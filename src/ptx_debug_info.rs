//! `--emit-ptx -g`: DWARF in the PTX module, so `cuda-gdb` shows a kernel's
//! Y variables - each thread's own - by their Y names and types.
//!
//! The format is the one `nvcc -G` writes (read from CUDA 13.4's output):
//!
//! * the module's `.target` gains `debug` (`.target sm_89, debug`); `ptxas`
//!   and the driver's JIT then compile it as debug code, unoptimised, which
//!   is what keeps every variable in a place the debugger can name;
//! * the line table is the `.file` / `.loc` that `--lineinfo` writes;
//! * `.section .debug_abbrev` and `.section .debug_info` hold DWARF 2 whose
//!   addresses are PTX labels. `ptxas` turns each label into the SASS address
//!   of the instruction after it, and each variable's PTX register into
//!   wherever that register lives in the SASS.
//!
//! A variable's location is `DW_OP_regx` of its PTX register's NAME, packed
//! big-endian as ASCII into the operand: `%r1` is `0x257231`. That is NVIDIA's
//! encoding (nvcc's `-G` output uses it, and LLVM's NVPTX backend writes the
//! same), with `DW_AT_address_class` 2, the register space.
//!
//! **What a variable's scope is.** A binding is visible from the instruction
//! after its `let` to the end of the block the `let` is in, so a stop on the
//! `let`'s own line, before the value exists, does not show it: `print x`
//! there says there is no `x`, rather than printing whatever the register held
//! before. Each `let` opens a lexical block nested in the previous one's, the
//! way Rust's debug information does, so the blocks nest as DWARF requires. A
//! `for` loop's variable is visible over the loop.
//!
//! **What it does not guess.** Every binding of a scalar type is described.
//! One the PTX does not hold in one register - a `v4` load's four registers,
//! or a `@ZeroDrift` grid accumulator's two halves - is listed with no
//! location, which `cuda-gdb` prints as `<optimized out>`: never a wrong value.
//! A parameter's location is its register too, not its `.param` slot: a Y
//! kernel may assign its parameters (`N = N + 1` writes the register), and the
//! slot would show the launch value after that.

use std::fmt::Write as _;

/// A DWARF base type: its Y name, `DW_ATE_*` encoding and size in bytes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BaseTy {
    pub name: String,
    pub encoding: u8,
    pub size: u8,
}

pub(crate) const DW_ATE_FLOAT: u8 = 0x04;
pub(crate) const DW_ATE_SIGNED: u8 = 0x05;
pub(crate) const DW_ATE_UNSIGNED: u8 = 0x08;

impl BaseTy {
    pub fn new(name: &str, encoding: u8, size: u8) -> Self {
        BaseTy { name: name.to_string(), encoding, size }
    }
}

/// The type a variable is described with.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum DbgType {
    Base(BaseTy),
    /// `GlobalMemory<T>`: a pointer to `T` (`None`: to an element type DWARF
    /// cannot describe, which reads as `void *`).
    Pointer(Option<BaseTy>),
    /// A vector value such as `U32x4`: a struct of `fields`, each an `elem`,
    /// each in a register of its own - so `print v.x` reads as the source
    /// does.
    Vector { name: String, elem: BaseTy, fields: Vec<String> },
    /// A type the debug information cannot describe; only ever given to a
    /// variable that has no location.
    Unknown(String),
}

/// A parameter or a binding.
#[derive(Clone, Debug)]
pub(crate) struct Var {
    /// The source name (`lexical_scope` renames shadowing bindings apart;
    /// this is the name before that).
    pub name: String,
    pub line: usize,
    pub ty: DbgType,
    /// The PTX registers holding it: one, one per field of a `Vector` (an
    /// empty name for a field no register holds), or none for
    /// `<optimized out>`.
    pub regs: Vec<String>,
}

#[derive(Debug)]
struct Scope {
    begin: String,
    end: String,
    vars: Vec<Var>,
    children: Vec<Scope>,
}

#[derive(Debug)]
struct Kernel {
    name: String,
    file: usize,
    line: usize,
    begin: String,
    end: String,
    params: Vec<Var>,
    scopes: Vec<Scope>,
}

/// The debug information of a module being emitted.
#[derive(Default, Debug)]
pub(crate) struct PtxDebugInfo {
    labels: usize,
    kernels: Vec<Kernel>,
    current: Option<Kernel>,
    /// Open scopes, innermost last, each with the depth of the source block
    /// it was opened in.
    open: Vec<(usize, Scope)>,
    depth: usize,
}

impl PtxDebugInfo {
    /// A label no other in the module has: the sections refer to labels from
    /// module scope, while the emitter's own labels are per kernel.
    pub fn fresh_label(&mut self) -> String {
        let l = format!("$L__ydbg{}", self.labels);
        self.labels += 1;
        l
    }

    pub fn begin_kernel(&mut self, name: &str, file: usize, line: usize, begin: String) {
        self.open.clear();
        self.depth = 0;
        self.current = Some(Kernel {
            name: name.to_string(),
            file,
            line,
            begin,
            end: String::new(),
            params: Vec::new(),
            scopes: Vec::new(),
        });
    }

    pub fn param(&mut self, v: Var) {
        if let Some(k) = self.current.as_mut() {
            k.params.push(v);
        }
    }

    /// `reg` no longer holds a lane of the vector that was loaded into it:
    /// a binding of that lane is assigned and still shares the register
    /// (`copy_if_lane_assigned` copies only where the program reads the
    /// lane again). That lane is described as `<optimized out>` from now on,
    /// for the whole of the vector's scope - its value before the assignment
    /// is the one place this gives up a value the program still has.
    pub fn forget_lane(&mut self, reg: &str) {
        for (_, s) in self.open.iter_mut() {
            for v in s.vars.iter_mut() {
                if matches!(v.ty, DbgType::Vector { .. }) {
                    for r in v.regs.iter_mut() {
                        if r == reg {
                            r.clear();
                        }
                    }
                }
            }
        }
    }

    pub fn in_kernel(&self) -> bool {
        self.current.is_some()
    }

    pub fn enter_block(&mut self) {
        self.depth += 1;
    }

    /// `v` is visible from `at` to the end of the innermost source block.
    pub fn bind(&mut self, v: Var, at: String) {
        if self.current.is_none() {
            return;
        }
        self.open.push((
            self.depth,
            Scope { begin: at, end: String::new(), vars: vec![v], children: Vec::new() },
        ));
    }

    /// The innermost source block ends at `at`: so does every scope opened in it.
    pub fn leave_block(&mut self, at: &str) {
        while matches!(self.open.last(), Some((d, _)) if *d >= self.depth) {
            let (_, mut s) = self.open.pop().unwrap();
            s.end = at.to_string();
            self.attach(s);
        }
        self.depth = self.depth.saturating_sub(1);
    }

    fn attach(&mut self, s: Scope) {
        match self.open.last_mut() {
            Some((_, parent)) => parent.children.push(s),
            None => {
                if let Some(k) = self.current.as_mut() {
                    k.scopes.push(s);
                }
            }
        }
    }

    pub fn end_kernel(&mut self, at: String) {
        while let Some((_, mut s)) = self.open.pop() {
            s.end = at.clone();
            self.attach(s);
        }
        self.depth = 0;
        if let Some(mut k) = self.current.take() {
            k.end = at;
            self.kernels.push(k);
        }
    }

    /// The `.section` directives, for the end of the module. `None` when no
    /// kernel was described.
    pub fn sections(&self, source: &std::path::Path, comp_dir: &std::path::Path) -> Option<String> {
        if self.kernels.is_empty() {
            return None;
        }
        let mut info = Out::default();
        // Unit header: length (patched below), version 2, abbrev offset,
        // address size.
        info.u16(2);
        info.sym32(".debug_abbrev");
        info.u8(8);

        info.uleb(ABBR_CU);
        info.string("Y compiler (--emit-ptx -g)");
        info.u16(DW_LANG_C99);
        info.string(&source.to_string_lossy());
        info.sym32(".debug_line");
        info.string(&comp_dir.to_string_lossy());
        info.u64(0);

        // Types first, so every reference to one points backwards.
        let mut types: Vec<DbgType> = Vec::new();
        let want = |t: &DbgType, types: &mut Vec<DbgType>| {
            let mut add = |t: DbgType| {
                if !types.contains(&t) {
                    types.push(t);
                }
            };
            match t {
                DbgType::Pointer(Some(b)) | DbgType::Vector { elem: b, .. } => add(DbgType::Base(b.clone())),
                _ => {}
            }
            add(t.clone());
        };
        fn walk(s: &Scope, f: &mut dyn FnMut(&DbgType)) {
            for v in &s.vars {
                f(&v.ty);
            }
            for c in &s.children {
                walk(c, f);
            }
        }
        for k in &self.kernels {
            for p in &k.params {
                want(&p.ty, &mut types);
            }
            for s in &k.scopes {
                walk(s, &mut |t| want(t, &mut types));
            }
        }
        let mut offset: std::collections::HashMap<DbgType, u32> = Default::default();
        for t in &types {
            let here = info.len as u32 + 4;
            match t {
                DbgType::Base(b) => {
                    info.uleb(ABBR_BASE);
                    info.string(&b.name);
                    info.u8(b.encoding);
                    info.u8(b.size);
                }
                DbgType::Pointer(Some(b)) => {
                    info.uleb(ABBR_PTR);
                    info.u32(offset[&DbgType::Base(b.clone())]);
                    info.u8(ADDR_GENERIC);
                }
                DbgType::Pointer(None) => {
                    info.uleb(ABBR_VOID_PTR);
                    info.u8(ADDR_GENERIC);
                }
                DbgType::Vector { name, elem, fields } => {
                    info.uleb(ABBR_STRUCT);
                    info.string(name);
                    info.u8(elem.size * fields.len() as u8);
                    let elem_ty = offset[&DbgType::Base(elem.clone())];
                    for (i, f) in fields.iter().enumerate() {
                        info.uleb(ABBR_MEMBER);
                        info.string(f);
                        info.u32(elem_ty);
                        info.u8(2);
                        info.u8(DW_OP_PLUS_UCONST);
                        info.u8(elem.size * i as u8);
                    }
                    info.u8(0);
                }
                DbgType::Unknown(name) => {
                    info.uleb(ABBR_UNSPECIFIED);
                    info.string(name);
                }
            }
            offset.insert(t.clone(), here);
        }

        for k in &self.kernels {
            info.uleb(ABBR_SUBPROGRAM);
            info.sym64(&k.begin);
            info.sym64(&k.end);
            info.u8(1);
            info.u8(DW_OP_CALL_FRAME_CFA);
            info.string(&k.name);
            info.u16(k.file as u16);
            info.u32(k.line as u32);
            info.u8(1);
            for p in &k.params {
                info.var(p, k.file, offset[&p.ty], ABBR_PARAM, ABBR_PARAM_NOLOC);
            }
            for s in &k.scopes {
                scope(&mut info, s, k.file, &offset);
            }
            info.u8(0);
        }
        info.u8(0);

        let mut text = String::new();
        text.push_str("\t.section\t.debug_abbrev\n\t{\n");
        write_bytes(&mut text, &abbrevs());
        text.push_str("\t}\n\t.section\t.debug_info\n\t{\n");
        writeln!(text, ".b32 {}", info.len).unwrap();
        info.write(&mut text);
        text.push_str("\t}\n");
        Some(text)
    }
}

fn scope(info: &mut Out, s: &Scope, file: usize, offset: &std::collections::HashMap<DbgType, u32>) {
    info.uleb(ABBR_BLOCK);
    info.sym64(&s.begin);
    info.sym64(&s.end);
    for v in &s.vars {
        info.var(v, file, offset[&v.ty], ABBR_VAR, ABBR_VAR_NOLOC);
    }
    for c in &s.children {
        scope(info, c, file, offset);
    }
    info.u8(0);
}

const DW_LANG_C99: u16 = 0x0c;
const DW_OP_REGX: u8 = 0x90;
const DW_OP_PIECE: u8 = 0x93;
const DW_OP_PLUS_UCONST: u8 = 0x23;
const DW_OP_CALL_FRAME_CFA: u8 = 0x9c;
/// NVIDIA's DWARF address classes: 2 is the register space, 12 generic.
const ADDR_REG: u8 = 2;
const ADDR_GENERIC: u8 = 12;

const ABBR_CU: u64 = 1;
const ABBR_BASE: u64 = 2;
const ABBR_PTR: u64 = 3;
const ABBR_VOID_PTR: u64 = 4;
const ABBR_UNSPECIFIED: u64 = 5;
const ABBR_SUBPROGRAM: u64 = 6;
const ABBR_PARAM: u64 = 7;
const ABBR_PARAM_NOLOC: u64 = 8;
const ABBR_VAR: u64 = 9;
const ABBR_VAR_NOLOC: u64 = 10;
const ABBR_BLOCK: u64 = 11;
const ABBR_STRUCT: u64 = 12;
const ABBR_MEMBER: u64 = 13;

// DW_TAG_*, DW_AT_* and DW_FORM_* codes, DWARF 2.
const DW_TAG_COMPILE_UNIT: u64 = 0x11;
const DW_TAG_BASE_TYPE: u64 = 0x24;
const DW_TAG_POINTER_TYPE: u64 = 0x0f;
const DW_TAG_UNSPECIFIED_TYPE: u64 = 0x3b;
const DW_TAG_SUBPROGRAM: u64 = 0x2e;
const DW_TAG_FORMAL_PARAMETER: u64 = 0x05;
const DW_TAG_VARIABLE: u64 = 0x34;
const DW_TAG_LEXICAL_BLOCK: u64 = 0x0b;
const DW_TAG_STRUCTURE_TYPE: u64 = 0x13;
const DW_TAG_MEMBER: u64 = 0x0d;
const AT_DATA_MEMBER_LOCATION: u64 = 0x38;
const AT_NAME: u64 = 0x03;
const AT_PRODUCER: u64 = 0x25;
const AT_LANGUAGE: u64 = 0x13;
const AT_STMT_LIST: u64 = 0x10;
const AT_COMP_DIR: u64 = 0x1b;
const AT_LOW_PC: u64 = 0x11;
const AT_HIGH_PC: u64 = 0x12;
const AT_FRAME_BASE: u64 = 0x40;
const AT_DECL_FILE: u64 = 0x3a;
const AT_DECL_LINE: u64 = 0x3b;
const AT_EXTERNAL: u64 = 0x3f;
const AT_LOCATION: u64 = 0x02;
const AT_ADDRESS_CLASS: u64 = 0x33;
const AT_TYPE: u64 = 0x49;
const AT_ENCODING: u64 = 0x3e;
const AT_BYTE_SIZE: u64 = 0x0b;
const FORM_ADDR: u64 = 0x01;
const FORM_BLOCK1: u64 = 0x0a;
const FORM_DATA1: u64 = 0x0b;
const FORM_DATA2: u64 = 0x05;
const FORM_DATA4: u64 = 0x06;
const FORM_STRING: u64 = 0x08;
const FORM_FLAG: u64 = 0x0c;
const FORM_REF4: u64 = 0x13;

fn abbrevs() -> Vec<u8> {
    let named_var = |tag: u64, located: bool| -> (u64, bool, Vec<(u64, u64)>) {
        let mut a = Vec::new();
        if located {
            a.push((AT_LOCATION, FORM_BLOCK1));
            a.push((AT_ADDRESS_CLASS, FORM_DATA1));
        }
        a.extend([
            (AT_NAME, FORM_STRING),
            (AT_DECL_FILE, FORM_DATA2),
            (AT_DECL_LINE, FORM_DATA4),
            (AT_TYPE, FORM_REF4),
        ]);
        (tag, false, a)
    };
    let table: Vec<(u64, (u64, bool, Vec<(u64, u64)>))> = vec![
        (
            ABBR_CU,
            (
                DW_TAG_COMPILE_UNIT,
                true,
                vec![
                    (AT_PRODUCER, FORM_STRING),
                    (AT_LANGUAGE, FORM_DATA2),
                    (AT_NAME, FORM_STRING),
                    (AT_STMT_LIST, FORM_DATA4),
                    (AT_COMP_DIR, FORM_STRING),
                    (AT_LOW_PC, FORM_ADDR),
                ],
            ),
        ),
        (
            ABBR_BASE,
            (
                DW_TAG_BASE_TYPE,
                false,
                vec![(AT_NAME, FORM_STRING), (AT_ENCODING, FORM_DATA1), (AT_BYTE_SIZE, FORM_DATA1)],
            ),
        ),
        (ABBR_PTR, (DW_TAG_POINTER_TYPE, false, vec![(AT_TYPE, FORM_REF4), (AT_ADDRESS_CLASS, FORM_DATA1)])),
        (ABBR_VOID_PTR, (DW_TAG_POINTER_TYPE, false, vec![(AT_ADDRESS_CLASS, FORM_DATA1)])),
        (ABBR_UNSPECIFIED, (DW_TAG_UNSPECIFIED_TYPE, false, vec![(AT_NAME, FORM_STRING)])),
        (
            ABBR_SUBPROGRAM,
            (
                DW_TAG_SUBPROGRAM,
                true,
                vec![
                    (AT_LOW_PC, FORM_ADDR),
                    (AT_HIGH_PC, FORM_ADDR),
                    (AT_FRAME_BASE, FORM_BLOCK1),
                    (AT_NAME, FORM_STRING),
                    (AT_DECL_FILE, FORM_DATA2),
                    (AT_DECL_LINE, FORM_DATA4),
                    (AT_EXTERNAL, FORM_FLAG),
                ],
            ),
        ),
        (ABBR_PARAM, named_var(DW_TAG_FORMAL_PARAMETER, true)),
        (ABBR_PARAM_NOLOC, named_var(DW_TAG_FORMAL_PARAMETER, false)),
        (ABBR_VAR, named_var(DW_TAG_VARIABLE, true)),
        (ABBR_VAR_NOLOC, named_var(DW_TAG_VARIABLE, false)),
        (ABBR_BLOCK, (DW_TAG_LEXICAL_BLOCK, true, vec![(AT_LOW_PC, FORM_ADDR), (AT_HIGH_PC, FORM_ADDR)])),
        (ABBR_STRUCT, (DW_TAG_STRUCTURE_TYPE, true, vec![(AT_NAME, FORM_STRING), (AT_BYTE_SIZE, FORM_DATA1)])),
        (
            ABBR_MEMBER,
            (
                DW_TAG_MEMBER,
                false,
                vec![(AT_NAME, FORM_STRING), (AT_TYPE, FORM_REF4), (AT_DATA_MEMBER_LOCATION, FORM_BLOCK1)],
            ),
        ),
    ];
    let mut b = Vec::new();
    for (code, (tag, children, attrs)) in table {
        uleb(&mut b, code);
        uleb(&mut b, tag);
        b.push(children as u8);
        for (a, f) in attrs {
            uleb(&mut b, a);
            uleb(&mut b, f);
        }
        b.extend([0, 0]);
    }
    b.push(0);
    b
}

/// The `DW_OP_regx` operand naming PTX register `reg`: its ASCII bytes, the
/// first one most significant. A name longer than eight bytes cannot be
/// encoded; the emitter's registers are at most `%rd` and five digits.
pub(crate) fn encode_register(reg: &str) -> Option<u64> {
    if reg.is_empty() || reg.len() > 8 || !reg.is_ascii() {
        return None;
    }
    Some(reg.bytes().fold(0u64, |v, c| (v << 8) | c as u64))
}

fn uleb(b: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            b.push(byte);
            return;
        }
        b.push(byte | 0x80);
    }
}

/// `v`'s location expression: `DW_OP_regx` of its register, or for a
/// `Vector` one `DW_OP_regx`, `DW_OP_piece` pair per field. `None` when it has
/// no registers or one that cannot be named.
fn location(v: &Var) -> Option<Vec<u8>> {
    let piece = match &v.ty {
        DbgType::Vector { elem, fields, .. } if fields.len() == v.regs.len() => Some(elem.size),
        DbgType::Vector { .. } => return None,
        _ if v.regs.len() == 1 => None,
        _ => return None,
    };
    let mut loc = Vec::new();
    for r in &v.regs {
        // A piece with no location before it is a piece the debugger
        // reports as `<optimized out>` (DWARF 2, 2.4.4).
        if !(r.is_empty() && piece.is_some()) {
            loc.push(DW_OP_REGX);
            uleb(&mut loc, encode_register(r)?);
        }
        if let Some(size) = piece {
            loc.push(DW_OP_PIECE);
            uleb(&mut loc, size as u64);
        }
    }
    (loc.len() <= 255).then_some(loc)
}

enum Piece {
    Bytes(Vec<u8>),
    Sym64(String),
    Sym32(String),
}

/// A section's contents: bytes, and the label/section references `ptxas`
/// resolves. `len` is the byte count so far, from which DIE offsets come.
#[derive(Default)]
struct Out {
    pieces: Vec<Piece>,
    len: usize,
}

impl Out {
    fn bytes(&mut self, b: &[u8]) {
        self.len += b.len();
        if let Some(Piece::Bytes(v)) = self.pieces.last_mut() {
            v.extend_from_slice(b);
        } else {
            self.pieces.push(Piece::Bytes(b.to_vec()));
        }
    }
    fn u8(&mut self, v: u8) {
        self.bytes(&[v]);
    }
    fn u16(&mut self, v: u16) {
        self.bytes(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.bytes(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }
    fn uleb(&mut self, v: u64) {
        let mut b = Vec::new();
        uleb(&mut b, v);
        self.bytes(&b);
    }
    fn string(&mut self, s: &str) {
        // A NUL inside would end the string early; the emitter's names and
        // paths have none, and one that did is cut there, not misread.
        let s = s.split('\0').next().unwrap_or("");
        self.bytes(s.as_bytes());
        self.u8(0);
    }
    fn sym64(&mut self, label: &str) {
        self.pieces.push(Piece::Sym64(label.to_string()));
        self.len += 8;
    }
    fn sym32(&mut self, section: &str) {
        self.pieces.push(Piece::Sym32(section.to_string()));
        self.len += 4;
    }
    fn var(&mut self, v: &Var, file: usize, ty: u32, located: u64, unlocated: u64) {
        match location(v) {
            Some(loc) => {
                self.uleb(located);
                self.u8(loc.len() as u8);
                self.bytes(&loc);
                self.u8(ADDR_REG);
            }
            None => self.uleb(unlocated),
        }
        self.string(&v.name);
        self.u16(file as u16);
        self.u32(v.line as u32);
        self.u32(ty);
    }
    fn write(&self, text: &mut String) {
        for p in &self.pieces {
            match p {
                Piece::Bytes(b) => write_bytes(text, b),
                Piece::Sym64(l) => writeln!(text, ".b64 {}", l).unwrap(),
                Piece::Sym32(s) => writeln!(text, ".b32 {}", s).unwrap(),
            }
        }
    }
}

fn write_bytes(text: &mut String, b: &[u8]) {
    for chunk in b.chunks(32) {
        let list: Vec<String> = chunk.iter().map(|x| x.to_string()).collect();
        writeln!(text, ".b8 {}", list.join(",")).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_register_name_is_its_ascii_bytes() {
        // nvcc -G's `x` lives in `%f4`: DW_OP_regx 180,204,149,1.
        let mut b = Vec::new();
        uleb(&mut b, encode_register("%f4").unwrap());
        assert_eq!(b, [180, 204, 149, 1]);
        assert_eq!(encode_register("%r1"), Some(0x257231));
        assert_eq!(encode_register("%rd12345"), Some(0x2572_6431_3233_3435));
        assert_eq!(encode_register("%rd123456"), None);
    }
}
