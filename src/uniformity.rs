//! Which values, and which code, can differ between the threads of a block.
//!
//! A warp shuffle with the full mask (`shfl.sync ... 0xffffffff`, which
//! `warp_reduce_sum`, `warp_reduce_max` and `shfl_sync_bfly` emit) must be
//! executed by all 32 lanes of the warp together, and `bar.sync` (which
//! `barrier_sync` emits) by every thread of the block. In code that only some
//! of them reach the GPU leaves both undefined: a shuffle reads registers of
//! lanes that are not there, and a barrier may hang. Until 2026-10-12 the PTX
//! backend emitted them wherever they were written, `if t < 20 {
//! warp_reduce_sum(x) }` included. It now asks this module how far each
//! branch's condition is the same across threads, and refuses a collective
//! instruction where the threads reaching it can differ
//! (`PtxEmitter::refuse_divergent_collectives`).
//!
//! The analysis over-approximates, soundly. A value is the same in every
//! thread of the block ([`Uniform::Block`]) if it is computed from constants,
//! kernel parameters and the block and grid built-ins; the same within each
//! warp ([`Uniform::Warp`]) if a warp reduction's result enters it; and may
//! differ between lanes ([`Uniform::Thread`]) otherwise. Every memory read is
//! `Thread`: another thread may have stored to it between two warps' loads.
//! A variable is no more uniform than anything assigned to it, and a value
//! assigned inside a branch is no more uniform than the branch's condition.
//! The analysis ignores statement order and iterates to a fixpoint, so a
//! loop's condition sees the assignments its body makes.
//!
//! A thread index divided by 32 is the same within a warp of a
//! one-dimensional block, but this analysis does not know the block's shape
//! and calls it `Thread`: such a branch is refused, not proved.

use std::collections::{HashMap, HashSet};

use crate::ast::{Block, Expr, Stmt};
use crate::lexical_scope::source_name;

/// How far a value is known to be the same: in every thread of the block, in
/// every lane of each warp, or not at all. Ordered from most to least uniform,
/// so `max` combines two.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Uniform {
    Block,
    Warp,
    Thread,
}

impl Uniform {
    /// Which threads code under a condition of this uniformity can run on.
    pub fn runs_on(self) -> &'static str {
        match self {
            Uniform::Block => "every thread of the block",
            Uniform::Warp => "some warps of the block and not others",
            Uniform::Thread => "some lanes of a warp and not others",
        }
    }
}

/// Built-ins whose result is the same in every thread of the block.
const BLOCK_BUILTINS: &[&str] = &[
    "block_idx_x", "block_idx_y", "block_idx_z", "block_dim_x", "block_dim_y", "block_dim_z",
    "grid_dim_x", "grid_dim_y", "grid_dim_z",
];

/// Built-ins that give every lane of the warp the same result: a butterfly
/// reduction gives each pair of partners `a + b` and `b + a`, which are equal.
const WARP_REDUCTIONS: &[&str] = &["warp_reduce_sum", "warp_reduce_max"];

/// Built-ins whose result depends on their arguments alone. The carry-in
/// forms (`addc_u32`, ...) are not here: they read the carry flag whatever
/// instruction last set it.
const PURE_BUILTINS: &[&str] = &[
    "block_cdiv", "mul_wide_u32", "mul_wide_s32", "u64_lo32", "u64_hi32", "add_cc_u32",
    "sub_cc_u32", "mad_lo_cc_u32", "mad_hi_cc_u32",
];

/// How far each variable of one kernel body is the same across threads.
#[derive(Debug, Default)]
pub struct Uniformity {
    /// A variable missing here is never assigned in the body: a parameter.
    vars: HashMap<String, (Uniform, String)>,
    /// Source names a `chisel` block mentions as `%name`: it may write them.
    chisel_names: HashSet<String>,
}

impl Uniformity {
    pub fn of_body(body: &Block) -> Self {
        let mut u = Uniformity::default();
        loop {
            let before: Vec<(String, Uniform)> =
                u.vars.iter().map(|(k, (t, _))| (k.clone(), *t)).collect();
            u.walk(body, Uniform::Block, "");
            let unchanged = before.len() == u.vars.len()
                && before.iter().all(|(k, t)| u.vars.get(k).map(|(v, _)| v) == Some(t));
            if unchanged {
                return u;
            }
        }
    }

    /// How far `e` is the same across threads.
    pub fn of(&self, e: &Expr) -> Uniform {
        self.leaf(e).0
    }

    /// The least uniform part of `e`, for an error message: the variable
    /// (and where its value came from), built-in or memory read that makes
    /// `e` differ.
    pub fn why(&self, e: &Expr) -> String {
        let (_, what, var) = self.leaf(e);
        match var.and_then(|v| self.vars.get(&v)) {
            Some((_, origin)) if !origin.is_empty() => format!("{} ({})", what, origin),
            _ => what,
        }
    }

    /// The uniformity of `e`, its least uniform part in a few words, and the
    /// variable that part is, if it is one.
    fn leaf(&self, e: &Expr) -> (Uniform, String, Option<String>) {
        type Leaf = (Uniform, String, Option<String>);
        let uniform = || (Uniform::Block, String::new(), None);
        let worst = |a: Leaf, b: Leaf| if b.0 > a.0 { b } else { a };
        match e {
            Expr::IntLit(..) | Expr::FloatLit(..) | Expr::BoolLit(..) => uniform(),
            // `Fragment::zero` is the one path with a value; the others are refused.
            Expr::Path { .. } => uniform(),
            Expr::Ident(name, _) => {
                let shown = format!("`{}`", source_name(name));
                if self.chisel_names.contains(source_name(name)) {
                    return (Uniform::Thread, format!("{}, which a `chisel` block may write", shown), None);
                }
                match self.vars.get(name) {
                    Some((Uniform::Block, _)) | None => uniform(),
                    Some((u, _)) => (*u, shown, Some(name.clone())),
                }
            }
            Expr::BinaryOp { left, right, .. } => worst(self.leaf(left), self.leaf(right)),
            Expr::UnaryOp { operand, .. } => self.leaf(operand),
            // A lane of a v4 value: as uniform as the value.
            Expr::MemberAccess { base, .. } => self.leaf(base),
            Expr::Index { .. } => (Uniform::Thread, "a value loaded from memory".to_string(), None),
            Expr::Call { func, args, .. } => {
                let name = match &**func {
                    Expr::Ident(n, _) => n.as_str(),
                    _ => "",
                };
                if BLOCK_BUILTINS.contains(&name) {
                    uniform()
                } else if WARP_REDUCTIONS.contains(&name) {
                    (Uniform::Warp, format!("`{}()`", name), None)
                } else if PURE_BUILTINS.contains(&name) {
                    args.iter().map(|a| self.leaf(a)).fold(uniform(), worst)
                } else if name.is_empty() {
                    (Uniform::Thread, "a call".to_string(), None)
                } else {
                    (Uniform::Thread, format!("`{}()`", name), None)
                }
            }
            // Refused by the PTX backend; nothing here is known to be uniform.
            Expr::GenericCall { .. }
            | Expr::BlockExpr(..)
            | Expr::StructLit { .. }
            | Expr::StringLit(..)
            | Expr::CharLit(..)
            | Expr::SelfLit(..)
            | Expr::ZeroInit(..) => (Uniform::Thread, "an expression this backend refuses".to_string(), None),
        }
    }

    fn raise(&mut self, name: &str, to: Uniform, origin: String) {
        let e = self.vars.entry(name.to_string()).or_insert((Uniform::Block, String::new()));
        if to > e.0 {
            *e = (to, origin);
        }
    }

    /// `ctx` is how far the threads running `block` are the same set
    /// (`Block`: all of them), and `ctx_why` says which branch narrows it.
    fn walk(&mut self, block: &Block, ctx: Uniform, ctx_why: &str) {
        for s in &block.stmts {
            self.walk_stmt(s, ctx, ctx_why);
        }
    }

    /// The uniformity of a value assigned on `line` under `ctx`, and where
    /// it came from.
    fn assigned(&self, value: Option<&Expr>, ctx: Uniform, ctx_why: &str, line: usize) -> (Uniform, String) {
        let (u, what, _) = value.map_or((Uniform::Block, String::new(), None), |v| self.leaf(v));
        if ctx > u {
            (ctx, format!("assigned on line {} {}", line, ctx_why))
        } else {
            (u, format!("from {} on line {}", what, line))
        }
    }

    fn walk_stmt(&mut self, s: &Stmt, ctx: Uniform, ctx_why: &str) {
        match s {
            Stmt::Let { name, init, span, .. } => {
                let (u, why) = self.assigned(init.as_ref(), ctx, ctx_why, span.line);
                self.raise(name, u, why);
            }
            Stmt::Assign { target, value, span } | Stmt::CompoundAssign { target, value, span, .. } => {
                let (u, why) = self.assigned(Some(value), ctx, ctx_why, span.line);
                // An element store (`Out[i] = v`) assigns no variable; a read
                // of memory is `Thread` anyway.
                if let Some(name) = assigned_variable(target) {
                    self.raise(name, u, why);
                }
            }
            Stmt::If { condition, then_block, else_block, span, .. } => {
                let c = ctx.max(self.of(condition));
                let why = if c > ctx { format!("inside the `if` on line {}", span.line) } else { ctx_why.to_string() };
                self.walk(then_block, c, &why);
                if let Some(e) = else_block {
                    self.walk(e, c, &why);
                }
            }
            Stmt::While { condition, body, span, .. } => {
                let c = ctx.max(self.of(condition));
                let why = if c > ctx { format!("inside the `while` on line {}", span.line) } else { ctx_why.to_string() };
                self.walk(body, c, &why);
            }
            Stmt::For { loop_var, start, end, step, body, span, .. } => {
                let mut c = ctx.max(self.of(start)).max(self.of(end));
                if let Some(st) = step {
                    c = c.max(self.of(st));
                }
                let why = if c > ctx { format!("in the `for` on line {}", span.line) } else { ctx_why.to_string() };
                let header = if c > ctx {
                    format!("the counter of the `for` on line {}, whose bounds can differ", span.line)
                } else {
                    format!("the counter of the `for` on line {}, {}", span.line, ctx_why)
                };
                self.raise(loop_var, c, header);
                self.walk(body, c, &why);
            }
            Stmt::Chisel(block, _) => {
                for inner in &block.stmts {
                    if let Stmt::Expr(Expr::StringLit(text, _)) = inner {
                        self.chisel_names.extend(chisel_registers(text));
                    } else {
                        self.walk_stmt(inner, ctx, ctx_why);
                    }
                }
            }
            Stmt::SafeBlock(b, _) | Stmt::GhostBlock(b, _) => self.walk(b, ctx, ctx_why),
            Stmt::HintBlock { body, .. } | Stmt::ClockDomainBlock { body, .. } => self.walk(body, ctx, ctx_why),
            // `match` and `break` are refused by the PTX backend.
            Stmt::Match { .. }
            | Stmt::Break { .. }
            | Stmt::Expr(_)
            | Stmt::Return(..)
            | Stmt::TypeAlias { .. }
            | Stmt::CompileTimeAssert { .. } => {}
        }
    }

    /// The least uniform region inside `block` (run under `ctx`) where some
    /// threads leave: a `return` under a condition that can differ, or a
    /// `chisel` block that may branch. A loop whose body holds one runs its
    /// later iterations without those threads, so the whole body is then that
    /// divergent. The line is the exit's.
    pub fn exit_in(&self, block: &Block, ctx: Uniform) -> Option<(Uniform, usize)> {
        let mut found = None;
        self.find_exits(block, ctx, &mut found);
        found
    }

    fn find_exits(&self, block: &Block, ctx: Uniform, found: &mut Option<(Uniform, usize)>) {
        for s in &block.stmts {
            self.find_exits_stmt(s, ctx, found);
        }
    }

    fn find_exits_stmt(&self, s: &Stmt, ctx: Uniform, found: &mut Option<(Uniform, usize)>) {
        fn note(found: &mut Option<(Uniform, usize)>, u: Uniform, line: usize) {
            if u > Uniform::Block && found.map_or(true, |(f, _)| u > f) {
                *found = Some((u, line));
            }
        }
        match s {
            Stmt::Return(_, span) => note(found, ctx, span.line),
            Stmt::Chisel(b, span) => {
                if chisel_transfers_control(b) {
                    note(found, Uniform::Thread, span.line);
                }
                for inner in &b.stmts {
                    if !matches!(inner, Stmt::Expr(Expr::StringLit(..))) {
                        self.find_exits_stmt(inner, ctx, found);
                    }
                }
            }
            Stmt::If { condition, then_block, else_block, .. } => {
                let c = ctx.max(self.of(condition));
                self.find_exits(then_block, c, found);
                if let Some(e) = else_block {
                    self.find_exits(e, c, found);
                }
            }
            Stmt::While { condition, body, .. } => self.find_exits(body, ctx.max(self.of(condition)), found),
            Stmt::For { start, end, step, body, .. } => {
                let mut c = ctx.max(self.of(start)).max(self.of(end));
                if let Some(st) = step {
                    c = c.max(self.of(st));
                }
                self.find_exits(body, c, found);
            }
            Stmt::SafeBlock(b, _) | Stmt::GhostBlock(b, _) => self.find_exits(b, ctx, found),
            Stmt::HintBlock { body, .. } | Stmt::ClockDomainBlock { body, .. } => self.find_exits(body, ctx, found),
            _ => {}
        }
    }
}

/// The variable an assignment writes: `x = ...`, or `v.x = ...` on a vector.
fn assigned_variable(target: &Expr) -> Option<&str> {
    match target {
        Expr::Ident(name, _) => Some(name),
        Expr::MemberAccess { base, .. } => assigned_variable(base),
        _ => None,
    }
}

/// The `%name`s a `chisel` line mentions, special registers included (they
/// are not variables, so including them costs nothing).
fn chisel_registers(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(at) = rest.find('%') {
        rest = &rest[at + 1..];
        let end = rest.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap_or(rest.len());
        if end > 0 {
            out.push(rest[..end].to_string());
        }
        rest = &rest[end..];
    }
    out
}

/// The instructions of a line of PTX, without predicates, labels, comments
/// and braces: `@%p1 bra L;` gives `bra L`.
fn instructions(line: &str) -> impl Iterator<Item = &str> {
    let code = line.split("//").next().unwrap_or("");
    code.split(';').filter_map(|piece| {
        let mut s = piece.trim().trim_start_matches('{').trim_end_matches('}').trim();
        // A label ends with ':' and may precede an instruction on its line.
        if let Some(colon) = s.find(':') {
            let label = &s[..colon];
            if !label.is_empty() && label.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$') {
                s = s[colon + 1..].trim();
            }
        }
        if let Some(rest) = s.strip_prefix('@') {
            s = rest.split_once(char::is_whitespace).map_or("", |(_, r)| r).trim();
        }
        if s.is_empty() || s.starts_with('.') {
            None
        } else {
            Some(s)
        }
    })
}

/// The opcode of an instruction: its first word.
fn opcode(instruction: &str) -> &str {
    instruction.split(|c: char| c.is_whitespace() || c == ',').next().unwrap_or("")
}

/// What an instruction requires of the threads that execute it: `Warp` if
/// every lane of the warp (every lane its mask names) must execute it
/// together, `Block` if every thread of the block must. `None` for an
/// instruction each thread executes on its own.
pub fn collective_scope(instruction: &str) -> Option<Uniform> {
    let op = opcode(instruction);
    let parts: Vec<&str> = op.split('.').collect();
    let head = parts.first().copied().unwrap_or("");
    if head == "bar" && parts.get(1) == Some(&"warp") {
        Some(Uniform::Warp)
    } else if head == "bar" || head == "barrier" {
        Some(Uniform::Block)
    } else if parts.iter().any(|p| *p == "sync" || *p == "aligned") {
        // shfl.sync, vote.sync, match.sync, redux.sync, mma.sync.aligned,
        // ldmatrix.sync.aligned, wmma.*.sync.aligned, ...
        Some(Uniform::Warp)
    } else {
        None
    }
}

/// The first instruction in `ptx` that code as divergent as `here` may not
/// execute, with its opcode and the threads it needs.
pub fn first_collective_violated(ptx: &str, here: Uniform) -> Option<(Uniform, String)> {
    for line in ptx.lines() {
        for ins in instructions(line) {
            if let Some(needs) = collective_scope(ins) {
                if here > needs {
                    return Some((needs, opcode(ins).to_string()));
                }
            }
        }
    }
    None
}

/// Whether a `chisel` block may branch or return: its PTX could then leave
/// some threads behind, and nothing here can say which.
pub fn chisel_transfers_control(block: &Block) -> bool {
    block.stmts.iter().any(|s| match s {
        Stmt::Expr(Expr::StringLit(text, _)) => text.lines().flat_map(instructions).any(|ins| {
            let head = opcode(ins).split('.').next().unwrap_or("");
            matches!(head, "bra" | "brx" | "ret" | "exit" | "call")
        }),
        _ => false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(src: &str) -> Block {
        let full = format!("kernel k(A: GlobalMemory<F32>, N: I32) {{\n{}\n}}\n", src);
        let tokens = crate::lexer::Lexer::new(&full).tokenize();
        let prog = crate::parser::Parser::new(tokens).parse_program().expect("parses");
        match &prog.items[0] {
            crate::ast::Item::Kernel(k) => k.body.clone(),
            _ => panic!("not a kernel"),
        }
    }

    fn tier(src: &str, var: &str) -> Uniform {
        let u = Uniformity::of_body(&body(src));
        u.of(&Expr::Ident(var.to_string(), crate::ast::Span { line: 0, col: 0 }))
    }

    #[test]
    fn instructions_are_classified_by_what_they_require() {
        for (ins, want) in [
            ("    shfl.sync.bfly.b32 %f1, %f0, 16, 0x1f, 0xffffffff;", Some(Uniform::Warp)),
            ("    @%p1 shfl.sync.idx.b32 %r1, %r0, 0, 0x1f, 0xffffffff;", Some(Uniform::Warp)),
            ("    bar.sync 0;", Some(Uniform::Block)),
            ("    barrier.sync.aligned 0;", Some(Uniform::Block)),
            ("    bar.warp.sync 0xffffffff;", Some(Uniform::Warp)),
            ("    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%f0}, {%r0}, {%r1}, {%f0};", Some(Uniform::Warp)),
            ("    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r0, %r1, %r2, %r3}, [%r4];", Some(Uniform::Warp)),
            ("    vote.sync.ballot.b32 %r1, %p1, 0xffffffff;", Some(Uniform::Warp)),
            ("    cp.async.ca.shared.global [%r1], [%rd1], 16;", None),
            ("    cp.async.wait_all;", None),
            ("    activemask.b32 %r1;", None),
            ("    membar.gl;", None),
            ("    add.f32 %f1, %f0, %f0; // shfl.sync in a comment", None),
            ("LOOP_START_3:", None),
            ("    .shared .align 4 .f32 smem_reduce[32];", None),
        ] {
            let got = instructions(ins).filter_map(collective_scope).next();
            assert_eq!(got, want, "{ins}");
        }
    }

    #[test]
    fn only_a_collective_the_region_cannot_run_is_reported() {
        let shfl = "    shfl.sync.bfly.b32 %f1, %f0, 1, 0x1f, 0xffffffff;\n";
        let bar = "    bar.sync 0;\n";
        assert_eq!(first_collective_violated(shfl, Uniform::Block), None);
        assert_eq!(first_collective_violated(shfl, Uniform::Warp), None);
        assert_eq!(first_collective_violated(shfl, Uniform::Thread).map(|v| v.0), Some(Uniform::Warp));
        assert_eq!(first_collective_violated(bar, Uniform::Block), None);
        assert_eq!(first_collective_violated(bar, Uniform::Warp).map(|v| v.0), Some(Uniform::Block));
    }

    #[test]
    fn thread_indices_memory_and_branches_make_values_differ() {
        let src = "let t: I32 = thread_idx_x();\n let b: I32 = block_idx_x() * N;\n \
                   let s: F32 = warp_reduce_sum(A[t]);\n let m: F32 = A[0];\n \
                   let c: I32 = 0;\n if t < 5 { c = 1; }\n let d: I32 = 0;\n if b < 5 { d = 1; }\n \
                   let w: I32 = 0;\n if s > 0.0 { w = 1; }\n let k: I32 = 0;\n \
                   for i in 0..N { k = k + i; }";
        assert_eq!(tier(src, "t"), Uniform::Thread);
        assert_eq!(tier(src, "b"), Uniform::Block);
        assert_eq!(tier(src, "N"), Uniform::Block);
        assert_eq!(tier(src, "s"), Uniform::Warp);
        assert_eq!(tier(src, "m"), Uniform::Thread);
        assert_eq!(tier(src, "c"), Uniform::Thread);
        assert_eq!(tier(src, "d"), Uniform::Block);
        assert_eq!(tier(src, "w"), Uniform::Warp);
        assert_eq!(tier(src, "k"), Uniform::Block);
    }

    #[test]
    fn a_loop_carried_value_reaches_the_loops_own_condition() {
        // `j` becomes thread-dependent only through `k`, assigned after the
        // test that reads it: the fixpoint, not one pass in order, finds it.
        let src = "let j: I32 = 0;\n let k: I32 = 0;\n let x: I32 = 0;\n \
                   while j < N { j = k; k = thread_idx_x(); }\n if j < 3 { x = 1; }";
        assert_eq!(tier(src, "j"), Uniform::Thread);
        assert_eq!(tier(src, "x"), Uniform::Thread);
    }

    #[test]
    fn a_return_some_threads_take_is_an_exit() {
        let u = |src: &str| {
            let b = body(src);
            Uniformity::of_body(&b).exit_in(&b, Uniform::Block).map(|e| e.0)
        };
        assert_eq!(u("let t: I32 = thread_idx_x();\n if t < 5 { return; }"), Some(Uniform::Thread));
        assert_eq!(u("if block_idx_x() < 5 { return; }"), None);
        assert_eq!(u("let s: F32 = warp_reduce_sum(A[0]);\n if s > 0.0 { return; }"), Some(Uniform::Warp));
        assert_eq!(u("for i in 0..N { if A[i] > 0.0 { return; } }"), Some(Uniform::Thread));
        assert_eq!(u("return;"), None);
    }

    #[test]
    fn a_chisel_block_may_write_what_it_names_and_may_branch() {
        let src = "let x: I32 = 0;\n chisel { \"mov.u32 %x, %tid.x;\"; }";
        assert_eq!(tier(src, "x"), Uniform::Thread);
        let b = body("chisel { \"@%p1 bra DONE;\"; }");
        assert_eq!(Uniformity::of_body(&b).exit_in(&b, Uniform::Block).map(|e| e.0), Some(Uniform::Thread));
        let b = body("chisel { \"add.u32 %r1, %r1, 1;\"; }");
        assert_eq!(Uniformity::of_body(&b).exit_in(&b, Uniform::Block), None);
    }
}
