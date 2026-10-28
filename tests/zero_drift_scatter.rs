//! `drift_add_at(ACC, i, acc)` hands a `@ZeroDrift` partial to total `i`, and
//! `drift_finish(ACC, OUT, n)` writes all `n` totals once every block has
//! handed its partials over. Each total is the same bits for every launch
//! shape, in whatever order the hand-overs land.
//!
//! Each property is checked by running the kernels:
//!
//! * **The totals.** A scatter (element `k` goes to total `IDX[k]`) and a
//!   pair scatter (`+X[k]` to `IDX[k]`, `-X[k]` to `JDX[k]`, the shape of a
//!   force between two atoms), over 98 launch shapes and 3, 1000 and 100,003
//!   totals: every launch stores the bits computed here from the definition
//!   for every total, and leaves `ACC` zero for the next launch. That
//!   includes totals whose terms pass 2^63 and cancel. A control kernel that
//!   scatters the same input with `f32` atomics must give more than one
//!   answer over the same shapes, or the input could not tell an exact sum
//!   from a lucky one.
//! * **Two kernels.** The scatter with `drift_close` and a second kernel's
//!   `drift_convert`, the second at its own shape (one thread to 1000 blocks):
//!   the same bits, and `ACC` zero after both.
//! * **The stops.** A program stops when a term, an index or a total is out
//!   of range, and then in every launch shape; otherwise in none, even where
//!   one thread's partial leaves the declared range. With two kernels it also
//!   stops when `drift_convert` runs without `drift_close` before it, twice,
//!   with another count or into another type, and when `drift_close` runs
//!   twice. Each launch runs in a child process, because a trap leaves the
//!   device unusable for its process.
//!
//! Hermetic checks pin the refusals that keep every total exact and the
//! emitted shape.
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

const NARROW: &str = "@bounds(min=-1000000.0, max=1000000.0)";
// 2.1e9 in Q32.32 is 9.0e18, just inside 2^63.
const WIDE: &str = "@bounds(min=-2100000000.0, max=2100000000.0)";

/// Thread `t` of block `b`, both flattened over three dimensions, takes
/// elements `k = b * threads + t + j * blocks * threads`, `j < K`, `k < N`,
/// and runs `body` on each.
fn kernel(elem: &str, body: &str, tail: &str) -> String {
    format!(
        "kernel k(X: GlobalMemory<{elem}>, IDX: GlobalMemory<I32>, JDX: GlobalMemory<I32>, ACC: GlobalMemory<I64>, OUT: GlobalMemory<{elem}>, N: I32, K: I32, NT: I32) {{
    let t: I32 = thread_idx_x() + block_dim_x() * (thread_idx_y() + block_dim_y() * thread_idx_z());
    let threads: I32 = block_dim_x() * block_dim_y() * block_dim_z();
    let b: I32 = block_idx_x() + grid_dim_x() * (block_idx_y() + grid_dim_y() * block_idx_z());
    let stride: I32 = grid_dim_x() * grid_dim_y() * grid_dim_z() * threads;
    @invariant(j >= 0)
    for j in 0..K step 1 {{
        let k: I32 = b * threads + t + j * stride;
        if k < N {{
{body}
        }}
    }}
    {tail}
}}

fn main() {{}}
"
    )
}

fn drift_let(acc_ty: &str, bounds: &str) -> String {
    let zero = if acc_ty == "I64" { "0" } else { "0.0" };
    format!("            @ZeroDrift\n            {bounds}\n            let mut acc: {acc_ty} = {zero};")
}

/// Element `k` to total `IDX[k]`.
fn scatter(acc_ty: &str, bounds: &str) -> String {
    let elem = if acc_ty == "I64" { "I64" } else { "F32" };
    let body = format!("{}\n            acc += X[k];\n            drift_add_at(ACC, IDX[k], acc);", drift_let(acc_ty, bounds));
    kernel(elem, &body, "drift_finish(ACC, OUT, NT);")
}

/// `scatter` closing its totals with `drift_close`, and two kernels that
/// convert them: `c` into `OUT`'s type and `ci` into `I64`.
fn scatter_closed(acc_ty: &str, bounds: &str) -> String {
    let elem = if acc_ty == "I64" { "I64" } else { "F32" };
    let body = format!("{}\n            acc += X[k];\n            drift_add_at(ACC, IDX[k], acc);", drift_let(acc_ty, bounds));
    kernel(elem, &body, "drift_close(ACC, NT);").replace(
        "fn main() {}",
        &format!(
            "kernel c(ACC: GlobalMemory<I64>, OUT: GlobalMemory<{elem}>, NT: I32) {{\n    drift_convert(ACC, OUT, NT);\n}}\n\n\
             kernel ci(ACC: GlobalMemory<I64>, OUT: GlobalMemory<I64>, NT: I32) {{\n    drift_convert(ACC, OUT, NT);\n}}\n\nfn main() {{}}"
        ),
    )
}

/// `+X[k]` to total `IDX[k]` and `-X[k]` to total `JDX[k]`, one partial
/// reused after its first hand-over set it back to zero.
fn pairs(bounds: &str) -> String {
    let body = format!(
        "{}\n            acc += X[k];\n            drift_add_at(ACC, IDX[k], acc);\n            acc -= X[k];\n            drift_add_at(ACC, JDX[k], acc);",
        drift_let("F32", bounds)
    );
    kernel("F32", &body, "drift_finish(ACC, OUT, NT);")
}

/// Every thread sums its elements first and hands the sum to total 0 once,
/// so one partial can leave the declared range while the total does not.
fn summed_first(bounds: &str) -> String {
    let src = kernel("F32", "            acc += X[k];", "drift_add_at(ACC, 0, acc);\n    drift_finish(ACC, OUT, NT);");
    src.replacen("    @invariant(j >= 0)", &format!("{}\n    @invariant(j >= 0)", drift_let("F32", bounds).replace("            ", "    ")), 1)
}

/// The control: the same scatter with `f32` atomics, which Y refuses to
/// emit, so it is written here in PTX. Same parameters and partition.
const F32_ATOMICS: &str = r#".version 7.0
.target sm_80
.address_size 64
.visible .entry k(.param .u64 X, .param .u64 IDX, .param .u64 JDX, .param .u64 ACC, .param .u64 OUT, .param .u32 N, .param .u32 K, .param .u32 NT)
{
    .reg .pred %p<3>;
    .reg .b32 %r<24>;
    .reg .f32 %f<2>;
    .reg .b64 %rd<12>;
    ld.param.u64 %rd1, [X];
    ld.param.u64 %rd2, [IDX];
    ld.param.u64 %rd3, [OUT];
    ld.param.u32 %r1, [N];
    ld.param.u32 %r2, [K];
    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %tid.y;
    mov.u32 %r5, %tid.z;
    mov.u32 %r6, %ntid.x;
    mov.u32 %r7, %ntid.y;
    mov.u32 %r8, %ntid.z;
    mad.lo.u32 %r9, %r5, %r7, %r4;
    mad.lo.u32 %r9, %r9, %r6, %r3;
    mul.lo.u32 %r10, %r6, %r7;
    mul.lo.u32 %r10, %r10, %r8;
    mov.u32 %r11, %ctaid.x;
    mov.u32 %r12, %ctaid.y;
    mov.u32 %r13, %ctaid.z;
    mov.u32 %r14, %nctaid.x;
    mov.u32 %r15, %nctaid.y;
    mov.u32 %r16, %nctaid.z;
    mad.lo.u32 %r17, %r13, %r15, %r12;
    mad.lo.u32 %r17, %r17, %r14, %r11;
    mul.lo.u32 %r18, %r14, %r15;
    mul.lo.u32 %r18, %r18, %r16;
    mul.lo.u32 %r18, %r18, %r10;
    mad.lo.u32 %r19, %r17, %r10, %r9;
    mov.u32 %r20, 0;
LOOP:
    setp.ge.s32 %p1, %r20, %r2;
    @%p1 bra DONE;
    setp.ge.s32 %p2, %r19, %r1;
    @%p2 bra NEXT;
    mul.wide.s32 %rd4, %r19, 4;
    add.u64 %rd5, %rd1, %rd4;
    ld.global.f32 %f1, [%rd5];
    add.u64 %rd6, %rd2, %rd4;
    ld.global.u32 %r21, [%rd6];
    mul.wide.s32 %rd7, %r21, 4;
    add.u64 %rd8, %rd3, %rd7;
    red.global.add.f32 [%rd8], %f1;
NEXT:
    add.u32 %r19, %r19, %r18;
    add.u32 %r20, %r20, 1;
    bra LOOP;
DONE:
    ret;
}
"#;

fn scratch(tag: &str) -> PathBuf {
    pinned::pinned_scratch(&format!("zdscat_{tag}"), pinned::SM_PINNED)
}

/// Compiles `src` (`args` added), returning the PTX path or the diagnostics.
fn try_compile(tag: &str, src: &str, args: &[&str]) -> Result<String, String> {
    let dir = scratch(tag);
    let path = dir.join(format!("{tag}.ysu"));
    std::fs::write(&path, src).unwrap();
    let out_path = path.with_extension("ptx");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .args(args)
        .args(["-o", out_path.to_str().unwrap()])
        .current_dir(&dir)
        .output()
        .expect("run Y");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    if out.status.success() && out_path.is_file() {
        Ok(out_path.display().to_string())
    } else {
        Err(text)
    }
}

fn compile(tag: &str, src: &str) -> String {
    try_compile(tag, src, &["--emit-ptx"]).unwrap_or_else(|e| panic!("{tag}: {e}"))
}

const BLOCKS: [(u32, u32, u32); 14] = [
    (1, 1, 1), (7, 1, 1), (31, 1, 1), (32, 1, 1), (33, 1, 1), (64, 1, 1), (100, 1, 1), (256, 1, 1),
    (1000, 1, 1), (1024, 1, 1), (8, 4, 2), (5, 3, 3), (32, 32, 1), (3, 1, 11),
];
const GRIDS: [(u32, u32, u32); 7] = [(1, 1, 1), (2, 1, 1), (3, 1, 1), (17, 1, 1), (128, 1, 1), (1000, 1, 1), (4, 3, 2)];

fn all_shapes() -> Vec<[u32; 6]> {
    BLOCKS
        .iter()
        .flat_map(|b| GRIDS.iter().map(move |g| [b.0, b.1, b.2, g.0, g.1, g.2]))
        .collect()
}

/// FNV-1a over `bytes`: one number names a whole `OUT`.
fn digest(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

/// One launch set: the kernel, its inputs and `ACC` at the first launch.
struct Case<'a> {
    ptx: &'a str,
    values: Vec<u8>,
    elem: usize,
    idx: Vec<i32>,
    jdx: Vec<i32>,
    totals: i32,
    ticket: i64,
    /// The byte `OUT` starts as: 0x55 shows a total that was never written;
    /// the f32 control adds into it, so it starts at zero.
    out_fill: u8,
    /// The launches per shape, in order: `k` (the PTX's kernel `k`), and for
    /// a `drift_close` kernel `c` (`drift_convert` into `OUT`), `c-1` (the
    /// same, told one total fewer) and `ci` (converting into `I64`s).
    plan: &'a str,
}

/// The child: run the plan's launches once per shape and print, for each,
/// the digest of `OUT` and whether `ACC` is all zero afterwards; or
/// `TRAPPED`.
fn child_run() -> Option<()> {
    let dir = PathBuf::from(std::env::var("Y_ZD_SCATTER_CASE").ok()?);
    let read = |name: &str| std::fs::read(dir.join(name)).unwrap();
    let meta = String::from_utf8(read("meta")).unwrap();
    let m: Vec<&str> = meta.split('|').collect();
    let (elem, totals, ticket): (usize, i32, i64) = (m[0].parse().unwrap(), m[1].parse().unwrap(), m[2].parse().unwrap());
    let fill: u8 = m[4].parse().unwrap();
    let plan: Vec<&str> = m[5].split(',').collect();
    let parse = |text: &str| -> Vec<[u32; 6]> {
        text.split(';').map(|s| s.split(',').map(|v| v.parse().unwrap()).collect::<Vec<u32>>().try_into().unwrap()).collect()
    };
    let (shapes, cshapes) = (parse(m[3]), parse(m[6]));
    let (values, idx, jdx) = (read("values"), read("idx"), read("jdx"));
    let n = values.len() / elem;
    // `drift_finish` keeps a ticket after the totals; `drift_close` a header
    // of seven words.
    let converts = plan.iter().any(|p| p.starts_with('c'));
    let words = 2 * totals.max(0) as usize + if converts { 7 } else { 1 };
    let ctx = y::cuda_runtime::CudaContext::new().expect("a context");
    let ptx = String::from_utf8(read("k.ptx")).unwrap();
    let module = ctx.load_ptx(&ptx, "k").expect("load");
    let convert = if converts { Some(ctx.load_ptx(&ptx, "c").expect("load c")) } else { None };
    let convert_i64 = if plan.contains(&"ci") { Some(ctx.load_ptx(&ptx, "ci").expect("load ci")) } else { None };
    let upload = |bytes: &[u8]| {
        let b = ctx.alloc(bytes.len().max(8)).unwrap();
        ctx.memcpy_htod_at(&b, 0, bytes).unwrap();
        b
    };
    let (x, i, j) = (upload(&values), upload(&idx), upload(&jdx));
    let mut acc0 = vec![0i64; words];
    acc0[2 * totals.max(0) as usize] = ticket;
    let acc = upload(&acc0.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>());
    let out_len = (totals.max(0) as usize * 8).max(8);
    for (s, sh) in shapes.iter().enumerate() {
        let all = (sh[0] * sh[1] * sh[2] * sh[3] * sh[4] * sh[5]) as usize;
        let rounds = n.div_ceil(all);
        let out = upload(&vec![fill; out_len]);
        // A `u64` holding an `I32` argument: CUDA reads the parameter's four
        // bytes, which are its low ones.
        let nt = |less: i32| (totals - less) as u32 as u64;
        for step in &plan {
            let (kernel, args, sh) = match *step {
                "k" => (&module, vec![x.device_ptr(), i.device_ptr(), j.device_ptr(), acc.device_ptr(), out.device_ptr(), n as u64, rounds as u64, nt(0)], *sh),
                "c" => (convert.as_ref().unwrap(), vec![acc.device_ptr(), out.device_ptr(), nt(0)], cshapes[s]),
                "c-1" => (convert.as_ref().unwrap(), vec![acc.device_ptr(), out.device_ptr(), nt(1)], cshapes[s]),
                "ci" => (convert_i64.as_ref().unwrap(), vec![acc.device_ptr(), out.device_ptr(), nt(0)], cshapes[s]),
                other => panic!("plan step {other}"),
            };
            if let Err(e) = ctx.launch(kernel, (sh[3], sh[4], sh[5]), (sh[0], sh[1], sh[2]), 0, &args).and_then(|_| ctx.synchronize()) {
                println!("TRAPPED at shape {s} {sh:?}, step {step}: {e}");
                return Some(());
            }
        }
        let mut got = vec![0u8; out_len];
        ctx.memcpy_dtoh_at(&mut got, &out, 0).unwrap();
        let mut after = vec![0u8; words * 8];
        ctx.memcpy_dtoh_at(&mut after, &acc, 0).unwrap();
        let zero = after.iter().all(|&b| b == 0);
        println!("SHAPE {s} OUT {} ACC {}", digest(&got[..totals.max(0) as usize * elem]), if zero { "zero" } else { "left" });
    }
    Some(())
}

/// What a child reported for each shape: `(digest of OUT, ACC left zero)`,
/// or `Err` naming the shape that trapped. `None` without a GPU (a reported
/// skip).
type Launches = Result<Vec<(u64, bool)>, String>;

/// The shapes `drift_convert` runs at, one per launch shape in turn: from one
/// thread to more threads than there are totals.
const CONVERT_SHAPES: [[u32; 6]; 7] = [
    [1, 1, 1, 1, 1, 1], [33, 1, 1, 2, 1, 1], [256, 1, 1, 17, 1, 1], [1024, 1, 1, 1000, 1, 1],
    [8, 4, 2, 4, 3, 2], [7, 1, 1, 1, 1, 1], [128, 1, 1, 64, 1, 1],
];

fn launch(test: &str, tag: &str, case: &Case, shapes: &[[u32; 6]]) -> Option<Launches> {
    if y::cuda_runtime::CudaContext::new().is_none() {
        eprintln!("SKIP: no CUDA device, so no kernel was run");
        return None;
    }
    let dir = scratch(&format!("run_{tag}"));
    let ints = |v: &[i32]| v.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>();
    std::fs::copy(case.ptx, dir.join("k.ptx")).unwrap();
    std::fs::write(dir.join("values"), &case.values).unwrap();
    std::fs::write(dir.join("idx"), ints(&case.idx)).unwrap();
    std::fs::write(dir.join("jdx"), ints(&case.jdx)).unwrap();
    let text = |shapes: &mut dyn Iterator<Item = [u32; 6]>| {
        shapes.map(|s| s.iter().map(u32::to_string).collect::<Vec<_>>().join(",")).collect::<Vec<_>>().join(";")
    };
    // One shape: the stop cases convert at the same shape; many: in turn.
    let cshapes = if shapes.len() == 1 { text(&mut shapes.iter().copied()) } else { text(&mut (0..shapes.len()).map(|s| CONVERT_SHAPES[s % 7])) };
    let meta = format!("{}|{}|{}|{}|{}|{}|{}", case.elem, case.totals, case.ticket, text(&mut shapes.iter().copied()), case.out_fill, case.plan, cshapes);
    std::fs::write(dir.join("meta"), meta).unwrap();
    let child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--nocapture"])
        .env("Y_ZD_SCATTER_CASE", &dir)
        .output()
        .expect("run the launches in a child");
    let text = String::from_utf8_lossy(&child.stdout).into_owned();
    if let Some(line) = text.lines().find(|l| l.starts_with("TRAPPED")) {
        return Some(Err(line.to_string()));
    }
    let rows: Vec<(u64, bool)> = text
        .lines()
        .filter(|l| l.starts_with("SHAPE "))
        .map(|l| {
            let w: Vec<&str> = l.split(' ').collect();
            (w[3].parse().unwrap(), w[5] == "zero")
        })
        .collect();
    assert_eq!(rows.len(), shapes.len(), "the child neither finished nor trapped: {text}");
    Some(Ok(rows))
}

/// round(x * 2^32), ties away from zero: the documented term.
fn q(x: f32) -> i128 {
    let y = x as f64 * 4294967296.0;
    (y.abs() + 0.5).floor().copysign(y) as i128
}

/// A total as `OUT` holds it: the f32 nearest to S * 2^-32, through f64.
fn read_back(s: i128) -> [u8; 4] {
    let s = i64::try_from(s).expect("the reference total fits 64 bits");
    ((s as f64 * 2f64.powi(-32)) as f32).to_le_bytes()
}

/// The digest of `OUT` computed from the definition: `+x` of every element
/// to its `idx` total and, with `jdx`, `-x` to its `jdx` total.
fn reference(xs: &[f32], idx: &[i32], jdx: Option<&[i32]>, totals: usize) -> u64 {
    let mut s = vec![0i128; totals];
    for (k, &x) in xs.iter().enumerate() {
        s[idx[k] as usize] += q(x);
        if let Some(j) = jdx {
            s[j[k] as usize] -= q(x);
        }
    }
    digest(&s.iter().flat_map(|&v| read_back(v)).collect::<Vec<u8>>())
}

fn f32_bytes(xs: &[f32]) -> Vec<u8> {
    xs.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i64_bytes(xs: &[i64]) -> Vec<u8> {
    xs.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn xorshift(seed: u32) -> impl FnMut() -> u32 {
    let mut x = seed;
    move || {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        x
    }
}

/// Values of mixed sign across eight decades, deterministic.
fn mixed(n: usize) -> Vec<f32> {
    let mut r = xorshift(0x9E37_79B9);
    (0..n)
        .map(|_| {
            let x = r();
            let mag = 10f32.powi((x % 9) as i32 - 6) * (1.0 + (x >> 8) as f32 / 16_777_216.0);
            if x & 1 == 0 { mag } else { -mag }
        })
        .collect()
}

fn indices(n: usize, totals: usize, seed: u32) -> Vec<i32> {
    let mut r = xorshift(seed);
    (0..n).map(|_| (r() as usize % totals) as i32).collect()
}

fn assert_every_shape(name: &str, rows: &[(u64, bool)], want: u64, shapes: &[[u32; 6]]) {
    let wrong: Vec<String> = rows
        .iter()
        .zip(shapes)
        .filter(|((got, zero), _)| *got != want || !zero)
        .map(|((got, zero), sh)| format!("  block {:?} grid {:?}: {got:#x}, ACC zero: {zero}", &sh[..3], &sh[3..]))
        .collect();
    assert!(wrong.is_empty(), "{name}: {} of {} launches differ from {want:#x}:\n{}", wrong.len(), rows.len(), wrong.join("\n"));
}

#[test]
fn every_launch_shape_gives_the_exact_totals() {
    if child_run().is_some() {
        return;
    }
    let test = "every_launch_shape_gives_the_exact_totals";
    let shapes = all_shapes();
    let n = 1_000_003;
    let xs = mixed(n);
    let one = compile("narrow", &scatter("F32", NARROW));
    let two = compile("pairs", &pairs(NARROW));
    for totals in [3usize, 1000, 100_003] {
        let (idx, jdx) = (indices(n, totals, 7), indices(n, totals, 11));
        let case = Case { ptx: &one, values: f32_bytes(&xs), elem: 4, idx: idx.clone(), jdx: jdx.clone(), totals: totals as i32, ticket: 0, out_fill: 0x55, plan: "k" };
        let Some(r) = launch(test, &format!("one_{totals}"), &case, &shapes) else { return };
        assert_every_shape(&format!("scatter, {totals} totals"), &r.expect("no trap"), reference(&xs, &idx, None, totals), &shapes);
        let case = Case { ptx: &two, ..case };
        let r = launch(test, &format!("two_{totals}"), &case, &shapes).unwrap();
        assert_every_shape(&format!("pairs, {totals} totals"), &r.expect("no trap"), reference(&xs, &idx, Some(&jdx), totals), &shapes);
    }

    // Two kernels: `drift_close`, then `drift_convert` at its own shape.
    let closed = compile("closed", &scatter_closed("F32", NARROW));
    for totals in [1000usize, 100_003] {
        let idx = indices(n, totals, 7);
        let case = Case { ptx: &closed, values: f32_bytes(&xs), elem: 4, idx: idx.clone(), jdx: vec![0; n], totals: totals as i32, ticket: 0, out_fill: 0x55, plan: "k,c" };
        let r = launch(test, &format!("closed_{totals}"), &case, &shapes).unwrap();
        assert_every_shape(&format!("close and convert, {totals} totals"), &r.expect("no trap"), reference(&xs, &idx, None, totals), &shapes);
    }

    // The control: f32 atomics on the same scatter do not agree, so the
    // agreement above is the exact sums', not this input's.
    let dir = scratch("control");
    let ctl = dir.join("control.ptx");
    std::fs::write(&ctl, F32_ATOMICS).unwrap();
    let ctl = ctl.display().to_string();
    let case = Case { ptx: &ctl, values: f32_bytes(&xs), elem: 4, idx: indices(n, 1000, 7), jdx: vec![0; n], totals: 1000, ticket: 0, out_fill: 0, plan: "k" };
    let rows = launch(test, "control", &case, &shapes).unwrap().expect("no trap");
    let distinct: std::collections::BTreeSet<u64> = rows.iter().map(|r| r.0).collect();
    assert!(distinct.len() > 1, "the f32 control gave one answer over {} shapes: the input cannot fail the test", shapes.len());

    // Totals whose running sums pass 2^63 in Q32.32, in some orders, and
    // cancel: each group of nine goes to one total.
    let xs: Vec<f32> = [2.0e9f32, 2.0e9, 2.0e9, -2.0e9, -2.0e9, -2.0e9, 0.75, -1.5e9, 1.5e9].repeat(1000);
    let idx: Vec<i32> = (0..xs.len()).map(|k| (k / 9 % 4) as i32).collect();
    let wide = compile("wide", &scatter("F32", WIDE));
    let case = Case { ptx: &wide, values: f32_bytes(&xs), elem: 4, idx: idx.clone(), jdx: vec![0; xs.len()], totals: 4, ticket: 0, out_fill: 0x55, plan: "k" };
    let r = launch(test, "wide", &case, &shapes).unwrap();
    assert_every_shape("cancelling", &r.expect("no trap: every total is 187.5"), reference(&xs, &idx, None, 4), &shapes);

    // The integer path: terms near 2^62, totals small.
    let xs: Vec<i64> = [4_000_000_000_000_000_000i64, 4e18 as i64, 3e18 as i64, -4e18 as i64, -4e18 as i64, -3e18 as i64, 12345].repeat(500);
    let idx: Vec<i32> = (0..xs.len()).map(|k| (k / 7 % 5) as i32).collect();
    let mut want = vec![0i64; 5];
    for (k, &x) in xs.iter().enumerate() {
        want[idx[k] as usize] += x;
    }
    let int = compile("int", &scatter("I64", ""));
    let case = Case { ptx: &int, values: i64_bytes(&xs), elem: 8, idx, jdx: vec![0; xs.len()], totals: 5, ticket: 0, out_fill: 0x55, plan: "k" };
    let r = launch(test, "int", &case, &shapes).unwrap();
    assert_every_shape("i64", &r.expect("no trap"), digest(&i64_bytes(&want)), &shapes);
}

#[test]
fn programs_stop_in_every_launch_shape_or_in_none() {
    if child_run().is_some() {
        return;
    }
    let test = "programs_stop_in_every_launch_shape_or_in_none";
    // One thread; partial warps over three blocks; full warps; 3-D both.
    let shapes: [[u32; 6]; 4] = [[1, 1, 1, 1, 1, 1], [33, 1, 1, 3, 1, 1], [256, 1, 1, 17, 1, 1], [8, 4, 2, 4, 3, 2]];
    let narrow = compile("stop_narrow", &scatter("F32", NARROW));
    let wide = compile("stop_wide", &scatter("F32", WIDE));
    let int = compile("stop_int", &scatter("I64", ""));
    let first = compile("stop_first", &summed_first(NARROW));
    let big = 4_000_000_000_000_000_000i64;
    let f = |ptx: &str, xs: &[f32], idx: &[i32], totals: i32| -> (String, Vec<u8>, usize, Vec<i32>, i32) {
        (ptx.to_string(), f32_bytes(xs), 4, idx.to_vec(), totals)
    };
    let fw = |xs: &[f32], idx: &[i32], totals: usize| Some(reference(xs, idx, None, totals));
    // (kernel, values, element size, indices, totals, ticket at the launch,
    // the digest of OUT, or None for a stop)
    let mut cases: Vec<((String, Vec<u8>, usize, Vec<i32>, i32), i64, Option<u64>)> = vec![
        (f(&narrow, &[1.5, -0.25, 3.0], &[0, 1, 0], 2), 0, fw(&[1.5, -0.25, 3.0], &[0, 1, 0], 2)),
        // Exactly on the bound is inside it.
        (f(&narrow, &[5e5, 5e5, 1.0], &[1, 1, 0], 2), 0, fw(&[5e5, 5e5, 1.0], &[1, 1, 0], 2)),
        // One total out of range, the other fine; either sign.
        (f(&narrow, &[6e5, 6e5, 1.0], &[1, 1, 0], 2), 0, None),
        (f(&narrow, &[-6e5, -6e5, 1.0], &[1, 1, 0], 2), 0, None),
        (f(&narrow, &[1.0, f32::NAN, 2.0], &[0, 1, 0], 2), 0, None),
        (f(&narrow, &[f32::INFINITY], &[0], 2), 0, None),
        // An index past the totals, or before them. Index n is the ticket's
        // word, where a term traps anyway as a stale ticket; a term of zero
        // there would change nothing, so only the index test stops it.
        (f(&narrow, &[1.0, 2.0], &[0, 2], 2), 0, None),
        (f(&narrow, &[1.0, 0.0], &[0, 2], 2), 0, None),
        (f(&narrow, &[1.0, 2.0], &[0, 3], 2), 0, None),
        (f(&narrow, &[1.0, 2.0], &[0, -1], 2), 0, None),
        // A term past 2^63 in Q32.32 cannot be converted: 3.9e9 * 2^32.
        (f(&wide, &[3.9e9, -3.9e9], &[0, 0], 1), 0, None),
        (f(&wide, &[2e9, 2e9, -2e9], &[0, 0, 0], 1), 0, fw(&[2e9], &[0], 1)),
        // A thread holding 9e5 + 9e5 runs to 1.8e6, outside the bounds: only
        // the total is tested, and it is 0.
        (f(&first, &[9e5, 9e5, -9e5, -9e5], &[0; 4], 1), 0, fw(&[0.0], &[0], 1)),
        (f(&first, &[6e5, 6e5], &[0; 2], 1), 0, None),
        // An ACC left non-zero by the host: its ticket is past the last.
        (f(&narrow, &[1.0], &[0], 2), 5, None),
        // A total of 2^32 + 1 whose low word alone reads 1.0.
        (f(&narrow, &[1073741824.0, 1073741824.0, 1073741824.0, 1073741824.0, 1.0], &[0; 5], 2), 0, None),
        // No elements: the totals are written, as zero.
        (f(&narrow, &[], &[], 2), 0, fw(&[], &[], 2)),
        // A negative count names no total.
        (f(&narrow, &[], &[], -1), 0, None),
    ];
    // The integer path: a total past 64 bits, either sign; and back inside.
    for (xs, want) in [(vec![big, big, big], None), (vec![-big, -big, -big], None), (vec![big, big, big, -big, -big], Some(big))] {
        let idx = vec![1; xs.len()];
        let want = want.map(|w| digest(&i64_bytes(&[0, w])));
        cases.push(((int.clone(), i64_bytes(&xs), 8, idx, 2), 0, want));
    }
    // Two kernels: the same stops, and every misuse of the pair.
    let closed = compile("stop_closed", &scatter_closed("F32", NARROW));
    let two: Vec<(&str, Vec<f32>, Vec<i32>, i32, i64, Option<u64>)> = vec![
        ("k,c", vec![1.5, -0.25, 3.0], vec![0, 1, 0], 2, 0, fw(&[1.5, -0.25, 3.0], &[0, 1, 0], 2)),
        ("k,c", vec![5e5, 5e5, 1.0], vec![1, 1, 0], 2, 0, fw(&[5e5, 5e5, 1.0], &[1, 1, 0], 2)),
        ("k,c", vec![6e5, 6e5, 1.0], vec![1, 1, 0], 2, 0, None),
        ("k,c", vec![-6e5, -6e5, 1.0], vec![1, 1, 0], 2, 0, None),
        ("k,c", vec![1.0, f32::NAN, 2.0], vec![0, 1, 0], 2, 0, None),
        ("k,c", vec![1.0, 0.0], vec![0, 2], 2, 0, None),
        ("k,c", vec![], vec![], 2, 0, fw(&[], &[], 2)),
        // ACC left non-zero by the host.
        ("k,c", vec![1.0], vec![0], 2, 5, None),
        // Converted with nothing closed, twice, with one total fewer, or
        // into I64; closed twice.
        ("c", vec![1.0], vec![0], 2, 0, None),
        ("k,c,c", vec![1.0], vec![0], 2, 0, None),
        ("k,c-1", vec![1.0], vec![0], 2, 0, None),
        ("k,ci", vec![1.0], vec![0], 2, 0, None),
        ("k,k,c", vec![1.0], vec![0], 2, 0, None),
        // A negative count names no total.
        ("k,c", vec![], vec![], -1, 0, None),
        // A total of 2^32 + 1 is past 64 bits in Q32.32, and its low word
        // alone reads 1.0, inside the bounds: only the 64-bit test stops it.
        ("k,c", vec![1073741824.0; 4].into_iter().chain([1.0]).collect(), vec![0; 5], 2, 0, None),
    ];
    let mut bad = Vec::new();
    for (c, (plan, xs, idx, totals, ticket, want)) in two.iter().enumerate() {
        let case = Case { ptx: &closed, values: f32_bytes(xs), elem: 4, jdx: vec![0; idx.len()], idx: idx.clone(), totals: *totals, ticket: *ticket, out_fill: 0x55, plan: *plan };
        for sh in &shapes {
            let Some(r) = launch(test, &format!("stop_two_{c}"), &case, &[*sh]) else { return };
            match (&r, want) {
                (Ok(rows), Some(w)) if rows[0].0 == *w && rows[0].1 => {}
                (Err(_), None) => {}
                _ => bad.push(format!("  two kernels, case {c} ({plan}), block {:?} grid {:?}: got {r:?}, want {want:?}", &sh[..3], &sh[3..])),
            }
        }
    }
    for (c, ((ptx, values, elem, idx, totals), ticket, want)) in cases.iter().enumerate() {
        let case = Case { ptx, values: values.clone(), elem: *elem, jdx: vec![0; idx.len()], idx: idx.clone(), totals: *totals, ticket: *ticket, out_fill: 0x55, plan: "k" };
        for sh in &shapes {
            let Some(r) = launch(test, &format!("stop_{c}"), &case, &[*sh]) else { return };
            match (&r, want) {
                (Ok(rows), Some(w)) if rows[0].0 == *w && rows[0].1 => {}
                (Err(_), None) => {}
                _ => bad.push(format!("  case {c}, block {:?} grid {:?}: got {r:?}, want {want:?}", &sh[..3], &sh[3..])),
            }
        }
    }
    assert!(bad.is_empty(), "a launch stopped (or ran) where the others did not:\n{}", bad.join("\n"));
}

/// The refusals, hermetic: each would make a total wrong or launch-dependent.
#[test]
fn what_would_make_a_total_wrong_or_launch_dependent_is_refused() {
    let k = |body: &str| {
        format!(
            "kernel k(X: GlobalMemory<F32>, IDX: GlobalMemory<I32>, ACC: GlobalMemory<I64>, OUT: GlobalMemory<F32>, P: GlobalMemory<I32>, NT: I32, R: F32) {{
    {body}
}}

fn main() {{}}
"
        )
    };
    let decl = format!("@ZeroDrift\n    {NARROW}\n    let mut acc: F32 = 0.0;\n    acc += X[0];");
    let hand = "drift_add_at(ACC, IDX[0], acc);";
    let fin = "drift_finish(ACC, OUT, NT);";
    let cases: Vec<(&str, String, &str)> = vec![
        ("finish_branch", k(&format!("{decl}\n    {hand}\n    if P[0] > 0 {{ {fin} }}")), "`drift_finish` must be a statement of the kernel's body itself"),
        ("finish_loop", k(&format!("{decl}\n    {hand}\n    @invariant(j >= 0)\n    for j in 0..2 step 1 {{ {fin} }}")), "`drift_finish` must be a statement of the kernel's body itself"),
        ("finish_twice", k(&format!("{decl}\n    {hand}\n    {fin}\n    {fin}")), "is called again on line"),
        ("no_finish", k(&format!("{decl}\n    {hand}")), "Without `drift_finish`"),
        ("nested_finish_only", k(&format!("{decl}\n    {hand}\n    if P[0] > 0 {{ {fin} }}\n    if P[1] > 0 {{ {fin} }}")), "`drift_finish` must be a statement of the kernel's body itself"),
        ("add_after_finish", k(&format!("{decl}\n    {fin}\n    {hand}")), "after `drift_finish` on line"),
        ("other_buffer", k(&format!("{decl}\n    drift_add_at(IDX, 0, acc);\n    {fin}")), "and the kernel's `drift_finish` finishes `ACC`"),
        ("not_declaring_block", k(&format!("{decl}\n    if P[0] > 0 {{ {hand} }}\n    {fin}")), "inside a block that does not declare it"),
        ("term_after_last", k(&format!("{decl}\n    {hand}\n    acc += X[1];\n    {fin}")), "is not called after this term is added"),
        ("term_after_last_nested", k(&format!("{decl}\n    {hand}\n    if P[0] > 0 {{ acc += X[1]; }}\n    {fin}")), "is not called after this term is added"),
        ("nonzero_start", k(&format!("@ZeroDrift\n    {NARROW}\n    let mut acc: F32 = 1.0;\n    {hand}\n    {fin}")), "does not start at zero"),
        ("read", k(&format!("{decl}\n    OUT[1] = acc;\n    {hand}\n    {fin}")), "is one thread's share of the total"),
        ("count_loaded", k(&format!("{decl}\n    {hand}\n    let n: I32 = P[0];\n    drift_finish(ACC, OUT, n);")), "scalar integer parameter of the kernel that the body never assigns"),
        ("count_assigned", k(&format!("{decl}\n    {hand}\n    NT = 3;\n    {fin}")), "scalar integer parameter of the kernel that the body never assigns"),
        ("count_float", k(&format!("{decl}\n    {hand}\n    drift_finish(ACC, OUT, R);")), "scalar integer parameter of the kernel that the body never assigns"),
        ("count_thread", k(&format!("{decl}\n    {hand}\n    drift_finish(ACC, OUT, thread_idx_x());")), "scalar integer parameter of the kernel that the body never assigns"),
        ("with_add_to", k(&format!("{decl}\n    {hand}\n    drift_add_to(ACC, OUT, acc);\n    {fin}")), "Each block takes one ticket, for one of them"),
        ("formats_differ", k(&format!("{decl}\n    {hand}\n    @ZeroDrift\n    {WIDE}\n    let mut b: F32 = 0.0;\n    b += X[1];\n    drift_add_at(ACC, 1, b);\n    {fin}")), "with another type, representation or `@bounds`"),
        ("finish_alone", k(&format!("{decl}\n    {fin}")), "no `drift_add_at` before it"),
        ("return_before", k(&format!("{decl}\n    {hand}\n    if P[0] > 3 {{ return; }}\n    {fin}")), "can take threads away first"),
        ("not_drift", k(&format!("let mut acc: F32 = 0.0;\n    acc += X[0];\n    {hand}\n    {fin}")), "is not a `@ZeroDrift` accumulator"),
        ("expression", k(&format!("{decl}\n    drift_add_at(ACC, 0, X[0]);\n    {fin}")), "takes a `@ZeroDrift` accumulator as its third argument"),
        ("acc_type", k(&format!("{decl}\n    drift_add_at(X, 0, acc);\n    drift_finish(X, OUT, NT);")), "give a GlobalMemory<I64>"),
        ("out_type", k(&format!("{decl}\n    {hand}\n    drift_finish(ACC, ACC, NT);")), "give a GlobalMemory<F32>"),
        ("float_index", k(&format!("{decl}\n    drift_add_at(ACC, X[1], acc);\n    {fin}")), "its index and count are integers"),
        ("arity", k(&format!("{decl}\n    drift_add_at(ACC, acc);\n    {fin}")), "drift_add_at"),
        ("close_branch", k(&format!("{decl}\n    {hand}\n    if P[0] > 0 {{ drift_close(ACC, NT); }}")), "`drift_close` must be a statement of the kernel's body itself"),
        ("close_twice", k(&format!("{decl}\n    {hand}\n    drift_close(ACC, NT);\n    drift_close(ACC, NT);")), "is called again on line"),
        ("close_and_finish", k(&format!("{decl}\n    {hand}\n    drift_close(ACC, NT);\n    {fin}")), "and `drift_finish` both end this kernel's hand-overs"),
        ("close_count", k(&format!("{decl}\n    {hand}\n    drift_close(ACC, R);")), "scalar integer parameter of the kernel that the body never assigns"),
        ("close_return", k(&format!("{decl}\n    {hand}\n    if P[0] > 3 {{ return; }}\n    drift_close(ACC, NT);")), "can take threads away first"),
        ("close_arity", k(&format!("{decl}\n    {hand}\n    drift_close(ACC);")), "drift_close"),
        ("convert_branch", k("if P[0] > 0 { drift_convert(ACC, OUT, NT); }"), "`drift_convert` must be a statement of the kernel's body itself"),
        ("convert_twice", k("drift_convert(ACC, OUT, NT);\n    drift_convert(ACC, OUT, NT);"), "is called again on line"),
        ("convert_with_hand_over", k(&format!("{decl}\n    {hand}\n    drift_convert(ACC, OUT, NT);")), "converts totals with `drift_convert` and hands partials"),
        ("convert_count", k("drift_convert(ACC, OUT, R);"), "scalar integer parameter of the kernel that the body never assigns"),
        ("convert_return", k("if P[0] > 3 { return; }\n    drift_convert(ACC, OUT, NT);"), "can take threads away first"),
        ("convert_out_type", k("drift_convert(ACC, P, NT);"), "give a GlobalMemory<F32> for fixed-point ones"),
        ("convert_acc_type", k("drift_convert(X, OUT, NT);"), "give the GlobalMemory<I64> a `drift_close` completed"),
    ];
    let mut bad = Vec::new();
    for (tag, src, want) in &cases {
        match try_compile(&format!("refuse_{tag}"), src, &["--emit-ptx"]) {
            Ok(_) => bad.push(format!("  {tag}: compiled")),
            Err(text) if !text.contains(want) => bad.push(format!("  {tag}: refused without `{want}`:\n{text}")),
            Err(_) => {}
        }
    }
    // The host backends have no grid: refused by name, never compiled.
    let ok = scatter("F32", NARROW);
    for (tag, args, want) in [("llvm", &[][..], "has no host lowering"), ("cpu", &["--emit-cpu"][..], "has no CPU lowering")] {
        for (form, src) in [("finish", ok.clone()), ("close", scatter_closed("F32", NARROW))] {
            match try_compile(&format!("host_{tag}_{form}"), &src, args) {
                Ok(_) => bad.push(format!("  {tag} ({form}): compiled")),
                Err(text) if !text.contains(want) || !text.contains("drift_") => {
                    bad.push(format!("  {tag} ({form}): refused without `{want}`:\n{text}"))
                }
                Err(_) => {}
            }
        }
    }
    // The JIT runs `main` on the CPU, and takes no `-o`.
    let dir = scratch("host_jit");
    let path = dir.join("host_jit.ysu");
    std::fs::write(&path, &ok).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_Y")).arg(&path).arg("--jit").current_dir(&dir).output().expect("run Y");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    if out.status.success() || !text.contains("has no host lowering") || !text.contains("drift_") {
        bad.push(format!("  jit: not refused by name:\n{text}"));
    }
    assert!(bad.is_empty(), "{} of {} forms were not refused as they should be:\n{}", bad.len(), cases.len() + 5, bad.join("\n"));
}

/// Hand-overs in loops and branches, declared in the block that hands them
/// over, compile: the rules refuse only what would lose or move a term.
#[test]
fn hand_overs_in_loops_and_branches_compile() {
    let body = format!(
        "{}\n            if X[k] > 0.0 {{\n                @ZeroDrift\n                {NARROW}\n                let mut pos: F32 = 0.0;\n                pos += X[k];\n                pos += X[k];\n                drift_add_at(ACC, IDX[k], pos);\n            }}\n            acc += X[k];\n            drift_add_at(ACC, JDX[k], acc);",
        drift_let("F32", NARROW)
    );
    compile("loops_branches", &kernel("F32", &body, "drift_finish(ACC, OUT, 7);"));
    compile("summed_first", &summed_first(NARROW));
}

/// The emitted shape, hermetic.
#[test]
fn a_hand_over_checks_its_index_and_carries_into_the_high_word() {
    let ptx = std::fs::read_to_string(compile("shape", &scatter("F32", NARROW))).unwrap();
    let start = ptx.find("one thread's partial").expect("the partial's declaration");
    let at = ptx.find("this thread's partial joins").expect("the hand-over");
    let fin = ptx.find("the grid's exact totals, once").expect("the finish");
    let body = &ptx[start..at];
    assert!(body.contains("add.cc.u64") && body.contains("addc.u64"), "the partial is not 128 bits:\n{body}");
    let hand = &ptx[at..fin];
    let lines: Vec<&str> = hand.lines().map(str::trim).collect();
    let trap = lines.iter().position(|l| l.ends_with("trap;")).expect("the index test");
    let atom = lines.iter().position(|l| l.starts_with("atom.global.add.u64")).expect("the atomic");
    assert!(trap < atom, "the index is tested after the atomic:\n{hand}");
    assert!(lines[..trap].iter().any(|l| l.starts_with("setp.lt.s64")) && lines[..trap].iter().any(|l| l.starts_with("setp.ge.s64")), "the index is not tested against both ends:\n{hand}");
    assert!(hand.contains("shl.b64") && hand.contains(", 4;"), "a total is not 16 bytes:\n{hand}");
    assert_eq!(hand.matches("red.global.add.u64").count(), 1, "the high word:\n{hand}");
    assert!(lines.iter().filter(|l| l.starts_with("mov.u64") && l.ends_with(", 0;")).count() == 2, "the partial is not set back to zero:\n{hand}");
    let finish = &ptx[fin..];
    assert!(finish.contains(".shared .align 4 .u32 drift_last;") && finish.matches("bar.sync 0;").count() == 2, "the last block's flag:\n{finish}");
    assert_eq!(finish.matches("atom.global.add.u64").count(), 1, "one ticket per block:\n{finish}");
    // The fences: a block's hand-overs before its ticket, every ticket before
    // the last block's reads. No launch shows a missing one on this GPU.
    let fence = finish.find("membar.gl;").expect("a fence");
    assert!(
        finish.matches("membar.gl;").count() == 2 && fence < finish.find("atom.global.add.u64").unwrap(),
        "the fences:\n{finish}"
    );
    // Two kernels: the closing block writes the header, then after a fence
    // the tag; the converter reads the tag and the count before any total,
    // and the rest of the header after a fence.
    let two = std::fs::read_to_string(compile("shape_closed", &scatter_closed("F32", NARROW))).unwrap();
    let close = &two[two.find("are complete once every block").expect("the close")..two.find(".entry c(").expect("kernel c")];
    let fence = close.rfind("membar.gl;").unwrap();
    assert!(
        close.matches("membar.gl;").count() == 3 && close[fence..].contains("+8], ") && close[..fence].contains("+40], "),
        "the tag is not written after the header and a fence:\n{close}"
    );
    let convert = &two[two.find("converted by every thread of the grid").expect("the convert")..two.find(".entry ci(").expect("kernel ci")];
    let loads: Vec<&str> = convert.lines().map(str::trim).filter(|l| l.starts_with("ld.volatile.global.u64")).collect();
    assert!(loads.len() >= 7 && loads[0].ends_with("+8];") && loads[1].ends_with("+16];"), "the tag and the count are not read first:\n{convert}");
    let first_total = convert.find("ld.volatile.global.u64").unwrap();
    assert!(convert[..convert.find("+24];").unwrap()].contains("membar.gl;") && first_total < convert.find("+24];").unwrap(), "the header after a fence:\n{convert}");
    assert!(convert.matches("bar.sync 0;").count() == 1 && convert.matches("atom.global.add.u64").count() == 1, "one ticket per block:\n{convert}");
    let Some(tool) = which("ptxas") else {
        eprintln!("SKIP: no ptxas, so the shape was not assembled");
        return;
    };
    for path in [compile("asm_int", &scatter("I64", "")), compile("asm_pairs", &pairs(WIDE)), compile("asm_closed", &scatter_closed("I64", ""))] {
        let out = Command::new(&tool).args(["-arch=sm_80", &path, "-o", "/dev/null"]).output().unwrap();
        assert!(out.status.success(), "ptxas: {}", String::from_utf8_lossy(&out.stderr));
    }
}

fn which(tool: &str) -> Option<PathBuf> {
    std::env::var_os("PATH")?
        .to_str()?
        .split(':')
        .map(|d| Path::new(d).join(tool))
        .chain([PathBuf::from("/opt/cuda/bin").join(tool), PathBuf::from("/usr/local/cuda/bin").join(tool)])
        .find(|p| p.is_file())
}
