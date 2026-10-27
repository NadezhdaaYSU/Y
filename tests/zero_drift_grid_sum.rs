//! `drift_add_to(ACC, OUT, acc)` sums every thread's `@ZeroDrift` partial into
//! one exact total for the whole grid, and the total is the same bits for
//! every launch shape.
//!
//! Each property is checked by running the kernel:
//!
//! * **The total.** One kernel, 98 launch shapes (block sizes that fill no
//!   warp, fill some, and three-dimensional ones; one to a thousand blocks,
//!   in three dimensions too): every launch stores the bits computed here from
//!   the definition, and leaves `ACC` zero for the next launch. That includes
//!   inputs whose partials pass 2^63 and cancel, which one 64-bit register
//!   could not hold. A control kernel that sums the same partitions in `f32`
//!   must give more than one answer over the same shapes, or the input could
//!   not tell an exact sum from a lucky one.
//! * **The stops.** A program stops when a term or the total is out of
//!   range, and then in every launch shape; otherwise in none, even where
//!   one thread's running sum leaves the declared range. Each launch runs in
//!   a child process, because a trap leaves the device unusable for its
//!   process.
//!
//! Hermetic checks pin the refusals that keep the total exact (the call in a
//! branch, in a loop, twice, after a `return`; a partial that is read, added
//! to afterwards or does not start at zero; buffers of the wrong type; the
//! host backends) and the emitted shape.
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

const NARROW: &str = "@bounds(min=-1000000.0, max=1000000.0)";
// 2.1e9 in Q32.32 is 9.0e18, just inside 2^63.
const WIDE: &str = "@bounds(min=-2100000000.0, max=2100000000.0)";

/// Thread `t` of block `b`, both flattened over three dimensions, adds `X[i]`
/// for `i = b * threads + t + j * blocks * threads`, `j < P[0]`, `i < P[1]`.
fn grid_sum(acc_ty: &str, bounds: &str, tail: &str) -> String {
    let (zero, elem) = if acc_ty == "I64" { ("0", "I64") } else { ("0.0", "F32") };
    format!(
        "kernel k(X: GlobalMemory<{elem}>, ACC: GlobalMemory<I64>, OUT: GlobalMemory<{elem}>, P: GlobalMemory<I32>) {{
    @ZeroDrift
    {bounds}
    let mut acc: {acc_ty} = {zero};
{}
    {tail}
}}

fn main() {{}}
",
        loop_body()
    )
}

fn loop_body() -> &'static str {
    "    let t: I32 = thread_idx_x() + block_dim_x() * (thread_idx_y() + block_dim_y() * thread_idx_z());
    let threads: I32 = block_dim_x() * block_dim_y() * block_dim_z();
    let b: I32 = block_idx_x() + grid_dim_x() * (block_idx_y() + grid_dim_y() * block_idx_z());
    let stride: I32 = grid_dim_x() * grid_dim_y() * grid_dim_z() * threads;
    let rounds: I32 = P[0];
    let n: I32 = P[1];
    @invariant(j >= 0)
    for j in 0..rounds step 1 {
        let i: I32 = b * threads + t + j * stride;
        if i < n {
            acc += X[i];
        }
    }"
}

/// The same partition summed in plain `f32`, each thread's partial stored at
/// its flat index: the host adds them in order.
fn f32_partials() -> String {
    format!(
        "kernel k(X: GlobalMemory<F32>, ACC: GlobalMemory<I64>, OUT: GlobalMemory<F32>, P: GlobalMemory<I32>) {{
    let mut acc: F32 = 0.0;
{}
    OUT[b * threads + t] = acc;
}}

fn main() {{}}
",
        loop_body()
    )
}

fn sum_kernel(acc_ty: &str, bounds: &str) -> String {
    grid_sum(acc_ty, bounds, "drift_add_to(ACC, OUT, acc);")
}

fn scratch(tag: &str) -> PathBuf {
    pinned::pinned_scratch(&format!("zdgrid_{tag}"), pinned::SM_PINNED)
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

/// The child: launch the PTX once per shape and print, for each, the bits
/// of `OUT[0]` (`total`) or the in-order f32 sum of every thread's slot
/// (`partials`), and `ACC` afterwards; or `TRAPPED`.
fn child_run() -> Option<()> {
    let case = std::env::var("Y_ZD_GRID_CASE").ok()?;
    let parts: Vec<&str> = case.split('|').collect();
    let (ptx_path, mode, values_path, acc_init, elem) = (parts[0], parts[1], parts[2], parts[3], parts[4]);
    let elem: usize = elem.parse().unwrap();
    let shapes: Vec<[u32; 6]> = parts[5]
        .split(';')
        .map(|s| s.split(',').map(|v| v.parse().unwrap()).collect::<Vec<u32>>().try_into().unwrap())
        .collect();
    let acc0: Vec<u8> = acc_init.split(',').flat_map(|w| w.parse::<i64>().unwrap().to_le_bytes()).collect();
    let values = std::fs::read(values_path).unwrap();
    let n = values.len() / elem;
    let ctx = y::cuda_runtime::CudaContext::new().expect("a context");
    let module = ctx.load_ptx(&std::fs::read_to_string(ptx_path).unwrap(), "k").expect("load");
    let x = ctx.alloc(values.len().max(8)).unwrap();
    ctx.memcpy_htod_at(&x, 0, &values).unwrap();
    let acc = ctx.alloc(24).unwrap();
    ctx.memcpy_htod_at(&acc, 0, &acc0).unwrap();
    let p = ctx.alloc(8).unwrap();
    for (s, sh) in shapes.iter().enumerate() {
        let threads = (sh[0] * sh[1] * sh[2]) as usize;
        let all = threads * (sh[3] * sh[4] * sh[5]) as usize;
        let rounds = n.div_ceil(all) as i32;
        let words: Vec<u8> = [rounds, n as i32].iter().flat_map(|w| w.to_le_bytes()).collect();
        ctx.memcpy_htod_at(&p, 0, &words).unwrap();
        let out_len = if mode == "partials" { all * 4 } else { 8 };
        let out = ctx.alloc(out_len).unwrap();
        ctx.memcpy_htod_at(&out, 0, &vec![0x55u8; out_len]).unwrap();
        let args = [x.device_ptr(), acc.device_ptr(), out.device_ptr(), p.device_ptr()];
        if let Err(e) = ctx.launch(&module, (sh[3], sh[4], sh[5]), (sh[0], sh[1], sh[2]), 0, &args).and_then(|_| ctx.synchronize()) {
            println!("TRAPPED at shape {s} {sh:?}: {e}");
            return Some(());
        }
        let mut got = vec![0u8; out_len];
        ctx.memcpy_dtoh_at(&mut got, &out, 0).unwrap();
        let result: u64 = if mode == "partials" {
            got.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).fold(0.0f32, |a, v| a + v).to_bits() as u64
        } else if elem == 4 {
            u32::from_le_bytes(got[..4].try_into().unwrap()) as u64
        } else {
            u64::from_le_bytes(got[..8].try_into().unwrap())
        };
        let mut after = vec![0u8; 24];
        ctx.memcpy_dtoh_at(&mut after, &acc, 0).unwrap();
        let after: Vec<String> = after.chunks(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()).to_string()).collect();
        println!("SHAPE {s} RESULT {result} ACC {}", after.join(","));
    }
    Some(())
}

/// What a child reported for each shape: `(result bits, ACC afterwards)`,
/// or `Err` naming the shape that trapped. `None` without a GPU (a reported
/// skip).
type Launches = Result<Vec<(u64, String)>, String>;

fn launch(test: &str, ptx: &str, mode: &str, values: &[u8], elem: usize, acc: [i64; 3], shapes: &[[u32; 6]]) -> Option<Launches> {
    if y::cuda_runtime::CudaContext::new().is_none() {
        eprintln!("SKIP: no CUDA device, so no kernel was run");
        return None;
    }
    let values_path = Path::new(ptx).with_extension(format!("{}.bin", shapes.len()));
    std::fs::write(&values_path, values).unwrap();
    let shapes_text: Vec<String> =
        shapes.iter().map(|s| s.iter().map(u32::to_string).collect::<Vec<_>>().join(",")).collect();
    let acc_text: Vec<String> = acc.iter().map(i64::to_string).collect();
    let case = [ptx, mode, values_path.to_str().unwrap(), &acc_text.join(","), &elem.to_string(), &shapes_text.join(";")].join("|");
    let child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--nocapture"])
        .env("Y_ZD_GRID_CASE", case)
        .output()
        .expect("run the launches in a child");
    let text = String::from_utf8_lossy(&child.stdout).into_owned();
    if let Some(line) = text.lines().find(|l| l.starts_with("TRAPPED")) {
        return Some(Err(line.to_string()));
    }
    let rows: Vec<(u64, String)> = text
        .lines()
        .filter(|l| l.starts_with("SHAPE "))
        .map(|l| {
            let w: Vec<&str> = l.split(' ').collect();
            (w[3].parse().unwrap(), w[5].to_string())
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

/// The total as `OUT` holds it: the f32 nearest to S * 2^-32, through f64.
fn reference_f32(xs: &[f32]) -> u64 {
    let s: i128 = xs.iter().map(|&x| q(x)).sum();
    let s = i64::try_from(s).expect("the reference total fits 64 bits");
    ((s as f64 * 2f64.powi(-32)) as f32).to_bits() as u64
}

fn f32_bytes(xs: &[f32]) -> Vec<u8> {
    xs.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i64_bytes(xs: &[i64]) -> Vec<u8> {
    xs.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A million values of mixed sign across eight decades, deterministic.
fn mixed(n: usize) -> Vec<f32> {
    let mut x: u32 = 0x9E37_79B9;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let mag = 10f32.powi((x % 9) as i32 - 6) * (1.0 + (x >> 8) as f32 / 16_777_216.0);
            if x & 1 == 0 { mag } else { -mag }
        })
        .collect()
}

fn assert_every_shape(name: &str, rows: &[(u64, String)], want: u64, shapes: &[[u32; 6]]) {
    let wrong: Vec<String> = rows
        .iter()
        .zip(shapes)
        .filter(|((got, acc), _)| *got != want || acc != "0,0,0")
        .map(|((got, acc), sh)| format!("  block {:?} grid {:?}: {got:#x}, ACC {acc}", &sh[..3], &sh[3..]))
        .collect();
    assert!(wrong.is_empty(), "{name}: {} of {} launches differ from {want:#x}:\n{}", wrong.len(), rows.len(), wrong.join("\n"));
}

#[test]
fn every_launch_shape_gives_the_exact_total() {
    if child_run().is_some() {
        return;
    }
    let test = "every_launch_shape_gives_the_exact_total";
    let shapes = all_shapes();
    let xs = mixed(1_000_003);
    let ptx = compile("narrow", &sum_kernel("F32", NARROW));
    let Some(r) = launch(test, &ptx, "total", &f32_bytes(&xs), 4, [0; 3], &shapes) else { return };
    assert_every_shape("mixed", &r.expect("no trap: every term and the total are in range"), reference_f32(&xs), &shapes);

    // The control: the same partitions in f32 do not agree, so agreement
    // above is the exact sum's and not this input's.
    let ctl = compile("control", &f32_partials());
    let rows = launch(test, &ctl, "partials", &f32_bytes(&xs), 4, [0; 3], &shapes).unwrap().expect("no trap");
    let distinct: std::collections::BTreeSet<u64> = rows.iter().map(|r| r.0).collect();
    assert!(distinct.len() > 1, "the f32 control gave one answer over {} shapes: the input cannot fail the test", shapes.len());

    // Partials past 2^63 in Q32.32 that cancel: a thread holding 2e9 + 2e9
    // holds 1.7e19, more than one 64-bit register.
    let xs: Vec<f32> = [2.0e9f32, 2.0e9, 2.0e9, -2.0e9, -2.0e9, -2.0e9, 0.75, -1.5e9, 1.5e9].repeat(1000);
    let ptx = compile("wide", &sum_kernel("F32", WIDE));
    let r = launch(test, &ptx, "total", &f32_bytes(&xs), 4, [0; 3], &shapes).unwrap();
    assert_every_shape("cancelling", &r.expect("no trap: the total is 750"), reference_f32(&xs), &shapes);

    // The integer path: terms near 2^62, partials past 2^63, total small.
    let xs: Vec<i64> = [4_000_000_000_000_000_000i64, 4e18 as i64, 3e18 as i64, -4e18 as i64, -4e18 as i64, -3e18 as i64, 12345].repeat(500);
    let ptx = compile("int", &sum_kernel("I64", ""));
    let r = launch(test, &ptx, "total", &i64_bytes(&xs), 8, [0; 3], &shapes).unwrap();
    assert_every_shape("i64", &r.expect("no trap"), xs.iter().sum::<i64>() as u64, &shapes);
}

#[test]
fn programs_stop_in_every_launch_shape_or_in_none() {
    if child_run().is_some() {
        return;
    }
    let test = "programs_stop_in_every_launch_shape_or_in_none";
    // One thread; partial warps over three blocks; full warps; 3-D both.
    let shapes: [[u32; 6]; 4] = [[1, 1, 1, 1, 1, 1], [33, 1, 1, 3, 1, 1], [256, 1, 1, 17, 1, 1], [8, 4, 2, 4, 3, 2]];
    let narrow = compile("stop_narrow", &sum_kernel("F32", NARROW));
    let wide = compile("stop_wide", &sum_kernel("F32", WIDE));
    let int = compile("stop_int", &sum_kernel("I64", ""));
    let big = 4_000_000_000_000_000_000i64;
    // (kernel, terms, ACC at the launch, the total, or None for a stop)
    let cases: Vec<(&str, Vec<u8>, usize, [i64; 3], Option<u64>)> = vec![
        (narrow.as_str(), f32_bytes(&[1.5, -0.25]), 4, [0; 3], Some(reference_f32(&[1.5, -0.25]))),
        // A thread holding both runs to 1.8e6, outside the bounds: only the
        // total is tested, and it is 0.
        (narrow.as_str(), f32_bytes(&[9e5, 9e5, -9e5, -9e5]), 4, [0; 3], Some(0)),
        // Exactly on the bound is inside it.
        (narrow.as_str(), f32_bytes(&[5e5, 5e5]), 4, [0; 3], Some(reference_f32(&[1e6]))),
        (narrow.as_str(), f32_bytes(&[6e5, 6e5]), 4, [0; 3], None),
        (narrow.as_str(), f32_bytes(&[-6e5, -6e5]), 4, [0; 3], None),
        (narrow.as_str(), f32_bytes(&[1.0, f32::NAN, 2.0]), 4, [0; 3], None),
        (narrow.as_str(), f32_bytes(&[f32::INFINITY]), 4, [0; 3], None),
        // A term past 2^63 in Q32.32 cannot be converted: 3.9e9 * 2^32.
        (wide.as_str(), f32_bytes(&[3.9e9, -3.9e9]), 4, [0; 3], None),
        (wide.as_str(), f32_bytes(&[2e9, 2e9, -2e9]), 4, [0; 3], Some(reference_f32(&[2e9]))),
        // The total past 64 bits, either sign; and back inside.
        (int.as_str(), i64_bytes(&[big, big, big]), 8, [0; 3], None),
        (int.as_str(), i64_bytes(&[-big, -big, -big]), 8, [0; 3], None),
        (int.as_str(), i64_bytes(&[big, big, big, -big, -big]), 8, [0; 3], Some(big as u64)),
        // An ACC left non-zero by the host: its ticket is past the last.
        (narrow.as_str(), f32_bytes(&[1.0]), 4, [0, 0, 5], None),
    ];
    let mut bad = Vec::new();
    for (i, (ptx, bytes, elem, acc, want)) in cases.iter().enumerate() {
        for sh in &shapes {
            let Some(r) = launch(test, ptx, "total", bytes, *elem, *acc, &[*sh]) else { return };
            match (&r, want) {
                (Ok(rows), Some(w)) if rows[0].0 == *w && rows[0].1 == "0,0,0" => {}
                (Err(_), None) => {}
                _ => bad.push(format!("  case {i}, block {:?} grid {:?}: got {r:?}, want {want:?}", &sh[..3], &sh[3..])),
            }
        }
    }
    assert!(bad.is_empty(), "a launch stopped (or ran) where the others did not:\n{}", bad.join("\n"));
}

/// The refusals, hermetic: each would make the total wrong or launch-dependent.
#[test]
fn what_would_make_the_total_depend_on_the_launch_is_refused() {
    let k = |body: &str| {
        format!(
            "kernel k(X: GlobalMemory<F32>, ACC: GlobalMemory<I64>, OUT: GlobalMemory<F32>, P: GlobalMemory<I32>) {{
    {body}
}}

fn main() {{}}
"
        )
    };
    let decl = format!("@ZeroDrift\n    {NARROW}\n    let mut acc: F32 = 0.0;\n    acc += X[0];");
    let cases: Vec<(&str, String, &str)> = vec![
        ("branch", k(&format!("{decl}\n    if P[0] > 0 {{ drift_add_to(ACC, OUT, acc); }}")), "must be a statement of the kernel's body itself"),
        // A branch every thread of the block takes alike is refused too: a
        // block that skipped it would take no ticket.
        ("uniform_branch", k(&format!("{decl}\n    if block_idx_x() > 0 {{ drift_add_to(ACC, OUT, acc); }}")), "must be a statement of the kernel's body itself"),
        ("loop", k(&format!("{decl}\n    @invariant(j >= 0)\n    for j in 0..2 step 1 {{ drift_add_to(ACC, OUT, acc); }}")), "must be a statement of the kernel's body itself"),
        // A second call nested in a branch, beside the one that is planned.
        ("nested_and_top", k(&format!("{decl}\n    if P[0] > 0 {{ drift_add_to(ACC, OUT, acc); }}\n    drift_add_to(ACC, OUT, acc);")), "must be a statement of the kernel's body itself"),
        ("top_and_nested", k(&format!("{decl}\n    drift_add_to(ACC, OUT, acc);\n    if P[0] > 0 {{ drift_add_to(ACC, OUT, acc); }}")), "must be a statement of the kernel's body itself"),
        ("twice", k(&format!("{decl}\n    drift_add_to(ACC, OUT, acc);\n    drift_add_to(ACC, OUT, acc);")), "is called again on line"),
        ("nonzero_start", k(&format!("@ZeroDrift\n    {NARROW}\n    let mut acc: F32 = 1.0;\n    drift_add_to(ACC, OUT, acc);")), "does not start at zero"),
        ("read", k(&format!("{decl}\n    OUT[1] = acc;\n    drift_add_to(ACC, OUT, acc);")), "is one thread's share of the total"),
        ("after", k(&format!("{decl}\n    drift_add_to(ACC, OUT, acc);\n    acc += X[1];")), "is added to after `drift_add_to`"),
        ("after_running_sum", k(&format!("{decl}\n    drift_add_to(ACC, OUT, acc);\n    acc = acc + X[1];")), "is added to after `drift_add_to`"),
        ("return_before", k(&format!("{decl}\n    if block_idx_x() > 3 {{ return; }}\n    drift_add_to(ACC, OUT, acc);")), "can take threads away first"),
        ("not_drift", k("let mut acc: F32 = 0.0;\n    acc += X[0];\n    drift_add_to(ACC, OUT, acc);"), "is not a `@ZeroDrift` accumulator"),
        // The type checker refuses this one first.
        ("declared_after", k(&format!("drift_add_to(ACC, OUT, acc);\n    {decl}")), "Undefined variable `acc`"),
        ("parameter", k(&format!("{decl}\n    drift_add_to(ACC, OUT, P);")), "is not a `@ZeroDrift` accumulator"),
        ("expression", k(&format!("{decl}\n    drift_add_to(ACC, OUT, X[0]);")), "takes a `@ZeroDrift` accumulator as its third argument"),
        ("acc_type", k(&format!("{decl}\n    drift_add_to(X, OUT, acc);")), "give a GlobalMemory<I64>"),
        ("out_type", k(&format!("{decl}\n    drift_add_to(ACC, ACC, acc);")), "give a GlobalMemory<F32>"),
        ("arity", k(&format!("{decl}\n    drift_add_to(ACC, acc);")), "drift_add_to"),
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
    let ok = sum_kernel("F32", NARROW);
    for (tag, args, want) in [("llvm", &[][..], "has no host lowering"), ("cpu", &["--emit-cpu"][..], "has no CPU lowering")] {
        match try_compile(&format!("host_{tag}"), &ok, args) {
            Ok(_) => bad.push(format!("  {tag}: compiled")),
            Err(text) if !text.contains(want) || !text.contains("drift_add_to") => {
                bad.push(format!("  {tag}: refused without `{want}`:\n{text}"))
            }
            Err(_) => {}
        }
    }
    // The JIT runs `main` on the CPU, and takes no `-o`.
    let dir = scratch("host_jit");
    let path = dir.join("host_jit.ysu");
    std::fs::write(&path, &ok).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_Y")).arg(&path).arg("--jit").current_dir(&dir).output().expect("run Y");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    if out.status.success() || !text.contains("`drift_add_to(...)` has no host lowering") {
        bad.push(format!("  jit: not refused by name:\n{text}"));
    }
    assert!(bad.is_empty(), "{} of {} forms were not refused as they should be:\n{}", bad.len(), cases.len() + 3, bad.join("\n"));
}

/// The emitted shape, hermetic.
#[test]
fn a_partial_is_added_in_128_bits_with_only_the_term_test() {
    let ptx = std::fs::read_to_string(compile("shape", &sum_kernel("F32", NARROW))).unwrap();
    let start = ptx.find("one thread's partial").expect("the partial's declaration");
    let end = ptx.find("every thread's partial joins").expect("the call");
    let body = &ptx[start..end];
    assert!(body.contains("add.cc.u64") && body.contains("addc.u64"), "the partial is not 128 bits:\n{body}");
    // The term test, s > 63 - 24, and nothing else stops the loop: no
    // overflow test (`xor.b64`) and no range test (`setp.gt.u64`).
    assert_eq!(body.matches("trap;").count(), 1, "the loop stops on more than its term:\n{body}");
    assert!(body.contains(", 39;"), "the term traps at 2^63:\n{body}");
    assert!(!body.contains("xor.b64") && !body.contains("setp.gt.u64"), "a partial is tested:\n{body}");
    let call = &ptx[end..];
    assert_eq!(call.matches("atom.global.add.u64").count(), 2, "one atomic for the total, one for the ticket:\n{call}");
    assert_eq!(call.matches("shfl.sync.bfly.b32").count(), 20, "five steps of two 64-bit words:\n{call}");
    assert!(call.contains("bar.sync 0;") && call.matches("membar.gl;").count() == 2, "the fences:\n{call}");
    // A warp the block does not fill skips the shuffles, decided by comparing
    // the warp's end with the block's size, both registers. The GPU cannot
    // show this (`python/y_lang/tests/test_drift_grid_sum.py` can).
    let lines: Vec<&str> = call.lines().map(str::trim).collect();
    let at = lines.iter().position(|l| l.contains("bra $DRIFT_OWN")).expect("the branch around the shuffles");
    let guard: Vec<&str> = lines[at - 1].trim_end_matches(';').split([' ', ',']).filter(|w| !w.is_empty()).collect();
    assert!(
        guard.len() == 4 && guard[0] == "setp.gt.u32" && guard[2].starts_with("%r") && guard[3].starts_with("%r"),
        "the partial-warp test is not `setp.gt.u32 p, end, threads`: {}",
        lines[at - 1]
    );
    assert!(lines[..at].iter().all(|l| !l.starts_with("shfl")), "a shuffle comes before the partial-warp test:\n{call}");
    let Some(tool) = which("ptxas") else {
        eprintln!("SKIP: no ptxas, so the shape was not assembled");
        return;
    };
    let p = compile("asm", &sum_kernel("I64", ""));
    for path in [p, compile("asm_f32", &sum_kernel("F32", WIDE))] {
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
