//! `rmsnorm_v4` normalises the row it is given, by the epsilon it is given.
//!
//! The built-in took `(x, w, out)` and assumed the rest: the mean was
//! `sum * 0.0009765625` (a 1024-wide row), epsilon was `1e-5`, and the second
//! reduction read exactly eight warp partials (a 256-thread block). Measured on
//! the device with this file's inputs: at `eps = 0.5` the output was 2.7% off,
//! a 512-wide row was scaled by `sqrt(2)` and a 128-wide one by `sqrt(8)`, and
//! a block covering half its row left the other half unwritten. The kernel's
//! `hidden_dim` and `eps` parameters were read by nothing.
//!
//! It takes `(x, w, out, hidden_dim, eps)` now, reduces over however many warps
//! the block has, and traps when the launch is not one row per block with four
//! elements per thread. Traps run in child processes (this test, re-run with
//! `Y_RMSNORM_CASE`): a trap leaves the device unusable for the rest of its
//! process.

use std::path::PathBuf;
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

const KERNEL: &str = "kernel rms(X: GlobalMemory<F32>, W: GlobalMemory<F32>, Out: GlobalMemory<F32>, hidden_dim: I32, eps: F32) {\n    \
    let row: I32 = block_idx_x();\n    let col: I32 = thread_idx_x() * 4;\n    let idx: I32 = row * hidden_dim + col;\n    \
    rmsnorm_v4(X[idx], W[col], Out[idx], hidden_dim, eps);\n}\n\nfn main() {}\n";

/// Two rows per block, selected by `thread_idx_y()`: the shape a
/// two-dimensional launch would have to mean.
const KERNEL_TWO_ROWS: &str = "kernel rms(X: GlobalMemory<F32>, W: GlobalMemory<F32>, Out: GlobalMemory<F32>, hidden_dim: I32, eps: F32) {\n    \
    let row: I32 = block_idx_x() * 2 + thread_idx_y();\n    let col: I32 = thread_idx_x() * 4;\n    let idx: I32 = row * hidden_dim + col;\n    \
    rmsnorm_v4(X[idx], W[col], Out[idx], hidden_dim, eps);\n}\n\nfn main() {}\n";

const ROWS: usize = 8;

fn compile(tag: &str, src: &str) -> (bool, String, PathBuf) {
    let dir = pinned::pinned_scratch(&format!("rmsnorm_{tag}"), pinned::SM_PINNED);
    let path = dir.join("k.ysu");
    std::fs::write(&path, src).unwrap();
    let ptx = dir.join("k.ptx");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .args(["--emit-ptx", "-o"])
        .arg(&ptx)
        .current_dir(&dir)
        .output()
        .expect("run Y");
    let log = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    (out.status.success(), log, ptx)
}

/// Inputs that are not all the same magnitude, so a wrong mean shows.
fn inputs(hidden: usize) -> (Vec<f32>, Vec<f32>) {
    let x = (0..ROWS * hidden).map(|i| ((i * 7919 % 1009) as f32 - 504.0) / 97.0).collect();
    let w = (0..hidden).map(|i| 0.5 + (i * 31 % 17) as f32 / 16.0).collect();
    (x, w)
}

/// One launch, reported on stdout: `TRAPPED`, or `RESULT <worst error relative
/// to the largest output>` against a float64 reference.
fn run_case(ptx_path: &str, hidden: usize, block: (u32, u32, u32), eps: f32) {
    let ctx = y::cuda_runtime::CudaContext::new().expect("a context");
    let ptx = std::fs::read_to_string(ptx_path).unwrap();
    let module = ctx.load_ptx(&ptx, "rms").expect("load");
    let (x, w) = inputs(hidden);
    let bytes = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
    let (dx, dw, dout) = (ctx.alloc(x.len() * 4).unwrap(), ctx.alloc(w.len() * 4).unwrap(), ctx.alloc(x.len() * 4).unwrap());
    ctx.memcpy_htod_at(&dx, 0, &bytes(&x)).unwrap();
    ctx.memcpy_htod_at(&dw, 0, &bytes(&w)).unwrap();
    ctx.memset_u8(&dout, 0).unwrap();
    let args = [dx.device_ptr(), dw.device_ptr(), dout.device_ptr(), hidden as u64, eps.to_bits() as u64];
    // One block per `block.1` rows.
    let grid = ((ROWS as u32) / block.1, 1, 1);
    if let Err(e) = ctx.launch(&module, grid, block, 0, &args).and_then(|_| ctx.synchronize()) {
        println!("TRAPPED {e}");
        return;
    }
    let mut raw = vec![0u8; x.len() * 4];
    ctx.memcpy_dtoh_at(&mut raw, &dout, 0).unwrap();
    let out: Vec<f32> = raw.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let (mut worst, mut largest) = (0f64, 0f64);
    for r in 0..ROWS {
        let row = &x[r * hidden..(r + 1) * hidden];
        let ms = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / hidden as f64;
        let inv = 1.0 / (ms + eps as f64).sqrt();
        for c in 0..hidden {
            let want = row[c] as f64 * inv * w[c] as f64;
            worst = worst.max((out[r * hidden + c] as f64 - want).abs());
            largest = largest.max(want.abs());
        }
    }
    println!("RESULT {:e}", worst / largest);
}

fn child(ptx: &PathBuf, hidden: usize, block: (u32, u32, u32), eps: f32) -> String {
    let out = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "every_row_shape_is_normalised_and_a_wrong_launch_traps", "--nocapture"])
        .env("Y_RMSNORM_CASE", format!("{}|{hidden}|{}|{}|{}|{eps}", ptx.display(), block.0, block.1, block.2))
        .output()
        .expect("run the launch in a child");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn every_row_shape_is_normalised_and_a_wrong_launch_traps() {
    if let Ok(case) = std::env::var("Y_RMSNORM_CASE") {
        let f: Vec<&str> = case.split('|').collect();
        let n = |i: usize| f[i].parse::<u32>().unwrap();
        run_case(f[0], f[1].parse().unwrap(), (n(2), n(3), n(4)), f[5].parse().unwrap());
        return;
    }
    if y::cuda_runtime::CudaContext::new().is_none() {
        eprintln!("SKIP: no CUDA device, so no kernel was run");
        return;
    }
    let (ok, log, ptx) = compile("device", KERNEL);
    assert!(ok, "{log}");
    let (ok, log, ptx_two_rows) = compile("two_rows", KERNEL_TWO_ROWS);
    assert!(ok, "{log}");

    // (hidden, block, eps). The first is the one shape the old lowering got
    // right; each other one breaks one of its three assumptions.
    for (hidden, block, eps) in [
        (1024, (256, 1, 1), 1e-5f32),
        (1024, (256, 1, 1), 0.5),
        (512, (128, 1, 1), 1e-5),
        (4096, (1024, 1, 1), 1e-5),
        (128, (32, 1, 1), 1e-6),
    ] {
        let text = child(&ptx, hidden, block, eps);
        let err: f64 = text
            .split("RESULT ")
            .nth(1)
            .and_then(|r| r.split_whitespace().next())
            .unwrap_or_else(|| panic!("hidden {hidden}, block {block:?}, eps {eps}: no result:\n{text}"))
            .parse()
            .unwrap();
        // rsqrt.approx is within 2^-22.9 relative; the f32 sum of squares over
        // a 4096-wide row adds a few ulps more.
        assert!(err < 1e-5, "hidden {hidden}, block {block:?}, eps {eps}: error {err:e} of the largest output");
    }

    // A launch that is not one row per block, four elements per thread, in
    // whole warps, would reduce something other than the row.
    for (kernel, hidden, block, why) in [
        (&ptx, 1024, (128, 1, 1), "the block covers half the row"),
        (&ptx, 192, (48, 1, 1), "48 threads is not a whole number of warps"),
        (&ptx_two_rows, 128, (32, 2, 1), "a two-dimensional block holds two rows"),
    ] {
        let text = child(kernel, hidden, block, 1e-5);
        assert!(text.contains("TRAPPED"), "{why} (hidden {hidden}, block {block:?}), and the kernel ran: {text}");
    }
}

/// The old three-operand form is refused rather than given a row length.
#[test]
fn the_row_length_and_epsilon_cannot_be_left_out() {
    let old = KERNEL.replace("rmsnorm_v4(X[idx], W[col], Out[idx], hidden_dim, eps)", "rmsnorm_v4(X[idx], W[col], Out[idx])");
    assert_ne!(old, KERNEL);
    let (ok, log, _) = compile("three", &old);
    assert!(!ok && log.contains("rmsnorm_v4"), "the three-operand form compiled:\n{log}");
    let (ok, log, _) = compile("five", KERNEL);
    assert!(ok, "{log}");
}
