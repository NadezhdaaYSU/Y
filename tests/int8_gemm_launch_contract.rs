//! The int8 tensor-core GEMM's launch contract, and the tie to
//! `proofs/Int8GemmSchedule.v`.
//!
//! ## The contract, since 2026-11-06
//!
//! The kernel stages its K tiles in shared memory
//! (`src/ptx_emitter/int8_staged_gemm.rs`); its header names the block and
//! the shared memory it needs.
//!
//! - **Block**: exactly the threads the header names. Every thread has a
//!   copy role and a warp tile, so any other block TRAPS: a smaller one would
//!   leave rows of a stage uncopied, a larger one would put a warp outside
//!   the tile.
//! - **Grid**: any. Output tiles are strided by a linear CTA id in grouped
//!   order, and K tiles are striped over `%ctaid.z`.
//! - **C**: a launch with `gridDim.z = 1` STORES every output; one that splits
//!   K adds its partials into C with `red.global.add.s32`, so C must then be
//!   zeroed. The scaled form ignores z and stores.
//!
//! ## The defect this file was written for, in the kernel it replaced
//!
//! That kernel gave one 16x8 output tile to one WARP, read A and B straight
//! from global memory, and launched a grid of `(N/8, M/16, splits)`. A CTA
//! had exactly 32 threads of work however many it was launched with, and
//! nothing checked. At M=64 N=32 K=128 with every element of A = 3 and of
//! B = 5, so every element of C must be exactly K*15 = 1920:
//!
//! | block | result |
//! |---|---|
//! | (32,1,1) | correct |
//! | (64,1,1) | 1344 of 2048 wrong, `C[256] = 3840` |
//! | (128,1,1) | same, and it reads row 79 of a 64-row A |
//!
//! 3840 is EXACTLY DOUBLE: warp 1's `g = tid/4` fell in 8..15, its second
//! A-row read landed in the next tile's rows, and `red.global.add.s32` summed
//! that product in. It was fixed with a warp-uniform guard (`tid < 32`). The
//! staged kernel cannot idle a warp - its copies need every thread - so it
//! traps instead, and the proof's theorems are the trap's.
//!
//! ## What each test covers
//!
//! The device tests SKIP without a driver, so the source-level tests carry
//! the trap's presence, the store/reduce branch and the tie to the proof on a
//! machine with no GPU. A trap poisons the CUDA context of its process, so
//! the launches expected to trap run in child processes.

use std::path::Path;
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;
#[path = "common/int8_launch.rs"]
mod int8_launch;

use int8_launch::int8_launch;

fn y_bin() -> std::path::PathBuf {
    let mut bin = std::env::current_exe().unwrap();
    bin.pop();
    if bin.ends_with("deps") {
        bin.pop();
    }
    bin.join("Y")
}

/// Write `source` as `file` into a per-test temp directory and return
/// `(dir, source path)`. The tag is in the SIGNATURE rather than a comment
/// asking the next author to remember: this helper materialises files in a
/// temp dir and that race has fired six times in this repository.
fn prepare(prefix: &str, tag: &str, file: &str, source: String) -> (std::path::PathBuf, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("{prefix}_{}_{}", std::process::id(), tag));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join(file);
    std::fs::write(&src, source).unwrap();
    (dir, src)
}

/// Compile against a PINNED profile, with `env` set for the compiler (the
/// tile override, for the tests that force one).
fn compile_pinned(dir: &Path, src: &Path, env: &[(&str, &str)]) -> std::process::Output {
    pinned::pin(dir, pinned::SM_PINNED);
    let mut cmd = Command::new(y_bin());
    cmd.arg(src).arg("--emit-ptx").current_dir(dir).env_remove("Y_INT8_TILE");
    for (k, v) in env {
        cmd.env(k, v);
    }
    cmd.output().expect("run Y")
}

fn read_back(dir: &Path, out: std::process::Output, ptx: &str) -> Result<String, String> {
    let r = if out.status.success() {
        Ok(std::fs::read_to_string(dir.join(ptx)).expect("no .ptx"))
    } else {
        Err(format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ))
    };
    let _ = std::fs::remove_dir_all(dir);
    r
}

fn plain_source(m: usize, n: usize, k: usize) -> String {
    format!(
        "@tile({}, {}, {})\n\
         kernel int8_gemm(A: GlobalMemory<I8>, B: GlobalMemory<I8>, C: GlobalMemory<I32>) {{\n}}\n\
         fn main() {{}}\n",
        m, n, k
    )
}

/// Compile a fixture at `(m, n, k)` and return its PTX. `--emit-ptx` writes
/// next to its input, so it compiles a COPY: a gate that emits must never
/// rewrite the committed artifacts it is checking.
fn emit(tag: &str, m: usize, n: usize, k: usize) -> Result<String, String> {
    emit_env(tag, m, n, k, &[])
}

fn emit_env(tag: &str, m: usize, n: usize, k: usize, env: &[(&str, &str)]) -> Result<String, String> {
    let (dir, src) = prepare("i8lc", tag, "lc.ysu", plain_source(m, n, k));
    let out = compile_pinned(&dir, &src, env);
    read_back(&dir, out, "lc.ptx")
}

/// The header's tile, `<cta_m>x<cta_n>`.
fn header_tile(ptx: &str) -> (usize, usize) {
    let hdr = ptx
        .lines()
        .find(|l| l.contains("[Y INT8 TENSOR CORE GEMM] staged"))
        .expect("the kernel must declare its tile in the header");
    let tile = hdr.split("| CTA ").nth(1).unwrap().split_whitespace().next().unwrap();
    let mut it = tile.split('x').map(|v| v.parse::<usize>().unwrap());
    (it.next().unwrap(), it.next().unwrap())
}

fn cpu_reference(m: usize, n: usize, k: usize, a: &[i8], b: &[i8]) -> Vec<i32> {
    let mut c = vec![0i32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0i32;
            for x in 0..k {
                s += a[i * k + x] as i32 * b[j * k + x] as i32;
            }
            c[i * n + j] = s;
        }
    }
    c
}

/// Run `ptx`'s `int8_gemm` on `(a, b)` with each grid in `grids`, C filled
/// with `fill` before each launch, and return how many outputs differ from
/// `want` per grid.
fn run_grids(
    ctx: &y::cuda_runtime::CudaContext,
    ptx: &str,
    dims: (usize, usize, usize),
    a: &[i8],
    b: &[i8],
    grids: &[((u32, u32, u32), u8)],
) -> Vec<(usize, usize, i32)> {
    let (m, n, k) = dims;
    let geo = int8_launch(ptx);
    let module = ctx.load_ptx(ptx, "int8_gemm").expect("PTX failed to load");
    module.set_max_dynamic_smem(geo.smem).unwrap();
    let want = cpu_reference(m, n, k, a, b);
    let d_a = ctx.alloc(m * k).unwrap();
    let d_b = ctx.alloc(n * k).unwrap();
    let d_c = ctx.alloc(m * n * 4).unwrap();
    ctx.memcpy_htod_at(&d_a, 0, &a.iter().map(|&v| v as u8).collect::<Vec<u8>>()).unwrap();
    ctx.memcpy_htod_at(&d_b, 0, &b.iter().map(|&v| v as u8).collect::<Vec<u8>>()).unwrap();
    let args = vec![d_a.device_ptr(), d_b.device_ptr(), d_c.device_ptr()];
    grids
        .iter()
        .map(|&(grid, fill)| {
            ctx.memset_u8(&d_c, fill).unwrap();
            ctx.launch(&module, grid, (geo.block, 1, 1), geo.smem, &args)
                .unwrap_or_else(|e| panic!("launch failed at grid {:?}: {:?}", grid, e));
            ctx.synchronize().unwrap_or_else(|e| panic!("faulted at grid {:?}: {:?}", grid, e));
            let mut raw = vec![0u8; m * n * 4];
            ctx.memcpy_dtoh_at(&mut raw, &d_c, 0).unwrap();
            let mut bad = 0usize;
            let mut first = (0usize, 0i32);
            for i in 0..m * n {
                let got = i32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]]);
                if got != want[i] {
                    if bad == 0 {
                        first = (i, got);
                    }
                    bad += 1;
                }
            }
            (bad, first.0, first.1)
        })
        .collect()
}

fn operands(m: usize, n: usize, k: usize, sa: usize, sb: usize) -> (Vec<i8>, Vec<i8>) {
    let a = (0..m * k).map(|i| (((i * sa + 11) % 251) as i32 - 125) as i8).collect();
    let b = (0..n * k).map(|i| (((i * sb + 7) % 251) as i32 - 125) as i8).collect();
    (a, b)
}

// ---------------------------------------------------------------------------
// The block: the declared one, and nothing else.
// ---------------------------------------------------------------------------

/// The child: one launch of the 64x32x128 fixture with the block size in
/// `Y_INT8_CHILD_BLOCK` (or the grid in `Y_INT8_CHILD_GRID`), printing
/// `CHILD_RESULT ok`, `wrong` or `stopped`. Does nothing unless spawned by
/// a parent below.
#[test]
fn child_int8_launch_with_a_given_geometry() {
    let block = std::env::var("Y_INT8_CHILD_BLOCK").ok();
    let grid = std::env::var("Y_INT8_CHILD_GRID").ok();
    if block.is_none() && grid.is_none() {
        return;
    }
    let ctx = y::cuda_runtime::CudaContext::new().expect("the parent checked for a driver");
    let (m, n, k) = (64usize, 32usize, 128usize);
    let ptx = emit("child", m, n, k).expect("the fixture must compile");
    let geo = int8_launch(&ptx);
    let bx: u32 = block.as_deref().map(|b| b.parse().unwrap()).unwrap_or(geo.block);
    let g: Vec<u32> = grid.as_deref().unwrap_or("1,1,1").split(',').map(|v| v.parse().unwrap()).collect();
    let module = ctx.load_ptx(&ptx, "int8_gemm").unwrap();
    module.set_max_dynamic_smem(geo.smem).unwrap();
    let (d_a, d_b, d_c) = (ctx.alloc(m * k).unwrap(), ctx.alloc(n * k).unwrap(), ctx.alloc(m * n * 4).unwrap());
    ctx.memset_u8(&d_a, 3).unwrap();
    ctx.memset_u8(&d_b, 5).unwrap();
    ctx.memset_u8(&d_c, 0).unwrap();
    let r = match ctx.launch(&module, (g[0], g[1], g[2]), (bx, 1, 1), geo.smem, &[d_a.device_ptr(), d_b.device_ptr(), d_c.device_ptr()]) {
        Err(e) => format!("launch refused {e}"),
        Ok(()) => match ctx.synchronize() {
            Err(e) => format!("stopped {e}"),
            Ok(()) => {
                let mut raw = vec![0u8; m * n * 4];
                ctx.memcpy_dtoh_at(&mut raw, &d_c, 0).unwrap();
                let wrong = (0..m * n)
                    .filter(|&i| i32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]]) != (k * 15) as i32)
                    .count();
                if wrong == 0 { "ok".to_string() } else { format!("wrong {wrong}") }
            }
        },
    };
    println!("CHILD_RESULT {}", r);
}

/// Spawn the child with `var=value` and return what it printed.
fn run_child(var: &str, value: &str) -> String {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["child_int8_launch_with_a_given_geometry", "--exact", "--nocapture", "--test-threads=1"])
        .env(var, value)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let start = std::time::Instant::now();
    while child.try_wait().unwrap().is_none() {
        if start.elapsed() > std::time::Duration::from_secs(120) {
            let _ = child.kill();
            panic!("{var}={value}: the child ran for 2 minutes, so nothing stopped it");
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let out = child.wait_with_output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    text.lines()
        .find_map(|l| l.find("CHILD_RESULT ").map(|at| l[at + "CHILD_RESULT ".len()..].to_string()))
        .unwrap_or_else(|| panic!("{var}={value}: the child printed no result:\n{text}{}", String::from_utf8_lossy(&out.stderr)))
}

/// The behavioural half: the declared block computes the right matrix, and
/// every other block stops the kernel. The old kernel's measurement (64 and
/// 128 threads double-counting) is the reason the wrong sizes are 64 and 256
/// as well as 32: one smaller, one larger, one the old contract's.
#[test]
fn the_declared_block_is_correct_and_any_other_traps() {
    if y::cuda_runtime::CudaContext::new().is_none() {
        eprintln!("SKIP: no CUDA driver. The source-level tests in this file still cover the trap's presence.");
        return;
    }
    let geo = int8_launch(&emit("blk", 64, 32, 128).expect("the fixture must compile"));
    assert_eq!(run_child("Y_INT8_CHILD_BLOCK", &geo.block.to_string()), "ok", "the declared block {} must compute the matrix", geo.block);
    for bx in [32u32, 64, 256] {
        assert_ne!(bx, geo.block);
        let r = run_child("Y_INT8_CHILD_BLOCK", &bx.to_string());
        assert!(r.starts_with("stopped"), "block ({bx},1,1) must trap, got `{r}`");
    }
}

/// The structural half, which runs with no GPU: the trap must be IN the
/// emitted kernel, test `%ntid.x` against the declared block (and y, z
/// against 1), and come before the first copy and the first mma.
#[test]
fn the_emitted_kernel_traps_on_any_other_block() {
    let ptx = emit("trap", 64, 32, 128).expect("the fixture must compile");
    let geo = int8_launch(&ptx);
    let body: Vec<&str> = ptx.lines().map(|l| l.trim()).filter(|l| !l.starts_with("//")).collect();
    let first_work = body
        .iter()
        .position(|l| l.starts_with("cp.async") || l.starts_with("mma.sync"))
        .expect("no copy or mma in the kernel");
    for (sreg, want) in [("%ntid.x", geo.block), ("%ntid.y", 1), ("%ntid.z", 1)] {
        let mov = body
            .iter()
            .position(|l| l.starts_with("mov.u32") && l.ends_with(&format!("{sreg};")))
            .unwrap_or_else(|| panic!("the kernel never reads {sreg}:\n{ptx}"));
        let reg = body[mov].split_whitespace().nth(1).unwrap().trim_end_matches(',');
        let setp = &body[mov + 1];
        assert!(
            setp.starts_with("setp.ne.u32") && setp.contains(&format!(" {reg}, {want};")),
            "{sreg} is read into {reg} but not compared against {want}: `{setp}`"
        );
        let pred = setp.split_whitespace().nth(1).unwrap().trim_end_matches(',');
        assert_eq!(body[mov + 2], format!("@{pred} trap;"), "the comparison of {sreg} does not trap");
        assert!(mov < first_work, "the {sreg} trap comes after the kernel has started copying or multiplying");
    }
}

// ---------------------------------------------------------------------------
// The grid: any.
// ---------------------------------------------------------------------------

/// The answer must not depend on the GRID - including the `(N/8, M/16, z)`
/// grid a host written for the old kernel launches - and a launch that does
/// not split K must OVERWRITE C (here C starts as 0xA5 garbage), while one
/// that does adds into a zeroed C.
#[test]
fn the_answer_does_not_depend_on_the_grid() {
    use y::cuda_runtime::CudaContext;
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver — grid invariance was not demonstrated.");
        return;
    };
    // 512x512 takes the 64x64 tile (64 tiles: 8 x 8), K = 1024 is 16 K tiles.
    let (m, n, k) = (512usize, 512usize, 1024usize);
    let ptx = emit("grid", m, n, k).expect("fixture must compile");
    assert_eq!(header_tile(&ptx), (64, 64));
    let tiles = int8_launch(&ptx).tiles;
    let (a, b) = operands(m, n, k, 29, 71);
    let grids = [
        ((tiles, 1, 1), 0xA5u8),                         // exactly the tiles
        (((n / 8) as u32, (m / 16) as u32, 1), 0xA5),    // the old kernel's host grid
        ((1, 1, 1), 0xA5),                               // one CTA does everything
        ((1, 2, 1), 0xA5),
        ((3, 5, 1), 0xA5),                               // coprime with the tile count
        ((tiles + 7, 1, 1), 0xA5),                       // more CTAs than tiles
        ((2, 2, 4), 0),                                  // split K as well
        (((n / 8) as u32, (m / 16) as u32, 3), 0),
        ((tiles, 1, 16), 0),                             // one K tile a CTA
        ((tiles, 1, 40), 0),                             // CTAs with an empty stripe
    ];
    for (r, (grid, fill)) in run_grids(&ctx, &ptx, (m, n, k), &a, &b, &grids).into_iter().zip(grids) {
        assert_eq!(
            r.0, 0,
            "grid {:?} with C filled with {:#x}: {} of {} wrong; C[{}] = {}",
            grid, fill, r.0, m * n, r.1, r.2
        );
    }
}

/// The CTA counts are 32-bit; a grid whose x*y plane has 2^32 CTAs or more
/// would wrap them, so the kernel traps. The grid is legal (CUDA accepts it),
/// so it is the trap that stops it: a kernel without the check would grind
/// through 4.3e9 CTAs, which the deadline turns into a failure.
#[test]
fn a_grid_of_2_to_the_32_ctas_traps() {
    if y::cuda_runtime::CudaContext::new().is_none() {
        eprintln!("SKIP: no CUDA driver - the grid check was not exercised.");
        return;
    }
    assert_eq!(run_child("Y_INT8_CHILD_GRID", "5,1,1"), "ok", "the control grid must run");
    // 65538 x 65535 = 4,295,032,830 CTAs, one more x column than 2^32 - 1 allows.
    let r = run_child("Y_INT8_CHILD_GRID", "65538,65535,1");
    assert!(r.starts_with("stopped"), "a grid of 2^32+ CTAs was not stopped by the kernel: {r}");
}

// ---------------------------------------------------------------------------
// The tile: chosen from the shape, and every one computes the same matrix.
// ---------------------------------------------------------------------------

/// The tile is the largest of 128x128 and 64x128 that fits inside the matrix
/// and leaves at least 64 output tiles, else 64x64 - and every M % 16, N % 8
/// still compiles. Runs with no GPU.
#[test]
fn the_tile_is_chosen_from_the_shape_and_narrows_nothing() {
    for (m, n) in [(16usize, 8usize), (48, 24), (80, 40), (16, 64), (64, 8), (128, 128), (512, 512),
                   (1024, 1024), (1008, 1008), (4096, 16), (4096, 64), (4096, 128), (32, 4096), (32, 11008),
                   (64, 11008), (112, 11008), (128, 4096), (128, 11008), (2048, 1024)] {
        let ptx = emit(&format!("sel{}_{}", m, n), m, n, 128)
            .unwrap_or_else(|e| panic!("M={} N={} must compile:\n{}", m, n, e));
        let tiles = |tm: usize, tn: usize| m.div_ceil(tm) * n.div_ceil(tn);
        // A tile taller than M or wider than N computes padding: 32 x 11008
        // took 128x128 and measured 0.60x cuBLASLt (the benchmark record).
        let fits = |tm: usize, tn: usize| tm <= m && tn <= n;
        let want = if fits(128, 128) && tiles(128, 128) >= 64 {
            (128, 128)
        } else if fits(64, 128) && tiles(64, 128) >= 64 {
            (64, 128)
        } else {
            (64, 64)
        };
        assert_eq!(header_tile(&ptx), want, "M={} N={}: {} tiles at 128x128, {} at 64x128", m, n, tiles(128, 128), tiles(64, 128));
        assert_eq!(int8_launch(&ptx).tiles as usize, tiles(want.0, want.1), "M={} N={}: the header's tile count", m, n);
    }
}

/// Shapes the rule sends to each tile, and forced tiles of other geometries
/// (the K step 32 and 128, 2 stages, 1x4 and 4x1 warps), all against a CPU
/// reference on a ragged shape: M and N not multiples of any tile, and
/// K % 64 == 32, so the last K tile is half zeros.
///
/// The reference is a plain CPU triple loop, not another GPU kernel: the tile
/// decides which lane reads which element, so checking one tiling against
/// another would compare two readings of the same possibly-wrong layout.
#[test]
fn every_tile_computes_the_same_matrix() {
    use y::cuda_runtime::CudaContext;
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver — the tiles were not exercised on a device.");
        return;
    };
    let mut seen = std::collections::BTreeSet::new();
    // (M, N, K, forced tile): None takes the rule's.
    let cases: [(usize, usize, usize, Option<&str>); 11] = [
        (1040, 1048, 352, None),                       // 128x128, ragged in M and N
        (208, 2056, 352, None),                        // 64x128
        (208, 200, 352, None),                         // 64x64
        (16, 8, 32, None),                             // the smallest legal shape
        (208, 200, 352, Some("128,128,64,2,2,3,2,32")),
        (208, 200, 352, Some("64,128,64,2,2,3,3,16")),
        (208, 200, 352, Some("128,64,64,2,2,3,3,8")),
        (208, 200, 352, Some("64,64,32,2,2,4,4,16")),  // one 32-byte K step a stage
        (208, 200, 352, Some("64,64,128,2,2,2,2,16")), // four K steps, two stages
        (208, 200, 352, Some("64,256,64,1,4,3,2,8")),  // 1x4 warps
        (208, 200, 352, Some("128,64,64,4,1,3,2,8")),  // 4x1 warps
    ];
    for (idx, (m, n, k, forced)) in cases.into_iter().enumerate() {
        let env: Vec<(&str, &str)> = forced.map(|t| vec![("Y_INT8_TILE", t)]).unwrap_or_default();
        let ptx = emit_env(&format!("tile{idx}"), m, n, k, &env).expect("fixture must compile");
        let tile = header_tile(&ptx);
        seen.insert(tile);
        let (a, b) = operands(m, n, k, 37, 53);
        let tiles = int8_launch(&ptx).tiles;
        let r = run_grids(&ctx, &ptx, (m, n, k), &a, &b, &[((tiles, 1, 1), 0xA5), ((tiles, 1, 3), 0)]);
        for (rr, z) in r.into_iter().zip([1, 3]) {
            assert_eq!(rr.0, 0, "M={m} N={n} K={k} tile {tile:?} ({forced:?}), z = {z}: {} wrong; C[{}] = {}", rr.0, rr.1, rr.2);
        }
    }
    // Non-vacuity: the cases must exercise distinct CTA tiles.
    assert!(seen.len() >= 5, "only {:?} distinct tiles were exercised", seen);
}

/// A malformed tile override is an error, not ignored.
#[test]
fn a_malformed_tile_override_is_refused() {
    let e = emit_env("badtile", 64, 64, 64, &[("Y_INT8_TILE", "128,128")]).expect_err("must be refused");
    assert!(e.contains("Y_INT8_TILE"), "{}", e);
    let e = emit_env("badgeo", 64, 64, 64, &[("Y_INT8_TILE", "64,64,48,2,2,3,2,8")]).expect_err("must be refused");
    assert!(e.contains("cannot be staged"), "{}", e);
}

// ---------------------------------------------------------------------------
// The epilogues: which may store.
// ---------------------------------------------------------------------------

fn scaled_source(m: usize, n: usize, k: usize) -> String {
    format!(
        "@tile({}, {}, {})\n\
         kernel int8_gemm_scaled(A: GlobalMemory<I8>, B: GlobalMemory<I8>, \
         Sa: GlobalMemory<F32>, Sb: GlobalMemory<F32>, Bias: GlobalMemory<F32>, \
         C: GlobalMemory<F32>) {{\n}}\n\
         fn main() {{}}\n",
        m, n, k
    )
}

/// The fused shape: `C = acc * Sa[m] * Sb[n] + Bias[n]`, f32 out.
fn emit_scaled(tag: &str, m: usize, n: usize, k: usize) -> Result<String, String> {
    let (dir, src) = prepare("i8sc", tag, "sc.ysu", scaled_source(m, n, k));
    let out = compile_pinned(&dir, &src, &[]);
    read_back(&dir, out, "sc.ptx")
}

/// The fused epilogue STORES, so it must walk the whole contraction at every
/// `gridDim.z`. Before it did, z = 2 gave 2048 of 2048 elements wrong, with a
/// different answer between launches. Three launches per geometry, because
/// that failure was a race.
#[test]
fn the_fused_epilogue_answers_the_same_at_every_split_k_geometry() {
    use y::cuda_runtime::CudaContext;

    const M: usize = 64;
    const N: usize = 32;
    const K: usize = 1024;

    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver — the fused epilogue's launch contract was not demonstrated.");
        return;
    };
    let ptx = emit_scaled("splitk", M, N, K).expect("scaled fixture did not compile");
    let geo = int8_launch(&ptx);
    let module = ctx.load_ptx(&ptx, "int8_gemm_scaled").expect("PTX failed to load");
    module.set_max_dynamic_smem(geo.smem).unwrap();
    let (a, b) = operands(M, N, K, 31, 67);

    let d_a = ctx.alloc(M * K).unwrap();
    let d_b = ctx.alloc(N * K).unwrap();
    let d_sa = ctx.alloc(M * 4).unwrap();
    let d_sb = ctx.alloc(N * 4).unwrap();
    let d_bias = ctx.alloc(N * 4).unwrap();
    let d_c = ctx.alloc(M * N * 4).unwrap();
    ctx.memcpy_htod_at(&d_a, 0, &a.iter().map(|&v| v as u8).collect::<Vec<u8>>()).unwrap();
    ctx.memcpy_htod_at(&d_b, 0, &b.iter().map(|&v| v as u8).collect::<Vec<u8>>()).unwrap();
    // Unit scales and zero bias, so the f32 output IS the integer
    // accumulation: |sum| <= 125*125*1024 = 16_000_000 < 2^24, exactly
    // representable in f32, so a mismatch is the schedule, never rounding.
    ctx.memcpy_htod_at(&d_sa, 0, &(0..M).flat_map(|_| 1.0f32.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    ctx.memcpy_htod_at(&d_sb, 0, &(0..N).flat_map(|_| 1.0f32.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    ctx.memset_u8(&d_bias, 0).unwrap();

    let run = |grid: (u32, u32, u32)| -> Vec<f32> {
        ctx.memset_u8(&d_c, 0xA5).unwrap();
        let args = vec![d_a.device_ptr(), d_b.device_ptr(), d_sa.device_ptr(), d_sb.device_ptr(), d_bias.device_ptr(), d_c.device_ptr()];
        ctx.launch(&module, grid, (geo.block, 1, 1), geo.smem, &args).expect("launch failed");
        ctx.synchronize().expect("kernel did not complete");
        let mut raw = vec![0u8; M * N * 4];
        ctx.memcpy_dtoh_at(&mut raw, &d_c, 0).unwrap();
        (0..M * N).map(|i| f32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]])).collect()
    };

    let want: Vec<f32> = cpu_reference(M, N, K, &a, &b).into_iter().map(|v| v as f32).collect();
    assert!(want.iter().any(|&v| v != 0.0), "the reference is all zeros; nothing below would mean anything");

    for splits in [1u32, 2, 3, 4, 8, 16] {
        for rep in 0..3 {
            let got = run((geo.tiles, 1, splits));
            let diffs = got.iter().zip(&want).filter(|(g, w)| g != w).count();
            assert_eq!(
                diffs, 0,
                "split-K = {splits} (repeat {rep}): {diffs} of {} elements wrong — got C[0] = {}, want {}. \
                 The fused epilogue STORES, so a striped split would make every z-CTA write its own partial.",
                M * N, got[0], want[0]
            );
        }
    }
}

/// The source-level half, which runs with no GPU. A BICONDITIONAL: the
/// plain kernel must still split K (it reads `%ctaid.z` and reduces) and the
/// scaled one must not (it never reads it, and only stores).
#[test]
fn only_the_reducing_epilogue_reads_the_split_k_index() {
    // Comments are stripped first: the storing kernel's header says why it
    // ignores %ctaid.z, and a raw substring search finds that.
    let code = |ptx: &str| -> String {
        ptx.lines()
            .map(|l| l.split("//").next().unwrap_or("").trim_end().to_string())
            .collect::<Vec<_>>()
            .join("\n")
    };
    let plain = code(&emit("reduce_epi", 64, 32, 128).expect("plain fixture did not compile"));
    let storing = code(&emit_scaled("store_epi", 64, 32, 128).expect("scaled fixture did not compile"));

    assert!(
        plain.contains("red.global.add.s32") && plain.contains("st.global.v2.s32") && !plain.contains(".f32 ["),
        "the plain kernel must both store (unsplit) and reduce (split), in int32"
    );
    assert!(
        // `red.global.add`, not `red.global`: every `cp.async...shared.global` contains that.
        storing.contains("st.global.v2.f32") && !storing.contains("red.global.add"),
        "the scaled kernel is supposed to STORE f32, and only that"
    );
    assert!(
        plain.contains("%ctaid.z") && plain.contains("%nctaid.z"),
        "the plain kernel stopped splitting K; `gpu_batch_invariance` would sweep a geometry it ignores"
    );
    assert!(
        !storing.contains("%ctaid.z") && !storing.contains("%nctaid.z"),
        "the fused epilogue reads the split-K index and then STORES: every z-CTA would write its \
         own partial to the same address\n{}",
        storing
    );
}

/// The plain kernel's stores are reached only when `nctaid.z` is 1: the
/// branch to the reductions must test exactly the register `%nctaid.z` was
/// read into, against 1, and every `st.global` must lie between that branch
/// and the reductions it jumps to.
#[test]
fn the_plain_kernel_stores_only_when_k_is_not_split() {
    let ptx = emit("store_branch", 512, 512, 1024).expect("fixture must compile");
    let body: Vec<&str> = ptx.lines().map(|l| l.trim()).filter(|l| !l.starts_with("//")).collect();
    let nz = body
        .iter()
        .find(|l| l.starts_with("mov.u32") && l.ends_with("%nctaid.z;"))
        .and_then(|l| l.split_whitespace().nth(1))
        .map(|r| r.trim_end_matches(',').to_string())
        .expect("the kernel never reads %nctaid.z");
    let test = body
        .iter()
        .position(|l| l.starts_with("setp.ne.u32") && l.ends_with(&format!(" {nz}, 1;")))
        .unwrap_or_else(|| panic!("no `setp.ne.u32 %p, {nz}, 1`: the store/reduce choice is not on nctaid.z\n{ptx}"));
    let pred = body[test].split_whitespace().nth(1).unwrap().trim_end_matches(',');
    let br = &body[test + 1];
    assert!(br.starts_with(&format!("@{pred} bra ")), "the split test does not branch: `{br}`");
    let target = br.rsplit(' ').next().unwrap().trim_end_matches(';');
    let reduce_at = body.iter().position(|l| *l == format!("{target}:")).expect("the branch target is not defined");
    let stores: Vec<usize> = body.iter().enumerate().filter(|(_, l)| l.contains("st.global")).map(|(i, _)| i).collect();
    let reds: Vec<usize> = body.iter().enumerate().filter(|(_, l)| l.contains("red.global.add")).map(|(i, _)| i).collect();
    assert!(!stores.is_empty() && !reds.is_empty());
    assert!(stores.iter().all(|&i| i > test + 1 && i < reduce_at), "a store lies outside the unsplit branch");
    assert!(reds.iter().all(|&i| i > reduce_at), "a reduction lies outside the split branch");
}

// ---------------------------------------------------------------------------
// The tie to the proof.
// ---------------------------------------------------------------------------

/// The proof's constants and statements must be the emitter's, and the
/// load-bearing ones must keep their content - a theorem weakened to
/// something nothing violates still compiles and still prints "Closed under
/// the global context", which is all `proofs_are_checked` asks.
#[test]
fn the_proof_states_the_schedule_the_emitter_makes() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let v = std::fs::read_to_string(repo.join("proofs/Int8GemmSchedule.v")).expect("proofs/Int8GemmSchedule.v");
    let statement = |name: &str| -> String {
        v.split(&format!("Theorem {name}"))
            .nth(1)
            .unwrap_or_else(|| panic!("the proof has no theorem {name}"))
            .split("Proof.")
            .next()
            .unwrap()
            .to_string()
    };
    for (name, val) in [("MMA_M", 16), ("MMA_N", 8), ("MMA_K", 32)] {
        assert!(v.contains(&format!("Definition {} : nat := {}.", name, val)), "the proof must state {} = {}", name, val);
    }
    // The confinement must stay TWO-SIDED: its lower bound holds for any
    // block size, the upper bound is what the trap buys.
    let confine = statement("the_trap_is_what_confines_a_warp_to_its_own_tile");
    assert!(confine.contains("tid < 32 * (wms * wns)") && confine.contains("< tn * (wns * (nj * MMA_N)) + wns * (nj * MMA_N)"),
            "the confinement theorem lost its hypothesis or its upper bound:\n{confine}");
    let refute = statement("without_the_trap_a_fifth_warp_lands_in_the_next_tile");
    assert!(refute.contains('~') && refute.contains("< 0 * (2 * (4 * MMA_N)) + 2 * (4 * MMA_N)"),
            "the refutation must negate the same bound:\n{refute}");
    // The output map is the epilogue's: tile, warp, mma, half, lane group.
    let row = v.split("Definition out_row").nth(1).unwrap().split('\n').take(2).collect::<String>();
    for term in ["tm * (warps_m * (num_i * MMA_M))", "wm * (num_i * MMA_M)", "i * MMA_M", "8 * h"] {
        assert!(row.contains(term), "out_row must carry `{term}`:\n{row}");
    }
    // The grouped order is the emitter's: groups of G M tiles, the last one short.
    assert!(v.contains("Definition g_size (TM G grp : nat) : nat := Nat.min (TM - grp * G) G."));
    let src = std::fs::read_to_string(repo.join("src/ptx_emitter/int8_staged_gemm.rs")).unwrap();
    assert!(src.contains("min.u32 {}, {}, {};\", gsize, gsize, group_m"), "the emitter's last group is no longer min(TM - first, G)");
    // The K split's owner is the emitter's stripe of K TILES.
    assert!(v.contains("Definition k_tile_owner (T nz s : nat) : nat := (s / T) mod nz."));
    assert!(v.contains("Definition n_my (KT z nz : nat) : nat := if z <? KT then (KT - z + nz - 1) / nz else 0."));
    assert!(src.contains("w(self, format!(\"sub.u32 {}, {}, {};\", n_my, k_tiles, cz));"), "the emitter's stripe count changed");
    // The shape refusal is the emitter's.
    assert!(emit("ok", 64, 32, 128).is_ok(), "a legal shape must compile");
    for (m, n, k) in [(63usize, 32usize, 128usize), (64, 31, 128), (64, 32, 127)] {
        let e = emit("bad", m, n, k).expect_err("a shape short of an mma must be refused");
        assert!(e.contains("M % 16 == 0") || e.contains("m16n8k32"), "{}", e);
    }
}

/// The epilogue half of the tie, in both directions.
#[test]
fn the_proof_states_the_epilogue_split_the_emitter_makes() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let v = std::fs::read_to_string(repo.join("proofs/Int8GemmSchedule.v")).expect("proofs/Int8GemmSchedule.v");
    assert!(
        v.contains("Definition emitted_class (stores : bool) (cz : nat) : nat := if stores then 0 else cz."),
        "the proof no longer says a storing CTA starts at class 0"
    );
    assert!(
        v.contains("Definition emitted_workers (stores : bool) (nz : nat) : nat := if stores then 1 else nz."),
        "the proof no longer says a storing CTA is the only worker"
    );
    let src = std::fs::read_to_string(repo.join("src/ptx_emitter/int8_staged_gemm.rs")).unwrap();
    assert!(src.contains("let stores = epi.is_some();"), "the emitter's split decision is no longer taken from the epilogue");
    assert!(
        v.contains("Theorem a_split_cta_holds_only_part_of_the_contraction")
            && v.contains("Theorem the_unsplit_launch_stores_the_whole_contraction"),
        "the refutation or the unsplit store's theorem was deleted"
    );
}

// ---------------------------------------------------------------------------
// compute-sanitizer: no access outside A, B and C, and no shared-memory race.
// ---------------------------------------------------------------------------

/// The runs the sanitizer checks: ragged M, N and K (K % 64 == 32), each
/// tile geometry, a grid with several tiles per CTA, and K split over z. The
/// buffers are exactly the matrices' size, so a copy or a store past an edge
/// is outside its allocation.
const SANITIZED: [(usize, usize, usize, Option<&str>, (u32, u32, u32)); 9] = [
    (208, 200, 352, None, (0, 1, 1)),
    (208, 200, 352, None, (3, 1, 1)),
    (208, 200, 352, None, (0, 1, 3)),
    (1040, 1048, 352, None, (0, 1, 1)),
    (208, 2056, 352, None, (5, 1, 2)),
    (208, 200, 352, Some("128,128,64,2,2,3,2,32"), (0, 1, 1)),
    (208, 200, 352, Some("64,64,32,2,2,4,4,16"), (2, 1, 1)),
    (208, 200, 352, Some("64,64,128,2,2,2,2,16"), (0, 1, 2)),
    (16, 8, 32, None, (0, 1, 1)),
];

/// The child the sanitizer runs: each of [`SANITIZED`], `grid.0 == 0`
/// meaning one CTA per tile, then the scaled form once.
#[test]
fn child_sanitized_int8_runs() {
    if std::env::var("Y_INT8_SANITIZE").is_err() {
        return;
    }
    let ctx = y::cuda_runtime::CudaContext::new().expect("the parent checked for a driver");
    for (idx, &(m, n, k, forced, grid)) in SANITIZED.iter().enumerate() {
        let env: Vec<(&str, &str)> = forced.map(|t| vec![("Y_INT8_TILE", t)]).unwrap_or_default();
        let ptx = emit_env(&format!("san{idx}"), m, n, k, &env).expect("fixture must compile");
        let geo = int8_launch(&ptx);
        let grid = if grid.0 == 0 { (geo.tiles, grid.1, grid.2) } else { grid };
        let (a, b) = operands(m, n, k, 41, 43);
        let r = run_grids(&ctx, &ptx, (m, n, k), &a, &b, &[(grid, if grid.2 == 1 { 0xA5 } else { 0 })]);
        assert_eq!(r[0].0, 0, "sanitized run {idx}: wrong bits");
        println!("SANITIZED_I8_RUN {idx}");
    }
    let (m, n, k) = (208usize, 200usize, 352usize);
    let ptx = emit_scaled("sanscaled", m, n, k).expect("scaled fixture did not compile");
    let geo = int8_launch(&ptx);
    let module = ctx.load_ptx(&ptx, "int8_gemm_scaled").unwrap();
    module.set_max_dynamic_smem(geo.smem).unwrap();
    let (a, b) = operands(m, n, k, 41, 43);
    let bufs: Vec<_> = [m * k, n * k, m * 4, n * 4, n * 4, m * n * 4].iter().map(|&s| ctx.alloc(s).unwrap()).collect();
    ctx.memcpy_htod_at(&bufs[0], 0, &a.iter().map(|&v| v as u8).collect::<Vec<u8>>()).unwrap();
    ctx.memcpy_htod_at(&bufs[1], 0, &b.iter().map(|&v| v as u8).collect::<Vec<u8>>()).unwrap();
    for buf in &bufs[2..] {
        ctx.memset_u8(buf, 0).unwrap();
    }
    let args: Vec<_> = bufs.iter().map(|d| d.device_ptr()).collect();
    ctx.launch(&module, (geo.tiles, 1, 2), (geo.block, 1, 1), geo.smem, &args).unwrap();
    ctx.synchronize().unwrap();
    println!("SANITIZED_I8_RUN scaled");
}

#[test]
fn compute_sanitizer_finds_no_invalid_access_and_no_race() {
    if y::cuda_runtime::CudaContext::new().is_none() {
        eprintln!("SKIP: no CUDA driver - nothing was sanitized.");
        return;
    }
    let tool = ["compute-sanitizer", "/opt/cuda/bin/compute-sanitizer", "/usr/local/cuda/bin/compute-sanitizer"]
        .into_iter()
        .find(|t| Command::new(t).arg("--version").output().is_ok_and(|o| o.status.success()));
    let Some(tool) = tool else {
        eprintln!("SKIP: no compute-sanitizer - the kernel was not checked for invalid accesses or races.");
        return;
    };
    // memcheck: every read inside A, B (and the scales), every write inside C.
    // racecheck: no shared-memory hazard, including a stage the next K tile
    // or the next output tile reuses.
    for (check, summary) in [("memcheck", "ERROR SUMMARY: 0 errors"), ("racecheck", "RACECHECK SUMMARY: 0 hazards displayed")] {
        let out = Command::new(tool)
            .args(["--tool", check, "--print-limit", "0"])
            .arg(std::env::current_exe().unwrap())
            .args(["child_sanitized_int8_runs", "--exact", "--nocapture", "--test-threads=1"])
            .env("Y_INT8_SANITIZE", "1")
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
        // Every run ran: a child that died early prints a clean summary for
        // the launches it made.
        assert_eq!(text.matches("SANITIZED_I8_RUN").count(), SANITIZED.len() + 1, "{check}: not every run ran:\n{text}");
        assert!(out.status.success() && text.contains(summary), "{check} found something:\n{text}");
    }
}
