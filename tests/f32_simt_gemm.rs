//! The FP32 GEMM on CUDA cores: `@tile(M, N, K) kernel g(A, B, C:
//! GlobalMemory<F32>)` (`src/ptx_emitter/f32_simt_gemm.rs`).
//!
//! Any M, N and K. Its claim is that every element of C is ONE `fma.rn.f32` chain over
//! k = 0..K-1, in that order, whatever the shape, the tile the shape selects
//! and the grid it is launched with. That is exactly what `f32::mul_add`
//! folded in the same order computes on the CPU, so the GPU results are
//! compared bit for bit - including on grids smaller and larger than one tile
//! per CTA, against an output pre-filled with NaN so an unwritten element
//! cannot pass. The operands span exponents 2^-3..2^4 with both signs, so a
//! different summation order would change bits.
//!
//! A launch with the wrong block traps; that runs in a child process (this
//! binary, re-run with `Y_F32_GEMM_CHILD_BLOCK`), since a trap leaves the
//! device unusable for the rest of its process.
//!
//! Speed is not tested here (it is measured, not asserted); the module doc
//! of `f32_simt_gemm.rs` has the numbers.

use std::path::Path;
use std::process::Command;
use y::cuda_runtime::CudaContext;

#[path = "common/pinned.rs"]
mod pinned;
#[path = "common/ptxas.rs"]
mod ptxas;

/// What a kernel's PTX header states.
struct Header {
    tile: String,
    block: u32,
    grid: (u32, u32),
    smem: u32,
}

fn header(ptx: &str) -> Header {
    let line = |pat: &str| ptx.lines().find(|l| l.contains(pat)).unwrap_or_else(|| panic!("no `{}` in the PTX", pat)).to_string();
    let h = line("[Y F32 SIMT GEMM]");
    let tile = h.split("tile `").nth(1).unwrap().split('`').next().unwrap().to_string();
    let launch = line("Launch: block (");
    let block: u32 = launch.split("block (").nth(1).unwrap().split(',').next().unwrap().parse().unwrap();
    let g = launch.split("works, (").nth(1).unwrap().split(", 1) is one tile").next().unwrap();
    let mut g = g.split(", ").map(|v| v.parse::<u32>().unwrap());
    let grid = (g.next().unwrap(), g.next().unwrap());
    let smem: u32 = line("Dynamic shared memory required:").split("required: ").nth(1).unwrap().split(' ').next().unwrap().parse().unwrap();
    Header { tile, block, grid, smem }
}

fn source(name: &str, m: u32, n: u32, k: u32) -> String {
    format!("@tile({m}, {n}, {k})\nkernel {name}(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>) {{\n}}\n\nfn main() {{\n}}\n")
}

/// Compile in `dir` (pinned profile: SM 8.0, 66 SMs). Ok(PTX) or Err(log).
fn compile(dir: &Path, name: &str, m: u32, n: u32, k: u32, backend: &str) -> Result<String, String> {
    let src = dir.join(format!("{name}.ysu"));
    std::fs::write(&src, source(name, m, n, k)).unwrap();
    let out_path = dir.join(format!("{name}.out"));
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&src)
        .arg(backend)
        .arg("-o")
        .arg(&out_path)
        .current_dir(dir)
        .output()
        .expect("run Y");
    let log = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    if !out.status.success() {
        return Err(log);
    }
    Ok(std::fs::read_to_string(&out_path).unwrap_or_else(|_| panic!("{name}: no output written\n{log}")))
}

#[test]
fn the_tile_follows_the_shape_and_each_kernel_assembles_without_spills() {
    let dir = pinned::pinned_scratch("f32_gemm_tiles", pinned::SM_PINNED);
    // (shape, tile, block, .maxnreg): the large tile needs ~2 waves of its
    // 128x256 tiles on 66 SMs, at least 80% full, and 128 rows; the mid tile
    // ~1 wave of 64x128. Ragged shapes count their edge tiles whole.
    let cases = [
        ((4096, 4096, 4096), "large", 256, 255),
        ((2048, 2048, 2048), "large", 256, 255),
        ((1536, 1536, 1536), "mid", 128, 170),
        ((1024, 1024, 1024), "mid", 128, 170),
        // 64 large tiles: one full wave is not enough for the large tile.
        ((1024, 2048, 1024), "mid", 128, 170),
        ((128, 4096, 4096), "mid", 128, 170),
        ((512, 512, 512), "small", 128, 128),
        ((64, 64, 16), "small", 128, 128),
        // 272 large tiles, 4.1 waves: 82% full.
        ((2048, 4160, 64), "large", 256, 255),
        // 153 large tiles, 2.3 waves: 77% full, so not the large tile.
        ((2049, 2049, 2048), "mid", 128, 170),
        // Odd N (4-byte B copies), many waves.
        ((1024, 50257, 768), "large", 256, 255),
        // One row: never the 128-row tile.
        ((1, 50257, 768), "mid", 128, 170),
        ((1023, 1025, 1027), "mid", 128, 170),
        ((7, 13, 5), "small", 128, 128),
        ((1, 1, 1), "small", 128, 128),
    ];
    let ptxas = ptxas::ptxas();
    for (i, ((m, n, k), tile, block, maxnreg)) in cases.iter().enumerate() {
        let name = format!("g{i}");
        let ptx = compile(&dir, &name, *m, *n, *k, "--emit-ptx").unwrap_or_else(|log| panic!("{m}x{n}x{k} refused:\n{log}"));
        let h = header(&ptx);
        assert_eq!((h.tile.as_str(), h.block), (*tile, *block), "{m}x{n}x{k}");
        assert!(ptx.contains(&format!(".maxnreg {maxnreg}")), "{m}x{n}x{k}: not .maxnreg {maxnreg}");
        assert!(ptx.contains(&format!(".extern .shared .align 16 .b8 smem_f32_gemm_{name}[];")), "{m}x{n}x{k}");
        // Reading the transposed A, B and the C tile: one fma per k step per
        // output, nothing else does arithmetic on the values.
        let fmas = ptx.matches("fma.rn.f32").count();
        assert!(fmas > 0 && !ptx.contains("add.f32") && !ptx.contains("mul.f32") && !ptx.contains("add.rn.f32"), "{m}x{n}x{k}");
        if let Some(ptxas) = &ptxas {
            let ptx_path = dir.join(format!("{name}.ptx"));
            std::fs::write(&ptx_path, &ptx).unwrap();
            let out = Command::new(ptxas).args(["-arch=sm_89", "-v", "-o"]).arg(dir.join(format!("{name}.cubin"))).arg(&ptx_path).output().unwrap();
            let log = String::from_utf8_lossy(&out.stderr).to_string();
            assert!(out.status.success(), "{m}x{n}x{k}: ptxas refused it:\n{log}");
            assert!(log.contains("0 bytes spill stores, 0 bytes spill loads"), "{m}x{n}x{k} spills:\n{log}");
        }
    }
    if ptxas.is_none() {
        eprintln!("SKIP (assembly): no ptxas - the kernels were not assembled.");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn what_cannot_be_addressed_is_refused_and_so_is_the_host_backend() {
    let dir = pinned::pinned_scratch("f32_gemm_refused", pinned::SM_PINNED);
    // One tile's rows are addressed with 32-bit offsets.
    for (i, (m, n, k)) in [(64, 64, 1 << 22), (64, 1 << 22, 64)].iter().enumerate() {
        let log = compile(&dir, &format!("r{i}"), *m, *n, *k, "--emit-ptx").expect_err("an unaddressable shape was compiled");
        assert!(log.contains("K and N must stay below 2^22"), "{m}x{n}x{k}:\n{log}");
    }
    // Just below the bound compiles.
    compile(&dir, "edge", 64, 64, (1 << 22) - 1, "--emit-ptx").unwrap_or_else(|log| panic!("{log}"));
    // The source has no body; on the host it would be an empty function.
    let log = compile(&dir, "host", 128, 128, 64, "--emit-llvm").expect_err("the host backend compiled an empty FP32 GEMM");
    assert!(log.contains("is an `@tile` FP32 GEMM, whose body only the PTX backend"), "{log}");
    // The signature is accepted by the type checker (the list names it) and
    // a near miss still gets the list.
    let src = dir.join("near.ysu");
    std::fs::write(&src, "@tile(128, 128, 64)\nkernel near(A: GlobalMemory<F32>, B: GlobalMemory<F16>, C: GlobalMemory<F32>) {\n}\n\nfn main() {\n}\n").unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_Y")).arg(&src).arg("--emit-ptx").current_dir(&dir).output().unwrap();
    let log = String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success() && log.contains("(A, B, C: GlobalMemory<F32>)") && log.contains("FP32 GEMM (CUDA cores)"), "{log}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Operands with both signs and exponents 2^-3..2^4, so that any other
/// summation order changes bits.
fn operand(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (s >> 32) as u32;
            let mant = 1.0 + (u & 0xffff) as f32 / 65536.0;
            let exp = ((u >> 16) & 7) as i32 - 3;
            let sign = if (u >> 19) & 1 == 1 { -1.0 } else { 1.0 };
            sign * mant * 2f32.powi(exp)
        })
        .collect()
}

/// `C[i][j]` as the kernel claims to compute it.
fn oracle(a: &[f32], b: &[f32], n: usize, k: usize, i: usize, j: usize) -> f32 {
    let mut acc = 0f32;
    for kk in 0..k {
        acc = a[i * k + kk].mul_add(b[kk * n + j], acc);
    }
    acc
}

fn as_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn each_element_is_the_in_order_fma_chain_on_every_grid() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver - no FP32 GEMM was executed.");
        return;
    };
    let dir = pinned::pinned_scratch("f32_gemm_run", pinned::SM_PINNED);
    // Each tile; K = 16 is one K tile (fewer than the pipeline's stages),
    // K = 32 two (the prologue loads them all), the others run the loop past
    // its last load.
    let shapes: [((u32, u32, u32), &str); 18] = [
        ((64, 64, 16), "small"),
        ((192, 320, 112), "small"),
        ((128, 4096, 80), "mid"),
        ((640, 1024, 48), "mid"),
        ((2048, 4096, 32), "large"),
        ((2048, 4096, 112), "large"),
        // Ragged: a single element; M, N and K all ragged with N odd; K
        // below one K tile; N a multiple of 4 but not of the tile; and each
        // tile with ragged M and N, odd N on the large tile.
        ((1, 1, 1), "small"),
        ((7, 13, 5), "small"),
        // N = 130: even, but a B row is not 16-byte aligned.
        ((66, 130, 24), "small"),
        ((33, 65, 17), "small"),
        ((129, 257, 15), "small"),
        ((300, 1028, 51), "small"),
        // N ragged with M whole (the column mask alone).
        ((128, 1028, 40), "small"),
        ((2048, 4100, 20), "large"),
        ((1023, 1025, 75), "mid"),
        ((1, 8193, 40), "mid"),
        ((2047, 4100, 20), "large"),
        ((2047, 4099, 36), "large"),
    ];
    for (si, ((m, n, k), tile)) in shapes.iter().enumerate() {
        let (m, n, k) = (*m as usize, *n as usize, *k as usize);
        let name = format!("run{si}");
        let ptx = compile(&dir, &name, m as u32, n as u32, k as u32, "--emit-ptx").unwrap_or_else(|log| panic!("{log}"));
        let h = header(&ptx);
        assert_eq!(h.tile, *tile, "{m}x{n}x{k}");
        let module = ctx.load_ptx(&ptx, &name).unwrap_or_else(|e| panic!("{m}x{n}x{k} did not load: {e}"));
        module.set_max_dynamic_smem(h.smem).unwrap();
        let a = operand(m * k, 11 + si as u64);
        let b = operand(k * n, 97 + si as u64);
        let (da, db, dc) = (ctx.alloc(m * k * 4).unwrap(), ctx.alloc(k * n * 4).unwrap(), ctx.alloc(m * n * 4).unwrap());
        ctx.memcpy_htod_at(&da, 0, &as_bytes(&a)).unwrap();
        ctx.memcpy_htod_at(&db, 0, &as_bytes(&b)).unwrap();
        // Rows checked: all of them up to 640 rows; on the 2048-row shapes
        // the first and last tile rows and every 7th row (7 is prime to
        // every tile height, so every residue of a tile is checked).
        let rows: Vec<usize> = (0..m).filter(|&i| m <= 640 || i < 128 || i >= m - 128 || i % 7 == 0).collect();
        let want: Vec<Vec<f32>> = rows.iter().map(|&i| (0..n).map(|j| oracle(&a, &b, n, k, i, j)).collect()).collect();
        for grid in [(h.grid.0, h.grid.1, 1), (3, 1, 1), (h.grid.0 + 5, h.grid.1, 2)] {
            ctx.memset_u8(&dc, 0xff).unwrap(); // NaN everywhere
            ctx.launch(&module, grid, (h.block, 1, 1), h.smem, &[da.device_ptr(), db.device_ptr(), dc.device_ptr()]).unwrap();
            ctx.synchronize().unwrap_or_else(|e| panic!("{m}x{n}x{k} grid {grid:?}: {e}"));
            let mut raw = vec![0u8; m * n * 4];
            ctx.memcpy_dtoh_at(&mut raw, &dc, 0).unwrap();
            let got: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            for (ri, &i) in rows.iter().enumerate() {
                for j in 0..n {
                    let (g, w) = (got[i * n + j], want[ri][j]);
                    assert!(
                        g.to_bits() == w.to_bits(),
                        "{m}x{n}x{k} ({tile}) grid {grid:?}: C[{i}][{j}] = {g:e} ({:#010x}), the in-order fma chain gives {w:e} ({:#010x})",
                        g.to_bits(),
                        w.to_bits()
                    );
                }
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Child: launch the 4096^3 kernel (block 256) with the block in
/// `Y_F32_GEMM_CHILD_BLOCK` and print what the device said.
#[test]
fn child_launch_with_a_given_block() {
    let Ok(block) = std::env::var("Y_F32_GEMM_CHILD_BLOCK") else { return };
    let block: u32 = block.parse().unwrap();
    let ctx = CudaContext::new().expect("the parent checked for a driver");
    let dir = pinned::pinned_scratch(&format!("f32_gemm_child_{block}"), pinned::SM_PINNED);
    let (m, n, k) = (2048usize, 4096usize, 16usize);
    let ptx = compile(&dir, "child", m as u32, n as u32, k as u32, "--emit-ptx").unwrap();
    let h = header(&ptx);
    let module = ctx.load_ptx(&ptx, "child").unwrap();
    module.set_max_dynamic_smem(h.smem).unwrap();
    let (da, db, dc) = (ctx.alloc(m * k * 4).unwrap(), ctx.alloc(k * n * 4).unwrap(), ctx.alloc(m * n * 4).unwrap());
    for d in [&da, &db, &dc] {
        ctx.memset_u8(d, 0).unwrap();
    }
    let r = ctx.launch(&module, (h.grid.0, h.grid.1, 1), (block, 1, 1), h.smem, &[da.device_ptr(), db.device_ptr(), dc.device_ptr()]).and_then(|_| ctx.synchronize());
    println!("CHILD_RESULT block={block}: {}", match r {
        Ok(()) => "ok".to_string(),
        Err(e) => format!("error {e}"),
    });
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_launch_with_the_wrong_block_traps() {
    if CudaContext::new().is_none() {
        eprintln!("SKIP: no CUDA driver - the launch contract was not exercised.");
        return;
    }
    let run = |block: u32| {
        let out = Command::new(std::env::current_exe().unwrap())
            .args(["child_launch_with_a_given_block", "--exact", "--nocapture", "--test-threads=1"])
            .env("Y_F32_GEMM_CHILD_BLOCK", block.to_string())
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout).to_string();
        // The harness prints `test name ... ` with no newline, so the
        // result shares that line: match it anywhere in the line.
        text.lines()
            .find_map(|l| l.find("CHILD_RESULT").map(|at| l[at..].to_string()))
            .unwrap_or_else(|| panic!("block {block}: the child printed no result:\n{text}{}", String::from_utf8_lossy(&out.stderr)))
    };
    // The control: the stated block runs.
    assert_eq!(run(256), "CHILD_RESULT block=256: ok");
    // Too few threads leave copies and outputs without an owner; too many
    // would split `bar.sync` across threads with no role.
    for block in [128, 512] {
        let r = run(block);
        assert!(r.starts_with(&format!("CHILD_RESULT block={block}: error")), "block {block} was not stopped: {r}");
    }
}

/// Child: small runs for compute-sanitizer - each tile, one K tile (fewer than
/// the pipeline's stages) and several, one tile per CTA and many per CTA (the
/// 3x1 grid reuses every stage across tiles). Prints the rows it checked.
#[test]
fn child_sanitized_runs() {
    if std::env::var("Y_F32_GEMM_SANITIZE").is_err() {
        return;
    }
    let ctx = CudaContext::new().expect("the parent checked for a driver");
    let dir = pinned::pinned_scratch("f32_gemm_sanitized", pinned::SM_PINNED);
    let shapes = [
        (64usize, 64usize, 16usize),
        (128, 4096, 48),
        (2048, 4096, 16),
        (2048, 4096, 64),
        // Ragged: rows past M and k past K are copies of size 0 from
        // addresses past the buffers, which must not be read.
        (1, 1, 1),
        (33, 65, 17),
        (129, 257, 15),
        (300, 1028, 51),
        (128, 1028, 40),
        (1023, 1025, 75),
        (2047, 4099, 36),
    ];
    for (si, &(m, n, k)) in shapes.iter().enumerate() {
        let name = format!("san{si}");
        let ptx = compile(&dir, &name, m as u32, n as u32, k as u32, "--emit-ptx").unwrap();
        let h = header(&ptx);
        let module = ctx.load_ptx(&ptx, &name).unwrap();
        module.set_max_dynamic_smem(h.smem).unwrap();
        let (a, b) = (operand(m * k, 5 + si as u64), operand(k * n, 7 + si as u64));
        // Exactly sized buffers: a read one element past A or B is outside
        // its allocation.
        let (da, db, dc) = (ctx.alloc(m * k * 4).unwrap(), ctx.alloc(k * n * 4).unwrap(), ctx.alloc(m * n * 4).unwrap());
        ctx.memcpy_htod_at(&da, 0, &as_bytes(&a)).unwrap();
        ctx.memcpy_htod_at(&db, 0, &as_bytes(&b)).unwrap();
        for grid in [(h.grid.0, h.grid.1, 1), (3, 1, 1)] {
            ctx.launch(&module, grid, (h.block, 1, 1), h.smem, &[da.device_ptr(), db.device_ptr(), dc.device_ptr()]).unwrap();
            ctx.synchronize().unwrap();
            let mut raw = vec![0u8; m * n * 4];
            ctx.memcpy_dtoh_at(&mut raw, &dc, 0).unwrap();
            for i in [0, m / 2, m - 1] {
                for j in 0..n {
                    let g = f32::from_le_bytes(raw[(i * n + j) * 4..(i * n + j) * 4 + 4].try_into().unwrap());
                    assert_eq!(g.to_bits(), oracle(&a, &b, n, k, i, j).to_bits(), "{m}x{n}x{k} grid {grid:?} C[{i}][{j}]");
                }
            }
        }
        println!("SANITIZED_RUN {m}x{n}x{k} {}", h.tile);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compute_sanitizer_finds_no_invalid_access_and_no_race() {
    if CudaContext::new().is_none() {
        eprintln!("SKIP: no CUDA driver - nothing was sanitized.");
        return;
    }
    let tool = ["compute-sanitizer", "/opt/cuda/bin/compute-sanitizer", "/usr/local/cuda/bin/compute-sanitizer"]
        .into_iter()
        .find(|t| Command::new(t).arg("--version").output().is_ok_and(|o| o.status.success()));
    let Some(tool) = tool else {
        eprintln!("SKIP: no compute-sanitizer - the kernels were not checked for invalid accesses or races.");
        return;
    };
    // memcheck: every read inside A and B, every write inside C. racecheck:
    // no shared-memory hazard, including a stage reused by the next tile.
    for (check, summary) in [("memcheck", "ERROR SUMMARY: 0 errors"), ("racecheck", "RACECHECK SUMMARY: 0 hazards displayed")] {
        let out = Command::new(tool)
            .args(["--tool", check, "--print-limit", "0"])
            .arg(std::env::current_exe().unwrap())
            .args(["child_sanitized_runs", "--exact", "--nocapture", "--test-threads=1"])
            .env("Y_F32_GEMM_SANITIZE", "1")
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
        // Every shape ran (a child that died early prints a clean summary
        // for the launches it made).
        assert_eq!(text.matches("SANITIZED_RUN").count(), 11, "{check}: not every shape ran:\n{text}");
        assert!(out.status.success() && text.contains(summary), "{check} found something:\n{text}");
    }
}

/// The last, partial K tile runs only its valid steps. Every product here
/// rounds to -0.0, so each element is -0.0; one padded step (0 * 0 + -0.0)
/// would make it +0.0.
#[test]
fn a_ragged_k_keeps_the_sign_of_a_zero_sum() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver - the signed zero was not checked.");
        return;
    };
    let dir = pinned::pinned_scratch("f32_gemm_negzero", pinned::SM_PINNED);
    for (si, &(m, n, k)) in [(64usize, 64usize, 17usize), (2047, 4100, 20), (1023, 1025, 75)].iter().enumerate() {
        let name = format!("z{si}");
        let ptx = compile(&dir, &name, m as u32, n as u32, k as u32, "--emit-ptx").unwrap();
        let h = header(&ptx);
        let module = ctx.load_ptx(&ptx, &name).unwrap();
        module.set_max_dynamic_smem(h.smem).unwrap();
        let (a, b) = (vec![-1e-30f32; m * k], vec![1e-30f32; k * n]);
        assert_eq!(oracle(&a, &b, n, k, 0, 0).to_bits(), (-0.0f32).to_bits(), "the premise: the in-order chain is -0.0");
        let (da, db, dc) = (ctx.alloc(m * k * 4).unwrap(), ctx.alloc(k * n * 4).unwrap(), ctx.alloc(m * n * 4).unwrap());
        ctx.memcpy_htod_at(&da, 0, &as_bytes(&a)).unwrap();
        ctx.memcpy_htod_at(&db, 0, &as_bytes(&b)).unwrap();
        ctx.memset_u8(&dc, 0xff).unwrap();
        ctx.launch(&module, (h.grid.0, h.grid.1, 1), (h.block, 1, 1), h.smem, &[da.device_ptr(), db.device_ptr(), dc.device_ptr()]).unwrap();
        ctx.synchronize().unwrap();
        let mut raw = vec![0u8; m * n * 4];
        ctx.memcpy_dtoh_at(&mut raw, &dc, 0).unwrap();
        let positive = raw.chunks_exact(4).filter(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) != 0x8000_0000).count();
        assert_eq!(positive, 0, "{m}x{n}x{k} ({}): {positive} of {} elements are not -0.0", h.tile, m * n);
    }
    let _ = std::fs::remove_dir_all(&dir);
}
