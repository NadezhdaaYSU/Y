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
//! The same GEMM with a workspace, `(A, B, C: GlobalMemory<F32>, W:
//! GlobalMemory<U64>)`, splits K in order where one tile per CTA would leave
//! SMs idle (`f32_gemm_streamk_plan`): a tile cut between two CTAs is
//! continued from the exact accumulators the first stored in C, so its
//! claim is the same chain, checked the same way - on grids from one CTA to
//! more CTAs than units (a tile cut in many pieces), on a W of garbage that
//! is never cleared between launches.
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
    compile_text(dir, name, &source(name, m, n, k), backend)
}

fn compile_text(dir: &Path, name: &str, text: &str, backend: &str) -> Result<String, String> {
    let src = dir.join(format!("{name}.ysu"));
    std::fs::write(&src, text).unwrap();
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
        ((512, 512, 512), "small", 128, 170),
        ((64, 64, 16), "small", 128, 170),
        // 272 large tiles, 4.1 waves: 82% full.
        ((2048, 4160, 64), "large", 256, 255),
        // 153 large tiles, 2.3 waves: 77% full, so not the large tile.
        ((2049, 2049, 2048), "mid", 128, 170),
        // Odd N (4-byte B copies), many waves.
        ((1024, 50257, 768), "large", 256, 255),
        // One row: never the 128-row tile.
        ((1, 50257, 768), "mid", 128, 170),
        ((1023, 1025, 1027), "mid", 128, 170),
        ((7, 13, 5), "small", 128, 170),
        ((1, 1, 1), "small", 128, 170),
        // All three tile sets in one kernel, under one register cap.
        ((4097, 4097, 64), "large", 256, 255),
        // B at each row's width, and with a K tail at one width.
        ((2048, 12803, 64), "large", 256, 255),
        ((2048, 12803, 67), "large", 256, 255),
        ((256, 12802, 80), "mid", 128, 170),
        ((2047, 2050, 2051), "mid", 128, 170),
        ((2049, 2049, 40), "mid", 128, 170),
        ((65, 70, 20), "small", 128, 170),
    ];
    // Leftover rows and columns of at most half a tile, on a side of at most
    // 50 tiles, go to remainder tiles in the same launch; anything else stays
    // in ragged whole tiles.
    let remainders = [
        ((1025, 1024, 1024), "rows 1024..1025 in 8 tiles `rows32x128`; none"),
        ((1024, 1028, 1024), "none; columns 1024..1028 in 16 tiles `rem64x32`"),
        ((2049, 2049, 2048), "rows 2048..2049 in 16 tiles `rows32x128`; columns 2048..2049 in 33 tiles `rem64x32`"),
        ((4097, 4097, 64), "rows 4096..4097 in 16 tiles `rows32x256`; columns 4096..4097 in 33 tiles `rem128x64`"),
        ((65, 70, 20), "rows 64..65 in 1 tiles `rows32x64`; columns 64..70 in 2 tiles `rem64x32`"),
        // 40 of 64 rows and 104 of 128 columns left: whole tiles.
        ((1000, 1000, 1000), "none; none"),
        // 196 tile columns: the ragged one is 0.5% of the work, and with N a
        // multiple of 4 splitting it off gains nothing else.
        ((1024, 50180, 768), "none; none"),
        // As wide, but N odd: splitting lets the whole tiles copy B at each
        // row's width.
        ((1024, 50257, 768), "none; columns 50176..50257 in 16 tiles `rem128x64`"),
    ];
    // How B is copied, from the header: one width where N allows it, each
    // row at its own alignment's width where it does not.
    let widths = [
        ((1024, 1024, 1024), "B by 16-byte copies"),
        ((1024, 30522, 768), "B by row-aligned 16-byte (rows 0 + 2j), 8-byte (rows 1 + 2j) copies"),
        ((1024, 50257, 768), "B by row-aligned 16-byte (rows 0 + 4j), 8-byte (rows 2 + 4j), 4-byte (rows 1 + 2j) copies"),
        // N odd on the 128-column mid tile: one width.
        ((1023, 1025, 1027), "B by 4-byte copies"),
        // An edge the remainder does not take (130 of 256 columns left):
        // one width that divides N, so no chunk straddles it.
        ((2048, 4226, 32), "B by 8-byte copies"),
        // N = 2 mod 4 on the small tile.
        ((66, 130, 64), "B by row-aligned 16-byte (rows 0 + 2j), 8-byte (rows 1 + 2j) copies"),
        // A K tail keeps one width.
        ((66, 130, 72), "B by 8-byte copies"),
        ((1024, 50257, 770), "B by 4-byte copies"),
    ];
    for (i, ((m, n, k), want)) in widths.iter().enumerate() {
        let ptx = compile(&dir, &format!("w{i}"), *m, *n, *k, "--emit-ptx").unwrap_or_else(|log| panic!("{log}"));
        assert!(ptx.contains(want), "{m}x{n}x{k}: not `{want}`:\n{}", ptx.lines().find(|l| l.contains("B by")).unwrap_or(""));
    }
    let ptxas = ptxas::ptxas();
    for (i, ((m, n, k), want)) in remainders.iter().enumerate() {
        let ptx = compile(&dir, &format!("rem{i}"), *m, *n, *k, "--emit-ptx").unwrap_or_else(|log| panic!("{log}"));
        let line = ptx.lines().find(|l| l.contains("// Remainder: ")).unwrap_or_else(|| panic!("{m}x{n}x{k}: no remainder line"));
        assert!(line.ends_with(&format!("Remainder: {want}.")), "{m}x{n}x{k}: {line}");
    }
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
fn odd_n_stores_c_in_aligned_pieces_on_the_256_column_tiles() {
    let dir = pinned::pinned_scratch("f32_gemm_odd_stores", pinned::SM_PINNED);
    // A thread's float4 of C sits at place (row * N + e) mod 4 of a 16-byte
    // line, and its four rows are 0..3 mod 4: with N odd the places are 0, 1,
    // 2 and 3 (in either order), stored as v4, f32 + v2 + f32, v2 + v2 and
    // f32 + v2 + f32. The large tile has 2 x 4 such (row group, column
    // group) pairs: 8 v4, 32 v2 and 32 f32. The 81 or 83 columns past its
    // whole tiles are a ragged `rem128x64` set, still 4 bytes at a time
    // (32 f32).
    for (m, n, k) in [(1024u32, 50257u32, 768u32), (1024, 50259, 768)] {
        let ptx = compile(&dir, &format!("odd_{n}"), m, n, k, "--emit-ptx").unwrap_or_else(|log| panic!("{log}"));
        assert_eq!(header(&ptx).tile, "large", "N = {n}");
        let count = |s: &str| ptx.matches(s).count();
        assert_eq!(
            (count("st.global.v4.f32 ["), count("st.global.v2.f32 ["), count("st.global.f32 [")),
            (8, 32, 64),
            "N = {n}: not the stores this test expects"
        );
    }
    // The 128-column mid tile keeps 4-byte stores: there the pieces ran
    // 1.02-1.09x the time at 1023 x 1025 x 1027.
    let ptx = compile(&dir, "odd_mid", 1023, 1025, 1027, "--emit-ptx").unwrap_or_else(|log| panic!("{log}"));
    assert_eq!(header(&ptx).tile, "mid");
    assert_eq!(ptx.matches("st.global.v2.f32").count() + ptx.matches("st.global.v4.f32").count(), 0, "a wide C store on the mid tile");
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
    let log = compile_text(&dir, "hostw", &source_w("hostw", 128, 128, 64), "--emit-llvm").expect_err("the host backend compiled an empty FP32 GEMM with W");
    assert!(log.contains("is an `@tile` FP32 GEMM, whose body only the PTX backend"), "{log}");
    // The workspace is U64 words; any other element type is not the shape.
    let wrong = source_w("wrongw", 128, 128, 64).replace("W: GlobalMemory<U64>", "W: GlobalMemory<U32>");
    let log = compile_text(&dir, "wrongw", &wrong, "--emit-ptx").expect_err("a U32 workspace was accepted");
    assert!(log.contains("FP32 GEMM, K split in order"), "{log}");
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

/// (shape, tile at the pinned sm_80 profile)
const EXACT_SHAPES: [((u32, u32, u32), &str); 43] = [
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
    ((66, 130, 64), "small"),
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
    // Remainder tiles: rows (`rows32x128`), rows and columns on the
    // large tile (`rows32x256`, `rem128x64`) and the small one
    // (`rows32x64`, `rem64x32`), N odd in a remainder.
    ((1025, 1024, 40), "mid"),
    ((2049, 4100, 24), "large"),
    ((65, 70, 20), "small"),
    ((1025, 1025, 33), "mid"),
    // Leftover rows that exactly fill their 32-row strips.
    ((1056, 1024, 24), "mid"),
    ((2112, 4096, 24), "large"),
    // B rows copied at their own alignment's width (16, 8 or 4 bytes):
    // N = 2 mod 4 on the mid and large tiles, N odd on the large one,
    // with K past the prologue's tiles so every width's pointer is
    // advanced in the loop; then the same with a K tail, which keeps one
    // width, and an edge the remainder does not take.
    ((256, 12802, 80), "mid"),
    ((2048, 12802, 16), "large"),
    ((2048, 12803, 64), "large"),
    ((2048, 4098, 48), "large"),
    ((256, 12802, 83), "mid"),
    ((2048, 12803, 67), "large"),
    ((2048, 4226, 32), "large"),
    // C stored in each float's widest aligned pieces (v4, v2 + v2, or
    // f32 + v2 + f32 by its place in a 16-byte line) on the 256-column
    // tiles with N odd: 12803 and 4099 above are 3 mod 4, these 1 mod 4,
    // the second with a 12-row `rows32x256` strip.
    ((2048, 4097, 40), "large"),
    ((2060, 4097, 24), "large"),
    // Built for sm_75 on 40 SMs, these two run on the 8 x 256 tile, which
    // no shape above reaches there (the 32 x 64 one is: 33 x 65 x 17; the
    // 128-thread GEMV 1 x 8193 x 40; the few-column 4 x 32, 8 x 16 and 16 x
    // 32 tiles 1 x 1 x 1, 7 x 13 x 5 and 64 x 64 x 16): whole, and with M
    // ragged, N = 2 mod 4 (8-byte B copies) and a K tail; then the 16 x 128
    // tile at N odd; the 4 x 32 tile with a K tail; the 8 x 16 tile on two
    // tile rows, M and N ragged, two K slices and a tail; the 32-thread GEMV
    // with a one-step tail; the 4 x 32 and 8 x 16 tiles over two and three
    // K slices and a tail; the 16 x 32 tile likewise; and the 32 x 32 tile.
    ((8, 4096, 40), "small"),
    ((5, 4098, 35), "small"),
    ((12, 4097, 24), "small"),
    ((3, 1001, 37), "small"),
    ((11, 300, 150), "small"),
    ((2, 3000, 33), "small"),
    ((3, 100, 200), "small"),
    ((7, 50, 150), "small"),
    ((11, 777, 77), "small"),
    ((96, 256, 24), "small"),
];

/// Every element of each EXACT_SHAPES GEMM, compiled in `dir` (whose
/// profile names the target), equal bit for bit to the in-order fma chain on
/// three grids. `tile_for(tile at sm_80, ptx, m, n, k)` checks the header's
/// tile and names it for the messages.
fn each_element_exact(ctx: &CudaContext, dir: &Path, tile_for: impl Fn(&str, &str, usize, usize, usize) -> String) {
    for (si, ((m, n, k), tile)) in EXACT_SHAPES.iter().enumerate() {
        let (m, n, k) = (*m as usize, *n as usize, *k as usize);
        let name = format!("run{si}");
        let ptx = compile(&dir, &name, m as u32, n as u32, k as u32, "--emit-ptx").unwrap_or_else(|log| panic!("{log}"));
        let h = header(&ptx);
        let tile = &tile_for(*tile, &ptx, m, n, k);
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
}

#[test]
fn each_element_is_the_in_order_fma_chain_on_every_grid() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver - no FP32 GEMM was executed.");
        return;
    };
    let dir = pinned::pinned_scratch("f32_gemm_run", pinned::SM_PINNED);
    each_element_exact(&ctx, &dir, |want, ptx, m, n, k| {
        let h = header(ptx);
        assert_eq!(h.tile, want, "{m}x{n}x{k}");
        h.tile
    });
    let _ = std::fs::remove_dir_all(&dir);
}

/// The instructions of a module, its comments left out (a header comment
/// names what the kernel does not use).
fn instructions(ptx: &str) -> String {
    ptx.lines().filter(|l| !l.trim_start().starts_with("//")).collect::<Vec<_>>().join("\n")
}

/// A profile for `sm` with `sms` SMs.
fn profile_for(tag: &str, sm: &str, sms: u32) -> std::path::PathBuf {
    let dir = pinned::pinned_scratch(tag, sm);
    let text = pinned::profile_text(sm).replace("SM_COUNT=66", &format!("SM_COUNT={sms}"));
    std::fs::write(dir.join(".ysu_hw_profile"), text).unwrap();
    dir
}

/// Below sm_80 there is no `cp.async`: each copy is a load and a shared
/// store (`f32_simt_gemm.rs`, `sync_copies`). Same arithmetic, so the same
/// 43 shapes give the same bits. Built for sm_75 (a Tesla T4: 40 SMs, 64 KB
/// of shared memory a block, so the large tile only at two stages, 49.7 KB)
/// and run here, where the driver compiles sm_75 PTX for this card.
#[test]
fn built_for_sm_75_each_element_is_the_same_fma_chain() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver - no sm_75 FP32 GEMM was executed.");
        return;
    };
    let dir = profile_for("f32_gemm_run_sm75", "7.5", 40);
    let seen = std::cell::RefCell::new(std::collections::BTreeSet::new());
    each_element_exact(&ctx, &dir, |_, ptx, m, n, k| {
        assert!(ptx.contains(".target sm_75"), "{m}x{n}x{k}: not built for sm_75");
        seen.borrow_mut().insert(header(ptx).tile);
        assert!(!instructions(ptx).contains("cp.async"), "{m}x{n}x{k}: cp.async on sm_75");
        let h = header(ptx);
        assert!(h.tile != "large" || ptx.contains("2 stages of plain loads"), "{m}x{n}x{k}: the 3-stage 74 KB tile on a 64 KB card");
        assert!(h.smem <= 64 * 1024, "{m}x{n}x{k}: {} bytes of shared memory", h.smem);
        h.tile
    });
    // Every tile Turing can choose ran here, each element checked.
    let want = ["large", "mid", "small", "thin32x32", "thin32x64", "skinny16x128", "skinny8x256", "gemv32", "gemv128",
                "slab4x32", "slab8x16", "slab16x32"];
    let seen = seen.into_inner();
    assert!(want.iter().all(|t| seen.contains(*t)), "sm_75 tiles run: {seen:?}, want all of {want:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A `ptxas` that still assembles sm_60..sm_72 (CUDA 13's does not):
/// Y_PTXAS_OLD_ARCH, a repository venv's Triton, or one on the usual paths.
fn old_arch_ptxas() -> Option<std::path::PathBuf> {
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Some(p) = std::env::var_os("Y_PTXAS_OLD_ARCH") {
        candidates.push(p.into());
    }
    for venv in ["venv", ".venv"] {
        if let Ok(dir) = std::fs::read_dir(pinned::repo().join(venv).join("lib")) {
            for py in dir.flatten() {
                candidates.push(py.path().join("site-packages/triton/backends/nvidia/bin/ptxas"));
            }
        }
    }
    candidates.extend(["ptxas".into(), "/opt/cuda/bin/ptxas".into(), "/usr/local/cuda/bin/ptxas".into()]);
    let probe = std::env::temp_dir().join(format!("y_f32_old_ptxas_{}.ptx", std::process::id()));
    std::fs::write(&probe, ".version 6.0\n.target sm_61\n.address_size 64\n.visible .entry p()\n{\n    ret;\n}\n").unwrap();
    let found = candidates.into_iter().find(|t| {
        Command::new(t).args(["-arch=sm_61", "-o", "/dev/null"]).arg(&probe).output().is_ok_and(|o| o.status.success())
    });
    let _ = std::fs::remove_file(&probe);
    found
}

/// Pascal, Volta and Turing: no `cp.async`, the tile within the target's
/// per-block shared memory (48, 96 and 64 KB), and `ptxas` for that target
/// assembles every kernel without spilling.
///
/// Until 2026-11-15 a K tail or a ragged M guarded every copy of the K loop,
/// and below sm_80 a guarded copy (a predicated load and a zero fill) spilled
/// 280-688 bytes at 255 registers; on a Tesla T4 1023 x 1025 x 1027 ran at
/// 0.22x cuBLAS. That was put down to N = 1 mod 4, because every such shape
/// here also had a K tail; so one shape per edge: N odd alone (1024 x 1025 x
/// 1024, which never spilled), K alone (1024 x 1024 x 1027) and M alone
/// (1023 x 1024 x 1024), and row remainder sets (67 x 37 x 1035, 400 x 2049 x
/// 1025, and at two or three K tiles, where `ptxas` unrolls the K loop, 67 x
/// 37 x 35 and 400 x 2049 x 40: those spilled 4-52 bytes until the loop's
/// loads went first and its stores last).
#[test]
fn below_sm_80_the_gemm_uses_plain_loads_fits_the_block_and_assembles() {
    let shapes = [(4096, 4096, 4096), (2049, 4100, 24), (1023, 1025, 1027), (64, 64, 16), (2048, 12803, 64),
                  (1, 8193, 40), (2048, 4097, 40), (1024, 1025, 1024), (1024, 1024, 1027), (1023, 1024, 1024),
                  (67, 37, 1035), (400, 2049, 1025), (67, 37, 35), (400, 2049, 40),
                  (3, 1001, 37), (8, 896, 896), (16, 777, 4097)];
    let new = ptxas::ptxas();
    let old = old_arch_ptxas();
    for (sm, sms, limit) in [("7.5", 40u32, 64 * 1024u32), ("7.0", 80, 96 * 1024), ("6.1", 6, 48 * 1024)] {
        let dir = profile_for(&format!("f32_gemm_old_{}", sm.replace('.', "")), sm, sms);
        let arch = format!("sm_{}", sm.replace('.', ""));
        let tool = if sm == "7.5" { new.clone() } else { old.clone() };
        if tool.is_none() {
            eprintln!("SKIP: no ptxas that targets {arch}; compiled only");
        }
        for (m, n, k) in shapes {
            let name = format!("g{m}_{n}_{k}");
            let ptx = compile(&dir, &name, m, n, k, "--emit-ptx").unwrap_or_else(|log| panic!("{arch} {m}x{n}x{k}: {log}"));
            assert!(ptx.contains(&format!(".target {arch}")), "{arch} {m}x{n}x{k}");
            assert!(!instructions(&ptx).contains("cp.async"), "{arch} {m}x{n}x{k}: cp.async");
            let h = header(&ptx);
            let gemv = h.tile.starts_with("gemv");
            assert!(gemv || ptx.contains("stages of plain loads (no cp.async below sm_80)"), "{arch} {m}x{n}x{k}: header");
            assert!(h.smem <= limit, "{arch} {m}x{n}x{k}: tile {} takes {} bytes, the block holds {limit}", h.tile, h.smem);
            // Turing's 64 KB holds one 3-stage mid tile and two 2-stage ones,
            // and the large tile at 2 stages only, so there both run 2;
            // elsewhere 3 (`f32_gemm_turing_stages`, `f32_gemm_tile`). The
            // few-column tiles (Turing only) stage two slices.
            let slab = h.tile.starts_with("slab");
            assert!(!slab || sm == "7.5", "{arch} {m}x{n}x{k}: the few-column kernel off Turing");
            let stages = if sm == "7.5" && (h.tile == "mid" || h.tile == "large" || slab) { 2 } else { 3 };
            assert!(gemv || ptx.contains(&format!("{stages} stages of plain loads")), "{arch} {m}x{n}x{k}: tile {} not {stages} stages", h.tile);
            if let Some(tool) = &tool {
                let path = dir.join(format!("{name}.ptx"));
                std::fs::write(&path, &ptx).unwrap();
                let out = Command::new(tool).args([format!("-arch={arch}").as_str(), "-v", "-o", "/dev/null"]).arg(&path).output().unwrap();
                let log = String::from_utf8_lossy(&out.stderr);
                assert!(out.status.success(), "{arch} {m}x{n}x{k}: ptxas failed:\n{log}");
                let spills = log.lines().filter(|l| l.contains("bytes spill")).any(|l| !l.contains(" 0 bytes spill stores, 0 bytes spill loads"));
                assert!(!spills, "{arch} {m}x{n}x{k}: spills\n{log}");
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    // The control: at sm_80 the same 4096^3 GEMM is the large tile on cp.async.
    let dir = profile_for("f32_gemm_old_control", "8.0", 66);
    let ptx = compile(&dir, "c4096", 4096, 4096, 4096, "--emit-ptx").unwrap();
    assert!(instructions(&ptx).contains("cp.async") && header(&ptx).tile == "large");
    let _ = std::fs::remove_dir_all(&dir);
}

/// On Turing the mid tile (two stages) is taken wherever the small tile
/// would put two CTAs on some SM (from 21 mid tiles on 40 SMs): on a Tesla T4 one
/// wave of mid tiles beats the small ones doubled up (512^3, 32 mid tiles on
/// 40 SMs: 0.68x cuBLAS on the small tile, 0.92x on the mid;
/// `f32_gemm_tile`). Where one small tile an SM covers C, or the mid tile
/// would be half empty, the small tile stays - unless it leaves SMs idle:
/// then a 32-row tile with more tiles that still fit a wave (128 x 512 x
/// 2048: 0.44x on the small tile, 0.54x on 32 x 64; `f32_gemm_turing_thin`).
/// Up to 16 rows, where no skinny tile is taken, the few-column tiles run
/// (8 x 896 x 896: 0.37x on the small tile, 0.73x on 8 x 16;
/// `f32_gemm_turing_slab`; 8 x 16 where even 16 x 32 tiles are few), up to
/// 4 rows until the GEMV's blocks would fill the SMs, and from 32 rows 16 x
/// 32 tiles where 32 x 32 ones fill half a wave or less. Sm_70 (Volta) on the same SM count, and this
/// machine's sm_89, keep the old rule.
#[test]
fn on_turing_the_tile_is_the_one_that_fills_a_wave() {
    let dir = profile_for("f32_gemm_turing_rule", "7.5", 40);
    // A T4's shared memory an SM, which the stream-K plan reads.
    let profile = dir.join(".ysu_hw_profile");
    std::fs::write(&profile, std::fs::read_to_string(&profile).unwrap() + "MAX_SMEM_PER_SM_BYTES=65536\n").unwrap();
    let cases = [
        // (shape, mid tiles, small tiles) -> tile
        ((512, 512, 512), "mid"),     // 32, 64
        ((500, 500, 500), "mid"),     // 32, 64, ragged
        ((2048, 128, 512), "mid"),    // 32, 64
        ((448, 512, 512), "mid"),     // 28, 56
        ((384, 512, 1024), "mid"),    // 24, 48: 60%
        ((448, 384, 512), "mid"),     // 21, 42: just over half a wave
        ((192, 896, 896), "mid"),     // 21, 42
        ((320, 512, 1024), "small"),  // 20, 40: one small tile an SM
        ((512, 256, 512), "small"),   // 16, 32: one small tile an SM
        ((2048, 64, 512), "small"),   // 32, 32: the mid tile half empty
        ((256, 512, 4096), "small"),  // 16, 32
        ((768, 768, 768), "mid"),     // 72: the rule before
        // Few tiles: the 32-row tile with the most tiles that fit a wave.
        ((128, 512, 2048), "thin32x64"), // 16 small tiles, 32 thin
        ((100, 500, 3000), "thin32x64"), // ragged, no remainder sets
        ((128, 256, 4096), "thin32x32"), // 8 small, 16 thin, 32 of 32x32
        ((384, 256, 2048), "small"),     // 24 small, 48 thin: over a wave
        ((128, 1024, 1024), "small"),    // 32 small, 64 thin
        ((32, 1024, 4096), "thin32x32"), // 16 small or 32x64, 32 of 32x32
        ((32, 1000, 4096), "small"),     // 32x32 needs N a multiple of 32,
        ((48, 1024, 4096), "thin32x64"), // and M
        // ... and where 32 x 32 tiles fill half a wave or less, 16 x 32 ones.
        ((32, 640, 4096), "slab16x32"),  // 20 of 32 x 32
        ((64, 128, 8192), "slab16x32"),  // 8
        ((32, 672, 4096), "thin32x32"),  // 21
        // 5 to 16 rows: a tile no taller than M.
        ((8, 4096, 4096), "skinny8x256"),
        ((6, 30522, 768), "skinny8x256"),  // N even: 8-byte B copies
        ((7, 50257, 768), "skinny16x128"), // N odd: 4-byte copies, 128 threads
        ((16, 4096, 4096), "skinny16x128"),
        ((17, 4096, 4096), "mid"),
        ((6, 4096, 11008), "skinny8x256"), // 16 of them: 40%
        // Where no skinny tile is taken, 5 to 16 rows run the few-column
        // tiles (`f32_gemm_turing_slab`), whatever the fragment tile was.
        ((6, 1024, 1024), "slab8x16"),   // 4 of 8 x 256: under 40% of the SMs
        ((8, 896, 896), "slab8x16"),
        ((6, 3072, 3072), "slab8x16"),   // the mid tile before
        ((16, 896, 4864), "slab16x32"),  // 28 of 16 x 32
        ((13, 999, 1001), "slab16x32"),  // 32
        ((16, 320, 2048), "slab8x16"),   // 10 of 16 x 32: a quarter of the SMs
        ((16, 256, 4096), "slab8x16"),
        ((16, 352, 2048), "slab16x32"),  // 11
        ((12, 2048, 2048), "skinny16x128"), // 16 of 16 x 128
        // One to 4 rows: the 4 x 32 tile while its CTAs number under
        // 0.6 (M + 1) an SM (N up to 1504 at one row, 3808 at four)...
        ((1, 1024, 1024), "slab4x32"),
        ((4, 896, 4864), "slab4x32"),
        ((1, 1504, 1024), "slab4x32"),   // 47 CTAs
        ((4, 3072, 1024), "slab4x32"),   // 96
        // ... then the GEMV, 128-thread blocks where they reach 75% of the
        // SMs, else 32.
        ((1, 1536, 1024), "gemv32"),     // 48
        ((1, 2048, 2048), "gemv32"),
        ((2, 3072, 768), "gemv32"),
        ((1, 50257, 768), "gemv128"),
        ((4, 4096, 4096), "gemv128"),
        ((4, 4096, 1024), "gemv128"),    // 128
        // ... unless the 8 x 256 tile has three waves of CTAs (N even).
        ((1, 24576, 768), "gemv128"),     // 96 of them
        ((1, 30522, 768), "skinny8x256"), // 120
        ((4, 30522, 768), "skinny8x256"),
    ];
    for (i, ((m, n, k), want)) in cases.iter().enumerate() {
        let ptx = compile(&dir, &format!("t{i}"), *m, *n, *k, "--emit-ptx").unwrap_or_else(|log| panic!("{m}x{n}x{k}: {log}"));
        let h = header(&ptx);
        assert_eq!(h.tile, *want, "{m}x{n}x{k} at sm_75 on 40 SMs");
        if h.tile.starts_with("gemv") {
            assert!(ptx.contains("Plain loads, B by 4-byte copies") && h.smem == 0, "{m}x{n}x{k}: the GEMV header");
            continue;
        }
        if h.tile.starts_with("slab") {
            assert!(ptx.contains("2 stages of plain loads") && ptx.contains("Remainder: none; none."), "{m}x{n}x{k}: the few-column header");
            continue;
        }
        let stages = if h.tile == "mid" { 2 } else { 3 };
        assert!(ptx.contains(&format!("{stages} stages of plain loads")), "{m}x{n}x{k}: tile {} not at {stages} stages", h.tile);
        if h.tile.starts_with("thin") {
            assert!(ptx.contains("Remainder: none; none."), "{m}x{n}x{k}: a thin tile with remainder sets");
        }
    }
    // With `W`, below 90% of a wave of mid tiles: the small tile split in
    // order where it has up to 1.3 tiles an SM, or where the mid tile's
    // launch, remainder tiles counted, exceeds one CTA an SM
    // (`f32_gemm_turing_split_small`); else the mid tile, `W` unused.
    let w_cases = [
        ((384, 512, 1024), Some("small")), // 48 small tiles
        ((832, 256, 1024), Some("small")), // 52
        ((576, 384, 1024), None),          // 54
        ((448, 512, 512), None),           // 56
        ((512, 512, 512), None),           // 64
        ((448, 576, 512), Some("small")),  // 63, but 28 + 14 mid CTAs
        ((768, 768, 768), Some("mid")),    // 72 mid tiles: the rule before
        // A K tail or a ragged M would guard a split's copies below sm_80.
        ((1000, 1000, 1000), None),
        ((768, 768, 770), None),
        // The 8- and 16-row tiles are not split, nor the few-column ones.
        ((8, 11008, 4096), None),
        ((8, 896, 896), None),
    ];
    for (i, ((m, n, k), want)) in w_cases.iter().enumerate() {
        let name = format!("w{i}");
        let ptx = compile_text(&dir, &name, &source_w(&name, *m, *n, *k), "--emit-ptx").unwrap_or_else(|log| panic!("{m}x{n}x{k}: {log}"));
        let got = sk_header(&ptx).map(|h| h.tile);
        assert_eq!(got.as_deref(), *want, "{m}x{n}x{k} with W at sm_75 on 40 SMs: the tile split, if any");
        if want.is_none() {
            let one_wave = if *m > 16 { "mid" } else if *n < 4096 { "slab8x16" } else { "skinny8x256" };
            assert_eq!(header(&ptx).tile, one_wave, "{m}x{n}x{k} with W: the tile");
            if one_wave.starts_with("slab") {
                assert!(ptx.contains("Workspace W: not used: the few-column kernel does not split K."), "{m}x{n}x{k} with W: the reason W is unused");
                continue;
            }
            let guarded = *m > 16 && (m % 64 != 0 || k % 16 != 0);
            assert_eq!(ptx.contains("a split's copies would be guarded"), guarded, "{m}x{n}x{k} with W: the reason W is unused");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    for (sm, sms) in [("7.0", 40u32), ("8.9", 66)] {
        let dir = profile_for(&format!("f32_gemm_turing_rule_ctl{}", sm.replace('.', "")), sm, sms);
        let ptx = compile(&dir, "c512", 512, 512, 512, "--emit-ptx").unwrap_or_else(|log| panic!("sm {sm}: {log}"));
        assert_eq!(header(&ptx).tile, "small", "512^3 at sm {sm} on {sms} SMs");
        let ptx = compile(&dir, "c128", 128, 512, 2048, "--emit-ptx").unwrap_or_else(|log| panic!("sm {sm}: {log}"));
        assert_eq!(header(&ptx).tile, "small", "128 x 512 x 2048 at sm {sm} on {sms} SMs");
        let ptx = compile(&dir, "c8", 8, 896, 896, "--emit-ptx").unwrap_or_else(|log| panic!("sm {sm}: {log}"));
        assert_eq!(header(&ptx).tile, "small", "8 x 896 x 896 at sm {sm} on {sms} SMs");
        let ptx = compile(&dir, "c4", 4, 896, 4864, "--emit-ptx").unwrap_or_else(|log| panic!("sm {sm}: {log}"));
        assert_eq!(header(&ptx).tile, "small", "4 x 896 x 4864 at sm {sm} on {sms} SMs");
        let _ = std::fs::remove_dir_all(&dir);
    }
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
        (1025, 1024, 40),
        (2049, 4100, 24),
        (65, 70, 20),
        (256, 12802, 80),
        (2048, 12803, 64),
        (256, 12802, 83),
        // C's v2 and v4 stores at N odd, both residues, a row strip.
        (2048, 4097, 40),
        (2060, 4097, 24),
    ];
    // Built for sm_75 on 40 SMs: the few-column tiles, whose copies past M
    // and N read row M-1 and column N-1, over two or more K slices (the 3x1
    // grid reuses both stages across tiles) and a K tail; N = 1.
    let dir75 = profile_for("f32_gemm_sanitized75", "7.5", 40);
    let shapes75 = [(3usize, 100usize, 200usize), (7, 50, 150), (11, 333, 77), (13, 1, 200), (4, 31, 65)];
    let runs = shapes.iter().map(|s| (&dir, *s)).chain(shapes75.iter().map(|s| (&dir75, *s)));
    for (si, (dir, (m, n, k))) in runs.enumerate() {
        let name = format!("san{si}");
        let ptx = compile(dir, &name, m as u32, n as u32, k as u32, "--emit-ptx").unwrap();
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
    let _ = std::fs::remove_dir_all(&dir75);
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
    // memcheck: every read inside A, B and W, every write inside C and W.
    // racecheck: no shared-memory hazard, including a stage reused by the
    // next tile or the next stream-K segment.
    for (check, summary) in [("memcheck", "ERROR SUMMARY: 0 errors"), ("racecheck", "RACECHECK SUMMARY: 0 hazards displayed")] {
        for (child, mark, runs) in [("child_sanitized_runs", "SANITIZED_RUN", 24), ("child_sanitized_streamk_runs", "SANITIZED_SK_RUN", STREAMK_SANITIZED.len())] {
            let out = Command::new(tool)
                .args(["--tool", check, "--print-limit", "0"])
                .arg(std::env::current_exe().unwrap())
                .args([child, "--exact", "--nocapture", "--test-threads=1"])
                .env("Y_F32_GEMM_SANITIZE", "1")
                .output()
                .unwrap();
            let text = String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
            // Every shape ran (a child that died early prints a clean summary
            // for the launches it made).
            assert_eq!(text.matches(mark).count(), runs, "{check} {child}: not every shape ran:\n{text}");
            assert!(out.status.success() && text.contains(summary), "{check} {child} found something:\n{text}");
        }
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

// ---------------------------------------------------------------------------
// The workspace form: K split in order where the plan takes it.
// ---------------------------------------------------------------------------

fn source_w(name: &str, m: u32, n: u32, k: u32) -> String {
    format!("@tile({m}, {n}, {k})\nkernel {name}(A: GlobalMemory<F32>, B: GlobalMemory<F32>, C: GlobalMemory<F32>, W: GlobalMemory<U64>) {{\n}}\n\nfn main() {{\n}}\n")
}

/// A scratch directory whose profile has `sms` SMs and 100 KB of shared
/// memory per SM. The pinned profile states no shared memory, and the
/// compiler's default of 48 KB holds one mid CTA where the card holds two,
/// which is a different plan.
fn profile_dir(tag: &str, sms: u32) -> std::path::PathBuf {
    let dir = pinned::pinned_scratch(tag, pinned::SM_PINNED);
    let text = pinned::profile_text(pinned::SM_PINNED).replace("SM_COUNT=66", &format!("SM_COUNT={sms}")) + "MAX_SMEM_PER_SM_BYTES=102400\n";
    std::fs::write(dir.join(".ysu_hw_profile"), text).unwrap();
    dir
}

/// What a stream-K kernel's header states, or None for one tile per CTA.
struct SkHeader {
    tile: String,
    block: u32,
    p: u32,
    tiles: u32,
    w_words: usize,
    smem: u32,
}

fn sk_header(ptx: &str) -> Option<SkHeader> {
    let line = |pat: &str| ptx.lines().find(|l| l.contains(pat)).unwrap_or_else(|| panic!("no `{}` in the PTX", pat)).to_string();
    let sk = ptx.lines().find(|l| l.contains("// Stream-K, in order: "))?;
    let tiles: u32 = sk.split("in order: ").nth(1).unwrap().split(' ').next().unwrap().parse().unwrap();
    let h = line("[Y F32 SIMT GEMM]");
    let tile = h.split("tile `").nth(1).unwrap().split('`').next().unwrap().to_string();
    let launch = line("Launch: block (");
    let block: u32 = launch.split("block (").nth(1).unwrap().split(',').next().unwrap().parse().unwrap();
    let p: u32 = launch.split("Grid (").nth(1).unwrap().split(',').next().unwrap().parse().unwrap();
    let w_words: usize = line("Workspace W: ").split("Workspace W: ").nth(1).unwrap().split(' ').next().unwrap().parse().unwrap();
    let smem: u32 = line("Dynamic shared memory required:").split("required: ").nth(1).unwrap().split(' ').next().unwrap().parse().unwrap();
    Some(SkHeader { tile, block, p, tiles, w_words, smem })
}

/// The plan on this machine's numbers (66 SMs, 100 KB of shared memory an
/// SM): stream-K where one tile per CTA leaves SMs idle, on the CTAs that
/// fit at once; one tile per CTA elsewhere, as the very kernel the
/// three-parameter form compiles to (its SASS, where cuobjdump is found).
#[test]
fn the_workspace_form_splits_k_where_the_model_says_and_is_otherwise_the_same_kernel() {
    let dir = profile_dir("f32_gemm_sk_plan", 66);
    let cases: [((u32, u32, u32), Option<(&str, u32)>); 17] = [
        // 72 mid tiles: one tile per CTA runs six SMs at two (0.82x cuBLAS
        // against 1.21x split).
        ((768, 768, 768), Some(("mid", 66))),
        // 288 tiles: two whole waves on 132 CTAs, the rest split.
        ((1536, 1536, 1536), Some(("mid", 132))),
        ((1500, 1500, 1500), Some(("mid", 132))),
        ((3072, 3072, 3072), Some(("large", 66))),
        ((512, 768, 768), Some(("small", 66))),
        ((1024, 1028, 1024), Some(("mid", 132))),
        // N odd on the mid tile, where the model gives at least 1.25x: the
        // plain path's one-column remainder tiles are a second wave.
        ((1023, 1025, 1027), Some(("mid", 132))),
        ((1025, 1025, 1025), Some(("mid", 132))),
        // 128 tiles in 132 slots; 7.8 waves of the large tile (3%, under
        // the 4% bar); N % 4 != 0, where the plain path's remainder tiles
        // let B's rows take their own widths; 64 tiles for 66 SMs.
        ((1024, 1024, 1024), None),
        ((4096, 4096, 4096), None),
        ((1024, 1024, 4096), None),
        ((1024, 1026, 1024), None),
        ((512, 512, 512), None),
        // N odd under the 1.25x bar (1.08x and 1.18x by the model; split,
        // they ran 1.13x and 1.02x the plain time), and N odd on the large
        // tile, whose plain path stores C in aligned pieces.
        ((768, 1025, 768), None),
        ((2048, 1025, 1024), None),
        ((4095, 4097, 512), None),
        ((1024, 50257, 768), None),
    ];
    let ptxas = ptxas::ptxas();
    let cuobjdump = ptxas.as_ref().map(|p| p.with_file_name("cuobjdump")).filter(|c| c.exists());
    let sass = |ptx: &str, name: &str| -> Option<Vec<String>> {
        let (ptxas, cuobjdump) = (ptxas.as_ref()?, cuobjdump.as_ref()?);
        let (pp, cb) = (dir.join(format!("{name}.ptx")), dir.join(format!("{name}.cubin")));
        std::fs::write(&pp, ptx).unwrap();
        let out = Command::new(ptxas).args(["-arch=sm_89", "-v", "-o"]).arg(&cb).arg(&pp).output().unwrap();
        let log = String::from_utf8_lossy(&out.stderr).to_string();
        assert!(out.status.success(), "{name}: ptxas refused it:\n{log}");
        assert!(log.contains("0 bytes spill stores, 0 bytes spill loads"), "{name} spills:\n{log}");
        let out = Command::new(cuobjdump).arg("-sass").arg(&cb).output().unwrap();
        // Instructions only, without their encodings' line numbers.
        Some(String::from_utf8_lossy(&out.stdout).lines().filter(|l| l.trim_start().starts_with("/*") && l.contains(';')).map(|l| l.split("/* 0x").next().unwrap().trim().to_string()).collect())
    };
    for ((m, n, k), want) in cases {
        let ptx = compile_text(&dir, "g", &source_w("g", m, n, k), "--emit-ptx").unwrap_or_else(|log| panic!("{m}x{n}x{k}:\n{log}"));
        match (sk_header(&ptx), want) {
            (Some(h), Some((tile, p))) => {
                assert_eq!((h.tile.as_str(), h.p), (tile, p), "{m}x{n}x{k}");
                assert_eq!(h.w_words, h.tiles as usize, "{m}x{n}x{k}: one W word a tile");
                // The handoff's instructions, and still nothing but fma on
                // the values.
                // A continuation loads C as float4, or at N odd a float at a
                // time (a row is only 4-byte aligned).
                let load = if n % 4 == 0 { "ld.global.cg.v4.f32" } else { "ld.global.cg.f32 " };
                for needle in ["ld.acquire.gpu.global.u64", "st.release.gpu.global.u64", "%gridid", load] {
                    assert!(ptx.contains(needle), "{m}x{n}x{k}: no `{needle}`");
                }
                if n % 4 != 0 {
                    assert!(!ptx.contains("ld.global.cg.v4.f32") && !ptx.contains("st.global.v4.f32"), "{m}x{n}x{k}: a float4 access of C at N odd");
                }
                assert!(!ptx.contains("add.f32") && !ptx.contains("mul.f32") && !ptx.contains("add.rn.f32"), "{m}x{n}x{k}");
                let _ = sass(&ptx, "sk");
            }
            (None, None) => {
                assert!(ptx.contains("// Workspace W: not used at this shape"), "{m}x{n}x{k}");
                let plain = compile(&dir, "g", m, n, k, "--emit-ptx").unwrap_or_else(|log| panic!("{log}"));
                if let (Some(a), Some(b)) = (sass(&ptx, "w"), sass(&plain, "p")) {
                    assert!(a == b && !a.is_empty(), "{m}x{n}x{k}: with W unused the kernel is not the three-parameter one");
                }
            }
            (got, _) => panic!("{m}x{n}x{k}: wanted {want:?}, got {}", got.map_or("one tile per CTA".to_string(), |h| format!("{} on {}", h.tile, h.p))),
        }
    }
    if ptxas.is_none() || cuobjdump.is_none() {
        eprintln!("SKIP (assembly): no ptxas or cuobjdump - not assembled, and the unused-W kernels' SASS was not compared.");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// (M, N, K, SMs of the profile, tile): small shapes made to take stream-K by
/// a profile of few SMs, so every element can be checked. Ragged M and N,
/// a K tail, one, two and many K tiles, each tile. The schedule depends on
/// the GRID, not the profile, so each runs on several.
const STREAMK_SHAPES: [(u32, u32, u32, u32, &str); 20] = [
    (241, 140, 80, 5, "mid"),
    (241, 140, 200, 5, "mid"),
    (144, 388, 7, 8, "mid"),
    (174, 84, 17, 2, "mid"),
    (130, 444, 40, 4, "mid"),
    (215, 180, 32, 6, "mid"),
    (260, 72, 20, 6, "small"),
    (258, 80, 15, 6, "small"),
    (180, 100, 48, 4, "small"),
    (35, 220, 40, 3, "small"),
    (296, 928, 33, 5, "large"),
    (296, 928, 300, 5, "large"),
    (590, 812, 15, 8, "large"),
    (683, 1036, 48, 8, "large"),
    (384, 1188, 40, 6, "large"),
    // N odd (the mid and small tiles only): a continuation loads C a float
    // at a time. 1 and 3 mod 4, a K tail with 2 and 63 K tiles, ragged M.
    (215, 181, 17, 6, "mid"),
    (215, 183, 256, 6, "mid"),
    (197, 181, 1000, 6, "mid"),
    (35, 221, 33, 3, "small"),
    (60, 221, 300, 3, "small"),
];

#[test]
fn the_workspace_form_is_the_in_order_fma_chain_on_every_grid() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver - no stream-K GEMM was executed.");
        return;
    };
    let mut cases: Vec<(u32, u32, u32, u32, &str)> = STREAMK_SHAPES.to_vec();
    // And at this machine's size, where the grid is 66 CTAs.
    cases.push((768, 768, 768, 66, "mid"));
    for (si, &(m, n, k, sms, tile)) in cases.iter().enumerate() {
        let dir = profile_dir(&format!("f32_gemm_sk_run{si}"), sms);
        let name = format!("sk{si}");
        let ptx = compile_text(&dir, &name, &source_w(&name, m, n, k), "--emit-ptx").unwrap_or_else(|log| panic!("{log}"));
        let h = sk_header(&ptx).unwrap_or_else(|| panic!("{m}x{n}x{k} on {sms} SMs: no stream-K"));
        assert_eq!(h.tile, tile, "{m}x{n}x{k}");
        let (m, n, k) = (m as usize, n as usize, k as usize);
        let module = ctx.load_ptx(&ptx, &name).unwrap_or_else(|e| panic!("{m}x{n}x{k} did not load: {e}"));
        module.set_max_dynamic_smem(h.smem).unwrap();
        let a = operand(m * k, 211 + si as u64);
        let b = operand(k * n, 307 + si as u64);
        let (da, db, dc) = (ctx.alloc(m * k * 4).unwrap(), ctx.alloc(k * n * 4).unwrap(), ctx.alloc(m * n * 4).unwrap());
        let dw = ctx.alloc(h.w_words * 8).unwrap();
        ctx.memcpy_htod_at(&da, 0, &as_bytes(&a)).unwrap();
        ctx.memcpy_htod_at(&db, 0, &as_bytes(&b)).unwrap();
        // W holds garbage and is never cleared: every launch's words are
        // tagged with its own grid id.
        ctx.memset_u8(&dw, 0xa5).unwrap();
        let rows: Vec<usize> = (0..m).filter(|&i| m <= 700 || i < 64 || i >= m - 64 || i % 7 == 0).collect();
        let want: Vec<Vec<f32>> = rows.iter().map(|&i| (0..n).map(|j| oracle(&a, &b, n, k, i, j)).collect()).collect();
        // The header's grid, twice (the second launch finds the first's
        // words in W); one CTA (every tile whole, in turn); two and three
        // (tiles cut between them); more CTAs than tiles (each tile cut into
        // several pieces, each waiting on the last).
        for grid in [h.p, h.p, 1, 2, 3, h.p + 5, 3 * h.tiles + 1] {
            ctx.memset_u8(&dc, 0xff).unwrap(); // NaN everywhere
            ctx.launch(&module, (grid, 1, 1), (h.block, 1, 1), h.smem, &[da.device_ptr(), db.device_ptr(), dc.device_ptr(), dw.device_ptr()]).unwrap();
            ctx.synchronize().unwrap_or_else(|e| panic!("{m}x{n}x{k} grid {grid}: {e}"));
            let mut raw = vec![0u8; m * n * 4];
            ctx.memcpy_dtoh_at(&mut raw, &dc, 0).unwrap();
            for (ri, &i) in rows.iter().enumerate() {
                for j in 0..n {
                    let g = f32::from_le_bytes(raw[(i * n + j) * 4..(i * n + j) * 4 + 4].try_into().unwrap());
                    let w = want[ri][j];
                    assert!(
                        g.to_bits() == w.to_bits(),
                        "{m}x{n}x{k} ({tile}, stream-K) grid {grid}: C[{i}][{j}] = {g:e} ({:#010x}), the in-order fma chain gives {w:e} ({:#010x})",
                        g.to_bits(),
                        w.to_bits()
                    );
                }
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// The workspace form built for sm_75 (a Tesla T4: 40 SMs, 64 KB an SM),
/// where each copy is a load and a shared store and a segment's loop loads
/// the next tile first and stores it after the FMAs (`prefetch`): stream-K at
/// every one of these shapes on 40 SMs, two with a K tail, run here (the
/// driver compiles sm_75 PTX for this card) on the grids the sm_80 test
/// uses: the small tile split over 40 CTAs (448 x 384 x 512, 42 tiles, 704
/// x 256 x 1024, 44, and 448 x 576 x 512, each taken over the mid tile with
/// `W`)
/// and the mid tile (768^3, 512 x 896 x 896). At a K tail or a ragged M a
/// split's copies would be guarded here, so there `W` is not used (the
/// Turing rule test). On a T4
/// this form took 384 x 512 x 1024 from 0.59x cuBLAS to 0.82x on the small
/// tile (docs/benchmarks/f32_gemm_sm75_guards_2026-11-15/t4_w/); 512^3 is
/// now the mid tile, one wave (0.92x), where `W` is not used.
#[test]
fn built_for_sm_75_the_workspace_form_is_the_in_order_fma_chain() {
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver - no sm_75 stream-K GEMM was executed.");
        return;
    };
    let cases: [(u32, u32, u32); 5] = [(448, 384, 512), (704, 256, 1024), (448, 576, 512), (768, 768, 768), (512, 896, 896)];
    for (si, &(m, n, k)) in cases.iter().enumerate() {
        let dir = pinned::pinned_scratch(&format!("f32_gemm_sk75_run{si}"), "7.5");
        let text = pinned::profile_text("7.5").replace("SM_COUNT=66", "SM_COUNT=40") + "MAX_SMEM_PER_SM_BYTES=65536\n";
        std::fs::write(dir.join(".ysu_hw_profile"), text).unwrap();
        let name = format!("sk75_{si}");
        let ptx = compile_text(&dir, &name, &source_w(&name, m, n, k), "--emit-ptx").unwrap_or_else(|log| panic!("{log}"));
        assert!(ptx.contains(".target sm_75") && !instructions(&ptx).contains("cp.async"), "{m}x{n}x{k}: not the sm_75 plain-load build");
        let h = sk_header(&ptx).unwrap_or_else(|| panic!("{m}x{n}x{k} on 40 SMs at sm_75: no stream-K"));
        assert_eq!(h.tile, if si < 3 { "small" } else { "mid" }, "{m}x{n}x{k} at sm_75 on 40 SMs");
        let (m, n, k) = (m as usize, n as usize, k as usize);
        let module = ctx.load_ptx(&ptx, &name).unwrap_or_else(|e| panic!("{m}x{n}x{k} did not load: {e}"));
        module.set_max_dynamic_smem(h.smem).unwrap();
        let a = operand(m * k, 911 + si as u64);
        let b = operand(k * n, 977 + si as u64);
        let (da, db, dc) = (ctx.alloc(m * k * 4).unwrap(), ctx.alloc(k * n * 4).unwrap(), ctx.alloc(m * n * 4).unwrap());
        let dw = ctx.alloc(h.w_words * 8).unwrap();
        ctx.memcpy_htod_at(&da, 0, &as_bytes(&a)).unwrap();
        ctx.memcpy_htod_at(&db, 0, &as_bytes(&b)).unwrap();
        ctx.memset_u8(&dw, 0xa5).unwrap();
        let want: Vec<Vec<f32>> = (0..m).map(|i| (0..n).map(|j| oracle(&a, &b, n, k, i, j)).collect()).collect();
        for grid in [h.p, h.p, 1, 2, 3, h.p + 5, 3 * h.tiles + 1] {
            ctx.memset_u8(&dc, 0xff).unwrap();
            ctx.launch(&module, (grid, 1, 1), (h.block, 1, 1), h.smem, &[da.device_ptr(), db.device_ptr(), dc.device_ptr(), dw.device_ptr()]).unwrap();
            ctx.synchronize().unwrap_or_else(|e| panic!("{m}x{n}x{k} grid {grid}: {e}"));
            let mut raw = vec![0u8; m * n * 4];
            ctx.memcpy_dtoh_at(&mut raw, &dc, 0).unwrap();
            for i in 0..m {
                for j in 0..n {
                    let g = f32::from_le_bytes(raw[(i * n + j) * 4..(i * n + j) * 4 + 4].try_into().unwrap());
                    let w = want[i][j];
                    assert!(g.to_bits() == w.to_bits(), "{m}x{n}x{k} (sm_75 stream-K) grid {grid}: C[{i}][{j}] = {g:e}, the in-order fma chain gives {w:e}");
                }
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// The stream-K shapes compute-sanitizer runs: each tile, a K tail, one and
/// two K tiles, ragged edges - and a K tail with many K tiles on the mid and
/// large tiles, where a continuation loads the partial last tile in the
/// loop, not the prologue. A loop K limit that forgot where its segment
/// started still gave the right bits (the tail never reads past K) and read
/// past A and B; only this run sees it. The N odd shapes: a continuation's
/// element loads past N, unguarded, read past C only here.
const STREAMK_SANITIZED: [usize; 12] = [0, 1, 2, 3, 6, 10, 11, 12, 15, 17, 18, 19];

/// Child: stream-K runs for compute-sanitizer, W exactly as large as the
/// header says.
#[test]
fn child_sanitized_streamk_runs() {
    if std::env::var("Y_F32_GEMM_SANITIZE").is_err() {
        return;
    }
    let ctx = CudaContext::new().expect("the parent checked for a driver");
    for &si in &STREAMK_SANITIZED {
        let (m, n, k, sms, _) = STREAMK_SHAPES[si];
        let dir = profile_dir(&format!("f32_gemm_sk_san{si}"), sms);
        let name = format!("sks{si}");
        let ptx = compile_text(&dir, &name, &source_w(&name, m, n, k), "--emit-ptx").unwrap();
        let h = sk_header(&ptx).unwrap();
        let (m, n, k) = (m as usize, n as usize, k as usize);
        let module = ctx.load_ptx(&ptx, &name).unwrap();
        module.set_max_dynamic_smem(h.smem).unwrap();
        let (a, b) = (operand(m * k, 5 + si as u64), operand(k * n, 7 + si as u64));
        let (da, db, dc) = (ctx.alloc(m * k * 4).unwrap(), ctx.alloc(k * n * 4).unwrap(), ctx.alloc(m * n * 4).unwrap());
        let dw = ctx.alloc(h.w_words * 8).unwrap();
        ctx.memcpy_htod_at(&da, 0, &as_bytes(&a)).unwrap();
        ctx.memcpy_htod_at(&db, 0, &as_bytes(&b)).unwrap();
        ctx.memset_u8(&dw, 0).unwrap();
        for grid in [h.p, 1, 3, 3 * h.tiles + 1] {
            ctx.launch(&module, (grid, 1, 1), (h.block, 1, 1), h.smem, &[da.device_ptr(), db.device_ptr(), dc.device_ptr(), dw.device_ptr()]).unwrap();
            ctx.synchronize().unwrap();
            let mut raw = vec![0u8; m * n * 4];
            ctx.memcpy_dtoh_at(&mut raw, &dc, 0).unwrap();
            for i in [0, m / 2, m - 1] {
                for j in 0..n {
                    let g = f32::from_le_bytes(raw[(i * n + j) * 4..(i * n + j) * 4 + 4].try_into().unwrap());
                    assert_eq!(g.to_bits(), oracle(&a, &b, n, k, i, j).to_bits(), "{m}x{n}x{k} grid {grid} C[{i}][{j}]");
                }
            }
        }
        println!("SANITIZED_SK_RUN {m}x{n}x{k} {}", h.tile);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// Child: launch a stream-K kernel on the grid in `Y_F32_GEMM_CHILD_GRID`
/// (`x,y,z`) and print whether the launch was refused or the kernel stopped.
#[test]
fn child_streamk_launch_on_a_given_grid() {
    let Ok(grid) = std::env::var("Y_F32_GEMM_CHILD_GRID") else { return };
    let g: Vec<u32> = grid.split(',').map(|v| v.parse().unwrap()).collect();
    let ctx = CudaContext::new().expect("the parent checked for a driver");
    let (m, n, k, sms, _) = STREAMK_SHAPES[0];
    let dir = profile_dir("f32_gemm_sk_child", sms);
    let ptx = compile_text(&dir, "skchild", &source_w("skchild", m, n, k), "--emit-ptx").unwrap();
    let h = sk_header(&ptx).unwrap();
    let module = ctx.load_ptx(&ptx, "skchild").unwrap();
    module.set_max_dynamic_smem(h.smem).unwrap();
    let (m, n, k) = (m as usize, n as usize, k as usize);
    let (da, db, dc, dw) = (ctx.alloc(m * k * 4).unwrap(), ctx.alloc(k * n * 4).unwrap(), ctx.alloc(m * n * 4).unwrap(), ctx.alloc(h.w_words * 8).unwrap());
    for d in [&da, &db, &dc, &dw] {
        ctx.memset_u8(d, 0).unwrap();
    }
    let r = match ctx.launch(&module, (g[0], g[1], g[2]), (h.block, 1, 1), h.smem, &[da.device_ptr(), db.device_ptr(), dc.device_ptr(), dw.device_ptr()]) {
        Err(e) => format!("launch refused {e}"),
        Ok(()) => match ctx.synchronize() {
            Ok(()) => "ok".to_string(),
            Err(e) => format!("stopped {e}"),
        },
    };
    println!("CHILD_RESULT grid={grid}: {r}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The CTA counts are 32-bit; a grid of 2^32 CTAs or more would wrap them
/// (a wrapped count of 0 divides by zero), so the kernel traps. The grid is
/// legal (CUDA accepts it), so it is the trap that stops it, not the launch:
/// a kernel without the check would grind through 8.6e9 CTAs, which the
/// deadline turns into a failure.
#[test]
fn a_streamk_grid_of_2_to_the_32_ctas_traps() {
    if CudaContext::new().is_none() {
        eprintln!("SKIP: no CUDA driver - the grid check was not exercised.");
        return;
    }
    let run = |grid: &str| {
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["child_streamk_launch_on_a_given_grid", "--exact", "--nocapture", "--test-threads=1"])
            .env("Y_F32_GEMM_CHILD_GRID", grid)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let start = std::time::Instant::now();
        while child.try_wait().unwrap().is_none() {
            if start.elapsed() > std::time::Duration::from_secs(120) {
                let _ = child.kill();
                panic!("grid {grid}: the child ran for 2 minutes, so nothing stopped it");
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        let out = child.wait_with_output().unwrap();
        let text = String::from_utf8_lossy(&out.stdout).to_string();
        text.lines()
            .find_map(|l| l.find("CHILD_RESULT").map(|at| l[at..].to_string()))
            .unwrap_or_else(|| panic!("grid {grid}: the child printed no result:\n{text}{}", String::from_utf8_lossy(&out.stderr)))
    };
    // The control: the plan's grid runs.
    assert_eq!(run("5,1,1"), "CHILD_RESULT grid=5,1,1: ok");
    // 65536 x 65535 x 2 = 8,589,803,520 CTAs.
    let r = run("65536,65535,2");
    assert!(r.starts_with("CHILD_RESULT grid=65536,65535,2: stopped"), "a grid of 2^32+ CTAs was not stopped by the kernel: {r}");
}
