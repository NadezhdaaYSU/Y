//! Where the F16 GEMM's accumulation happens.
//!
//! The tensor core adds into its accumulator with less than float32 rounding,
//! so a K loop chained in it - `mma d, a, b, d`, what cuBLAS's and Triton's
//! tensor-core kernels do - leaves same-sign results about 1.2e-5 per 1024 of
//! K below the true sum. Starting each mma from zero and adding its result in
//! float32 (`add.rn.f32`) keeps them within float32 rounding at any K, and
//! compensating those adds (`kahan`) narrows the error further. Compensated,
//! it is the default for GEMMs of at most 32 rows (split K keeps plain
//! float32). Above 32 rows, from K = 4096, the chain runs one block of K at a
//! time into a block sum added in float32 (`blocked`) where the block sums fit
//! the registers. `Y_F16_ACCUMULATE=f32|kahan|blocked|mma` selects one
//! everywhere. This file checks the selection, and on the device that each
//! mode computes exact integer products, that only the float32 ones keep
//! same-sign error at float32's level, that compensation narrows it, and that
//! blocks shrink the chain's error.

use std::path::Path;
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

/// The module the compiler emits for an `m x n x k` F16 GEMM, with
/// `Y_F16_ACCUMULATE` set to `knob` (or unset), or its error output.
fn compile(dir: &Path, tag: &str, m: u32, n: u32, k: u32, knob: Option<&str>) -> Result<String, String> {
    let params = "A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>";
    compile_kernel(dir, tag, m, n, k, params, knob)
}

/// `compile` for a kernel taking `params` (the fused SwiGLU's, say).
fn compile_kernel(dir: &Path, tag: &str, m: u32, n: u32, k: u32, params: &str, knob: Option<&str>) -> Result<String, String> {
    compile_source(dir, tag, &format!("@tile({m}, {n}, {k})\nkernel {tag}({params}) {{\n}}\n\nfn main() {{\n}}\n"), knob)
}

/// The module the compiler emits for `source`, under `knob` as `compile`.
fn compile_source(dir: &Path, tag: &str, source: &str, knob: Option<&str>) -> Result<String, String> {
    compile_source_tiled(dir, tag, source, knob, None)
}

/// `compile_source` with `Y_SWIGLU_TILE` set to `swiglu_tile` (or unset).
fn compile_source_tiled(dir: &Path, tag: &str, source: &str, knob: Option<&str>, swiglu_tile: Option<&str>) -> Result<String, String> {
    compile_source_env(dir, tag, source, knob, swiglu_tile, None)
}

/// `compile_source_tiled` with `Y_F16_BLOCK_K` set to `block_k` (or unset).
fn compile_source_env(dir: &Path, tag: &str, source: &str, knob: Option<&str>, swiglu_tile: Option<&str>, block_k: Option<&str>) -> Result<String, String> {
    let src = dir.join(format!("{tag}.ysu"));
    std::fs::write(&src, source).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_Y"));
    cmd.arg(&src).arg("--emit-ptx").current_dir(dir).env_remove("Y_CTA_OVERRIDE");
    for (var, value) in [("Y_F16_ACCUMULATE", knob), ("Y_SWIGLU_TILE", swiglu_tile), ("Y_F16_BLOCK_K", block_k)] {
        match value {
            Some(v) => cmd.env(var, v),
            None => cmd.env_remove(var),
        };
    }
    let out = cmd.output().expect("run Y");
    if !out.status.success() {
        return Err(format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)));
    }
    Ok(std::fs::read_to_string(src.with_extension("ptx")).expect("no .ptx"))
}

/// The K loop's mmas start from zero and fold in with float32 adds, and none
/// chains its own accumulator. (Any rounding of the add: which one is the
/// precision test's to judge.)
fn accumulates_in_f32(ptx: &str) -> bool {
    let mmas: Vec<&str> = ptx.lines().map(str::trim).filter(|l| l.starts_with("mma.sync.aligned.m16n8k16")).collect();
    assert!(!mmas.is_empty(), "no f16 mma");
    let chained = |l: &str| {
        let ops: Vec<&str> = l.split('{').skip(1).map(|o| o.split('}').next().unwrap()).collect();
        ops.len() == 4 && ops[0] == ops[3]
    };
    let from_zero = mmas.iter().filter(|l| !chained(l)).count();
    assert!(from_zero == 0 || from_zero == mmas.len(), "a module mixing the two forms");
    let adds = ptx.lines().map(str::trim).filter(|l| l.starts_with("add.") && l.contains(".f32 ")).count();
    from_zero == mmas.len() && adds >= 4 * mmas.len()
}

/// `line`'s operands when it is an `op` instruction.
fn operands(line: &str, op: &str) -> Option<Vec<String>> {
    let rest = line.trim().strip_prefix(op)?.strip_suffix(';')?;
    Some(rest.split(',').map(|o| o.trim().to_string()).collect())
}

/// The registers the mmas' results are added into, once per add:
/// `add.rn.f32 x, x, t` for an mma's output `t`.
fn mma_result_adds(body: &str) -> Vec<String> {
    let outs: std::collections::HashSet<String> = body
        .lines()
        .filter(|l| l.trim().starts_with("mma.sync.aligned.m16n8k16"))
        .flat_map(|l| l.split('{').nth(1).unwrap().split('}').next().unwrap().split(',').map(|r| r.trim().to_string()).collect::<Vec<_>>())
        .collect();
    body.lines()
        .filter_map(|l| operands(l, "add.rn.f32 "))
        .filter(|o| o[0] == o[1] && outs.contains(&o[2]))
        .map(|o| o[0].clone())
        .collect()
}

/// The compensated folds of one kernel's body - `add.rn.f32 s', s, x`,
/// `sub.rn.f32 b, s, s'`, `add.rn.f32 e, b, x`, `add.rn.f32 c, c, e` - as
/// (x, s, c): the sum folded in, the accumulator and its compensation.
fn kahan_folds(body: &str) -> Vec<(String, String, String)> {
    let lines: Vec<&str> = body.lines().collect();
    let mut folds = Vec::new();
    for w in lines.windows(4) {
        let (Some(sum), Some(back), Some(e), Some(c)) = (
            operands(w[0], "add.rn.f32 "),
            operands(w[1], "sub.rn.f32 "),
            operands(w[2], "add.rn.f32 "),
            operands(w[3], "add.rn.f32 "),
        ) else {
            continue;
        };
        if back[1] == sum[1] && back[2] == sum[0] && e[1] == back[0] && e[2] == sum[2] && c[0] == c[1] && c[2] == e[0] {
            folds.push((sum[2].clone(), sum[1].clone(), c[0].clone()));
        }
    }
    folds
}

/// Every float32 add of an mma's result goes into a sum that a compensated
/// fold adds into an accumulator.
fn compensated(ptx: &str) -> bool {
    let adds = mma_result_adds(ptx);
    let folded: std::collections::HashSet<String> = kahan_folds(ptx).into_iter().map(|(x, _, _)| x).collect();
    !adds.is_empty() && adds.iter().all(|x| folded.contains(x))
}

/// The two levels: each tile sum takes one add per 16-deep step of the
/// kernel's K-tile (`CTA MxNxK` in its header) and is folded in twice, at
/// the end of the tile and once more after the K loop.
fn assert_tile_sums(ptx: &str, what: &str) {
    let cta = ptx.split("| CTA ").nth(1).unwrap_or_else(|| panic!("{what}: no CTA in the header"));
    let cta_k: usize = cta.split(|c: char| c == 'x' || c == ' ').nth(2).unwrap().parse().unwrap();
    let adds = mma_result_adds(ptx);
    let folds = kahan_folds(ptx);
    for x in adds.iter().collect::<std::collections::HashSet<_>>() {
        assert_eq!(adds.iter().filter(|a| a == &x).count(), cta_k / 16, "{what}: mma results added into {x}");
        assert_eq!(folds.iter().filter(|(f, _, _)| f == x).count(), 2, "{what}: folds of {x}");
    }
}

/// The block folds of one kernel's body - `add.rn.f32 s, s, x` and then
/// `mov.f32 x, 0f00000000;` - as (s, x): the accumulator and its block sum.
fn block_folds(body: &str) -> Vec<(String, String)> {
    let lines: Vec<&str> = body.lines().collect();
    lines
        .windows(2)
        .filter_map(|w| {
            let (add, zero) = (operands(w[0], "add.rn.f32 ")?, operands(w[1], "mov.f32 ")?);
            (add[0] == add[1] && zero[0] == add[2] && zero[1] == "0f00000000").then(|| (add[0].clone(), add[2].clone()))
        })
        .collect()
}

/// Blocked summation in every kernel of `ptx`: each mma chains (its `d` is
/// its `c`) in registers that are block sums, never accumulators. Each block
/// sum is added into its accumulator, and zeroed, exactly once, in one run of
/// folds on the exit of an inner loop over a block's `tiles` K-tiles: below
/// the K loop's exit test the block's end, `min(k + tiles, K-tiles)`, then
/// the inner loop's top; after the K loop counts a tile done, it branches
/// back there while the count is below the block's end; after the folds,
/// back to the K loop's exit test. With one-tile blocks there is no inner
/// loop.
fn assert_blocked(ptx: &str, tiles: u32, what: &str) {
    use std::collections::HashSet;
    for body in ptx.split(".visible .entry ").skip(1) {
        let mmas: Vec<Vec<String>> = body
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with("mma.sync.aligned.m16n8k16"))
            .map(|l| l.split('{').skip(1).map(|o| o.split('}').next().unwrap().to_string()).collect())
            .collect();
        assert!(!mmas.is_empty(), "{what}: no f16 mma");
        assert!(mmas.iter().all(|o| o.len() == 4 && o[0] == o[3]), "{what}: an mma does not chain");
        let sums: HashSet<String> = mmas.iter().flat_map(|o| o[0].split(',').map(|r| r.trim().to_string())).collect();
        let folds = block_folds(body);
        let folded: HashSet<String> = folds.iter().map(|(_, x)| x.clone()).collect();
        assert_eq!(folded, sums, "{what}: the block sums folded are not the registers the mmas chain in");
        assert_eq!(folds.len(), sums.len(), "{what}: a block sum folded more than once");
        let accs: HashSet<&str> = folds.iter().map(|(s, _)| s.as_str()).collect();
        assert_eq!(accs.len(), folds.len(), "{what}: two block sums fold into one accumulator");
        assert!(accs.iter().all(|s| !sums.contains(*s)), "{what}: an accumulator is also a block sum");
        for (s, x) in &folds {
            let adds = body.lines().filter_map(|l| operands(l, "add.rn.f32 ")).filter(|o| o[0] == *s && o[1] == *s && o[2] == *x).count();
            assert_eq!(adds, 1, "{what}: {x} is added into {s} {adds} times");
        }
        let lines: Vec<&str> = body.lines().map(str::trim).collect();
        let first = lines.iter().position(|l| operands(l, "add.rn.f32 ").is_some_and(|o| o[0] == folds[0].0 && o[2] == folds[0].1)).unwrap();
        let after = first + 2 * folds.len();
        assert_eq!(block_folds(&lines[first..after].join("\n")).len(), folds.len(), "{what}: the folds are not one run");
        // After the folds, the back edge to the K loop's exit test.
        let outer = lines[after].strip_prefix("bra ").and_then(|l| l.strip_suffix(';')).unwrap_or_else(|| panic!("{what}: {} after the folds", lines[after]));
        let top = lines.iter().position(|l| *l == format!("{outer}:")).unwrap_or_else(|| panic!("{what}: no {outer}"));
        let exit = operands(lines[top + 1], "setp.ge.u32 ").unwrap_or_else(|| panic!("{what}: {outer} is not the K loop's exit test"));
        assert!(lines[top + 2].starts_with(&format!("@{} bra ", exit[0])), "{what}: {}", lines[top + 2]);
        let (k, k_tiles) = (&exit[1], &exit[2]);
        // Above them the count of tiles done, then the inner loop's back edge.
        let inc_at = if tiles == 1 { first - 1 } else { first - 3 };
        let inc = operands(lines[inc_at], "add.u32 ").unwrap_or_else(|| panic!("{what}: {} where the count should be", lines[inc_at]));
        assert!(inc[0] == *k && inc[1] == *k && inc[2] == "1", "{what}: {inc:?} does not count the K loop's tiles");
        if tiles > 1 {
            // The block's end, below the exit test, and the inner loop's top.
            let start = operands(lines[top + 3], "add.u32 ").unwrap_or_else(|| panic!("{what}: {} below the exit test", lines[top + 3]));
            let cap = operands(lines[top + 4], "min.u32 ").unwrap_or_else(|| panic!("{what}: {} below the exit test", lines[top + 4]));
            let end = &start[0];
            assert!(start[1] == *k && start[2] == tiles.to_string(), "{what}: a block is not {tiles} K-tiles from its start: {start:?}");
            assert!(cap[0] == *end && cap[1] == *end && cap[2] == *k_tiles, "{what}: the block's end is not capped at K: {cap:?}");
            let lt = operands(lines[first - 2], "setp.lt.u32 ").unwrap_or_else(|| panic!("{what}: no end-of-block test above the folds"));
            assert!(lt[1] == *k && lt[2] == *end, "{what}: the inner loop does not run to the block's end: {lt:?}");
            let inner = lines[first - 1].strip_prefix(&format!("@{} bra ", lt[0])).and_then(|l| l.strip_suffix(';'))
                .unwrap_or_else(|| panic!("{what}: {}", lines[first - 1]));
            assert_eq!(lines[top + 5], format!("{inner}:"), "{what}: the inner loop's top is not below the block's end");
        } else {
            assert!(!body.contains("F16_BLOCK_TILE"), "{what}: an inner loop for one-tile blocks");
        }
    }
}

/// The tile in a kernel's header (`CTA MxNxK ... | WMxWN warps`) as the
/// float32 accumulators each thread holds and the CTA's threads.
fn accumulators_and_threads(ptx: &str) -> (u32, u32) {
    let cta = ptx.split("| CTA ").nth(1).expect("no CTA in the header");
    let nums: Vec<u32> = cta.split(|c: char| !c.is_ascii_digit()).filter(|x| !x.is_empty()).take(5).map(|x| x.parse().unwrap()).collect();
    let [cm, cn, _, wm, wn] = nums[..] else { panic!("header {cta:?}") };
    ((cm / wm / 16) * (cn / wn / 16) * 8, wm * wn * 32)
}

#[test]
fn blocks_start_at_k_4096_above_32_rows_where_the_block_sums_fit() {
    let dir = pinned::pinned_scratch("f16_acc_blocked", pinned::SM_PINNED);
    // The pinned profile gives 64 x 4096 a 64x64 tile of 2x2 warps, 32
    // accumulators a thread, and 4096^2 a 128x128 one, 128 a thread: their
    // block sums would take 2 x 128 registers.
    for (m, n, k, accs, want) in [(64u32, 4096u32, 4096u32, 32u32, true), (64, 4096, 8192, 32, true), (64, 4096, 2048, 32, false),
                                  (4096, 4096, 4096, 128, false), (32, 4096, 4096, 16, false)] {
        let what = format!("{m}x{n}x{k}, no knob");
        let ptx = compile(&dir, &format!("blk_{m}x{n}x{k}"), m, n, k, None).unwrap();
        assert_eq!(accumulators_and_threads(&ptx).0, accs, "{what}: not the tile this case checks");
        assert_eq!(!block_folds(&ptx).is_empty(), want, "{what}");
        if want {
            assert!(!accumulates_in_f32(&ptx), "{what}");
            // 256 of K a block: 8 of the tile's 32-deep K-tiles.
            assert_blocked(&ptx, 8, &what);
        } else if m > 32 {
            assert!(!accumulates_in_f32(&ptx) && !compensated(&ptx), "{what}: not the chain");
        }
    }
    // The knob blocks any shape, decode and short K included, and
    // `Y_F16_BLOCK_K` sets the block: 128 of K is 4 K-tiles, 32 one.
    let src = |tag: &str, m: u32, k: u32| format!("@tile({m}, 256, {k})\nkernel {tag}(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>) {{\n}}\n\nfn main() {{\n}}\n");
    for (m, k, block, tiles) in [(8u32, 512u32, None, 8u32), (256, 512, None, 8), (256, 512, Some("128"), 4), (256, 512, Some("32"), 1)] {
        let tag = format!("blk_knob_{m}_{}", block.unwrap_or("dflt"));
        let ptx = compile_source_env(&dir, &tag, &src(&tag, m, k), Some("blocked"), None, block).unwrap();
        assert_eq!(ptx.split("| CTA ").nth(1).and_then(|c| c.split('x').nth(2)).and_then(|c| c.split(' ').next()), Some("32"), "{tag}: not 32-deep K-tiles");
        assert!(!accumulates_in_f32(&ptx) && !compensated(&ptx), "{tag}");
        assert_blocked(&ptx, tiles, &tag);
    }
    // K = 32 is one K-tile: the single-stage loop (no cp.async), folding too.
    let ptx = compile_source_env(&dir, "blk_one_tile", &src("blk_one_tile", 64, 32), Some("blocked"), None, None).unwrap();
    assert!(!ptx.contains("cp.async"), "K = 32 took the pipelined loop");
    assert_blocked(&ptx, 8, "K = 32");
    let err = compile_source_env(&dir, "blk_bad", &src("blk_bad", 256, 512), Some("blocked"), None, Some("100")).expect_err("a block of 100 compiled");
    assert!(err.contains("Y_F16_BLOCK_K must be"), "{err}");
    // The fused SwiGLU kernel blocks both of its accumulator arrays, by
    // default from K = 4096 (its fixed 64x64 tile fits).
    let swiglu = "X: GlobalMemory<F16>, Wgate: GlobalMemory<F16>, Wup: GlobalMemory<F16>, Out: GlobalMemory<F32>";
    for (k, knob, want) in [(256u32, Some("blocked"), true), (32, Some("blocked"), true), (4096, None, true), (2048, None, false)] {
        let what = format!("SwiGLU K = {k}, {knob:?}");
        let ptx = compile_kernel(&dir, &format!("blk_swiglu_{k}"), 64, 128, k, swiglu, knob).unwrap();
        assert!(ptx.contains("EPI_SWIGLU"), "not the fused SwiGLU kernel");
        assert_eq!(ptx.contains("cp.async"), k > 32, "{what}: K = 32 is the single-stage loop, longer K the pipelined one");
        assert_eq!(!block_folds(&ptx).is_empty(), want, "{what}");
        if want {
            assert_blocked(&ptx, 8, &what);
            let (accs, _) = accumulators_and_threads(&ptx);
            assert_eq!(block_folds(&ptx).len() as u32, 2 * accs, "{what}: not both arrays");
        }
    }
    // Its fit counts both arrays: a 128x64 tile of 2x2 warps holds 64
    // accumulators a thread in each, whose block sums would take 2 x 128
    // registers (one array's alone would fit).
    let tag = "blk_swiglu_wide";
    let wide = format!("@tile(128, 128, 4096)\nkernel {tag}({swiglu}) {{\n}}\n\nfn main() {{\n}}\n");
    let ptx = compile_source_tiled(&dir, tag, &wide, None, Some("128,64,32,2,2")).unwrap();
    assert!(ptx.contains("EPI_SWIGLU"), "not the fused SwiGLU kernel");
    assert_eq!(accumulators_and_threads(&ptx), (64, 128), "not the tile this case checks");
    assert!(block_folds(&ptx).is_empty() && !accumulates_in_f32(&ptx), "SwiGLU with 2 x 64 accumulators a thread is not the chain");
}

/// The `.maxnreg` a kernel of `ptx` declares.
fn maxnreg(ptx: &str) -> u32 {
    let at = ptx.find(".maxnreg ").expect("no .maxnreg") + ".maxnreg ".len();
    ptx[at..].split_whitespace().next().unwrap().parse().unwrap()
}

#[test]
fn blocked_swiglu_asks_for_the_ctas_shared_memory_allows() {
    // The fused SwiGLU kernel's 64x64 tile of 4x2 warps takes 48 KiB of
    // shared memory at K = 4096: two CTAs an SM where it has 100 KiB. The
    // chain fits two in its registers; blocked, the kernel asks for a
    // register limit that leaves two (65,536 / (256 x 2) = 128).
    let swiglu = "X: GlobalMemory<F16>, Wgate: GlobalMemory<F16>, Wup: GlobalMemory<F16>, Out: GlobalMemory<F32>";
    let dir = pinned::pinned_scratch("f16_blk_swiglu_regs", pinned::SM_PINNED);
    let profile = dir.join(".ysu_hw_profile");
    let pinned_text = std::fs::read_to_string(&profile).unwrap();
    std::fs::write(&profile, format!("{pinned_text}MAX_SMEM_PER_SM_BYTES=102400\nMAX_REGS_PER_SM=65536\n")).unwrap();
    let blocked = compile_kernel(&dir, "blk_sw_regs", 64, 128, 4096, swiglu, None).unwrap();
    assert!(!block_folds(&blocked).is_empty(), "SwiGLU at K = 4096 is not blocked by default");
    assert!(blocked.contains("Dynamic shared memory required: 49152 bytes"), "not the 48 KiB this case checks");
    assert_eq!(maxnreg(&blocked), 128, "blocked SwiGLU's register limit");
    // The controls: the chain, and blocked where shared memory allows one
    // CTA an SM (the pinned profile's 48 KiB), ask for nothing.
    let chain = compile_kernel(&dir, "blk_sw_regs_chain", 64, 128, 4096, swiglu, Some("mma")).unwrap();
    assert!(maxnreg(&chain) > 128, "the chain's register limit is {}", maxnreg(&chain));
    let one = pinned::pinned_scratch("f16_blk_swiglu_regs_one", pinned::SM_PINNED);
    let ptx = compile_kernel(&one, "blk_sw_regs_one", 64, 128, 4096, swiglu, None).unwrap();
    assert!(!block_folds(&ptx).is_empty() && maxnreg(&ptx) > 128, "one CTA an SM, register limit {}", maxnreg(&ptx));
    // Under that limit ptxas fits it without spilling (it took 138
    // registers without one).
    let Ok(out) = Command::new("ptxas").args(["-arch=sm_89", "-v", "-o", "/dev/null"]).arg(dir.join("blk_sw_regs.ptx")).output() else {
        eprintln!("SKIP: no ptxas - the blocked SwiGLU kernel's spills were not checked.");
        return;
    };
    let log = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{log}");
    assert!(log.contains(" 0 bytes spill stores"), "blocked SwiGLU under its register limit spills: {log}");
}

#[test]
fn the_default_follows_the_row_count_and_the_knob_overrides_it() {
    let dir = pinned::pinned_scratch("f16_acc_mode", pinned::SM_PINNED);
    for (m, want) in [(1u32, true), (8, true), (32, true), (33, false), (256, false)] {
        let ptx = compile(&dir, &format!("acc_m{m}"), m, 256, 512, None).unwrap();
        assert_eq!(accumulates_in_f32(&ptx), want, "M = {m}, no knob");
        assert_eq!(compensated(&ptx), want, "M = {m}, no knob");
        if want {
            assert_tile_sums(&ptx, &format!("M = {m}, no knob"));
        }
    }
    for (m, knob, want) in [(256u32, "f32", true), (8, "f32", true), (8, "mma", false), (8, "kahan", true), (256, "kahan", true)] {
        let ptx = compile(&dir, &format!("acc_{knob}_{m}"), m, 256, 512, Some(knob)).unwrap();
        assert_eq!(accumulates_in_f32(&ptx), want, "M = {m}, Y_F16_ACCUMULATE={knob}");
        assert_eq!(compensated(&ptx), knob == "kahan", "M = {m}, Y_F16_ACCUMULATE={knob}");
        if knob == "kahan" {
            assert_tile_sums(&ptx, &format!("M = {m}"));
        }
    }
    // The fused SwiGLU kernel's two accumulator arrays are compensated too.
    let swiglu = "X: GlobalMemory<F16>, Wgate: GlobalMemory<F16>, Wup: GlobalMemory<F16>, Out: GlobalMemory<F32>";
    for knob in ["f32", "kahan"] {
        let ptx = compile_kernel(&dir, &format!("acc_swiglu_{knob}"), 64, 128, 256, swiglu, Some(knob)).unwrap();
        assert!(ptx.contains("EPI_SWIGLU"), "not the fused SwiGLU kernel");
        assert!(accumulates_in_f32(&ptx), "SwiGLU, Y_F16_ACCUMULATE={knob}");
        assert_eq!(compensated(&ptx), knob == "kahan", "SwiGLU, Y_F16_ACCUMULATE={knob}");
        if knob == "kahan" {
            assert_tile_sums(&ptx, "SwiGLU");
        }
    }
    // With no knob it follows the row count as the GEMM does. Its fixed
    // tile's 64 rows must divide M, so 32 rows need a `Y_SWIGLU_TILE` of 32.
    for (m, tile, want) in [(32u32, Some("32,64,32,2,2"), true), (64, None, false)] {
        let tag = format!("acc_swiglu_m{m}");
        let src = format!("@tile({m}, 128, 256)\nkernel {tag}({swiglu}) {{\n}}\n\nfn main() {{\n}}\n");
        let ptx = compile_source_tiled(&dir, &tag, &src, None, tile).unwrap();
        assert!(ptx.contains("EPI_SWIGLU"), "not the fused SwiGLU kernel");
        assert_eq!(accumulates_in_f32(&ptx), want, "SwiGLU, M = {m}, no knob");
        assert_eq!(compensated(&ptx), want, "SwiGLU, M = {m}, no knob");
    }
    // Two kernels with different numbers of accumulators in one module: each
    // kernel's compensations and tile sums are its own registers, zeroed in
    // that kernel, never an accumulator, and the compensations written only
    // by the folds (register numbers start again in each kernel, so a
    // pairing left over from the first kernel would name another register
    // in the second).
    let gemm = "A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>";
    let two = format!("@tile(512, 512, 512)\nkernel acc_two_a({gemm}) {{\n}}\n\n@tile(8, 256, 512)\nkernel acc_two_b({gemm}) {{\n}}\n\nfn main() {{\n}}\n");
    let ptx = compile_source(&dir, "acc_two", &two, Some("kahan")).unwrap();
    let bodies: Vec<&str> = ptx.split(".visible .entry ").skip(1).collect();
    assert_eq!(bodies.len(), 2);
    let mut acc_counts = Vec::new();
    for body in bodies {
        let name = body.split('(').next().unwrap();
        let folds = kahan_folds(body);
        assert!(!folds.is_empty(), "{name}: no compensated fold");
        assert!(compensated(body), "{name}: an mma result added outside a tile sum");
        let accs: std::collections::HashSet<&str> = folds.iter().map(|(_, s, _)| s.as_str()).collect();
        acc_counts.push(accs.len());
        for (x, _, c) in &folds {
            for (reg, role) in [(x, "tile sum"), (c, "compensation")] {
                assert!(!accs.contains(reg.as_str()), "{name}: {role} {reg} is also an accumulator");
                assert!(body.contains(&format!("mov.f32 {reg}, 0f00000000;")), "{name}: {role} {reg} is never zeroed here");
            }
            let writes = body.lines().map(str::trim).filter(|l| l.split_whitespace().nth(1) == Some(&format!("{c},"))).count();
            let updates = folds.iter().filter(|(_, _, k)| k == c).count();
            assert_eq!(writes, updates + 1, "{name}: compensation {c} is written outside the folds");
        }
    }
    assert!(acc_counts[0] > acc_counts[1], "the two tiles hold {acc_counts:?} accumulators: not the case this checks");
    let err = compile(&dir, "acc_bad", 8, 256, 512, Some("fp32")).expect_err("a misspelt knob compiled");
    assert!(err.contains("Y_F16_ACCUMULATE must be"), "{err}");
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The value of a normal f16 bit pattern.
fn f16_value(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1F) as i32;
    let f = (h & 0x3FF) as f64;
    assert!(e > 0 && e < 31, "normal values only");
    sign * (1.0 + f / 1024.0) * 2f64.powi(e - 15)
}

/// The f16 bit pattern of a small integer.
fn f16_of_int(v: i32) -> u16 {
    if v == 0 {
        return 0;
    }
    let (sign, a) = (if v < 0 { 0x8000u16 } else { 0 }, v.unsigned_abs());
    let e = 31 - a.leading_zeros();
    sign | (((e + 15) as u16) << 10) | (((a << (10 - e)) & 0x3FF) as u16)
}

/// A runner of the `m x n x k` F16 GEMM `ptx` (entry `tag`, launched as its
/// header says) on f16 bit patterns, returning C.
fn launcher<'c>(ctx: &'c y::cuda_runtime::CudaContext, ptx: &str, tag: &str, m: u32, n: u32, k: u32) -> impl Fn(&[u16], &[u16]) -> Vec<f32> + 'c {
    let (mu, nu, ku) = (m as usize, n as usize, k as usize);
    let field = |key: &str| -> u32 {
        let at = ptx.find(key).unwrap_or_else(|| panic!("no {key:?}")) + key.len();
        ptx[at..].chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap()
    };
    let (cm, cn) = (field("| CTA "), field(&format!("| CTA {}x", field("| CTA "))));
    let warps_at = ptx.find(" warps").unwrap();
    let w: Vec<u32> = ptx[..warps_at].rsplit("| ").next().unwrap().split('x').map(|x| x.trim().parse().unwrap()).collect();
    let smem = ptx.find("Dynamic shared memory required: ").map(|_| field("Dynamic shared memory required: ")).unwrap_or(0);
    let module = ctx.load_ptx(ptx, tag).unwrap();
    if smem > 48 * 1024 {
        module.set_max_dynamic_smem(smem).unwrap();
    }
    let d_a = ctx.alloc(mu * ku * 2).unwrap();
    let d_b = ctx.alloc(ku * nu * 2).unwrap();
    let d_c = ctx.alloc(mu * nu * 4).unwrap();
    move |a: &[u16], b: &[u16]| -> Vec<f32> {
        ctx.memcpy_htod_at(&d_a, 0, &a.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        ctx.memcpy_htod_at(&d_b, 0, &b.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        ctx.memset_u8(&d_c, 0xFF).unwrap();
        let grid = (n.div_ceil(cn), m.div_ceil(cm), 1);
        ctx.launch(&module, grid, (w[0] * w[1] * 32, 1, 1), smem, &[d_a.device_ptr(), d_b.device_ptr(), d_c.device_ptr()]).unwrap();
        ctx.synchronize().unwrap();
        let mut raw = vec![0u8; mu * nu * 4];
        ctx.memcpy_dtoh_at(&mut raw, &d_c, 0).unwrap();
        raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }
}

#[test]
fn float32_accumulation_keeps_same_sign_error_at_float32_rounding_on_the_gpu() {
    use y::cuda_runtime::CudaContext;
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver - no F16 GEMM was executed.");
        return;
    };
    let dir = pinned::pinned_scratch("f16_acc_gpu", pinned::SM_PINNED);
    let k = 4096u32;
    let mut state = 0xACC0_F16Fu64;
    // A decode shape (compensated by default) and a square one (the
    // tensor-core chain by default), each in every mode.
    for (m, n) in [(8u32, 256u32), (128, 128)] {
        let (mu, nu, ku) = (m as usize, n as usize, k as usize);
        // |N|-like positive f16 in [0.25, 2): no cancellation hides an error.
        let a: Vec<u16> = (0..mu * ku).map(|_| { let r = splitmix(&mut state); ((13 + r % 2) << 10) as u16 | ((r >> 8) & 0x3FF) as u16 }).collect();
        let b: Vec<u16> = (0..ku * nu).map(|_| { let r = splitmix(&mut state); ((13 + r % 2) << 10) as u16 | ((r >> 8) & 0x3FF) as u16 }).collect();
        let (av, bv): (Vec<f64>, Vec<f64>) = (a.iter().map(|&h| f16_value(h)).collect(), b.iter().map(|&h| f16_value(h)).collect());
        let mut exact = vec![0f64; mu * nu];
        for i in 0..mu {
            for j in 0..nu {
                exact[i * nu + j] = (0..ku).map(|kk| av[i * ku + kk] * bv[kk * nu + j]).sum();
            }
        }
        let ia: Vec<i32> = (0..mu * ku).map(|_| (splitmix(&mut state) % 9) as i32 - 4).collect();
        let ib: Vec<i32> = (0..ku * nu).map(|_| (splitmix(&mut state) % 9) as i32 - 4).collect();
        let mut iref = vec![0f32; mu * nu];
        for i in 0..mu {
            for j in 0..nu {
                iref[i * nu + j] = (0..ku).map(|kk| (ia[i * ku + kk] * ib[kk * nu + j]) as i64).sum::<i64>() as f32;
            }
        }
        let mut errs = Vec::new();
        for knob in ["mma", "f32", "kahan", "blocked"] {
            let tag = format!("acc_{knob}_{m}x{n}");
            let ptx = compile(&dir, &tag, m, n, k, Some(knob)).unwrap();
            assert_eq!(accumulates_in_f32(&ptx), knob == "f32" || knob == "kahan");
            let run = launcher(&ctx, &ptx, &tag, m, n, k);
            let ints = run(&ia.iter().map(|&v| f16_of_int(v)).collect::<Vec<_>>(), &ib.iter().map(|&v| f16_of_int(v)).collect::<Vec<_>>());
            let wrong = ints.iter().zip(&iref).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            assert_eq!(wrong, 0, "{m}x{n}x{k} {knob}: {wrong} integer products wrong");
            let got = run(&a, &b);
            let rel: Vec<f64> = got.iter().zip(&exact).map(|(&g, &e)| (g as f64 - e) / e).collect();
            let mean_rel = rel.iter().sum::<f64>() / rel.len() as f64;
            let spread = (rel.iter().map(|r| (r - mean_rel) * (r - mean_rel)).sum::<f64>() / rel.len() as f64).sqrt();
            errs.push((mean_rel, spread));
            // One +Inf in row 3 of A, B positive: that row's sums are +Inf in
            // every mode (compensated, `Inf - Inf` would make them NaN), and
            // no other row changes.
            let mut a_inf = a.clone();
            a_inf[3 * ku + 100] = 0x7C00;
            let inf = run(&a_inf, &b);
            for (idx, (&v, &g)) in inf.iter().zip(&got).enumerate() {
                if idx / nu == 3 {
                    assert_eq!(v, f32::INFINITY, "{m}x{n}x{k} {knob}: output {idx} of the row holding +Inf is {v}");
                } else {
                    assert_eq!(v.to_bits(), g.to_bits(), "{m}x{n}x{k} {knob}: output {idx} moved with another row's +Inf");
                }
            }
        }
        let ((chained, _), (f32_added, f32_spread), (kahan, kahan_spread), (blocked, _)) = (errs[0], errs[1], errs[2], errs[3]);
        // The control: the chain's known bias, about -5.0e-5 at K = 4096.
        assert!(chained < -2e-5, "{m}x{n}x{k}: the chained accumulator is off by only {chained:e} on average");
        // float32 rounding of a 256-term sum of exact-ish partials: ~1e-7.
        assert!(f32_added.abs() < 1e-6, "{m}x{n}x{k}: float32 accumulation is off by {f32_added:e} on average");
        assert!(kahan.abs() < 1e-6, "{m}x{n}x{k}: compensated accumulation is off by {kahan:e} on average");
        // Compensation removes the adds' rounding, the spread's main part:
        // what is left is each mma's own truncation. Measured 2.7e-8 against
        // 2.5e-7 (0.11x) at both shapes; without compensation it is 1x.
        assert!(kahan_spread < 0.3 * f32_spread,
                "{m}x{n}x{k}: compensated spread {kahan_spread:e} against float32's {f32_spread:e}");
        // Blocks of 256 chain 16 mmas, not 256: measured -2.6e-6 at both
        // shapes, 1/19 of the chain's bias. It is still the chain's kind of
        // error, a truncation bias, 40x float32's: the mmas did chain.
        assert!(blocked.abs() < chained.abs() / 8.0, "{m}x{n}x{k}: blocked {blocked:e} against the chain's {chained:e}");
        assert!(blocked < 10.0 * f32_added.min(kahan), "{m}x{n}x{k}: blocked {blocked:e}: no chain's bias");
    }
}

#[test]
fn blocked_sums_count_every_product_once_on_the_gpu() {
    use y::cuda_runtime::CudaContext;
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver - no F16 GEMM was executed.");
        return;
    };
    let dir = pinned::pinned_scratch("f16_blk_gpu", pinned::SM_PINNED);
    let mut state = 0xB10C_F16Fu64;
    // 32-deep K-tiles, 8 to a block: K = 32 is one K-tile (the single-stage
    // loop), 96 ends inside the first block, 2080 one K-tile into the ninth,
    // 4096 at the end of the sixteenth.
    // Small integer products sum exactly at any order, so a product left out
    // (a block sum not added after the loop) or counted twice (a fold that
    // does not zero its block sum) changes an output.
    let (m, n) = (64u32, 128u32);
    for k in [32u32, 96, 2080, 4096] {
        let (mu, nu, ku) = (m as usize, n as usize, k as usize);
        let ia: Vec<i32> = (0..mu * ku).map(|_| (splitmix(&mut state) % 9) as i32 - 4).collect();
        let ib: Vec<i32> = (0..ku * nu).map(|_| (splitmix(&mut state) % 9) as i32 - 4).collect();
        let mut iref = vec![0f32; mu * nu];
        for i in 0..mu {
            for j in 0..nu {
                iref[i * nu + j] = (0..ku).map(|kk| (ia[i * ku + kk] * ib[kk * nu + j]) as i64).sum::<i64>() as f32;
            }
        }
        let tag = format!("blk_gpu_{k}");
        let ptx = compile(&dir, &tag, m, n, k, Some("blocked")).unwrap();
        assert_blocked(&ptx, 8, &tag);
        let run = launcher(&ctx, &ptx, &tag, m, n, k);
        let got = run(&ia.iter().map(|&v| f16_of_int(v)).collect::<Vec<_>>(), &ib.iter().map(|&v| f16_of_int(v)).collect::<Vec<_>>());
        let wrong = got.iter().zip(&iref).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        assert_eq!(wrong, 0, "{m}x{n}x{k} blocked: {wrong} of {} integer products wrong", got.len());
    }
}
