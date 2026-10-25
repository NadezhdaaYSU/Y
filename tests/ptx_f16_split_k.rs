//! The split-K F16 GEMM: an `@tile(M, N, K)` kernel named `<name>_splitk<S>`
//! taking (A, B: GlobalMemory<F16>, C, Partial: GlobalMemory<F32>). Its main
//! entry runs S equal K ranges in `%ctaid.z`, each written as an M x N float32
//! partial to slot z of Partial; by default each range of more than 32 rows
//! and longer than a block is summed in blocks (`Y_F16_ACCUMULATE=blocked`),
//! any other adds each mma in float32. Its `_reduce` entry adds the slots
//! into C in slot order, with no atomics.
//!
//! A kernel named `<name>_splitk` takes the split-K rule's split and tile
//! (`Autotuner::split_k_plan`, `tests/f16_split_k_rule.rs`): its module is
//! then the `_splitk<S>` one pinned to that tile, or, where the rule does not
//! split, the plain kernel writing C beside a `_reduce` that does nothing.
//!
//! This file checks the name grammar and the shapes refused, and on the
//! device: exact integer products; every slot equal, bit for bit, to the
//! unsplit kernel run on that slot's K range; C equal to the slots summed in
//! order on the host; the same bits on every launch; same-sign error at
//! float32's level, or blocked well below the chained control's; and the
//! tuner measuring the split kernel with its reduction. For `_splitk`: the
//! module against its pinned and plain twins, which tile it takes, the note a
//! plain GEMM gets, and on the device the launch its header states.

use std::path::Path;
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

/// Compiles an `m x n x k` F16 tile GEMM named `name` (with the workspace
/// parameter when `partial`), under `Y_F16_ACCUMULATE=knob` and
/// `Y_CTA_OVERRIDE=tile` when given; the module or the compiler's output.
fn compile(dir: &Path, name: &str, m: u32, n: u32, k: u32, partial: bool, knob: Option<&str>, tile: Option<&str>) -> Result<String, String> {
    let src = dir.join(format!("{name}.ysu"));
    let p = if partial { ", P: GlobalMemory<F32>" } else { "" };
    std::fs::write(&src, format!("@tile({m}, {n}, {k})\nkernel {name}(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>{p}) {{\n}}\n\nfn main() {{\n}}\n")).unwrap();
    compile_source(dir, &src, knob, tile)
}

fn compile_source(dir: &Path, src: &Path, knob: Option<&str>, tile: Option<&str>) -> Result<String, String> {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_Y"));
    cmd.arg(src).arg("--emit-ptx").current_dir(dir);
    match knob {
        Some(v) => cmd.env("Y_F16_ACCUMULATE", v),
        None => cmd.env_remove("Y_F16_ACCUMULATE"),
    };
    match tile {
        Some(v) => cmd.env("Y_CTA_OVERRIDE", v),
        None => cmd.env_remove("Y_CTA_OVERRIDE"),
    };
    let out = cmd.output().expect("run Y");
    let ptx = src.with_extension("ptx");
    if !out.status.success() || !ptx.exists() {
        return Err(format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)));
    }
    Ok(std::fs::read_to_string(ptx).unwrap())
}

/// The main entry adds block sums into its accumulators in the K loop
/// (`add.rn.f32 s, s, x` and then `mov.f32 x, 0f00000000;`); see
/// `ptx_f16_accumulate.rs` for the structure checked in full.
fn blocked(ptx: &str) -> bool {
    let lines: Vec<&str> = ptx.lines().map(str::trim).collect();
    lines.windows(2).any(|w| {
        let add = w[0].strip_prefix("add.rn.f32 ").and_then(|l| l.strip_suffix(';')).map(|l| l.split(", ").collect::<Vec<_>>());
        matches!(add, Some(o) if o.len() == 3 && o[0] == o[1] && w[1] == format!("mov.f32 {}, 0f00000000;", o[2]))
    })
}

fn entries(ptx: &str) -> Vec<String> {
    ptx.lines()
        .filter_map(|l| l.trim().strip_prefix(".visible .entry "))
        .map(|l| l.split('(').next().unwrap().to_string())
        .collect()
}

/// The main entry's mmas start from zero and fold in with float32 adds
/// (true), or every one chains its own accumulator (false).
fn accumulates_in_f32(ptx: &str) -> bool {
    let mmas: Vec<&str> = ptx.lines().map(str::trim).filter(|l| l.starts_with("mma.sync.aligned.m16n8k16")).collect();
    assert!(!mmas.is_empty(), "no f16 mma");
    let chained = |l: &str| {
        let ops: Vec<&str> = l.split('{').skip(1).map(|o| o.split('}').next().unwrap()).collect();
        ops.len() == 4 && ops[0] == ops[3]
    };
    let from_zero = mmas.iter().filter(|l| !chained(l)).count();
    assert!(from_zero == 0 || from_zero == mmas.len(), "a module mixing the two forms");
    from_zero == mmas.len()
}

#[test]
fn split_k_names_select_the_shape_and_everything_else_is_refused() {
    let dir = pinned::pinned_scratch("f16_splitk_names", pinned::SM_PINNED);

    let ptx = compile(&dir, "g_splitk4", 256, 256, 4096, true, None, None).unwrap();
    assert_eq!(entries(&ptx), ["g_splitk4_reduce", "g_splitk4"], "the reduction entry beside the main one");
    assert!(ptx.contains("K=4096 in 4 splits of K=1024 | grid ("), "the launch contract is stated");
    assert!(ptx.contains("%ctaid.z;"), "the split index");
    // 256 rows, splits of 1024: blocked, its block sums folded in the K
    // loop (the knob is the control).
    assert!(!accumulates_in_f32(&ptx) && blocked(&ptx), "a split of 256 rows and K = 1024 is not blocked");
    let f32_split = compile(&dir, "g_f32_splitk4", 256, 256, 4096, true, Some("f32"), None).unwrap();
    assert!(accumulates_in_f32(&f32_split) && !blocked(&f32_split), "Y_F16_ACCUMULATE=f32 does not add each mma in a split");
    // Where the block sums do not fit - a 128x128 tile of 2x2 warps, 128
    // accumulators a thread - a split adds each mma in float32.
    let wide = compile(&dir, "g_wide_splitk4", 256, 256, 4096, true, None, Some("128,128,32,2,2,3")).unwrap();
    assert!(wide.contains("| CTA 128x128x32 | 2x2 warps"), "not the tile this case checks");
    assert!(accumulates_in_f32(&wide) && !blocked(&wide), "a split of 128 accumulators a thread is not plain float32");
    // Splits no longer than a block (256 of K) add each mma in float32.
    let short = compile(&dir, "g_short_splitk4", 256, 256, 1024, true, None, None).unwrap();
    assert!(accumulates_in_f32(&short) && !blocked(&short), "a split of K = 256 is not plain float32");
    // Uncompensated at 32 rows or fewer too, where the unsplit GEMM's adds
    // are compensated (`sub.rn.f32` is the compensated fold's; the knob is
    // the control), and not blocked.
    let decode = compile(&dir, "d_splitk4", 8, 256, 4096, true, None, None).unwrap();
    assert!(accumulates_in_f32(&decode) && !decode.contains("sub.rn.f32"), "a split of 8 rows is not plain float32");
    let compensated = compile(&dir, "d_kahan_splitk4", 8, 256, 4096, true, Some("kahan"), None).unwrap();
    assert!(compensated.contains("sub.rn.f32"), "Y_F16_ACCUMULATE=kahan does not compensate a split");
    let chained = compile(&dir, "g_mma_splitk4", 256, 256, 4096, true, Some("mma"), None).unwrap();
    assert!(!accumulates_in_f32(&chained), "Y_F16_ACCUMULATE=mma still selects the chain");

    // The control: the same four parameters without the suffix are the
    // fused bias + ReLU GEMM, with one entry.
    let bias = compile(&dir, "g_bias", 256, 256, 4096, true, None, None).unwrap();
    assert_eq!(entries(&bias), ["g_bias"]);
    assert!(bias.contains("EPI_BIAS_RELU"), "the bias shape's epilogue");

    for (name, m, n, k, partial, why) in [
        ("g_splitk0", 256, 256, 4096, true, "must end in `_splitk<S>` with S from 1 to 64"),
        ("g_splitk04", 256, 256, 4096, true, "must end in `_splitk<S>` with S from 1 to 64"),
        ("g_splitk65", 256, 256, 4096, true, "must end in `_splitk<S>` with S from 1 to 64"),
        ("g_splitkx", 256, 256, 4096, true, "must end in `_splitk<S>` with S from 1 to 64"),
        ("g3_splitk4", 256, 256, 4096, false, "takes exactly (A, B: GlobalMemory<F16>, C, Partial: GlobalMemory<F32>)"),
        ("k_splitk3", 256, 256, 4096, true, "does not split into 3 parts that are each a multiple of 8"),
        ("k8_splitk4", 256, 256, 48, true, "does not split into 4 parts that are each a multiple of 8"),
        ("n_splitk2", 256, 258, 4096, true, "is not a multiple of 4"),
    ] {
        let err = compile(&dir, name, m, n, k, partial, None, None).expect_err(&format!("{name} compiled"));
        assert!(err.contains(why), "{name}: {err}");
    }
    // A FIVE-parameter split name is refused too (it would otherwise be read
    // as the FP8 shape's parameter list).
    let src = dir.join("g5_splitk2.ysu");
    std::fs::write(&src, "@tile(256, 256, 4096)\nkernel g5_splitk2(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>, P: GlobalMemory<F32>, Q: GlobalMemory<F32>) {\n}\n\nfn main() {\n}\n").unwrap();
    let err = compile_source(&dir, &src, None, None).expect_err("a 5-parameter split compiled");
    assert!(err.contains("takes exactly"), "{err}");
    // The smallest legal splits: one, and parts of exactly 8.
    compile(&dir, "s1_splitk1", 256, 256, 4096, true, None, None).unwrap();
    compile(&dir, "s8_splitk3", 256, 256, 24, true, None, None).unwrap();
    // The rule's form takes the same four parameters, and no split's shape
    // check where it does not split: K = 4104 (refused in 2, the control)
    // compiles unsplit.
    let err = compile(&dir, "a3_splitk", 256, 256, 4096, false, None, None).expect_err("a 3-parameter `_splitk` compiled");
    assert!(err.contains("takes exactly"), "{err}");
    let err = compile(&dir, "o_splitk2", 256, 256, 4104, true, None, None).expect_err("K = 4104 split in 2");
    assert!(err.contains("does not split into 2 parts"), "{err}");
    let odd = compile(&dir, "a_odd_splitk", 256, 256, 4104, true, None, None).unwrap();
    assert_eq!(splits_of(&odd), 1);
}

/// The split count a split-K module's header states ("in S split(s)").
fn splits_of(ptx: &str) -> u32 {
    let at = ptx.find("[Y SPLIT-K GEMM] K=").expect("no split-K header");
    let line = ptx[at..].lines().next().unwrap();
    line[line.find(" in ").unwrap() + 4..].split(' ').next().unwrap().parse().unwrap()
}

/// The text of entry `name` (its `.visible .entry` line to its closing brace).
fn entry_text<'a>(ptx: &'a str, name: &str) -> &'a str {
    let start = ptx.find(&format!(".visible .entry {name}(")).unwrap_or_else(|| panic!("no entry {name}"));
    let end = start + ptx[start..].find("\n}\n").expect("an entry's closing brace") + 3;
    &ptx[start..end]
}

/// `entry` without the lines `drop` matches and the `.reg` declarations,
/// every register renamed in order of first use, so two entries that differ
/// only by one register load compare equal operand for operand.
fn canonical(entry: &str, drop: impl Fn(&str) -> bool) -> String {
    let mut names = std::collections::HashMap::new();
    let mut out = String::new();
    for line in entry.lines().filter(|l| !drop(l) && !l.trim_start().starts_with(".reg ")) {
        let mut rest = line;
        while let Some(at) = rest.find('%') {
            out.push_str(&rest[..at]);
            let tail = &rest[at + 1..];
            let class: String = tail.chars().take_while(|c| c.is_ascii_alphabetic()).collect();
            let digits: String = tail[class.len()..].chars().take_while(|c| c.is_ascii_digit()).collect();
            if digits.is_empty() || !matches!(class.as_str(), "r" | "rd" | "f" | "p" | "h") {
                out.push('%');
                rest = tail;
                continue;
            }
            let next = names.keys().filter(|(c, _)| *c == class).count();
            let id = *names.entry((class.clone(), digits.clone())).or_insert(next);
            out.push_str(&format!("%{class}#{id}"));
            rest = &tail[class.len() + digits.len()..];
        }
        out.push_str(rest);
        out.push('\n');
    }
    out
}

/// The machine code `ptxas -arch=sm_89` gives `entry` of `ptx`, without
/// addresses; `None` when `ptxas` or `cuobjdump` is not installed.
fn sass_of(dir: &Path, ptx: &str, entry: &str) -> Option<String> {
    let src = dir.join(format!("{entry}_sass.ptx"));
    let cubin = src.with_extension("cubin");
    std::fs::write(&src, ptx).unwrap();
    let asm = Command::new("ptxas").arg("-arch=sm_89").arg(&src).arg("-o").arg(&cubin).output().ok()?;
    assert!(asm.status.success(), "ptxas: {}", String::from_utf8_lossy(&asm.stderr));
    let dump = Command::new("cuobjdump").arg("-sass").arg("-fun").arg(entry).arg(&cubin).output().ok()?;
    let text = String::from_utf8_lossy(&dump.stdout).to_string();
    let lines: Vec<String> = text
        .lines()
        .filter_map(|l| l.trim().strip_prefix("/*").and_then(|r| r.split_once("*/")).map(|(_, i)| i.split(';').next().unwrap().trim().to_string()))
        .filter(|i| !i.is_empty())
        .collect();
    assert!(lines.len() > 50, "{entry}: no SASS dumped");
    Some(lines.join("\n"))
}

#[test]
fn the_rules_kernel_is_the_pinned_split_or_else_the_plain_kernel() {
    let dir = pinned::pinned_scratch("f16_splitk_auto_twins", pinned::SM_PINNED);
    // 512 x 512 x 16384: 64 CTAs unsplit; the rule splits it in 2 on 64x128.
    let auto = compile(&dir, "a_splitk", 512, 512, 16384, true, None, None).unwrap();
    assert_eq!(entries(&auto), ["a_splitk_reduce", "a_splitk"]);
    assert_eq!(splits_of(&auto), 2);
    assert!(auto.contains("K=16384 in 2 splits of K=8192 | grid (4, 8, 2)"), "the launch is not stated");
    assert!(auto.contains("[Y AUTO SPLIT-K] S=2 by the split-K rule: 64 CTAs of 64x64 unsplit on 66 SMs; 2 splits of K=8192 on 64x128x32 tiles, 4x2 warps: 64 CTAs"));
    // The same module as `_splitk2` pinned to the rule's tile, but for its
    // name and the line saying why.
    let pinned2 = compile(&dir, "p_splitk2", 512, 512, 16384, true, None, Some("64,128,32,4,2,4")).unwrap();
    let without = |ptx: &str| ptx.lines().filter(|l| !l.contains("[Y AUTO SPLIT-K]")).collect::<Vec<_>>().join("\n");
    assert_eq!(without(&auto), without(&pinned2.replace("p_splitk2", "a_splitk")), "`_splitk` is not `_splitk2` on the rule's tile");
    // The control: the analytic tile `_splitk2` takes unpinned is not it.
    let analytic2 = compile(&dir, "q_splitk2", 512, 512, 16384, true, None, None).unwrap();
    assert_ne!(without(&auto), without(&analytic2.replace("q_splitk2", "a_splitk")));

    // 512 x 512 x 4096: splits of 2048 at most, so not split.
    let unsplit = compile(&dir, "u_splitk", 512, 512, 4096, true, None, None).unwrap();
    let plain = compile(&dir, "u", 512, 512, 4096, false, None, None).unwrap();
    assert_eq!(splits_of(&unsplit), 1);
    assert!(unsplit.contains("K=4096 in 1 split (not split) | grid (8, 8, 1) writes C; Partial is neither read nor written"));
    // Its reduction does nothing: no load, no store, one `ret`.
    let reduce = entry_text(&unsplit, "u_splitk_reduce");
    assert!(!reduce.contains("ld.") && !reduce.contains("st.") && reduce.matches("ret;").count() == 1, "{reduce}");
    // Its main entry is the plain kernel's operand for operand, but for the
    // unused Partial's parameter and load; C is the third parameter.
    // (C's parameter line then ends the list, without its comma.)
    let ours = canonical(entry_text(&unsplit, "u_splitk"), |l| l.contains("P_3") || l.contains("[Y SPLIT-K GEMM]") || l.contains("[Y AUTO SPLIT-K]") || l.contains("Estimated registers"))
        .replace("    .param .u64 C_2,\n)", "    .param .u64 C_2\n)");
    let theirs = canonical(entry_text(&plain, "u"), |l| l.contains("Estimated registers"));
    assert_eq!(ours.replace("u_splitk", "u"), theirs, "the unsplit `_splitk` is not the plain kernel");
    assert!(!ours.contains("P_3") && entry_text(&unsplit, "u_splitk").matches("P_3").count() == 2, "Partial is used beyond its declaration and load");
    // The control: the same comparison with the plain kernel's C swapped for
    // its A tells them apart.
    let swapped = canonical(&entry_text(&plain, "u").replace("[C_2]", "[A_0]"), |l| l.contains("Estimated registers"));
    assert_ne!(ours.replace("u_splitk", "u"), swapped);
    // And ptxas makes the same machine code of both.
    match (sass_of(&dir, &unsplit, "u_splitk"), sass_of(&dir, &plain, "u")) {
        (Some(a), Some(b)) => assert_eq!(a, b, "different SASS"),
        _ => eprintln!("SKIP: ptxas or cuobjdump not found - the SASS was not compared."),
    }
}

#[test]
fn the_rules_tile_yields_only_to_a_measurement_of_its_split() {
    let dir = pinned::pinned_scratch("f16_splitk_auto_tile", pinned::SM_PINNED);
    // A measurement of a NEARBY shape in 2 splits: `_splitk2` borrows it,
    // `_splitk` keeps the rule's tile.
    let profile = dir.join(".ysu_hw_profile");
    let base = std::fs::read_to_string(&profile).unwrap();
    std::fs::write(&profile, format!("{base}AUTOTUNE_F16_SPLITK2_512x512x32768_{}=64,64,32,2,2,3\n", pinned::GPU_NAME)).unwrap();
    let auto = compile(&dir, "a_splitk", 512, 512, 16384, true, None, None).unwrap();
    assert!(auto.contains("| CTA 64x128x32 | 4x2 warps |"), "the rule's tile gave way to a borrowed one");
    let borrow = compile(&dir, "b_splitk2", 512, 512, 16384, true, None, None).unwrap();
    assert!(borrow.contains("| CTA 64x64x32 | 2x2 warps |"), "the control: `_splitk2` did not borrow the nearby measurement");
    // A measurement of this shape in 2 splits is taken, and said so.
    std::fs::write(&profile, format!("{base}AUTOTUNE_F16_SPLITK2_512x512x16384_{}=64,128,32,2,4,4\n", pinned::GPU_NAME)).unwrap();
    let measured = compile(&dir, "m_splitk", 512, 512, 16384, true, None, None).unwrap();
    assert!(measured.contains("| CTA 64x128x32 | 2x4 warps |"), "a measurement of the split did not win");
    assert!(measured.contains("(this kernel takes the measured or overriding tile 64x128x32 2x4 s4 instead)"));
    // So is an override.
    let over = compile(&dir, "o_splitk", 512, 512, 16384, true, None, Some("64,64,32,4,2,4")).unwrap();
    assert!(over.contains("| CTA 64x64x32 | 4x2 warps |") && over.contains("overriding tile 64x64x32 4x2 s4 instead"));

    // One module holding both forms at one shape, in both orders: each takes
    // its own tile (the cache of tiles resolved in one compile keeps them apart).
    std::fs::write(&profile, &base).unwrap();
    for (first, second) in [("x_splitk2", "y_splitk"), ("y_splitk", "x_splitk2")] {
        let src = dir.join(format!("both_{first}.ysu"));
        let kernel = |name: &str| format!("@tile(512, 512, 16384)\nkernel {name}(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>, P: GlobalMemory<F32>) {{\n}}\n\n");
        std::fs::write(&src, format!("{}{}fn main() {{\n}}\n", kernel(first), kernel(second))).unwrap();
        let ptx = compile_source(&dir, &src, None, None).unwrap();
        let tile_of = |name: &str| launch_of(entry_text(&ptx, name));
        assert_eq!(tile_of("y_splitk").1, 128, "{first} first: `_splitk` did not take the rule's tile");
        assert_eq!(tile_of("x_splitk2").1, 64, "{first} first: `_splitk2` took the rule's tile");
    }
}

#[test]
fn a_plain_gemm_is_told_the_rules_split_and_keeps_its_launch() {
    let dir = pinned::pinned_scratch("f16_splitk_auto_note", pinned::SM_PINNED);
    let run = |name: &str, k: u32| {
        let src = dir.join(format!("{name}.ysu"));
        let bias = if name.contains("bias") { ", Bias: GlobalMemory<F32>" } else { "" };
        std::fs::write(&src, format!("@tile(512, 512, {k})\nkernel {name}(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>{bias}) {{\n}}\n\nfn main() {{\n}}\n")).unwrap();
        let out = Command::new(env!("CARGO_BIN_EXE_Y")).arg(&src).arg("--emit-ptx").current_dir(&dir).env_remove("Y_CTA_OVERRIDE").env_remove("Y_F16_ACCUMULATE").output().unwrap();
        assert!(out.status.success());
        (String::from_utf8_lossy(&out.stdout).to_string(), std::fs::read_to_string(src.with_extension("ptx")).unwrap())
    };
    let (said, ptx) = run("g", 16384);
    assert!(said.contains("[Y split-K rule] `g` would split K 2 ways: 64 CTAs of 64x64 unsplit on 66 SMs; 2 splits of K=8192 on 64x128x32 tiles, 4x2 warps: 64 CTAs. Named `g_splitk`, with a fourth parameter Partial: GlobalMemory<F32> of 524288 floats (2.1 MB)"), "{said}");
    assert_eq!(entries(&ptx), ["g"], "a plain GEMM grew an entry");
    assert!(!ptx.contains("SPLIT-K") && !ptx.contains("%ctaid.z"), "a plain GEMM was split");
    let (said, _) = run("h", 4096);
    assert!(!said.contains("[Y split-K rule]"), "a GEMM the rule does not split was told it would: {said}");
    // The fused bias + ReLU GEMM has no split form, so it is told nothing.
    let (said, _) = run("g_bias", 16384);
    assert!(!said.contains("[Y split-K rule]"), "the bias GEMM was offered a split: {said}");
}

/// Launches a `_splitk` module as its header states: the main entry on
/// (N/cn, M/cm, S) and then `_reduce` on its grid, from 0xFF-filled C and
/// Partial; C and Partial afterwards.
#[allow(clippy::too_many_arguments)]
fn run_auto(ctx: &y::cuda_runtime::CudaContext, ptx: &str, entry: &str, m: u32, n: u32, k: u32, a: &[u16], b: &[u16]) -> (Vec<f32>, Vec<u8>) {
    let s = splits_of(ptx);
    let (cm, cn, threads, smem) = launch_of(ptx);
    let main = ctx.load_ptx(ptx, entry).unwrap();
    if smem > 48 * 1024 {
        main.set_max_dynamic_smem(smem).unwrap();
    }
    let red = ctx.load_ptx(ptx, &format!("{entry}_reduce")).unwrap();
    let (mu, nu, ku) = (m as usize, n as usize, k as usize);
    let d_a = ctx.alloc(mu * ku * 2).unwrap();
    let d_b = ctx.alloc(ku * nu * 2).unwrap();
    let d_c = ctx.alloc(mu * nu * 4).unwrap();
    let d_p = ctx.alloc(s as usize * mu * nu * 4).unwrap();
    ctx.memcpy_htod_at(&d_a, 0, &bytes_of_u16(a)).unwrap();
    ctx.memcpy_htod_at(&d_b, 0, &bytes_of_u16(b)).unwrap();
    ctx.memset_u8(&d_c, 0xFF).unwrap();
    ctx.memset_u8(&d_p, 0xFF).unwrap();
    let args = [d_a.device_ptr(), d_b.device_ptr(), d_c.device_ptr(), d_p.device_ptr()];
    ctx.launch(&main, (n.div_ceil(cn), m.div_ceil(cm), s), (threads, 1, 1), smem, &args).unwrap();
    ctx.launch(&red, ((m * n / 4).div_ceil(256), 1, 1), (256, 1, 1), 0, &args).unwrap();
    ctx.synchronize().unwrap();
    let mut c = vec![0u8; mu * nu * 4];
    ctx.memcpy_dtoh_at(&mut c, &d_c, 0).unwrap();
    let mut p = vec![0u8; s as usize * mu * nu * 4];
    ctx.memcpy_dtoh_at(&mut p, &d_p, 0).unwrap();
    (f32s(&c), p)
}

/// The pipeline depth the module's header states.
fn stages_of(ptx: &str) -> u32 {
    let key = "Autotuner selected ";
    let at = ptx.find(key).expect("no pipeline depth") + key.len();
    ptx[at..].split(' ').next().unwrap().parse().unwrap()
}

/// The checks of a split module's run on (a, b), C `c` and Partial bytes
/// `p` (`run_auto`): each slot equal, bit for bit, to the unsplit kernel under
/// `Y_F16_ACCUMULATE=knob` on that slot's K range on the same tile, and C the
/// slots summed in order.
#[allow(clippy::too_many_arguments)]
fn assert_slots_are_the_slices(ctx: &y::cuda_runtime::CudaContext, dir: &Path, auto: &str, name: &str, (m, n, k): (u32, u32, u32), (a, b): (&[u16], &[u16]), (c, p): (&[f32], &[u8]), knob: &str) {
    let (mu, nu, ku) = (m as usize, n as usize, k as usize);
    let s = splits_of(auto);
    let (cm, cn, threads, smem) = launch_of(auto);
    let ks = ku / s as usize;
    let warps = &auto[auto.find("| CTA ").unwrap()..];
    let w: Vec<u32> = warps[..warps.find(" warps").unwrap()].rsplit("| ").next().unwrap().split('x').map(|x| x.trim().parse().unwrap()).collect();
    let tile = format!("{cm},{cn},32,{},{},{}", w[0], w[1], stages_of(&auto));
    let slice_name = format!("u{m}x{n}x{ks}");
    let slice = compile(dir, &slice_name, m, n, ks as u32, false, Some(knob), Some(&tile)).unwrap();
    assert_eq!(launch_of(&slice), (cm, cn, threads, smem), "{name}: the slice kernel's launch differs");
    let module = ctx.load_ptx(&slice, &slice_name).unwrap();
    if smem > 48 * 1024 {
        module.set_max_dynamic_smem(smem).unwrap();
    }
    let p = f32s(p);
    let (d_sa, d_sb, d_sc) = (ctx.alloc(mu * ks * 2).unwrap(), ctx.alloc(ks * nu * 2).unwrap(), ctx.alloc(mu * nu * 4).unwrap());
    let mut wrong_slots = Vec::new();
    for z in 0..s as usize {
        let sa: Vec<u16> = (0..mu * ks).map(|x| a[(x / ks) * ku + z * ks + x % ks]).collect();
        ctx.memcpy_htod_at(&d_sa, 0, &bytes_of_u16(&sa)).unwrap();
        ctx.memcpy_htod_at(&d_sb, 0, &bytes_of_u16(&b[z * ks * nu..(z + 1) * ks * nu])).unwrap();
        ctx.memset_u8(&d_sc, 0xFF).unwrap();
        ctx.launch(&module, (n.div_ceil(cn), m.div_ceil(cm), 1), (threads, 1, 1), smem, &[d_sa.device_ptr(), d_sb.device_ptr(), d_sc.device_ptr()]).unwrap();
        ctx.synchronize().unwrap();
        let mut raw = vec![0u8; mu * nu * 4];
        ctx.memcpy_dtoh_at(&mut raw, &d_sc, 0).unwrap();
        if f32s(&raw).iter().zip(&p[z * mu * nu..(z + 1) * mu * nu]).any(|(x, y)| x.to_bits() != y.to_bits()) {
            wrong_slots.push(z);
        }
    }
    assert!(wrong_slots.is_empty(), "{name}: slots {wrong_slots:?} differ from the unsplit kernel on their K range");
    let in_order: Vec<f32> = (0..mu * nu).map(|x| (1..s as usize).fold(p[x], |t, z| t + p[z * mu * nu + x])).collect();
    assert!(c.iter().zip(&in_order).all(|(x, y)| x.to_bits() == y.to_bits()), "{name}: C is not the in-order sum of the slots");
}

#[test]
fn the_rules_kernels_run_as_their_headers_state_on_the_gpu() {
    use y::cuda_runtime::CudaContext;
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver - no `_splitk` GEMM was executed.");
        return;
    };
    let dir = pinned::pinned_scratch("f16_splitk_auto_gpu", pinned::SM_PINNED);
    let mut state = 0xA_0705_F16u64;
    // A shape the rule splits, and one it does not.
    for (m, n, k, split) in [(256u32, 256u32, 16384u32, true), (256, 256, 4096, false)] {
        let (mu, nu, ku) = (m as usize, n as usize, k as usize);
        let name = format!("a{m}x{n}x{k}_splitk");
        let auto = compile(&dir, &name, m, n, k, true, None, None).unwrap();
        let s = splits_of(&auto);
        assert_eq!(s > 1, split, "{name}: S = {s}");

        // 1. Small integers: exact.
        let ia: Vec<i32> = (0..mu * ku).map(|_| (splitmix(&mut state) % 9) as i32 - 4).collect();
        let ib: Vec<i32> = (0..ku * nu).map(|_| (splitmix(&mut state) % 9) as i32 - 4).collect();
        let iref: Vec<f32> = (0..mu * nu)
            .map(|x| (0..ku).map(|kk| (ia[(x / nu) * ku + kk] * ib[kk * nu + x % nu]) as i64).sum::<i64>() as f32)
            .collect();
        let (ha, hb): (Vec<u16>, Vec<u16>) = (ia.iter().map(|&v| f16_of_int(v)).collect(), ib.iter().map(|&v| f16_of_int(v)).collect());
        let (c, _) = run_auto(&ctx, &auto, &name, m, n, k, &ha, &hb);
        let wrong = c.iter().zip(&iref).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        assert_eq!(wrong, 0, "{name}: {wrong} integer products wrong");

        // Same-sign f16 in [0.25, 2).
        let a: Vec<u16> = (0..mu * ku).map(|_| { let r = splitmix(&mut state); ((13 + r % 2) << 10) as u16 | ((r >> 8) & 0x3FF) as u16 }).collect();
        let b: Vec<u16> = (0..ku * nu).map(|_| { let r = splitmix(&mut state); ((13 + r % 2) << 10) as u16 | ((r >> 8) & 0x3FF) as u16 }).collect();
        let (c, p) = run_auto(&ctx, &auto, &name, m, n, k, &a, &b);
        let (cm, cn, threads, smem) = launch_of(&auto);
        if split {
            // 2. Each slot is the unsplit blocked kernel on its K range, on the
            // same tile; and C their in-order sum.
            assert_slots_are_the_slices(&ctx, &dir, &auto, &name, (m, n, k), (&a, &b), (&c, &p), "blocked");
        } else {
            // 2. C is the plain kernel's, bit for bit, and Partial untouched.
            let plain_name = format!("p{m}x{n}x{k}");
            let plain = compile(&dir, &plain_name, m, n, k, false, None, None).unwrap();
            assert_eq!(launch_of(&plain), (cm, cn, threads, smem));
            let module = ctx.load_ptx(&plain, &plain_name).unwrap();
            let (d_a, d_b, d_c) = (ctx.alloc(mu * ku * 2).unwrap(), ctx.alloc(ku * nu * 2).unwrap(), ctx.alloc(mu * nu * 4).unwrap());
            ctx.memcpy_htod_at(&d_a, 0, &bytes_of_u16(&a)).unwrap();
            ctx.memcpy_htod_at(&d_b, 0, &bytes_of_u16(&b)).unwrap();
            ctx.launch(&module, (n.div_ceil(cn), m.div_ceil(cm), 1), (threads, 1, 1), smem, &[d_a.device_ptr(), d_b.device_ptr(), d_c.device_ptr()]).unwrap();
            ctx.synchronize().unwrap();
            let mut raw = vec![0u8; mu * nu * 4];
            ctx.memcpy_dtoh_at(&mut raw, &d_c, 0).unwrap();
            assert!(f32s(&raw).iter().zip(&c).all(|(x, y)| x.to_bits() == y.to_bits()), "{name}: C is not the plain kernel's");
            assert!(p.iter().all(|&byte| byte == 0xFF), "{name}: Partial was written");
        }
        // 3. The same bits on every launch.
        for _ in 0..3 {
            let (again, _) = run_auto(&ctx, &auto, &name, m, n, k, &a, &b);
            assert!(again.iter().zip(&c).all(|(x, y)| x.to_bits() == y.to_bits()), "{name}: a relaunch changed C");
        }
    }
}

#[test]
fn a_split_kernel_takes_its_own_measured_tile() {
    // Measured tiles are keyed by the split count: the profile's
    // `F16_SPLITK4` entry for the whole K, not the unsplit kernel's entry for
    // one split's depth.
    let dir = pinned::pinned_scratch("f16_splitk_key", pinned::SM_PINNED);
    let profile = dir.join(".ysu_hw_profile");
    let mut text = std::fs::read_to_string(&profile).unwrap();
    text.push_str(&format!("AUTOTUNE_F16_SPLITK4_256x256x4096_{0}=32,64,32,1,2,3\nAUTOTUNE_F16_256x256x1024_{0}=64,64,32,2,2,3\n", pinned::GPU_NAME));
    std::fs::write(&profile, text).unwrap();
    let split = compile(&dir, "t_splitk4", 256, 256, 4096, true, None, None).unwrap();
    assert!(split.contains("| CTA 32x64x32 | 1x2 warps |"), "the split kernel did not take its SPLITK4 entry");
    let unsplit = compile(&dir, "t_unsplit", 256, 256, 1024, false, None, None).unwrap();
    assert!(unsplit.contains("| CTA 64x64x32 | 2x2 warps |"), "the control: the unsplit entry is read");
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn f16_value(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1F) as i32;
    let f = (h & 0x3FF) as f64;
    assert!(e > 0 && e < 31, "normal values only");
    sign * (1.0 + f / 1024.0) * 2f64.powi(e - 15)
}

fn f16_of_int(v: i32) -> u16 {
    if v == 0 {
        return 0;
    }
    let (sign, a) = (if v < 0 { 0x8000u16 } else { 0 }, v.unsigned_abs());
    let e = 31 - a.leading_zeros();
    sign | (((e + 15) as u16) << 10) | (((a << (10 - e)) & 0x3FF) as u16)
}

/// The CTA tile, warp grid and dynamic shared memory the module's header states.
fn launch_of(ptx: &str) -> (u32, u32, u32, u32) {
    let field = |key: &str| -> u32 {
        let at = ptx.find(key).unwrap_or_else(|| panic!("no {key:?}")) + key.len();
        ptx[at..].chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap()
    };
    let cm = field("| CTA ");
    let cn = field(&format!("| CTA {cm}x"));
    let warps_at = ptx.find(" warps").unwrap();
    let w: Vec<u32> = ptx[..warps_at].rsplit("| ").next().unwrap().split('x').map(|x| x.trim().parse().unwrap()).collect();
    let smem = ptx.find("Dynamic shared memory required: ").map(|_| field("Dynamic shared memory required: ")).unwrap_or(0);
    (cm, cn, w[0] * w[1] * 32, smem)
}

fn bytes_of_u16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f32s(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

#[test]
fn split_k_is_exact_sums_its_slots_in_order_and_repeats_on_the_gpu() {
    use y::cuda_runtime::CudaContext;
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver - no split-K GEMM was executed.");
        return;
    };
    let dir = pinned::pinned_scratch("f16_splitk_gpu", pinned::SM_PINNED);
    let mut state = 0x5_917_F16u64;
    // A square shape, and a ragged one: M and N past whole tiles, and each
    // split's 200 deep, not a whole number of 32-deep K-tiles.
    for (m, n, k, s, tile) in [(128u32, 128u32, 4096u32, 4u32, "64,64,32,2,2,3"), (72, 136, 1000, 5, "32,32,32,2,1,3")] {
        let (mu, nu, ku, su, ks) = (m as usize, n as usize, k as usize, s as usize, (k / s) as usize);
        let name = format!("g{m}x{n}x{k}_splitk{s}");
        let split = compile(&dir, &name, m, n, k, true, None, Some(tile)).unwrap();
        let chained = compile(&dir, &format!("c{m}x{n}x{k}_splitk{s}"), m, n, k, true, Some("mma"), Some(tile)).unwrap();
        // The square shape's splits (1024 deep) are blocked by default, the
        // ragged one's (200) float32.
        let mode = if blocked(&split) { "blocked" } else { "f32" };
        assert_eq!(mode, if k / s > 256 { "blocked" } else { "f32" }, "{name}: not the default this case checks");
        // The unsplit kernel of one split's depth in the same mode, same tile.
        let slice_name = format!("u{m}x{n}x{}", k / s);
        let slice = compile(&dir, &slice_name, m, n, k / s, false, Some(mode), Some(tile)).unwrap();
        assert_eq!(launch_of(&split), launch_of(&slice), "{name}: the slice kernel's tile differs");

        let d_a = ctx.alloc(mu * ku * 2).unwrap();
        let d_b = ctx.alloc(ku * nu * 2).unwrap();
        let d_c = ctx.alloc(mu * nu * 4).unwrap();
        let d_p = ctx.alloc(su * mu * nu * 4).unwrap();
        let d_sa = ctx.alloc(mu * ks * 2).unwrap();
        let d_sb = ctx.alloc(ks * nu * 2).unwrap();
        let d_sc = ctx.alloc(mu * nu * 4).unwrap();
        let load = |ptx: &str, entry: &str| {
            let module = ctx.load_ptx(ptx, entry).unwrap();
            let (.., smem) = launch_of(ptx);
            if smem > 48 * 1024 && !entry.ends_with("_reduce") {
                module.set_max_dynamic_smem(smem).unwrap();
            }
            module
        };
        // Runs a split module on (a, b): C and every slot, from 0xFF-filled buffers.
        let run_split = |ptx: &str, entry: &str, a: &[u16], b: &[u16]| -> (Vec<f32>, Vec<f32>) {
            let (main, red) = (load(ptx, entry), load(ptx, &format!("{entry}_reduce")));
            let (cm, cn, threads, smem) = launch_of(ptx);
            ctx.memcpy_htod_at(&d_a, 0, &bytes_of_u16(a)).unwrap();
            ctx.memcpy_htod_at(&d_b, 0, &bytes_of_u16(b)).unwrap();
            ctx.memset_u8(&d_c, 0xFF).unwrap();
            ctx.memset_u8(&d_p, 0xFF).unwrap();
            let args = [d_a.device_ptr(), d_b.device_ptr(), d_c.device_ptr(), d_p.device_ptr()];
            ctx.launch(&main, (n.div_ceil(cn), m.div_ceil(cm), s), (threads, 1, 1), smem, &args).unwrap();
            ctx.launch(&red, ((m * n / 4).div_ceil(256), 1, 1), (256, 1, 1), 0, &args).unwrap();
            ctx.synchronize().unwrap();
            let mut c = vec![0u8; mu * nu * 4];
            ctx.memcpy_dtoh_at(&mut c, &d_c, 0).unwrap();
            let mut p = vec![0u8; su * mu * nu * 4];
            ctx.memcpy_dtoh_at(&mut p, &d_p, 0).unwrap();
            (f32s(&c), f32s(&p))
        };

        // 1. Small integers: every product and partial sum is exact.
        let ia: Vec<i32> = (0..mu * ku).map(|_| (splitmix(&mut state) % 9) as i32 - 4).collect();
        let ib: Vec<i32> = (0..ku * nu).map(|_| (splitmix(&mut state) % 9) as i32 - 4).collect();
        let iref: Vec<f32> = (0..mu * nu)
            .map(|x| (0..ku).map(|kk| (ia[(x / nu) * ku + kk] * ib[kk * nu + x % nu]) as i64).sum::<i64>() as f32)
            .collect();
        let (ha, hb): (Vec<u16>, Vec<u16>) = (ia.iter().map(|&v| f16_of_int(v)).collect(), ib.iter().map(|&v| f16_of_int(v)).collect());
        for (ptx, entry) in [(&split, name.clone()), (&chained, format!("c{m}x{n}x{k}_splitk{s}"))] {
            let (c, _) = run_split(ptx, &entry, &ha, &hb);
            let wrong = c.iter().zip(&iref).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            assert_eq!(wrong, 0, "{entry}: {wrong} integer products wrong");
        }

        // Same-sign f16 in [0.25, 2): no cancellation hides an error.
        let a: Vec<u16> = (0..mu * ku).map(|_| { let r = splitmix(&mut state); ((13 + r % 2) << 10) as u16 | ((r >> 8) & 0x3FF) as u16 }).collect();
        let b: Vec<u16> = (0..ku * nu).map(|_| { let r = splitmix(&mut state); ((13 + r % 2) << 10) as u16 | ((r >> 8) & 0x3FF) as u16 }).collect();
        let (c, p) = run_split(&split, &name, &a, &b);

        // 2. Slot z is the unsplit kernel on K range z, bit for bit.
        let slice_mod = load(&slice, &slice_name);
        let (cm, cn, threads, smem) = launch_of(&slice);
        let mut wrong_slots = Vec::new();
        let mut slices = Vec::new();
        for z in 0..su {
            let sa: Vec<u16> = (0..mu * ks).map(|x| a[(x / ks) * ku + z * ks + x % ks]).collect();
            let sb: Vec<u16> = b[z * ks * nu..(z + 1) * ks * nu].to_vec();
            ctx.memcpy_htod_at(&d_sa, 0, &bytes_of_u16(&sa)).unwrap();
            ctx.memcpy_htod_at(&d_sb, 0, &bytes_of_u16(&sb)).unwrap();
            ctx.memset_u8(&d_sc, 0xFF).unwrap();
            ctx.launch(&slice_mod, (n.div_ceil(cn), m.div_ceil(cm), 1), (threads, 1, 1), smem, &[d_sa.device_ptr(), d_sb.device_ptr(), d_sc.device_ptr()]).unwrap();
            ctx.synchronize().unwrap();
            let mut raw = vec![0u8; mu * nu * 4];
            ctx.memcpy_dtoh_at(&mut raw, &d_sc, 0).unwrap();
            let got = f32s(&raw);
            if got.iter().zip(&p[z * mu * nu..(z + 1) * mu * nu]).any(|(x, y)| x.to_bits() != y.to_bits()) {
                wrong_slots.push(z);
            }
            slices.push(got);
        }
        assert!(wrong_slots.is_empty(), "{name}: slots {wrong_slots:?} differ from the unsplit kernel on their K range");
        // The control: slot 0 is not slice 1's.
        assert!(slices[1].iter().zip(&p[..mu * nu]).any(|(x, y)| x.to_bits() != y.to_bits()), "{name}: slot 0 equals slice 1");

        // 3. C is the slots added in order, z = 0, 1, ..., in float32.
        let in_order: Vec<f32> = (0..mu * nu).map(|x| (1..su).fold(p[x], |t, z| t + p[z * mu * nu + x])).collect();
        let reversed: Vec<f32> = (0..mu * nu).map(|x| (0..su - 1).rev().fold(p[(su - 1) * mu * nu + x], |t, z| t + p[z * mu * nu + x])).collect();
        let off = c.iter().zip(&in_order).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        assert_eq!(off, 0, "{name}: {off} outputs are not the in-order sum of the slots");
        let rev = c.iter().zip(&reversed).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        assert!(rev > 0, "{name}: the reverse order gave the same bits everywhere - the order check checks nothing");

        // 4. The same bits on every launch - and through the timer the tuner
        // uses, which must run the reduction too (C is refilled first).
        for _ in 0..3 {
            let (again, _) = run_split(&split, &name, &a, &b);
            assert!(again.iter().zip(&c).all(|(x, y)| x.to_bits() == y.to_bits()), "{name}: a relaunch changed C");
        }
        let (main, red) = (load(&split, &name), load(&split, &format!("{name}_reduce")));
        let (scm, scn, sthreads, ssmem) = launch_of(&split);
        ctx.memset_u8(&d_c, 0xFF).unwrap();
        let args = vec![d_a.device_ptr(), d_b.device_ptr(), d_c.device_ptr(), d_p.device_ptr()];
        let us = ctx
            .time_launch_sequence(
                &[(&main, (n.div_ceil(scn), m.div_ceil(scm), s), (sthreads, 1, 1), ssmem), (&red, ((m * n / 4).div_ceil(256), 1, 1), (256, 1, 1), 0)],
                &[args],
                2,
            )
            .unwrap();
        assert!(us > 0.0);
        let mut raw = vec![0u8; mu * nu * 4];
        ctx.memcpy_dtoh_at(&mut raw, &d_c, 0).unwrap();
        assert!(f32s(&raw).iter().zip(&c).all(|(x, y)| x.to_bits() == y.to_bits()), "{name}: the timed sequence did not leave C as the reduction writes it");

        // 5. Same-sign error: float32 accumulation at float32's level, blocked
        // sums a fraction of the chained splits' (the control) bias low.
        if k / s >= 1024 {
            let exact: Vec<f64> = (0..mu * nu)
                .map(|x| (0..ku).map(|kk| f16_value(a[(x / nu) * ku + kk]) * f16_value(b[kk * nu + x % nu])).sum())
                .collect();
            let mean_rel = |got: &[f32]| got.iter().zip(&exact).map(|(&g, &e)| (g as f64 - e) / e).sum::<f64>() / (mu * nu) as f64;
            let (cc, _) = run_split(&chained, &format!("c{m}x{n}x{k}_splitk{s}"), &a, &b);
            let (split_err, chained_err) = (mean_rel(&c), mean_rel(&cc));
            assert!(chained_err < -5e-6, "{name}: the chained splits are off by only {chained_err:e} on average");
            assert_eq!(mode, "blocked");
            // Blocks of 256 chain 16 mmas, a split of 1024 64: measured
            // -2.6e-6 against the chain's -1.2e-5.
            assert!(split_err < 0.0 && split_err.abs() < chained_err.abs() / 3.0, "{name}: blocked splits are off by {split_err:e} on average, the chained by {chained_err:e}");
            let f32_name = format!("f{m}x{n}x{k}_splitk{s}");
            let plain = compile(&dir, &f32_name, m, n, k, true, Some("f32"), Some(tile)).unwrap();
            let (cf, _) = run_split(&plain, &f32_name, &a, &b);
            let plain_err = mean_rel(&cf);
            assert!(plain_err.abs() < 1e-6, "{name}: float32 accumulation is off by {plain_err:e} on average");
        }
    }
}

#[test]
fn the_tuner_measures_a_split_kernel_with_its_reduction_on_the_gpu() {
    use y::autotuner::AutotuneCandidate;
    use y::cuda_runtime::CudaContext;
    if CudaContext::new().is_none() {
        eprintln!("SKIP: no CUDA driver - the split-K tuner was not run.");
        return;
    }
    let hw = y::sentinel::HardwareProfile::default();
    let cand = |cta_m, cta_n, cta_k, warps_m, warps_n, num_stages| AutotuneCandidate {
        cta_m, cta_n, cta_k, warps_m, warps_n, num_stages, num_warps: warps_m * warps_n,
    };
    let candidates = [cand(64, 64, 32, 2, 2, 3), cand(32, 64, 32, 1, 2, 3)];
    // Every candidate must pass the tuner's correctness gate, which reads C:
    // without the reduction C stays zero and every candidate is rejected.
    let ranked = y::empirical_autotune::tune_gemm_f16_split(256, 256, 2048, 4, &hw, &candidates, false)
        .unwrap_or_else(|e| panic!("the split-K tuner failed: {e}"));
    assert_eq!(ranked.len(), 2, "both candidates were compiled, verified and timed");
    assert!(ranked.iter().all(|r| r.timing.best_us > 0.0 && r.timing.best_us.is_finite()));
}

#[test]
fn compensated_splits_are_the_compensated_kernel_on_each_slot_on_the_gpu() {
    use y::cuda_runtime::CudaContext;
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver - no compensated split was executed.");
        return;
    };
    let dir = pinned::pinned_scratch("f16_splitk_bk_gpu", pinned::SM_PINNED);
    let mut state = 0xB10C_5917u64;
    let (m, n, k) = (256u32, 256u32, 16384u32);
    let (mu, nu, ku) = (m as usize, n as usize, k as usize);
    let name = "c256_splitk";
    let auto = compile(&dir, name, m, n, k, true, Some("blocked_kahan"), None).unwrap();
    assert_eq!(splits_of(&auto), 4);
    assert!(auto.contains("sub.rn.f32"), "the split is not compensated");
    let ia: Vec<i32> = (0..mu * ku).map(|_| (splitmix(&mut state) % 9) as i32 - 4).collect();
    let ib: Vec<i32> = (0..ku * nu).map(|_| (splitmix(&mut state) % 9) as i32 - 4).collect();
    let iref: Vec<f32> = (0..mu * nu)
        .map(|x| (0..ku).map(|kk| (ia[(x / nu) * ku + kk] * ib[kk * nu + x % nu]) as i64).sum::<i64>() as f32)
        .collect();
    let (ha, hb): (Vec<u16>, Vec<u16>) = (ia.iter().map(|&v| f16_of_int(v)).collect(), ib.iter().map(|&v| f16_of_int(v)).collect());
    let (c, _) = run_auto(&ctx, &auto, name, m, n, k, &ha, &hb);
    let wrong = c.iter().zip(&iref).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    assert_eq!(wrong, 0, "{name}: {wrong} integer products wrong");
    let a: Vec<u16> = (0..mu * ku).map(|_| { let r = splitmix(&mut state); ((13 + r % 2) << 10) as u16 | ((r >> 8) & 0x3FF) as u16 }).collect();
    let b: Vec<u16> = (0..ku * nu).map(|_| { let r = splitmix(&mut state); ((13 + r % 2) << 10) as u16 | ((r >> 8) & 0x3FF) as u16 }).collect();
    let (c, p) = run_auto(&ctx, &auto, name, m, n, k, &a, &b);
    assert_slots_are_the_slices(&ctx, &dir, &auto, name, (m, n, k), (&a, &b), (&c, &p), "blocked_kahan");
    // The control: the plain blocked split's C differs.
    let plain = compile(&dir, "p256_splitk", m, n, k, true, None, None).unwrap();
    let (cp, _) = run_auto(&ctx, &plain, "p256_splitk", m, n, k, &a, &b);
    assert!(cp.iter().zip(&c).any(|(x, y)| x.to_bits() != y.to_bits()), "compensation changed no output");
}

/// `compile` with `envs` set and `args` after `--emit-ptx` (`Y_AUTOTUNE`,
/// `Y_F16_ACCUMULATE` and `Y_CTA_OVERRIDE` unset otherwise): the module, if
/// one was written, and the compiler's output.
#[allow(clippy::too_many_arguments)]
fn compile_with(dir: &Path, name: &str, m: u32, n: u32, k: u32, partial: bool, envs: &[(&str, &str)], args: &[&str]) -> (Option<String>, String) {
    let src = dir.join(format!("{name}.ysu"));
    let p = if partial { ", P: GlobalMemory<F32>" } else { "" };
    std::fs::write(&src, format!("@tile({m}, {n}, {k})\nkernel {name}(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>{p}) {{\n}}\n\nfn main() {{\n}}\n")).unwrap();
    let ptx = src.with_extension("ptx");
    let _ = std::fs::remove_file(&ptx);
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_Y"));
    cmd.arg(&src).arg("--emit-ptx").args(args).current_dir(dir);
    for var in ["Y_AUTOTUNE", "Y_F16_ACCUMULATE", "Y_CTA_OVERRIDE"] {
        cmd.env_remove(var);
    }
    cmd.envs(envs.iter().copied());
    let out = cmd.output().expect("run Y");
    let said = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    (out.status.success().then(|| std::fs::read_to_string(&ptx).ok()).flatten(), said)
}

/// Sets the pinned profile's measured split choice for `m x n x k` to `value`.
fn set_split_entry(dir: &Path, base: &str, m: u32, n: u32, k: u32, value: &str) {
    std::fs::write(dir.join(".ysu_hw_profile"), format!("{base}AUTOTUNE_F16_SPLITK_AUTO_{m}x{n}x{k}_{}={value}\n", pinned::GPU_NAME)).unwrap();
}

#[test]
fn a_measured_split_choice_replaces_the_rules_and_only_a_valid_one() {
    let dir = pinned::pinned_scratch("f16_splitk_measured", pinned::SM_PINNED);
    let base = std::fs::read_to_string(dir.join(".ysu_hw_profile")).unwrap();
    let (m, n, k) = (512, 512, 16384);
    // The rule's own choice, the control: 2 splits on 64x128.
    let (rule, _) = compile_with(&dir, "r_splitk", m, n, k, true, &[], &[]);
    assert_eq!(splits_of(&rule.unwrap()), 2);
    // A measurement of 4 splits on 128x128 is taken, and said so.
    set_split_entry(&dir, &base, m, n, k, "4,128,128,32,4,2,3");
    let (ptx, said) = compile_with(&dir, "m_splitk", m, n, k, true, &[], &[]);
    let ptx = ptx.unwrap_or_else(|| panic!("{said}"));
    assert_eq!(splits_of(&ptx), 4);
    assert!(ptx.contains("| CTA 128x128x32 | 4x2 warps |"));
    assert!(ptx.contains(&format!("[Y AUTO SPLIT-K] S=4 as measured: measured (AUTOTUNE_F16_SPLITK_AUTO_512x512x16384_{}): 4 splits of K=4096 on 128x128x32 tiles, 4x2 warps", pinned::GPU_NAME)), "{ptx}");
    assert!(said.contains("[Y split-K choice] `m_splitk`: S=4"), "{said}");
    // The plain GEMM's note reads it too.
    let (_, note) = compile_with(&dir, "g", m, n, k, false, &[], &[]);
    assert!(note.contains("`g` would split K 4 ways: measured (AUTOTUNE_F16_SPLITK_AUTO"), "{note}");
    // A measurement of no split leaves the plain kernel where the rule splits.
    set_split_entry(&dir, &base, m, n, k, "1");
    let (one, _) = compile_with(&dir, "o_splitk", m, n, k, true, &[], &[]);
    let one = one.unwrap();
    assert_eq!(splits_of(&one), 1);
    assert!(one.contains("[Y AUTO SPLIT-K] S=1 as measured: measured (AUTOTUNE_F16_SPLITK_AUTO_512x512x16384_"));
    // `Y_AUTOTUNE=off` ignores persisted measurements: the rule decides.
    set_split_entry(&dir, &base, m, n, k, "4,128,128,32,4,2,3");
    let (off, _) = compile_with(&dir, "f_splitk", m, n, k, true, &[("Y_AUTOTUNE", "off")], &[]);
    let off = off.unwrap();
    assert_eq!(splits_of(&off), 2);
    assert!(off.contains("S=2 by the split-K rule"));
    // An entry codegen could not emit is refused, said so, and the rule decides.
    for bad in ["3,64,64,32,4,2,4", "2,64,48,32,4,2,4", "2,64,64,32,4,2,0", "x"] {
        set_split_entry(&dir, &base, m, n, k, bad);
        let (ptx, said) = compile_with(&dir, "b_splitk", m, n, k, true, &[], &[]);
        let ptx = ptx.unwrap_or_else(|| panic!("{bad}: {said}"));
        assert!(said.contains(&format!("ignoring AUTOTUNE_F16_SPLITK_AUTO_512x512x16384_{}={bad}", pinned::GPU_NAME)), "{bad}: {said}");
        assert!(splits_of(&ptx) == 2 && ptx.contains("S=2 by the split-K rule"), "{bad}");
    }
    // Where the rule's gate does not pass, no measured split is taken:
    // 64 x 4096 x 11008 is bound to DRAM.
    set_split_entry(&dir, &base, 64, 4096, 11008, "2,64,128,32,4,2,4");
    let (dram, _) = compile_with(&dir, "d_splitk", 64, 4096, 11008, true, &[], &[]);
    let dram = dram.unwrap();
    assert!(splits_of(&dram) == 1 && dram.contains("bound to DRAM"));
}

#[test]
fn the_split_choice_is_measured_and_then_kept_on_the_gpu() {
    use y::cuda_runtime::CudaContext;
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver - no split choice was measured.");
        return;
    };
    let dir = pinned::pinned_scratch("f16_splitk_measure_gpu", pinned::SM_PINNED);
    let (m, n, k) = (256u32, 256u32, 8192u32);
    let key = format!("AUTOTUNE_F16_SPLITK_AUTO_{m}x{n}x{k}_{}", pinned::GPU_NAME);
    // --autotune measures the rule's candidates and keeps the winner.
    let (ptx, said) = compile_with(&dir, "a_splitk", m, n, k, true, &[], &["--autotune"]);
    let ptx = ptx.unwrap_or_else(|| panic!("{said}"));
    // Two measurements: the unsplit kernel's tile, then the split choice -
    // once, though the CLI and codegen both ask, and no split's own tile
    // measurement after it.
    assert_eq!(said.matches("[Y autotuner] measuring on").count(), 2, "{said}");
    assert_eq!(said.matches("split 1-8 ways").count(), 1, "{said}");
    let profile = std::fs::read_to_string(dir.join(".ysu_hw_profile")).unwrap();
    let value = profile.lines().find_map(|l| l.strip_prefix(&format!("{key}="))).unwrap_or_else(|| panic!("no {key} in the profile"));
    let (s, tile) = y::autotuner::Autotuner::parse_split_k_entry(value, m, n, k).unwrap();
    assert_eq!(splits_of(&ptx), s);
    assert!(ptx.contains("as measured: measured by --autotune among"), "{ptx}");
    if let Some(t) = &tile {
        assert!(ptx.contains(&format!("| CTA {}x{}x{} | {}x{} warps |", t.cta_m, t.cta_n, t.cta_k, t.warps_m, t.warps_n)));
    }
    // A later build reads it, and makes the same module but for the line
    // naming the measurement's source.
    let (again, said) = compile_with(&dir, "a_splitk", m, n, k, true, &[], &[]);
    let again = again.unwrap();
    assert!(!said.contains("[Y autotuner] measuring on"), "measured again: {said}");
    assert!(again.contains(&format!("as measured: measured ({key})")));
    let body = |p: &str| p.lines().filter(|l| !l.contains("[Y AUTO SPLIT-K]")).collect::<Vec<_>>().join("\n");
    assert_eq!(body(&again), body(&ptx));
    // A plain GEMM under --autotune measures its own tile, never a split.
    let (_, plain) = compile_with(&dir, "p", m, n, 2 * k, false, &[], &["--autotune"]);
    assert!(plain.contains("[Y autotuner] measuring on") && !plain.contains("split 1-"), "{plain}");
    assert!(!std::fs::read_to_string(dir.join(".ysu_hw_profile")).unwrap().contains(&format!("SPLITK_AUTO_{m}x{n}x{}", 2 * k)));
    // And the chosen kernel is exact on small integers.
    let mut state = 0x5E1F_C401u64;
    let (mu, nu, ku) = (m as usize, n as usize, k as usize);
    let ia: Vec<i32> = (0..mu * ku).map(|_| (splitmix(&mut state) % 9) as i32 - 4).collect();
    let ib: Vec<i32> = (0..ku * nu).map(|_| (splitmix(&mut state) % 9) as i32 - 4).collect();
    let iref: Vec<f32> = (0..mu * nu)
        .map(|x| (0..ku).map(|kk| (ia[(x / nu) * ku + kk] * ib[kk * nu + x % nu]) as i64).sum::<i64>() as f32)
        .collect();
    let (ha, hb): (Vec<u16>, Vec<u16>) = (ia.iter().map(|&v| f16_of_int(v)).collect(), ib.iter().map(|&v| f16_of_int(v)).collect());
    let (c, _) = run_auto(&ctx, &again, "a_splitk", m, n, k, &ha, &hb);
    let wrong = c.iter().zip(&iref).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    assert_eq!(wrong, 0, "{wrong} integer products wrong");
}

