//! The split-K F16 GEMM: an `@tile(M, N, K)` kernel named `<name>_splitk<S>`
//! taking (A, B: GlobalMemory<F16>, C, Partial: GlobalMemory<F32>). Its main
//! entry runs S equal K ranges in `%ctaid.z`, each written as an M x N float32
//! partial to slot z of Partial, accumulating each mma in float32 by default;
//! its `_reduce` entry adds the slots into C in slot order, with no atomics.
//!
//! This file checks the name grammar and the shapes refused, and on the
//! device: exact integer products; every slot equal, bit for bit, to the
//! unsplit kernel run on that slot's K range; C equal to the slots summed in
//! order on the host; the same bits on every launch; same-sign error at
//! float32's level against the chained control; and the tuner measuring the
//! split kernel with its reduction.

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
    assert!(accumulates_in_f32(&ptx), "a split accumulates in float32 by default, at any M");
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
        ("g_splitk", 256, 256, 4096, true, "must end in `_splitk<S>` with S from 1 to 64"),
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
        // The unsplit float32 kernel of one split's depth, same tile.
        let slice_name = format!("u{m}x{n}x{}", k / s);
        let slice = compile(&dir, &slice_name, m, n, k / s, false, Some("f32"), Some(tile)).unwrap();
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

        // 5. Same-sign error: float32 accumulation at float32's level, the
        // chained splits (the control) visibly biased low.
        if k / s >= 1024 {
            let exact: Vec<f64> = (0..mu * nu)
                .map(|x| (0..ku).map(|kk| f16_value(a[(x / nu) * ku + kk]) * f16_value(b[kk * nu + x % nu])).sum())
                .collect();
            let mean_rel = |got: &[f32]| got.iter().zip(&exact).map(|(&g, &e)| (g as f64 - e) / e).sum::<f64>() / (mu * nu) as f64;
            let (cc, _) = run_split(&chained, &format!("c{m}x{n}x{k}_splitk{s}"), &a, &b);
            let (f32_err, chained_err) = (mean_rel(&c), mean_rel(&cc));
            assert!(chained_err < -5e-6, "{name}: the chained splits are off by only {chained_err:e} on average");
            assert!(f32_err.abs() < 1e-6, "{name}: float32 accumulation is off by {f32_err:e} on average");
        }
    }
}

#[test]
fn the_tuner_measures_a_split_kernel_with_its_reduction() {
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
