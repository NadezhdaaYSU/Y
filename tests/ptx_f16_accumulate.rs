//! Where the F16 GEMM's accumulation happens.
//!
//! The tensor core adds into its accumulator with less than float32 rounding,
//! so a K loop chained in it - `mma d, a, b, d`, what cuBLAS's and Triton's
//! tensor-core kernels do - leaves same-sign results about 1.2e-5 per 1024 of
//! K below the true sum. Starting each mma from zero and adding its result in
//! float32 (`add.rn.f32`) keeps them within float32 rounding at any K. It is
//! the default for GEMMs of at most 32 rows, where it is no slower, and
//! `Y_F16_ACCUMULATE=f32|mma` selects either everywhere. This file checks the
//! selection, and on the device that each mode computes exact integer products
//! and that only the float32 one keeps same-sign error at float32's level.

use std::path::Path;
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

/// The module the compiler emits for an `m x n x k` F16 GEMM, with
/// `Y_F16_ACCUMULATE` set to `knob` (or unset), or its error output.
fn compile(dir: &Path, tag: &str, m: u32, n: u32, k: u32, knob: Option<&str>) -> Result<String, String> {
    let src = dir.join(format!("{tag}.ysu"));
    std::fs::write(&src, format!("@tile({m}, {n}, {k})\nkernel {tag}(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>) {{\n}}\n\nfn main() {{\n}}\n")).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_Y"));
    cmd.arg(&src).arg("--emit-ptx").current_dir(dir).env_remove("Y_CTA_OVERRIDE");
    match knob {
        Some(v) => cmd.env("Y_F16_ACCUMULATE", v),
        None => cmd.env_remove("Y_F16_ACCUMULATE"),
    };
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

#[test]
fn the_default_follows_the_row_count_and_the_knob_overrides_it() {
    let dir = pinned::pinned_scratch("f16_acc_mode", pinned::SM_PINNED);
    for (m, want) in [(1u32, true), (8, true), (32, true), (33, false), (256, false)] {
        let ptx = compile(&dir, &format!("acc_m{m}"), m, 256, 512, None).unwrap();
        assert_eq!(accumulates_in_f32(&ptx), want, "M = {m}, no knob");
    }
    for (m, knob, want) in [(256u32, "f32", true), (8, "mma", false)] {
        let ptx = compile(&dir, &format!("acc_{knob}_{m}"), m, 256, 512, Some(knob)).unwrap();
        assert_eq!(accumulates_in_f32(&ptx), want, "M = {m}, Y_F16_ACCUMULATE={knob}");
    }
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
    // A decode shape (float32 by default) and a square one (the tensor-core
    // chain by default), each in both modes.
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
        for knob in ["mma", "f32"] {
            let tag = format!("acc_{knob}_{m}x{n}");
            let ptx = compile(&dir, &tag, m, n, k, Some(knob)).unwrap();
            assert_eq!(accumulates_in_f32(&ptx), knob == "f32");
            let field = |key: &str| -> u32 {
                let at = ptx.find(key).unwrap_or_else(|| panic!("no {key:?}")) + key.len();
                ptx[at..].chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap()
            };
            let (cm, cn) = (field("| CTA "), field(&format!("| CTA {}x", field("| CTA "))));
            let warps_at = ptx.find(" warps").unwrap();
            let w: Vec<u32> = ptx[..warps_at].rsplit("| ").next().unwrap().split('x').map(|x| x.trim().parse().unwrap()).collect();
            let smem = ptx.find("Dynamic shared memory required: ").map(|_| field("Dynamic shared memory required: ")).unwrap_or(0);
            let module = ctx.load_ptx(&ptx, &tag).unwrap();
            if smem > 48 * 1024 {
                module.set_max_dynamic_smem(smem).unwrap();
            }
            let d_a = ctx.alloc(mu * ku * 2).unwrap();
            let d_b = ctx.alloc(ku * nu * 2).unwrap();
            let d_c = ctx.alloc(mu * nu * 4).unwrap();
            let run = |a: &[u16], b: &[u16]| -> Vec<f32> {
                ctx.memcpy_htod_at(&d_a, 0, &a.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
                ctx.memcpy_htod_at(&d_b, 0, &b.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
                ctx.memset_u8(&d_c, 0xFF).unwrap();
                let grid = (n.div_ceil(cn), m.div_ceil(cm), 1);
                ctx.launch(&module, grid, (w[0] * w[1] * 32, 1, 1), smem, &[d_a.device_ptr(), d_b.device_ptr(), d_c.device_ptr()]).unwrap();
                ctx.synchronize().unwrap();
                let mut raw = vec![0u8; mu * nu * 4];
                ctx.memcpy_dtoh_at(&mut raw, &d_c, 0).unwrap();
                raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
            };
            let ints = run(&ia.iter().map(|&v| f16_of_int(v)).collect::<Vec<_>>(), &ib.iter().map(|&v| f16_of_int(v)).collect::<Vec<_>>());
            let wrong = ints.iter().zip(&iref).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            assert_eq!(wrong, 0, "{m}x{n}x{k} {knob}: {wrong} integer products wrong");
            let got = run(&a, &b);
            let mean_rel: f64 = got.iter().zip(&exact).map(|(&g, &e)| (g as f64 - e) / e).sum::<f64>() / (mu * nu) as f64;
            errs.push(mean_rel);
        }
        let (chained, f32_added) = (errs[0], errs[1]);
        // The control: the chain's known bias, about -5.0e-5 at K = 4096.
        assert!(chained < -2e-5, "{m}x{n}x{k}: the chained accumulator is off by only {chained:e} on average");
        // float32 rounding of a 256-term sum of exact-ish partials: ~1e-7.
        assert!(f32_added.abs() < 1e-6, "{m}x{n}x{k}: float32 accumulation is off by {f32_added:e} on average");
    }
}
