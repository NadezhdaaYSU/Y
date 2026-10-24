//! Where the F16 GEMM's accumulation happens.
//!
//! The tensor core adds into its accumulator with less than float32 rounding,
//! so a K loop chained in it - `mma d, a, b, d`, what cuBLAS's and Triton's
//! tensor-core kernels do - leaves same-sign results about 1.2e-5 per 1024 of
//! K below the true sum. Starting each mma from zero and adding its result in
//! float32 (`add.rn.f32`) keeps them within float32 rounding at any K. It is
//! the default for GEMMs of at most 32 rows, where it is no slower, and
//! `Y_F16_ACCUMULATE=f32|mma` selects either everywhere; `=kahan` compensates
//! the float32 adds. This file checks the selection, and on the device that
//! each mode computes exact integer products, that only the float32 ones keep
//! same-sign error at float32's level, and that compensation narrows it.

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
    let src = dir.join(format!("{tag}.ysu"));
    std::fs::write(&src, source).unwrap();
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

#[test]
fn the_default_follows_the_row_count_and_the_knob_overrides_it() {
    let dir = pinned::pinned_scratch("f16_acc_mode", pinned::SM_PINNED);
    for (m, want) in [(1u32, true), (8, true), (32, true), (33, false), (256, false)] {
        let ptx = compile(&dir, &format!("acc_m{m}"), m, 256, 512, None).unwrap();
        assert_eq!(accumulates_in_f32(&ptx), want, "M = {m}, no knob");
        assert!(!compensated(&ptx), "M = {m}: compensated with no knob");
    }
    for (m, knob, want) in [(256u32, "f32", true), (8, "mma", false), (8, "kahan", true), (256, "kahan", true)] {
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
        for knob in ["mma", "f32", "kahan"] {
            let tag = format!("acc_{knob}_{m}x{n}");
            let ptx = compile(&dir, &tag, m, n, k, Some(knob)).unwrap();
            assert_eq!(accumulates_in_f32(&ptx), knob != "mma");
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
        let ((chained, _), (f32_added, f32_spread), (kahan, kahan_spread)) = (errs[0], errs[1], errs[2]);
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
    }
}
