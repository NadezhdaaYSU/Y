//! `BF16`: bfloat16 (`torch.bfloat16`) as a `GlobalMemory` element, and the
//! `@tile` tensor-core GEMM over bf16 operands.
//!
//! A load widens exactly: the bf16's bits become the f32's high half. A store
//! rounds an F32 to nearest-even (`cvt.rn.bf16.f32`, sm_80 and later). An
//! F64, a literal or an integer is rounded once, not through a nearest f32:
//! sm_80 and sm_89 convert to bf16 from f32 only, so they reach f32 by
//! rounding to odd. A `BF16` value - a `let`, a scalar parameter - is refused,
//! as `F16`'s is, and so is every backend without the conversion.
//!
//! `@tile(M, N, K) kernel f(A, B: GlobalMemory<BF16>, C: GlobalMemory<F32>)`
//! is the F16 tensor-core GEMM with `.bf16` mmas (A `[M][K]`, B `[K][N]`,
//! row-major), and its bias and split-K forms too. The device checks compare
//! bits against references built from the format's definition;
//! python/y_lang/tests/test_bf16.py sweeps the conversions against torch.

use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

#[path = "common/pinned.rs"]
mod pinned;

struct Emitted {
    ok: bool,
    log: String,
    ptx: Option<String>,
}

/// Compiles `src` (`fn main() {}` appended) with `args` under a profile
/// pinned to `sm`, in its own temp dir; returns the log and any `k.ptx`.
fn compile(tag: &str, src: &str, sm: &str, args: &[&str]) -> Emitted {
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let mut bin = std::env::current_exe().unwrap();
    bin.pop();
    if bin.ends_with("deps") {
        bin.pop();
    }
    let dir = std::env::temp_dir().join(format!(
        "y_bf16_{}_{}_{}",
        std::process::id(),
        tag,
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("k.ysu");
    std::fs::write(&file, format!("{}\nfn main() {{}}\n", src)).unwrap();
    pinned::pin(&dir, sm);
    let out = Command::new(bin.join("Y"))
        .arg(&file)
        .args(args)
        .current_dir(&dir)
        .env_remove("Y_CTA_OVERRIDE")
        .env_remove("Y_F16_ACCUMULATE")
        .output()
        .expect("run Y");
    let log = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    let ptx = std::fs::read_to_string(dir.join("k.ptx")).ok();
    let _ = std::fs::remove_dir_all(&dir);
    Emitted { ok: out.status.success(), log, ptx }
}

fn ptx_at(tag: &str, src: &str, sm: &str) -> String {
    let e = compile(tag, src, sm, &["--emit-ptx"]);
    assert!(e.ok, "{} must compile:\n{}", tag, e.log);
    e.ptx.unwrap_or_else(|| panic!("{} compiled but wrote no .ptx:\n{}", tag, e.log))
}

fn refused(tag: &str, src: &str, sm: &str, args: &[&str], why: &str) {
    let e = compile(tag, src, sm, args);
    assert!(!e.ok, "{} must be refused, and compiled:\n{}", tag, e.log);
    assert!(e.log.contains(why), "{} was refused, but not saying `{}`:\n{}", tag, why, e.log);
}

/// Every finite non-negative bf16, by bit pattern, with its exact value.
fn finite_bf16() -> Vec<(u16, f64)> {
    (0u16..0x7F80).map(|p| (p, f64::from(f32::from_bits(u32::from(p) << 16)))).collect()
}

/// The bf16 nearest `x`, ties to the even pattern, overflowing to infinity
/// past the midpoint above the largest finite value; `None` for NaN. A
/// search over every finite pattern, so it rests on nothing but the format.
/// `x - c` is exact for the two candidates nearest `x` (Sterbenz), so a tie
/// is seen as one.
fn bf16_of(x: f64, table: &[(u16, f64)]) -> Option<u16> {
    if x.is_nan() {
        return None;
    }
    let sign = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = x.abs();
    let max = table.last().unwrap().1;
    let half_ulp = (max - table[table.len() - 2].1) / 2.0;
    if a >= max + half_ulp {
        return Some(sign | 0x7F80);
    }
    let mut best = (0u16, f64::INFINITY);
    let i = table.partition_point(|&(_, v)| v <= a);
    for &(p, v) in &table[i.saturating_sub(1)..(i + 1).min(table.len())] {
        let d = (v - a).abs();
        if d < best.1 || (d == best.1 && p % 2 == 0) {
            best = (p, d);
        }
    }
    Some(sign | best.0)
}

/// The bf16 nearest the integer `x`, exactly: past 256 the candidates are
/// integers, so the distances are compared in i128.
fn bf16_of_int(x: i64, table: &[(u16, f64)]) -> u16 {
    let sign = if x < 0 { 0x8000 } else { 0 };
    let a = i128::from(x).abs();
    if a <= 256 {
        return sign | bf16_of(a as f64, table).unwrap();
    }
    let i = table.partition_point(|&(_, v)| v <= a as f64);
    let mut best = (0u16, i128::MAX);
    for &(p, v) in &table[i.saturating_sub(2)..(i + 2).min(table.len())] {
        let d = (v as i128 - a).abs();
        if d < best.1 || (d == best.1 && p % 2 == 0) {
            best = (p, d);
        }
    }
    sign | best.0
}

/// Launches `ptx`'s entry `k` and returns every buffer after it, or `None`
/// without a CUDA driver or an sm_80 GPU.
fn run(ptx: &str, grid: (u32, u32, u32), threads: u32, shared: u32, bufs: &[Vec<u8>], scalars: &[u64]) -> Option<Vec<Vec<u8>>> {
    use y::cuda_runtime::CudaContext;
    let ctx = CudaContext::new()?;
    // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR / _MINOR
    let cc = (ctx.device_attribute(75)?, ctx.device_attribute(76)?);
    if cc < (8, 0) {
        return None;
    }
    let module = ctx.load_ptx(ptx, "k").expect("PTX failed to load");
    if shared > 48 * 1024 {
        module.set_max_dynamic_smem(shared).unwrap();
    }
    let mut dev = Vec::new();
    for init in bufs {
        let b = ctx.alloc(init.len()).unwrap();
        ctx.memcpy_htod_at(&b, 0, init).unwrap();
        dev.push(b);
    }
    let mut args: Vec<u64> = dev.iter().map(|b| b.device_ptr()).collect();
    args.extend_from_slice(scalars);
    ctx.launch(&module, grid, (threads, 1, 1), shared, &args).expect("launch failed");
    ctx.synchronize().expect("kernel did not complete");
    Some(
        dev.iter()
            .map(|b| {
                let mut v = vec![0u8; b.len_bytes()];
                ctx.memcpy_dtoh_at(&mut v, b, 0).unwrap();
                v
            })
            .collect(),
    )
}

fn skip(what: &str) {
    eprintln!("SKIP: no CUDA driver or no sm_80 GPU - {} was not checked on the device.", what);
}

fn halves(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

// ─────────────────────────── the element type ────────────────────────────

/// Every one of the 65,536 patterns, NaNs and infinities included, widens to
/// the f32 whose high half it is.
#[test]
fn a_load_widens_every_pattern_exactly() {
    let ptx = ptx_at(
        "widen",
        "kernel k(Q: GlobalMemory<BF16>, Out: GlobalMemory<F32>) {\n    let i: U32 = global_thread_id();\n    Out[i] = Q[i];\n}",
        pinned::SM_PINNED,
    );
    assert!(!ptx.contains("cvt.f32.bf16"), "sm_80 has no cvt.f32.bf16; it would not load:\n{}", ptx);
    let q: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
    let Some(out) = run(&ptx, (256, 1, 1), 256, 0, &[q, vec![0xAB; 4 << 16]], &[]) else {
        return skip("the BF16 load");
    };
    for (p, w) in words(&out[1]).into_iter().enumerate() {
        assert_eq!(w, (p as u32) << 16, "pattern {:#06x} widened to {:#010x}", p, w);
    }
}

/// From f32: ties both ways, values a hair past them, subnormals, the
/// overflow boundary, infinities, signed zero, NaN, and a sweep of ordinary
/// values, all against the format's definition.
#[test]
fn a_store_rounds_an_f32_to_nearest_even() {
    let table = finite_bf16();
    let mut xs: Vec<f32> = vec![
        1.0, 1.00390625, 1.01171875, 1.0039063, 1.0039062, -2.5, 3.0e38, 3.3961776e38, 3.3961778e38, 3.4028235e38,
        f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0, f32::from_bits(1), f32::from_bits(0x8000), f32::from_bits(0x18000),
        f32::from_bits(0x007F_FFFF), f32::MIN_POSITIVE, 1.0e-40, -1.0e-40,
    ];
    let mut s = 0x1234_5678u32;
    for _ in 0..4000 {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        let f = f32::from_bits(s);
        if !f.is_nan() {
            xs.push(f);
        }
    }
    let n = xs.len() as u32;
    let ptx = ptx_at(
        "narrow",
        "kernel k(X: GlobalMemory<F32>, Q: GlobalMemory<BF16>, n: U32) {\n    let i: U32 = global_thread_id();\n    if i < n {\n        Q[i] = X[i];\n    }\n}",
        pinned::SM_PINNED,
    );
    assert!(ptx.contains("cvt.rn.bf16.f32"), "{}", ptx);
    let x: Vec<u8> = xs.iter().flat_map(|v| v.to_le_bytes()).collect();
    let Some(out) = run(&ptx, (n.div_ceil(256), 1, 1), 256, 0, &[x, vec![0xAB; 2 * n as usize]], &[u64::from(n)]) else {
        return skip("the BF16 store from f32");
    };
    for (v, got) in xs.iter().zip(halves(&out[1])) {
        let want = bf16_of(f64::from(*v), &table).unwrap();
        assert_eq!(got, want, "{:e} ({:#010x}) stored as {:#06x}, nearest-even is {:#06x}", v, v.to_bits(), got, want);
    }
}

/// From f64 and i64, at bf16 midpoints and one f64 ulp either side, where
/// rounding to the nearest f32 first would round twice; the control counts
/// how many of them that would get wrong, so the inputs do test it.
#[test]
fn an_f64_and_an_integer_round_once() {
    let table = finite_bf16();
    let mut xs: Vec<f64> = Vec::new();
    for w in table.windows(2).step_by(97) {
        let mid = (w[0].1 + w[1].1) / 2.0;
        for v in [mid, f64::from_bits(mid.to_bits() - 1), f64::from_bits(mid.to_bits() + 1)] {
            xs.push(v);
            xs.push(-v);
        }
    }
    let double = xs.iter().filter(|&&v| bf16_of(f64::from(v as f32), &table) != bf16_of(v, &table)).count();
    assert!(double > 100, "only {} of these inputs would round twice through f32; the test is too weak", double);
    let ints: Vec<i64> = (8..63)
        .flat_map(|k| {
            let base = 1i64 << k;
            let step = 1i64 << (k - 8);
            [base + step / 2, base + step / 2 + 1, base + step / 2 - 1, base + 3 * step / 2, base - 1, -(base + step / 2)]
        })
        .collect();
    let (n, ni) = (xs.len() as u32, ints.len() as u32);
    let ptx = ptx_at(
        "narrow64",
        "kernel k(X: GlobalMemory<F64>, Q: GlobalMemory<BF16>, I: GlobalMemory<I64>, R: GlobalMemory<BF16>, n: U32, ni: U32) {\n    let i: U32 = global_thread_id();\n    if i < n {\n        Q[i] = X[i];\n    }\n    if i < ni {\n        R[i] = I[i];\n    }\n}",
        pinned::SM_PINNED,
    );
    let bufs = [
        xs.iter().flat_map(|v| v.to_le_bytes()).collect(),
        vec![0xAB; 2 * n as usize],
        ints.iter().flat_map(|v| v.to_le_bytes()).collect(),
        vec![0xAB; 2 * ni as usize],
    ];
    let Some(out) = run(&ptx, (n.max(ni).div_ceil(256), 1, 1), 256, 0, &bufs, &[u64::from(n), u64::from(ni)]) else {
        return skip("the BF16 store from f64 and i64");
    };
    for (v, got) in xs.iter().zip(halves(&out[1])) {
        assert_eq!(Some(got), bf16_of(*v, &table), "{:e} ({:#018x}) stored as {:#06x}", v, v.to_bits(), got);
    }
    for (v, got) in ints.iter().zip(halves(&out[3])) {
        assert_eq!(got, bf16_of_int(*v, &table), "{} stored as {:#06x}", v, got);
    }
}

#[test]
fn a_predicated_load_past_the_edge_reads_zero() {
    let ptx = ptx_at(
        "pred_load",
        "kernel k(Q: GlobalMemory<BF16>, Out: GlobalMemory<F32>) {
    Out[0] = block_ptr2d_load(Q, 0, 1, 4, 1, 4);
    Out[1] = block_ptr2d_load(Q, 1, 1, 4, 1, 4);
}",
        pinned::SM_PINNED,
    );
    // The zero is written by an instruction of its own, under the opposite
    // predicate: on the card an unwritten register usually holds 0 already.
    let lines: Vec<&str> = ptx.lines().map(str::trim).collect();
    let loads: Vec<usize> = (0..lines.len()).filter(|&i| lines[i].contains("ld.global.b16")).collect();
    assert_eq!(loads.len(), 2, "{}", ptx);
    for i in loads {
        let (pred, rest) = lines[i].split_once(' ').unwrap();
        let reg = rest.split_whitespace().nth(1).unwrap().trim_end_matches(',');
        let want = format!("@!{} mov.b16 {}, 0;", pred.trim_start_matches('@'), reg);
        assert_eq!(lines[i + 1], want, "the load `{}` is not followed by its zeroing", lines[i]);
    }
    let q: Vec<u8> = [0u16, 0xC040, 0, 0, 0, 0x40A0, 0, 0].iter().flat_map(|v| v.to_le_bytes()).collect();
    let Some(out) = run(&ptx, (1, 1, 1), 1, 0, &[q, vec![0xAB; 8]], &[]) else {
        return skip("the predicated BF16 load");
    };
    let at = |i: usize| f32::from_le_bytes(out[1][i * 4..i * 4 + 4].try_into().unwrap());
    assert_eq!((at(0), at(1)), (-3.0, 0.0), "row 1 is past max_r = 1, so it must read 0, not 5");
}

#[test]
fn the_v4_forms_are_refused() {
    refused(
        "v4",
        "kernel k(Q: GlobalMemory<BF16>, Out: GlobalMemory<F32>) {\n    let v: F32 = block_ptr2d_load_v4(Q, 0, 0, 4, 1, 4);\n    Out[0] = v;\n}",
        pinned::SM_PINNED,
        &["--emit-ptx"],
        "no BF16 form",
    );
}

#[test]
fn the_conversion_needs_sm_80() {
    let src = "kernel k(X: GlobalMemory<F32>, Q: GlobalMemory<BF16>) {\n    Q[0] = X[0];\n    X[1] = Q[1];\n}";
    refused("sm75", src, "7.5", &["--emit-ptx"], "exists only on sm_80 (Ampere) and later");
    let ptx = ptx_at("sm80", src, pinned::SM_PINNED);
    assert!(ptx.contains("cvt.rn.bf16.f32"), "{}", ptx);
}

#[test]
fn a_bf16_value_is_refused() {
    let why = "`BF16` is a buffer element type in this backend, not a value type";
    refused("let", "kernel k(X: GlobalMemory<F32>) {\n    let v: BF16 = X[0];\n    X[1] = v;\n}", pinned::SM_PINNED, &["--emit-ptx"], why);
    refused("param", "kernel k(s: BF16, X: GlobalMemory<F32>) {\n    X[0] = 1.0;\n}", pinned::SM_PINNED, &["--emit-ptx"], why);
}

#[test]
fn backends_without_the_conversion_refuse_it() {
    let src = "kernel k(X: GlobalMemory<F32>, Q: GlobalMemory<BF16>) {\n    Q[0] = X[0];\n}";
    refused("cpu", src, pinned::SM_PINNED, &["--emit-cpu"], "`GlobalMemory<BF16>` cannot be lowered");
    refused("llvm", src, pinned::SM_PINNED, &["--emit-llvm"], "element type this backend does not know");
}

// ──────────────────────────── the bf16 GEMM ─────────────────────────────

fn gemm_src(name: &str, elem: &str, m: u32, n: u32, k: u32, params: &str) -> String {
    format!("@tile({m}, {n}, {k})\nkernel {name}(A: GlobalMemory<{elem}>, B: GlobalMemory<{elem}>, {params}) {{\n}}")
}

/// `(cta_m, cta_n, threads, dynamic shared bytes)` from the GEMM's header.
fn geometry(ptx: &str) -> (u32, u32, u32, u32) {
    let line = ptx.lines().find(|l| l.contains("[Y TENSOR CORE GEMM]")).expect("no GEMM header");
    let cta = line.split("| CTA ").nth(1).unwrap().split(' ').next().unwrap();
    let dims: Vec<u32> = cta.split('x').map(|v| v.parse().unwrap()).collect();
    let warps: Vec<u32> = line.split("| ").nth(2).unwrap().split(' ').next().unwrap().split('x').map(|v| v.parse().unwrap()).collect();
    let smem = ptx
        .split("Dynamic shared memory required: ")
        .nth(1)
        .and_then(|s| s.split(' ').next())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    (dims[0], dims[1], warps[0] * warps[1] * 32, smem)
}

/// Small integers in [-8, 8]: every product and partial sum is exact in f32,
/// so C must be the integer product bit for bit, whatever the tiling.
fn operands(m: usize, n: usize, k: usize, seed: u32) -> (Vec<i32>, Vec<i32>) {
    let mut s = seed;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        (s % 17) as i32 - 8
    };
    let a = (0..m * k).map(|_| next()).collect();
    let b = (0..k * n).map(|_| next()).collect();
    (a, b)
}

fn as_bf16(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|&x| ((f32::to_bits(x as f32) >> 16) as u16).to_le_bytes()).collect()
}

fn product(a: &[i32], b: &[i32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = (0..k).map(|p| a[i * k + p] * b[p * n + j]).sum::<i32>() as f32;
        }
    }
    c
}

/// The BF16 GEMM's PTX is the F16 GEMM's with `bf16` for `f16` in the mma
/// and the header: the same tile, staging, swizzle and accumulation. Runs
/// without a GPU.
#[test]
fn the_gemm_is_the_f16_gemm_with_bf16_mmas() {
    for (m, n, k) in [(512u32, 384u32, 320u32), (33, 4096, 896), (1024, 1024, 4096)] {
        let b = ptx_at("bf16", &gemm_src("k", "BF16", m, n, k, "C: GlobalMemory<F32>"), pinned::SM_PINNED);
        let f = ptx_at("f16", &gemm_src("k", "F16", m, n, k, "C: GlobalMemory<F32>"), pinned::SM_PINNED);
        assert!(b.contains("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"), "{m}x{n}x{k}");
        assert!(!b.contains(".f16.f16"), "{m}x{n}x{k}: an f16 mma in the bf16 GEMM");
        assert_eq!(b.replace("bf16", "f16"), f, "{m}x{n}x{k}: the BF16 GEMM differs from the F16 one beyond the type");
    }
}

#[test]
fn the_gemm_is_exact_on_small_integers() {
    for (m, n, k) in [(512usize, 384usize, 320usize), (200, 136, 72), (33, 256, 896), (16, 64, 4096)] {
        let ptx = ptx_at("gemm", &gemm_src("k", "BF16", m as u32, n as u32, k as u32, "C: GlobalMemory<F32>"), pinned::SM_PINNED);
        let (cm, cn, threads, smem) = geometry(&ptx);
        let (a, b) = operands(m, n, k, (m * 31 + n * 7 + k) as u32);
        let poison: Vec<u8> = f32::NAN.to_le_bytes().repeat(m * n);
        let grid = ((n as u32).div_ceil(cn), (m as u32).div_ceil(cm), 1);
        let Some(out) = run(&ptx, grid, threads, smem, &[as_bf16(&a), as_bf16(&b), poison], &[]) else {
            return skip("the BF16 GEMM");
        };
        let want = product(&a, &b, m, n, k);
        let got: Vec<f32> = words(&out[2]).into_iter().map(f32::from_bits).collect();
        let bad = got.iter().zip(&want).filter(|(g, w)| g.to_bits() != w.to_bits()).count();
        assert_eq!(bad, 0, "{m}x{n}x{k}: {bad} of {} outputs are not the integer product", m * n);
    }
}

#[test]
fn the_bias_and_split_k_forms_are_exact() {
    let (m, n, k) = (256usize, 384usize, 1024usize);
    let (a, b) = operands(m, n, k, 99);
    let prod = product(&a, &b, m, n, k);
    let bias: Vec<f32> = (0..n).map(|j| (j as f32) - 200.0).collect();
    let poison: Vec<u8> = f32::NAN.to_le_bytes().repeat(m * n);

    let ptx = ptx_at("bias", &gemm_src("k", "BF16", m as u32, n as u32, k as u32, "Bias: GlobalMemory<F32>, C: GlobalMemory<F32>"), pinned::SM_PINNED);
    let (cm, cn, threads, smem) = geometry(&ptx);
    let bias_bytes: Vec<u8> = bias.iter().flat_map(|v| v.to_le_bytes()).collect();
    let grid = ((n as u32).div_ceil(cn), (m as u32).div_ceil(cm), 1);
    let Some(out) = run(&ptx, grid, threads, smem, &[as_bf16(&a), as_bf16(&b), bias_bytes, poison.clone()], &[]) else {
        return skip("the BF16 GEMM's bias form");
    };
    let got: Vec<f32> = words(&out[3]).into_iter().map(f32::from_bits).collect();
    for (idx, g) in got.iter().enumerate() {
        let want = (prod[idx] + bias[idx % n]).max(0.0);
        assert_eq!(g.to_bits(), want.to_bits(), "bias form, output {idx}: {g} against {want}");
    }

    // Split-K: four splits into Partial, then the `_reduce` entry sums them.
    let src = format!(
        "@tile({m}, {n}, {k})\nkernel k_splitk4(A: GlobalMemory<BF16>, B: GlobalMemory<BF16>, C: GlobalMemory<F32>, Partial: GlobalMemory<F32>) {{\n}}"
    );
    let ptx = ptx_at("splitk", &src, pinned::SM_PINNED);
    let head = ptx.lines().find(|l| l.contains("[Y SPLIT-K GEMM]")).expect("no split-K header");
    let nums = |after: &str| -> Vec<u32> {
        head.split(after).nth(1).unwrap().split(')').next().unwrap().split(", ").map(|v| v.trim().parse().unwrap()).collect()
    };
    let g = nums("grid (");
    let r = nums("_reduce, grid (");
    let (_, _, threads, smem) = geometry(&ptx);
    use y::cuda_runtime::CudaContext;
    let Some(ctx) = CudaContext::new() else {
        return skip("the BF16 GEMM's split-K form");
    };
    let main = ctx.load_ptx(&ptx, "k_splitk4").expect("PTX failed to load");
    let reduce = ctx.load_ptx(&ptx, "k_splitk4_reduce").expect("PTX failed to load");
    if smem > 48 * 1024 {
        main.set_max_dynamic_smem(smem).unwrap();
    }
    let bufs: Vec<_> = [as_bf16(&a), as_bf16(&b), poison.clone(), f32::NAN.to_le_bytes().repeat(4 * m * n)]
        .iter()
        .map(|init| {
            let d = ctx.alloc(init.len()).unwrap();
            ctx.memcpy_htod_at(&d, 0, init).unwrap();
            d
        })
        .collect();
    let args: Vec<u64> = bufs.iter().map(|b| b.device_ptr()).collect();
    ctx.launch(&main, (g[0], g[1], g[2]), (threads, 1, 1), smem, &args).expect("split launch failed");
    ctx.launch(&reduce, (r[0], 1, 1), (256, 1, 1), 0, &args).expect("reduce launch failed");
    ctx.synchronize().expect("kernels did not complete");
    let mut c = vec![0u8; 4 * m * n];
    ctx.memcpy_dtoh_at(&mut c, &bufs[2], 0).unwrap();
    let got: Vec<f32> = words(&c).into_iter().map(f32::from_bits).collect();
    let bad = got.iter().zip(&prod).filter(|(g, w)| g.to_bits() != w.to_bits()).count();
    assert_eq!(bad, 0, "split-K form: {bad} of {} outputs are not the integer product", m * n);
}

#[test]
fn the_type_checker_accepts_the_bf16_shapes_and_refuses_a_mixed_one() {
    let mixed = "@tile(64, 64, 64)\nkernel k(A: GlobalMemory<BF16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>) {\n}";
    refused("mixed", mixed, pinned::SM_PINNED, &["--emit-ptx"], "B (expected GlobalMemory<BF16>)");
    let swiglu = "@tile(64, 64, 64)\nkernel k(X: GlobalMemory<BF16>, Wg: GlobalMemory<BF16>, Wu: GlobalMemory<BF16>, Out: GlobalMemory<F32>) {\n}";
    refused("swiglu", swiglu, pinned::SM_PINNED, &["--emit-ptx"], "Wu (expected GlobalMemory<F32>)");
}
