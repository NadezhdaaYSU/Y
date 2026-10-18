//! `E4M3`: FP8 e4m3 (`torch.float8_e4m3fn`) as a `GlobalMemory` element, and
//! the `@tile` GEMM over e4m3 bytes.
//!
//! A store rounds an F32 to nearest-even and saturates at +-448 (NaN stays
//! NaN); a load widens exactly; an F64 or a literal is rounded once, not
//! through float32. An `E4M3` value - a `let`, a scalar parameter - is
//! refused, as `F16`'s is, and so is every backend without the conversion.
//! The conversions exist from sm_89, where ptxas wants PTX ISA 8.1.
//!
//! `@tile(M, N, K) kernel f(A, B: GlobalMemory<E4M3>, scale_a, scale_b: F32,
//! C: GlobalMemory<F32>)` computes `C = scale_a * scale_b * (A @ B^T)` with A
//! `[M][K]` and B `[N][K]`. The device checks here compare BYTES against an
//! exhaustive nearest-code reference; python/y_lang/tests/test_fp8_e4m3.py
//! sweeps the conversions against torch and runs the GEMM's tiles.

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
fn compile(tag: &str, src: &str, sm: &str, args: &[&str], env: &[(&str, &str)]) -> Emitted {
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let mut bin = std::env::current_exe().unwrap();
    bin.pop();
    if bin.ends_with("deps") {
        bin.pop();
    }
    let dir = std::env::temp_dir().join(format!(
        "y_e4m3_{}_{}_{}",
        std::process::id(),
        tag,
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("k.ysu");
    std::fs::write(&file, format!("{}\nfn main() {{}}\n", src)).unwrap();
    pinned::pin(&dir, sm);
    let mut cmd = Command::new(bin.join("Y"));
    cmd.arg(&file).args(args).current_dir(&dir).env_remove("Y_FP8_E4M3_TILE");
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("run Y");
    let log = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    let ptx = std::fs::read_to_string(dir.join("k.ptx")).ok();
    let _ = std::fs::remove_dir_all(&dir);
    Emitted { ok: out.status.success(), log, ptx }
}

fn ptx_at(tag: &str, src: &str, sm: &str) -> String {
    let e = compile(tag, src, sm, &["--emit-ptx"], &[]);
    assert!(e.ok, "{} must compile:\n{}", tag, e.log);
    e.ptx.unwrap_or_else(|| panic!("{} compiled but wrote no .ptx:\n{}", tag, e.log))
}

fn refused(tag: &str, src: &str, sm: &str, args: &[&str], env: &[(&str, &str)], why: &str) {
    let e = compile(tag, src, sm, args, env);
    assert!(!e.ok, "{} must be refused, and compiled:\n{}", tag, e.log);
    assert!(e.log.contains(why), "{} was refused, but not saying `{}`:\n{}", tag, why, e.log);
}

/// The value of e4m3 code `c` (sign bit clear).
fn e4m3_value(c: u8) -> f64 {
    let (exp, man) = ((c >> 3) & 0xF, f64::from(c & 7));
    if exp == 0 {
        man * 2f64.powi(-9)
    } else {
        (8.0 + man) * 2f64.powi(i32::from(exp) - 10)
    }
}

/// The e4m3 code nearest `x`, ties to the even code, saturating at 448;
/// `None` for NaN. Exhaustive over the 127 finite magnitudes, so it rests on
/// nothing but the format's definition.
fn e4m3_of(x: f64) -> Option<u8> {
    if x.is_nan() {
        return None;
    }
    let sign = if x.is_sign_negative() { 0x80 } else { 0 };
    let a = x.abs();
    let mut best = (0u8, f64::INFINITY);
    for c in 0u8..=0x7E {
        let d = (e4m3_value(c) - a).abs();
        if d < best.1 || (d == best.1 && c % 2 == 0) {
            best = (c, d);
        }
    }
    Some(sign | best.0)
}

/// Launches `k` on `threads` threads of one CTA with `shared` bytes of
/// dynamic shared memory; returns every buffer after it, or `None` without a
/// CUDA driver or an sm_89 GPU.
fn run(ptx: &str, grid: (u32, u32, u32), threads: u32, shared: u32, bufs: &[Vec<u8>], scalars: &[u64]) -> Option<Vec<Vec<u8>>> {
    use y::cuda_runtime::CudaContext;
    let ctx = CudaContext::new()?;
    // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR / _MINOR
    let cc = (ctx.device_attribute(75)?, ctx.device_attribute(76)?);
    if cc < (8, 9) {
        return None;
    }
    let module = ctx.load_ptx(ptx, "k").expect("PTX failed to load");
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
    eprintln!("SKIP: no CUDA driver or no sm_89 GPU - {} was not checked on the device.", what);
}

const POISON: u8 = 0xAB;

// ─────────────────────────── the element type ────────────────────────────

const STORE_EIGHT: &str = "kernel k(X: GlobalMemory<F32>, Q: GlobalMemory<E4M3>) {
    Q[0] = X[0];
    Q[1] = X[1];
    Q[2] = X[2];
    Q[3] = X[3];
    Q[4] = X[4];
    Q[5] = X[5];
    Q[6] = X[6];
    Q[7] = X[7];
}";

/// The rounding decisions: two ties (to even), a value just past one, the
/// saturation (finite and infinite), a subnormal tie each way, and NaN; the
/// ninth byte is never written.
#[test]
fn a_store_rounds_to_nearest_even_and_saturates() {
    let ptx = ptx_at("store", STORE_EIGHT, pinned::SM_FP8);
    let x: [f32; 8] = [
        1.0625,                        // tie between 1.0 and 1.125: even is 1.0
        f32::from_bits(1.0625f32.to_bits() + 1), // just above the tie: 1.125
        449.0,                         // past the largest finite: 448
        f32::NEG_INFINITY,             // -448, not NaN
        2f32.powi(-10),                // tie between 0 and 2^-9: 0
        3.0 * 2f32.powi(-10),          // tie between 2^-9 and 2^-8: 2^-8, the even code
        -0.0,
        f32::NAN,
    ];
    let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let Some(out) = run(&ptx, (1, 1, 1), 1, 0, &[bytes, vec![POISON; 9]], &[]) else {
        return skip("the E4M3 store");
    };
    let q = &out[1];
    for (i, v) in x.iter().enumerate() {
        match e4m3_of(f64::from(*v)) {
            Some(want) => assert_eq!(q[i], want, "{:?} stored {:#04x}, want {:#04x}", v, q[i], want),
            None => assert_eq!(q[i] & 0x7F, 0x7F, "NaN stored {:#04x}, not a NaN code", q[i]),
        }
    }
    assert_eq!(q[8], POISON, "a store wrote the byte after the eighth element");
}

/// Every byte widens to the value it codes, exactly; 0x7F and 0xFF to NaN.
#[test]
fn a_load_widens_every_byte_exactly() {
    let ptx = ptx_at(
        "load",
        "kernel k(Q: GlobalMemory<E4M3>, X: GlobalMemory<F32>) {\n    let i: I32 = thread_idx_x();\n    X[i] = Q[i];\n}",
        pinned::SM_FP8,
    );
    let all: Vec<u8> = (0..=255u8).collect();
    let Some(out) = run(&ptx, (1, 1, 1), 256, 0, &[all, vec![POISON; 1024]], &[]) else {
        return skip("the E4M3 load");
    };
    for c in 0..=255u8 {
        let got = f32::from_le_bytes(out[1][c as usize * 4..c as usize * 4 + 4].try_into().unwrap());
        if c & 0x7F == 0x7F {
            assert!(got.is_nan(), "{:#04x} widened to {}, not NaN", c, got);
            continue;
        }
        let mag = e4m3_value(c & 0x7F);
        let want = if c & 0x80 != 0 { -mag } else { mag };
        assert_eq!(got.to_bits(), (want as f32).to_bits(), "{:#04x} widened to {}, want {}", c, got, want);
    }
}

/// An F64, and a float literal (its exact double), round once: just above
/// e4m3's 1.0/1.125 tie by less than a float32 ulp is 1.125, where rounding
/// to float32 first lands on the tie and gives 1.0.
#[test]
fn an_f64_and_a_literal_round_once() {
    let ptx = ptx_at(
        "f64",
        "kernel k(V: GlobalMemory<F64>, Q: GlobalMemory<E4M3>) {\n    let a: F64 = V[0];\n    Q[0] = a;\n    let b: F64 = V[1];\n    Q[1] = b;\n    Q[2] = 1.0625000001;\n    Q[3] = 1.0625;\n}",
        pinned::SM_FP8,
    );
    let v = [1.0625 * (1.0 + 2f64.powi(-40)), -1.0625 * (1.0 + 2f64.powi(-40))];
    assert_eq!((v[0] as f32) as f64, 1.0625, "the fixture must sit inside one float32 ulp of the tie");
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let Some(out) = run(&ptx, (1, 1, 1), 1, 0, &[bytes, vec![POISON; 5]], &[]) else {
        return skip("the F64 store");
    };
    let one_125 = e4m3_of(1.125).unwrap();
    assert_eq!(&out[1][..4], &[one_125, 0x80 | one_125, one_125, e4m3_of(1.0).unwrap()], "{:02x?}", out[1]);
    assert_eq!(out[1][4], POISON);
}

/// Four e4m3 bytes in one `block_ptr2d_store_v4` and back through
/// `block_ptr2d_load_v4`; out of range, neither touches memory.
#[test]
fn the_v4_forms_move_four_bytes() {
    let ptx = ptx_at(
        "v4",
        "kernel k(X: GlobalMemory<F32>, Q: GlobalMemory<E4M3>, Out: GlobalMemory<F32>) {
    block_ptr2d_store_v4(Q, 0, 4, 8, 1, 8, X[0], X[1], X[2], X[3]);
    block_ptr2d_store_v4(Q, 1, 0, 8, 1, 8, X[0], X[1], X[2], X[3]);
    let v: U32x4 = block_ptr2d_load_v4(Q, 0, 4, 8, 1, 8);
    Out[0] = v.x;
    Out[1] = v.y;
    Out[2] = v.z;
    Out[3] = v.w;
}",
        pinned::SM_FP8,
    );
    assert!(ptx.contains("st.global.b32") && ptx.contains("ld.global.b32"), "{}", ptx);
    let x = [1.0f32, -2.5, 1000.0, 0.3];
    let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let Some(out) = run(&ptx, (1, 1, 1), 1, 0, &[bytes, vec![POISON; 16], vec![POISON; 16]], &[]) else {
        return skip("the E4M3 v4 forms");
    };
    let want: Vec<u8> = x.iter().map(|v| e4m3_of(f64::from(*v)).unwrap()).collect();
    assert_eq!(&out[1][..4], &[POISON; 4], "columns 0..3 of row 0 were not stored to");
    assert_eq!(&out[1][4..8], &want[..], "row 0, columns 4..7");
    assert_eq!(&out[1][8..], &[POISON; 8], "row 1 is past max_r and must not be written");
    for i in 0..4 {
        let got = f32::from_le_bytes(out[2][i * 4..i * 4 + 4].try_into().unwrap());
        let mag = e4m3_value(want[i] & 0x7F);
        assert_eq!(got as f64, if want[i] & 0x80 != 0 { -mag } else { mag }, "lane {}", i);
    }
}

/// `block_ptr2d_load`'s predicated load: an element past `max_r` reads as
/// zero, one inside it as its value.
#[test]
fn a_predicated_load_past_the_edge_reads_zero() {
    let ptx = ptx_at(
        "pred_load",
        "kernel k(Q: GlobalMemory<E4M3>, Out: GlobalMemory<F32>) {
    Out[0] = block_ptr2d_load(Q, 0, 1, 4, 1, 4);
    Out[1] = block_ptr2d_load(Q, 1, 1, 4, 1, 4);
}",
        pinned::SM_FP8,
    );
    // The zero is written by an instruction of its own: on the card an
    // unwritten register usually holds 0 already, so the device check below
    // passes without it. Every predicated byte load is followed by the move
    // that zeroes its register under the opposite predicate.
    let lines: Vec<&str> = ptx.lines().map(str::trim).collect();
    let loads: Vec<usize> = (0..lines.len()).filter(|&i| lines[i].contains("ld.global.u8")).collect();
    assert_eq!(loads.len(), 2, "{}", ptx);
    for i in loads {
        let (pred, rest) = lines[i].split_once(' ').unwrap();
        let reg = rest.split_whitespace().nth(1).unwrap().trim_end_matches(',');
        let want = format!("@!{} mov.b16 {}, 0;", pred.trim_start_matches('@'), reg);
        assert_eq!(lines[i + 1], want, "the load `{}` is not followed by its zeroing", lines[i]);
    }
    let q = vec![0u8, e4m3_of(-3.0).unwrap(), 0, 0, 0, e4m3_of(5.0).unwrap(), 0, 0];
    let Some(out) = run(&ptx, (1, 1, 1), 1, 0, &[q, vec![POISON; 8]], &[]) else {
        return skip("the predicated E4M3 load");
    };
    let at = |i: usize| f32::from_le_bytes(out[1][i * 4..i * 4 + 4].try_into().unwrap());
    assert_eq!((at(0), at(1)), (-3.0, 0.0), "row 1 is past max_r = 1, so it must read 0, not 5");
}

#[test]
fn the_conversions_need_sm_89_and_ptx_8_1() {
    let src = "kernel k(X: GlobalMemory<F32>, Q: GlobalMemory<E4M3>) {\n    Q[0] = X[0];\n    X[1] = Q[1];\n}";
    refused("sm80", src, pinned::SM_PINNED, &["--emit-ptx"], &[], "exist only on sm_89 (Ada) and later");
    let ptx = ptx_at("sm89", src, pinned::SM_FP8);
    assert!(ptx.starts_with(".version 8.1"), "{}", &ptx[..60]);
    assert!(ptx.contains("cvt.rn.satfinite.e4m3x2.f32") && ptx.contains("cvt.rn.f16x2.e4m3x2"), "{}", ptx);
    if Command::new("ptxas").arg("--version").output().is_err() {
        return eprintln!("SKIP: no ptxas - the module was not assembled");
    }
    let path = std::env::temp_dir().join(format!("y_e4m3_asm_{}.ptx", std::process::id()));
    std::fs::write(&path, &ptx).unwrap();
    let out = Command::new("ptxas").args(["-arch=sm_89", "-o", "/dev/null"]).arg(&path).output().unwrap();
    let _ = std::fs::remove_file(&path);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn an_e4m3_value_is_refused() {
    let why = "`E4M3` is a buffer element type, not a value type";
    refused("let", "kernel k(X: GlobalMemory<F32>) {\n    let v: E4M3 = X[0];\n    X[1] = v;\n}", pinned::SM_FP8, &["--emit-ptx"], &[], why);
    refused("param", "kernel k(s: E4M3, X: GlobalMemory<F32>) {\n    X[0] = 1.0;\n}", pinned::SM_FP8, &["--emit-ptx"], &[], why);
}

#[test]
fn backends_without_the_conversion_refuse_it() {
    let src = "kernel k(X: GlobalMemory<F32>, Q: GlobalMemory<E4M3>) {\n    Q[0] = X[0];\n}";
    refused("cpu", src, pinned::SM_FP8, &["--emit-cpu"], &[], "`GlobalMemory<E4M3>` cannot be lowered");
    refused("llvm", src, pinned::SM_FP8, &["--emit-llvm"], &[], "element type this backend does not know");
}

// ──────────────────────────── the e4m3 GEMM ─────────────────────────────

fn gemm_src(m: u32, n: u32, k: u32) -> String {
    format!(
        "@tile({m}, {n}, {k})\nkernel k(A: GlobalMemory<E4M3>, B: GlobalMemory<E4M3>, scale_a: F32, scale_b: F32, C: GlobalMemory<F32>) {{\n}}"
    )
}

fn launch_line(ptx: &str) -> ((u32, u32, u32), u32, u32) {
    let line = ptx.lines().find(|l| l.contains("Launch grid required")).expect("no launch line");
    let nums: Vec<u32> = line
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();
    let smem: u32 = ptx
        .split("Dynamic shared memory required: ")
        .nth(1)
        .and_then(|s| s.split(' ').next())
        .and_then(|s| s.parse().ok())
        .expect("no shared memory line");
    ((nums[0], nums[1], nums[2]), nums[3], smem)
}

/// A ragged GEMM - M and N not whole tiles, K one chunk past a whole K tile -
/// equal to the integer product scaled by 8, every element, C poisoned with
/// NaN first.
#[test]
fn the_gemm_is_exact_on_a_ragged_shape() {
    let (m, n, k) = (200u32, 136u32, 80u32);
    let ptx = ptx_at("gemm", &gemm_src(m, n, k), pinned::SM_FP8);
    assert!(ptx.contains("e4m3 inputs"), "the e4m3 GEMM was not the one emitted");
    let (grid, threads, smem) = launch_line(&ptx);
    let int = |i: u32, j: u32, s: u32| ((i * 7 + j * 3 + s) % 17) as i64 - 8;
    let a: Vec<i64> = (0..m * k).map(|x| int(x / k, x % k, 1)).collect();
    let b: Vec<i64> = (0..n * k).map(|x| int(x / k, x % k, 5)).collect();
    let enc = |v: &[i64]| -> Vec<u8> { v.iter().map(|&x| e4m3_of(x as f64).unwrap()).collect() };
    let c0: Vec<u8> = std::iter::repeat(f32::NAN.to_le_bytes()).take((m * n) as usize).flatten().collect();
    let scalars = [u64::from(2.0f32.to_bits()), u64::from(4.0f32.to_bits())];
    // Buffers in parameter order around the scalars: A, B, then C after them.
    let Some(out) = run_gemm(&ptx, grid, threads, smem, &enc(&a), &enc(&b), &c0, &scalars) else {
        return skip("the e4m3 GEMM");
    };
    for i in 0..m {
        for j in 0..n {
            let want: i64 = (0..k).map(|x| a[(i * k + x) as usize] * b[(j * k + x) as usize]).sum::<i64>() * 8;
            let at = ((i * n + j) * 4) as usize;
            let got = f32::from_le_bytes(out[at..at + 4].try_into().unwrap());
            assert_eq!(got, want as f32, "C[{}][{}]", i, j);
        }
    }
}

fn run_gemm(ptx: &str, grid: (u32, u32, u32), threads: u32, smem: u32, a: &[u8], b: &[u8], c: &[u8], scales: &[u64; 2]) -> Option<Vec<u8>> {
    use y::cuda_runtime::CudaContext;
    let ctx = CudaContext::new()?;
    if (ctx.device_attribute(75)?, ctx.device_attribute(76)?) < (8, 9) {
        return None;
    }
    let module = ctx.load_ptx(ptx, "k").expect("PTX failed to load");
    let bufs: Vec<_> = [a, b, c]
        .iter()
        .map(|init| {
            let d = ctx.alloc(init.len()).unwrap();
            ctx.memcpy_htod_at(&d, 0, init).unwrap();
            d
        })
        .collect();
    let args = [bufs[0].device_ptr(), bufs[1].device_ptr(), scales[0], scales[1], bufs[2].device_ptr()];
    ctx.launch(&module, grid, (threads, 1, 1), smem, &args).expect("launch failed");
    ctx.synchronize().expect("kernel did not complete");
    let mut out = vec![0u8; c.len()];
    ctx.memcpy_dtoh_at(&mut out, &bufs[2], 0).unwrap();
    Some(out)
}

#[test]
fn the_gemm_states_its_launch_and_needs_sm_89() {
    let ptx = ptx_at("gemm1024", &gemm_src(1024, 1024, 1024), pinned::SM_FP8);
    let (grid, threads, smem) = launch_line(&ptx);
    assert_eq!((grid, threads, smem), ((16, 16, 1), 128, 24576), "the default tile is 64x64x64 on 4 warps, 3 stages");
    assert!(ptx.starts_with(".version 8.4"), "the FP8 mma needs PTX 8.4");
    // Four CTAs of 128 threads share an SM's 65,536 registers.
    assert!(ptx.contains(".maxnreg 128\n"), "the default tile is sized for four CTAs per SM");
    refused("gemm_sm80", &gemm_src(1024, 1024, 1024), pinned::SM_PINNED, &["--emit-ptx"], &[], "exist only on sm_89");
}

#[test]
fn the_gemm_refuses_what_it_cannot_compute() {
    refused("k8", &gemm_src(64, 64, 72), pinned::SM_FP8, &["--emit-ptx"], &[], "need K a multiple of 16");
    // B in f32 beside an e4m3 A is neither form.
    refused(
        "mixed",
        "@tile(64, 64, 64)\nkernel k(A: GlobalMemory<E4M3>, B: GlobalMemory<F32>, scale_a: F32, scale_b: F32, C: GlobalMemory<F32>) {\n}",
        pinned::SM_FP8,
        &["--emit-ptx"],
        &[],
        "B (expected GlobalMemory<E4M3>)",
    );
    refused("malformed_tile", &gemm_src(64, 64, 64), pinned::SM_FP8, &["--emit-ptx"], &[("Y_FP8_E4M3_TILE", "128,128,64")], "Y_FP8_E4M3_TILE='128,128,64' is not");
    refused("smem", &gemm_src(64, 64, 64), pinned::SM_FP8, &["--emit-ptx"], &[("Y_FP8_E4M3_TILE", "128,128,64,4,2,7,1,8")], "past the 101376 a CTA can have");
    // Tiles the kernel cannot stage, each refused by name rather than
    // emitted: 2048 threads; one stage; one n8 tile per warp (an ldmatrix.x4
    // loads two); one warp on 128-byte rows, which copies 4 rows a pass, so a
    // thread's chunks would sit in rows of different swizzle; and an empty
    // launch group.
    for (tag, spec, why) in [
        ("threads", "512,128,64,16,4,2,1,8", "past a CTA's 1024"),
        ("one_stage", "128,128,64,4,2,1,1,8", "the K loop needs at least 2"),
        ("odd_n8", "64,32,64,2,4,2,1,8", "not whole pairs of n8 tiles"),
        ("swizzle", "32,32,128,1,1,2,1,8", "would differ in swizzle"),
        ("group0", "128,128,64,4,2,3,2,0", "must be at least 1"),
        // A K step of 48 is whole 16-wide fragments but not whole k32 ones:
        // refused by the warp tiling guard with this kernel's own fragment.
        ("cta_k48", "128,128,48,4,2,3,2,8", "cta_k=48 does not tile"),
    ] {
        refused(tag, &gemm_src(256, 256, 256), pinned::SM_FP8, &["--emit-ptx"], &[("Y_FP8_E4M3_TILE", spec)], why);
    }
}

/// Only the F16 tile GEMMs take their tile from the autotuner; the CLI used
/// to print an F16 tile for every `@tile` kernel.
#[test]
fn the_cli_reports_an_autotuned_tile_for_f16_only() {
    let e = compile("cli_e4m3", &gemm_src(256, 256, 256), pinned::SM_FP8, &["--emit-ptx"], &[]);
    assert!(e.ok && !e.log.contains("[JIT Autotuner Result]"), "{}", e.log);
    let f16 = "@tile(256, 256, 256)\nkernel k(A: GlobalMemory<F16>, B: GlobalMemory<F16>, C: GlobalMemory<F32>) {\n}";
    let e = compile("cli_f16", f16, pinned::SM_FP8, &["--emit-ptx"], &[]);
    assert!(e.ok && e.log.contains("[JIT Autotuner Result] `k`"), "{}", e.log);
}
