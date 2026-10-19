//! Unsigned division and remainder by a constant, as the PTX emitter writes
//! them.
//!
//! `ptxas` lowers `div.u32 d, a, 4` and `rem.u32 d, a, 4` - an immediate, a
//! power of two - into a reciprocal division of about fifteen instructions
//! (`I2F.U32.RP`, `MUFU.RCP`, `F2I`, `IMAD.HI`, two correction steps), and it
//! does so after constant propagation, so a dividend known at compile time is
//! divided at run time too. 27 committed kernels, all GEMMs, held 8 to 57
//! such divisions each; a K loop took its stage index modulo the stage count
//! every K-tile. `strength_reduce_constant_division` now rewrites them
//! over the finished module; this file checks that the rewrite computes what
//! the division computed, on the device, for every form it produces and for
//! the forms it must leave alone.

use std::path::PathBuf;
use std::process::Command;

use y::ptx_emitter::{strength_reduce_constant_division as reduce, udiv32_magic};

#[path = "common/pinned.rs"]
mod pinned;

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The multiplier the pass uses gives `n / c` at the dividends where an
/// approximate one breaks first - next to multiples of `c`, and at the top of
/// the range - and at random ones, for every divisor up to 100000.
#[test]
fn the_multiplier_divides_exactly_where_it_exists() {
    let mut state = 7u64;
    let mut with_magic = 0u32;
    for c in 2u32..=100_000 {
        let Some((m, l)) = udiv32_magic(c) else { continue };
        with_magic += 1;
        let q = |n: u32| (((n as u64 * m as u64) >> 32) >> l) as u32;
        let top = u32::MAX / c;
        let mut ns = vec![0, 1, c - 1, c, c.wrapping_add(1), u32::MAX, u32::MAX - 1];
        for k in [1u32, 2, top.saturating_sub(1), top] {
            let base = k.wrapping_mul(c);
            ns.extend([base, base.wrapping_sub(1), base.wrapping_add(c - 1)]);
        }
        for _ in 0..64 {
            ns.push(splitmix(&mut state) as u32);
        }
        for n in ns {
            assert_eq!(q(n), n / c, "{n} / {c} with m = {m}, l = {l}");
        }
    }
    // Not a vacuous loop: most divisors have one, and the known exceptions
    // (7 needs a 33-bit multiplier) do not.
    assert!(with_magic > 50_000, "only {with_magic} divisors had a multiplier");
    assert_eq!(udiv32_magic(3), Some((0xAAAA_AAAB, 1)));
    assert_eq!(udiv32_magic(7), None);
    assert_eq!(udiv32_magic(0x7FFF_FFFF), None);
    assert_eq!(udiv32_magic(1), None);
    assert_eq!(udiv32_magic(0), None);
}

#[test]
fn each_form_is_rewritten_and_the_rest_is_left_alone() {
    let cases: &[(&str, &[&str])] = &[
        ("    div.u32 %r1, %r2, 8;", &["    shr.u32 %r1, %r2, 3;"]),
        ("    rem.u32 %r1, %r2, 8;", &["    and.b32 %r1, %r2, 7;"]),
        ("    div.u64 %rd1, %rd2, 4096;", &["    shr.u64 %rd1, %rd2, 12;"]),
        ("    rem.u64 %rd1, %rd2, 4294967296;", &["    and.b64 %rd1, %rd2, 4294967295;"]),
        ("    div.u32 %r1, %r2, 1;", &["    mov.b32 %r1, %r2;"]),
        ("    rem.u32 %r1, %r2, 1;", &["    mov.b32 %r1, 0;"]),
        ("    div.u32 %r1, %r2, 3;", &["    mul.hi.u32 %r1, %r2, 2863311531;", "    shr.u32 %r1, %r1, 1;"]),
        (
            "    rem.u32 %r1, %r2, 3;",
            &["    mul.hi.u32 %r1, %r2, 2863311531;", "    shr.u32 %r1, %r1, 1;", "    mul.lo.u32 %r1, %r1, 3;", "    sub.u32 %r1, %r2, %r1;"],
        ),
        ("    @%p3 rem.u32 %r1, %r2, 16;", &["    @%p3 and.b32 %r1, %r2, 15;"]),
        (
            "    @!%p0 rem.u32 %r1, %r2, 3;",
            &["    @!%p0 mul.hi.u32 %r1, %r2, 2863311531;", "    @!%p0 shr.u32 %r1, %r1, 1;", "    @!%p0 mul.lo.u32 %r1, %r1, 3;", "    @!%p0 sub.u32 %r1, %r2, %r1;"],
        ),
    ];
    for (line, want) in cases {
        let got = reduce(&format!("{line}\n"));
        assert_eq!(got, format!("{}\n", want.join("\n")), "{line}");
    }
    // Left exactly as written: a register divisor, signed division (it rounds
    // toward zero, so a shift is not it), zero, a divisor with no 32-bit
    // multiplier, a u64 divisor that is not a power of two, a remainder whose
    // destination is its dividend (the sequence would overwrite it before
    // its last read), a comment, and a u32 immediate out of range.
    for line in [
        "    div.u32 %r1, %r2, %r3;",
        "    rem.u32 %r1, %r2, %r3;",
        "    div.s32 %r1, %r2, 8;",
        "    rem.s32 %r1, %r2, 8;",
        "    div.u32 %r1, %r2, 0;",
        "    div.u32 %r1, %r2, 7;",
        "    rem.u32 %r1, %r2, 7;",
        "    div.u64 %rd1, %rd2, 3;",
        "    rem.u32 %r2, %r2, 3;",
        "    // div.u32 %r1, %r2, 8;",
        "    div.u32 %r1, %r2, 4294967296;",
        "    div.u32 %r1, %r2, 0x10;",
    ] {
        assert_eq!(reduce(&format!("{line}\n")), format!("{line}\n"), "{line}");
    }
    // Text without a trailing newline keeps that shape.
    assert_eq!(reduce("    div.u32 %r1, %r2, 2;"), "    shr.u32 %r1, %r2, 1;");
}

/// A kernel with nothing to rewrite comes back byte for byte.
#[test]
fn a_module_with_no_constant_division_is_unchanged() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for k in ["rope_128", "rmsnorm_residual_4096", "ptx_integer_ops"] {
        let ptx = std::fs::read_to_string(root.join(format!("tests/{k}.ptx"))).expect(k);
        assert_eq!(reduce(&ptx), ptx, "{k}");
    }
}

const U32_DIVISORS: [u32; 29] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 24, 25, 36, 40, 64, 72, 100, 641, 1000, 4096, 65535, 65536,
    65537, 1 << 20, 0x7FFF_FFFF, 1 << 31, 0xFFFF_FFFF,
];
const U64_DIVISORS: [u64; 7] = [1, 2, 8, 1 << 20, 1 << 32, 1 << 63, 3];

/// The device kernel: every input divided by every divisor, `div` and `rem`,
/// one output slot each, plus a predicated remainder (the guard must hold on
/// every instruction it becomes) and a remainder into its own dividend.
fn kernel() -> String {
    let mut body = String::new();
    let mut r = 10u32;
    let mut rd = 10u32;
    let mut slot = 0u32;
    let store32 = |body: &mut String, reg: &str, slot: &mut u32| {
        body.push_str(&format!(
            "    mad.wide.u32 %rd8, %r3, 4, %rd2;\n    mul.wide.u32 %rd9, %r4, {};\n    add.s64 %rd8, %rd8, %rd9;\n    st.global.u32 [%rd8], {reg};\n",
            *slot * 4
        ));
        *slot += 1;
    };
    for c in U32_DIVISORS {
        for op in ["div", "rem"] {
            let d = format!("%r{r}");
            r += 1;
            body.push_str(&format!("    {op}.u32 {d}, %r5, {c};\n"));
            store32(&mut body, &d, &mut slot);
        }
    }
    // Guarded: a < 2^31 ? a % 3 : 12345.
    let d = format!("%r{r}");
    r += 1;
    body.push_str(&format!(
        "    mov.u32 {d}, 12345;\n    setp.lt.u32 %p1, %r5, 2147483648;\n    @%p1 rem.u32 {d}, %r5, 3;\n"
    ));
    store32(&mut body, &d, &mut slot);
    let d = format!("%r{r}");
    r += 1;
    body.push_str(&format!(
        "    mov.u32 {d}, 54321;\n    @!%p1 div.u32 {d}, %r5, 24;\n"
    ));
    store32(&mut body, &d, &mut slot);
    // A remainder into its own dividend: left to ptxas, still right.
    let d = format!("%r{r}");
    r += 1;
    body.push_str(&format!("    mov.u32 {d}, %r5;\n    rem.u32 {d}, {d}, 3;\n"));
    store32(&mut body, &d, &mut slot);
    let slots32 = slot;
    // u64 slots follow, 8 bytes each, from a 64-bit input.
    for c in U64_DIVISORS {
        for op in ["div", "rem"] {
            let d = format!("%rd{rd}");
            rd += 1;
            body.push_str(&format!(
                "    {op}.u64 {d}, %rd7, {c};\n    mad.wide.u32 %rd8, %r3, 8, %rd2;\n    mul.wide.u32 %rd9, %r4, {};\n    add.s64 %rd8, %rd8, %rd9;\n    st.global.u64 [%rd8], {d};\n",
                slots32 * 4 + (slot - slots32) * 8
            ));
            slot += 1;
        }
    }
    format!(
        ".version 7.8\n.target sm_89\n.address_size 64\n\n.visible .entry cdiv(\n    .param .u64 A,\n    .param .u64 B,\n    .param .u64 O,\n    .param .u32 N\n)\n{{\n    .reg .b32 %r<{r}>;\n    .reg .b64 %rd<{rd}>;\n    .reg .pred %p<2>;\n    ld.param.u64 %rd0, [A];\n    ld.param.u64 %rd1, [B];\n    ld.param.u64 %rd2, [O];\n    ld.param.u32 %r4, [N];\n    cvta.to.global.u64 %rd0, %rd0;\n    cvta.to.global.u64 %rd1, %rd1;\n    cvta.to.global.u64 %rd2, %rd2;\n    mov.u32 %r0, %ctaid.x;\n    mov.u32 %r1, %ntid.x;\n    mov.u32 %r2, %tid.x;\n    mad.lo.u32 %r3, %r0, %r1, %r2;\n    mad.wide.u32 %rd5, %r3, 4, %rd0;\n    ld.global.u32 %r5, [%rd5];\n    mad.wide.u32 %rd6, %r3, 8, %rd1;\n    ld.global.u64 %rd7, [%rd6];\n{body}    ret;\n}}\n"
    )
}

/// What the kernel's slots must hold for one input pair: the CPU's `/` and `%`.
fn reference(a: u32, b: u64) -> (Vec<u32>, Vec<u64>) {
    let mut v32 = Vec::new();
    for c in U32_DIVISORS {
        v32.push(a / c);
        v32.push(a % c);
    }
    v32.push(if a < 1 << 31 { a % 3 } else { 12345 });
    v32.push(if a < 1 << 31 { 54321 } else { a / 24 });
    v32.push(a % 3);
    let mut v64 = Vec::new();
    for c in U64_DIVISORS {
        v64.push(b / c);
        v64.push(b % c);
    }
    (v32, v64)
}

/// The original module, which `ptxas` divides, and the rewritten one, both run
/// on the device against the CPU on full-range inputs and the boundary ones.
/// The original is the control: if the harness compared nothing, both would
/// pass; here the original must pass too, and the rewrite must actually have
/// removed the divisions it claims to.
#[test]
fn the_rewritten_divisions_match_the_cpu_on_the_gpu() {
    use y::cuda_runtime::CudaContext;
    let Some(ctx) = CudaContext::new() else {
        eprintln!("SKIP: no CUDA driver - the rewritten divisions were not executed.");
        return;
    };
    let original = kernel();
    let rewritten = reduce(&original);
    let count = |s: &str, pat: &str| s.lines().filter(|l| l.trim_start().contains(pat)).count();
    let divs = |s: &str| count(s, "div.u") + count(s, "rem.u");
    assert_eq!(divs(&original), 2 * U32_DIVISORS.len() + 3 + 2 * U64_DIVISORS.len());
    // Left: 7 and 2^31 - 1 (div and rem; neither has a 32-bit multiplier),
    // the remainder into its own dividend, and u64 by 3 (div and rem).
    assert_eq!(divs(&rewritten), 7, "{rewritten}");
    assert!(rewritten.contains("@%p1 mul.hi.u32") && rewritten.contains("@!%p1 shr.u32"), "a guard was dropped");

    const N: usize = 1 << 16;
    let mut a: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 23, 24, 25, 641, 65535, 65536, 65537, 0x7FFF_FFFE, 0x7FFF_FFFF, 0x8000_0000, 0x8000_0001, 0xFFFF_FFFD, 0xFFFF_FFFE, 0xFFFF_FFFF];
    let mut b: Vec<u64> = vec![0, 1, 2, 3, 7, 8, (1 << 20) - 1, 1 << 20, (1 << 32) - 1, 1 << 32, (1 << 32) + 1, (1 << 63) - 1, 1 << 63, u64::MAX - 1, u64::MAX];
    let mut state = 0xC0FF_EE00_1234_5678u64;
    while a.len() < N {
        a.push(splitmix(&mut state) as u32);
    }
    while b.len() < N {
        b.push(splitmix(&mut state));
    }
    let (r32, r64) = reference(0, 0);
    let (s32, s64) = (r32.len(), r64.len());
    let out_bytes = N * (s32 * 4 + s64 * 8);

    for (name, ptx) in [("original", &original), ("rewritten", &rewritten)] {
        let module = ctx.load_ptx(ptx, "cdiv").unwrap_or_else(|e| panic!("{name}: {e}"));
        let d_a = ctx.alloc(N * 4).unwrap();
        let d_b = ctx.alloc(N * 8).unwrap();
        let d_o = ctx.alloc(out_bytes).unwrap();
        let a_bytes: Vec<u8> = a.iter().flat_map(|x| x.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b.iter().flat_map(|x| x.to_le_bytes()).collect();
        ctx.memcpy_htod_at(&d_a, 0, &a_bytes).unwrap();
        ctx.memcpy_htod_at(&d_b, 0, &b_bytes).unwrap();
        ctx.memset_u8(&d_o, 0xA5).unwrap();
        let args = [d_a.device_ptr(), d_b.device_ptr(), d_o.device_ptr(), N as u64];
        ctx.launch(&module, ((N / 256) as u32, 1, 1), (256, 1, 1), 0, &args).expect("launch");
        ctx.synchronize().expect("kernel did not complete");
        let mut raw = vec![0u8; out_bytes];
        ctx.memcpy_dtoh_at(&mut raw, &d_o, 0).unwrap();
        let mut bad = 0usize;
        let mut first = None;
        for i in 0..N {
            let (w32, w64) = reference(a[i], b[i]);
            for (k, w) in w32.iter().enumerate() {
                let off = (k * N + i) * 4;
                let got = u32::from_le_bytes(raw[off..off + 4].try_into().unwrap());
                if got != *w {
                    bad += 1;
                    first.get_or_insert(format!("u32 slot {k}, a = {}: GPU {got}, CPU {w}", a[i]));
                }
            }
            for (k, w) in w64.iter().enumerate() {
                let off = s32 * N * 4 + (k * N + i) * 8;
                let got = u64::from_le_bytes(raw[off..off + 8].try_into().unwrap());
                if got != *w {
                    bad += 1;
                    first.get_or_insert(format!("u64 slot {k}, b = {}: GPU {got}, CPU {w}", b[i]));
                }
            }
        }
        assert_eq!(bad, 0, "{name}: {bad} of {} results wrong; first: {}", N * (s32 + s64), first.unwrap_or_default());
    }
}

/// What the emitter now writes for a GEMM: no division by an immediate that
/// the pass can reduce. The F16 GEMMs carried 10-57 each.
#[test]
fn the_emitted_gemm_divides_by_no_reducible_constant() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = pinned::pinned_scratch("cdiv_gemm", pinned::SM_PINNED);
    let src = pinned::copy_fixture(&dir, "tests/gemm_f16_512.ysu");
    let out = Command::new(env!("CARGO_BIN_EXE_Y")).arg(&src).arg("--emit-ptx").current_dir(&dir).output().expect("run Y");
    assert!(out.status.success(), "{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    let ptx = std::fs::read_to_string(src.with_extension("ptx")).expect("no .ptx");
    assert_eq!(reduce(&ptx), ptx, "the emitted module still holds a reducible division");
    // The control: the committed parent of this kernel had them, so the check
    // above is not passing on a kernel that never divided.
    let parent = std::fs::read_to_string(root.join("docs/benchmarks/f16_square_2026-10-18/512/ptx/64_64_32_4_2_4.ptx")).expect("parent");
    assert!(reduce(&parent) != parent, "the control kernel has no reducible division to remove");
    assert!(ptx.contains("mma.sync"), "not a GEMM");
}

/// The plain epilogue's copy loop stays rolled. With its divisions gone,
/// `ptxas -O3` unrolled it four times, which took ten GEMM kernels out of the
/// validator's loop census; the emitter marks it `.pragma "nounroll"`.
///
/// Checked on the SASS, as the census checks it: `ptxas` cannot invent a
/// global store, so a rolled loop holds exactly the PTX's stores and an
/// unrolled one more (16 against 2 here, without the pragma).
#[test]
fn the_plain_epilogue_loop_stays_rolled() {
    if Command::new("ptxas").arg("--version").output().is_err() {
        eprintln!("SKIP: no ptxas on PATH - the epilogue's unrolling was not checked.");
        return;
    }
    let dir = pinned::pinned_scratch("cdiv_rolled", pinned::SM_PINNED);
    let src = pinned::copy_fixture(&dir, "tests/gemm_f16_4096.ysu");
    let out = Command::new(env!("CARGO_BIN_EXE_Y")).arg(&src).arg("--emit-ptx").current_dir(&dir).output().expect("run Y");
    assert!(out.status.success(), "{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    let ptx_path = src.with_extension("ptx");
    let ptx = std::fs::read_to_string(&ptx_path).expect("no .ptx");
    let target = ptx.lines().find_map(|l| l.trim().strip_prefix(".target ")).expect("no .target").trim().to_string();
    let cubin = dir.join("k.cubin");
    let asm = Command::new("ptxas").arg(format!("-arch={target}")).arg(&ptx_path).arg("-o").arg(&cubin).output().expect("run ptxas");
    assert!(asm.status.success(), "{}", String::from_utf8_lossy(&asm.stderr));
    let dis = Command::new("cuobjdump").arg("-sass").arg(&cubin).output().expect("run cuobjdump");
    let sass = String::from_utf8_lossy(&dis.stdout);
    let ptx_stores = ptx.lines().filter(|l| l.contains("st.global.v4.f32")).count();
    let sass_stores = sass.lines().filter(|l| l.contains("STG.E.128")).count();
    assert!(ptx_stores > 0, "the control: the kernel has no epilogue store to count");
    assert_eq!(
        sass_stores, ptx_stores,
        "ptxas unrolled the plain epilogue ({sass_stores} SASS stores for {ptx_stores} in PTX)"
    );
}
