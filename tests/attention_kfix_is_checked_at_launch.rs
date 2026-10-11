//! Exact attention's temperature is checked by the kernel that reads it.
//!
//! `attn_accum` and `attn_accum_naive` take `KFix = round(C * 2^32)` as a
//! `.param .u32` and consume it with a SIGNED `mul.wide.s32`. Two ranges are
//! proved wrong answers in `proofs/SoftmaxErrorBound.v`: `KFix = 0` gives every
//! key the same weight (a uniform softmax), and at or above `2^31` the multiply
//! reads a negative number and keys get the weight of an unrelated argument.
//! `exact_attention::temperature_fixed_point` refuses both, but it is a host
//! helper a caller may skip, and the kernel took any `u32`: both values
//! launched successfully and returned an answer. The entries now trap outside
//! `1..2^31`.
//!
//! Each launch runs in a child process (this test, re-run with
//! `Y_KFIX_CASE`): a trap leaves the device unusable for the rest of its
//! process, so even a fresh context cannot be created after one.
//!
//! Run with: cargo test --release --test attention_kfix_is_checked_at_launch -- --nocapture

use std::process::Command;
use y::cuda_runtime::CudaContext;
use y::exact_attention::attention_ptx;

const D: usize = 32;
const S: usize = 256;
const B: usize = 2;

const ENTRIES: [&str; 2] = ["attn_accum", "attn_accum_naive"];

/// The values either side of both edges, and one ordinary temperature.
const CASES: [(u32, bool); 6] = [
    (0, false),
    (1, true),
    ((8 << 16) + 1, true),
    ((1 << 31) - 1, true),
    (1 << 31, false),
    (u32::MAX, false),
];

fn inputs() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let gen = |n: usize, a: usize, c: usize| -> Vec<u8> {
        (0..n).map(|i| ((((i * a + c) % 251) as i32 - 125) as i8) as u8).collect()
    };
    (gen(B * D, 37, 11), gen(S * D, 53, 7), gen(S * D, 97, 29))
}

fn to_i32(raw: &[u8]) -> Vec<i32> {
    raw.chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// One launch of `attn_scores` then `entry` at `kfix`, reported on stdout:
/// `TRAPPED`, or `RESULT` with whether every weight is the host exp of the
/// device's own score at that temperature.
fn run_case(entry: &str, kfix: u32) {
    let ctx = CudaContext::new().expect("a context");
    let ptx = attention_ptx(D, S).expect("D and S are within the exactness bound");
    let scores_mod = ctx.load_ptx(&ptx, "attn_scores").expect("attn_scores");
    let accum_mod = ctx.load_ptx(&ptx, entry).expect(entry);

    let (q, k, v) = inputs();
    let d_q = ctx.alloc(B * D).unwrap();
    let d_k = ctx.alloc(S * D).unwrap();
    let d_v = ctx.alloc(S * D).unwrap();
    let d_s = ctx.alloc(B * S * 4).unwrap();
    let d_m = ctx.alloc(B * 4).unwrap();
    let d_l = ctx.alloc(B * 8).unwrap();
    let d_o = ctx.alloc(B * D * 8).unwrap();
    let d_p = ctx.alloc(B * S * 4).unwrap();
    ctx.memcpy_htod_at(&d_q, 0, &q).unwrap();
    ctx.memcpy_htod_at(&d_k, 0, &k).unwrap();
    ctx.memcpy_htod_at(&d_v, 0, &v).unwrap();
    for buf in [&d_s, &d_m, &d_l, &d_o, &d_p] {
        ctx.memset_u8(buf, 0).unwrap();
    }

    let sargs = [d_q.device_ptr(), d_k.device_ptr(), d_s.device_ptr(), d_m.device_ptr()];
    ctx.launch(&scores_mod, ((S as u32).div_ceil(128), B as u32, 1), (128, 1, 1), 0, &sargs)
        .and_then(|_| ctx.synchronize())
        .expect("attn_scores takes no temperature and must run");

    let aargs = [
        d_s.device_ptr(),
        d_v.device_ptr(),
        d_m.device_ptr(),
        d_l.device_ptr(),
        d_o.device_ptr(),
        d_p.device_ptr(),
        kfix as u64,
    ];
    if let Err(e) = ctx
        .launch(&accum_mod, ((S as u32).div_ceil(128), B as u32, 1), (128, 1, 1), 0, &aargs)
        .and_then(|_| ctx.synchronize())
    {
        println!("TRAPPED {e}");
        return;
    }

    let mut raw = vec![0u8; B * S * 4];
    ctx.memcpy_dtoh_at(&mut raw, &d_s, 0).unwrap();
    let scores = to_i32(&raw);
    let mut raw_m = vec![0u8; B * 4];
    ctx.memcpy_dtoh_at(&mut raw_m, &d_m, 0).unwrap();
    let maxima: Vec<i32> = to_i32(&raw_m).iter().map(|&m| ((m as u32) ^ 0x8000_0000) as i32).collect();
    let mut raw_p = vec![0u8; B * S * 4];
    ctx.memcpy_dtoh_at(&mut raw_p, &d_p, 0).unwrap();
    let p = to_i32(&raw_p);

    // The kernel's arithmetic on the host, reading KFix as the unsigned
    // temperature it is meant to be.
    let mut agree = true;
    let mut distinct = std::collections::BTreeSet::new();
    for b in 0..B {
        for i in 0..S {
            let delta = maxima[b].wrapping_sub(scores[b * S + i]) as i64;
            let t = ((delta * kfix as i64 + 32768) >> 16).min(1 << 30) as u32;
            let want = y::fixed_exp::exp2_neg_q16_16(t) as i32;
            agree &= p[b * S + i] == want;
            distinct.insert(want);
        }
    }
    println!("RESULT agree={agree} distinct={}", distinct.len());
}

#[test]
fn a_temperature_out_of_range_stops_the_kernel() {
    if let Ok(case) = std::env::var("Y_KFIX_CASE") {
        let (entry, kfix) = case.split_once('|').unwrap();
        run_case(entry, kfix.parse().unwrap());
        return;
    }
    if CudaContext::new().is_none() {
        eprintln!("SKIP: no CUDA device, so no kernel was run");
        return;
    }
    for entry in ENTRIES {
        for (kfix, in_range) in CASES {
            let child = Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "a_temperature_out_of_range_stops_the_kernel", "--nocapture"])
                .env("Y_KFIX_CASE", format!("{entry}|{kfix}"))
                .output()
                .expect("run the launch in a child");
            let text = String::from_utf8_lossy(&child.stdout).into_owned();
            if in_range {
                assert!(
                    text.contains("RESULT agree=true"),
                    "{entry} at KFix = {kfix} (in range) did not compute the host's weights:\n{text}{}",
                    String::from_utf8_lossy(&child.stderr)
                );
                // An ordinary temperature must produce a softmax that is not
                // uniform, or agreeing with the host shows nothing.
                if kfix == (8 << 16) + 1 {
                    let n: usize = text.split("distinct=").nth(1).and_then(|r| r.split_whitespace().next())
                        .unwrap().parse().unwrap();
                    assert!(n > 10, "{entry} at KFix = {kfix}: only {n} distinct weights");
                }
            } else {
                assert!(
                    text.contains("TRAPPED"),
                    "{entry} at KFix = {kfix} (out of range) ran to completion: {text}"
                );
            }
        }
    }
}
