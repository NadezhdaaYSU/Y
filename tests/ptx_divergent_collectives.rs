//! A warp shuffle or a block barrier that only some threads reach is refused.
//!
//! `warp_reduce_sum`, `warp_reduce_max` and `shfl_sync_bfly` emit `shfl.sync`
//! with the full mask, which all 32 lanes of a warp must execute together;
//! `barrier_sync` emits `bar.sync`, which every thread of the block must reach.
//! Until 2026-10-12 the PTX backend emitted both wherever they were written,
//! `if t < 20 { warp_reduce_sum(x) }` included, and the GPU leaves that
//! undefined: the shuffle reads registers of lanes that are not there, and the
//! barrier may hang. Only the CPU interpreter stopped at it, when run.
//!
//! Every refused kernel here compiled before. The accepted ones are the same
//! computations written where every thread runs them, or under conditions the
//! threads agree on, and run on the device with an exact answer (small
//! integers, so every float sum is exact in any order).

use std::path::PathBuf;
use std::process::Command;

#[path = "common/pinned.rs"]
mod pinned;

fn compile(tag: &str, src: &str) -> (bool, String, PathBuf) {
    let dir = pinned::pinned_scratch(&format!("divergent_{tag}"), pinned::SM_PINNED);
    let path = dir.join("k.ysu");
    std::fs::write(&path, format!("{src}\nfn main() {{}}\n")).unwrap();
    let ptx = dir.join("k.ptx");
    let out = Command::new(env!("CARGO_BIN_EXE_Y"))
        .arg(&path)
        .args(["--emit-ptx", "-o"])
        .arg(&ptx)
        .current_dir(&dir)
        .output()
        .expect("run Y");
    let log = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    (out.status.success(), log, ptx)
}

/// The kernel every refused case starts with: line 2 binds `t`, and the
/// case's own lines start at line 3.
fn kernel(body: &str) -> String {
    format!(
        "kernel k(A: GlobalMemory<F32>, Out: GlobalMemory<F32>, N: I32) {{\n    \
         let t: I32 = global_thread_id();\n{body}}}\n"
    )
}

#[test]
fn a_collective_some_threads_skip_is_refused() {
    let shuffle = ["this statement emits `shfl.sync.bfly.b32`", "all 32 lanes of a warp"];
    let barrier = ["this statement emits `bar.sync`", "every thread of the block must reach"];
    let cases: &[(&str, &str, &[&str], &[&str])] = &[
        (
            "sum",
            "    if t < 20 {\n        Out[t] = warp_reduce_sum(A[t]);\n    }\n",
            &shuffle,
            &["Line 4:", "under the `if` on line 3, whose condition reads `t` (from `global_thread_id()` on line 2)",
              "some lanes of a warp and not others"],
        ),
        ("max", "    if t < 20 {\n        let s: F32 = warp_reduce_max(A[t]);\n        Out[t] = s;\n    }\n", &shuffle, &["Line 4:"]),
        ("bfly", "    if t < 20 {\n        Out[t] = shfl_sync_bfly(A[t], 1);\n    }\n", &shuffle, &["Line 4:"]),
        ("bfly_b32", "    if t < 20 {\n        let s: I32 = shfl_sync_bfly_b32(N, 16);\n        Out[t] = A[s];\n    }\n", &shuffle, &["Line 4:"]),
        ("barrier", "    if t < 16 {\n        barrier_sync();\n    }\n", &barrier, &["Line 4:", "the `if` on line 3"]),
        ("membar", "    if t < 16 {\n        membar();\n    }\n", &barrier, &["Line 4:"]),
        ("barrier_path", "    if t < 16 {\n        barrier::sync();\n    }\n", &barrier, &["Line 4:"]),
        (
            "else",
            "    if t < 20 {\n        Out[t] = 1.0;\n    } else {\n        Out[t] = warp_reduce_sum(A[t]);\n    }\n",
            &shuffle,
            &["Line 6:", "the `if` on line 3"],
        ),
        (
            "after_return",
            "    if t >= N {\n        return;\n    }\n    Out[t] = warp_reduce_sum(A[t]);\n",
            &shuffle,
            &["Line 6:", "after the `return` on line 4, which only some threads take"],
        ),
        (
            "for_bound",
            "    @invariant(i >= 0)\n    for i in 0..t {\n        Out[t] = warp_reduce_sum(A[i]);\n    }\n",
            &shuffle,
            &["Line 5:", "in the `for` on line 4, whose bounds read `t`"],
        ),
        (
            "while_condition",
            "    let i: I32 = 0;\n    @invariant(i >= 0)\n    while i < t {\n        Out[t] = warp_reduce_sum(A[i]);\n        i = i + 1;\n    }\n",
            &shuffle,
            &["Line 6:", "in the `while` on line 5, whose condition reads"],
        ),
        (
            "assigned_in_branch",
            "    let c: I32 = 0;\n    if t < 5 {\n        c = 1;\n    }\n    if c == 1 {\n        Out[t] = warp_reduce_sum(A[t]);\n    }\n",
            &shuffle,
            &["Line 8:", "the `if` on line 7, whose condition reads `c` (assigned on line 5 inside the `if` on line 4)"],
        ),
        (
            // `c` differs only from the second iteration on, so `d`, set
            // under a test of `c` before `c = t`, differs too: one pass in
            // source order would call `d` uniform.
            "loop_carried",
            "    let c: I32 = 0;\n    let d: I32 = 0;\n    @invariant(i >= 0)\n    for i in 0..N {\n        if c > 0 {\n            \
             d = 1;\n        }\n        c = t;\n    }\n    if d == 1 {\n        Out[t] = warp_reduce_sum(A[t]);\n    }\n",
            &shuffle,
            &["Line 13:", "the `if` on line 12, whose condition reads `d` (assigned on line 8 inside the `if` on line 7)"],
        ),
        (
            // Another thread may store to `A[0]` between two warps' loads.
            "memory_read",
            "    if A[0] > 0.0 {\n        Out[t] = warp_reduce_sum(A[t]);\n    }\n",
            &shuffle,
            &["Line 4:", "whose condition reads a value loaded from memory"],
        ),
        (
            "barrier_under_warp_value",
            "    let s: F32 = warp_reduce_sum(A[t]);\n    if s > 0.0 {\n        barrier_sync();\n    }\n",
            &barrier,
            &["Line 5:", "reads `s` (from `warp_reduce_sum()` on line 3)", "some warps of the block and not others"],
        ),
        (
            // The shuffle comes before the `return`, but the threads that
            // took it are gone from the next iteration's.
            "loop_with_exit",
            "    @invariant(i >= 0)\n    for i in 0..N {\n        Out[t] = warp_reduce_sum(A[i]);\n        if A[t] > 0.0 {\n            return;\n        }\n    }\n",
            &shuffle,
            &["Line 5:", "in the loop on line 4, which threads can leave at different iterations (through the exit on line 7)"],
        ),
        (
            "after_branching_chisel",
            "    chisel {\n        \"bra.uni Y_CHISEL_NEXT;\";\n        \"Y_CHISEL_NEXT:\";\n    }\n    Out[t] = warp_reduce_sum(A[t]);\n",
            &shuffle,
            &["Line 7:", "after the `chisel` block on line 3, which may branch or return"],
        ),
        (
            "nested_condition",
            "    if t < 20 {\n        if warp_reduce_sum(A[t]) > 0.0 {\n            Out[t] = 1.0;\n        }\n    }\n",
            &shuffle,
            &["Line 4:"],
        ),
        (
            // A built-in whose lowering holds the shuffles and the barriers.
            "rmsnorm_in_branch",
            "    if t < 20 {\n        rmsnorm_v4(A[t], A[0], Out[t], N, 0.5);\n    }\n",
            &shuffle,
            &["Line 4:"],
        ),
    ];
    for (tag, body, kind, phrases) in cases {
        let src = kernel(body);
        let (ok, log, _) = compile(tag, &src);
        assert!(!ok, "{tag}: compiled:\n{src}\n{log}");
        for p in kind.iter().chain(phrases.iter()) {
            assert!(log.contains(p), "{tag}: the refusal does not say {p:?}:\n{src}\n{log}");
        }
    }
}

/// Raw 32-bit words in, raw words out: `Out` (the last buffer) is filled
/// with `fill` first, so a word no thread writes shows.
fn run(ptx: &PathBuf, grid: u32, block: u32, inputs: &[Vec<u32>], out_len: usize, scalars: &[u64]) -> Vec<u32> {
    let ctx = y::cuda_runtime::CudaContext::new().expect("a context");
    let module = ctx.load_ptx(&std::fs::read_to_string(ptx).unwrap(), "k").expect("load");
    let bytes = |v: &[u32]| v.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>();
    let mut args = Vec::new();
    let mut keep = Vec::new();
    for v in inputs {
        let d = ctx.alloc(v.len() * 4).unwrap();
        ctx.memcpy_htod_at(&d, 0, &bytes(v)).unwrap();
        args.push(d.device_ptr());
        keep.push(d);
    }
    let out = ctx.alloc(out_len * 4).unwrap();
    ctx.memset_u8(&out, 0xff).unwrap();
    args.push(out.device_ptr());
    args.extend_from_slice(scalars);
    ctx.launch(&module, (grid, 1, 1), (block, 1, 1), 0, &args).expect("launch");
    ctx.synchronize().expect("run");
    let mut raw = vec![0u8; out_len * 4];
    ctx.memcpy_dtoh_at(&mut raw, &out, 0).unwrap();
    raw.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn small_ints(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 7 + 3) % 11) as f32 - 5.0).collect()
}

/// A float sum from `+0.0`, as the kernels start theirs: Rust's `sum()`
/// starts from `-0.0`, so a sum of nothing is `-0.0` there and `+0.0` here.
fn sum(v: impl Iterator<Item = f32>) -> f32 {
    v.fold(0.0, |a, b| a + b)
}

fn words(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

#[test]
fn the_same_reductions_where_the_threads_agree_compile_and_are_exact() {
    // The rewrite the refusal suggests: a neutral value under the branch,
    // the reduction outside it.
    let masked = "kernel k(A: GlobalMemory<F32>, Out: GlobalMemory<F32>, N: I32) {\n    \
        let t: I32 = global_thread_id();\n    let v: F32 = 0.0;\n    if t < N {\n        v = A[t];\n    }\n    \
        Out[t] = warp_reduce_sum(v);\n}\n";
    // Branches every thread of the block, or every lane of the warp, agrees
    // on: a block index, a warp's reduction, a parameter's trip count, and a
    // `while` on a flag set only under a test of a warp's sum. (The
    // invariant prover reads integer conditions only, hence the flag.)
    let agreed = "kernel k(A: GlobalMemory<F32>, Out: GlobalMemory<F32>, N: I32) {\n    \
        let t: I32 = global_thread_id();\n    let x: F32 = A[t];\n    let s: F32 = warp_reduce_sum(x);\n    \
        let r: F32 = 0.0;\n    if block_idx_x() == 1 {\n        r = warp_reduce_max(x);\n    }\n    \
        if s > 0.0 {\n        r = r + warp_reduce_sum(x + x);\n    }\n    \
        @invariant(i >= 0)\n    for i in 0..N {\n        r = r + shfl_sync_bfly(x, 1);\n    }\n    \
        let n: I32 = 0;\n    let total: F32 = warp_reduce_sum(x);\n    let more: I32 = 0;\n    \
        if total < 100.0 {\n        more = 1;\n    }\n    \
        @invariant(n >= 0)\n    while more == 1 && n < 64 {\n        x = x + 1.0;\n        n = n + 1;\n        \
        total = warp_reduce_sum(x);\n        more = 0;\n        if total < 100.0 {\n            more = 1;\n        }\n    }\n    \
        Out[t] = r + x;\n}\n";
    // A barrier after a `return` every thread of a block takes or none does.
    let block_return = "kernel k(Out: GlobalMemory<U32>, M: I32) {\n    if block_idx_x() >= M {\n        return;\n    }\n    \
        let smem: U64 = shared_alloc_u32(256);\n    let t: I32 = thread_idx_x();\n    \
        let base: I32 = block_idx_x() * 64 + t;\n    shared_store_v4(smem, t, base, base + 1, base + 2, base + 3);\n    \
        barrier_sync();\n    let v: U32x4 = shared_load_v4(smem, 63 - t);\n    let v0: U32 = v.x;\n    Out[base] = v0;\n}\n";

    let compiled: Vec<PathBuf> = [("masked", masked), ("agreed", agreed), ("block_return", block_return)]
        .iter()
        .map(|(tag, src)| {
            let (ok, log, ptx) = compile(tag, src);
            assert!(ok, "{tag} was refused:\n{src}\n{log}");
            ptx
        })
        .collect();

    if y::cuda_runtime::CudaContext::new().is_none() {
        eprintln!("SKIP: no CUDA device, so the accepted kernels were compiled and not run");
        return;
    }

    // 128 threads; N = 50: warp 0 sums 32 values, warp 1 eighteen, warps 2-3 none.
    let a = small_ints(128);
    let got = run(&compiled[0], 2, 64, &[words(&a)], 128, &[50]);
    for t in 0..128 {
        let w = t / 32;
        let want = sum((w * 32..(w + 1) * 32).filter(|&i| i < 50).map(|i| a[i]));
        assert_eq!(got[t], want.to_bits(), "masked: thread {t}");
    }

    // The same arithmetic on the host, warp by warp.
    let (n_iter, a) = (3u64, small_ints(128));
    let got = run(&compiled[1], 2, 64, &[words(&a)], 128, &[n_iter]);
    let mut warps_with_s_positive = 0;
    for w in 0..4 {
        let x: Vec<f32> = a[w * 32..(w + 1) * 32].to_vec();
        let s = sum(x.iter().copied());
        warps_with_s_positive += (s > 0.0) as usize;
        let block = (w * 32) / 64;
        let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut x_end = x.clone();
        let mut n = 0;
        while sum(x_end.iter().copied()) < 100.0 && n < 64 {
            x_end.iter_mut().for_each(|v| *v += 1.0);
            n += 1;
        }
        for lane in 0..32 {
            let mut r = if block == 1 { max } else { 0.0 };
            if s > 0.0 {
                r += sum(x.iter().map(|v| v + v));
            }
            for _ in 0..n_iter {
                r += x[lane ^ 1];
            }
            let t = w * 32 + lane;
            assert_eq!(got[t], (r + x_end[lane]).to_bits(), "agreed: thread {t} (warp {w}, s = {s})");
        }
    }
    assert!(warps_with_s_positive > 0 && warps_with_s_positive < 4, "the inputs take both sides of `s > 0.0`");

    // Three blocks, the third returns: its words keep the fill.
    let got = run(&compiled[2], 3, 64, &[], 192, &[2]);
    for i in 0..192 {
        let (b, t) = (i / 64, i % 64);
        let want = if b < 2 { (b * 64 + 63 - t) as u32 } else { u32::MAX };
        assert_eq!(got[i], want, "block_return: word {i}");
    }
}
