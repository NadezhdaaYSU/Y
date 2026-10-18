"""Mutants of the paged decode attention loop (lanes per token, warp blocks,
masking, lazy rescale, prefetch, in-warp combine). Each is applied to the
attn worktree's ptx_emitter.rs, built, and run against
python/y_lang/tests/test_paged_decode_attention.py; a mutant is killed when
test_paged_decode_attention.py (both classes, the second under memcheck)
fails. The original is restored by writing it back, touching it and
rebuilding.

    python3 mutate_attn.py [name ...]
"""
import os
import subprocess
import sys
import time

P = "/tmp/claude-1000/-home-yumin-NVME-files-YSU-engine-main-YSU-engine-main-src-Y-lang-Y/35eca213-da5a-4f11-9ada-c7b8b4745295/scratchpad"
W = f"{P}/attn"
E = f"{W}/src/ptx_emitter.rs"

MUTANTS = {
    # ---- masking of lanes past the warp's end ----
    "p_mask_dropped": ('                writeln!(&mut self.ptx_buffer, "    @!{} mov.f32 {}, 0f00000000;", pv, p_w).unwrap();',
                       '                let _ = (pv, &p_w);'),
    "valid_uses_le": ('"    setp.lt.s32 {}, {}, {};", pv, t_tok, w_end)', '"    setp.le.s32 {}, {}, {};", pv, t_tok, w_end)'),
    "clamp_dropped": ('"    min.s32 {}, {}, {};", t_ld, t_tok, f.w_last)', '"    mov.u32 {}, {};", t_ld, t_tok)'),
    # ---- the warp's token block ----
    "per_warp_not_whole_groups": ('"    mul.lo.s32 {}, {}, {};", per_warp, iters, groups)', '"    mov.u32 {}, {};", per_warp, iters)'),
    "w_end_unclamped": ('"    min.s32 {}, {}, {};", w_end, w_hi, t_end)', '"    mov.u32 {}, {};", w_end, w_hi)'),
    "n_tok_unclamped": ('"    max.s32 {}, {}, 0;", n_tok, n_tok)', '"    max.s32 {}, {}, -1000000;", n_tok, n_tok)'),
    # ---- lanes per token ----
    "bfly_crosses_tokens": ("            let mut off = lpt / 2;\n            while off >= 1 {", "            let mut off = 16;\n            while off >= 1 {"),
    "bfly_too_short": ("            let mut off = lpt / 2;\n            while off >= 1 {", "            let mut off = lpt / 4;\n            while off >= 1 {"),
    "sub_offset_is_lane": ('"    mul.lo.s32 {}, {}, {};", r, sub, epl_k)', '"    mul.lo.s32 {}, {}, {};", r, lane, epl_k)'),
    "kv_lane_stride_halved": ('"    mad.lo.s32 {}, {}, {}, {};", s3, f.sub, f.epl_k, s2)', '"    mad.lo.s32 {}, {}, {}, {};", s3, f.sub, f.epl_k / 2, s2)'),
    "q_in_merge_layout": ("        let q_byte_off = if groups == 1 {\n            byte_off.clone()", "        let q_byte_off = if true {\n            byte_off.clone()"),
    "smem_store_merge_layout": ('"    add.s32 {}, {}, {};", acc_off2, acc_off, sub_off_k)', '"    add.s32 {}, {}, {};", acc_off2, acc_off, lane_off)'),
    "smem_store_unpredicated": ('"    setp.lt.u32 {}, {}, {};", pg, lane, lpt)', '"    setp.lt.u32 {}, {}, {};", pg, lane, 32)'),
    # ---- lazy rescale ----
    "rescale_acc_dropped": ('                writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", accs[h][j], accs[h][j], corr).unwrap();\n',
                            '                let _ = &corr;\n'),
    "rescale_l_dropped": ('            writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", l_regs[h], l_regs[h], corr).unwrap();\n            for j in 0..epl_k as usize {\n                writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", accs[h][j], accs[h][j], corr)',
                          '            for j in 0..epl_k as usize {\n                writeln!(&mut self.ptx_buffer, "    mul.f32 {}, {}, {};", accs[h][j], accs[h][j], corr)'),
    "rescale_d_stale": ('            writeln!(&mut self.ptx_buffer, "    mov.f32 {}, {};", m_regs[h], m_new).unwrap();\n            writeln!(&mut self.ptx_buffer, "    sub.f32 {}, {}, {};", ds[h], scores[h], m_regs[h]).unwrap();',
                        '            writeln!(&mut self.ptx_buffer, "    mov.f32 {}, {};", m_regs[h], m_new).unwrap();'),
    "rescale_never": ('"    @!{} bra ATTN_NO_RESCALE_{};", need_w, kernel_name)', '"    bra ATTN_NO_RESCALE_{};", kernel_name)'),
    "vote_is_lane_local": ('"    vote.sync.any.pred {}, {}, 0xffffffff;", need_w, need)', '"    mov.pred {}, {};", need_w, need)'),
    "need_skips_last_head": ("            t_max = Some(t);\n            ds.push(d);", "            if h + 1 < heads_per_cta || heads_per_cta == 1 { t_max = Some(t); }\n            ds.push(d);"),
    "need_inverted": ('"    setp.ne.b32 {}, {}, 0x{:08X};", need, t_max.unwrap()', '"    setp.eq.b32 {}, {}, 0x{:08X};", need, t_max.unwrap()'),
    "threshold_100": ("const ATTN_LAZY_RESCALE: f32 = 8.0;", "const ATTN_LAZY_RESCALE: f32 = 100.0;"),
    # ---- prefetch ----
    "prefetch_not_rotated": ("        for (cur, nx) in k_words.iter().zip(&k_next).chain(v_words.iter().zip(&v_next)) {\n            writeln!(&mut self.ptx_buffer, \"    mov.b32 {}, {};\", cur, nx).unwrap();\n        }",
                             "        let _ = (&k_next, &v_next);"),
    "loop_test_dropped": ('"    @{} bra ATTN_LOOP_END_{};", p_done, kernel_name)', '"    // @{} bra ATTN_LOOP_END_{};", p_done, kernel_name)'),
    "prefetch_same_token": ('"    add.s32 {}, {}, {};", t_n, t_w, groups)', '"    add.s32 {}, {}, {};", t_n, t_w, 0)'),
    "prologue_skip_dropped": ('"    @{} bra ATTN_LOOP_END_{};", p_none, kernel_name)', '"    // @{} bra ATTN_LOOP_END_{};", p_none, kernel_name)'),
    # ---- page lookup ----
    "page_reload_inverted": ('"    setp.ne.s32 {}, {}, {};", p_new_page, page, f.cached_page)', '"    setp.eq.s32 {}, {}, {};", p_new_page, page, f.cached_page)'),
    "page_cache_not_updated": ('"    mov.u32 {}, {};", f.cached_page, page)', '"    // mov.u32 {}, {};", f.cached_page, page)'),
    # ---- in-warp combine ----
    "combine_skipped": ("        let mut dist = lpt;\n        while dist < 32 {", "        let mut dist = 32;\n        while dist < 32 {"),
    "combine_one_step": ("        let mut dist = lpt;\n        while dist < 32 {", "        let mut dist = lpt * 2;\n        while dist < 32 {"),
    "combine_other_acc_dropped": ('"    fma.rn.f32 {}, {}, {}, {};", accs[h][j], a_o, c_o, a_s)', '"    mov.f32 {}, {};", accs[h][j], a_s)'),
    "combine_other_l_dropped": ('"    fma.rn.f32 {}, {}, {}, {};", l_regs[h], l_o, c_o, l_s)', '"    mov.f32 {}, {};", l_regs[h], l_s)'),
    "combine_m_not_updated": ('                writeln!(&mut self.ptx_buffer, "    mov.f32 {}, {};", m_regs[h], m_n).unwrap();\n            }\n            dist *= 2;',
                              '            }\n            dist *= 2;'),
    # ---- the layout policy (perf only; any value is correct) ----
    "policy_always_32_lanes": ("        epl.max(head_dim / 32)\n", "        let _ = epl;\n        head_dim / 32\n"),
}

EXPECTED_SURVIVORS = {
    # after the combine every group holds the same state: group 1 writes the
    # same values to the same addresses.
    "smem_store_unpredicated",
    # w_end lies in [1, seq_len] whenever the loop is reached, so an empty
    # warp's clamped first token is a real one: a wasted fetch, in bounds
    # (memcheck agrees), not a wrong answer.
    "prologue_skip_dropped",
    # a reload every token instead of every page: slower, not wrong.
    "page_cache_not_updated",
    # the old layout is a correct layout.
    "policy_always_32_lanes",
    # every lane of a token group holds the same score after the butterfly,
    # so a per-lane rescale decision is the same within a group and only
    # costs divergence; the combine already merges groups with different m.
    "vote_is_lane_local",
}


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, cwd=W, capture_output=True, text=True, **kw)


def build():
    r = sh("cargo build --release 2>&1 | tail -3")
    return "error" not in r.stdout


def run_test():
    """A hang (a loop that never advances) counts as a kill; the whole
    process group goes, so no kernel is left running on the GPU."""
    env = dict(os.environ, Y_LIB_PATH=f"{W}/target/release/liby.so", PYTHONPATH="python")
    p = subprocess.Popen(["python3", "-m", "unittest", "discover", "-s", "python/y_lang/tests", "-p",
                          "test_paged_decode_attention.py"], cwd=W, env=env, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        out, _ = p.communicate(timeout=300)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, 9)
        p.wait()
        return False, "TIMEOUT (hang) after 300 s"
    open(f"{P}/pa/mut_logs/{CUR[0]}.log", "w").write(out)
    tail = out.strip().splitlines()
    fail = [l for l in tail if "AssertionError" in l or "Error" in l]
    return p.returncode == 0, (fail[:1] or tail[-1:] or [""])[0][:150]


CUR = [""]


def main():
    os.makedirs(f"{P}/pa/mut_logs", exist_ok=True)
    names = sys.argv[1:] or list(MUTANTS)
    orig = open(E).read()
    results = []
    try:
        for n in names:
            old, new = MUTANTS[n]
            c = orig.count(old)
            if c != 1:
                results.append((n, "NOT APPLIED (%d matches)" % c, ""))
                print(results[-1], flush=True)
                continue
            CUR[0] = n
            open(E, "w").write(orig.replace(old, new))
            t = time.time()
            if not build():
                results.append((n, "BUILD FAILED", ""))
            else:
                ok, why = run_test()
                verdict = "SURVIVED" if ok else "killed"
                if n in EXPECTED_SURVIVORS:
                    verdict += " (expected survivor)"
                results.append((n, verdict, why))
            print("%-28s %-30s %4.0fs  %s" % (results[-1][0], results[-1][1], time.time() - t, results[-1][2]), flush=True)
    finally:
        open(E, "w").write(orig)
        os.utime(E, None)
        print("restored; rebuild:", build(), flush=True)
    bad = [r for r in results if r[1] == "SURVIVED" or r[1].startswith("NOT") or r[1].startswith("BUILD")]
    print("\n%d mutants, %d unexpected survivors / not applied: %s" % (len(results), len(bad), [b[0] for b in bad]))


if __name__ == "__main__":
    main()
