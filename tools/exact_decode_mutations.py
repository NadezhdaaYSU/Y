"""Negative controls for the decode-step proof and its executable check.

    PYTHONPATH=python python3 tools/exact_decode_mutations.py [--out FILE]

Each mutation is applied to a COPY of `python/`, `proofs/` and the two
`tools/exact_decode_*.py` scripts in a temporary directory; the working tree
is never edited. Then the gate the mutation must break is run against the
copy:

  * `coq`  - regenerate proofs/ExactDecodeConstants.v from the mutated
             Python, then compile the proof chain up to ExactDecodeStep.v
             (each `coqc` capped at 3 GB);
  * `exec` - run tools/exact_decode_check.py on the CPU (layers 0, 11, 23;
             two decode steps) with the mutated package.

An unmutated copy is run first through both gates and must PASS, so a
mutation that "fails" is not failing for an unrelated reason. Exit status 0
only if the control passes and every mutation is caught by every gate it
lists as `must_break`; a mutation that survives a gate is reported as such.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COQ_CHAIN = ["MixedRadix", "Decomposition", "ExactGemmSchedule", "ExactGemmMicro", "ExactPvExact",
             "NormGridSum", "ExactReduce", "ExactDecodeConstants", "ExactDecodeWidths", "ExactDecodeStep"]

# (name, what it plants, [(file, old, new)], gates that must break, gates run but allowed to survive)
MUTATIONS = [
    ("float32 denominator",
     "a float accumulation planted in a reduction: the softmax denominator summed in float32",
     [("python/y_lang/exact/attention.py",
       "    l = p.sum(dim=-1, keepdim=True)           # exact: int64, the TRUE denominator",
       "    l = p.to(torch.float32).sum(dim=-1, keepdim=True).to(torch.int64)")],
     ["exec"], []),
    ("float32 denominator in the proof",
     "the same float sum allowed by the proof's carrier table",
     [("proofs/ExactDecodeStep.v",
       "  | (NormHi | NormLo | Denom), ER.I64 => True",
       "  | (NormHi | NormLo | Denom), ER.I64 => True\n  | Denom, ER.F32 => True")],
     ["coq"], []),
    ("float32 linear accumulation",
     "a float accumulation planted in every linear layer: the int8 GEMM run as a float32 matmul",
     [("python/y_lang/exact/model.py",
       "                acc = torch._int_mm(xp, self.w8t)[:m].to(torch.float32)",
       "                acc = torch.matmul(x8.to(torch.float32), self.w8t.to(torch.float32))")],
     ["exec"], []),
    ("wrong index",
     "a wrong index: each query head reads the K scales of head h mod n_kv instead of h // rep",
     [("python/y_lang/exact/attention.py",
       "    sk = sk.transpose(-1, -2).repeat_interleave(rep, dim=1).to(torch.float64)",
       "    sk = sk.transpose(-1, -2).repeat(1, rep, 1, 1).to(torch.float64)")],
     ["exec"], []),
    ("digit ceiling widened",
     "a bound widened past overflow: two int8 digits declared to hold 32767",
     [("python/y_lang/exact/model.py",
       "DIGIT_MAX = 127 * 128 + 127          # 16383",
       "DIGIT_MAX = 255 * 128 + 127          # 32767")],
     ["coq"], ["exec"]),
    ("score budget widened",
     "a bound widened past overflow: the float32 score budget at 2^26 with K at 4095 levels",
     [("python/y_lang/exact/kv.py",
       "SCORE_BUDGET = 1 << 24          # fp32 holds every integer below this exactly",
       "SCORE_BUDGET = 1 << 26          # fp32 holds every integer below this exactly"),
      ("python/y_lang/exact/kv.py",
       'K_LEVELS = _levels_from_env("Y_EXACT_K_LEVELS", 1023, 127)',
       'K_LEVELS = _levels_from_env("Y_EXACT_K_LEVELS", 4095, 127)')],
     ["coq", "exec"], []),
    ("accumulator margin dropped",
     "the `- 128` dropped from act_levels_for, the bound on the interleaved digit sums",
     [("python/y_lang/exact/model.py",
       "    cap = min(ACC_BUDGET // max(k * w_levels, 1) - 128, DIGIT_MAX,",
       "    cap = min(ACC_BUDGET // max(k * w_levels, 1), DIGIT_MAX,")],
     ["coq"], []),
]


def copy_tree(dst):
    ign = shutil.ignore_patterns("__pycache__", "*.vo", "*.vok", "*.vos", "*.glob", ".*.aux", "*.so")
    shutil.copytree(os.path.join(ROOT, "python"), os.path.join(dst, "python"), symlinks=True, ignore=ign)
    shutil.copytree(os.path.join(ROOT, "proofs"), os.path.join(dst, "proofs"), ignore=ign)
    os.makedirs(os.path.join(dst, "tools"))
    for f in ("exact_decode_constants.py", "exact_decode_check.py"):
        shutil.copy(os.path.join(ROOT, "tools", f), os.path.join(dst, "tools", f))


def apply(dst, edits):
    for rel, old, new in edits:
        path = os.path.join(dst, rel)
        text = open(path).read()
        if text.count(old) != 1:
            raise SystemExit(f"mutation anchor not found exactly once in {rel}: {old!r}")
        open(path, "w").write(text.replace(old, new))


def env_for(dst):
    env = {k: v for k, v in os.environ.items() if not k.startswith("Y_EXACT_")}
    env.update(PYTHONPATH=os.path.join(dst, "python"), CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1")
    return env


def gate_coq(dst):
    """0 if the chain compiles; else the first failure."""
    env = env_for(dst)
    gen = subprocess.run([sys.executable, os.path.join(dst, "tools", "exact_decode_constants.py"), "--write"],
                         env=env, capture_output=True, text=True, timeout=600)
    if gen.returncode:
        return 1, "constants generator failed: " + gen.stderr[-400:]
    # The generator writes beside itself: ../proofs of the copy.
    pdir = os.path.join(dst, "proofs")
    for name in COQ_CHAIN:
        r = subprocess.run(["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=3G",
                            "-p", "MemorySwapMax=0", "timeout", "300", "coqc", "-Q", ".", "", f"{name}.v"],
                           cwd=pdir, capture_output=True, text=True)
        out = r.stdout + r.stderr
        if r.returncode or "Axioms:" in out:
            first = next((l for l in out.splitlines() if "Error" in l), out[-300:])
            return 1, f"{name}.v: {first.strip()}"
    return 0, "the chain compiles"


def gate_exec(dst):
    env = env_for(dst)
    r = subprocess.run([sys.executable, os.path.join(dst, "tools", "exact_decode_check.py"),
                        "--layers", "0", "11", "23", "--steps", "2"],
                       env=env, capture_output=True, text=True, timeout=1800)
    lines = [l for l in r.stdout.splitlines() if "compared tensors" in l or l.startswith("MISMATCH")]
    tail = lines[-1] if lines else (r.stderr.strip().splitlines() or ["no output"])[-1]
    n_mis = sum(l.startswith("MISMATCH") for l in lines)
    return r.returncode, (f"{n_mis} mismatches; " if n_mis else "") + tail[:300]


GATES = {"coq": gate_coq, "exec": gate_exec}


def run_case(base, name, edits, gates):
    dst = os.path.join(base, name.replace(" ", "_"))
    copy_tree(dst)
    apply(dst, edits)
    res = {}
    for g in gates:
        t0 = time.time()
        code, detail = GATES[g](dst)
        res[g] = {"broke": code != 0, "detail": detail, "seconds": round(time.time() - t0, 1)}
        print(f"  {g:4} {'BROKE' if code else 'held '}  {detail}", flush=True)
    shutil.rmtree(dst, ignore_errors=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None, help="run one mutation by name")
    args = ap.parse_args()
    base = tempfile.mkdtemp(prefix="y_decode_mut_", dir=os.environ.get("TMPDIR"))
    report = {"control": None, "mutations": []}
    ok = True
    try:
        print("control (no mutation): both gates must hold", flush=True)
        ctl = run_case(base, "control", [], ["coq", "exec"])
        report["control"] = ctl
        if any(r["broke"] for r in ctl.values()):
            print("CONTROL FAILED: the gates break without a mutation; nothing below means anything")
            ok = False
        for name, what, edits, must, may in MUTATIONS:
            if args.only and name != args.only:
                continue
            print(f"{name}: {what}", flush=True)
            res = run_case(base, name, edits, must + may)
            caught = all(res[g]["broke"] for g in must)
            ok &= caught
            report["mutations"].append({"name": name, "plants": what, "must_break": must,
                                        "also_run": may, "results": res, "caught": caught})
            print(f"  -> {'caught' if caught else 'NOT CAUGHT'}", flush=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
