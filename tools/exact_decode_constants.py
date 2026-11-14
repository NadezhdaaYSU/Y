"""Emit proofs/ExactDecodeConstants.v from the constants in python/y_lang/exact.

    PYTHONPATH=python python3 tools/exact_decode_constants.py            # print
    PYTHONPATH=python python3 tools/exact_decode_constants.py --write    # rewrite

The Coq proofs of the exact decode path (proofs/ExactDecodeWidths.v) take
their constants from the file this writes, and their width functions are
checked against the samples recorded here. python/tests/test_exact_decode_constants.py runs this
and compares the result with the committed file byte for byte, so a constant
or a width function changed in the Python without the proof fails there; and
the proof does not compile against a changed file whose values break a bound.

Run it with `Y_EXACT_ACT_LEVELS`, `Y_EXACT_K_LEVELS` and `Y_EXACT_V_LEVELS`
unset: the file records the defaults.
"""
import inspect
import os
import pathlib
import sys

OUT = pathlib.Path(__file__).resolve().parents[1] / "proofs" / "ExactDecodeConstants.v"
ENV = ("Y_EXACT_ACT_LEVELS", "Y_EXACT_K_LEVELS", "Y_EXACT_V_LEVELS")

# Contraction lengths: the linear layers of Qwen2.5 0.5B, 1.5B, 3B and 7B,
# small and ragged ones, and the values either side of each ceiling.
ACT_K = (1, 2, 64, 100, 896, 1536, 2048, 3584, 4864, 8960, 11008, 18944, 28672,
         32768, 65536, 66051, 66052, 131072, 133152, 133153, 1 << 20, 1 << 24, 1 << 31)
ACT_REQUESTED = (127, 255, 511, 1023, 2047, 4095, 16383, 32767)
K_HEAD_DIMS = (1, 16, 32, 64, 96, 128, 129, 192, 256, 258, 259, 512, 1024, 1040, 1041, 4096)
K_Q_LEVELS = (127, 255)
PV_T = (1, 2, 64, 256, 517, 518, 519, 1024, 2048, 4096, 8192, 8738, 8739, 32768,
        65536, 132104, 132105, 1 << 20)
PV_V_LEVELS = (127, 1023)


def exp_arguments():
    """A spread of exp arguments: both ends of every table step near the
    bottom, each power of two and its neighbours, and a fixed stride."""
    args = set()
    for i in range(0, 66):
        for d in (0, 1, 511, 1023):
            args.add(i * 1024 + d)
    for n in range(0, 31):
        for d in (-1, 0, 1, 40000):
            args.add(max(0, (n << 16) + d))
    args.update(range(0, 31 << 16, 104729))
    args.update((30 << 16, (31 << 16) - 1, 31 << 16, 1 << 30))
    return sorted(args)


def zlist(values, per_line=8):
    values = [str(v) for v in values]
    rows = ["; ".join(values[i:i + per_line]) for i in range(0, len(values), per_line)]
    return "[ " + ";\n    ".join(rows) + " ]"


def triples(rows):
    return "[ " + ";\n    ".join(f"({a}, {b}, {c})" for a, b, c in rows) + " ]"


def pairs(rows, per_line=4):
    cells = [f"({a}, {b})" for a, b in rows]
    lines = ["; ".join(cells[i:i + per_line]) for i in range(0, len(cells), per_line)]
    return "[ " + ";\n    ".join(lines) + " ]"


def generate():
    set_ = [name for name in ENV if name in os.environ]
    if set_:
        raise SystemExit(f"unset {', '.join(set_)}: the file records the defaults")
    import torch
    from y_lang.exact import attention, kv, model

    w_levels = inspect.signature(model.act_levels_for).parameters["w_levels"].default
    table = [int(v) for v in attention._table("cpu").tolist()]
    exp_args = exp_arguments()
    exp_vals = attention.exp2_neg_q16_16(
        torch.tensor(exp_args, dtype=torch.int64), attention._table("cpu")).tolist()
    act = [(k, r, model.act_levels_for(k, w_levels, r)) for k in ACT_K for r in ACT_REQUESTED]
    klv = [(d, q, kv.k_levels_for(d, q)) for d in K_HEAD_DIMS for q in K_Q_LEVELS]
    dig = [(t, v, attention.digit_width(t, v)) for t in PV_T for v in PV_V_LEVELS]
    qs = sorted({q for a in (127, 511, model.DIGIT_MAX) for q in (-a, -a + 1, -129, -128,
                 -127, -1, 0, 1, 127, 128, a - 1, a)})
    split = []
    for q in qs:
        hi, lo = model.split_digits(torch.tensor([q], dtype=torch.int32), model.DIGIT_MAX)
        split.append((q, int(hi[0]), int(lo[0])))

    return f"""(** * The exact decode path's constants.

    GENERATED from `python/y_lang/exact/` by `tools/exact_decode_constants.py`.
    Do not edit: regenerate with

      << PYTHONPATH=python python3 tools/exact_decode_constants.py --write >>

    `python/tests/test_exact_decode_constants.py` fails on any byte of
    difference between this file and what the Python code generates now.

    The constants are the ones the Python reads at each bound. The sample
    lists record what the Python width functions and the integer `exp2`
    return; [ExactDecodeWidths.v] proves its own definitions of those
    functions return the same values, so a transcription that drifts from the
    code does not compile. *)

From Stdlib Require Import ZArith List.
Import ListNotations.
Open Scope Z_scope.

(** `model.ACC_BUDGET`: the int32 accumulator of an int8 dot product. *)
Definition ACC_BUDGET : Z := {model.ACC_BUDGET}.
(** `model.DIGIT_MAX`: the widest activation two int8 digits hold. *)
Definition DIGIT_MAX : Z := {model.DIGIT_MAX}.
(** The weight width: `act_levels_for`'s default `w_levels`. *)
Definition W_LEVELS : Z := {w_levels}.
(** `model.ACT_LEVELS`: the default requested activation width. *)
Definition ACT_LEVELS_DEFAULT : Z := {model.ACT_LEVELS}.
(** `attention.Q_LEVELS`, `kv.V_LEVELS`, and `kv.K_LEVELS` (a requested maximum). *)
Definition Q_LEVELS : Z := {attention.Q_LEVELS}.
Definition V_LEVELS : Z := {kv.V_LEVELS}.
Definition K_LEVELS : Z := {kv.K_LEVELS}.
(** `kv.SCORE_BUDGET`: float32 holds every integer up to this. *)
Definition SCORE_BUDGET : Z := {kv.SCORE_BUDGET}.
(** `attention.P_BITS`: the softmax weights are Q0.P_BITS. *)
Definition P_BITS : Z := {attention.P_BITS}.
(** `model.SUMSQ_F`, `SUMSQ_W`, `SUMSQ_MAX_H`: the norm's grid, limb and row limit. *)
Definition SUMSQ_F : Z := {model.SUMSQ_F}.
Definition SUMSQ_W : Z := {model.SUMSQ_W}.
Definition SUMSQ_MAX_H : Z := {model.SUMSQ_MAX_H}.
(** `attention.LN2_Q30`, `RECIP6_Q32` and `_table`: the integer `exp2`. *)
Definition LN2_Q30 : Z := {attention.LN2_Q30}.
Definition RECIP6_Q32 : Z := {attention.RECIP6_Q32}.
Definition EXP_TABLE : list Z :=
  {zlist(table, 4)}.
(** `attention.EXP2_DOMAIN_DIGEST`: FNV-1a of `exp2` over `0 .. 31 << 16`. *)
Definition EXP2_DOMAIN_DIGEST : Z := {attention.EXP2_DOMAIN_DIGEST}.

(** `(k, requested, model.act_levels_for(k, W_LEVELS, requested))`. *)
Definition ACT_LEVELS_SAMPLES : list (Z * Z * Z) :=
  {triples(act)}.

(** `(head_dim, q_levels, kv.k_levels_for(head_dim, q_levels))`. *)
Definition K_LEVELS_SAMPLES : list (Z * Z * Z) :=
  {triples(klv)}.

(** `(T, v_levels, attention.digit_width(T, v_levels))`. *)
Definition DIGIT_WIDTH_SAMPLES : list (Z * Z * Z) :=
  {triples(dig)}.

(** `(q, hi, lo)` from `model.split_digits`. *)
Definition SPLIT_SAMPLES : list (Z * Z * Z) :=
  {triples(split)}.

(** `(t, attention.exp2_neg_q16_16(t))`. *)
Definition EXP2_SAMPLES : list (Z * Z) :=
  {pairs(list(zip(exp_args, exp_vals)))}.

Theorem the_exp_table_has_64_entries_from_one : length EXP_TABLE = 64%nat
  /\\ nth 0 EXP_TABLE 0 = 2 ^ 30.
Proof. split; reflexivity. Qed.

Theorem the_samples_are_not_empty :
  length ACT_LEVELS_SAMPLES = {len(act)}%nat /\\ length K_LEVELS_SAMPLES = {len(klv)}%nat
  /\\ length DIGIT_WIDTH_SAMPLES = {len(dig)}%nat /\\ length SPLIT_SAMPLES = {len(split)}%nat
  /\\ length EXP2_SAMPLES = {len(exp_args)}%nat.
Proof. repeat split; reflexivity. Qed.

Print Assumptions the_exp_table_has_64_entries_from_one.
Print Assumptions the_samples_are_not_empty.
"""


def main():
    text = generate()
    if "--write" in sys.argv[1:]:
        OUT.write_text(text)
        print(f"wrote {OUT}")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
