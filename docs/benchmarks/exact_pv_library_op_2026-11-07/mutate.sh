#!/usr/bin/env bash
# Each mutant: copy the file, apply one replacement, run the self-test in int8-V mode, restore.
set -u
S="$(realpath "$1")"   # usage: mutate.sh SCRATCH_DIR (needs SCRATCH_DIR/pvmut/); backups and logs go there
cd "$(dirname "$0")/../../.."
mut() {  # name file old new
  local name="$1" f="$2" old="$3" new="$4"
  cp "$f" "$S/pvmut/backup"
  python3 - "$f" "$old" "$new" <<'PY'
import sys
f, old, new = sys.argv[1:]
s = open(f).read()
assert s.count(old) == 1, (f, old, s.count(old))
open(f, "w").write(s.replace(old, new))
PY
  Y_EXACT_V_INT8=1 timeout 900 python3 tools/exact_selftest.py > "$S/pvmut/$name.log" 2>&1
  local rc=$?
  cp "$S/pvmut/backup" "$f"
  local line; line=$(grep -E "check_pv_int8_kernel|exact_pv's kernel" -A1 "$S/pvmut/$name.log" | head -2 | tr -s ' ' | tr '\n' ' ')
  echo "$name exit=$rc :: $line"
}
mut default_stream tools/batch_invariance_demo.py "        kern.launch((Qn, B, 1), (Dh, 1, 1), args)
    else:" "        kern.launch((Qn, B, 1), (Dh, 1, 1), args, stream=torch.cuda.default_stream(dev))
    else:"
mut no_op_under_compile tools/batch_invariance_demo.py "            acc = _PV_OP(p32, v)" "            acc = _y_exact_pv(p32, v)"
mut fake_float64 tools/batch_invariance_demo.py "    return p.new_empty((*p.shape[:-1], v.shape[-1]), dtype=torch.int64)" "    return p.new_empty((*p.shape[:-1], v.shape[-1]), dtype=torch.float64)"
mut short_key tools/batch_invariance_demo.py "    args = [p32, v8, out, T, Dh, Qn," "    args = [p32, v8, out, max(T - 1, 1), Dh, Qn,"
mut no_cpu_refusal tools/batch_invariance_demo.py '    if dev.type != "cuda" or v.device != dev:' '    if v.device != dev:'
git status --short tools/
