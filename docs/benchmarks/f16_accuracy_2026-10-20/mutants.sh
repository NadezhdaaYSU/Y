#!/bin/bash
# mutants.sh <bench_ab.py dir>: check_metrics.py against four mutants of
# accuracy.py's metric code (each must fail it) and one control it cannot see.
# Mutants are written into a temporary directory; accuracy.py is not touched.
set -u
here=$(cd "$(dirname "$0")" && pwd); M=$(mktemp -d); cp "$here/check_metrics.py" "$M/"
mut() {
  python3 - "$here/accuracy.py" "$M/accuracy.py" "$1" "$2" <<'PY'
import sys
s = open(sys.argv[1]).read(); old, new = sys.argv[3], sys.argv[4]
assert s.count(old) == 1, old
open(sys.argv[2], "w").write(s.replace(old, new))
PY
  if (cd "$M" && CUDA_VISIBLE_DEVICES="" python3 check_metrics.py "$BENCH" > "$M/out.txt" 2>&1); then echo "SURVIVED: $3"
  else echo "killed: $3 ($(grep -v USDT "$M/out.txt" | tail -1 | cut -c1-100))"; fi
}
BENCH=$1
mut "en = e[pos] / s[pos]" "en = e / s" "no zero-scale mask"
mut "(c != 0)).sum()" "(c == 0)).sum()" "zero-scale count inverted"
mut '"better": (ec < eb)' '"better": (ec <= eb)' "ties counted as better"
mut "((ax > 0) & (ax < 2.0 ** -14))" "(ax < 2.0 ** -14)" "zeros counted as subnormal"
mut "return a * 2.0 ** -12, b * 2.0 ** -12" "return a * 2.0 ** -12, b" "control, not checked: small's scale (acc_1.json records each input's max_abs)"
rm -rf "$M"
