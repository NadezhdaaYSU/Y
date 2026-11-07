"""Moved to `python/y_lang/exact/attention.py` (`y_lang.exact.attention`), so
that an installed y-lang carries the exact attention. Importing
`batch_invariance_demo` from here gives that module itself, not a copy
(`_exact_pkg`); running this file runs its demo.

    python3 tools/batch_invariance_demo.py
"""
import sys

import _exact_pkg

if __name__ == "__main__":
    sys.exit(_exact_pkg.alias("batch_invariance_demo", "attention").main())
_exact_pkg.alias(__name__, "attention")
