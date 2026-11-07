"""The exact-inference modules live in the y-lang package (`y_lang.exact`), so
that an installed y-lang carries them. The harnesses here still import them by
their old names (`exact_model`, `exact_kv`, `w8a8_gemv`,
`batch_invariance_demo`); each of those files makes its name refer to the
package module itself, not a copy, so a harness that sets
`exact_model.GEMM = "y"` or replaces `batch_invariance_demo._y_exact_pv`
changes what the package runs.
"""

import importlib
import os
import sys

PYTHON = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "python"))


def add_path():
    """Put this checkout's `python/` first, so `y_lang` is the one built from
    these sources (and finds this checkout's `liby.so`)."""
    if PYTHON not in sys.path:
        sys.path.insert(0, PYTHON)


def alias(name, submodule):
    """Make `sys.modules[name]` the package's `y_lang.exact.<submodule>`."""
    add_path()
    module = importlib.import_module(f"y_lang.exact.{submodule}")
    sys.modules[name] = module
    return module
