"""Moved to `python/y_lang/exact/w8a8_gemv.py` (`y_lang.exact.w8a8_gemv`), so that an
installed y-lang carries it. Importing `w8a8_gemv` from here gives that module
itself, not a copy (`_exact_pkg`)."""
import _exact_pkg

_exact_pkg.alias(__name__, "w8a8_gemv")
