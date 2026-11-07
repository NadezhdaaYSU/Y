"""Moved to `python/y_lang/exact/model.py` (`y_lang.exact.model`), so that an
installed y-lang carries it. Importing `exact_model` from here gives that module
itself, not a copy (`_exact_pkg`)."""
import _exact_pkg

_exact_pkg.alias(__name__, "model")
