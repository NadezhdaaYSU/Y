"""Build the `y-lang` wheel with the compiler library inside it.

Installed, the package loads `liby.so` from beside itself (`_find_liby` in
`y_lang/compiler.py`). A wheel built without it installed cleanly and failed
at the first compile, so this build refuses instead:
  * no library: build it with `cargo build --release` in the cargo project
    that holds this directory, or name one in `Y_WHEEL_LIBY`;
  * a library older than any compiler source beside it (`src/`, `crates/`,
    `Cargo.toml`, `Cargo.lock`): it would answer with an older compiler, and
    nothing at run time would say so.

The wheel is tagged `py3-none-<platform>`. The library is native code, so
the wheel is not pure, but Python loads it through ctypes, so it does not
depend on the Python version.
"""

import os
import shutil
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:  # setuptools < 70.1
    from wheel.bdist_wheel import bdist_wheel

HERE = Path(__file__).resolve().parent
CARGO_ROOT = HERE.parent
SOURCE_DIRS = ("src", "crates")
SOURCE_FILES = ("Cargo.toml", "Cargo.lock")


def _sources(root):
    for name in SOURCE_FILES:
        path = root / name
        if path.is_file():
            yield path
    for name in SOURCE_DIRS:
        for top, dirs, files in os.walk(root / name):
            dirs[:] = [d for d in dirs if d not in ("target", "__pycache__")]
            for f in files:
                yield Path(top) / f


def compiler_library(root=CARGO_ROOT, named=None):
    """The `liby.so` to put in the wheel; raises SystemExit naming what is wrong."""
    named = os.environ.get("Y_WHEEL_LIBY") if named is None else named
    lib = Path(named) if named else root / "target" / "release" / "liby.so"
    if not lib.is_file():
        raise SystemExit(
            f"y-lang wheel: no compiler library at {lib}. Run `cargo build "
            f"--release` in {root}, or name the library in Y_WHEEL_LIBY.")
    if (root / "Cargo.toml").is_file():
        built = lib.stat().st_mtime_ns
        newer = sorted(str(p.relative_to(root)) for p in _sources(root)
                       if p.stat().st_mtime_ns > built)
        if newer:
            shown = ", ".join(newer[:5]) + (f" and {len(newer) - 5} more" if len(newer) > 5 else "")
            raise SystemExit(
                f"y-lang wheel: {lib} is older than {shown}. Rebuild it with "
                f"`cargo build --release` in {root}.")
    return lib


class BuildPy(build_py):
    def run(self):
        super().run()
        lib = compiler_library()
        dest = Path(self.build_lib) / "y_lang" / "liby.so"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(lib, dest)
        os.chmod(dest, 0o755)


class NativeDistribution(Distribution):
    """Builds into the platform's library directory, so the wheel is not pure
    and its files sit at the wheel's root."""

    def has_ext_modules(self):
        return True


class BdistWheel(bdist_wheel):
    def get_tag(self):
        _, _, platform = super().get_tag()
        return "py3", "none", platform


setup(distclass=NativeDistribution,
      cmdclass={"build_py": BuildPy, "bdist_wheel": BdistWheel})
