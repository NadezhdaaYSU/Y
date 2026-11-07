"""Build the `y-lang` wheel with the compiler library and GPU probe inside it.

Installed, the package loads `liby.so` from beside itself (`_find_liby` in
`y_lang/compiler.py`). A wheel built without it installed cleanly and failed
at the first compile. The compiler runs `ysu_gpu_probe` from beside `liby.so`
when there is no `.ysu_hw_profile` (`sentinel::check_or_probe_hardware`);
without it the profile names an unknown GPU and kernels target `sm_80`
whatever the card. So this build refuses:
  * a missing library or probe: build both with `cargo build --release` in
    the cargo project that holds this directory, or name them in
    `Y_WHEEL_LIBY` and `Y_WHEEL_PROBE`;
  * one older than any compiler source beside it (`src/`, `crates/`,
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


# (file in target/release and in the package, variable naming another, what it is)
ARTIFACTS = (
    ("liby.so", "Y_WHEEL_LIBY", "compiler library"),
    ("ysu_gpu_probe", "Y_WHEEL_PROBE", "GPU probe"),
)


def built_artifact(name, variable, what, root=CARGO_ROOT):
    """The file to put in the wheel; raises SystemExit naming what is wrong."""
    named = os.environ.get(variable)
    path = Path(named) if named else root / "target" / "release" / name
    if not path.is_file():
        raise SystemExit(
            f"y-lang wheel: no {what} at {path}. Run `cargo build --release` "
            f"in {root}, or name the file in {variable}.")
    if (root / "Cargo.toml").is_file():
        built = path.stat().st_mtime_ns
        newer = sorted(str(p.relative_to(root)) for p in _sources(root)
                       if p.stat().st_mtime_ns > built)
        if newer:
            shown = ", ".join(newer[:5]) + (f" and {len(newer) - 5} more" if len(newer) > 5 else "")
            raise SystemExit(
                f"y-lang wheel: {path} is older than {shown}. Rebuild it with "
                f"`cargo build --release` in {root}.")
    return path


class BuildPy(build_py):
    def run(self):
        super().run()
        for name, variable, what in ARTIFACTS:
            source = built_artifact(name, variable, what)
            dest = Path(self.build_lib) / "y_lang" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
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
