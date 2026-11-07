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
  * one older than a source it was built from: it would answer with an older
    compiler, and nothing at run time would say so. The sources are the ones
    cargo's dep-info beside the file names (`liby.d`, `ysu_gpu_probe.d`).
    Without it, or if it names a file that does not exist here, they are
    every file under `src/` and `crates/` and `Cargo.toml` and `Cargo.lock`:
    stricter, and an edit to a file only the `Y` executable uses then
    refuses a library cargo rightly did not rebuild.

Both binaries contain the Rust standard library, whose notice depends on the
toolchain, so the build copies the toolchain's `COPYRIGHT-library.html` in as
`RUST_STD_COPYRIGHT.html` (or `Y_WHEEL_RUST_NOTICE`), and refuses without
it. The crates' notices are `THIRD_PARTY_NOTICES`, kept current by
`tools/third_party_notices.py`.

The wheel is tagged `py3-none-<platform>`. The library is native code, so
the wheel is not pure, but Python loads it through ctypes, so it does not
depend on the Python version.
"""

import os
import re
import shutil
import subprocess
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
# `sentinel::USER_CACHE_MARKER` and `y_lang.compiler.USER_CACHE_MARKER`.
USER_CACHE_MARKER = "PROFILE_IN_USER_CACHE"
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


def _shown(path, root):
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _dep_info_sources(path):
    """The source files cargo's dep-info (`<artifact>.d` beside it) says
    `path` was built from, or None where there is no usable dep-info."""
    dep_info = path.with_suffix(".d")
    try:
        text = dep_info.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        head, sep, deps = line.partition(": ")
        if not sep or Path(head.replace("\\ ", " ")).name != path.name:
            continue
        sources = [Path(d.replace("\\ ", " ")) for d in re.split(r"(?<!\\) ", deps.strip()) if d]
        if sources and all(src.is_file() for src in sources):
            return sources
        return None
    return None


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
        sources = _dep_info_sources(path) or list(_sources(root))
        newer = sorted(_shown(p, root) for p in sources if p.stat().st_mtime_ns > built)
        if newer:
            shown = ", ".join(newer[:5]) + (f" and {len(newer) - 5} more" if len(newer) > 5 else "")
            raise SystemExit(
                f"y-lang wheel: {path} is older than {shown}. Rebuild it with "
                f"`cargo build --release` in {root}.")
    return path


def rust_std_notice():
    """The toolchain's notice for the Rust standard library."""
    named = os.environ.get("Y_WHEEL_RUST_NOTICE")
    if named:
        path = Path(named)
    else:
        try:
            sysroot = subprocess.run(["rustc", "--print", "sysroot"], check=True,
                                     capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as e:
            raise SystemExit(f"y-lang wheel: `rustc --print sysroot` failed ({e}); "
                             "name the standard library's notice in Y_WHEEL_RUST_NOTICE.")
        path = Path(sysroot) / "share" / "doc" / "rust" / "COPYRIGHT-library.html"
    if not path.is_file():
        raise SystemExit(f"y-lang wheel: no Rust standard library notice at {path}; "
                         "name it in Y_WHEEL_RUST_NOTICE.")
    return path


class BuildPy(build_py):
    def run(self):
        super().run()
        package = Path(self.build_lib) / "y_lang"
        package.mkdir(parents=True, exist_ok=True)
        for name, variable, what in ARTIFACTS:
            dest = package / name
            shutil.copyfile(built_artifact(name, variable, what), dest)
            os.chmod(dest, 0o755)
        shutil.copyfile(rust_std_notice(), package / "RUST_STD_COPYRIGHT.html")
        # Beside liby.so, this sends the hardware profile and the PTX cache
        # to the user cache directory instead of the working directory
        # (`sentinel::hw_profile_path`, `compiler.get_cache_dir`).
        (package / USER_CACHE_MARKER).write_text(
            "This copy of the Y compiler keeps its hardware profile and caches in\n"
            "$YSU_CACHE_DIR, else $XDG_CACHE_HOME/y-lang, else ~/.cache/y-lang.\n"
            "A .ysu_hw_profile in the working directory is still used if present,\n"
            "and YSU_HW_PROFILE names a profile file outright.\n", encoding="utf-8")


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
