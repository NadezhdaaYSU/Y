"""Exact training (T2) on an H100: the fast path's 5000-step record, compared
on the development machine against the CPU record of
docs/benchmarks/exact_train_lm_2026-11-21/, and step times against float32.

    git archive --format=tar.gz -o /abs/y_src.tar.gz HEAD     # in the repository
    Y_SRC_TAR=/abs/y_src.tar.gz modal run modal_train.py --out DIR

Y is built from the tarball in the image (Rust 1.97.1, on Modal's CPU
builder), so the GPU is rented only for the runs. The function has a hard
timeout, and each command a share of a budget below it: a command still
running when the budget is spent is killed and its output so far kept.
"""
import os
import pathlib
import subprocess
import sys
import time

import modal

TORCH = "torch==2.13.0"
REPO = "/root/Y"
PATH = "/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
TEXT = os.path.expanduser("~/.cache/y-lang/data/tinyshakespeare.txt")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ca-certificates", "build-essential")
    .pip_install(TORCH, index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("numpy", "transformers==5.15.0", "tokenizers==0.22.2", "safetensors", "z3-solver", "mpmath")
    .env({"PATH": PATH})
    .run_commands(
        "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal -q --default-toolchain 1.97.1",
    )
    .add_local_file(TEXT, "/root/.cache/y-lang/data/tinyshakespeare.txt", copy=True)
    .add_local_file(os.environ.get("Y_SRC_TAR", "y_src.tar.gz"), "/root/y_src.tar.gz", copy=True)
    .run_commands(
        f"mkdir -p {REPO} && tar -xzf /root/y_src.tar.gz -C {REPO}",
        f"cd {REPO} && CARGO_TERM_COLOR=never cargo build --release --lib --bin ysu_gpu_probe 2>&1 | tail -3",
        f"sha256sum {REPO}/target/release/liby.so",
    )
    .env({"PYTHONPATH": f"{REPO}/python", "Y_LIB_PATH": f"{REPO}/target/release/liby.so",
          "YSU_HW_PROFILE": "/root/y_hw_profile"})
)

app = modal.App("y-exact-train", image=image)

REC = "exact_train_lm.py --device cuda --record {out}/"
RUNS = [
    ("exact_y_g12b8_fast", REC + "h100_exact_y_g12b8_fast.json --mm y --graph --compile"),
    ("float32_fast", REC + "h100_float32_fast.json --arm float32 --graph --compile"),
    ("float32_eager", REC + "h100_float32_eager.json --arm float32"),
    ("speed_medium", "exact_train_speed.py --size medium --check 3 --steady 50 "
                     "--out {out}/h100_speed_medium.json"),
    ("speed_tiny", "exact_train_speed.py --size tiny --check 30 --steady 500 "
                   "--out {out}/h100_speed_tiny.json"),
]


def _sh(cmd):
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout


@app.function(gpu="H100!", timeout=35 * 60, cpu=16.0, memory=32768)
def on_h100():
    end = time.time() + 32 * 60
    out = pathlib.Path("/root/out")
    out.mkdir(exist_ok=True)
    files = {}
    head = _sh("nvidia-smi --query-gpu=name,compute_cap,driver_version,clocks.max.sm,memory.total --format=csv; "
               "lscpu | grep -E 'Model name|^CPU\\(s\\)'; "
               "python -c 'import torch, triton; print(torch.__version__, triton.__version__)'; "
               f"sha256sum {REPO}/target/release/liby.so")
    files["machine.txt"] = head
    print(head, flush=True)
    for name, cmd in RUNS:
        t0 = time.time()
        left = end - time.time()
        if left < 60:
            files[f"{name}.log"] = f"not run: {left:.0f} s of the budget left\n"
            continue
        try:
            # `exec`, so a kill at the budget reaches python, not just the shell.
            p = subprocess.run(f"cd {REPO}/tools && exec {sys.executable} {cmd.format(out=out)}",
                               shell=True, timeout=left, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log = p.stdout + f"\nexit {p.returncode} {time.time() - t0:.0f} s\n"
        except subprocess.TimeoutExpired as e:
            got = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            log = got + f"\nKILLED at the budget after {time.time() - t0:.0f} s\n"
        files[f"{name}.log"] = log
        print(log[-2500:], flush=True)
        for js in out.glob("*.json"):
            files[js.name] = js.read_text()
    return files


@app.local_entrypoint()
def main(out: str = "results"):
    d = pathlib.Path(out)
    d.mkdir(parents=True, exist_ok=True)
    for name, text in on_h100.remote().items():
        (d / name).write_text(text)
    print("wrote", d)
