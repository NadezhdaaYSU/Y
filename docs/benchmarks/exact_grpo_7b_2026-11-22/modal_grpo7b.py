"""Exact GRPO of Qwen2.5-7B-Instruct on an H100 and an A100 80GB, at once:
the same 20 steps on each (`tools/exact_lora_grpo.py`, three-digit
multiplication answered with the number, dynamic sampling), whose records the
development machine compares; the A100 also runs the RL tests.

    git archive --format=tar.gz -o /abs/y_src.tar.gz HEAD     # in the repository
    Y_SRC_TAR=/abs/y_src.tar.gz modal run modal_grpo7b.py --out DIR

The model is pinned to a revision (the image build fails if the hub's main
has moved). Y is built from the tarball in the image, so the GPUs are rented
only for the runs. Each function has a hard timeout and each command a
share of a budget below it; a command still running when the budget is
spent is killed and its output so far kept.
"""
import os
import pathlib
import subprocess
import sys
import time

import modal

TORCH = "torch==2.13.0"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
REPO = "/root/Y"
PATH = "/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ca-certificates", "build-essential")
    .pip_install(TORCH, index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("numpy", "transformers==5.15.0", "tokenizers==0.22.2", "safetensors", "mpmath",
                 "z3-solver", "huggingface_hub")
    .env({"HF_HOME": "/root/hf", "PATH": PATH})
    .run_commands(
        "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal -q --default-toolchain 1.97.1",
        # The hub's main, checked to be the pinned revision: a download by
        # revision alone writes no `main` ref, and the tools load `main`
        # offline (the first run of this job failed so, in 6 s).
        f"python -c \"from huggingface_hub import snapshot_download as d; p = d('{MODEL}'); "
        f"assert p.endswith('{REVISION}'), p\"",
    )
    .add_local_file(os.environ.get("Y_SRC_TAR", "y_src.tar.gz"), "/root/y_src.tar.gz", copy=True)
    .run_commands(
        f"mkdir -p {REPO} && tar -xzf /root/y_src.tar.gz -C {REPO}",
        f"cd {REPO} && CARGO_TERM_COLOR=never cargo build --release --lib --bin ysu_gpu_probe 2>&1 | tail -3",
        f"sha256sum {REPO}/target/release/liby.so",
    )
    .env({"HF_HUB_OFFLINE": "1", "PYTHONPATH": f"{REPO}/python",
          "Y_LIB_PATH": f"{REPO}/target/release/liby.so", "YSU_HW_PROFILE": "/root/y_hw_profile",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

app = modal.App("y-exact-grpo-7b", image=image)

GRPO = (f"exact_lora_grpo.py --device cuda --model {MODEL} --digits 3 --new 8 --eval-new 8 "
        "--sample-prompts 32 --chunk-tokens 4096 --steps 20 --eval-every 20 --checkpoint 5 --log 1 ")
TESTS = ("cd {repo} && exec env PYTHONPATH=python:python/tests " + sys.executable +
         " -m unittest -v python.tests.test_exact_rl")


def _sh(cmd):
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout


def _runs(name, runs, minutes):
    end = time.time() + minutes * 60
    out = pathlib.Path("/root/out")
    out.mkdir(exist_ok=True)
    files = {}
    head = _sh("nvidia-smi --query-gpu=name,compute_cap,driver_version,clocks.max.sm,memory.total --format=csv; "
               "lscpu | grep -E 'Model name|^CPU\\(s\\)'; "
               "python -c 'import torch, triton; print(torch.__version__, triton.__version__)'; "
               f"sha256sum {REPO}/target/release/liby.so; "
               "sha256sum /root/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/*.safetensors")
    files[f"{name}_machine.txt"] = head
    print(head, flush=True)
    for run, cmd in runs:
        t0 = time.time()
        left = end - time.time()
        if left < 60:
            files[f"{run}.log"] = f"not run: {left:.0f} s of the budget left\n"
            continue
        line = (cmd[4:].format(repo=REPO) if cmd.startswith("RAW:")
                else f"cd {REPO}/tools && exec {sys.executable} {cmd.format(out=out)}")
        try:
            p = subprocess.run(line, shell=True, timeout=left, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log = p.stdout + f"\nexit {p.returncode} {time.time() - t0:.0f} s\n"
        except subprocess.TimeoutExpired as e:
            got = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            log = got + f"\nKILLED at the budget after {time.time() - t0:.0f} s\n"
        files[f"{run}.log"] = log
        print(log[-2500:], flush=True)
    for f in out.iterdir():
        files[f.name] = f.read_text()
    return files


@app.function(gpu="H100!", timeout=30 * 60, cpu=8.0, memory=65536)
def on_h100():
    return _runs("h100", [("h100_grpo20", GRPO + "--record {out}/h100_grpo20.json")], 27)


@app.function(gpu="A100-80GB", timeout=30 * 60, cpu=8.0, memory=65536)
def on_a100():
    return _runs("a100", [("a100_grpo20", GRPO + "--record {out}/a100_grpo20.json"),
                          ("a100_rl_tests", "RAW:" + TESTS)], 27)


@app.local_entrypoint()
def main(out: str = "results"):
    d = pathlib.Path(out)
    d.mkdir(parents=True, exist_ok=True)
    calls = [on_h100.spawn(), on_a100.spawn()]
    for call in calls:
        for name, data in call.get().items():
            (d / name).write_text(data)
    print("wrote", d)
