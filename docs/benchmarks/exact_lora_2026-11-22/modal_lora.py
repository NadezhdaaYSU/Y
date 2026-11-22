"""Exact LoRA fine-tuning (T3) on an H100: the GPU tests, a fine-tune of
Qwen2.5-0.5B-Instruct on two-digit multiplication (`tools/exact_lora_sft.py`)
whose records the development machine compares with its CPU's, and the
CPU's adapters served on the H100.

    git archive --format=tar.gz -o /abs/y_src.tar.gz HEAD     # in the repository
    Y_SRC_TAR=/abs/y_src.tar.gz Y_CPU_ADAPTERS=/abs/cpu_sft20_adapters.pt \\
        modal run modal_lora.py --out DIR

The model is the revision the development machine has (pinned; the image
build fails if the hub's main has moved). Y is built from the tarball in
the image, so the GPU is rented only for the runs. The function has a hard
timeout and each command a share of a budget below it; a command still
running when the budget is spent is killed and its output so far kept.
"""
import os
import pathlib
import subprocess
import sys
import time

import modal

TORCH = "torch==2.13.0"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
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
        f"python -c \"from huggingface_hub import snapshot_download as d; p = d('{MODEL}'); "
        f"assert p.endswith('{REVISION}'), p\"",
    )
    .add_local_file(os.environ.get("Y_SRC_TAR", "y_src.tar.gz"), "/root/y_src.tar.gz", copy=True)
    .add_local_file(os.environ.get("Y_CPU_ADAPTERS", "cpu_sft20_adapters.pt"), "/root/cpu_adapters.pt",
                    copy=True)
    .run_commands(
        f"mkdir -p {REPO} && tar -xzf /root/y_src.tar.gz -C {REPO}",
        f"cd {REPO} && CARGO_TERM_COLOR=never cargo build --release --lib --bin ysu_gpu_probe 2>&1 | tail -3",
        f"sha256sum {REPO}/target/release/liby.so",
    )
    .env({"HF_HUB_OFFLINE": "1", "PYTHONPATH": f"{REPO}/python",
          "Y_LIB_PATH": f"{REPO}/target/release/liby.so", "YSU_HW_PROFILE": "/root/y_hw_profile"})
)

app = modal.App("y-exact-lora", image=image)

TESTS = ("cd {repo} && exec env PYTHONPATH=python:python/tests " + sys.executable +
         " -m unittest -v python.tests.test_exact_lora")
SFT = "exact_lora_sft.py --device cuda --batch 64 --log 10 "
RUNS = [
    ("gpu_tests", "RAW:" + TESTS),
    ("h100_sft400", SFT + "--steps 400 --eval-every 20 --record {out}/h100_sft400.json "
                    "--save {out}/h100_sft400_adapters.pt"),
    ("h100_serve_cpu20", SFT + "--load /root/cpu_adapters.pt --eval-only "
                         "--record {out}/h100_serve_cpu20.json"),
]


def _sh(cmd):
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout


@app.function(gpu="H100!", timeout=30 * 60, cpu=8.0, memory=32768)
def on_h100():
    end = time.time() + 27 * 60
    out = pathlib.Path("/root/out")
    out.mkdir(exist_ok=True)
    files = {}
    head = _sh("nvidia-smi --query-gpu=name,compute_cap,driver_version,clocks.max.sm,memory.total --format=csv; "
               "lscpu | grep -E 'Model name|^CPU\\(s\\)'; "
               "python -c 'import torch, triton; print(torch.__version__, triton.__version__)'; "
               f"sha256sum {REPO}/target/release/liby.so /root/cpu_adapters.pt; "
               "sha256sum /root/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*/model.safetensors")
    files["machine.txt"] = head
    print(head, flush=True)
    for name, cmd in RUNS:
        t0 = time.time()
        left = end - time.time()
        if left < 60:
            files[f"{name}.log"] = f"not run: {left:.0f} s of the budget left\n"
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
        files[f"{name}.log"] = log
        print(log[-2500:], flush=True)
    for f in out.iterdir():
        files[f.name] = f.read_bytes() if f.suffix == ".pt" else f.read_text()
    return files


@app.local_entrypoint()
def main(out: str = "results"):
    d = pathlib.Path(out)
    d.mkdir(parents=True, exist_ok=True)
    for name, data in on_h100.remote().items():
        p = d / name
        (p.write_bytes if isinstance(data, bytes) else p.write_text)(data)
    print("wrote", d)
