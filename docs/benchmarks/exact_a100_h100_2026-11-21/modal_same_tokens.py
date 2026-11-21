"""Same tokens on an A100 and an H100: tools/exact_cross_device.py records,
made on Modal (modal.com) and compared on the development machine against the
CPU, T4 and RTX 4070 Ti records.

    tar -czf y_src.tar.gz -T src_files.txt        # in the repository
    Y_SRC_TAR=/abs/y_src.tar.gz modal run modal_same_tokens.py --stage eager --out DIR

Stage `eager`: 0.5B eager, both arms (stock is the control: it must change
across cards), then 0.5B eager with every switch on Y's int8 GEMM. Stages
`compiled_0p5b` and `compiled_1p5b` (run with `--gpus h100`): compiled, every
switch on Y's GEMM.

Y is built from the tarball in the image (Rust 1.97.1, on Modal's CPU
builder), and both models are downloaded there too, so the GPU is rented only
for the records. Every function has a hard timeout.
"""
import os
import pathlib
import subprocess
import sys
import time

import modal

TORCH = "torch==2.13.0"
HUB = ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct")
REPO = "/root/Y"
PATH = "/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ca-certificates", "build-essential")
    .pip_install(TORCH, index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("transformers==5.15.0", "tokenizers==0.22.2", "safetensors", "mpmath",
                 "z3-solver", "huggingface_hub")
    .env({"HF_HOME": "/root/hf", "PATH": PATH})
    .run_commands(
        "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal -q --default-toolchain 1.97.1",
        "python -c \"import sysconfig, os; p = sysconfig.get_paths()['include']; "
        "assert os.path.exists(os.path.join(p, 'Python.h')), p\"",
        "python -c \"from huggingface_hub import snapshot_download as d; "
        + "; ".join(f"d('{m}')" for m in HUB) + "\"",
    )
    .add_local_file(os.environ.get("Y_SRC_TAR", "y_src.tar.gz"), "/root/y_src.tar.gz", copy=True)
    .run_commands(
        f"mkdir -p {REPO} && tar -xzf /root/y_src.tar.gz -C {REPO}",
        f"cd {REPO} && CARGO_TERM_COLOR=never cargo build --release --lib --bin ysu_gpu_probe 2>&1 | tail -3",
        f"sha256sum {REPO}/target/release/liby.so",
    )
    .env({"HF_HUB_OFFLINE": "1", "PYTHONPATH": f"{REPO}/python",
          "Y_LIB_PATH": f"{REPO}/target/release/liby.so", "YSU_HW_PROFILE": "/root/y_hw_profile"})
)

app = modal.App("y-same-tokens", image=image)

EVERY = {"Y_EXACT_V_INT8": "1", "Y_EXACT_NORM": "triton", "Y_EXACT_ATTN": "y",
         "Y_EXACT_KV_PACK": "inplace", "Y_EXACT_GEMM": "y"}
NO_Y_GEMM = {k: v for k, v in EVERY.items() if k != "Y_EXACT_GEMM"}
REC = "exact_cross_device.py --device cuda --record {json} "
# Speed: stock bf16 (what ships, as on the 4070 Ti) against the exact path,
# both compiled, in turns inside one process, best of 5. Without Y_EXACT_GEMM
# the exact path's GEMM on sm_80+ is Triton's int8 dot.
TP = ("exact_throughput.py --batch 32 --tokens 64 --reps 5 --compile "
      "--model Qwen/Qwen2.5-0.5B-Instruct --loop graphed --stock-loop ")
BOTH = {"Y_THR_ARMS": "stock,exact"}
STAGES = {
    "eager": [
        ("eager", {}, REC + "--arms stock,exact --model Qwen/Qwen2.5-0.5B-Instruct"),
        ("ygemm_every_eager", EVERY, REC + "--arms exact --model Qwen/Qwen2.5-0.5B-Instruct"),
    ],
    "compiled_0p5b": [
        ("ygemm_every_compiled", EVERY, REC + "--arms exact --compile --model Qwen/Qwen2.5-0.5B-Instruct"),
    ],
    "compiled_1p5b": [
        ("ygemm_1p5b_every_compiled", EVERY, REC + "--arms exact --compile --model Qwen/Qwen2.5-1.5B-Instruct"),
    ],
    "speed": [
        ("speed_every_ygemm_stock_graphed", dict(EVERY, **BOTH), TP + "graphed"),
        ("speed_every_triton_stock_graphed", dict(NO_Y_GEMM, **BOTH), TP + "graphed"),
        ("speed_every_ygemm_stock_greedy", dict(EVERY, **BOTH), TP + "greedy"),
    ],
}


def _sh(cmd):
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout


def _record(tag, stage, budget=None):
    """Runs the stage's commands in turn. With `budget` (seconds), a command
    still running when it is spent is killed and its output so far kept, so a
    stage that runs long still returns its finished logs before the
    function's hard timeout."""
    end = time.time() + budget if budget else None
    out = pathlib.Path("/root/out")
    out.mkdir(exist_ok=True)
    files = {}
    head = _sh("nvidia-smi --query-gpu=name,compute_cap,driver_version,clocks.max.sm,memory.total --format=csv; "
               "lscpu | grep -E 'Model name|^CPU\\(s\\)'; "
               f"python -c 'import torch, transformers; print(torch.__version__, transformers.__version__)'; "
               f"sha256sum {REPO}/target/release/liby.so")
    files["machine.txt"] = head
    print(head, flush=True)
    for name, env, cmd in STAGES[stage]:
        t0 = time.time()
        cmd = cmd.replace("{json}", f"{out}/{tag}_{name}.json")
        left = None if end is None else end - time.time()
        if left is not None and left < 60:
            files[f"{tag}_{name}.log"] = f"not run: {left:.0f} s of the budget left\n"
            continue
        try:
            # `exec`, so a kill at the budget reaches python, not just the shell.
            p = subprocess.run(f"cd {REPO}/tools && exec {sys.executable} {cmd}",
                               shell=True, env=dict(os.environ, **env), timeout=left,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            log = p.stdout + f"\nexit {p.returncode} {time.time() - t0:.0f} s\n"
        except subprocess.TimeoutExpired as e:
            got = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            log = got + f"\nKILLED at the budget after {time.time() - t0:.0f} s\n"
        files[f"{tag}_{name}.log"] = log
        print(log[-2500:], flush=True)
        js = out / f"{tag}_{name}.json"
        if js.exists():
            files[js.name] = js.read_text()
    return files


COMMON = dict(timeout=20 * 60, cpu=4.0, memory=16384)


@app.function(gpu="A100-40GB", **COMMON)
def on_a100(stage: str):
    return _record("a100", stage)


@app.function(gpu="H100!", **COMMON)
def on_h100(stage: str):
    return _record("h100", stage)


# Compiling is CPU work (Inductor); on the T4's 4 cores the compiled records
# took 36 and 41 minutes. More cores cost little next to the GPU.
@app.function(gpu="H100!", timeout=40 * 60, cpu=16.0, memory=32768)
def on_h100_long(stage: str):
    return _record("h100", stage)


# Three timed processes, each compiling both arms.
@app.function(gpu="H100!", timeout=60 * 60, cpu=16.0, memory=32768)
def on_h100_speed(stage: str):
    return _record("h100", stage, budget=57 * 60)


@app.local_entrypoint()
def main(stage: str = "eager", out: str = "results", gpus: str = "a100,h100"):
    if stage not in STAGES:
        raise SystemExit(f"unknown stage {stage!r}")
    h100 = on_h100_speed if stage == "speed" else on_h100_long if stage.startswith("compiled") else on_h100
    fns = {"a100": on_a100, "h100": h100}
    calls = [(g, fns[g].spawn(stage)) for g in gpus.split(",")]
    for g, call in calls:
        d = pathlib.Path(out) / f"{g}_{stage}"
        d.mkdir(parents=True, exist_ok=True)
        for name, text in call.get().items():
            (d / name).write_text(text)
        print("wrote", d)
