"""Exact training (T2) on an H100: the fast path's 5000-step record, compared
on the development machine against the CPU record of
docs/benchmarks/exact_train_lm_2026-11-21/, and step times against float32.

    git archive --format=tar.gz -o /abs/y_src.tar.gz HEAD     # in the repository
    Y_SRC_TAR=/abs/y_src.tar.gz modal run modal_train.py --stage t2 --out DIR
    Y_SRC_TAR=/abs/y_src.tar.gz modal run modal_train.py --stage act32 --out DIR

Stage `act32` (float32 activations) also runs its CPU reference record on a
CPU-only container.

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
# Inductor's and Triton's caches, kept between runs: compiling on Modal's
# CPUs took 4 to 9 minutes a model, more than the runs themselves. The cache
# keys cover the sources, the config and y_lang.exact.model's tag, so a
# changed graph compiles again.
CACHE = modal.Volume.from_name("y-inductor-cache", create_if_missing=True)

REC = "exact_train_lm.py --device cuda --record {out}/"
SPEED_MODES = "exact32:compile+graph,exact:compile+graph,f32:compile+graph,bf16:compile+graph"
STAGES = {"t2": [
    ("exact_y_g12b8_fast", REC + "h100_exact_y_g12b8_fast.json --mm y --graph --compile"),
    ("float32_fast", REC + "h100_float32_fast.json --arm float32 --graph --compile"),
    ("float32_eager", REC + "h100_float32_eager.json --arm float32"),
    ("speed_medium", "exact_train_speed.py --size medium --check 3 --steady 50 "
                     "--out {out}/h100_speed_medium.json"),
    ("speed_tiny", "exact_train_speed.py --size tiny --check 30 --steady 500 "
                   "--out {out}/h100_speed_tiny.json"),
],
# Float32 activations (Ops(act=torch.float32)), with bf16 mixed precision as
# a second baseline.
"act32": [
    ("exact32_y_g12b8_fast", REC + "h100_exact32_y_g12b8_fast.json --mm y --act float32 --graph --compile"),
    ("speed32_medium", f"exact_train_speed.py --size medium --check 3 --steady 50 --modes {SPEED_MODES} "
                       "--out {out}/h100_speed32_medium.json"),
    ("speed32_tiny", f"exact_train_speed.py --size tiny --check 30 --steady 500 --modes {SPEED_MODES} "
                     "--out {out}/h100_speed32_tiny.json"),
]}
STAGES["profile"] = [
    ("profile_medium_exact32", "exact_train_profile.py --size medium --arm exact32 "
                               "--out {out}/h100_profile_medium_exact32.txt"),
    ("profile_medium_bf16", "exact_train_profile.py --size medium --arm bf16 "
                            "--out {out}/h100_profile_medium_bf16.txt"),
]
# Inductor's own tuning (block sizes by coordinate descent, autotuned
# pointwise configs): environment variables only, the same code.
TUNED = {"TORCHINDUCTOR_COORDINATE_DESCENT_TUNING": "1", "TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE": "1"}
STAGES["tune"] = [
    ("speed32_medium_tuned", "exact_train_speed.py --size medium --check 3 --steady 50 "
                             "--modes exact32:compile+graph --out {out}/h100_speed32_medium_tuned.json", TUNED),
    ("profile_medium_exact32_tuned", "exact_train_profile.py --size medium --arm exact32 "
                                     "--out {out}/h100_profile_medium_exact32_tuned.txt", TUNED),
    ("profile_medium_exact32", "exact_train_profile.py --size medium --arm exact32 "
                               "--out {out}/h100_profile_medium_exact32.txt"),
]
# The float32 quantizers as float32 arithmetic (5f9d9c1), timed and profiled.
STAGES["f32formula"] = [
    ("speed32f_medium", "exact_train_speed.py --size medium --check 3 --steady 50 "
                        "--modes exact32:compile+graph,f32:compile+graph,bf16:compile+graph "
                        "--out {out}/h100_speed32f_medium.json"),
    ("profile32f_medium", "exact_train_profile.py --size medium --arm exact32 "
                          "--out {out}/h100_profile32f_medium_exact32.txt"),
]
# Attention's products on int8 tensor cores (int8_bmm) and the causal softmax
# with its quantizer as one kernel (softmax_kernel): the GPU tests, the
# 5000-step record against the CPU's, speed with Y's and cuBLAS's GEMM.
TESTS = ("cd {repo} && exec env PYTHONPATH=python:python/tests " + sys.executable + " -m unittest "
         "python.tests.test_exact_int8_bmm python.tests.test_exact_softmax_kernel "
         "python.tests.test_exact_train_lm.OnTheGpu.test_float32_activations_give_the_cpu_weights "
         "python.tests.test_exact_train_lm.OnTheGpu.test_compiled_graph_gives_the_cpu_weights")
STAGES["kernels"] = [
    ("gpu_tests", "RAW:" + TESTS),
    ("exact32_kernels_fast", REC + "h100_exact32_kernels_fast.json --mm y --act float32 --graph --compile"),
    ("speedk_medium", "exact_train_speed.py --size medium --check 3 --steady 50 "
                      "--modes exact32:compile+graph,f32:compile+graph,bf16:compile+graph "
                      "--out {out}/h100_speedk_medium.json"),
    ("speedk_medium_cublas", "exact_train_speed.py --size medium --check 3 --steady 50 --mm torch "
                             "--modes exact32:compile+graph --out {out}/h100_speedk_medium_cublas.json"),
    ("profilek_medium", "exact_train_profile.py --size medium --arm exact32 "
                        "--out {out}/h100_profilek_medium_exact32.txt"),
]
# The CPU reference for a stage, on a CPU-only container.
CPU_RUNS = {"act32": [("cpu_exact32_g12b8", "exact_train_lm.py --device cpu --record "
                                            "{out}/cpu_exact32_g12b8.json --act float32")]}


def _sh(cmd):
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout


def _run(runs, end):
    out = pathlib.Path("/root/out")
    out.mkdir(exist_ok=True)
    files = {}
    head = _sh("nvidia-smi --query-gpu=name,compute_cap,driver_version,clocks.max.sm,memory.total --format=csv; "
               "lscpu | grep -E 'Model name|^CPU\\(s\\)'; "
               "python -c 'import torch, triton; print(torch.__version__, triton.__version__)'; "
               f"sha256sum {REPO}/target/release/liby.so")
    files["machine.txt"] = head
    print(head, flush=True)
    for name, cmd, *env in runs:
        env = dict(os.environ, **(env[0] if env else {}))
        t0 = time.time()
        left = end - time.time()
        if left < 60:
            files[f"{name}.log"] = f"not run: {left:.0f} s of the budget left\n"
            continue
        try:
            # `exec`, so a kill at the budget reaches python, not just the shell.
            line = (cmd[4:].format(repo=REPO) if cmd.startswith("RAW:")
                    else f"cd {REPO}/tools && exec {sys.executable} {cmd.format(out=out)}")
            p = subprocess.run(line,
                               shell=True, timeout=left, text=True, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log = p.stdout + f"\nexit {p.returncode} {time.time() - t0:.0f} s\n"
        except subprocess.TimeoutExpired as e:
            got = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            log = got + f"\nKILLED at the budget after {time.time() - t0:.0f} s\n"
        files[f"{name}.log"] = log
        print(log[-2500:], flush=True)
        for js in [*out.glob("*.json"), *out.glob("*.txt")]:
            files[js.name] = js.read_text()
    return files


@app.function(gpu="H100!", timeout=45 * 60, cpu=16.0, memory=32768, volumes={"/cache": CACHE})
def on_h100(stage: str):
    os.environ.update(TORCHINDUCTOR_CACHE_DIR="/cache/inductor", TRITON_CACHE_DIR="/cache/triton")
    try:
        return _run(STAGES[stage], time.time() + 42 * 60)
    finally:
        CACHE.commit()


@app.function(timeout=60 * 60, cpu=8.0, memory=16384)
def on_cpu(stage: str):
    return _run(CPU_RUNS[stage], time.time() + 57 * 60)


@app.local_entrypoint()
def main(stage: str = "t2", out: str = "results"):
    d = pathlib.Path(out)
    d.mkdir(parents=True, exist_ok=True)
    calls = [("gpu", on_h100.spawn(stage))]
    if stage in CPU_RUNS:
        calls.append(("cpu", on_cpu.spawn(stage)))
    for tag, call in calls:
        for name, text in call.get().items():
            (d / (f"{tag}_{name}" if name == "machine.txt" else name)).write_text(text)
    print("wrote", d)
