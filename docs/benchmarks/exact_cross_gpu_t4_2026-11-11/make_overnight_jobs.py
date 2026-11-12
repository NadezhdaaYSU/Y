"""Writes a Kaggle notebook + kernel-metadata.json for one overnight job.

    python3 make_job.py JOB OUT_DIR

Record jobs run tools/exact_cross_device.py from the y-lang-src dataset with
the wheel's liby.so; the cargo job clones the public repository and runs the
Rust suites on the CPU.
"""
import json
import os
import sys

M05, M15, M3, M7 = ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct",
                    "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct")
ALL = {"Y_EXACT_V_INT8": "1", "Y_EXACT_NORM": "triton", "Y_EXACT_ATTN": "y", "Y_EXACT_KV_PACK": "inplace"}

JOBS = {
    # name: (gpu?, [(record name, args, env)])
    "y-night-t4": (True, [
        ("t4_0p5b_full", ["--arms", "stock,exact", "--model", M05, "--full"], {}),
        ("t4_1p5b", ["--arms", "stock,exact", "--model", M15], {}),
        ("t4_3b", ["--arms", "stock,exact", "--model", M3], {}),
        ("t4_7b_all_compiled", ["--arms", "exact", "--model", M7, "--low-memory", "--compile"], ALL),
    ]),
    "y-night-cpu-0p5b": (False, [("cpu_0p5b_full", ["--arms", "stock,exact", "--model", M05, "--full"], {})]),
    "y-night-cpu-1p5b": (False, [("cpu_1p5b", ["--arms", "stock,exact", "--model", M15], {})]),
    "y-night-cpu-3b": (False, [("cpu_3b", ["--arms", "stock,exact", "--model", M3], {})]),
    "y-t4-3b": (True, [("t4_3b", ["--arms", "stock,exact", "--model", M3, "--low-memory"], {})]),
    "y-night-cargo": (False, None),
}

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": t.strip("\n")})


def records(gpu, runs):
    md(f"# Y overnight: {'Tesla T4' if gpu else 'Kaggle CPU'} records")
    code("""
!nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv || true
!lscpu | grep -E 'Model name|^CPU\\(s\\)'
!free -g | head -2
import sys, os, platform
print(sys.version.split()[0], platform.platform(), 'cpus', os.cpu_count())
""")
    index = "cu126" if gpu else "cpu"
    code(f"""
import glob, subprocess, shutil, time, json
WHEEL = sorted(glob.glob('/kaggle/input/**/y_lang-*.whl', recursive=True))[-1]
!pip uninstall -y -q torchvision torchaudio
!pip install -q torch==2.13.0 --index-url https://download.pytorch.org/whl/{index}
!pip install -q transformers==5.15.0 tokenizers==0.22.2 safetensors mpmath z3-solver
!pip install -q --no-deps --force-reinstall "{{WHEEL}}"
""")
    code("""
OUT, REPO = '/kaggle/working', '/kaggle/working/Y'
shutil.rmtree(REPO, ignore_errors=True)
roots = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/python/y_lang/__init__.py', recursive=True)]
shutil.copytree(os.path.dirname(os.path.dirname(roots[0])), REPO, dirs_exist_ok=True)
empty = [p for p in glob.glob(f'{REPO}/**/*.ysu', recursive=True) if os.path.getsize(p) == 0]
assert not empty, empty
LIBY = subprocess.run([sys.executable, '-c',
                       'import y_lang, os; print(os.path.join(os.path.dirname(y_lang.__file__), "liby.so"))'],
                      capture_output=True, text=True).stdout.strip()
ENV = dict(os.environ, PYTHONPATH=f'{REPO}/python', Y_LIB_PATH=LIBY, OMP_NUM_THREADS=str(os.cpu_count()))
ENV.pop('HF_HUB_OFFLINE', None)

def run(name, args, extra):
    t0 = time.time()
    with open(f'{OUT}/{name}.log', 'w') as log:
        p = subprocess.run([sys.executable, 'exact_cross_device.py', '--record', f'{OUT}/{name}.json'] + args,
                           cwd=f'{REPO}/tools', env=dict(ENV, **extra), stdout=log, stderr=subprocess.STDOUT)
    lines = open(f'{OUT}/{name}.log').read().splitlines()
    print(f'--- {name}: exit {p.returncode}, {(time.time() - t0) / 60:.1f} min')
    print('\\n'.join(l for l in lines if 'tokens ' in l or 'self-check' in l or 'Error' in l)[-1500:])
""")
    dev = ["--device", "cuda"] if gpu else ["--device", "cpu"]
    for name, args, env in runs:
        code(f"run({name!r}, {dev + args!r}, {env!r})")


def cargo():
    md("# Y overnight: the Rust suites on Kaggle's CPU (no GPU)")
    code("""
!lscpu | grep -E 'Model name|^CPU\\(s\\)'; free -g | head -2; df -h /kaggle/working | tail -1
!pip install -q z3-solver
!curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal -q
import os, subprocess, time
os.environ['PATH'] = os.path.expanduser('~/.cargo/bin') + ':' + os.environ['PATH']
!rustc --version; cargo --version; which z3 clang gdb ptxas 2>&1 | head
!git clone -q --depth 1 https://github.com/NadezhdaaYSU/Y.git /kaggle/working/Y
!cd /kaggle/working/Y && git log -1 --format='source: %h %s'
""")
    code("""
t0 = time.time()
with open('/kaggle/working/cargo_test.log', 'w') as log:
    p = subprocess.run(['cargo', 'test', '--no-fail-fast'], cwd='/kaggle/working/Y', stdout=log,
                       stderr=subprocess.STDOUT, env=dict(os.environ, CARGO_TERM_COLOR='never'))
print('cargo test exit', p.returncode, f'{(time.time() - t0) / 60:.1f} min')
text = open('/kaggle/working/cargo_test.log').read()
import re
res = re.findall(r'test result: \\w+\\. (\\d+) passed; (\\d+) failed; (\\d+) ignored', text)
print('binaries', len(res), 'passed', sum(int(a) for a, _, _ in res), 'failed', sum(int(b) for _, b, _ in res))
print('\\n'.join(sorted(set(re.findall(r'^test (\\S+) \\.\\.\\. FAILED', text, re.M))))[:4000])
print('\\n'.join(l for l in text.splitlines() if l.startswith('error') )[:3000])
""")


def main():
    job, out = sys.argv[1], sys.argv[2]
    gpu, runs = JOBS[job]
    if runs is None:
        cargo()
    else:
        records(gpu, runs)
    os.makedirs(out, exist_ok=True)
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                      "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"n{i:02d}"
        c["source"] = c["source"].splitlines(keepends=True)
    json.dump(nb, open(f"{out}/{job}.ipynb", "w"), indent=1)
    sources = [] if runs is None else ["umutkorkmazysu/y-lang-wheel", "umutkorkmazysu/y-lang-src"]
    json.dump({"id": f"umutkorkmazysu/{job}", "title": job, "code_file": f"{job}.ipynb",
               "language": "python", "kernel_type": "notebook", "is_private": True,
               "enable_gpu": gpu, "enable_internet": True, "dataset_sources": sources,
               "competition_sources": [], "kernel_sources": []},
              open(f"{out}/kernel-metadata.json", "w"), indent=2)
    print("wrote", out, len(cells), "cells")


main()
