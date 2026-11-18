"""Writes a Kaggle T4 job that times the exact path with each decode step one
CUDA graph (`--loop graphed`, y_lang.exact.graphed) against the greedy loop.

    python3 make_t4_graphed_job.py OUT_DIR SLUG PART

PART e: exact at its defaults, --loop graphed then --loop greedy.
PART f: exact with every switch, --loop greedy then --loop graphed.
PART g: exact --loop graphed against stock --stock-loop greedy (stock's
faster loop on the T4, job e) in one process: defaults, then every switch.
PART h: docs/benchmarks/exact_graphed_2026-11-18/capacity_sweep.py, the
graphed step's time for buffers of 92 to 256 rows: stock fp16 and exact at
its defaults, then exact with every switch.
PART i: docs/benchmarks/exact_graphed_2026-11-18/profile_step.py, the
graphed step's GPU time kernel by kernel: stock fp16 and exact at its
defaults, then exact with every switch.
Stock is float16 in each process. Both loops of a setup run on one machine,
so graphed against greedy is not a comparison across two Kaggle machines.
Y's source comes from the y-lang-src dataset, liby.so from y-lang-wheel.
Qwen2.5-0.5B-Instruct at 7ae5576.
"""
import json
import os
import sys

out, SLUG, PART = sys.argv[1], sys.argv[2], sys.argv[3]
QWEN_REV = "7ae557604adf67be50417f59c2c2f167def9a775"
cells = []


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n")})


code("""
!nvidia-smi --query-gpu=index,name,compute_cap,driver_version,clocks.max.sm --format=csv
!lscpu | grep -E 'Model name|^CPU\\(s\\)' | cut -c1-200
""")
code("""
import glob, os, subprocess, shutil, sys, time, hashlib
WHEEL = sorted(glob.glob('/kaggle/input/**/y_lang-*.whl', recursive=True))[-1]
!pip uninstall -y -q torchvision torchaudio
!pip install -q torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
!pip install -q transformers==5.15.0 tokenizers==0.22.2 safetensors mpmath z3-solver huggingface_hub
!pip install -q --no-deps --force-reinstall "{WHEEL}"
""")
code(f"""
OUT, REPO = '/kaggle/working', '/kaggle/working/Y'
shutil.rmtree(REPO, ignore_errors=True)
roots = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/python/y_lang/__init__.py', recursive=True)]
shutil.copytree(os.path.dirname(os.path.dirname(roots[0])), REPO, dirs_exist_ok=True)
LIBY = subprocess.run([sys.executable, '-c', 'import y_lang, os; print(os.path.join(os.path.dirname(y_lang.__file__), "liby.so"))'],
                      capture_output=True, text=True).stdout.strip()
os.environ.update(PYTHONPATH=f'{{REPO}}/python', Y_LIB_PATH=LIBY)
print('liby.so sha256', hashlib.sha256(open(LIBY, 'rb').read()).hexdigest()[:16])
print('exact_throughput has --stock-dtype:', '--stock-dtype' in open(f'{{REPO}}/tools/exact_throughput.py').read())
print('exact_throughput has with_attention:', 'def with_attention' in open(f'{{REPO}}/tools/exact_throughput.py').read())
assert 'm.w8t = m.w8t.t().contiguous().t()' in open(f'{{REPO}}/python/y_lang/exact/model.py').read(), 'model.py lacks the layout change'
assert '"graphed"' in open(f'{{REPO}}/tools/exact_throughput.py').read(), 'exact_throughput lacks --loop graphed'
assert 'def time_ratio' in open(f'{{REPO}}/tools/exact_throughput.py').read(), 'exact_throughput lacks time_ratio'
assert '--stock-loop' in open(f'{{REPO}}/tools/exact_throughput.py').read(), 'exact_throughput lacks --stock-loop'
assert 'def keeps_signature' in open(f'{{REPO}}/tools/exact_throughput.py').read(), 'exact_throughput lacks keeps_signature'
# A symlinked .ysu once arrived as an empty file and Y refused to build it.
empty = [p for p in glob.glob(f'{{REPO}}/**/*.ysu', recursive=True) if os.path.getsize(p) == 0]
assert not empty, f'empty .ysu files (symlinks lost in the dataset?): {{empty}}'
from huggingface_hub import snapshot_download
MODEL = snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', revision='{QWEN_REV}', local_dir=f'{{OUT}}/qwen05')
import torch, transformers
print('torch', torch.__version__, 'transformers', transformers.__version__, torch.cuda.get_device_name(0))
""")


def run(name, cmd):
    code(f"""
t0 = time.time()
p = subprocess.run(f'cd {{REPO}}/tools && {cmd}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
open(f'{{OUT}}/{name}.log', 'w').write(p.stdout)
print(p.stdout[-4000:]); print('exit', p.returncode, f'{{time.time() - t0:.0f}} s')
!nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader
""")




EVERY = "Y_EXACT_V_INT8=1 Y_EXACT_NORM=triton Y_EXACT_ATTN=y Y_EXACT_KV_PACK=inplace "
TP = "Y_THR_ARMS=stock,exact {sys.executable} exact_throughput.py --batch 32 --tokens 64 --reps 5 --compile --stock-dtype float16 --model {MODEL} --loop "
SWEEP = "{sys.executable} {REPO}/docs/benchmarks/exact_graphed_2026-11-18/capacity_sweep.py --stock-dtype float16 --model {MODEL}"
PROF = "{sys.executable} {REPO}/docs/benchmarks/exact_graphed_2026-11-18/profile_step.py --stock-dtype float16 --model {MODEL}"
if PART == "i":
    run("profile_default", PROF)
    run("profile_every_switch", EVERY + "Y_THR_ARMS=exact " + PROF)
elif PART == "h":
    run("sweep_default", SWEEP)
    run("sweep_every_switch", EVERY + "Y_THR_ARMS=exact " + SWEEP)
elif PART == "g":
    run("mixed_default", TP + "graphed --stock-loop greedy")
    run("mixed_every_switch", EVERY + TP + "graphed --stock-loop greedy")
else:
    SETUP, ENV, LOOPS = {"e": ("default", "", ("graphed", "greedy")),
                         "f": ("every_switch", EVERY, ("greedy", "graphed"))}[PART]
    for loop in LOOPS:
        run(f"{loop}_{SETUP}", ENV + TP + loop)

os.makedirs(out, exist_ok=True)
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"s{i:02d}"; c["source"] = c["source"].splitlines(keepends=True)
json.dump(nb, open(f"{out}/{SLUG}.ipynb", "w"), indent=1)
json.dump({"id": f"umutkorkmazysu/{SLUG}", "title": SLUG, "code_file": f"{SLUG}.ipynb",
           "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
           "enable_internet": True, "dataset_sources": ["umutkorkmazysu/y-lang-wheel", "umutkorkmazysu/y-lang-src"],
           "competition_sources": [], "kernel_sources": []},
          open(f"{out}/kernel-metadata.json", "w"), indent=2)
print("wrote", out)
