"""Writes the Kaggle T4 job that times Y's exact path against stock float16 on
a Tesla T4 (for the paper's second speed table).

    python3 make_t4_speed_job.py OUT_DIR [SETUPS, default A,B,C]

Y's source comes from the y-lang-src dataset (67b8fdc: exact_throughput.py with_attention, earlier --stock-dtype
), liby.so from the y-lang-wheel dataset. The model is
Qwen2.5-0.5B-Instruct at 7ae5576. Three processes of tools/exact_throughput.py,
compiled, batch 32, 64 new tokens, best of 5, arms interleaved inside each:
  A: stock fp16, exact (127 levels), fixed-order fp32 control
  B: stock fp16, exact at 511 levels (Y_EXACT_ACT_LEVELS=511)
  C: stock fp16, exact with every Y switch (V int8 on exact_pv, Triton norm,
     Y's decode attention, the in-place packed cache)
"""
import json
import os
import sys

out = sys.argv[1]
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
# A symlinked .ysu once arrived as an empty file and Y refused to build it.
empty = [p for p in glob.glob(f'{{REPO}}/**/*.ysu', recursive=True) if os.path.getsize(p) == 0]
assert not empty, f'empty .ysu files (symlinks lost in the dataset?): {{empty}}'
from huggingface_hub import snapshot_download
MODEL = snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', revision='{QWEN_REV}', local_dir=f'{{OUT}}/qwen05')
import torch, transformers
print('torch', torch.__version__, 'transformers', transformers.__version__, torch.cuda.get_device_name(0))
""")


def run(name, arms, env=""):
    code(f"""
t0 = time.time()
cmd = (f'cd {{REPO}}/tools && {env} Y_THR_ARMS={arms} {{sys.executable}} exact_throughput.py --model {{MODEL}} '
       f'--batch 32 --tokens 64 --reps 5 --compile --stock-dtype float16')
p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
open(f'{{OUT}}/speed_{name}.log', 'w').write(p.stdout)
print(p.stdout[-3000:]); print('exit', p.returncode, f'{{time.time() - t0:.0f}} s')
!nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader
""")


SETUPS = {"A": ("A_127_control", "stock,exact,fixedfloat", ""),
          "B": ("B_511", "stock,exact", "Y_EXACT_ACT_LEVELS=511"),
          "C": ("C_every_switch", "stock,exact",
                "Y_EXACT_V_INT8=1 Y_EXACT_NORM=triton Y_EXACT_ATTN=y Y_EXACT_KV_PACK=inplace")}
for key in (sys.argv[2] if len(sys.argv) > 2 else "A,B,C").split(","):
    run(*SETUPS[key])

os.makedirs(out, exist_ok=True)
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"s{i:02d}"; c["source"] = c["source"].splitlines(keepends=True)
json.dump(nb, open(f"{out}/y-t4-speed.ipynb", "w"), indent=1)
json.dump({"id": "umutkorkmazysu/y-t4-speed", "title": "y-t4-speed", "code_file": "y-t4-speed.ipynb",
           "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
           "enable_internet": True, "dataset_sources": ["umutkorkmazysu/y-lang-wheel", "umutkorkmazysu/y-lang-src"],
           "competition_sources": [], "kernel_sources": []},
          open(f"{out}/kernel-metadata.json", "w"), indent=2)
print("wrote", out)
