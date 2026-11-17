"""Writes a Kaggle T4 job that tests the int8 weight's storage order for
`torch._int_mm` (the 2026-11-17 profile found cuBLAS running a CUDA-core
`volta_igemm_int8_..._nn` kernel for the row-major [K][N] weight).

    python3 make_t4_layout_job.py OUT_DIR SLUG PART

PART a: gemm_shapes (nn, tn, fp16), layout_ab default, profiles of stock
        float16 and exact default (with the decode-only difference).
PART b: layout_ab with every Y switch, layout_ab at 511 levels, and
        logits_to_keep_ab (the cost of the hidden `logits_to_keep`).
PART c: profiles (with host time per op) of stock float16, exact default
        and exact with every Y switch: why is every switch slow on a T4?
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
assert 'host_ops' in open(f'{{REPO}}/docs/benchmarks/exact_t4_profile_2026-11-17/profile_05b.py').read(), 'old profile_05b'
assert 'def keeps_signature' in open(f'{{REPO}}/tools/exact_throughput.py').read(), 'exact_throughput lacks keeps_signature'
# A symlinked .ysu once arrived as an empty file and Y refused to build it.
empty = [p for p in glob.glob(f'{{REPO}}/**/*.ysu', recursive=True) if os.path.getsize(p) == 0]
assert not empty, f'empty .ysu files (symlinks lost in the dataset?): {{empty}}'
from huggingface_hub import snapshot_download
MODEL = snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', revision='{QWEN_REV}', local_dir=f'{{OUT}}/qwen05')
import torch, transformers
print('torch', torch.__version__, 'transformers', transformers.__version__, torch.cuda.get_device_name(0))
""")


P = "docs/benchmarks/exact_t4_profile_2026-11-17"



def run(name, cmd):
    code(f"""
t0 = time.time()
p = subprocess.run(f'cd {{REPO}}/tools && {cmd}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
open(f'{{OUT}}/{name}.log', 'w').write(p.stdout)
print(p.stdout[-4000:]); print('exit', p.returncode, f'{{time.time() - t0:.0f}} s')
!nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader
""")



AB = f"{{sys.executable}} ../{P}/layout_ab.py --stock-dtype float16 --model {{MODEL}}"
EVERY = "Y_EXACT_V_INT8=1 Y_EXACT_NORM=triton Y_EXACT_ATTN=y Y_EXACT_KV_PACK=inplace "
PROF = f"{{sys.executable}} ../{P}/profile_05b.py"
if PART == "a":
    run("gemm_shapes_layout", f"{{sys.executable}} ../{P}/gemm_shapes.py {{OUT}}/gemm_shapes_layout.json")
    run("ab_default", AB)
    run("prof_stock_fp16", f"{PROF} stock {{OUT}}/prof_stock_fp16.json --stock-dtype float16 --model {{MODEL}}")
    run("prof_exact_default", f"{PROF} exact {{OUT}}/prof_exact_default.json --model {{MODEL}}")
elif PART == "b":
    run("ab_every_switch", EVERY + AB)
    run("ab_511", "Y_EXACT_ACT_LEVELS=511 " + AB)
    run("logits_to_keep_ab", f"{{sys.executable}} ../docs/benchmarks/exact_speed_05b_2026-11-17/logits_to_keep_ab.py "
        f"--stock-dtype float16 --model {{MODEL}}")
elif PART == "c":
    run("prof_stock_fp16", f"{PROF} stock {{OUT}}/prof_stock_fp16.json --stock-dtype float16 --model {{MODEL}}")
    run("prof_exact_default", f"{PROF} exact {{OUT}}/prof_exact_default.json --model {{MODEL}}")
    run("prof_exact_every_switch", EVERY + f"{PROF} exact {{OUT}}/prof_exact_every_switch.json --model {{MODEL}}")
else:
    raise SystemExit(f"PART {PART!r}: expected a, b or c")

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
