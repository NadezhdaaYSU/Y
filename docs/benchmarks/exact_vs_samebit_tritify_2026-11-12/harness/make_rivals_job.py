"""Writes the Kaggle T4 job comparing Y's exact path with samebit and tritify.

    python3 make_rivals_job.py RIVALS_DIR OUT_DIR

RIVALS_DIR holds the harness scripts (prefill_vs_samebit.py,
accuracy_same_tokens.py, chat_request_y2.py), tritify_bench.patch (timing
only) and samebit/tools/qwen2/calibration.json (the pinned maxima the local
run used, so samebit builds the same integer model: fingerprint
aba0455663f2b5ac). Everything is pinned: Qwen2.5-0.5B-Instruct@7ae5576,
Salesforce/wikitext@b08601e, samebit@c6f3530, tritify@db52e0c, Rust 1.90.0.
Y comes from the y-lang-src dataset with the wheel's liby.so.
"""
import json
import os
import sys

rivals, out = sys.argv[1], sys.argv[2]
QWEN_REV = "7ae557604adf67be50417f59c2c2f167def9a775"
WIKI_REV = "b08601e04326c79dfdd32d625aee71d232d685c3"
SAMEBIT, TRITIFY = "c6f3530c9d21345f4b41c63c26409582897c2593", "db52e0c3d5cc3fbd1771fe7fa8f595a4b1266ae5"
files = {n: open(os.path.join(rivals, n)).read() for n in
         ("prefill_vs_samebit.py", "accuracy_same_tokens.py", "chat_request_y2.py", "tritify_bench.patch")}
calib = open(os.path.join(rivals, "samebit/tools/qwen2/calibration.json")).read()
cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n")})


md("# Y vs samebit vs tritify on a Tesla T4 (accuracy, prefill, decode, bits)")
code("""
!nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv
!lscpu | grep -E 'Model name|^CPU\\(s\\)|Flags' | cut -c1-400
!free -g | head -2
""")
code("""
import glob, os, subprocess, shutil, sys, time, json
WHEEL = sorted(glob.glob('/kaggle/input/**/y_lang-*.whl', recursive=True))[-1]
!pip uninstall -y -q torchvision torchaudio
!pip install -q torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
!pip install -q transformers==5.15.0 tokenizers==0.22.2 safetensors mpmath z3-solver pyarrow cupy-cuda12x huggingface_hub
!pip install -q --no-deps --force-reinstall "{WHEEL}"
""")
code("""
OUT, REPO, RIV = '/kaggle/working', '/kaggle/working/Y', '/kaggle/working/rivals'
shutil.rmtree(REPO, ignore_errors=True)
roots = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/python/y_lang/__init__.py', recursive=True)]
shutil.copytree(os.path.dirname(os.path.dirname(roots[0])), REPO, dirs_exist_ok=True)
LIBY = subprocess.run([sys.executable, '-c', 'import y_lang, os; print(os.path.join(os.path.dirname(y_lang.__file__), "liby.so"))'],
                      capture_output=True, text=True).stdout.strip()
os.makedirs(RIV, exist_ok=True)
os.environ.update(PYTHONPATH=f'{REPO}/python', Y_LIB_PATH=LIBY, OMP_NUM_THREADS=str(os.cpu_count()))
os.environ.pop('HF_HUB_OFFLINE', None)
""")
code("FILES = " + repr(files) + "\nfor n, t in FILES.items():\n    open(f'{RIV}/{n}', 'w').write(t)\nprint(sorted(os.listdir(RIV)))")
code(f"""
from huggingface_hub import snapshot_download, hf_hub_download
W = f'{{RIV}}/samebit_work'
os.makedirs(f'{{W}}/data', exist_ok=True)
snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', revision='{QWEN_REV}', local_dir=f'{{W}}/model')
for split in ('test', 'train'):
    p = hf_hub_download('Salesforce/wikitext', f'wikitext-2-raw-v1/{{split}}-00000-of-00001.parquet', repo_type='dataset',
                        revision='{WIKI_REV}', local_dir=f'{{RIV}}/wt')
    shutil.copy(p, f'{{W}}/data/wt2-{{split}}.parquet')
!cd {{RIV}} && git clone -q https://github.com/Shakhtar-Sankur/samebit.git && cd samebit && git checkout -q {SAMEBIT} && git log -1 --format='samebit %h %ci'
""")
code("open(f'{RIV}/samebit/tools/qwen2/calibration.json', 'w').write(" + repr(calib) + ")\n" + """
t0 = time.time()
!cd {RIV}/samebit/tools/qwen2 && QWEN_WORK={W} PYTHONPATH={RIV}/samebit/src python convert.py 2>&1 | tail -2
print(f'convert {time.time() - t0:.0f} s (local fingerprint aba0455663f2b5ac)')
""")


def run(name, args, env=""):
    code(f"""
t0 = time.time()
p = subprocess.run(f"cd {{RIV}} && {env} {{sys.executable}} {args} > {{OUT}}/{name}.log 2>&1", shell=True)
print(f'{name}: exit {{p.returncode}}, {{(time.time() - t0) / 60:.1f}} min')
lines = [l for l in open(f'{{OUT}}/{name}.log').read().splitlines()
         if not any(k in l for k in ('USDT', 'Loading weights', 'Warning', 'warn'))]
print('\\n'.join(lines[-25:]))
""")


for act in ("127", "511"):
    e = f"Y_EXACT_ACT_LEVELS={act}"
    run(f"accuracy_act{act}", f"accuracy_same_tokens.py {{RIV}}/samebit/src {{W}} {{REPO}} --windows 40 --out {{OUT}}/accuracy_act{act}", e)
    run(f"prefill_act{act}", f"prefill_vs_samebit.py {{RIV}}/samebit/src {{W}} {{REPO}} --rounds 11 --reps 5 --compiled --out {{OUT}}/prefill_act{act}.json", e)
    run(f"y_chat_eager_act{act}", f"chat_request_y2.py {{REPO}} {{W}}/model 32 --rounds 7 --out {{OUT}}/y_chat_eager_act{act}.json", e)
    run(f"y_chat_compiled_act{act}", f"chat_request_y2.py {{REPO}} {{W}}/model 32 --rounds 7 --compiled --devices cuda --out {{OUT}}/y_chat_compiled_act{act}.json", e)
    run(f"y_128_eager_act{act}", f"chat_request_y2.py {{REPO}} {{W}}/model 128 --no-eos --rounds 5 --out {{OUT}}/y_128_eager_act{act}.json", e)
    run(f"y_128_compiled_act{act}", f"chat_request_y2.py {{REPO}} {{W}}/model 128 --no-eos --rounds 5 --compiled --devices cuda --out {{OUT}}/y_128_compiled_act{act}.json", e)

md("## tritify (pinned commit, timing patch only)")
code(f"""
!curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal -q --default-toolchain 1.90.0
os.environ['PATH'] = os.path.expanduser('~/.cargo/bin') + ':/usr/local/cuda/bin:' + os.environ['PATH']
!cd {{RIV}} && git clone -q https://github.com/pelloran-inc/tritify.git && cd tritify && git checkout -q {TRITIFY} && git apply {{RIV}}/tritify_bench.patch && git log -1 --format='tritify %h %ci' && git diff --stat
t0 = time.time()
p = subprocess.run(f"cd {{RIV}}/tritify && CUDA_HOME=/usr/local/cuda KT_CUDA_ARCHS=75 cargo build --release -p kt-convert "
                   f"--example generate_backends --features kt-runtime/cuda > {{OUT}}/tritify_build.log 2>&1", shell=True)
print(f'build exit {{p.returncode}}, {{(time.time() - t0) / 60:.1f}} min')
print(open(f'{{OUT}}/tritify_build.log').read()[-600:])
""")
code("""
for name, n, extra in (('tritify_chat', 32, 'KT_BENCH_REPEAT=7'), ('tritify_128', 128, 'KT_BENCH_REPEAT=5 KT_NO_EOS=1')):
    t0 = time.time()
    p = subprocess.run(f"cd {RIV}/tritify && LD_LIBRARY_PATH=/usr/local/cuda/lib64 {extra} "
                       f"./target/release/examples/generate_backends {W}/model {n} > {OUT}/{name}.log 2>&1", shell=True)
    print(f'{name}: exit {p.returncode}, {(time.time() - t0) / 60:.1f} min')
    print('\\n'.join(l[:220] for l in open(f'{OUT}/{name}.log').read().splitlines()
                    if any(k in l for k in ('PROMPT_IDS', 'BENCH', 'logits_hash', 'VERDICT'))))
""")

os.makedirs(out, exist_ok=True)
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"r{i:02d}"
    c["source"] = c["source"].splitlines(keepends=True)
json.dump(nb, open(f"{out}/y-rivals-t4.ipynb", "w"), indent=1)
json.dump({"id": "umutkorkmazysu/y-rivals-t4", "title": "y-rivals-t4", "code_file": "y-rivals-t4.ipynb",
           "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
           "enable_internet": True, "dataset_sources": ["umutkorkmazysu/y-lang-wheel", "umutkorkmazysu/y-lang-src"],
           "competition_sources": [], "kernel_sources": []}, open(f"{out}/kernel-metadata.json", "w"), indent=2)
print("wrote", out, len(cells), "cells")
