"""Writes a Kaggle T4 job for Y's int8 GEMM on Turing (sm_75):
docs/benchmarks/int8_turing_2026-11-18/.

    python3 make_t4_graphed_job.py OUT_DIR SLUG PART

PART p: probe_m8n8k16.py, the int8 m8n8k16 fragment layout on the T4.
PART b: liby.so built here, then gemm_t4.py (same bits and time against
        torch._int_mm) and python/y_lang/tests/test_int8_mm.py.
PART x: liby.so built here, then exact_cross_device.py records with Y_EXACT_GEMM=y
        (0.5B eager both arms, 0.5B and 1.5B compiled with every switch).
PART s: liby.so built here, then exact_throughput at batch 1 and 8, every switch on
        Y's GEMM, graphed against stock greedy.
PART t: liby.so built here, then test_int8_mm.py and gemm_t4.py (round 1, the
        default now the Turing tile rule).
PART b2: liby.so built here, then gemm_t4.py round2 (tiles near round 1's best,
         at M = 32, 128 and 2048).
PART e: liby.so built here, then exact_throughput, exact graphed against
        stock greedy: every switch on torch._int_mm, every switch on Y's
        GEMM, Y's GEMM at the defaults.
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
assert 'probe_m8n8k16' in str(glob.glob(f'{{REPO}}/docs/benchmarks/int8_turing_2026-11-18/*')), 'no probe'
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





def build_liby():
    """liby.so and ysu_gpu_probe built here from the dataset's sources (Rust
    1.97.1, as on the machine that wrote them), over the installed wheel's,
    which predates the Turing kernel. (A liby.so built on that machine needs
    GLIBC_2.39; this image turned out to have 2.39 too, so it would also have
    loaded. Building here keeps the job from depending on the wheel.)"""
    code(r"""
!curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal -q --default-toolchain 1.97.1
os.environ['PATH'] = os.path.expanduser('~/.cargo/bin') + ':' + os.environ['PATH']
!rustc --version; ldd --version | head -1
t0 = time.time()
p = subprocess.run('cargo build --release --lib --bin ysu_gpu_probe', shell=True, cwd=REPO,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=dict(os.environ, CARGO_TERM_COLOR='never'))
open(f'{OUT}/cargo_build.log', 'w').write(p.stdout)
print(p.stdout[-1500:]); print('cargo build exit', p.returncode, f'{(time.time() - t0) / 60:.1f} min')
assert p.returncode == 0, 'cargo build failed'
built = f'{REPO}/target/release/liby.so'
assert b'four per m16n8k32 step' in open(built, 'rb').read(), 'the built liby.so has no Turing int8 kernel'
for name in ('liby.so', 'ysu_gpu_probe'):
    shutil.copy2(f'{REPO}/target/release/{name}', os.path.join(os.path.dirname(LIBY), name))
print('liby.so now', hashlib.sha256(open(LIBY, 'rb').read()).hexdigest()[:16])
""")


D = "{REPO}/docs/benchmarks/int8_turing_2026-11-18"
EVERY = "Y_EXACT_V_INT8=1 Y_EXACT_NORM=triton Y_EXACT_ATTN=y Y_EXACT_KV_PACK=inplace "
TP = ("Y_THR_ARMS=stock,exact {sys.executable} exact_throughput.py --batch 32 --tokens 64 --reps 5 "
      "--compile --stock-dtype float16 --model {MODEL} --loop graphed --stock-loop greedy")
# `unittest discover`: `python -m unittest /abs/path/test_x.py` is not a module name and
# fails to import (job b, 2026-11-18).
TESTS = ("PYTHONPATH={REPO}/python:{REPO}/python/y_lang/tests {sys.executable} -m unittest discover -v "
         "-s {REPO}/python/y_lang/tests -p test_int8_mm.py")
if PART == "p":
    run("probe_m8n8k16", "{sys.executable} " + D + "/probe_m8n8k16.py")
elif PART == "b":
    build_liby()
    run("gemm_t4", "{sys.executable} " + D + "/gemm_t4.py")
    run("test_int8_mm", TESTS)
elif PART == "x":
    # Same tokens as the CPU with Y's Turing GEMM: tools/exact_cross_device.py
    # records, as exact_cross_gpu_t4_2026-11-11 made them (hub model ids, so the
    # workload hashes match the reference records); compared on the
    # development machine against them.
    build_liby()
    REC = "{sys.executable} exact_cross_device.py --device cuda --record {OUT}/"
    run("t4_ygemm_eager", "Y_EXACT_GEMM=y " + REC + "t4_ygemm_eager.json --arms stock,exact --model Qwen/Qwen2.5-0.5B-Instruct")
    run("t4_ygemm_all_compiled", EVERY + "Y_EXACT_GEMM=y " + REC + "t4_ygemm_all_compiled.json --arms exact --compile --model Qwen/Qwen2.5-0.5B-Instruct")
    run("t4_ygemm_1p5b_all_compiled", EVERY + "Y_EXACT_GEMM=y " + REC + "t4_ygemm_1p5b_all_compiled.json --arms exact --compile --model Qwen/Qwen2.5-1.5B-Instruct")
elif PART == "s":
    # Small batches, every switch on Y's GEMM, graphed against stock greedy.
    build_liby()
    for bs in (1, 8):
        run(f"every_switch_y_gemm_batch{bs}", EVERY + "Y_EXACT_GEMM=y " + TP.replace("--batch 32", f"--batch {bs}"))
elif PART == "t":
    build_liby()
    run("test_int8_mm", TESTS)
    run("gemm_t4", "{sys.executable} " + D + "/gemm_t4.py")
elif PART == "b2":
    build_liby()
    run("gemm_t4_round2", "{sys.executable} " + D + "/gemm_t4.py round2")
elif PART == "e":
    build_liby()
    run("every_switch_int_mm", EVERY + TP)
    run("every_switch_y_gemm", EVERY + "Y_EXACT_GEMM=y " + TP)
    run("default_y_gemm", "Y_EXACT_GEMM=y " + TP)
else:
    raise SystemExit(f"unknown part {PART!r}")

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
