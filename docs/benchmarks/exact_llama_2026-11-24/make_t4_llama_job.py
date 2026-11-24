"""Writes a Kaggle T4 job for exact Llama: docs/benchmarks/exact_llama_2026-11-24/.

    python3 make_t4_llama_job.py OUT_DIR SLUG PART

PART a: liby.so built here (Y's Turing int8 GEMM), then
        - llama_check.py on SmolLM2-135M and Llama 3.2 1B: greedy tokens on
          this machine's CPU and on the T4, alone and in a batch (saved, to
          compare with the RTX 4070 Ti SUPER and the Ryzen 9 9950X);
        - 3 GRPO steps of Llama 3.2 1B as the local record ran them
          (compiled, the defaults), and again with the graphed sampler and
          every inference switch on Y's GEMM;
        - the same 3 steps eagerly on this machine's CPU (Intel);
        - the exact training tests on the T4.
PART t: liby.so built here, then the exact tests on the T4 (after the move fix).
PART b: liby.so built here, then the worked-answer GRPO run of Llama 3.2 1B
        (150 steps, held-out 600 every 25), the 0.5B run's settings, with a
        wall-clock limit; the record is written at every checkpoint, so a
        stopped run keeps its steps. Out of memory: again with 512-token pieces
        and 32 draw rows.
Y's source comes from the y-lang-src dataset, liby.so from y-lang-wheel (then
rebuilt here). Models pinned to the revisions the local records used.
"""
import json
import os
import sys

out, SLUG, PART = sys.argv[1], sys.argv[2], sys.argv[3]
LLAMA_REV = "5a8abab4a5d6f164389b1079fb721cfab8d7126c"
SMOL_REV = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
cells = []


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n")})


code("""
!nvidia-smi --query-gpu=index,name,compute_cap,driver_version,clocks.max.sm,memory.total --format=csv
!lscpu | grep -E 'Model name|^CPU\\(s\\)' | cut -c1-200
!free -g | head -2
""")
code("""
import glob, os, subprocess, shutil, sys, time, hashlib
START = time.time()
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
# The source must be the version with Llama and the T4 guard (a dataset
# version still being built would attach the old one).
assert '"llama"' in open(f'{{REPO}}/python/y_lang/exact/__init__.py').read(), 'no Llama support in the source'
assert 'get_device_capability(device) >= (8, 0)' in open(f'{{REPO}}/python/y_lang/exact/qmm.py').read(), 'qmm lacks the sm_80 guard'
assert 'get_device_capability() >= (8, 0)' in open(f'{{REPO}}/python/tests/test_exact_qmm.py').read(), 'qmm tests not skipped on sm_75'
assert 'peak_gb' in open(f'{{REPO}}/tools/exact_lora_grpo.py').read(), 'no peak memory in the record'
assert 'def save_partial' in open(f'{{REPO}}/tools/exact_lora_grpo.py').read(), 'exact_lora_grpo lacks partial records'
assert 'sys.argv[3]' in open(f'{{REPO}}/docs/benchmarks/exact_llama_2026-11-24/llama_check.py').read(), 'llama_check cannot save'
empty = [p for p in glob.glob(f'{{REPO}}/**/*.ysu', recursive=True) + glob.glob(f'{{REPO}}/**/*.py', recursive=True)
         if os.path.getsize(p) == 0 and not p.endswith('__init__.py')]
print('empty files:', empty[:10])
from huggingface_hub import snapshot_download
LLAMA = snapshot_download('unsloth/Llama-3.2-1B-Instruct', revision='{LLAMA_REV}', local_dir='/tmp/models/llama1b')
SMOL = snapshot_download('HuggingFaceTB/SmolLM2-135M-Instruct', revision='{SMOL_REV}', local_dir='/tmp/models/smol')
import torch, transformers, triton
print('torch', torch.__version__, 'transformers', transformers.__version__, 'triton', triton.__version__,
      torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
""")


def run(name, cmd, timeout=None):
    """Runs cmd in REPO/tools; its output to OUT/name.log. `timeout` (s), or
    None: until the job's own deadline (DEADLINE)."""
    limit = str(timeout) if timeout is not None else "max(60, int(DEADLINE - time.time()))"
    code(f"""
t0 = time.time()
limit = int({limit})
# coreutils timeout signals its whole process group: bash and the python under it.
p = subprocess.run(['timeout', '-k', '60', str(limit), 'bash', '-c', f'cd {{REPO}}/tools && {cmd}'],
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
text, rc = p.stdout, ('timeout' if p.returncode == 124 else p.returncode)
open(f'{{OUT}}/{name}.log', 'w').write(text)
print(text[-3000:]); print('{name} exit', rc, f'{{time.time() - t0:.0f}} s')
!nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,memory.used --format=csv,noheader
""")


def build_liby():
    """liby.so and ysu_gpu_probe built here from the dataset's sources (Rust
    1.97.1), over the installed wheel's, which predates the Turing kernel."""
    code(r"""
!curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal -q --default-toolchain 1.97.1
os.environ['PATH'] = os.path.expanduser('~/.cargo/bin') + ':' + os.environ['PATH']
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


PY = "{sys.executable}"
CHECK = PY + " {REPO}/docs/benchmarks/exact_llama_2026-11-24/llama_check.py"
GRPO = PY + " exact_lora_grpo.py"
FAST = "Y_EXACT_GEMM=y Y_EXACT_KV_PACK=inplace Y_EXACT_ATTN=y Y_EXACT_V_INT8=1 "
# The local record's settings (rtx4070tis_and_cpu/llama_gpu.json), with the model as a pinned local path.
THREE = (" --model {LLAMA} --prompts 4 --group 4 --new 6 --steps 3 --eval 0 --checkpoint 3 --log 1")
if PART == "a":
    code("DEADLINE = START + 11.3 * 3600")
    build_liby()
    run("llama_check_smollm2", CHECK + " {SMOL} 24 {OUT}/t4_smollm2.json", 3600)
    run("llama_check_llama1b", CHECK + " {LLAMA} 24 {OUT}/t4_llama1b.json", 7200)
    run("grpo3_t4", GRPO + " --device cuda --compile" + THREE + " --record {OUT}/t4_grpo3.json", 3600)
    run("grpo3_t4_fast", FAST + GRPO + " --device cuda --compile --graphed --sample-rows 256" + THREE
        + " --record {OUT}/t4_grpo3_fast.json", 3600)
    run("grpo3_cpu", GRPO + " --device cpu" + THREE + " --record {OUT}/kaggle_cpu_grpo3.json", 4 * 3600)
    run("exact_tests_t4", "cd {REPO} && PYTHONPATH={REPO}/python:{REPO}/python/tests " + PY
        + ' -m unittest discover -v -s python/tests -p "test_exact_*.py"', None)
elif PART == "t":
    # The exact tests again, after ExactLinear._apply (726cc1b) and with proofs/ in the source.
    code("DEADLINE = START + 3 * 3600")
    code("""
assert 'def _apply' in open(f'{REPO}/python/y_lang/exact/model.py').read(), 'model.py lacks the move fix'
assert os.path.exists(f'{REPO}/proofs/ExactDecodeConstants.v'), 'no proofs in the source'
""")
    build_liby()
    run("exact_tests_t4", "cd {REPO} && PYTHONPATH={REPO}/python:{REPO}/python/tests " + PY
        + ' -m unittest discover -v -s python/tests -p "test_exact_*.py"', None)
elif PART == "b":
    code("DEADLINE = START + 11.2 * 3600")
    build_liby()
    LONG = (FAST + GRPO + " --device cuda --model {LLAMA} --mode steps --new 128 --eval 600 --eval-every 25"
            " --eval-new 160 --sample-prompts 32 --early-stop --compile --graphed"
            " --steps 150 --checkpoint 10 --log 5")
    # 64 rows of the draw at a time: Llama's 128k vocabulary at 256 rows ran
    # a 16 GB card out of memory in step 2's prefill (the same bits either way).
    run("llama_steps150", LONG + " --sample-rows 64 --chunk-tokens 1024 --record {OUT}/t4_llama_steps150.json"
        " --save {OUT}/t4_llama_steps150_adapters.pt", None)
    code("""
import json
log = open(f'{OUT}/llama_steps150.log').read()
rec = f'{OUT}/t4_llama_steps150.json'
done = len(json.load(open(rec))['steps']) if os.path.exists(rec) else 0
RETRY = 'out of memory' in log.lower() and done < 50
print('steps recorded with 1024-token pieces:', done, '; again with 512:', RETRY)
""")
    code("""
if RETRY:
    t0 = time.time()
    cmd = f'cd {REPO}/tools && """ + LONG + """ --sample-rows 32 --chunk-tokens 512 --record {OUT}/t4_llama_steps150_c512.json'
    p = subprocess.run(['timeout', '-k', '60', str(max(60, int(DEADLINE - time.time()))), 'bash', '-c', cmd],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    open(f'{OUT}/llama_steps150_c512.log', 'w').write(p.stdout)
    print(p.stdout[-3000:]); print('retry exit', p.returncode, f'{time.time() - t0:.0f} s')
""")
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
