"""Writes y_cross_gpu.ipynb: Y's exact path on a GPU it has never run on (a
Kaggle T4), compared bit for bit with the records made on a Zen 5 CPU and an
RTX 4070 Ti SUPER."""
import json
import sys

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md("""
# Y's exact inference on a GPU it has never seen

Qwen2.5-0.5B-Instruct decodes 5 batches (160 tokens each) on this GPU. The tokens are
compared, bit for bit, with records made on a **Zen 5 CPU** and an **RTX 4070 Ti SUPER**:

1. **Eager, stock and exact.** The stock model is expected to *differ* from the 4070 Ti
   (the control: the comparison can see a difference). Y's exact path must give the
   CPU's tokens: `0/5 cases changed`.
2. **Compiled, with every Y switch this GPU can run** (Y's attention kernel, the Triton
   norm, the in-place cache). Again `0/5`.
3. **Y's own int8 GEMM** needs sm_80 tensor-core instructions. On an older card it must be
   *refused* by name, not run wrongly.

Needs: **Accelerator GPU T4 x2**, **Internet on**, and the y-lang wheel attached as an
input (Add Input → your dataset). Takes about 40-60 minutes; just press **Run All**.
Everything is logged to `/kaggle/working/` (Output panel on the right).
""")

code("""
# The GPU and the Python this runs on.
!nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total --format=csv
import sys, platform
print(sys.version.split()[0], platform.platform())
""")

code("""
# Install the same torch and transformers the reference records were made with
# (torch 2.13.0 for CUDA 12.6, which still supports a T4), then Y from the wheel.
import glob
WHEELS = sorted(glob.glob('/kaggle/input/**/y_lang-*.whl', recursive=True))
assert WHEELS, ('No y-lang wheel found. Right panel -> Add Input -> your dataset that '
                'holds y_lang-1.1.0-py3-none-linux_x86_64.whl')
WHEEL = WHEELS[-1]
print('wheel:', WHEEL)
!pip uninstall -y -q torchvision torchaudio
!pip install -q torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
!pip install -q transformers==5.15.0 tokenizers==0.22.2 safetensors mpmath z3-solver
!pip install -q --no-deps --force-reinstall "{WHEEL}"
""")

code("""
# Check the install. If torch is not 2.13.0 here: Run -> Restart & clear outputs, then
# run this cell and the ones below again (skip the install cell).
import subprocess
out = subprocess.run([sys.executable, '-c',
    'import torch, transformers, y_lang; '
    'print("torch", torch.__version__, "cuda", torch.version.cuda); '
    'print("transformers", transformers.__version__); '
    'print("y_lang", y_lang.__version__, y_lang.__file__); '
    'print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'],
    capture_output=True, text=True)
print(out.stdout, out.stderr[-2000:])
# Which int8 GEMM the exact path will use on this card.
probe = '''
import torch
from y_lang.exact import model as em
print("Triton int8 GEMM usable here:", em.triton_int8_dot_ok("cuda"), "(default GEMM:", em.GEMM + ")")
a = torch.randint(-127, 128, (32, 64), dtype=torch.int8, device="cuda")
b = torch.randint(-127, 128, (64, 64), dtype=torch.int8, device="cuda")
try:
    r = torch._int_mm(a, b)
    exact = torch.equal(r.double(), a.double() @ b.double())
    print("torch._int_mm works here, exact:", exact)
except Exception as e:
    print("torch._int_mm refused here, so float64 is used:", str(e)[:200])
'''
out = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True)
print(out.stdout, out.stderr[-2000:])
""")

code("""
# Y's record tool and the reference records, from GitHub. The clone's python/ is removed
# so the tool imports y_lang from the wheel, not from the sources.
!rm -rf /kaggle/working/Y && git clone -q --depth 1 https://github.com/NadezhdaaYSU/Y.git /kaggle/working/Y
!rm -rf /kaggle/working/Y/python
!cd /kaggle/working/Y && git log -1 --format='source: %h %s'

import os, subprocess, time
TOOLS = '/kaggle/working/Y/tools'
REF = '/kaggle/working/Y/docs/benchmarks'
OUT = '/kaggle/working'
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
CPU_REC = f'{REF}/exact_norm_sum_2026-11-10/cpu_0p5b.json'                # Zen 5, exact arm
SM89_REC = f'{REF}/exact_y_gemm_2026-11-07/gpu_sm89_triton_eager.json'    # 4070 Ti, both arms

def run(name, args, env=None):
    \"\"\"Run the record tool, show its progress (without the hardware probe's lines and
    warnings), keep the whole output in /kaggle/working/<name>.log; return the exit code.\"\"\"
    e = dict(os.environ, **(env or {}))
    e.pop('HF_HUB_OFFLINE', None)
    t0 = time.time()
    with open(f'{OUT}/{name}.log', 'w') as log:
        p = subprocess.Popen([sys.executable, 'exact_cross_device.py'] + args, cwd=TOOLS, env=e,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            log.write(line)
            s = line.strip()
            if s and not s.startswith('->') and 'Warning' not in s and 'warnings.warn' not in s:
                print(line, end='')
        p.wait()
    print(f'--- {name}: exit {p.returncode}, {(time.time() - t0) / 60:.1f} min')
    return p.returncode

def compare(a, b):
    p = subprocess.run([sys.executable, 'exact_cross_device.py', '--compare', a, b],
                       cwd=TOOLS, capture_output=True, text=True)
    print(p.stdout[-3000:], p.stderr[-1500:])
""")

md("""
## 1. Eager, stock and exact
""")

code("""
rc_eager = run('t4_eager', ['--record', f'{OUT}/t4_eager.json', '--device', 'cuda',
                            '--arms', 'stock,exact', '--model', MODEL])
""")

code("""
print('=== exact: this GPU against the Zen 5 CPU (want 0/5 changed) ===')
compare(CPU_REC, f'{OUT}/t4_eager.json')
print('=== stock and exact: this GPU against the RTX 4070 Ti SUPER ===')
print('    (stock is expected to change: that is the control; exact must not)')
compare(SM89_REC, f'{OUT}/t4_eager.json')
""")

md("""
## 2. Compiled, every Y switch a T4 can run
""")

code("""
rc_all = run('t4_all_compiled', ['--record', f'{OUT}/t4_all_compiled.json', '--device', 'cuda',
                                 '--arms', 'exact', '--compile', '--model', MODEL],
             env={'Y_EXACT_V_INT8': '1', 'Y_EXACT_NORM': 'triton', 'Y_EXACT_ATTN': 'y',
                  'Y_EXACT_KV_PACK': 'inplace'})
print('=== this GPU, compiled, against the Zen 5 CPU (want 0/5 changed) ===')
compare(CPU_REC, f'{OUT}/t4_all_compiled.json')
""")

md("""
## 3. Y's own GEMM: must be refused below sm_80
""")

code("""
rc_ygemm = run('t4_ygemm', ['--record', f'{OUT}/t4_ygemm.json', '--device', 'cuda', '--arms', 'exact',
                            '--model', MODEL, '--cases', 'alone (batch 1)', '--tokens', '8'],
               env={'Y_EXACT_GEMM': 'y'})
log = open(f'{OUT}/t4_ygemm.log').read()
print('refused by name:', rc_ygemm != 0 and 'sm_80' in log)
""")

md("""
## Summary
Download `t4_eager.json`, `t4_all_compiled.json` and the `.log` files from the Output panel
(right side, `/kaggle/working`) and bring them back.
""")

code("""
import json
for name in ('t4_eager', 't4_all_compiled'):
    try:
        d = json.load(open(f'{OUT}/{name}.json'))
        for arm, a in d['arms'].items():
            print(f"{name:16s} {arm:6s} digest {a['digest']}  self-check {a.get('self_check')}")
    except FileNotFoundError:
        print(name, 'no record (see its .log)')
print('CPU (Zen 5) exact digest:', json.load(open(CPU_REC))['arms']['exact']['digest'])
print('exit codes:', dict(eager=rc_eager, compiled=rc_all, y_gemm_refused=rc_ygemm))
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                  "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"c{i:02d}"
    c["source"] = c["source"].splitlines(keepends=True)
json.dump(nb, open(sys.argv[1], "w"), indent=1)
print("wrote", sys.argv[1], len(cells), "cells")
