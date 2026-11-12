"""Writes y_round2.ipynb: Y's Python test suites on a Kaggle T4, then the 7B
exact record against the Zen 5 CPU's."""
import json
import sys

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md("""
# Y on a T4, round 2: the test suites, then 7B

1. **Y's two Python test suites** (about 280 tests) on this GPU. Every Y kernel in them is
   compiled for this card (sm_75) and checked against a reference, most bit for bit. Some
   tests need a full build of the repository (cargo, the CLI) and will fail or skip here:
   that is expected, and the full logs say which.
2. **Qwen2.5-7B-Instruct**, exact path, eager, against the record made on a Zen 5 CPU
   (digest `7700e3e28f1d21d1`): want `0/5 cases changed`. Downloads about 15 GB.

Needs: **GPU T4 x2**, **Internet on**, the **y-lang-wheel** input. Best run with
**Save Version → Save & Run All** (2-3 hours; the tab can be closed).
""")

code("""
!nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total --format=csv
import sys, platform
print(sys.version.split()[0], platform.platform())
!free -g | head -2; df -h /kaggle/working /root | tail -2
""")

code("""
import glob
WHEELS = sorted(glob.glob('/kaggle/input/**/y_lang-*.whl', recursive=True))
assert WHEELS, 'Add Input -> y-lang-wheel'
WHEEL = WHEELS[-1]
print('wheel:', WHEEL)
!pip uninstall -y -q torchvision torchaudio
!pip install -q torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
!pip install -q transformers==5.15.0 tokenizers==0.22.2 safetensors mpmath z3-solver
!pip install -q --no-deps --force-reinstall "{WHEEL}"
""")

code("""
import os, subprocess, time, json
OUT = '/kaggle/working'
REPO = '/kaggle/working/Y'
!rm -rf {REPO} && git clone -q --depth 1 https://github.com/NadezhdaaYSU/Y.git {REPO}
!cd {REPO} && git log -1 --format='source: %h %s'
LIBY = subprocess.run([sys.executable, '-c',
                       'import y_lang, os; print(os.path.join(os.path.dirname(y_lang.__file__), "liby.so"))'],
                      capture_output=True, text=True).stdout.strip()
print('liby.so from the wheel:', LIBY, os.path.exists(LIBY))

def stream(name, cmd, cwd, env=None, show=lambda s: True):
    e = dict(os.environ, **(env or {}))
    e.pop('HF_HUB_OFFLINE', None)
    t0 = time.time()
    with open(f'{OUT}/{name}.log', 'w') as log:
        p = subprocess.Popen(cmd, cwd=cwd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            log.write(line)
            if show(line):
                print(line, end='')
        p.wait()
    print(f'--- {name}: exit {p.returncode}, {(time.time() - t0) / 60:.1f} min')
    return p.returncode
""")

md("""
## 1. Y's Python test suites on this GPU
""")

code("""
# The repository's Python sources (the same commit as the wheel), with the wheel's liby.so.
SUITE_ENV = {'PYTHONPATH': 'python', 'Y_LIB_PATH': LIBY}
quiet = lambda s: s.startswith(('FAIL:', 'ERROR:', 'Ran ', 'OK', 'FAILED'))
rc_py = stream('suite_python_tests', [sys.executable, '-m', 'unittest', 'discover', '-s', 'python/tests',
                                      '-p', 'test_*.py', '-v'], REPO, SUITE_ENV, quiet)
rc_yl = stream('suite_y_lang_tests', [sys.executable, '-m', 'unittest', 'discover', '-s', 'python/y_lang/tests',
                                      '-p', 'test_*.py', '-v'], REPO, SUITE_ENV, quiet)
""")

code("""
# Failures a clean clone shows on any machine (they need a full build: cargo, the CLI,
# a fresh liby.so beside the sources), measured on the development machine. Anything
# else is about this GPU.
import re
EXPECTED = set(['test_carries_the_library_and_the_probe', 'test_dep_info_limits_the_sources_to_the_ones_used', 'test_dep_info_naming_a_missing_file_falls_back_to_every_source', 'test_installed_package_probes_into_the_user_cache', 'test_named_library_is_the_one_carried_and_is_checked_too', 'test_refuses_a_library_older_than_a_source', 'test_refuses_a_missing_library', 'test_refuses_a_missing_probe', 'test_refuses_without_the_rust_std_notice', 'test_the_loaded_library_is_not_older_than_the_c_api', 'test_tools_names_are_the_package_modules', 'test_ysu_cache_dir_holds_the_profile_too', 'test_ysu_hw_profile_names_the_profile'])
for name in ('suite_python_tests', 'suite_y_lang_tests'):
    text = open(f'{OUT}/{name}.log').read()
    bad = sorted(set(re.findall(r'^(?:FAIL|ERROR): (\\S+) ', text, re.M)))
    new = [t for t in bad if t not in EXPECTED]
    print(f'{name}: {len(bad)} failing, {len(bad) - len(new)} expected (need a build), {len(new)} NEW:')
    for t in new:
        print('   ', t)
""")

md("""
## 2. Qwen2.5-7B-Instruct, exact, against the Zen 5 CPU
""")

code("""
TOOLS = f'{REPO}/tools'
MODEL7 = 'Qwen/Qwen2.5-7B-Instruct'
CPU7 = f'{REPO}/docs/benchmarks/exact_norm_sum_2026-11-10/cpu_zen5_eager_7b.json'
noise = lambda s: (s.strip() and not s.strip().startswith('->') and 'Warning' not in s
                   and 'warnings.warn' not in s and 'it/s]' not in s)
# The record tool from the clone, but y_lang from the wheel: hide the clone's sources.
!mv {REPO}/python {REPO}/python_sources
rc_7b = stream('t4_7b', [sys.executable, 'exact_cross_device.py', '--record', f'{OUT}/t4_7b.json',
                         '--device', 'cuda', '--arms', 'exact', '--model', MODEL7, '--low-memory'],
               TOOLS, show=noise)
p = subprocess.run([sys.executable, 'exact_cross_device.py', '--compare', CPU7, f'{OUT}/t4_7b.json'],
                   cwd=TOOLS, capture_output=True, text=True)
print(p.stdout[-3000:], p.stderr[-1500:])
""")

md("""
## Summary
Download everything in `/kaggle/working` (the `.log` and `.json` files) from the Output panel.
""")

code("""
for name in ('suite_python_tests', 'suite_y_lang_tests'):
    lines = open(f'{OUT}/{name}.log').read().splitlines()
    print(name, [l for l in lines if l.startswith(('Ran ', 'OK', 'FAILED'))][-2:])
try:
    d = json.load(open(f'{OUT}/t4_7b.json'))
    print('7B T4 exact digest', d['arms']['exact']['digest'], '| CPU', json.load(open(CPU7))['arms']['exact']['digest'])
except FileNotFoundError:
    print('7B: no record (see t4_7b.log)')
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                  "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"r{i:02d}"
    c["source"] = c["source"].splitlines(keepends=True)
json.dump(nb, open(sys.argv[1], "w"), indent=1)
print("wrote", sys.argv[1], len(cells), "cells")
