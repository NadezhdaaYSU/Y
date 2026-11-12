"""Writes y_round3.ipynb: Y's Python suites on a Kaggle T4 from a source
dataset (not GitHub), then a CPU record on Kaggle's own processor against the
Zen 5's and the T4's."""
import json
import sys

EXPECTED = ['test_carries_the_library_and_the_probe', 'test_dep_info_limits_the_sources_to_the_ones_used',
            'test_dep_info_naming_a_missing_file_falls_back_to_every_source',
            'test_installed_package_probes_into_the_user_cache',
            'test_named_library_is_the_one_carried_and_is_checked_too', 'test_refuses_a_library_older_than_a_source',
            'test_refuses_a_missing_library', 'test_refuses_a_missing_probe',
            'test_refuses_without_the_rust_std_notice', 'test_the_loaded_library_is_not_older_than_the_c_api',
            'test_tools_names_are_the_package_modules', 'test_ysu_cache_dir_holds_the_profile_too',
            'test_ysu_hw_profile_names_the_profile']

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md("""
# Y on a T4, round 3: the suites with fixed sources, and Kaggle's CPU

1. Y's two Python suites on this GPU, from the `y-lang-src` dataset (the working tree).
2. Qwen2.5-0.5B, stock and exact, on **this machine's CPU**, compared with a Zen 5 CPU's
   record and with this T4's.
""")

code("""
!nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv
!lscpu | grep -E 'Model name|^CPU\\(s\\)|Flags' | cut -c1-200
import sys, platform, os
print(sys.version.split()[0], platform.platform(), 'cpus', os.cpu_count())
""")

code("""
import glob
WHEEL = sorted(glob.glob('/kaggle/input/**/y_lang-*.whl', recursive=True))[-1]
print('wheel:', WHEEL)
!pip uninstall -y -q torchvision torchaudio
!pip install -q torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
!pip install -q transformers==5.15.0 tokenizers==0.22.2 safetensors mpmath z3-solver
!pip install -q --no-deps --force-reinstall "{WHEEL}"
""")

code("""
import subprocess, time, json, re, shutil
OUT = '/kaggle/working'
REPO = '/kaggle/working/Y'
shutil.rmtree(REPO, ignore_errors=True)
os.makedirs(REPO)
tars = sorted(glob.glob('/kaggle/input/**/y_src.tar.gz', recursive=True))
if tars:
    subprocess.run(['tar', 'xzf', tars[-1], '-C', REPO], check=True)
else:                                   # Kaggle unpacked the archive on upload
    roots = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/python/y_lang/__init__.py', recursive=True)]
    src = os.path.dirname(os.path.dirname(roots[0]))
    shutil.copytree(src, REPO, dirs_exist_ok=True, symlinks=True)
print(sorted(os.listdir(REPO)))
empty = [p for p in glob.glob(f'{REPO}/**/*.ysu', recursive=True) if os.path.getsize(p) == 0]
assert not empty, f'empty kernel sources (a symlink lost on upload?): {empty}'
print(len(glob.glob(f'{REPO}/**/*', recursive=True)), 'files')
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

md("## 1. The suites")

code("""
SUITE_ENV = {'PYTHONPATH': 'python', 'Y_LIB_PATH': LIBY}
quiet = lambda s: s.startswith(('Ran ', 'OK', 'FAILED'))
for name, d in (('suite_python_tests', 'python/tests'), ('suite_y_lang_tests', 'python/y_lang/tests')):
    stream(name, [sys.executable, '-m', 'unittest', 'discover', '-s', d, '-p', 'test_*.py', '-v'],
           REPO, SUITE_ENV, quiet)
EXPECTED = set(""" + repr(EXPECTED) + """)
for name in ('suite_python_tests', 'suite_y_lang_tests'):
    text = open(f'{OUT}/{name}.log').read()
    bad = sorted(set(re.findall(r'^(?:FAIL|ERROR): (\\S+) ', text, re.M)))
    new = [t for t in bad if t not in EXPECTED]
    skipped = re.findall(r'skipped=(\\d+)', text)
    print(f'{name}: {len(bad)} failing, {len(bad) - len(new)} expected (need a build), '
          f'{len(new)} NEW; skipped {skipped[-1] if skipped else 0}')
    for t in new:
        print('   ', t)
""")

md("## 2. This machine's CPU against a Zen 5 and a T4")

code("""
TOOLS = f'{REPO}/tools'
REF = f'{REPO}/docs/benchmarks'
noise = lambda s: (s.strip() and not s.strip().startswith('->') and 'Warning' not in s
                   and 'warnings.warn' not in s and 'it/s]' not in s)
rc_cpu = stream('kaggle_cpu', [sys.executable, 'exact_cross_device.py', '--record', f'{OUT}/kaggle_cpu.json',
                               '--device', 'cpu', '--arms', 'stock,exact', '--model', 'Qwen/Qwen2.5-0.5B-Instruct'],
                TOOLS, {'PYTHONPATH': f'{REPO}/python', 'Y_LIB_PATH': LIBY,
                        'OMP_NUM_THREADS': str(os.cpu_count())}, noise)
for ref, what in ((f'{REF}/exact_y_gemm_2026-11-07/cpu_zen5_0p5b.json', 'Zen 5 CPU, both arms'),
                  (f'{REF}/exact_norm_sum_2026-11-10/cpu_0p5b.json', 'Zen 5 CPU, integer norm sum'),
                  (f'{REF}/exact_cross_gpu_t4_2026-11-11/t4_eager.json', 'Tesla T4, both arms')):
    p = subprocess.run([sys.executable, 'exact_cross_device.py', '--compare', ref, f'{OUT}/kaggle_cpu.json'],
                       cwd=TOOLS, capture_output=True, text=True)
    print('===', what)
    print('\\n'.join(l for l in (p.stdout + p.stderr).splitlines()
                    if 'cases changed' in l or 'control' in l or '***' in l or 'exact:' in l))
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                  "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"t{i:02d}"
    c["source"] = c["source"].splitlines(keepends=True)
json.dump(nb, open(sys.argv[1], "w"), indent=1)
print("wrote", sys.argv[1], len(cells), "cells")
