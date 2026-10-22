"""End-to-end run on fully synthetic data.

Creates a working folder, writes synthetic inputs (make_synthetic_data.py), and runs the same pipeline that produced
the paper: price preprocessing -> scenarios, duals and robustness -> rolling-horizon tests -> Tables 3 and 5.
The confidential extraction step (00c_extract_purchases.py) is replaced by the synthetic purchase file.

Usage:  python run_demo.py [work_folder] [--figures] [--bootstrap]
Runtime: a few minutes on a laptop (the rolling-horizon ARIMA fits take most of it).
"""
import os, sys, shutil, subprocess, glob, time

HERE = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not a.startswith('--')]
WORK = os.path.abspath(args[0] if args else os.path.join(HERE, '..', 'demo_run'))
os.makedirs(WORK, exist_ok=True)
for f in glob.glob(os.path.join(HERE, '*.py')):
    if os.path.basename(f) not in ('run_demo.py', 'company_config_TEMPLATE.py'):
        shutil.copy(f, WORK)
sys.path.insert(0, HERE)
from make_synthetic_data import generate
generate(WORK)

steps = ['prep_prices_v3.py', 'run_main3.py', 'run_rolling3.py']
if '--bootstrap' in sys.argv: steps.append('run_bootstrap3.py')
steps.append('make_tables.py')
if '--figures' in sys.argv: steps.append('figures3.py')
t0 = time.time()
for s in steps:
    print(f'--- {s}', flush=True)
    r = subprocess.run([sys.executable, s], cwd=WORK, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-4000:]); sys.exit(f'{s} failed')
    if s == 'make_tables.py': print(r.stdout)
print(f'done in {time.time()-t0:.0f} s; outputs in {WORK}')
