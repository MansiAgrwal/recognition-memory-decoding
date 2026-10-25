"""Planted-confound check for notebook 03's category control.

Simulated patients call one image category "old" more often, and some neurons respond to that category, but no neuron
carries any memory or decision signal. The primary Q2 metric should be fooled (agreement > 0.5); the excess over the
within-category null should not. Usage: python tests/test_03_category_confound.py [n_patients=40]"""
import sys, io, contextlib, warnings, tempfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, nbformat
from pynwb import NWBFile, NWBHDF5IO
ROOT = Path(__file__).resolve().parents[1]
NB03 = ROOT / 'notebooks' / '03_Recognition_Across_Sessions.ipynb'
TMP = Path(tempfile.mkdtemp(prefix='nb03_confound_'))
def make(path, seed):
    rng = np.random.default_rng(seed)
    nwb = NWBFile(session_description='s', identifier=path.stem, session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc))
    for n in ['stim_on_time', 'stim_phase', 'category_name', 'external_image_file', 'new_old_labels_recog', 'response_value', 'response_time']:
        nwb.add_trial_column(n, n)
    on = 10 + 4.0 * np.arange(200)
    truth = np.r_[np.full(100, -1), rng.permutation(np.r_[np.zeros(50), np.ones(50)])].astype(int)
    cat = rng.choice(['houses', 'animals'], 200)
    # report: memory-based with accuracy .72, but 'houses' pushed toward "old" (report_new=0)
    report = np.where(rng.random(200) < .72, truth, 1 - truth)
    report = np.where((cat == 'houses') & (rng.random(200) < .35), 0, report)
    files = [f'L{i}.jpg' for i in range(100)]; pool = iter(rng.permutation(files)); k = iter(range(999))
    files += [next(pool) if truth[i] == 0 else f'N{next(k)}.jpg' for i in range(100, 200)]
    for i, t0 in enumerate(on):
        rec = i >= 100
        code = (rng.choice([31, 32, 33]) if report[i] == 1 else rng.choice([34, 35, 36])) if rec else np.nan
        nwb.add_trial(start_time=t0, stop_time=t0 + 3.5, stim_on_time=t0, stim_phase='recog' if rec else 'learn', category_name=str(cat[i]),
                      external_image_file=files[i], new_old_labels_recog=str(1 - truth[i]) if rec else 'NA',
                      response_value=float(code), response_time=t0 + 1.8 + rng.random())
    for u in range(20):
        st = list(rng.uniform(0, on[-1] + 5, rng.poisson(4 * (on[-1] + 5))))
        if u < 8:   # category-selective neurons, no memory signal
            for i in range(100, 200):
                if cat[i] == 'houses':
                    st += list(rng.uniform(on[i] + .1, on[i] + .8, rng.poisson(6)))
        nwb.add_unit(spike_times=np.sort(np.array(st)))
    with NWBHDF5IO(str(path), 'w') as w: w.write(nwb)
nb = nbformat.read(NB03, as_version=4)
code = [c.source for c in nb.cells if c.cell_type == 'code']
sources = []
for p in range(int(sys.argv[1]) if len(sys.argv) > 1 else 40):
    path = TMP / f'sub-C{p:02d}_ses-0.nwb'; make(path, 900 + p)
    sources.append(dict(path=f'sub-C{p:02d}/{path.name}', asset_id=f'c-{p}', url=str(path)))
ns = {'display': lambda *a, **k: None}
exec(code[1].replace("BASE = Path('/content') if Path('/content').exists() else Path('.')", f"BASE = Path(r'{TMP}')"), ns)
ns['sources'] = sources
with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for c in (code[3], code[4], code[5], code[6], code[7], code[8], code[9]): exec(c, ns)
cc = ns['category_control']; print(cc.to_string())
ok = (cc.loc['Q2 agreement, observed', 'p'] < .05 and cc.loc['Q2 agreement, observed', 'mean'] > .5
      and cc.loc['Q2 excess over the null', 'p'] > .05)
print('CATEGORY CONFOUND CHECK:', 'PASS (primary metric fooled, null-corrected metric not)' if ok else 'FAIL')
sys.exit(0 if ok else 1)
