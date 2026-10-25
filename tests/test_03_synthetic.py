"""Run notebook 03's analysis cells on synthetic NWB files with a known answer.

Two scenarios, 8 synthetic patients each (one with two sessions, one with half its learning phase missing):
neurons that follow the patient's report, and neurons that follow the true status. Usage: python tests/test_03_synthetic.py"""
import json, sys, io, contextlib, warnings, tempfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import nbformat
from pynwb import NWBFile, NWBHDF5IO

ROOT = Path(__file__).resolve().parents[1]
NB03 = ROOT / 'notebooks' / '03_Recognition_Across_Sessions.ipynb'
TMP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix='nb03_synthetic_'))

def make_session(path, follows, seed, n_units=20, p_correct=0.72, learn_in_file=100):
    rng = np.random.default_rng(seed)
    nwb = NWBFile(session_description='synthetic', identifier=path.stem, session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc))
    for name in ['stim_on_time', 'stim_phase', 'category_name', 'external_image_file', 'new_old_labels_recog', 'response_value', 'response_time']:
        nwb.add_trial_column(name, name)
    onsets = 10 + 4.0 * np.arange(200)
    truth = np.r_[np.full(100, -1), rng.permutation(np.r_[np.zeros(50), np.ones(50)])].astype(int)
    report = np.where(rng.random(200) < p_correct, truth, 1 - truth)
    signal = report if follows == 'report' else truth
    # Mimic the real files: old recognition images reuse learning-phase files; stored label 1 = OLD
    learn_files = [f'newolddelay\\set\\{i}.jpg' for i in range(100)]
    old_pool, new_count = iter(rng.permutation(learn_files)), iter(range(1000))
    files = learn_files + [next(old_pool) if truth[i] == 0 else f'newolddelay\\set\\new{next(new_count)}.jpg' for i in range(100, 200)]
    for i, t0 in enumerate(onsets):
        recog = i >= 100
        if not recog and i >= learn_in_file:      # mimic files that lack part of the learning phase
            continue
        code = (rng.choice([31, 32, 33]) if report[i] == 1 else rng.choice([34, 35, 36])) if recog else np.nan
        nwb.add_trial(start_time=t0, stop_time=t0 + 3.5, stim_on_time=t0, stim_phase='recog' if recog else 'learn',
                      category_name=str(rng.choice(['houses', 'animals'])),
                      external_image_file=files[i],
                      new_old_labels_recog=str(1 - truth[i]) if recog else 'NA',
                      response_value=float(code), response_time=t0 + 1.8 + rng.random())
    duration = onsets[-1] + 5
    for u in range(n_units):
        st = list(np.sort(rng.uniform(0, duration, rng.poisson(4 * duration))))
        if u < 8:
            for i in range(100, 200):
                if signal[i] == 1:
                    st += list(rng.uniform(onsets[i] + .1, onsets[i] + .8, rng.poisson(7)))
        nwb.add_unit(spike_times=np.sort(np.array(st)))
    with NWBHDF5IO(str(path), 'w') as w:
        w.write(nwb)

nb = nbformat.read(NB03, as_version=4)
code = [c.source for c in nb.cells if c.cell_type == 'code']
cfg, read, models, run, sanity, statcell, sens, cat = code[1], code[3], code[4], code[5], code[6], code[7], code[8], code[9]
results = {}
for follows in ['report', 'truth']:
    d = TMP / follows; d.mkdir(parents=True, exist_ok=True)
    sources = []
    for p in range(8):
        for s in range(2 if p == 0 else 1):        # patient 0 has two sessions (tests averaging)
            path = d / f'sub-P{p:02d}_ses-{s}.nwb'
            make_session(path, follows, seed=100 * p + s + (0 if follows == 'report' else 7),
                         learn_in_file=50 if p == 3 else 100)   # patient 3: half the learning phase missing
            sources.append(dict(path=f'sub-P{p:02d}/{path.name}', asset_id=f'{follows}-{p}-{s}', url=str(path)))
    ns = {'display': lambda *a, **k: None}
    exec(cfg.replace("BASE = Path('/content') if Path('/content').exists() else Path('.')", f"BASE = Path(r'{d}')"), ns)
    ns['sources'] = sources
    out = io.StringIO()
    with contextlib.redirect_stdout(out), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for cell in (read, models, run, sanity, statcell, sens, cat):
            exec(cell, ns)
    results[follows] = (ns['q1'], ns['q2'], len(ns['sessions']), ns['sessions'].patient.nunique(),
                        bool(ns['sessions'].set_index('patient').stored_label_marks_old.lt(1).to_dict()
                             == {f'P{p:02d}': p == 3 for p in range(8)}
                             and (ns['sessions'].learning_first == True).all()
                             and ns['sensitivity'].shape[0] == 4
                             and abs(ns['category_control'].loc['Q2 excess over the null', 'mean']) > .2
                             and ns['category_control'].loc['Q1 excess over the null', 'mean'] > .1))
    print(f"\n=== planted signal follows the {follows.upper()} ===")
    print(ns['sensitivity'].round(3).to_string())
    print(ns['category_control'].to_string())
    print(f"sessions {results[follows][2]}, patients {results[follows][3]}; behaviour accuracy median {ns['sessions'].behaviour_accuracy.median():.2f}")
    print(f"Q1 AUC post {ns['q1']['auc_post_mean']} (p={ns['q1']['p_post_vs_chance']:.3g}), pre {ns['q1']['auc_pre_mean']} (p={ns['q1']['p_pre_vs_chance']:.3g})")
    for w in ('post', 'pre'):
        q = ns['q2'][w]; print(f"Q2 {w}: agreement with report {q['agreement_mean']} CI {q['agreement_ci']} p={q['p_vs_half']:.3g} ({q['n_patients']} patients, {q['n_error_trials']} error trials)")

ok = (results['report'][1]['post']['agreement_mean'] > .7 and results['truth'][1]['post']['agreement_mean'] < .3
      and abs(results['report'][1]['pre']['agreement_mean'] - .5) < .15 and abs(results['truth'][1]['pre']['agreement_mean'] - .5) < .15
      and results['report'][0]['auc_post_mean'] > .6 and results['truth'][0]['auc_post_mean'] > .6
      and abs(results['report'][0]['auc_pre_mean'] - .5) < .1 and results['report'][3] == 8
      and results['report'][4] and results['truth'][4])
print('\nSYNTHETIC CHECK:', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
