"""Run notebook 02's analysis cells on a synthetic session built like the real file.

The synthetic session has 5 image categories with 10 old and 10 new recognition images each, stores the label as
1 = old (as the real files do), and plants a memory signal in 6 of 15 neurons. Checks: the label check passes, the
decoder finds the signal, the within-category null sits near chance, and the category-only model falls below chance
(the fixed-margin effect explained in the notebook). Usage: python tests/test_02_synthetic.py"""
import io, json, contextlib, sys, tempfile, warnings
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from pynwb import NWBFile, NWBHDF5IO

ROOT = Path(__file__).resolve().parents[1]
NB02 = ROOT / 'notebooks' / '02_Human_Recognition_Neural_Decoding.ipynb'
TMP = Path(tempfile.mkdtemp(prefix='nb02_synthetic_'))
rng = np.random.default_rng(3)

nwb = NWBFile(session_description='synthetic', identifier='synthetic_02', session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc))
for name in ['stim_on_time', 'stim_phase', 'category_name', 'external_image_file', 'new_old_labels_recog', 'response_value', 'response_time']:
    nwb.add_trial_column(name, name)
categories = ['houses', 'animals', 'phones', 'landscapes', 'cars']
learn_cat = np.repeat(categories, 20)
learn_files = [f'newolddelay\\{c}\\{i}.jpg' for i, c in enumerate(learn_cat)]
recognition = []
for c in categories:                                   # 10 old + 10 new per category
    old = [f for f in learn_files if f'\\{c}\\' in f][:10]
    recognition += [(c, f, '1') for f in old] + [(c, f'newolddelay\\{c}\\new{j}.jpg', '0') for j in range(10)]
recognition = [recognition[i] for i in rng.permutation(100)]
rows = [(learn_cat[i], learn_files[i], 'NA', 'learn') for i in range(100)] + [(c, f, lab, 'recog') for c, f, lab in recognition]
onsets = 10 + 4.0 * np.arange(200)
for i, (c, f, lab, phase) in enumerate(rows):
    nwb.add_trial(start_time=onsets[i], stop_time=onsets[i] + 3.5, stim_on_time=onsets[i], stim_phase=phase, category_name=c,
                  external_image_file=f, new_old_labels_recog=lab, response_value=np.nan, response_time=onsets[i] + 2.5)
for u in range(15):
    spikes = list(rng.uniform(0, onsets[-1] + 5, rng.poisson(4 * (onsets[-1] + 5))))
    if u < 6:                                          # memory signal: extra spikes for old images
        for i in range(100, 200):
            if rows[i][2] == '1':
                spikes += list(rng.uniform(onsets[i] + .1, onsets[i] + .8, rng.poisson(2.5)))
    nwb.add_unit(spike_times=np.sort(np.array(spikes)))
with NWBHDF5IO(str(TMP / 'synthetic_02.nwb'), 'w') as w:
    w.write(nwb)

nb = json.loads(NB02.read_text(encoding='utf-8'))
code = {i: ''.join(c['source']) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code'}
config = (code[2]
          .replace("DATA=Path('/content/neurotech_data') if Path('/content').exists() else Path('neurotech_data')", f"DATA=Path(r'{TMP}')")
          .replace("OUT=Path('/content/neurotech_results/02_human_recognition') if Path('/content').exists() else Path('neurotech_results/02_human_recognition')", f"OUT=Path(r'{TMP}/out')")
          .replace("FILE=DATA/'sub-P11HMH_ses-20061101_ecephys+image.nwb'", "FILE=DATA/'synthetic_02.nwb'"))
assert 'synthetic_02.nwb' in config, 'notebook 02 configuration cell changed; update this test'
ns = {'display': lambda *a, **k: None}
log = io.StringIO()
with contextlib.redirect_stdout(log), warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for i in (2, 4, 6, 7, 8, 10, 11, 12, 13, 16):     # skips pip install (1) and the download (3)
        exec(config if i == 2 else code[i], ns)
out = log.getvalue()
scores = ns['scores'].set_index('model')
null = ns['null_summary']
print(scores[['balanced_accuracy', 'roc_auc']].round(3).to_string())
print('within-category null:', null)
checks = {
    'label check passed': 'Label check' in out,
    'decoder finds planted signal (AUC > .8)': scores.loc['post_stimulus_neural', 'roc_auc'] > .8,
    'null AUC near chance': abs(null['roc_auc']['null_mean'] - .5) < .05,
    'signal beats the null (p < .05)': null['roc_auc']['p_one_sided'] < .05,
    'category-only below chance (fixed-margin effect)': scores.loc['category_only', 'roc_auc'] < .5,
    'settings record the label coding': json.loads((Path(ns['OUT']) / 'settings.json').read_text())['y_coding'] == {'0': 'old', '1': 'new'},
}
for name, ok in checks.items():
    print(('PASS ' if ok else 'FAIL ') + name)
sys.exit(0 if all(checks.values()) else 1)
