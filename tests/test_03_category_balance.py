"""Category-balance check for notebook 03's Q1 null.

60 simulated patients see a stimulus set with old and new balanced within each of 5 categories; neurons are tuned to
category and carry no memory signal. Chronological-fold AUC is then biased below 0.5; the excess over the
within-category null should not be. Usage: python tests/test_03_category_balance.py"""
import sys, tempfile
from pathlib import Path
import numpy as np, warnings, nbformat
from scipy import stats
warnings.simplefilter('ignore')
ROOT = Path(__file__).resolve().parents[1]
NB03 = ROOT / 'notebooks' / '03_Recognition_Across_Sessions.ipynb'
nb = nbformat.read(NB03, as_version=4)
code = [c.source for c in nb.cells if c.cell_type == 'code']
ns = {}
TMP = Path(tempfile.mkdtemp(prefix='nb03_balance_'))
exec(code[1].replace("BASE = Path('/content') if Path('/content').exists() else Path('.')", f"BASE = Path(r'{TMP}')"), ns)
exec(code[4], ns)
cat_cell = code[9]
exec(cat_cell[cat_cell.index('def shuffle_within_category'):cat_cell.index('rng = np.random.default_rng(SEED)')], ns)
rng = np.random.default_rng(1)
obs, nul = [], []
for s in range(60):   # 60 simulated patients: category-balanced stimulus set, category-tuned neurons, no memory signal
    cats = np.repeat(np.arange(5), 20)
    y = np.concatenate([rng.permutation(np.r_[np.zeros(10), np.ones(10)]) for _ in range(5)])
    order = rng.permutation(100); cats, y = cats[order], y[order].astype(int)
    rates = np.clip(3 + 1.5 * rng.normal(0, 1, (5, 20)), .1, None)
    X = rng.poisson(rates[cats]).astype(float)
    obs.append(ns['chrono_auc'](X, y))
    nul.append(np.nanmean([ns['chrono_auc'](X[ns['shuffle_within_category'](cats.astype(str), rng)], y) for _ in range(50)]))
obs, nul = np.array(obs), np.array(nul); ex = obs - nul
print(f'observed AUC {obs.mean():.3f} (p vs .5 {stats.wilcoxon(obs - .5).pvalue:.3g}); null {nul.mean():.3f}; excess {ex.mean():+.3f} (p {stats.wilcoxon(ex).pvalue:.3g})')
ok = obs.mean() < .49 and stats.wilcoxon(ex).pvalue > .05
print('CATEGORY BALANCE CHECK:', 'PASS (raw AUC biased below chance, null-corrected AUC not)' if ok else 'FAIL')
sys.exit(0 if ok else 1)
