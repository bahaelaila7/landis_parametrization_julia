"""Checks of the nested-CV folds and role files (docs/nested_cv_gafl.md §5.1). Exit 1 on any FAIL.
  python -I tools/ncv/verify_folds.py <folds dir> [k]"""
import sys, numpy as np, pandas as pd
d = sys.argv[1]; K = int(sys.argv[2]) if len(sys.argv) > 2 else 5
ids = ['statecd', 'unitcd', 'countycd', 'plot']
fails = []
def check(name, ok, msg=''):
    print(f"{'PASS' if ok else 'FAIL'} {name} {msg}"); ok or fails.append(name)
F = pd.read_csv(f'{d}/folds.csv')
check('F1 one row per plot', not F.duplicated(ids).any(), f'{len(F)} plots')
check('F2 folds 1..K', set(F.fold) == set(range(1, K + 1)))
check('F3 blocks whole', (F.groupby('block').fold.nunique() == 1).all(), f'{F.block.nunique()} blocks')
n = F.fold.value_counts().sort_index(); check('F4 fold sizes within 10 % of the mean', (abs(n / n.mean() - 1) < 0.10).all(), str(n.tolist()))
sh = pd.crosstab(F.l3, F.fold, normalize='index'); big = F.l3.value_counts(); big = big[big >= 200].index
check('F5 L3 share per fold in [0.10, 0.30] (L3 with >= 200 plots)', ((sh.loc[big] >= 0.10) & (sh.loc[big] <= 0.30)).all().all())
roles = [(f'k{k}', k, k % K + 1) for k in range(1, K + 1)] + [('final', None, 1)]
for name, t, v in roles:
    R = pd.read_csv(f'{d}/split_{name}.csv')
    m = R.merge(F[ids + ['fold']], on=ids, how='outer', indicator=True)
    check(f'R1 {name} same plots as folds.csv', (m._merge == 'both').all() and not R.duplicated(ids).any())
    exp = np.where(m.fold == t, 'test', np.where(m.fold == v, 'val', 'train')) if t is not None else np.where(m.fold == v, 'val', 'train')
    check(f'R2 {name} sets follow the fold roles', (m.set.values == exp).all(), ', '.join(f'{s} {int((m.set == s).sum())}' for s in ('train', 'val', 'test')))
# spatial separation: distance from every test plot to its nearest TRAIN plot (k1), vs a random plot split of the same sizes
R = pd.read_csv(f'{d}/split_k1.csv').merge(F[ids + ['lat', 'lon']], on=ids)
def nn_km(a, b):
    ya, xa = np.radians(a.lat.values)[:, None], np.radians(a.lon.values)[:, None]; yb, xb = np.radians(b.lat.values)[None, :], np.radians(b.lon.values)[None, :]
    h = np.sin((yb - ya) / 2) ** 2 + np.cos(ya) * np.cos(yb) * np.sin((xb - xa) / 2) ** 2
    return (2 * 6371 * np.arcsin(np.sqrt(h))).min(axis=1)
te, tr = R[R.set == 'test'], R[R.set == 'train']
d_blk = nn_km(te, tr)
rng = np.random.default_rng(1); p = rng.permutation(len(R)); rr = R.iloc[p]
d_rnd = nn_km(rr.iloc[:len(te)], rr.iloc[len(te):len(te) + len(tr)])
print(f'INFO nearest train plot of a test plot (k1), km: blocked median {np.median(d_blk):.1f} (p10 {np.percentile(d_blk, 10):.1f}) vs random split median {np.median(d_rnd):.1f} (p10 {np.percentile(d_rnd, 10):.1f})')
check('S1 blocked split is farther apart than a random split', np.median(d_blk) > 1.5 * np.median(d_rnd))
print('ALL PASS' if not fails else f'FAILED: {fails}'); sys.exit(1 if fails else 0)
