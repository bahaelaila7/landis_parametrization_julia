"""Outer folds of the nested-CV calibration (docs/nested_cv_gafl.md §1): every FIA plot of GA+FL in one of K spatially blocked folds, and the role files.
  Blocks: BLOCK_DEG x BLOCK_DEG lat/lon cells of the plot's (last-visit) public coordinates. Blocks are dealt in a seeded random order, each to the fold
  with the fewest plots of the block's dominant L3 (ties: fewest plots overall, then the lowest fold number), so folds are L3-proportional and whole blocks.
  L3 of a plot = the most frequent epa_l3 of its curated cohorts (else of its curated trees; plots with neither = "NA").
Outputs in <out>: folds.csv (statecd, unitcd, countycd, plot, fold, block, l3, lat, lon) and one role file per role (statecd, unitcd, countycd, plot,
  eco, dom, set): split_k<k>.csv (test = fold k, val = fold k mod K + 1, train = the rest) and split_final.csv (val = fold 1, train = the rest, no test).
  python -I tools/ncv/make_folds.py --db <slim_r4.duckdb> --out <dir> [--k 5] [--block-deg 0.25] [--seed 20261023] [--states 12,13]"""
import argparse, os, sys
import numpy as np, pandas as pd, duckdb

ap = argparse.ArgumentParser()
ap.add_argument('--db', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--k', type=int, default=5); ap.add_argument('--block-deg', type=float, default=0.25)
ap.add_argument('--seed', type=int, default=20261023); ap.add_argument('--states', default='12,13')
a = ap.parse_args()
states = [int(s) for s in a.states.split(',')]
con = duckdb.connect(a.db, read_only=True)
st = ','.join(map(str, states))
plots = con.execute(f"""
  SELECT STATECD statecd, UNITCD unitcd, COUNTYCD countycd, PLOT plot, arg_max(LAT, INVYR) lat, arg_max(LON, INVYR) lon
  FROM PLOT WHERE STATECD IN ({st}) GROUP BY 1, 2, 3, 4""").df()
l3c = con.execute(f"""SELECT statecd, unitcd, countycd, plot, mode(epa_l3) l3c FROM curated_cohorts_landis_stdorg WHERE statecd IN ({st}) GROUP BY 1, 2, 3, 4""").df()
l3t = con.execute(f"""SELECT STATECD statecd, UNITCD unitcd, COUNTYCD countycd, PLOT plot, mode(epa_l3) l3t FROM curated_trees WHERE STATECD IN ({st}) GROUP BY 1, 2, 3, 4""").df()
con.close()
ids = ['statecd', 'unitcd', 'countycd', 'plot']
for d in (plots, l3c, l3t):
    for c in ids: d[c] = d[c].astype(int)
df = plots.merge(l3c, on=ids, how='left').merge(l3t, on=ids, how='left')
df['l3'] = df.l3c.fillna(df.l3t).fillna('NA').astype(str)
if df.lat.isna().any() or df.lon.isna().any(): sys.exit(f'{int(df.lat.isna().sum())} plots without coordinates')
df['block'] = [f'{int(np.floor(la / a.block_deg))}_{int(np.floor(lo / a.block_deg))}' for la, lo in zip(df.lat, df.lon)]
df = df.sort_values(ids).reset_index(drop=True)
# deal the blocks
bl = df.groupby('block').agg(n=('plot', 'size'), l3=('l3', lambda s: s.value_counts().sort_index().idxmax())).reset_index().sort_values('block')
rng = np.random.default_rng(a.seed); order = rng.permutation(len(bl))
fold_n = np.zeros(a.k, int); fold_l3 = {}
assign = {}
for i in order:
    b = bl.iloc[i]; c = fold_l3.setdefault(b.l3, np.zeros(a.k, int))
    f = min(range(a.k), key=lambda j: (c[j], fold_n[j], j))
    assign[b.block] = f + 1; c[f] += b.n; fold_n[f] += b.n
df['fold'] = df.block.map(assign).astype(int)
os.makedirs(a.out, exist_ok=True)
df[ids + ['fold', 'block', 'l3', 'lat', 'lon']].to_csv(f'{a.out}/folds.csv', index=False)
def role(sets, name):
    r = df[ids].copy(); r['eco'] = df.l3; r['dom'] = ''; r['set'] = sets
    r.to_csv(f'{a.out}/split_{name}.csv', index=False)
    print(f'split_{name}: ' + ', '.join(f'{s} {int((r.set == s).sum())}' for s in ('train', 'val', 'test')))
for k in range(1, a.k + 1):
    v = k % a.k + 1
    role(np.where(df.fold == k, 'test', np.where(df.fold == v, 'val', 'train')), f'k{k}')
role(np.where(df.fold == 1, 'val', 'train'), 'final')
print(f'{len(df)} plots, {len(bl)} blocks of {a.block_deg} deg; plots per fold {fold_n.tolist()}')
print(pd.crosstab(df.l3, df.fold, margins=True).to_string())
