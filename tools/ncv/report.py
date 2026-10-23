"""Pooled out-of-fold report of a nested-CV campaign (docs/nested_cv_gafl.md §3). Only OUTER TEST data enter the headline numbers.
  python -I tools/ncv/report.py <root> [roles k1,...,k5]
BS / HBAD v1 / PnET: the test pairs of every fold's val-selected set (test_pairs/pairs_cand1.csv, written by tools/tost_select.jl TOST_DUMP) are pooled per
group (label = species, or species·cell for the cell-split species, as tools/tost_common.jl). Per group: n pairs, mean log1p-difference (bias %),
mean |log1p-difference| (accuracy), and two TOSTs at ±5/10/20 % (alpha 0.05, 90 % CI inside ±log1p(m)):
  pair   the one-sample t CI of tools/tost_common.jl on the pooled pairs (pairs treated as independent, as in every earlier report)
  plot   a plot-cluster percentile bootstrap of the mean (plots = (fold, plot) resampled with replacement, B 2000, seed 1): pairs of one plot are not
         independent (several cohorts and visits), so this CI is the honest one and the headline.
Also per fold: the val-selected family / candidate and the fold's own test pass counts (spread across folds). HBAD v2 / HBAD3P: their test lines.
Writes <root>/report/{<model>_groups.csv, folds.csv, summary.md}."""
import sys, os, re, glob, numpy as np, pandas as pd
from scipy import stats
R = sys.argv[1]; ROLES = sys.argv[2].split(',') if len(sys.argv) > 2 else [f'k{k}' for k in range(1, 6)]
OUT = f'{R}/report'; os.makedirs(OUT, exist_ok=True)
MARG = [0.05, 0.10, 0.20]; ALPHA = 0.05; B = 2000; MIN_PLOTS = 10   # plot-cluster TOST only for groups with >= MIN_PLOTS test plots
md = ['# Nested-CV calibration — pooled out-of-fold results', '', f'Root `{R}`; folds {", ".join(ROLES)}. Headline TOST = plot-cluster bootstrap.', '']
def tost_t(d):
    n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n); q = stats.t.ppf(1 - ALPHA, n - 1)
    return m - q * se, m + q * se
def tost_boot(df, rng):
    g = df.groupby(['fold', 'plot_id']).dd.agg(['sum', 'count']).values; k = len(g)
    idx = rng.integers(0, k, size=(B, k)); s = g[idx, 0].sum(1); c = g[idx, 1].sum(1)
    return tuple(np.quantile(s / c, [ALPHA, 1 - ALPHA]))
fold_rows = []
for model in ('bs', 'hbad', 'pnet'):
    parts = []
    for r in ROLES:
        f = f'{R}/{model}/{r}/test_pairs/pairs_cand1.csv'
        if not os.path.exists(f): continue
        p = pd.read_csv(f); p['fold'] = r; parts.append(p)
        src = open(f'{R}/{model}/{r}/cands/selected_val/source.txt').read().strip()
        tv = pd.read_csv(f'{R}/{model}/{r}/cands/tost_select_val.csv').iloc[0]
        tt = pd.read_csv(f'{R}/{model}/{r}/cands/selected_val/tost_select_test.csv').iloc[0]
        fold_rows.append(dict(model=model, fold=r, selected=os.path.basename(src).replace('.jld2', ''), groups_val=int(tv.groups),
                              val_pass_5=int(tv.pass_5), val_pass_10=int(tv.pass_10), val_pass_20=int(tv.pass_20), groups_test=int(tt.groups),
                              test_pass_5=int(tt.pass_5), test_pass_10=int(tt.pass_10), test_pass_20=int(tt.pass_20),
                              test_median_abs_bias=float(tt.median_abs_bias), test_mean_abs_lr=float(tt.mean_abs_lr)))
    if not parts: continue
    P = pd.concat(parts, ignore_index=True)
    P['label'] = [s if g == 'pooled' else f'{s}·{g}' for s, g in zip(P.species, P.pgrp)]
    P['dd'] = np.log1p(P.sim_agb.astype(float)) - np.log1p(P.obs_agb.astype(float))
    rng = np.random.default_rng(1); rows = []
    for lab, d in P.groupby('label'):
        if len(d) < 3: continue
        lo_t, hi_t = tost_t(d.dd.values); lo_b, hi_b = tost_boot(d, rng)
        row = dict(group=lab, n_pairs=len(d), n_plots=d.groupby(['fold', 'plot_id']).ngroups, n_folds=d.fold.nunique(),
                   bias_pct=100 * np.expm1(d.dd.mean()), mean_abs_lr=d.dd.abs().mean())
        for m in MARG:
            row[f'pair_{int(100*m)}'] = bool(lo_t > -np.log1p(m) and hi_t < np.log1p(m))
            row[f'plot_{int(100*m)}'] = bool(row['n_plots'] >= MIN_PLOTS and lo_b > -np.log1p(m) and hi_b < np.log1p(m))
        row.update(ci_plot_lo_pct=100 * np.expm1(lo_b), ci_plot_hi_pct=100 * np.expm1(hi_b)); rows.append(row)
    G = pd.DataFrame(rows).sort_values('n_pairs', ascending=False); G.to_csv(f'{OUT}/{model}_groups.csv', index=False)
    md += [f'## {model} (pooled test pairs of {P.fold.nunique()} folds, {len(G)} groups)', '',
           '| TOST | ±5 % | ±10 % | ±20 % |', '|---|---|---|---|',
           f'| plot-cluster bootstrap (headline; groups with < {MIN_PLOTS} test plots count as not equivalent) | ' + ' | '.join(f"{int(G[f'plot_{int(100*m)}'].sum())}/{len(G)}" for m in MARG) + ' |',
           '| pair t (as earlier reports) | ' + ' | '.join(f"{int(G[f'pair_{int(100*m)}'].sum())}/{len(G)}" for m in MARG) + ' |', '',
           f'Median |bias| over groups {np.median(np.abs(G.bias_pct)):.1f} %; mean |log ratio| (mean over groups) {G.mean_abs_lr.mean():.3f}.', '']
F = pd.DataFrame(fold_rows)
if len(F):
    F.to_csv(f'{OUT}/folds.csv', index=False)
    tab = ['| ' + ' | '.join(F.columns) + ' |', '|' + '---|' * len(F.columns)] + ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in F.itertuples(index=False)]
    md += ['## Per fold (selection on inner val; test reported)', '', *tab, '']
# HBAD v2 / HBAD3P: the test lines their own tools log
for model, pat, files in (('hbad2', r'\btest\b', 'logs/hbad2/{r}/joint.log'), ('hbad3p', r'^TEST|GAIN', 'hbad3p/{r}/fit3p_{r}/log.txt')):
    lines = []
    for r in ROLES:
        for f in sorted(glob.glob(f'{R}/' + files.format(r=r))):
            lines += [f'{r}: {l.strip()}' for l in open(f) if re.search(pat, l)]
    if lines: md += [f'## {model} test lines', '', '```', *lines, '```', '']
open(f'{OUT}/summary.md', 'w').write('\n'.join(md)); print('\n'.join(md))
