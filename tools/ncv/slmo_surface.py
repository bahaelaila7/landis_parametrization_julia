"""Species-level trade-off surface + summary of the species-level MO experiment (tools/ncv/slmo_bs.sh).
  python -I tools/ncv/slmo_surface.py <experiment dir>
TRAIN (a_g, w_g) of every sweep set = the last round of its minimize refinement (round*_groups.csv; gdigel groups species[·cell]@L3, aggregated to the
TOST group label by pair count: a = n-weighted mean, w = n-weighted mean). VAL per group from cands/tost_select_val_groups.csv; the per-group choice
from assembled_choices.csv. Writes surface.csv, surface.png (one panel per group: train a vs w, colour = lambda, marker = start pick, red ring =
the set chosen on val for that group) and summary.md (final val decision and the test report)."""
import sys, os, re, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
X = sys.argv[1]
files = [l.strip() for l in open(f'{X}/cands/list.txt') if l.strip()]
names = [os.path.basename(f)[:-5] for f in files]
rows = []
for k, nm in enumerate(names, start=1):
    m = re.match(r'sweep_(min_A|knee|min_W)_([0-9.]+)', nm); start, lam = m.group(1), float(m.group(2))
    rr = sorted(glob.glob(f'{X}/{nm}/round*_groups.csv'), key=lambda f: int(re.search(r'round(\d+)_', os.path.basename(f)).group(1)))
    g = pd.read_csv(rr[-1], keep_default_na=False, na_values=['NaN']); g['label'] = g.group.str.replace(r'@.*$', '', regex=True)
    g = g[g.n > 0]
    def wavg(v, n):   # NaN-aware pair-weighted mean (a sub-group without pairs / without a W value does not count; review 2026-10-24)
        ok = v.notna() & (n > 0); return np.average(v[ok], weights=n[ok]) if ok.any() else np.nan
    agg = g.groupby('label').apply(lambda d: pd.Series(dict(n_train=d.n.sum(), a_train=wavg(d.a, d.n), w_train=wavg(d.w, d.n))), include_groups=False).reset_index()
    agg['cand'] = k; agg['set'] = nm; agg['start'] = start; agg['lam'] = lam; rows.append(agg)
S = pd.concat(rows, ignore_index=True)
V = pd.read_csv(f'{X}/cands/tost_select_val_groups.csv', keep_default_na=False)
V = V.rename(columns={'group': 'label', 'pct_bias': 'val_bias_pct', 'mean_abs_lr': 'val_mabs', 'pass_5': 'val_p5', 'pass_10': 'val_p10', 'pass_20': 'val_p20'})
S = S.merge(V[['cand', 'label', 'n', 'val_bias_pct', 'val_mabs', 'val_p5', 'val_p10', 'val_p20']].rename(columns={'n': 'n_val'}), on=['cand', 'label'], how='left')
ch = pd.read_csv(f'{X}/assembled_choices.csv', keep_default_na=False)
chosen = dict(zip(ch.group, ch.file.str.replace('.jld2', '', regex=False)))
S['chosen'] = [chosen.get(l) == s for l, s in zip(S.label, S.set)]
S.to_csv(f'{X}/surface.csv', index=False)
labs = sorted(S.label.unique(), key=lambda l: -S[S.label == l].n_train.max())
nc = 7; nr = int(np.ceil(len(labs) / nc))
fig, axs = plt.subplots(nr, nc, figsize=(nc * 2.6, nr * 2.3), squeeze=False)
mk = {'min_A': 'o', 'knee': 's', 'min_W': '^'}
for ax, l in zip(axs.flat, labs):
    d = S[S.label == l]
    for st, dd in d.groupby('start'):
        sc = ax.scatter(dd.a_train, dd.w_train, c=dd.lam, cmap='viridis', vmin=0, vmax=1, marker=mk[st], s=22, edgecolors='none')
    c = d[d.chosen]
    if len(c): ax.scatter(c.a_train, c.w_train, s=90, facecolors='none', edgecolors='red', linewidths=1.4)
    pas = '' if not len(c) or pd.isna(c.val_p20.iloc[0]) else f" val±20 {'✓' if str(c.val_p20.iloc[0]).lower() == 'true' else '✗'}"
    ax.set_title(f'{l} (n {int(d.n_train.max())}){pas}', fontsize=7); ax.tick_params(labelsize=6)
for ax in list(axs.flat)[len(labs):]: ax.axis('off')
fig.supxlabel('train accuracy a = mean |log ratio| (lower better)'); fig.supylabel('train age-structure W1 (lower better)')
cb = fig.colorbar(sc, ax=axs, shrink=0.4, label='λ (1 = accuracy only)'); fig.suptitle(f'Species-level trade-off ({os.path.basename(X)}): o min_A, ■ knee, ▲ min_W start; red = chosen on val', fontsize=10)
fig.savefig(f'{X}/surface.png', dpi=110, bbox_inches='tight')
md = [f'# Species-level MO experiment — {os.path.basename(X)}', '', f'{len(names)} sweep sets (3 starts × λ grid), {V.label.nunique()} val groups ({len(labs)} with train values); per-group choice on val assembled from '
      f'{ch.file.nunique()} sets. NOTE: this first-run val comparison is circular (choices made on the same val) and D was not species-tied — see v2/summary.md.', '']
for split, f in (('val (decides: assembled vs best single)', f'{X}/final/val/tost_select_val.csv'), ('test (report; c = production selection, comparison only)', f'{X}/final/test/tost_select_test.csv')):
    if os.path.exists(f):
        t = pd.read_csv(f); nm = {1: 'a_assembled', 2: 'b_best_single', 3: 'c_production'}
        md += [f'## {split}', '', '| set | groups | ±5 % | ±10 % | ±20 % | median abs bias % | mean abs log ratio |', '|---|---|---|---|---|---|---|']
        md += [f"| {nm[int(r.cand)]} | {r.groups} | {r.pass_5} | {r.pass_10} | {r.pass_20} | {r.median_abs_bias} | {r.mean_abs_lr} |" for r in t.sort_values('cand').itertuples()]
        md += ['']
open(f'{X}/summary.md', 'w').write('\n'.join(md)); print('\n'.join(md))
