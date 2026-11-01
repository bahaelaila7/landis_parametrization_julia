"""Fit FIA tree volume equations ln(V)=b0+b1 ln(ba)+b2 ln(HT) for EPA Region 4 states.
NOTE: VOLBSGRS (Scribner) is an empty VARCHAR column in this DB (0 positive rows), so board feet use
VOLBFGRS (gross International 1/4-inch board feet; same 426,965 sawtimber rows as VOLCSGRS).
Run: /workspace/standmap/.venv/bin/python -I tools/fit_volume_eqs.py
Writes data/volume_eqs_r4.csv; prints stats/validation tables (markdown) to stdout."""
import os, sys
import duckdb, numpy as np, pandas as pd

DB = os.environ.get('PAN_FIA_DB', os.path.join(os.environ.get('WORKSPACE', '/workspace'), 'FIA/truth/fia_truth_r4.duckdb'))
OUT = '/workspace/landis_scen/data/volume_eqs_r4.csv'
STATES = (1, 12, 13, 21, 28, 37, 45, 47)
K = 0.005454154
MAXN = 200_000
MIN_N = {'tcuft': 300, 'bdft': 150}
VCOL = {'tcuft': 'vt', 'bdft': 'vb'}  # DB columns VOLTSGRS, VOLBFGRS

def load():
    c = duckdb.connect(DB, read_only=True)
    q = f"""select t.CN tcn, t.PLT_CN plt, t.SPCD spcd, t.SPGRPCD grp, r.SPECIES_SYMBOL sym,
        r.COMMON_NAME cname, t.DIA dia, t.HT ht, t.VOLTSGRS vt, t.VOLBFGRS vb, t.TPA_UNADJ tpa
        from TREE t join PLOT p on t.STATECD=p.STATECD and t.UNITCD=p.UNITCD and t.COUNTYCD=p.COUNTYCD and t.PLOT=p.PLOT and t.INVYR=p.INVYR join REF_SPECIES r on t.SPCD=r.SPCD
        where t.STATUSCD=1 and t.DIA>=1.0 and t.HT>0 and p.MEASYEAR>=2010 and p.STATECD in {STATES}"""
    df = c.sql(q).df(); c.close()
    df = df.sort_values('tcn').reset_index(drop=True)
    df['major'] = np.where(df.spcd < 300, 'S', 'H')
    df['ba'] = K * df.dia ** 2
    return df

def fit_one(d, vcol, seed=1):
    d = d[d[vcol] > 0]
    n_all = len(d)
    if n_all > MAXN:
        d = d.sample(MAXN, random_state=seed)
    X = np.column_stack([np.ones(len(d)), np.log(d.ba), np.log(d.ht)])
    y = np.log(d[vcol].values)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    s = np.sqrt(res @ res / (len(y) - 3))
    r2 = 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))
    return dict(b0=beta[0], b1=beta[1], b2=beta[2], cf=np.exp(s * s / 2), n=n_all,
                rmse_log=s, r2_log=r2)

SAW_MIN = {'S': 9.0, 'H': 11.0}  # sawtimber thresholds (verified from data)

def fit_logit(x, y, w):
    """TPA-weighted logistic y ~ a + c*x (x = ln DIA); Newton/IRLS with tiny ridge."""
    w = w / w.mean(); m = np.average(y, weights=w)
    if m > 0.999: return 12.0, 0.0
    if m < 0.001: return -12.0, 0.0
    X = np.column_stack([np.ones(len(x)), x]); b = np.array([np.log(m / (1 - m)), 0.0])
    for _ in range(50):
        pr = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        Wd = w * pr * (1 - pr) + 1e-12
        H = (X * Wd[:, None]).T @ X + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, X.T @ (w * (y - pr)) - 1e-6 * b)
        b = b + step
        if np.abs(step).max() < 1e-8: break
    return float(b[0]), float(b[1])

def fit_sq(sub):
    thr = np.where(sub.major.values == 'S', SAW_MIN['S'], SAW_MIN['H'])
    d = sub[sub.dia.values >= thr]
    if len(d) < 20: return np.nan, np.nan
    return fit_logit(np.log(d.dia.values), (d.vb.values > 0).astype(float), d.tpa.values)

def fit_all(df):
    rows = []
    for model, vcol in VCOL.items():
        pos = df[df[vcol] > 0]
        mins = {m: float(np.percentile(pos[pos.major == m].dia, 1)) for m in 'SH'}
        for level, col in (('spcd', 'spcd'), ('spgrpcd', 'grp'), ('major', 'major')):
            cnt = pos.groupby(col).size()
            for key, n in cnt.items():
                if level != 'major' and n < MIN_N[model]:
                    continue
                sub = df[df[col] == key].sort_values("tcn")
                f = fit_one(sub, vcol)
                if level == 'spcd':
                    r0 = sub.iloc[0]; sym, name = r0.sym, r0.cname
                elif level == 'major':
                    sym, name = '', 'Softwood' if key == 'S' else 'Hardwood'
                else:
                    sym, name = '', ''
                if model == 'bdft':
                    sp = sub[sub[vcol] > 0]
                    md = float(np.percentile(sp.dia, 1))
                    sa, sc = fit_sq(sub)
                else:
                    md = sa = sc = np.nan
                rows.append(dict(model=model, level=level, key=key, symbol=sym, name=name, **f, min_dbh=md,
                                 sq_a=sa, sq_c=sc))
    return pd.DataFrame(rows), mins

def predict(df, eqs, model, sq=False):
    """most specific available level: spcd -> spgrpcd -> major. returns pred (per tree) and level used"""
    e = eqs[eqs.model == model]
    pred = np.full(len(df), np.nan); lvl = np.array([''] * len(df), dtype=object)
    for level, col in (('major', 'major'), ('spgrpcd', 'grp'), ('spcd', 'spcd')):  # later overrides earlier
        el = e[e.level == level].copy()
        el['key'] = el.key.astype(str)
        m = el.set_index('key')
        k = df[col].astype(str)
        has = k.isin(m.index).values
        if not has.any(): continue
        p = m.loc[k[has]]
        v = (p.cf.values * np.exp(p.b0.values) * df.ba.values[has] ** p.b1.values *
             df.ht.values[has] ** p.b2.values)
        if sq:  # sawtimber-qualifying probability; 0 below the sawtimber DBH threshold
            dia = np.sqrt(df.ba.values[has] / K)
            thr = np.where(df.major.values[has] == 'S', SAW_MIN['S'], SAW_MIN['H'])
            pq = 1 / (1 + np.exp(-(p.sq_a.values + p.sq_c.values * np.log(dia))))
            v = np.where(dia >= thr - 1e-9, pq * v, 0.0)
        pred[has] = v; lvl[has] = level
    return pred, lvl

def pred_with_ba_ht(ba, ht, spcd, grp, major, eqs, model, sq=False):
    d = pd.DataFrame(dict(ba=ba, ht=ht, spcd=spcd, grp=grp, major=major))
    return predict(d, eqs, model, sq)[0]

def plot_metrics(g, obs, pred):
    """g: plot ids; obs, pred per-tree contributions (already tpa weighted)"""
    t = pd.DataFrame(dict(g=g, o=obs, p=pred)).groupby('g').sum()
    t = t[t.o > 0]
    r2 = 1 - ((t.o - t.p) ** 2).sum() / ((t.o - t.o.mean()) ** 2).sum()
    bias = 100 * (t.p.sum() - t.o.sum()) / t.o.sum()
    mbias = (100 * (t.p - t.o) / t.o).mean()
    mape = (100 * (t.p - t.o).abs() / t.o).median()
    return dict(n=len(t), R2=r2, bias_pct_total=bias, mean_plot_bias_pct=mbias, med_abs_pct_err=mape)

def md(df, floatfmt='.3f', index=True):
    d = df.reset_index() if index else df
    f = lambda v: format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v)
    L = ['| ' + ' | '.join(map(str, d.columns)) + ' |', '|' + '---|' * len(d.columns)]
    for r in d.itertuples(index=False):
        L.append('| ' + ' | '.join(f(v) for v in r) + ' |')
    return '\n'.join(L)

def main():
    df = load()
    print(f"## Data\nTrees: {len(df):,}; plots: {df.plt.nunique():,}\n")
    # null/zero stats
    bins = [1, 5, 9, 11, 15, 20, 30, 200]
    df['dcls'] = pd.cut(df.dia, bins, right=False)
    s = df.groupby('dcls', observed=True).agg(n=('dia', 'size'),
        vt_null=('vt', lambda x: x.isna().mean() * 100), vt_zero=('vt', lambda x: (x == 0).mean() * 100),
        vb_null=('vb', lambda x: x.isna().mean() * 100), vb_zero=('vb', lambda x: (x == 0).mean() * 100))
    print("### Null/zero rates (%) by DBH class\n" + md(s.round(2), '.2f') + "\n")
    for m in 'SH':
        pv = df[(df.major == m) & (df.vb > 0)].dia
        print(f"VOLBFGRS>0 DIA percentiles {m}: min {pv.min():.1f}, p1 {np.percentile(pv,1):.2f}, p5 {np.percentile(pv,5):.2f}; n={len(pv):,}")
        nb = df[(df.major == m) & (df.dia >= (9 if m=='S' else 11)) & (df.vb.isna() | (df.vb == 0))]
        print(f"  trees >= threshold with null/zero VOLBFGRS: {len(nb):,} of {((df.major==m)&(df.dia>=(9 if m=='S' else 11))).sum():,}")
    # drop null-volume trees from the validation population (keep them out of obs totals)
    # ---- holdout split by plot
    rng = np.random.RandomState(42)
    plots = np.sort(df.plt.unique()); hold = set(rng.choice(plots, int(0.2 * len(plots)), replace=False))
    is_h = df.plt.isin(hold).values
    train, test = df[~is_h], df[is_h]
    eq_tr, mins_tr = fit_all(train)
    eq_all, mins_all = fit_all(df)
    eq_all['min_dbh'] = eq_all.min_dbh.round(1)
    out = eq_all[['model','level','key','symbol','name','b0','b1','b2','cf','n','rmse_log','r2_log','min_dbh','sq_a','sq_c']]
    out.to_csv(OUT, index=False, float_format='%.6g')
    print("\n### Row counts\n" + out.groupby(['model','level']).size().to_string() + "\n")
    print("Major min_dbh (full fit, 1st pct):", mins_all, "\nlevel dbh p1 for bdft major:\n",
          out[(out.model=='bdft')&(out.level=='major')][['key','min_dbh']].to_string())
    print("\n### Major equations\n", md(out[out.level=='major'], '.4f', index=False))
    top = out[out.level=='spcd'].sort_values('n', ascending=False)
    print("\n### Top species tcuft\n", md(top[top.model=='tcuft'].head(12).drop(columns=['model','level','min_dbh']), '.4f', index=False))
    print("\n### Top species bdft\n", md(top[top.model=='bdft'].head(12).drop(columns=['model','level']), '.4f', index=False))

    # ---- per-tree log RMSE on holdout
    print("\n### Per-tree holdout log RMSE (train-fit equations, most specific level)")
    tmin = {m: eq_tr[(eq_tr.model=='bdft')&(eq_tr.level=='major')&(eq_tr.key==m)].min_dbh.iloc[0] for m in 'SH'}
    print("train major min_dbh:", tmin)
    res = {}
    for model, vcol in VCOL.items():
        t = test[test[vcol] > 0]
        p, lvl = predict(t, eq_tr, model)
        e = np.log(p) - np.log(t[vcol].values)
        # note: pred includes cf; report on log scale of bias-corrected prediction
        print(f"{model}: n={len(t):,} RMSE_log={np.sqrt(np.mean(e**2)):.4f} mean_log_err={e.mean():.4f}; level mix: {pd.Series(lvl).value_counts().to_dict()}")
        # Pan approx per-tree
        pa = t.ba.values * t.ht.values * 0.42 * (4.0 if model=='bdft' else 1.0)
        ea = np.log(pa) - np.log(t[vcol].values)
        print(f"   Pan 0.42 approx: RMSE_log={np.sqrt(np.mean(ea**2)):.4f} mean_log_err={ea.mean():.4f}")

    # ---- plot-level
    print("\n### Plot-level totals (held-out plots)")
    rows = []
    T = test[test.vt.notna()].copy()
    pt, _ = predict(T, eq_tr, 'tcuft')
    T['pt_new'] = pt * T.tpa; T['ot'] = T.vt * T.tpa; T['pt_pan'] = T.ba * T.ht * 0.42 * T.tpa
    rows.append(('tcuft ft3/ac', 'new', plot_metrics(T.plt, T.ot, T.pt_new)))
    rows.append(('tcuft ft3/ac', 'Pan 0.42', plot_metrics(T.plt, T.ot, T.pt_pan)))
    B = test.copy(); B['vb0'] = B.vb.fillna(0)
    pb, _ = predict(B, eq_tr, 'bdft')
    mdb = B.major.map(tmin).values
    B['pb_nosq'] = np.where(B.dia.values >= mdb, pb, 0) * B.tpa
    B['pb_new'] = predict(B, eq_tr, 'bdft', sq=True)[0] * B.tpa
    B['pb_pan'] = np.where(B.dia >= 9, 4.0 * 0.42 * B.ba * B.ht, 0) * B.tpa
    B['ob'] = B.vb0 * B.tpa
    rows.append(('bdft bf/ac', 'new, no sq prob', plot_metrics(B.plt, B.ob, B.pb_nosq)))
    rows.append(('bdft bf/ac', 'new + sq prob', plot_metrics(B.plt, B.ob, B.pb_new)))
    rows.append(('bdft bf/ac', 'Pan 4x0.42', plot_metrics(B.plt, B.ob, B.pb_pan)))
    Bp = B[B.vb0 > 0]  # diagnostic: only sawtimber-size trees that actually carry board feet
    rows.append(('bdft bf/ac (only trees with VOLBFGRS>0)', 'new, no sq prob', plot_metrics(Bp.plt, Bp.ob, Bp.pb_nosq)))
    rows.append(('bdft bf/ac (only trees with VOLBFGRS>0)', 'Pan 4x0.42', plot_metrics(Bp.plt, Bp.ob, Bp.pb_pan)))
    pm = pd.DataFrame([dict(quantity=a, method=b, **c) for a, b, c in rows])
    print(md(pm, ".3f", index=False))

    # ---- cohort level
    print("\n### Cohort-level (plot x species x 2-inch DBH class), held-out plots")
    crow = []
    for model in ('tcuft', 'bdft'):
        X = T.copy() if model == 'tcuft' else B[B.dia.values >= B.major.map(tmin).values].copy()
        X['ov'] = (X.vt if model == 'tcuft' else X.vb0) * X.tpa
        X['d2w'] = X.dia ** 2 * X.tpa; X['hw'] = X.ht * X.tpa
        X['cls'] = (X.dia // 2).astype(int)
        g = X.groupby(['plt', 'spcd', 'cls']).agg(tpa=('tpa', 'sum'), d2w=('d2w', 'sum'), hw=('hw', 'sum'),
            ov=('ov', 'sum'), grp=('grp', 'first'), major=('major', 'first'), ntree=('tpa', 'size')).reset_index()
        qmd = np.sqrt(g.d2w / g.tpa); hbar = g.hw / g.tpa
        ba = K * qmd ** 2
        pn = g.tpa * pred_with_ba_ht(ba.values, hbar.values, g.spcd.values, g.grp.values, g.major.values, eq_tr, model)
        pp = g.tpa * ba * hbar * 0.42 * (4.0 if model == 'bdft' else 1.0)
        for name, p in (('new', pn), ('Pan approx', pp)):
            r2 = 1 - ((g.ov - p) ** 2).sum() / ((g.ov - g.ov.mean()) ** 2).sum()
            crow.append(dict(model=model, method=name, cohorts=len(g), bias_pct=100 * (p.sum() - g.ov.sum()) / g.ov.sum(),
                             R2=r2, med_abs_pct_err=np.nanmedian(100 * np.abs(p - g.ov) / g.ov.replace(0, np.nan))))
    print(md(pd.DataFrame(crow), '.3f', index=False))

    # ---- bdft cohorts with sq probability: cohorts of ALL trees (as Pan holds them)
    from math import erf
    print("\n### bdft cohort-level with sq probability (all trees in plot x species x 2-in class; obs = summed VOLBFGRS*TPA, null=0)")
    X = B.copy(); X['d2w'] = X.dia ** 2 * X.tpa; X['hw'] = X.ht * X.tpa; X['cls'] = (X.dia // 2).astype(int)
    g = X.groupby(['plt', 'spcd', 'cls']).agg(tpa=('tpa', 'sum'), d2w=('d2w', 'sum'), hw=('hw', 'sum'),
        ov=('ob', 'sum'), grp=('grp', 'first'), major=('major', 'first')).reset_index()
    qmd = np.sqrt(g.d2w / g.tpa); hbar = g.hw / g.tpa; ba = K * qmd ** 2
    thr = g.major.map(SAW_MIN).values
    f0 = pred_with_ba_ht(ba.values, hbar.values, g.spcd.values, g.grp.values, g.major.values, eq_tr, 'bdft', sq=False)
    pqf = pred_with_ba_ht(ba.values, hbar.values, g.spcd.values, g.grp.values, g.major.values, eq_tr, 'bdft', sq=True)
    # pqf = p(QMD)*f0 where QMD>=thr else 0  -> p(QMD) = pqf/f0
    pq = np.where(f0 > 0, pqf / np.where(f0 > 0, f0, 1), 0.0)
    pq_raw = np.where(qmd.values >= thr - 1e-9, pq, 0.0)
    # p(QMD) irrespective of threshold, for the normal-share variant
    pq_any = pred_with_ba_ht(np.maximum(ba.values, K * (thr + 1e-6) ** 2), hbar.values, g.spcd.values, g.grp.values, g.major.values, eq_tr, 'bdft', sq=True)
    f_any = pred_with_ba_ht(ba.values, hbar.values, g.spcd.values, g.grp.values, g.major.values, eq_tr, 'bdft', sq=False)
    # exact p(QMD): recompute from equation coefficients
    e = eq_tr[eq_tr.model == 'bdft']; pcoef = np.full((len(g), 2), np.nan)
    for level, col in (('major', 'major'), ('spgrpcd', 'grp'), ('spcd', 'spcd')):
        m = e[e.level == level].assign(key=lambda d: d.key.astype(str)).set_index('key')
        k = g[col].astype(str); has = k.isin(m.index).values
        pcoef[has, 0] = m.loc[k[has]].sq_a.values; pcoef[has, 1] = m.loc[k[has]].sq_c.values
    pQ = 1 / (1 + np.exp(-(pcoef[:, 0] + pcoef[:, 1] * np.log(qmd.values))))
    Phi = np.vectorize(lambda z: 0.5 * (1 + erf(z / np.sqrt(2))))
    share = 1 - Phi((thr - qmd.values) / (0.25 * qmd.values))
    cr = []
    for name, p in (('A: p(QMD) if QMD>=min_dbh', g.tpa * np.where(qmd.values >= thr, pQ, 0) * f_any),
                    ('B: normal share (CV .25) * p(QMD)', g.tpa * share * pQ * f_any),
                    ('no sq prob, QMD>=min_dbh', g.tpa * np.where(qmd.values >= thr, 1.0, 0) * f_any),
                    ('Pan 4x0.42, QMD>=9', g.tpa * np.where(qmd.values >= 9, 4 * 0.42 * ba * hbar, 0))):
        p = np.asarray(p); o = g.ov.values
        r2 = 1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum()
        # cohorts with any observed board feet
        mk = o > 0
        cr.append(dict(method=name, cohorts=len(g), bias_pct=100 * (p.sum() - o.sum()) / o.sum(), R2=r2,
                       med_abs_pct_err_obs_pos=np.median(100 * np.abs(p[mk] - o[mk]) / o[mk])))
    print(md(pd.DataFrame(cr), '.3f', index=False))
    # same cohorts, plot-level aggregation
    pr = []
    for name, p in (('A', g.tpa * np.where(qmd.values >= thr, pQ, 0) * f_any), ('B', g.tpa * share * pQ * f_any)):
        pr.append(dict(method=name, **plot_metrics(g.plt, g.ov, p)))
    print("\nPlot totals from cohort-level predictions\n" + md(pd.DataFrame(pr), '.3f', index=False))
    sqc = out[(out.model=='bdft')&(out.level=='major')][['key','sq_a','sq_c']]
    print("\nsq major:\n" + md(sqc, '.4f', index=False))
    print(md(out[(out.model=='bdft')&(out.level=='spcd')].sort_values('n', ascending=False).head(8)[['key','symbol','sq_a','sq_c']], '.4f', index=False))

main()
