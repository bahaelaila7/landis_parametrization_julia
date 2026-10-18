"""HBAD3 viability test: GA+FL-fitted models scored WITHOUT refitting on Region 4 plots outside GA+FL (R4 split val; tools/hbad3/transfer_eval.jl).
For every model side by side: observed/simulated diameter and height increment of survivors (ratio of sums) and observed/expected deaths,
by EPA L3, by climate-normal class (annual T, CWD), by interval drought anomaly (growing-season CWD anomaly) and by soil AWC.
A transferable model has ratios near 1 in every class and no trend along the climate axes.
  python scripts/hbad3/transfer_climate.py <repo> <model> [<model> ...]      model = file stem after resid_transfer_ (e.g. v2_v2fv5, h3_H3b)"""
import sys, numpy as np, pandas as pd
R = sys.argv[1]; models = sys.argv[2:]
ic = pd.read_csv(f"{R}/results/hbad3/interval_climate_r4.csv"); pc = pd.read_csv(f"{R}/results/hbad3/plot_climate_r4.csv")
iw = pd.read_csv(f"{R}/results/hbad3/interval_water_r4.csv")
ic = ic.merge(pc[["st", "u", "c", "p", "awc_mm"]], how="left").merge(iw, on="iid")
ic["l3"] = ic.l4.str.rsplit(".", n=1).str[0]
def load(m):
  d = pd.read_csv(f"{R}/results/hbad3/resid_transfer_{m}.csv").merge(ic, on="iid", how="left")
  d["gDo"] = (d.D2o - d.D1) / d.dt; d["gDs"] = (d.D2s - d.D1) / d.dt; d["gHo"] = (d.H2o - d.H1) / d.dt; d["gHs"] = (d.H2s - d.H1) / d.dt
  d["do"] = d.n1 - d.s; d["de"] = d.n1 * (1 - d.ps); return d
D = {m: load(m) for m in models}
ref = D[models[0]]
cuts = {"tmean_n": np.quantile(ref.tmean_n, [0, .2, .4, .6, .8, 1]), "cwd_n": np.quantile(ref.cwd_n, [0, .2, .4, .6, .8, 1]),
        "cwd_gs_an": np.quantile(ref.cwd_gs_an, [0, 1/3, 2/3, 1]), "aetpet_iv": np.quantile(ref.aetpet_iv, [0, .2, .4, .6, .8, 1]),
        "awc_mm": np.quantile(ref.awc_mm, [0, 1/3, 2/3, 1])}
def ratios(x):
  s = x[x.s > 0]
  return pd.Series({"trees": x.n1.sum(), "dD": (s.s * s.gDo).sum() / (s.s * s.gDs).sum(), "dH": (s.s * s.gHo).sum() / (s.s * s.gHs).sum(), "mOE": x.do.sum() / x.de.sum()})
def table(by):
  parts = []
  for m in models:
    d = D[m].copy()
    if by in cuts:
      e = cuts[by].copy(); e[0] -= 1e-9; e[-1] += 1e-9
      d["cls"] = pd.cut(d[by], e).astype(str)
    else:
      d["cls"] = d[by]
    t = d.groupby("cls", observed=True).apply(ratios, include_groups=False)
    t.columns = pd.MultiIndex.from_product([[m], t.columns]); parts.append(t)
  t = pd.concat(parts, axis=1)
  keep = [c for c in t.columns if c[1] != "trees" or c[0] == models[0]]
  return t[keep]
print("== OVERALL (transfer set)")
print(pd.concat([ratios(D[m]).rename(m) for m in models], axis=1).round(3).to_string())
for by, lab in (("l3", "EPA L3"), ("tmean_n", "annual mean T normal (°C)"), ("cwd_n", "climatic water deficit normal (mm)"),
                ("aetpet_iv", "interval AET/PET (process water index)"), ("cwd_gs_an", "interval growing-season CWD anomaly (mm)"), ("awc_mm", "soil AWC (mm)")):
  print(f"\n== by {lab}"); print(table(by).round(3).to_string())
# one-number summaries: trees-weighted mean |ln ratio| across L3 classes and across T / CWD classes (0 = no structured bias)
print("\n== structured bias: tree-weighted mean |ln ratio| over classes (lower = more transferable)")
rows = []
for m in models:
  for by in ("l3", "tmean_n", "cwd_n", "aetpet_iv"):
    t = table(by)[m]; w = table(by)[(models[0], "trees")]
    rows.append({"model": m, "axis": by, **{k: float(np.sum(w * np.abs(np.log(t[k]))) / w.sum()) for k in ("dD", "dH", "mOE")}})
print(pd.DataFrame(rows).pivot(index="axis", columns="model").round(3).to_string())
