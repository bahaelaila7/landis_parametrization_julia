"""S2b: is the interval water-balance signal a calendar-period artefact? Wet vs dry (CWD_gs anomaly terciles) within interval-midpoint periods,
and within L3 x period. Ratios as in climate_signal.py (observed / simulated diameter increment of survivors; observed / expected deaths)."""
import sys, numpy as np, pandas as pd, pyarrow.ipc as ip
R = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else "v2j"
d = pd.read_csv(f"{R}/results/hbad3/resid_{tag}.csv"); ic = pd.read_csv(f"{R}/results/hbad3/interval_climate.csv")
iv = ip.open_file(f"{R}/results/hbad2/intervals.arrow").read_all().to_pandas()[["iid", "d1", "d2"]]
iv["d1"] = pd.to_datetime(iv.d1); iv["d2"] = pd.to_datetime(iv.d2); iv["mid"] = iv.d1.dt.year + (iv.d2.dt.year - iv.d1.dt.year) / 2
d = d.merge(ic, on="iid").merge(iv, on="iid"); d["l3"] = d.l4.str.rsplit(".", n=1).str[0]
d["gDo"] = (d.D2o - d.D1) / d.dt; d["gDs"] = (d.D2s - d.D1) / d.dt; d["dead_o"] = d.n1 - d.s; d["dead_e"] = d.n1 * (1 - d.ps)
d["per"] = pd.cut(d.mid, [0, 2006, 2010, 2014, 2030], labels=["<=2006", "2007-10", "2011-14", ">=2015"])
d["wet"] = d.groupby("per", observed=True).cwd_gs_an.transform(lambda x: pd.qcut(x, 3, labels=["wet", "mid", "dry"]))
def r(x):
  sv = x[x.s > 0]
  return pd.Series(dict(trees=x.n1.sum(), cwd_an=np.average(x.cwd_gs_an, weights=x.n1), dD=(sv.s * sv.gDo).sum() / (sv.s * sv.gDs).sum(), mortOE=x.dead_o.sum() / x.dead_e.sum()))
for set_ in ["train", "val"]:
  s = d[d.set == set_]
  print(f"===== {set_}: wet/mid/dry terciles of CWD_gs anomaly WITHIN each period"); print(s.groupby(["per", "wet"], observed=True).apply(r, include_groups=False).round(3).to_string())
  print("pooled over periods"); print(s.groupby("wet", observed=True).apply(r, include_groups=False).round(3).to_string())
