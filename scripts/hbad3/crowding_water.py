"""HBAD3 S2c: is symmetric crowding (stand RD) a water-competition effect? If so, v2j (whose crowding response ignores water) should
under-predict the crowding penalty on growth in DRY conditions and over-predict it in WET ones: the observed/simulated diameter-increment
ratio should fall with RD more steeply when water is scarce. Cross-tables RD class x water class (soil AWC, normal CWD, interval CWD anomaly);
also survivors' height increment and mortality O/E."""
import sys, numpy as np, pandas as pd
R = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else "v2j"
d = pd.read_csv(f"{R}/results/hbad3/resid_{tag}.csv"); ic = pd.read_csv(f"{R}/results/hbad3/interval_climate.csv"); pc = pd.read_csv(f"{R}/results/hbad3/plot_climate.csv")
ic = ic.merge(pc[["st", "u", "c", "p", "awc_mm"]], how="left"); d = d.merge(ic, on="iid")
d["gDo"] = (d.D2o - d.D1) / d.dt; d["gDs"] = (d.D2s - d.D1) / d.dt; d["gHo"] = (d.H2o - d.H1) / d.dt; d["gHs"] = (d.H2s - d.H1) / d.dt
d["dead_o"] = d.n1 - d.s; d["dead_e"] = d.n1 * (1 - d.ps)
d["RDc"] = pd.cut(d.RD, [0, 0.25, 0.5, 0.75, 10], labels=["RD<.25", ".25-.5", ".5-.75", ">.75"])
def r(x):
  sv = x[x.s > 0]
  return pd.Series(dict(trees=x.n1.sum(), dD=(sv.s * sv.gDo).sum() / (sv.s * sv.gDs).sum(), dH=(sv.s * sv.gHo).sum() / (sv.s * sv.gHs).sum(), mOE=x.dead_o.sum() / x.dead_e.sum()))
for w in ["awc_mm", "cwd_n", "cwd_gs_an"]:
  d["Wc"] = pd.qcut(d[w], 3, labels=(["dry", "mid", "wet"] if w == "awc_mm" else ["wet", "mid", "dry"]))
  for set_ in ["train", "val"]:
    t = d[d.set == set_].groupby(["Wc", "RDc"], observed=True).apply(r, include_groups=False)
    print(f"===== {set_} water class by {w} x stand RD"); print(t.unstack("RDc").round(3).to_string())
