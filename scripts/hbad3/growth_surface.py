"""HBAD3 S2d: observed vs simulated diameter / height increment of all-survivor cohorts by overtopping O x stand RD (start of interval).
Shows the functional form the data imply for light (O) and crowding (RD) competition and how a fitted v2 model reproduces it."""
import sys, pandas as pd
R = sys.argv[1]; tag = sys.argv[2]
d = pd.read_csv(f"{R}/results/hbad3/resid_{tag}.csv"); d["gDo"] = (d.D2o - d.D1) / d.dt; d["gDs"] = (d.D2s - d.D1) / d.dt; d["gHo"] = (d.H2o - d.H1) / d.dt; d["gHs"] = (d.H2s - d.H1) / d.dt
d["RDc"] = pd.cut(d.RD, [0, .25, .5, .75, 10]); d["Oc"] = pd.cut(d.O, [-1, .1, .3, .6, 10])
x = d[(d.s > 0) & (d.s == d.n1)]
for v in ["gDo", "gDs", "gHo", "gHs"]:
  print(f"-- {tag}: mean annual {v} (survivor-weighted), rows O, cols RD")
  print(x.groupby(["Oc", "RDc"], observed=True).apply(lambda y: (y.s * y[v]).sum() / y.s.sum(), include_groups=False).unstack().round(3).to_string())
