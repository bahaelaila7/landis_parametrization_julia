"""HBAD3 design study, step S2: is there climate signal in the residuals of the biomass-free v2 model (v2j)?
Per climate quintile: ratio observed / simulated diameter increment, height increment (ratio of sums over survivors, cm/yr and m/yr)
and observed / expected deaths. Normals = spatial gradient (confounded with region and species mix); anomalies = interval climate
departures (temporal, cleaner but small). Also within-L3 versions of the normals (L3-demeaned)."""
import sys, numpy as np, pandas as pd
R = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else "v2j"
d = pd.read_csv(f"{R}/results/hbad3/resid_{tag}.csv")
ic = pd.read_csv(f"{R}/results/hbad3/interval_climate.csv"); pc = pd.read_csv(f"{R}/results/hbad3/plot_climate.csv")
ic = ic.merge(pc[["st", "u", "c", "p", "awc_mm"]], how="left")
d = d.merge(ic, on="iid", how="left"); d["l3"] = d.l4.str.rsplit(".", n=1).str[0]
d["gDo"] = (d.D2o - d.D1) / d.dt; d["gDs"] = (d.D2s - d.D1) / d.dt; d["gHo"] = (d.H2o - d.H1) / d.dt; d["gHs"] = (d.H2s - d.H1) / d.dt
d["dead_o"] = d.n1 - d.s; d["dead_e"] = d.n1 * (1 - d.ps)
def table(df, col, q=5):
  df = df[df[col].notna()].copy(); df["q"] = pd.qcut(df[col], q, labels=False, duplicates="drop")
  sv = df[df.s > 0]
  g = sv.groupby("q").apply(lambda x: pd.Series(dict(lo=x[col].min(), hi=x[col].max(),
        dD=(x.s * x.gDo).sum() / (x.s * x.gDs).sum(), dH=(x.s * x.gHo).sum() / (x.s * x.gHs).sum())), include_groups=False)
  m = df.groupby("q").apply(lambda x: pd.Series(dict(trees=x.n1.sum(), mortOE=x.dead_o.sum() / x.dead_e.sum())), include_groups=False)
  return g.join(m)
for set_ in ["train", "val"]:
  s = d[d.set == set_]
  print(f"\n===== {set_}: {len(s)} cohorts, overall dD o/s {(s.s*s.gDo).sum()/(s.s*s.gDs).sum():.3f} dH {(s.s*s.gHo).sum()/(s.s*s.gHs).sum():.3f} mort O/E {s.dead_o.sum()/s.dead_e.sum():.3f}")
  for col in ["tmean_n", "cwd_n", "ppt_n", "vpd_jja_n", "awc_mm", "tmean_an", "cwd_an", "cwd_gs_an", "ppt_gs_an", "vpd_jja_an"]:
    print(f"-- {col}"); print(table(s, col).round(3).to_string())
  for col in ["tmean_n", "cwd_n"]:
    s = s.copy(); s[col + "_w"] = s[col] - s.groupby("l3")[col].transform("mean")
    print(f"-- {col} within L3"); print(table(s, col + "_w").round(3).to_string())
