"""HBAD3 S1b: process water index per plot and interval — monthly soil bucket (capacity = plot AWC from SOLUS/Saxton-Rawls, plot_soil_r4.csv),
driven by PRISM L4 monthly ppt and Thornthwaite PET (same as climate_join.py). Each month: W += P; AET = min(PET, W·(1 - exp(-PET/AWC))) …
simplified to the standard 'supply-limited' form AET = min(PET, W); W = min(AWC, W - AET) (no runoff before capacity). Spun up 1980-1989.
Index: growing-season (Apr-Oct) ratio R = ΣAET / ΣPET per year (1 = no water limitation); interval value = mean over its years; normal = 1991-2020.
Output results/hbad3/interval_water{SUF}.csv (iid, aetpet_iv, aetpet_n)."""
import sys, numpy as np, pandas as pd, duckdb, pyarrow.ipc as ip
R = sys.argv[1]
IVP = sys.argv[2] if len(sys.argv) > 2 else f"{R}/results/hbad2/intervals.arrow"; SUF = sys.argv[3] if len(sys.argv) > 3 else ""
pc = pd.read_csv(f"{R}/results/hbad3/plot_climate{SUF}.csv"); ic = pd.read_csv(f"{R}/results/hbad3/interval_climate{SUF}.csv")
iv = ip.open_file(IVP).read_all().to_pandas()[["iid", "d1", "d2"]]
iv["y1"] = pd.to_datetime(iv.d1).dt.year; iv["y2"] = np.maximum(pd.to_datetime(iv.d2).dt.year - 1, iv.y1)
pr = duckdb.connect("/workspace/PRISM_LT/PRISM_LT_EPA/prism_lt_epa.duckdb", read_only=True)
ids = sorted(pc.l4.dropna().unique())
m = pr.execute("select EPA_L4_ID l4, YEAR y, MONTH mo, CAST(CAST(ppt_mean AS VARCHAR) AS DOUBLE) ppt, CAST(CAST(tmean_mean AS VARCHAR) AS DOUBLE) tmean from EPA_L4_monthly where EPA_L4_ID in (select unnest(?)) and YEAR >= 1980 order by l4, y, mo", [ids]).df()
lat = pc.groupby("l4").lat.mean()
J = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]); dec = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39); days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
pet_by = {}
for (l, y), g in m.groupby(["l4", "y"]):
  if len(g) < 12: continue
  t = np.clip(g.tmean.values, 0, None); I = ((t / 5) ** 1.514).sum(); a = 6.75e-7 * I**3 - 7.71e-5 * I**2 + 1.792e-2 * I + 0.49239
  ws = np.arccos(np.clip(-np.tan(np.radians(lat[l])) * np.tan(dec), -1, 1)); N = 24 / np.pi * ws
  pet_by[(l, y)] = (g.ppt.values, np.where(t > 0, 16 * (10 * t / max(I, 1e-9)) ** a, 0) * N / 12 * days / 30)
years = sorted({y for (_, y) in pet_by})
gs = slice(3, 10)
rows = []
for r in pc.itertuples():
  awc = max(r.awc_mm, 5.0); W = awc; ratio = {}
  for y in years:
    if (r.l4, y) not in pet_by: continue
    P, E = pet_by[(r.l4, y)]; aet = np.zeros(12)
    for k in range(12):
      W += P[k]; aet[k] = min(E[k], W); W = min(awc, W - aet[k])
    ratio[y] = aet[gs].sum() / max(E[gs].sum(), 1e-9)
  rows.append((r.st, r.u, r.c, r.p, ratio))
plot_ratio = {(a, b, c, d): rt for a, b, c, d, rt in rows}
ivp = iv.merge(ic[["iid", "st", "u", "c", "p"]], on="iid")
out = []
for r in ivp.itertuples():
  rt = plot_ratio[(r.st, r.u, r.c, r.p)]
  vals = [rt[y] for y in range(r.y1, r.y2 + 1) if y in rt]; nv = [rt[y] for y in range(1991, 2021) if y in rt]
  out.append(dict(iid=r.iid, aetpet_iv=np.mean(vals) if vals else np.nan, aetpet_n=np.mean(nv), aetpet_min_iv=np.min(vals) if vals else np.nan))
O = pd.DataFrame(out); O.to_csv(f"{R}/results/hbad3/interval_water{SUF}.csv", index=False)
print(O.describe().T)
