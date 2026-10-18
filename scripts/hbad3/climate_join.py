"""HBAD3 design study, step S1: climate + soil covariates per FIA plot and per remeasurement interval.
Plot -> EPA L4 (curated_cohorts_landis_stdorg.epa_l4, slim_r4.duckdb, read-only) -> PRISM_LT_EPA L4 monthly (°C / mm / hPa; BLOB text).
Normals 1991-2020: tmean, Jan tmin, Jul tmax, annual ppt, JJA vpdmax, Thornthwaite PET and climatic water deficit CWD = Σ max(0, PET - ppt).
Interval values: means over the calendar years spanned by the interval [year(d1), year(d2)-1] and anomalies vs the normals.
Soil: plot_soil_r4.csv (awc_mm). Output results/hbad3/{plot_climate,interval_climate}.csv."""
import sys, duckdb, numpy as np, pandas as pd, pyarrow.ipc as ip
R = sys.argv[1]  # repo root
iv = ip.open_file(f"{R}/results/hbad2/intervals.arrow").read_all().to_pandas()
slim = duckdb.connect("/workspace/landis_region4/data_r4/slim_r4.duckdb", read_only=True)
l4 = slim.execute("select statecd st, unitcd u, countycd c, plot p, any_value(epa_l4) l4, count(distinct epa_l4) n from curated_cohorts_landis_stdorg group by all").df()
assert (l4.n == 1).all(), "plot with several L4"
soil = pd.read_csv("/workspace/landis_region4/data_r4/plot_soil_r4.csv").rename(columns={"statecd": "st", "unitcd": "u", "countycd": "c", "plot": "p"})
pc = iv[["st", "u", "c", "p"]].drop_duplicates().merge(l4[["st", "u", "c", "p", "l4"]], how="left").merge(soil[["st", "u", "c", "p", "lat", "lon", "awc_mm"]], how="left")
print("plots", len(pc), "with L4", pc.l4.notna().mean(), "with soil", pc.awc_mm.notna().mean())
pr = duckdb.connect("/workspace/PRISM_LT/PRISM_LT_EPA/prism_lt_epa.duckdb", read_only=True)
cols = ["ppt", "tmin", "tmean", "tmax", "vpdmax"]
sel = ", ".join(f"CAST(CAST({c}_mean AS VARCHAR) AS DOUBLE) {c}" for c in cols)
ids = sorted(pc.l4.dropna().unique())
m = pr.execute(f"select EPA_L4_ID l4, YEAR y, MONTH mo, {sel} from EPA_L4_monthly where EPA_L4_ID in (select unnest(?)) and YEAR >= 1980", [ids]).df()
lat = pc.groupby("l4").lat.mean()
def thornthwaite(g):  # monthly PET mm, Thornthwaite with daylength correction at the L4 mean latitude
  t = np.clip(g.tmean.values, 0, None); I = (np.clip(g.groupby('y').tmean.transform(lambda x: x), 0, None) / 5) ** 1.514
  return t
# per (l4, y): heat index from that year's months
def pet_year(df, phi):
  df = df.sort_values("mo"); t = np.clip(df.tmean.values, 0, None); I = ((t / 5) ** 1.514).sum()
  a = 6.75e-7 * I**3 - 7.71e-5 * I**2 + 1.792e-2 * I + 0.49239
  J = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]); dec = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39)
  ws = np.arccos(np.clip(-np.tan(np.radians(phi)) * np.tan(dec), -1, 1)); N = 24 / np.pi * ws
  days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
  pet = np.where(t > 0, 16 * (10 * t / max(I, 1e-9)) ** a, 0) * N / 12 * days / 30
  return pet
rows = []
for (l, y), g in m.groupby(["l4", "y"]):
  if g.mo.nunique() < 12: continue
  g = g.sort_values("mo"); pet = pet_year(g, lat[l])
  rows.append(dict(l4=l, y=y, tmean=g.tmean.mean(), tmin_jan=g.tmin.values[0], tmax_jul=g.tmax.values[6], ppt=g.ppt.sum(),
                   vpd_jja=g.vpdmax.values[5:8].mean(), pet=pet.sum(), cwd=np.clip(pet - g.ppt.values, 0, None).sum(),
                   ppt_gs=g.ppt.values[3:9].sum(), cwd_gs=np.clip(pet[3:9] - g.ppt.values[3:9], 0, None).sum()))
Y = pd.DataFrame(rows); V = ["tmean", "tmin_jan", "tmax_jul", "ppt", "vpd_jja", "pet", "cwd", "ppt_gs", "cwd_gs"]
nrm = Y[(Y.y >= 1991) & (Y.y <= 2020)].groupby("l4")[V].mean().add_suffix("_n").reset_index()
pc = pc.merge(nrm, how="left"); pc.to_csv(f"{R}/results/hbad3/plot_climate.csv", index=False)
iv = iv.merge(pc[["st", "u", "c", "p", "l4"]], how="left")
Yi = Y.set_index(["l4", "y"])
out = []
for r in iv.itertuples():
  y1, y2 = r.d1.year, max(r.d2.year - 1, r.d1.year)
  if pd.isna(r.l4): out.append({}); continue
  sub = Yi.loc[(r.l4, slice(y1, y2)), V] if (r.l4, y1) in Yi.index else None
  out.append({} if sub is None or len(sub) == 0 else sub.mean().to_dict())
IC = pd.concat([iv[["iid", "st", "u", "c", "p", "l4"]].reset_index(drop=True), pd.DataFrame(out).add_suffix("_iv")], axis=1)
IC = IC.merge(nrm, how="left")
for v in V: IC[v + "_an"] = IC[v + "_iv"] - IC[v + "_n"]
IC.to_csv(f"{R}/results/hbad3/interval_climate.csv", index=False)
print(IC[[v + "_n" for v in V]].describe().T[["mean", "std", "min", "max"]])
print(IC[[v + "_an" for v in V]].describe().T[["mean", "std", "min", "max"]])
print("intervals with climate", IC.tmean_iv.notna().mean())
