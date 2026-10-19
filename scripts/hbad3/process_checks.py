"""HBAD3 process-grounding checks per REGION, from the data alone (no model fit): is each process of the model visible in the remeasurements of
regions other than GA+FL? Tracked trees of undisturbed intervals (Pan R4 interval product, results/hbad2_r4/*.arrow); sites = FIA subplots.
  P1 light competition: relative diameter growth (÷ the mean of open-grown trees of the same species-key × diameter class) by tree-level overtopping
     O = (SDI of taller trees + ½ own) / 1500 (SDI per ha of the subplot) — saturating? fitted e^{-aO} vs L(1+K)/(L+K), L = e^{-kO}
  P2 crowding: relative growth of open-grown trees (O < 0.1) by subplot RD = ΣSDI / 1500
  P3 height from diameter: sd of ln(H2pred/H2obs) of survivors: no growth | H–D allometry only | allometry + shade elongation/relaxation (4 globals)
  P4 growth-dependent mortality: death in interval k+1 by relative growth in interval k (trees measured on 3 visits)
  P5 drought-anomaly mortality: annual mortality by within-period tercile of the growing-season CWD anomaly
  python scripts/hbad3/process_checks.py <repo>"""
import sys, numpy as np, pandas as pd, pyarrow.ipc as ip
from scipy.optimize import least_squares, minimize
R = sys.argv[1]
iv = ip.open_file(f"{R}/results/hbad2_r4/intervals.arrow").read_all().to_pandas()
tr = ip.open_file(f"{R}/results/hbad2_r4/interval_trees.arrow").read_all().to_pandas()
ic = pd.read_csv(f"{R}/results/hbad3/interval_climate_r4.csv", usecols=["iid", "cwd_gs_an"])
iv = iv.merge(ic, on="iid", how="left")
iv["d1"] = pd.to_datetime(iv.d1); iv["d2"] = pd.to_datetime(iv.d2)
def region(r):
  if r.st in (12, 13): return "GAFL"
  if r.l3.startswith("8.4."): return "APP"
  if r.l3 == "8.3.3": return "IP"
  if r.st in (1, 28) and r.l3 in ("8.3.5", "8.3.6", "8.5.2"): return "WSP"
  if r.st in (37, 45) and r.l3 in ("8.3.4", "8.3.5", "8.5.1"): return "NCSC"
  return None
iv["reg"] = [region(r) for r in iv.itertuples()]
clean = iv[~(iv.trt | iv.dst | iv.harv) & (iv.dt > 2) & iv.reg.notna()]
t = tr.merge(clean[["iid", "st", "u", "c", "p", "k", "dt", "reg", "planted", "d1", "cwd_gs_an"]], on="iid")
t = t[(t.dia_cm > 0) & (t.ht_m > 0)].copy()
t["w"] = t.w_ha * 4.0; t["sdi"] = t.w * (t.dia_cm / 25.4) ** 1.605
t["site"] = t.iid * 10 + t.sb
# species keys per region: own if >= 2000 trees in the region, else _S/_H
cnt = t.groupby(["reg", "sym"]).size()
t["key"] = [s if cnt.get((r, s), 0) >= 2000 else "_" + g for r, s, g in zip(t.reg, t.sym, t.grp)]
# tree-level overtopping (taller trees of the same subplot) and subplot RD
t = t.sort_values(["site", "ht_m"], ascending=[True, False])
t["sdi_above"] = t.groupby("site").sdi.cumsum() - t.sdi            # strictly taller first (ties ordered arbitrarily)
t["O"] = (t.sdi_above + 0.5 * t.sdi) / 1500; t["RD"] = t.groupby("site").sdi.transform("sum") / 1500
t["Dc"] = pd.cut(t.dia_cm, [0, 5, 12.7, 25, 40, 1e9], labels=False)
al = t[(t.out == "alive") & (t.dia2_cm > 0) & (t.ht2_m > 0) & (~t.planted)].copy()
al["gD"] = (al.dia2_cm - al.dia_cm) / al.dt
al = al[al.gD.between(-0.5, 3.0)]
out = []
print("trees (alive survivors, natural) per region:", al.groupby("reg").size().to_dict())
# ---------------- P1 / P2
ref = al[al.O < 0.1].groupby(["reg", "key", "Dc"]).gD.mean().rename("g0")
al = al.join(ref, on=["reg", "key", "Dc"]); al = al[al.g0 > 0.05]
al["rel"] = al.gD / al.g0
oc = [-1, 0.1, 0.3, 0.6, 1.0, 1.6, 1e9]; ocl = ["<.1", ".1-.3", ".3-.6", ".6-1", "1-1.6", ">1.6"]
al["Oc"] = pd.cut(al.O, oc, labels=ocl)
p1 = al.groupby(["reg", "Oc"], observed=True).agg(n=("rel", "size"), rel=("rel", "mean"), Om=("O", "mean")).reset_index()
print("\n== P1 light competition: mean relative diameter growth by overtopping class (1 = open-grown of same key and size)")
print(p1.pivot(index="reg", columns="Oc", values="rel")[ocl].round(3).to_string())
print(p1.pivot(index="reg", columns="Oc", values="n")[ocl].to_string())
print("fitted response to the class means (weighted): exponential e^{-aO} vs saturating L(1+K)/(L+K), L=e^{-kO}; weighted RMSE")
for rg, g in p1.groupby("reg"):
  O, y, w = g.Om.values, g.rel.values, np.sqrt(g.n.values)
  e1 = least_squares(lambda p: w * (np.exp(-p[0] * O) - y), [1.0], bounds=([0], [50]))
  e2 = least_squares(lambda p: w * (np.exp(-p[0] * O) * (1 + p[1]) / (np.exp(-p[0] * O) + p[1]) - y), [2.0, 0.5], bounds=([0, 1e-4], [50, 1e3]))
  rm = lambda r: np.sqrt(np.sum(r.fun ** 2) / np.sum(w ** 2))
  print(f"  {rg:5s} exp a={e1.x[0]:.2f} rmse {rm(e1):.3f} | saturating k={e2.x[0]:.2f} K={e2.x[1]:.3f} rmse {rm(e2):.3f} | floor at O=2: exp {np.exp(-2*e1.x[0]):.3f} sat {np.exp(-2*e2.x[0])*(1+e2.x[1])/(np.exp(-2*e2.x[0])+e2.x[1]):.3f}")
rc = [0, 0.25, 0.5, 0.75, 1.0, 1e9]; rcl = ["<.25", ".25-.5", ".5-.75", ".75-1", ">1"]
op = al[al.O < 0.1].copy(); op["RDc"] = pd.cut(op.RD, rc, labels=rcl)
print("\n== P2 crowding: relative growth of OPEN-GROWN trees (O < 0.1) by subplot RD")
print(op.groupby(["reg", "RDc"], observed=True).rel.mean().unstack()[rcl].round(3).to_string())
# ---------------- P3 height from diameter
print("\n== P3 height from diameter (survivors): sd ln(H2pred / H2obs)   [none | allometry | allometry + 4 globals]  (fit on 50 % of plots, scored on the other 50 %)")
for rg, g in al.groupby("reg"):
  g = g[(g.ht2_m > 1.37)].copy()
  hd = {}
  for k, h in g.groupby("key"):
    D, H = h.dia_cm.values, h.ht_m.values
    f = lambda p: np.log(1.37 + np.exp(p[0]) * (1 - np.exp(-np.exp(p[1]) * D)) ** np.exp(p[2])) - np.log(H)
    hd[k] = np.exp(least_squares(f, [np.log(30), np.log(0.03), 0.0]).x)
  Hs = lambda k, D: 1.37 + hd[k][0] * (1 - np.exp(-hd[k][1] * D)) ** hd[k][2]
  g["Hs1"] = [Hs(k, D) for k, D in zip(g.key, g.dia_cm)]; g["Hs2"] = [Hs(k, D) for k, D in zip(g.key, g.dia2_cm)]
  half = (g.p * 7 + g.c) % 2 == 0
  def pred(p, z):
    eta, lam, k, mu = p; L = np.exp(-k * z.O)
    return z.ht_m + (z.Hs2 - z.Hs1) * (1 + eta * (1 - L)) + lam * z.dt * (z.Hs1 * (1 + mu * (1 - L)) - z.ht_m)
  loss = lambda p, z: np.mean(np.log(np.maximum(pred(p, z), 0.5) / z.ht2_m) ** 2)
  r = minimize(loss, [0.3, 0.02, 2.0, 0.3], args=(g[half],), method="Nelder-Mead", options=dict(maxiter=3000))
  v = g[~half]; sd = lambda e: np.sqrt(np.mean(e ** 2))
  print(f"  {rg:5s} n {len(v):7d} | none {sd(np.log(v.ht_m / v.ht2_m)):.4f} | allometry {sd(np.log(v.Hs2 * v.ht_m / v.Hs1 / v.ht2_m)):.4f} | +4 globals {sd(np.log(np.maximum(pred(r.x, v), .5) / v.ht2_m)):.4f}"
        f"  (eta {r.x[0]:.2f} lam {r.x[1]:.4f} k {r.x[2]:.2f} mu {r.x[3]:.2f})")
# ---------------- P4 growth-dependent mortality (tree measured at v1 of interval k and k+1 → its growth in k and fate in k+1)
nx = clean[["st", "u", "c", "p", "k", "iid"]].rename(columns={"iid": "iid_next"}); nx["k"] -= 1
a = al[["iid", "st", "u", "c", "p", "k", "sb", "tr", "reg", "key", "rel", "dia2_cm"]].merge(nx, on=["st", "u", "c", "p", "k"])
f = tr[["iid", "sb", "tr", "out"]].rename(columns={"iid": "iid_next", "out": "out_next"})
a = a.merge(f, on=["iid_next", "sb", "tr"]); a = a[a.out_next.isin(["alive", "dead"])]
a = a.merge(clean[["iid", "dt"]].rename(columns={"iid": "iid_next", "dt": "dt_next"}), on="iid_next")
a["dead"] = a.out_next == "dead"
a["gq"] = a.groupby("reg").rel.transform(lambda x: pd.qcut(x, 5, labels=False, duplicates="drop"))
p4 = a.groupby(["reg", "gq"]).apply(lambda x: pd.Series({"n": len(x), "ann_mort_pct": 100 * (1 - (1 - x.dead.mean()) ** (1 / x.dt_next.mean()))}), include_groups=False).reset_index()
print("\n== P4 growth-dependent mortality: annual mortality % in interval k+1 by quintile of relative growth in interval k (0 = slowest)")
print(p4.pivot(index="reg", columns="gq", values="ann_mort_pct").round(2).to_string())
print(p4.pivot(index="reg", columns="gq", values="n").to_string())
# ---------------- P5 drought-anomaly mortality
m = t[t.out.isin(["alive", "dead"]) & (~t.planted)].copy(); m["dead"] = m.out == "dead"
m["mid"] = m.d1.dt.year + m.dt / 2; m["per"] = pd.cut(m.mid, [0, 2006, 2010, 2014, 2030], labels=False)
m = m[m.cwd_gs_an.notna()]
m["wq"] = m.groupby(["reg", "per"]).cwd_gs_an.transform(lambda x: pd.qcut(x.rank(method="first"), 3, labels=["wet", "mid", "dry"]))
p5 = m.groupby(["reg", "wq"], observed=True).apply(lambda x: 100 * (1 - (1 - x.dead.mean()) ** (1 / x.dt.mean())), include_groups=False).unstack()
print("\n== P5 drought-anomaly mortality: annual mortality % by within-(region, period) tercile of growing-season CWD anomaly")
print(p5[["wet", "mid", "dry"]].round(3).to_string())
# ---------------- P6 recruitment: sapling ingrowth (first tallied, D < 12.7 cm; microplot) per subplot-year and species key, by seed source and light
ig = ip.open_file(f"{R}/results/hbad2_r4/interval_ingrowth.arrow").read_all().to_pandas()
ig = ig.merge(clean[["iid", "reg", "dt", "planted"]], on="iid"); ig = ig[(ig.dia_cm < 12.7) & (~ig.planted)]
ig["key"] = [s if cnt.get((r, s), 0) >= 2000 else "_" + g for r, s, g in zip(ig.reg, ig.sym, ig.grp)]
ig["site"] = ig.iid * 10 + ig.sb
nat = t[~t.planted]
st = nat.groupby("site").agg(reg=("reg", "first"), dt=("dt", "first"), SDI=("sdi", "sum")).reset_index()
nat_big = nat[nat.dia_cm >= 12.7].assign(ba=lambda x: np.pi / 4 * (x.dia_cm / 100) ** 2 * x.w)
basp = nat_big.groupby(["site", "key"]).ba.sum()
keys_top = nat.groupby(["reg", "key"]).size().reset_index().sort_values(0, ascending=False).groupby("reg").head(8)
rows = []
for rg, kk in keys_top.groupby("reg"):
  s = st[st.reg == rg]
  for k in kk.key:
    b = basp.xs(k, level="key") if k in basp.index.get_level_values("key") else pd.Series(dtype=float)
    y = ig[(ig.reg == rg) & (ig.key == k)].groupby("site").size()
    rows.append(pd.DataFrame({"reg": rg, "key": k, "site": s.site.values, "dt": s.dt.values, "RD": s.SDI.values / 1500,
                              "BA": b.reindex(s.site.values).fillna(0).values, "y": y.reindex(s.site.values).fillna(0).values}))
rc6 = pd.concat(rows)
rc6["seed"] = pd.cut(rc6.BA, [-1, 0, 2, 6, 1e9], labels=["0", "0-2", "2-6", ">6"]); rc6["RDc"] = pd.cut(rc6.RD, [-1, 0.25, 0.5, 0.75, 1e9], labels=["<.25", ".25-.5", ".5-.75", ">.75"])
rate = lambda x: x.y.sum() / x.dt.sum()
print("\n== P6 recruitment: ingrowth saplings per subplot-year, top-8 keys per region pooled — by conspecific seed-source BA (m²/ha, trees ≥ 12.7 cm)")
print(rc6.groupby(["reg", "seed"], observed=True).apply(rate, include_groups=False).unstack().round(4).to_string())
print("by subplot RD (light), sites WITH a conspecific seed source")
print(rc6[rc6.BA > 0].groupby(["reg", "RDc"], observed=True).apply(rate, include_groups=False).unstack().round(4).to_string())
