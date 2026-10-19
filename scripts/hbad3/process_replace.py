"""HBAD3 process-replacement checks per region (data only; tracked survivors / deaths of undisturbed natural intervals, R4 product).
R1 growth: can the empirical potential-growth curve r (D/20)^c e^{-D/Dmax} (3 per key) be replaced by a stem-volume balance
     dV/dt = ε LA f(L) − ρ V,   LA = D^1.605 (pipe-model leaf area ∝ Reineke size), V = D² H (stem volume up to a form factor), f(L) = e^{-a O}
   with ε per key and ρ, a global?  dD follows from dV with H fixed over the interval: dD = dV / (2 D H)  (+ height growth neglected in the inversion).
   Compared on held-out plots (50 / 50): sd of ln(D2pred / D2obs) of survivors, and the shape of the growth-size curve.
R2 mortality: can the empirical U-shaped background hazard exp(mbg + mD ln(D/20) + mD2 D/20) (2 per key + 1 global) be replaced by
     h = h0(longevity) + ms (stress from relative growth) + mH (H / Hmax_key)^q   (intrinsic + carbon starvation + large-tree risk)
   with h0 = m0 / LONGEVITY (m0 global; Botkin's 4.605 = 1 % reach the longevity), relative growth from the previous interval (P4), and globals ms, β, mH, q?
   Compared on held-out plots: binomial deviance per tree, and the size pattern of mortality.
  python scripts/hbad3/process_replace.py <repo>"""
import sys, numpy as np, pandas as pd, pyarrow.ipc as ip
from scipy.optimize import minimize
R = sys.argv[1]
iv = ip.open_file(f"{R}/results/hbad2_r4/intervals.arrow").read_all().to_pandas()
tr = ip.open_file(f"{R}/results/hbad2_r4/interval_trees.arrow").read_all().to_pandas()
def region(r):
  if r.st in (12, 13): return "GAFL"
  if r.l3.startswith("8.4."): return "APP"
  if r.l3 == "8.3.3": return "IP"
  if r.st in (1, 28) and r.l3 in ("8.3.5", "8.3.6", "8.5.2"): return "WSP"
  if r.st in (37, 45) and r.l3 in ("8.3.4", "8.3.5", "8.5.1"): return "NCSC"
  return None
iv["reg"] = [region(r) for r in iv.itertuples()]
clean = iv[~(iv.trt | iv.dst | iv.harv) & (iv.dt > 2) & iv.reg.notna() & ~iv.planted]
t = tr.merge(clean[["iid", "st", "u", "c", "p", "k", "dt", "reg"]], on="iid")
t = t[(t.dia_cm > 0) & (t.ht_m > 1.37) & t.out.isin(["alive", "dead"])].copy()
t["w"] = t.w_ha * 4.0; t["sdi"] = t.w * (t.dia_cm / 25.4) ** 1.605; t["site"] = t.iid * 10 + t.sb
cnt = t.groupby(["reg", "sym"]).size()
t["key"] = [s if cnt.get((r, s), 0) >= 2000 else "_" + g for r, s, g in zip(t.reg, t.sym, t.grp)]
t = t.sort_values(["site", "ht_m"], ascending=[True, False])
t["O"] = (t.groupby("site").sdi.cumsum() - 0.5 * t.sdi) / 1500
t["half"] = (t.p * 7 + t.c) % 2 == 0
stc = pd.read_csv("/workspace/landis_region4/data_r4/shadetol_all_species.csv"); std_ = dict(zip(stc.sym, stc.shade_class))
t["ST"] = [std_.get(x, 3.0) for x in t.sym]
t["Hmx"] = t.groupby(["reg", "key"]).ht_m.transform(lambda x: x.quantile(0.995))
lon = pd.read_csv("/workspace/species_longevity_ref.csv"); lond = dict(zip(lon.species_symbol, lon.longevity_yr))
t["L"] = [lond.get(s, 300.0) for s in t.sym]
import os
RUN_R1 = os.environ.get("RUN_R1", "1") == "1"
# ---------------- R1 growth
s = t[(t.out == "alive") & (t.dia2_cm > 0)].copy(); s["gD"] = (s.dia2_cm - s.dia_cm) / s.dt; s = s[s.gD.between(-0.5, 3)]
print("== R1 potential growth: empirical power-exponential (3/key + light a) vs stem-volume balance (1/key: ε; globals ρ, a)")
print("   held-out sd of ln(D2pred/D2obs) | growth (cm/yr) by D class obs / emp / vol")
for rg, g in (s.groupby("reg") if RUN_R1 else []):
  keys = sorted(g.key.unique()); ki = {k: i for i, k in enumerate(keys)}; kk = g.key.map(ki).values
  D, H, O, dt, D2 = g.dia_cm.values, g.ht_m.values, g.O.values, g.dt.values, g.dia2_cm.values
  tr_ = g.half.values; nk = len(keys)
  def emp(p):        # p: per key ln r, c, ln Dmax (3 nk) + a
    r, c, Dm = np.exp(p[:nk])[kk], p[nk:2*nk][kk], np.exp(p[2*nk:3*nk])[kk]; a = np.exp(p[-1])
    return r * (D / 20) ** c * np.exp(-D / Dm) * np.exp(-a * O)
  def vol(p):        # p: per key ln ε (nk) + ln ρ + ln a + b (leaf-area exponent, pipe model; 1.605 = Reineke)
    e, rho, a, b = np.exp(p[:nk])[kk], np.exp(p[nk]), np.exp(p[nk+1]), p[nk+2]
    dV = e * (D / 20) ** b * 20 ** 1.605 * np.exp(-a * O) - rho * D ** 2 * H          # [cm² m / yr] up to the form factor
    return dV / (2 * D * H)
  Hx = g.Hmx.values
  def volh(p):       # + hydraulic limitation of tall trees: × max(0, 1 − (H/Hmax)^θ), θ global (p[-1] = ln θ)
    e, a, b, th = np.exp(p[:nk])[kk], np.exp(p[nk]), p[nk+1], np.exp(p[nk+2])
    return e * (D / 20) ** b * 20 ** 1.605 * np.exp(-a * O) * np.maximum(1 - np.minimum(H / (1.1 * Hx), 1) ** th, 0) / (2 * D * H)
  def loss(f, p, m):
    d2 = D[m] + np.maximum(f(p)[m], -0.2) * dt[m]
    return np.mean(np.log(np.maximum(d2, 0.1) / D2[m]) ** 2)
  pe = np.concatenate([np.full(nk, np.log(0.4)), np.full(nk, 0.4), np.full(nk, np.log(40)), [np.log(2)]])
  pv = np.concatenate([np.full(nk, np.log(2.0)), [np.log(0.002)], [np.log(2)], [1.605]])
  re = minimize(lambda p: loss(emp, p, tr_), pe, method="L-BFGS-B"); rv = minimize(lambda p: loss(vol, p, tr_), pv, method="L-BFGS-B")
  rh = minimize(lambda p: loss(volh, p, tr_), np.concatenate([rv.x[:nk] + 0.3, [rv.x[nk+1], rv.x[nk+2]], [np.log(3.0)]]), method="L-BFGS-B"); th_ = np.sqrt(loss(volh, rh.x, ~tr_))
  te = np.sqrt(loss(emp, re.x, ~tr_)); tv = np.sqrt(loss(vol, rv.x, ~tr_)); t0 = np.sqrt(np.mean(np.log(D[~tr_] / D2[~tr_]) ** 2))
  dc = pd.cut(D, [0, 5, 12.7, 25, 40, 1e9], labels=["<5", "5-13", "13-25", "25-40", ">40"])
  tab = pd.DataFrame({"dc": dc, "o": g.gD.values, "e": emp(re.x), "v": vol(rv.x), "h": volh(rh.x)})[~tr_].groupby("dc", observed=True).mean()
  print(f"  {rg:5s} keys {nk:2d} | none {t0:.4f} emp({3*nk+1}) {te:.4f} vol({nk+3}) {tv:.4f} hyd({nk+3}) {th_:.4f} | b {rh.x[nk+1]:.2f} θ {np.exp(rh.x[nk+2]):.2f} | obs/emp/vol/hyd "
        + " ".join(f"{i}:{r.o:.3f}/{r.e:.3f}/{r.v:.3f}/{r.h:.3f}" for i, r in tab.iterrows()))
# ---------------- R2 mortality (trees with growth in the previous interval: P4 linkage)
print("\n== R2 mortality: empirical U-shape (2/key + 1) + growth stress vs process (intrinsic 4.605/LONGEVITY + growth stress + height risk; 4 globals)")
print("   held-out binomial deviance per tree (lower better) | annual mortality % by D class obs / emp / proc")
nx = clean[["st", "u", "c", "p", "k", "iid"]].rename(columns={"iid": "iid_prev"}); nx["k"] += 1
prev = tr[["iid", "sb", "tr", "dia_cm", "dia2_cm"]].rename(columns={"iid": "iid_prev", "dia_cm": "d0", "dia2_cm": "d1"})
m = t.merge(nx, on=["st", "u", "c", "p", "k"]).merge(prev, on=["iid_prev", "sb", "tr"])
m = m.merge(clean[["iid", "dt"]].rename(columns={"iid": "iid_prev", "dt": "dt_prev"}), on="iid_prev")
m = m[m.d1 > 0]; m["gprev"] = (m.d1 - m.d0) / m.dt_prev
ref = m.groupby(["reg", "key", pd.cut(m.dia_cm, [0, 5, 12.7, 25, 40, 1e9])], observed=True).gprev.transform(lambda x: x.clip(lower=0).mean())
m["grel"] = (m.gprev.clip(lower=0) / ref).clip(upper=3).fillna(1.0)
m["Hmax"] = m.groupby(["reg", "key"]).ht_m.transform(lambda x: x.quantile(0.99))
m["dead"] = (m.out == "dead").astype(float)
for rg, g in m.groupby("reg"):
  keys = sorted(g.key.unique()); ki = {k: i for i, k in enumerate(keys)}; kk = g.key.map(ki).values; nk = len(keys)
  D, dt, y, gr, Hr, L = g.dia_cm.values, g.dt.values, g.dead.values, g.grel.values, (g.ht_m / g.Hmax).values, g.L.values
  stz = (g.ST.values - 3) / 1.0
  trm = g.half.values
  def prob(h): return np.clip(1 - np.exp(-h * dt), 1e-9, 1 - 1e-9)
  def h_emp(p):      # per key mbg, mD (2 nk) + mD2 + stress ms, β
    return np.exp(p[:nk][kk] + p[nk:2*nk][kk] * np.log(D / 20) + p[2*nk] * D / 20) + np.exp(p[2*nk+1]) * np.exp(-np.exp(p[2*nk+2]) * gr)
  def h_proc(p):     # intrinsic ∝ 1/LONGEVITY (global scale m0) + stress + height risk: globals ln ms, ln β, ln mH, ln q, ln m0
    return np.exp(p[4]) / L + np.exp(p[0]) * np.exp(-np.exp(p[1]) * gr) * (D / 20) ** p[5] + np.exp(p[2]) * Hr ** np.exp(p[3])
  def nll(f, p, msk):
    pr = prob(f(p))[msk]; return -np.mean(y[msk] * np.log(pr) + (1 - y[msk]) * np.log(1 - pr))
  pe = np.concatenate([np.full(nk, -5.0), np.full(nk, -0.5), [0.2, np.log(0.05), np.log(2.0)]]); pp = np.array([np.log(0.05), np.log(2.0), np.log(0.01), np.log(3.0), np.log(1.0), -0.5])   # + ζ: stress × (D/20)^ζ (reserve-limited survival of small trees)
  def h_proc2(p):    # + shade tolerance: tolerant species survive low growth (Kobe): stress × e^{σST·(ST−3)}, sensitivity β × e^{βST·(ST−3)}
    return np.exp(p[4]) / L + np.exp(p[0] + p[6] * stz) * np.exp(-np.exp(p[1] + p[7] * stz) * gr) * (D / 20) ** p[5] + np.exp(p[2]) * Hr ** np.exp(p[3])
  re = minimize(lambda p: nll(h_emp, p, trm), pe, method="L-BFGS-B"); rp = minimize(lambda p: nll(h_proc, p, trm), pp, method="L-BFGS-B")
  rq = minimize(lambda p: nll(h_proc2, p, trm), np.concatenate([rp.x, [0.0, 0.0]]), method="L-BFGS-B")
  ann = lambda h: 100 * (1 - np.exp(-h))
  dc = pd.cut(D, [0, 5, 12.7, 25, 40, 1e9], labels=["<5", "5-13", "13-25", "25-40", ">40"])
  tab = pd.DataFrame({"dc": dc, "y": y, "dt": dt, "e": h_emp(re.x), "q": h_proc(rp.x)})[~trm].groupby("dc", observed=True).apply(
    lambda x: pd.Series({"o": 100 * (1 - (1 - x.y.mean()) ** (1 / x.dt.mean())), "e": ann(x.e.mean()), "q": ann(x.q.mean())}), include_groups=False)
  null = nll(lambda p: np.full(len(y), p[0]), [y[trm].mean() / dt[trm].mean()], ~trm)
  print(f"  {rg:5s} n {len(y):6d} | null {null:.5f} emp({2*nk+3}) {nll(h_emp, re.x, ~trm):.5f} proc(6) {nll(h_proc, rp.x, ~trm):.5f} +ST(8) {nll(h_proc2, rq.x, ~trm):.5f} [σST {rq.x[6]:.2f} βST {rq.x[7]:.2f}] | ζ {rp.x[5]:.2f} m0 {np.exp(rp.x[4]):.2f} ms {np.exp(rp.x[0]):.3f} β {np.exp(rp.x[1]):.2f} mH {np.exp(rp.x[2]):.4f} q {np.exp(rp.x[3]):.2f} | "
        + " ".join(f"{i}:{r.o:.2f}/{r.e:.2f}/{r.q:.2f}" for i, r in tab.iterrows()))
