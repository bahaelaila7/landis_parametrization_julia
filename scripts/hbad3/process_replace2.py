"""HBAD3 process-replacement checks, part 2 (data only, per region; held-out = the other half of the plots).
R3 recruitment: empirical per-key Poisson  E[n] = dt exp(ρ0_k + ρS_k ln(1 + BA_k,sub) + ρL_k RD)          (3 per key)
   vs process:  E[n] = dt F_k (BA_k,sub + δ BA_k,plot + ε0)^ρS exp(−λ_ST RD),  λ_ST = λ0 e^{λ1 (ST−3)}      (1 per key: fecundity F_k; globals δ ρS λ0 λ1 ε0)
   (seed rain from the subplot's own seed trees + dispersal from the rest of the plot + a background; light window narrowing with shade INtolerance)
R4 H–D allometry: per-key Chapman–Richards (3 per key) vs  H* = 1.37 + (Hmax_k − 1.37)(1 − exp(−s D / (Hmax_k − 1.37)))^c  with Hmax_k the key's
   99.5th-percentile height (a trait) and s (initial slenderness, m/cm), c global — scored as the height-growth rule of process_checks P3.
  python scripts/hbad3/process_replace2.py <repo>"""
import sys, numpy as np, pandas as pd, pyarrow.ipc as ip
from scipy.optimize import minimize, least_squares
R = sys.argv[1]
iv = ip.open_file(f"{R}/results/hbad2_r4/intervals.arrow").read_all().to_pandas()
tr = ip.open_file(f"{R}/results/hbad2_r4/interval_trees.arrow").read_all().to_pandas()
ig = ip.open_file(f"{R}/results/hbad2_r4/interval_ingrowth.arrow").read_all().to_pandas()
def region(r):
  if r.st in (12, 13): return "GAFL"
  if r.l3.startswith("8.4."): return "APP"
  if r.l3 == "8.3.3": return "IP"
  if r.st in (1, 28) and r.l3 in ("8.3.5", "8.3.6", "8.5.2"): return "WSP"
  if r.st in (37, 45) and r.l3 in ("8.3.4", "8.3.5", "8.5.1"): return "NCSC"
  return None
iv["reg"] = [region(r) for r in iv.itertuples()]
clean = iv[~(iv.trt | iv.dst | iv.harv) & (iv.dt > 2) & iv.reg.notna() & ~iv.planted]
t = tr.merge(clean[["iid", "st", "u", "c", "p", "dt", "reg"]], on="iid"); t = t[(t.dia_cm > 0) & (t.ht_m > 1.37)].copy()
t["w"] = t.w_ha * 4.0; t["sdi"] = t.w * (t.dia_cm / 25.4) ** 1.605; t["site"] = t.iid * 10 + t.sb
cnt = t.groupby(["reg", "sym"]).size()
keyf = lambda r, s, g: s if cnt.get((r, s), 0) >= 2000 else "_" + g
t["key"] = [keyf(r, s, g) for r, s, g in zip(t.reg, t.sym, t.grp)]
stc = pd.read_csv("/workspace/landis_region4/data_r4/shadetol_all_species.csv"); std_ = dict(zip(stc.sym, stc.shade_class))
t["ST"] = [std_.get(x, 3.0) for x in t.sym]
kst = t.groupby(["reg", "key"]).ST.mean()
# ---------------- R3 recruitment
ig = ig.merge(clean[["iid", "reg", "dt"]], on="iid"); ig = ig[ig.dia_cm < 12.7].copy()
ig["key"] = [keyf(r, s, g) for r, s, g in zip(ig.reg, ig.sym, ig.grp)]; ig["site"] = ig.iid * 10 + ig.sb
sites = t.groupby("site").agg(reg=("reg", "first"), iid=("iid", "first"), dt=("dt", "first"), SDI=("sdi", "sum"), p=("p", "first"), c=("c", "first")).reset_index()
big = t[t.dia_cm >= 12.7].assign(ba=lambda x: np.pi / 4 * (x.dia_cm / 100) ** 2 * x.w)
bsub = big.groupby(["site", "key"]).ba.sum(); bplot = big.groupby(["iid", "key"]).ba.sum() / 4
print("== R3 recruitment: held-out Poisson deviance per site-key (lower better)")
for rg, S in sites.groupby("reg"):
  keys = sorted(t[t.reg == rg].key.unique()); rows = []
  for k in keys:
    bs = bsub.xs(k, level="key").reindex(S.site.values).fillna(0).values if k in bsub.index.get_level_values(1) else np.zeros(len(S))
    bp = bplot.xs(k, level="key").reindex(S.iid.values).fillna(0).values if k in bplot.index.get_level_values(1) else np.zeros(len(S))
    y = ig[(ig.reg == rg) & (ig.key == k)].groupby("site").size().reindex(S.site.values).fillna(0).values
    rows.append(pd.DataFrame({"k": keys.index(k), "dt": S.dt.values, "RD": S.SDI.values / 1500, "bs": bs, "bp": np.maximum(bp - bs / 4, 0), "y": y,
                              "half": ((S.p.values * 7 + S.c.values) % 2 == 0), "st": kst.get((rg, k), 3.0)}))
  d = pd.concat(rows); kk = d.k.values; nk = len(keys); y = d.y.values; dt = d.dt.values; RD = d.RD.values; bs = d.bs.values; bp = d.bp.values; st = d.st.values - 3
  h = d.half.values
  def mu_emp(p): return dt * np.exp(p[:nk][kk] + p[nk:2*nk][kk] * np.log1p(bs) + p[2*nk:][kk] * RD)
  def mu_proc(p):  # p: ln F_k (nk), ln δ, ρS, ln λ0, λ1, ln ε0
    return dt * np.exp(p[:nk][kk]) * (bs + np.exp(p[nk]) * bp + np.exp(p[nk+4])) ** p[nk+1] * np.exp(-np.exp(p[nk+2] + p[nk+3] * st) * RD)
  def dev(mu, m): mu = np.maximum(mu[m], 1e-12); yy = y[m]; return 2 * np.mean(np.where(yy > 0, yy * np.log(yy / mu), 0) - (yy - mu))
  base = np.log(max(y[h].sum() / dt[h].sum() / nk, 1e-4))
  re = minimize(lambda p: dev(mu_emp(p), h), np.concatenate([np.full(nk, base), np.zeros(nk), np.zeros(nk)]), method="L-BFGS-B")
  rp = minimize(lambda p: dev(mu_proc(p), h), np.concatenate([np.full(nk, base), [np.log(0.5), 0.5, np.log(1.0), 0.0, np.log(0.1)]]), method="L-BFGS-B")
  null = dev(dt * y[h].sum() / dt[h].sum(), ~h)
  print(f"  {rg:5s} keys {nk:2d} sites {len(S):6d} | null {null:.5f} emp({3*nk}) {dev(mu_emp(re.x), ~h):.5f} proc({nk+5}) {dev(mu_proc(rp.x), ~h):.5f} | "
        f"δ {np.exp(rp.x[nk]):.2f} ρS {rp.x[nk+1]:.2f} λ0 {np.exp(rp.x[nk+2]):.2f} λ1 {rp.x[nk+3]:.2f} ε0 {np.exp(rp.x[nk+4]):.3f}")
# ---------------- R4 H–D allometry from traits
print("\n== R4 height rule with the H–D curve per key (3/key) vs from the trait Hmax_k + 2 globals: held-out sd ln(H2pred/H2obs)")
t = t.sort_values(["site", "ht_m"], ascending=[True, False]); t["O"] = (t.groupby("site").sdi.cumsum() - 0.5 * t.sdi) / 1500
al = t[(t.out == "alive") & (t.dia2_cm > 0) & (t.ht2_m > 1.37)].copy()
al["Hmx"] = al.groupby(["reg", "key"]).ht_m.transform(lambda x: x.quantile(0.995))
for rg, g in al.groupby("reg"):
  g = g.copy(); half = ((g.p * 7 + g.c) % 2 == 0).values
  hd = {}
  for k, x in g[half].groupby("key"):
    D, H = x.dia_cm.values, x.ht_m.values
    hd[k] = np.exp(least_squares(lambda p: np.log(1.37 + np.exp(p[0]) * (1 - np.exp(-np.exp(p[1]) * D)) ** np.exp(p[2])) - np.log(H), [np.log(30), np.log(0.03), 0.0]).x)
  def rule(Hs1, Hs2, z, p):
    eta, lam, k, mu = p; L = np.exp(-k * z.O.values)
    return z.ht_m.values + (Hs2 - Hs1) * (1 + eta * (1 - L)) + lam * z.dt.values * (Hs1 * (1 + mu * (1 - L)) - z.ht_m.values)
  sd = lambda e: np.sqrt(np.mean(e ** 2))
  # (a) per-key curve
  A1 = np.array([1.37 + hd[k][0] * (1 - np.exp(-hd[k][1] * D)) ** hd[k][2] for k, D in zip(g.key, g.dia_cm)])
  A2 = np.array([1.37 + hd[k][0] * (1 - np.exp(-hd[k][1] * D)) ** hd[k][2] for k, D in zip(g.key, g.dia2_cm)])
  la = lambda p, m: np.mean(np.log(np.maximum(rule(A1[m], A2[m], g[m], p), .5) / g.ht2_m.values[m]) ** 2)
  ra = minimize(lambda p: la(p, half), [0.0, 0.03, 4.0, 0.3], method="Nelder-Mead", options=dict(maxiter=2000))
  # (b) trait curve: Hmax_k + globals s, c (+ the same 4 rule globals), fitted jointly
  Hm = g.Hmx.values; D1 = g.dia_cm.values; D2 = g.dia2_cm.values
  cur = lambda D, s, c: 1.37 + (Hm - 1.37) * (1 - np.exp(-s * D / (Hm - 1.37))) ** c
  def lb(p, m):
    s, c = np.exp(p[4]), np.exp(p[5]); B1, B2 = cur(D1, s, c), cur(D2, s, c)
    return np.mean(np.log(np.maximum(rule(B1[m], B2[m], g[m], p[:4]), .5) / g.ht2_m.values[m]) ** 2)
  rb = minimize(lambda p: lb(p, half), np.concatenate([ra.x, [np.log(1.0), 0.0]]), method="Nelder-Mead", options=dict(maxiter=4000))
  nkk = g.key.nunique()
  print(f"  {rg:5s} keys {nkk:2d} | per-key curve ({3*nkk}+4) {sd(np.log(np.maximum(rule(A1[~half], A2[~half], g[~half], ra.x), .5) / g.ht2_m.values[~half])):.4f} | "
        f"trait curve (Hmax + 2 + 4) {np.sqrt(lb(rb.x, ~half)):.4f}  (s {np.exp(rb.x[4]):.2f} m/cm, c {np.exp(rb.x[5]):.2f})")
