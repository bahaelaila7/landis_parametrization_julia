"""HBAD3 S4: can height growth be derived from diameter growth through a fixed species H-D allometry (+ light-dependent elongation and
relaxation toward the allometric height), instead of v2's 3 free height-curve parameters + 2 competition responses per species?
1) Fit per key on TRAIN live trees: H* = 1.37 + a (1 - exp(-b D))^c (least squares on ln H).
2) Cohort rule (given the OBSERVED D2): H2 = H1 + [H*(D2) - H*(D1)] (1 + eta (1 - L)) + lam dt (H*(D1)·m(L) - H1), L = exp(-k O),
   m(L) = 1 + mu (1 - L) (shaded trees taller at the same D), globals (eta, lam, k, mu) fitted on TRAIN cohorts.
Reported: sd of ln(H2pred / H2obs) on VAL (survivor-weighted) vs the v2fv5 unconditional prediction (whose error also includes its D error),
and vs "allometry only" (H2 = H*(D2)·H1/H*(D1)). This is a conditional test of the allocation rule, not a full model comparison."""
import sys, numpy as np, pandas as pd, pyarrow.ipc as ip
from scipy.optimize import least_squares, minimize
R = sys.argv[1]
tr = ip.open_file(f"{R}/results/hbad2/interval_trees.arrow").read_all().to_pandas()
iv = ip.open_file(f"{R}/results/hbad2/intervals.arrow").read_all().to_pandas()[["iid", "set"]]
tr = tr.merge(iv, on="iid"); tr = tr[(tr.dia_cm > 0) & (tr.ht_m > 0)]
d = pd.read_csv(f"{R}/results/hbad3/resid_v2fv5.csv")
keys = sorted(d.key.unique()); big = set(k for k in keys if not k.startswith("_"))
tr["key"] = [s if s in big else "_" + g for s, g in zip(tr.sym, tr.grp)]
hd = {}
for k in keys:
  t = tr[(tr.key == k) & (tr.set == "train")]
  D, H = t.dia_cm.values, t.ht_m.values
  f = lambda p: np.log(1.37 + np.exp(p[0]) * (1 - np.exp(-np.exp(p[1]) * D)) ** np.exp(p[2])) - np.log(H)
  r = least_squares(f, [np.log(30), np.log(0.03), np.log(1.0)]); hd[k] = np.exp(r.x)
  print(f"{k:6s} n {len(t):6d}  a {hd[k][0]:6.1f} b {hd[k][1]:.4f} c {hd[k][2]:.3f}  sd ln H {np.std(r.fun):.3f}")
Hs = lambda k, D: 1.37 + hd[k][0] * (1 - np.exp(-hd[k][1] * D)) ** hd[k][2]
x = d[d.s > 0].copy()
x["Hs1"] = [Hs(k, D) for k, D in zip(x.key, x.D1)]; x["Hs2"] = [Hs(k, D) for k, D in zip(x.key, x.D2o)]
def pred(p, z):
  eta, lam, k, mu = p; L = np.exp(-k * z.O)
  return z.H1 + (z.Hs2 - z.Hs1) * (1 + eta * (1 - L)) + lam * z.dt * (z.Hs1 * (1 + mu * (1 - L)) - z.H1)
def loss(p, z):
  e = np.log(np.maximum(pred(p, z), 0.5) / z.H2o); return np.sum(z.s * e**2) / z.s.sum()
trc = x[x.set == "train"]; vac = x[x.set == "val"]
r = minimize(loss, [0.5, 0.02, 2.0, 0.2], args=(trc,), method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-6, fatol=1e-10))
sd = lambda e, z: np.sqrt(np.sum(z.s * e**2) / z.s.sum())
print("fitted globals eta lam k mu:", np.round(r.x, 4))
for nm, z in (("train", trc), ("val", vac)):
  e_rule = np.log(np.maximum(pred(r.x, z), 0.5) / z.H2o); e_allo = np.log(z.Hs2 * z.H1 / z.Hs1 / z.H2o)
  e_none = np.log(z.H1 / z.H2o); e_v2 = np.log(np.maximum(z.H2s, 0.1) / z.H2o)
  print(f"{nm}: sd ln(H2pred/H2obs)  no-growth {sd(e_none, z):.4f} | allometry-only {sd(e_allo, z):.4f} | rule (4 globals) {sd(e_rule, z):.4f} | v2fv5 unconditional {sd(e_v2, z):.4f}")
  # by O class: mean bias
  z = z.assign(e_rule=e_rule, e_v2=e_v2, Oc=pd.cut(z.O, [-1, .1, .3, .6, 10]))
  print(z.groupby("Oc", observed=True).apply(lambda y: pd.Series(dict(trees=y.s.sum(), bias_rule=np.average(y.e_rule, weights=y.s), bias_v2=np.average(y.e_v2, weights=y.s))), include_groups=False).round(4).to_string())
pd.DataFrame([dict(key=k, a=v[0], b=v[1], c=v[2]) for k, v in hd.items()]).to_csv(f"{R}/results/hbad3/hd_allometry.csv", index=False)
