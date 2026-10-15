"""Why does the Sim A age-structure term W barely respond to parameters? Data-only analysis (no simulation).

For every train (plot, visit T, species) record under the survivors scope (cohorts present at T-Δ and again at T with age+Δ;
recruits at entry, disturbance partials and full removals excluded, as in the Sim A scoring rule), compare the observed biomass-
by-age distribution at T with predictions from the previous visit:
  M0 persistence  b_T = b_{T-Δ}                          (= any growth that is uniform across a record's ages: W is scale-free)
  M1 linear trend b_T = b_{T-Δ}·exp(Δ·(a_c + b_c·log age))  fitted per (L3, effective species) cell on train pairs
  M2 binned trend b_T = b_{T-Δ}·exp(Δ·g_c[agebin])       best per-cell age-bin growth rates (most flexible age-only rule)
W1 = Σ over the record's age support of |F_obs − F_pred| · gap (ages in years; biomass CDFs normalised per record).
    python -I w_floor.py <split_plots.csv> <species_map.csv> <out_dir>
"""
import sys, os, math, collections
import duckdb, numpy as np, pandas as pd

split_csv, spmap_csv, outd = sys.argv[1:4]
os.makedirs(outd, exist_ok=True)
DB = "/workspace/landis_region4/data_r4/slim_r4.duckdb"
DIST = "/workspace/landis_mogd/data/cohort_disturbance_full.parquet"

c = duckdb.connect()
c.sql(f"ATTACH '{DB}' AS s (READ_ONLY)")
sp = pd.read_csv(split_csv); sm = pd.read_csv(spmap_csv)
c.register("sp", sp); c.register("sm", sm)
coh = c.sql("""
  select t.statecd, t.unitcd, t.countycd, t.plot, sp.eco, split_part(sp.eco, '|', 1) as l3, sp.set,
         coalesce(sm.effective_species, '_UNMAPPED') as esp, t.species_symbol, t.sim_year, t.measdate, t.age_calc as age, t.agb
  from s.curated_cohorts_landis_stdorg t
  join sp using (statecd, unitcd, countycd, plot)
  left join sm using (species_symbol)""").df()
dist = c.sql(f"""
  select d.statecd, d.unitcd, d.countycd, d.plot, coalesce(sm.effective_species, '_UNMAPPED') esp, cast(d.measdate as date) md, d.age_calc age,
         max(d.disturbance_drop_pct) as dpct, bool_or(d.full_removal) as dfull
  from '{DIST}' d left join sm using (species_symbol) group by all""").df()
coh["md"] = pd.to_datetime(coh.measdate).dt.date
key = ["statecd", "unitcd", "countycd", "plot"]
# Pan aggregates cohorts per (plot, effective species, age) at each visit
agg = coh.groupby(key + ["eco", "l3", "set", "sim_year", "md", "esp", "age"], as_index=False).agg(agb=("agb", "sum"))
dkey = dist.set_index(key + ["esp", "md", "age"])
dset_partial = set(dkey[(dkey["dpct"] > 0) & (~dkey["dfull"])].index)
dset_full = set(dkey[dkey["dfull"]].index)

records = []   # (plot, T, esp, l3, set, Δ, ages, b_prev, b_obs)
stats = collections.Counter()
for pk, g in agg.groupby(key):
    visits = sorted(g.sim_year.unique())
    byv = {v: g[g.sim_year == v] for v in visits}
    for py, yr in zip(visits[:-1], visits[1:]):
        dlt = yr - py
        P = {(r.esp, r.age): r.agb for r in byv[py].itertuples()}
        C = byv[yr]; md = C.md.iloc[0]
        for esp, gc in C.groupby("esp"):
            ages, bp, bo = [], [], []
            for r in gc.itertuples():
                k = pk + (esp, md, r.age)
                if k in dset_partial: stats["excluded_partial_disturbance"] += 1; continue
                prev = P.get((esp, r.age - dlt))
                if prev is None: stats["recruit_or_age_shift"] += 1; continue
                ages.append(r.age); bp.append(prev); bo.append(r.agb); stats["matched"] += 1
            # previous cohorts of this species that vanished: death (natural) or full disturbance removal
            for (e2, a2), b2 in P.items():
                if e2 != esp: continue
                if (e2, a2 + dlt) not in set(zip(gc.esp, gc.age)):
                    stats["vanished_full_removal" if pk + (esp, md, a2 + dlt) in dset_full else "vanished_natural"] += 1
            if ages:
                records.append(dict(plot=pk, T=yr, esp=esp, l3=gc.l3.iloc[0], set=gc.set.iloc[0], dlt=dlt,
                                    ages=np.array(ages, float), bp=np.array(bp, float), bo=np.array(bo, float)))
print("cohort bookkeeping:", dict(stats))

def w1(ages, a, b):
    o = np.argsort(ages); x = ages[o]; fa = np.cumsum(a[o]) / a.sum(); fb = np.cumsum(b[o]) / b.sum()
    if len(x) < 2: return 0.0
    return float(np.sum(np.abs(fa[:-1] - fb[:-1]) * np.diff(x)))

R = pd.DataFrame(records)
R["n"] = R.ages.apply(len)
tr = R[R.set == "train"]
# pairwise annual log growth rates (train) for the trend fits
rows = []
for r in tr.itertuples():
    for a, p, o in zip(r.ages, r.bp, r.bo):
        if p > 0 and o > 0: rows.append((r.l3, r.esp, a - r.dlt, math.log(o / p) / r.dlt))
G = pd.DataFrame(rows, columns=["l3", "esp", "age0", "r"])
G["lage"] = np.log(np.maximum(G.age0, 1))
BINS = [0, 10, 20, 30, 40, 50, 60, 80, 100, 150, 1000]
G["bin"] = pd.cut(G.age0, BINS, labels=False, include_lowest=True)
lin = {}
for (l3, esp), g in G.groupby(["l3", "esp"]):
    if len(g) >= 10 and g.lage.std() > 0:
        b, a = np.polyfit(g.lage, g.r, 1); lin[(l3, esp)] = (a, b)
binned = G.groupby(["l3", "esp", "bin"]).r.median().to_dict()
# variance decomposition of the annual log growth rate (train pairs)
tot = G.r.var(); within_bin = G.groupby(["l3", "esp", "bin"]).r.transform(lambda s: s - s.median()).var()
print(f"annual log growth r: n={len(G)}  var={tot:.4f}  residual var after per-cell age-bin medians={within_bin:.4f}  "
      f"explained={(1 - within_bin / tot) * 100:.1f}%  share of declines (r<0)={(G.r < 0).mean() * 100:.1f}%  "
      f"median r={G.r.median():.4f}")

def pred(r, mode):
    if mode == "M0": return r.bp
    a0 = r.ages - r.dlt
    if mode == "M1":
        if (r.l3, r.esp) not in lin: return r.bp
        a, b = lin[(r.l3, r.esp)]; return r.bp * np.exp(r.dlt * (a + b * np.log(np.maximum(a0, 1))))
    if mode == "M2":
        bins = pd.cut(a0, BINS, labels=False, include_lowest=True)
        g = np.array([binned.get((r.l3, r.esp, b), 0.0) for b in bins]); return r.bp * np.exp(r.dlt * g)

out = []
for split in ("train", "val", "test"):
    S = R[R.set == split]
    res = {m: sum(w1(r.ages, pred(r, m), r.bo) for r in S.itertuples()) for m in ("M0", "M1", "M2")}
    multi = S[S.n >= 2]
    res_dec = sum(w1(r.ages, r.bp, np.maximum(r.bo, r.bp)) for r in S.itertuples())   # M0 with every decline removed
    out.append(dict(split=split, records=len(S), single_cohort_share=float((S.n == 1).mean()), multi=len(multi),
                    W_M0=res["M0"], W_M1=res["M1"], W_M2=res["M2"], W_M0_no_declines=res_dec))
O = pd.DataFrame(out)
O["reducible_by_age_trend_%"] = 100 * (1 - O.W_M2 / O.W_M0)
O["due_to_declines_%"] = 100 * (1 - O.W_M0_no_declines / O.W_M0)
print(O.to_string(index=False))
# where W sits: by number of cohorts in the record, and per-record W share of the top 10 % records (train, M0)
tr = R[R.set == "train"].copy(); tr["w"] = [w1(r.ages, r.bp, r.bo) for r in tr.itertuples()]
print(tr.groupby(pd.cut(tr.n, [0, 1, 2, 3, 5, 10, 100])).w.agg(["count", "sum", "mean"]).to_string())
q = tr.w.sort_values(ascending=False); print(f"top 10% of records carry {q.head(len(q) // 10).sum() / q.sum() * 100:.1f}% of train W(M0)")
O.to_csv(os.path.join(outd, "w_floor_summary.csv"), index=False)
G.to_csv(os.path.join(outd, "growth_pairs_train.csv"), index=False)
