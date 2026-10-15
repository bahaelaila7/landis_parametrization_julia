"""Follow-up to w_floor.py: (1) age trend of annual log growth WITHIN records vs BETWEEN records (Simpson check);
(2) re-aging: vanished previous cohorts vs new cohorts of the same plot/species within ±3 years of the expected age."""
import sys, math, collections, numpy as np, pandas as pd
exec(open("w_floor.py").read().split("records = []")[0])   # reuse the data loading (agg, dset_partial, dset_full)
within, between, reage = [], [], collections.Counter(); reage_bio = collections.Counter()
for pk, g in agg[agg.set == "train"].groupby(key):
    visits = sorted(g.sim_year.unique()); byv = {v: g[g.sim_year == v] for v in visits}
    for py, yr in zip(visits[:-1], visits[1:]):
        dlt = yr - py; P = byv[py]; C = byv[yr]
        for esp in set(C.esp) | set(P.esp):
            pa = dict(zip(P[P.esp == esp].age, P[P.esp == esp].agb)); ca = dict(zip(C[C.esp == esp].age, C[C.esp == esp].agb))
            m = [(a - dlt, pa[a - dlt], ca[a]) for a in ca if (a - dlt) in pa]
            rs = [(math.log(max(a0, 1)), math.log(o / p) / dlt) for a0, p, o in m if p > 0 and o > 0]
            if len(rs) >= 2:
                la = np.array([x for x, _ in rs]); r = np.array([y for _, y in rs])
                within += list(zip(la - la.mean(), r - r.mean())); between.append((la.mean(), r.mean()))
            van = [a + dlt for a in pa if (a + dlt) not in ca]; new = [a for a in ca if (a - dlt) not in pa]
            used = set()
            for e in van:
                cand = [a for a in new if abs(a - e) <= 3 and a not in used]
                if cand:
                    a = min(cand, key=lambda a: abs(a - e)); used.add(a); reage[abs(a - e)] += 1
                    reage_bio["vanished"] += pa[e - dlt]; reage_bio["new"] += ca[a]
                else: reage["vanished_no_mate"] += 1
            reage["new_total"] += len(new); reage["vanished_total"] += len(van)
W = np.array(within); B = np.array(between)
print(f"within-record slope of r on log age: {np.polyfit(W[:,0], W[:,1], 1)[0]:+.4f}  (n={len(W)})")
print(f"between-record slope (record means): {np.polyfit(B[:,0], B[:,1], 1)[0]:+.4f}  (n={len(B)})")
print("re-aging candidates (|age diff| of the nearest new cohort to a vanished one):", dict(sorted(reage.items(), key=str)))
print("biomass of matched vanished/new pairs:", dict(reage_bio))
