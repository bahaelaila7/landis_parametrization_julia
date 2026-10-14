# TreeMap 2022 donor table

## Purpose
About 8,541 of 65,043 TreeMap 2022 plots (mostly FIA visits before 2008) have no rows in `curated_cohorts_landis_stdorg`, so Pan cannot initialize them. `data/treemap_donors.csv` assigns each such plot a structurally similar TreeMap plot that does have curated data. Built by `tools/build_treemap_donors.py`.

## Rule
- "Has data": the plot's location key (STATECD_UNITCD_COUNTYCD_PLOT, via PLOT.CN = PLT_CN) appears in `curated_cohorts_landis_stdorg` with a row where `age_calc > 0` and `coalesce(intro_type,'') <> 'phantom_backfill'`. Result: 56,502 with data, 8,541 without (as expected).
- Nonstocked targets (BALIVE=0 and TPA_LIVE=0, or no live trees in the tree table): no donor, reason `nonstocked`. Donors are also required to be stocked.
- Pool: same FORTYPCD if it has >= 20 candidates (`fortypcd`); else same forest-type group FORTYPCD//10*10 (`fortypgrp`); else all candidates (`any`).
- Nearest neighbour (Euclidean) on features standardized over all TreeMap plots: log1p(BALIVE), log1p(QMD), STANDHT, log1p(TPA_LIVE), CANOPYPCT, softwood BA fraction (live trees, SPCD<300, BA=0.005454154*DIA^2*TPA_UNADJ), plus LAT and LON each scaled by 0.5.
- Columns: cn, donor_cn, reason, dist, cn_measyear, donor_measyear, cn_statecd, donor_statecd.

## Results
Rows: 8,541. Reasons: fortypcd 8,514; fortypgrp 27; any 0; nonstocked 0 (every TreeMap plot has live trees: min BALIVE 1.80, min TPA_LIVE 1.99).

Target-vs-donor differences (median absolute, median absolute percent):

| Variable | All (n=8,541) | Region 4 (n=1,136) |
|---|---|---|
| BALIVE | 5.02 / 6.56% | 4.98 / 6.31% |
| QMD | 0.27 / 4.38% | 0.21 / 4.36% |
| STANDHT | 3.0 / 6.76% | 3.0 / 5.63% |
| TPA_LIVE | 30.1 / 8.58% | 44.9 / 9.10% |

Donor in same state: 46.3% overall; 30.6% in Region 4 states (1,12,13,21,28,37,45,47). Median match distance 0.37 (all), 0.39 (Region 4). Median MEASYEAR: targets 2004, donors 2016.

Examples (cn -> donor_cn, reason, dist, states): 188758664020004 -> 188782546020004 (fortypcd, 0.309, 8 -> 35); 55020149010661 -> 104969801010661 (fortypcd, 0.775, 26 -> 26); 102509448010661 -> 104967506010661 (fortypcd, 0.820, 26 -> 26); 473803218489998 -> 107527197010478 (fortypcd, 0.255, 48 -> 48); 143600046010854 -> 259086470010854 (fortypcd, 0.382, 48 -> 22).

## Limitations
- Donors are mostly newer visits (median 2016 vs 2004): structure is matched to the TreeMap attributes, not to the older stand history; species composition is only constrained through forest type and softwood fraction.
- Only 46% of donors are in the same state (31% in Region 4), so donors can come from different ecoregions; geography has weight 0.5 only. Pan uses the donor's epa_l3 cohorts, so ecoregion parameters follow the donor, not the target location.
- Many-to-one: popular donors may be reused by many targets. No distance cutoff is applied; check `dist` for poor matches.
- The rule was not changed from the specification; the `any` and `nonstocked` branches are unused with the current data.
