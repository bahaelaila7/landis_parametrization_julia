# Region 4 tree volume equations (FIA)

**Note: `bdft` is gross International 1/4-inch board feet (FIA `VOLBFGRS`), NOT Scribner, because `VOLBSGRS` is empty in this DB.**

Script: `tools/fit_volume_eqs.py` (deterministic; run with `/workspace/standmap/.venv/bin/python -I`). Output: `data/volume_eqs_r4.csv`.

## Data
FIADB.duckdb (read-only), TREE x PLOT x REF_SPECIES; live trees (STATUSCD=1), DIA >= 1.0 in, HT > 0 (total height, ft), MEASYEAR >= 2010, STATECD in 1,12,13,21,28,37,45,47. 1,971,188 trees on 69,088 plots (789k softwood, 1.18M hardwood).

**Deviation from the brief: `VOLBSGRS` (Scribner) is an empty VARCHAR column in this DB (0 positive values).** Board feet therefore use `VOLBFGRS` (gross board feet, International 1/4-inch rule in FIADB), which is populated for the same 426,965 sawtimber-size trees as `VOLCSGRS`. The `bdft` model is thus International 1/4", not Scribner; it is typically larger than Scribner. Rescale if Scribner is needed.

Null/zero handling. `VOLTSGRS` has 0 nulls and 0 zeros in every DBH class (1-5, 5-9, 9-11, 11-15, 15-20, 20-30, 30+ in), so no filtering was needed. Note it is evidently FIA-model-derived (some species fit with log RMSE ~0.000-0.01), so the fits recover FIA's own volume equations rather than independent measurements. `VOLBFGRS` is null for all trees < 9 in (100%), and for 54% (9-11 in), 18% (11-15), 16% (15-20), 18% (20-30), 34% (30+) of larger trees; it is never 0. Among trees at/above the threshold, 16,937 of 273,286 softwoods (6%) and 60,405 of 231,021 hardwoods (26%) have null board feet (presumably no merchantable sawlog; not verified). Fits use only VOL > 0 rows; in validation null board feet count as 0 observed.

## Model
ba = 0.005454154 DIA^2 (ft^2); ln V = b0 + b1 ln(ba) + b2 ln(HT), OLS; prediction = cf * exp(b0) * ba^b1 * HT^b2, cf = exp(s^2/2), s = residual SD. Up to 200k trees per fit (seeded random subsample); `n` in the CSV is the full count of V>0 rows.
`min_dbh` (bdft): 1st percentile of DIA among VOLBFGRS>0 rows: softwood 9.0 in, hardwood 11.0 in (confirmed; also the observed minimum).

### Sawtimber-qualifying probability (bdft only)
Many trees at or above `min_dbh` carry no board feet (6% of softwood, 26% of hardwood). For trees with DIA >= 9 (S) / 11 (H) in, `p(VOLBFGRS>0)` is a TPA_UNADJ-weighted logistic, logit p = sq_a + sq_c*ln(DIA), fitted at each bdft key (spcd, spgrpcd, major; columns `sq_a`, `sq_c`, empty for tcuft rows). Predicted bdft per tree = p(DIA) * cf*exp(b0)*ba^b1*HT^b2 for DIA >= min_dbh, else 0. Major level: softwood a=1.7414, c=0.3933; hardwood a=-0.6933, c=0.6427. Species-level fits can be erratic in sign (e.g. white oak c=-0.14, chestnut oak c=-1.11).

## Levels and row counts (CSV: 258 rows)
| model | spcd (n>=300 / 150) | spgrpcd | major |
|---|---|---|---|
| tcuft | 124 | 29 | 2 |
| bdft | 75 | 26 | 2 |

Prediction uses the most specific level available (spcd, then spgrpcd, then major).

## Coefficients (full-data fit)
Major: tcuft S b=(-1.3112, 0.9741, 1.0907) cf 1.0048 RMSE_log 0.098; tcuft H (-1.1065, 0.9422, 1.0350) cf 1.0044 RMSE 0.093; bdft S (-0.1448, 1.4419, 1.1733) RMSE 0.103; bdft H (-0.4583, 1.2880, 1.1825) RMSE 0.113.

| species | tcuft b0 / b1 / b2 | RMSE_log | bdft b0 / b1 / b2 | RMSE_log |
|---|---|---|---|---|
| loblolly pine (131) | -1.7913 / 0.9598 / 1.2012 | 0.040 | -0.4891 / 1.4702 / 1.2552 | 0.068 |
| sweetgum (611) | -1.2431 / 0.9585 / 1.0583 | 0.001 | -0.2036 / 1.3919 / 1.1241 | 0.043 |
| red maple (316) | -1.0769 / 0.9069 / 1.0217 | 0.042 | -0.2327 / 1.2938 / 1.1112 | 0.056 |
| slash pine (111) | -1.0873 / 1.0347 / 1.0370 | 0.026 | 0.1464 / 1.4954 / 1.1058 | 0.059 |
| yellow-poplar (621) | -1.6505 / 0.9221 / 1.1529 | 0.084 | -0.6709 / 1.3202 / 1.2388 | 0.085 |
| white oak (802) | -1.4882 / 0.9340 / 1.1312 | 0.045 | -0.4055 / 1.2878 / 1.1744 | 0.058 |

## Validation (20% of plots held out by PLOT.CN, seed 42; equations refit on the other 80%)
Per-tree log RMSE (held-out trees with V>0, new equations vs ba*HT*0.42, and board feet 4x that): tcuft 0.0405 vs 0.1179 (n=394,712); bdft 0.0630 vs 0.2390 (n=85,317). Level used: tcuft 99.6% spcd; bdft 99.4% spcd.

Plot totals (TPA_UNADJ-weighted, ft3/ac or bf/ac; 13,817 plots for tcuft, 11,615 with observed board feet):

| quantity | method | R2 | total bias % | mean plot bias % | median abs % err |
|---|---|---|---|---|---|
| tcuft | new | 0.997 | +0.01 | -0.02 | 1.5 |
| tcuft | Pan 0.42 | 0.976 | +6.06 | +6.04 | 6.1 |
| bdft (all trees >= min_dbh; null = 0) | new, no sq prob | 0.864 | +16.4 | +39.4 | 7.7 |
| bdft | **new + sq prob** | **0.931** | **+1.4** | +15.9 | 9.8 |
| bdft | Pan 4x0.42, DIA>=9 | 0.820 | +14.2 | +62.9 | 24.0 |
| bdft, only trees with VOLBFGRS>0 (diagnostic) | new, no sq prob | 0.979 | +2.6 | +0.9 | 2.9 |
| bdft, only trees with VOLBFGRS>0 (diagnostic) | Pan 4x0.42 | 0.925 | -8.4 | +1.5 | 12.2 |

Cohort level (plot x species x 2-inch DBH class; cohort volume = TPA * f(ba(QMD), TPA-weighted mean HT); bdft cohorts use trees >= min_dbh only):

| model | method | cohorts | bias % | R2 | median abs % err |
|---|---|---|---|---|---|
| tcuft | new | 197,578 | -0.18 | 0.997 | 1.1 |
| tcuft | Pan 0.42 | 197,578 | +5.87 | 0.978 | 7.1 |
| bdft (cohorts of sawtimber-size trees only, no sq prob) | new | 62,789 | +17.8 | 0.753 | 3.7 |
| bdft (same cohorts) | Pan 4x0.42 | 62,789 | +6.7 | 0.759 | 18.1 |

bdft cohorts as Pan holds them: ALL trees in plot x species x 2-inch class (197,578 cohorts, QMD/Hbar from all trees); observed = summed TPA*VOLBFGRS (null = 0). Variant A applies p(QMD) only if QMD >= min_dbh; variant B multiplies by p(QMD) and the sawtimber share 1 - Phi((min_dbh - QMD)/(0.25 QMD)) (normal DBH, CV 0.25). Median error is over cohorts with observed board feet > 0.

| method | bias % | R2 | median abs % err | plot-total R2 / bias % / mean bias % / median abs % (cohort preds summed to 11,615 plots) |
|---|---|---|---|---|
| A: p(QMD), QMD >= min_dbh | +1.0 | 0.873 | 12.6 | 0.928 / +0.1 / +10.6 / 13.0 |
| B: normal share x p(QMD) | -4.7 | 0.866 | 27.3 | 0.920 / -5.7 / +9.5 / 17.5 |
| no sq prob, QMD >= min_dbh | +16.4 | 0.820 | 4.0 | |
| Pan 4x0.42, QMD >= 9 | +14.2 | 0.802 | 18.3 | |

## Limitations
- Board-foot bias: the sq probability removes most of the total bias (plot +16.4% to +1.4%; cohort A +1.0%), but mean plot bias stays positive (+16%, cause not investigated) and species-level coefficients can be non-monotonic in DBH). Variant B under-predicts (-4.7%) and has larger median error (the normal-share assumption was not tested against actual within-cohort DBH spread).
- Board feet are International 1/4", not Scribner (see Data).
- VOLTSGRS is an FIA model output, so tcuft accuracy reflects reproduction of FIA equations, not measurement error. Region 4 states only, measurements since 2010; no extrapolation guidance beyond DIA ~ 1-40 in.
- Cohort evaluation uses QMD/mean height of observed trees, so within-cohort height variance (Jensen effect) is small by construction (cohorts are species x 2-in classes).
- cf is near 1 (<= 1.01) at all major levels, so log-bias correction is negligible.
