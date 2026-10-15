# Sim A/B campaign: interim analysis notes (2026-10-15)

## Sim A growth fails in every variant so far (criterion 1, decided on val)

| variant | val ±20 % | val median abs bias % |
|---|---|---|
| V1 knee | 5/48 | 31.9 |
| V2 / V2r knee | 8/47 | 29.7 |
| V2f knee | 4/47 | 42.1 |
| TOST-targeted reference (val) | 18/48 | 14.2 |
| V1 min_A archive point | 9/48 (test 15/46) | 20.0 (test 17.6) |

- The MO fronts are narrow in A: A_rel ranges from 0.70 (min_A) to about 1.0–1.08 at the W end. The chord knee sits toward W (V2f knee
  A_rel 1.06), so the knee choice costs growth accuracy.
- Even min_A is worse than the reference on group bias. a_g = mean |log1p Σsim − log1p Σobs| (L1) makes the MEDIAN pair unbiased,
  not the mean, while held-out TOST tests the group MEAN log ratio.
- Candidate change for the user to decide (not p-hacking; it is still pair-level accuracy, not the TOST statistic): a squared-log
  accuracy a_g = mean (log1p sim − log1p obs)², i.e. bias² + variance, which is mean-unbiased. Also refine and evaluate the min_A end
  (or select the point on the front by val), as the original design said (knee + both extremes).

## Sim B (after the B23 normalisation fix) is much better
- V2r val ±20 % 19/48, cycle bias −4.3 / −7.3 % (PASS).
- V2f mortality pooled bias +12 % (PASS), 71 % of cells within ×2 (rule 80 %).

## Why the age-structure term W barely improves (2026-10-15; scripts w_floor.py, w_floor2.py; train/val/test of V1's split)

### 1. Natural deaths (w_scope all)
About 35 % of W comes from cohorts that die between visits but stay alive in Sim A, where PROB_MORT is 0. This part is constant
across parameter sets (13,770 cohorts every time).
- It is removed by w_scope survivors (V1, V2*) and modelled by the death term (V2*, V4).
- Data: 97k of the previous cohorts vanish between visits, about 17 %. Only 3.4 % of vanished cohorts have a "new" cohort of the same
  species within ±3 yr, so these are real deaths, not re-ageing.

### 2. Matched-set shape (the rest)
This part is mostly not reducible by Biomass Succession parameters.

W1 compares biomass-weighted age CDFs per (plot, visit, species). These CDFs are scale-free, so parameters can change W only
through DIFFERENTIAL growth across the ages of one species at one plot. The data:
- An age-only growth rule per (L3, species) explains 5.2 % of the variance of cohort annual log growth between visits.
- Used as a predictor, the best age-only rule makes W WORSE than uniform growth: W(M2)/W(M0) = +6 % on train, +7 % on val/test.
- Simpson effect: the within-record age slope of growth is −0.012 per log-age, against −0.033 between records. Young cohorts grow
  faster ACROSS plots, but only weakly within a plot, so a model whose growth curve fits the population over-differentiates within a
  plot.
- 11 % of surviving cohort-intervals lose biomass (partial mortality inside a cohort, or trees reassigned). Biomass Succession cannot
  produce that, and those declines carry 22 % of the matched-set W.
- 36 % of records are single-cohort and have W = 0 whatever the parameters. The top 10 % of records carry 59 % of W.

This matches the earlier sensitivity result: matched-shape W moved 2.6 % across 8 parameter sets (random included) while A moved 36 %.

### Consequences for the MO calibration
W is close to a noise floor at the plot-record level, so the A–W front is narrow in W (W_rel 0.87–0.96 in V1) and wide in A. The
chord knee on range-normalised objectives inflates W's tiny range and pulls the selected point toward the W end, at the cost of
growth accuracy (V2f knee A_rel 1.06).

### Options
1. Score age structure at the population level: species × cell × age class, pooled over plots, like the tier-4 aggregate. Cohort
   noise averages out there and what remains is the age pattern the parameters can change. Or keep W only for Sim B, where
   establishment and mortality shape age structure.
2. Keep plot-level W but select on A, with W as a constraint (e.g. W ≤ (1 + ε)·min W). No range-normalised knee.
3. Model side: within-cohort partial mortality is absent from BS (whole-cohort PROB_MORT only). It explains a fifth of the
   matched-set W.

## Does W work? Pipeline check (w_check.jl; V1 knee params, gafl_A, survivors scope, train context)

The check uses Pan's own W kernel and the same references the Sim A loss uses, on 46,869 scored (site, visit, species) records.

**Oracle.** With the observed cohorts as the "sim", W = 0 on every record. The kernel, the references and the exclusions are correct: identical
curves score 0.

**Trajectory vs one interval:**

| mode | ΣW model | ΣW persistence (previous visit's survivors, aged) | corr |
|---|---|---|---|
| sync (the real Sim A scoring: matched cohorts keep simulated biomass across visits) | 28,965 | 14,028 | 0.585 |
| replace (every site restarts from the observed cohorts each visit = one-interval prediction) | 13,941 | 14,028 | 0.995 |

**Per visit, sync mode (multi-cohort records):**

| visit | mean W per record | W persistence | sim − obs biomass-weighted mean age (yr), mean ± sd |
|---|---|---|---|
| 1 | 0.51 | 0.51 | −0.01 ± 1.5 |
| 2 | 0.86 | 0.49 | −0.09 ± 2.3 |
| 3 | 1.15 | 0.42 | −0.17 ± 2.8 |
| 4 | 1.39 | 0.41 | −0.25 ± 3.1 |
| 5 | 1.59 | 0.40 | −0.31 ± 3.5 |

In replace mode W stays flat at the persistence level (0.51 → 0.39).

**Reading:**
- W is correct, and the model's one-interval growth is as good as "uniform growth": it neither helps nor hurts the shape.
- The one-interval W (≈ 14k) is the noise floor of the data. Cohorts change idiosyncratically between visits (partial mortality,
  ingrowth, measurement), and 43 % of that W is in records with a declining cohort.
- Along the trajectory, those idiosyncratic changes ACCUMULATE in the observed cohorts but not in the smooth simulation, so the
  sim–obs shape distance grows each visit. The age error is a random spread (sd 1.5 → 3.5 yr), not a bias (mean −0.01 → −0.31 yr).
  At visit 5 more than 99 % of the squared age error is variance.
- W is an unsigned per-record distance. Shared species×cell parameters can move the mean of the age error, but they cannot follow
  per-plot random walks, so W is nearly flat in the parameters (2.6 % spread). The small systematic part (simulated cohorts slightly
  younger, −0.3 yr) is drowned out.

**What would make W informative:** compare pooled distributions. Use species × cell × visit, summed over plots, so that per-plot random
deviations cancel and the systematic part remains. A signed age-shift diagnostic would also isolate the bias.

## Is the departure of W from its floor a real signal? (w_signal.jl, V1 archive, 41 points, survivors scope)

The test evaluates every archive point on TRAIN and on VAL, using the training loss settings.

| | train range (max−min)/min | val range | Spearman train vs val |
|---|---|---|---|
| W | 5.0 % | 5.3 % | **0.980** |
| A | 23.1 % | 22.9 % | 0.991 |

W's departure from its floor is reproducible held-out signal, so as an MO objective it works. **Pooling is withdrawn**, for two
reasons (the user's argument, 2026-10-15):
- it only removes a floor, which is irrelevant to Pareto ranking;
- it can hurt: per-plot errors that swing both ways would cancel to 0, the same score as a solution with nothing out of place.

What remains:
1. **Point selection.** Only about 5 % of W is attainable along the front, against 23 % of A, and the range-normalised chord knee
   values them equally. This preference should be set explicitly, e.g. select on val with a stated W tolerance.
2. **Visit weighting.** W's floor grows with visit index (per-record 0.51 at visit 1 → 1.59 at visit 5, sync mode), so plots with
   long records weigh more. This is an implicit weighting.

## Scoring schemes against the record-length imbalance (w_schemes.jl; V1 knee, train, survivors scope)
These were the user's ideas (2026-10-15):
- **plotlast:** score only each plot's last visit;
- **cohlast:** score each cohort (plot, species, birth year) only at its LAST eligible observation, i.e. the visit after which it is
  not seen again, or the plot's last visit.

The references are rebuilt from cohort subsets. Sanity check: the rebuilt references reproduce Pan's ΣW exactly (28,965.212).

| visits/plot | plot share | W share all | plotlast | cohlast |
|---|---|---|---|---|
| 1 | 13.8 % | 1.7 % | 4.8 % | 3.9 % |
| 2 | 15.8 % | 5.6 % | 9.6 % | 8.5 % |
| 3 | 28.4 % | 22.1 % | 26.6 % | 26.3 % |
| 4 | 12.5 % | 14.5 % | 13.9 % | 15.5 % |
| 5 | 29.5 % | 56.3 % | 45.1 % | 45.8 % |

| | all | plotlast | cohlast |
|---|---|---|---|
| cohort-observations scored | 159,608 | 42,025 | 59,623 |
| effective number of plots (1/Σ share²) | 977 | 986 | 1,132 |
| W per plot, 1 → 5 visits | 1.3 → 19.7 | 1.3 → 5.6 | 1.3 → 6.9 |

- Both schemes reduce the over-weighting of long records (5-visit plots go from 1.91× their plot share to 1.53–1.55×).
- cohlast keeps 42 % more anchors than plotlast. 18 % of its W comes from cohorts last seen before the plot's last visit, and it spreads
  W over more plots.
- The rest of the imbalance comes from trajectory length: the last observation is where accumulated drift is largest.

### Reproducibility of each scheme (V1 archive, 41 points; train vs val Spearman of the scheme's W) + imbalance
| scheme | cohort-obs scored | 5-visit W/plot-share | effective #plots | Spearman train↔val | range above min (train) | Spearman vs `all` (train) |
|---|---|---|---|---|---|---|
| all (current) | 159,608 | 1.91 | 977 | **0.980** | 5.0 % | 1 |
| plotlast | 42,025 | 1.53 | 986 | 0.878 | 5.9 % | 0.991 |
| cohlast | 59,623 | 1.55 | 1,132 | 0.878 | 5.5 % | 0.993 |
| plotmean (all visits, each plot's W averaged over its visits) | 159,608 | **1.43** | **1,170** | **0.973** | 5.2 % | 0.992 |

**Reading:**
- Scoring only the last visit (per plot or per cohort) reduces the long-record imbalance, but it drops consistent anchors. Held-out
  reproducibility falls from 0.98 to 0.88, and the ranking on train is almost unchanged (0.99 vs `all`).
- The intermediate visits do not amplify noise in the ranking sense; they reinforce it. The last observation also carries the largest
  accumulated drift, so it does not lower the per-record floor.
- Equal weight per plot (plotmean) removes MORE of the imbalance than either last-visit scheme and keeps the reproducibility
  (0.973).
- What remains in every scheme is drift growth with time (later visits have diverged further).
- Recommendation: if the scoring is made flexible in the refactor, implement plot-level averaging of W as an option. cohlast is still
  available as a variant, but it is not better here.
