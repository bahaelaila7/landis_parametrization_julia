# Sim A/B options campaign: summary (2026-10-15)

Design: docs/simAB_options_experiment.md. Criteria are decided on VAL; test is reported only. Model: Biomass Succession, GA+FL.
All Sim B runs use the corrected Sim B normalisation (3ab8fef). V1's Sim B was redone; its first run is kept as *_oldnorm.

| variant | Sim A val ±20 % | Sim A median abs bias % | mortality cells within ×2 | pooled implied p/yr (bias; empirical 0.0145) | Sim B val ±20 % | Sim B median abs bias % | Sim B cycle bias | criteria 1–5 |
|---|---|---|---|---|---|---|---|---|
| V1 | 5/48 | 31.9 | 2 % | 0.0000 (-100 %) | 18/48 | 12.4 | c1: +2.5, c2: +6.0 | FAIL FAIL FAIL PASS PASS |
| V2 | 8/47 | 29.7 | 25 % | 0.0059 (-59 %) | 18/48 | 12.2 | c1: -2.9, c2: -3.5 | FAIL FAIL FAIL PASS PASS |
| V2r | 8/47 | 29.7 | 31 % | 0.0098 (-33 %) | 19/48 | 10.6 | c1: -4.3, c2: -7.3 | FAIL FAIL FAIL PASS PASS |
| V2f | 4/47 | 42.1 | 71 % | 0.0163 (+12 %) | 14/48 | 18.0 | c1: -0.6, c2: -4.9 | FAIL FAIL FAIL PASS PASS |
| V3 | 10/48 | 27.7 | 2 % | 0.0000 (-100 %) | 14/48 | 18.8 | c1: +8.1, c2: +15.7 | FAIL FAIL FAIL PASS FAIL |
| V4 | 8/48 | 38.8 | 32 % | 0.0074 (-49 %) | 8/48 | 21.2 | c1: -0.0, c2: +0.8 | FAIL FAIL FAIL PASS PASS |

Criteria: 1 Sim A growth, 2 bounds, 3 mortality, 4 establishment, 5 Sim B held-out.

**No variant passes all of 1–5.**
1. **Sim A growth (1) fails everywhere.**
   - Val ±20 % is 4–10 of 48 groups against the TOST-targeted reference's 18; median |bias| 28–42 % vs 14 %.
   - Causes (docs/simAB_analysis_notes.md):
     - The chord knee on range-normalised objectives trades the ~23 % attainable A range against the ~5 % attainable W range
       one-for-one.
     - a_g is an L1 (median-unbiased) pair error, while TOST tests the group mean.
     - The min_A archive point is better (V1: 9/48, 20 %) but still worse than the reference.
2. **Mortality (3).**
   - PROB_MORT is NOT identified by Sim B: fitted freely there (V1, V3) it collapses to 0.
   - Fitted by the Sim A death likelihood it is identified. The folded variant V2f gets the pooled rate right (+12 %, PASS on the pooled
     rule), but only 71 % of cells are within ×2 (rule 80 %), and it costs Sim A growth (42 %).
   - With the 3-objective variants (V2, V2r, V4) the rate is 33–59 % too low.
3. **Establishment (4)** passes everywhere, but only 81 of 504 cells have data.
4. **Sim B held-out (5)** is good after the normalisation fix: cycle bias within ±15 % in 5 of 6 variants. Best is V2r (19/48 at
   ±20 %, median 10.6 %; cycle −4.3 / −7.3 %).
5. **Bounds (2)** fail everywhere: Sim A dims hug the box.

## Per-variant reports (generated)


Design: docs/simAB_options_experiment.md.


## Variant V2

### Sim A/B evaluation: V2

Sim A config `gafl_A.yml`, params `/workspace/landis_mogd/runs_local/simab_V2/refine/calibrated.jld2`; Sim B config `simab_B.yml`, params `/workspace/landis_mogd/runs_local/simab_V2/simB_params.jld2`

#### 1. Growth (Sim A held-out TOST): VAL decides, TEST reported

| split | groups | ±5 % | ±10 % | ±20 % | median abs bias % |
|---|---|---|---|---|---|
| val | 47 | 0 | 1 | 8 | 29.7 |
| test (information only) | 46 | 0 | 2 | 7 | 28.6 |
| reference (TOST-targeted BS coordinate set), val | 48 | 0 | 4 | 18 | 14.2 |
| reference (TOST-targeted BS coordinate set), test | 46 | 1 | 8 | 22 | 13.2 |

The reference's TEST numbers from the design: ±10 % 15/46, ±20 % 28/46 (information only).

Rule (decided on VAL): ±10 % passes ≥ 3.9 and ±20 % ≥ 17.6 (the reference's val numbers 4 / 18 of 48 groups, scaled to 47 groups) and median |bias| ≤ 10 %: **FAIL**

#### 2. Bound hugging (u ≤ 0.01 or ≥ 0.99), % of fitted dimensions

| stage | family | dims | hugging | % | low / high |
|---|---|---|---|---|---|
| Sim A | ANPP_ratio | 61 | 15 | 24.6 | 2 / 13 |
| Sim A | B_MAX | 61 | 14 | 23.0 | 11 / 3 |
| Sim A | D | 41 | 4 | 9.8 | 0 / 4 |
| Sim A | S | 61 | 5 | 8.2 | 4 / 1 |
| Sim B | PROB_ESTAB_SPP | 504 | 67 | 13.3 | 45 / 22 |
| Sim B | PROB_MORT_SPP | 504 | 11 | 2.2 | 6 / 5 |

Rule: ≤ 5 % per family: **FAIL**

#### 3. Mortality: implied vs empirical natural-death rate (train)

Implied annual rate p from Sim B final params; empirical = per-cell MLE of the survive/die likelihood over the same cohort-intervals (cells with ≥ 20 intervals: 374).

- cells within ×2 of the empirical rate: 25 % (rule ≥ 80 %)
- pooled implied 0.0059 vs empirical 0.0145 per year: bias -59 % (rule |bias| ≤ 20 %)

**FAIL**

| eco | species | intervals | deaths | p_emp | p_implied |
|---|---|---|---|---|---|
| 8.5.3|lu=C | NYBI | 3106 | 175 | 0.0100 | 0.0030 |
| 8.5.3|lu=C | PIEL | 2876 | 336 | 0.0206 | 0.0200 |
| 8.3.5|lu=B | PITA | 2793 | 294 | 0.0213 | 0.0070 |
| 8.3.5|lu=C | NYBI | 2750 | 137 | 0.0093 | 0.0030 |
| 8.3.5|lu=B | LIST2 | 2640 | 194 | 0.0143 | 0.0030 |
| 8.4.4|lu=X | _GRP_43_HST | 2492 | 84 | 0.0067 | 0.0070 |
| 8.3.5|lu=C | PITA | 2432 | 217 | 0.0175 | 0.0030 |
| 8.3.5|lu=C | QUNI | 2326 | 209 | 0.0169 | 0.0030 |
| 8.5.3|lu=C | QULA3 | 2294 | 247 | 0.0186 | 0.0070 |
| 8.4.4|lu=X | ACRU | 2272 | 47 | 0.0041 | 0.0030 |
| 8.3.4|lu=B | LIST2 | 2260 | 155 | 0.0139 | 0.0030 |
| 8.5.3|lu=C | ACRU | 2208 | 145 | 0.0118 | 0.0030 |
| 8.3.5|lu=B | QUNI | 1970 | 167 | 0.0169 | 0.0030 |
| 8.3.4|lu=C | LIST2 | 1937 | 113 | 0.0118 | 0.0030 |
| 8.3.4|lu=B | PITA | 1777 | 170 | 0.0199 | 0.0200 |

(15 largest cells shown.)

#### 4. Establishment (Sim B PROB_ESTAB)

- cells with a data value: 81 of 504 (the table covers only the L3s it lists); inside the window [mult × data, 1]: 81; pinned at the lower bound: 8 (10 %, rule ≤ 25 %)

**PASS**

#### 5. Sim B held-out (plot-cluster bootstrap on log Σsim/Σobs): VAL decides, TEST reported

| split | groups | ±5 % | ±10 % | ±20 % | median abs bias % | cycle bias % |
|---|---|---|---|---|---|---|
| val | 48 | 0 | 3 | 18 | 12.2 | c1: -2.9, c2: -3.5 |
| test (information only) | 47 | 0 | 7 | 11 | 15.6 | c1: -4.2, c2: -2.4 |

Rule (decided on VAL): cycle-1 and cycle-2 total AGB bias within ±15 % (val: -2.9, -3.5): **PASS**; the ±20 % pass counts are the held-out score (no fixed threshold in the design).

#### 6. Identifiability (archive spread, reported)

Behavioural points: ref-normalised F ≤ 1.5 × knee F (0.960 at the knee).

| family | points | dims | median max/min | share of dims > 5× | flag |
|---|---|---|---|---|---|
| ANPP | 251 | 508 | 2.67 | 2 % |  |
| B_MAX | 251 | 508 | 2.52 | 0 % |  |
| D | 251 | 41 | 4.63 | 0 % |  |
| PROB_MORT | 251 | 508 | 200.00 | 100 % | not identified |
| S | 251 | 508 | 100.00 | 95 % | not identified |

#### Verdict

| criterion | 1 growth | 2 bounds | 3 mortality | 4 establishment | 5 Sim B |
|---|---|---|---|---|---|
| result | FAIL | FAIL | FAIL | PASS | PASS |

Proper parameterization (1–5 all pass): **NO**

## Variant V2r

### Sim A/B evaluation: V2r

Sim A config `gafl_A.yml`, params `/workspace/landis_mogd/runs_local/simab_V2r/refine/calibrated.jld2`; Sim B config `simab_B.yml`, params `/workspace/landis_mogd/runs_local/simab_V2r/simB_params.jld2`

#### 1. Growth (Sim A held-out TOST): VAL decides, TEST reported

| split | groups | ±5 % | ±10 % | ±20 % | median abs bias % |
|---|---|---|---|---|---|
| val | 47 | 0 | 1 | 8 | 29.7 |
| test (information only) | 46 | 0 | 2 | 7 | 28.6 |
| reference (TOST-targeted BS coordinate set), val | 48 | 0 | 4 | 18 | 14.2 |
| reference (TOST-targeted BS coordinate set), test | 46 | 1 | 8 | 22 | 13.2 |

The reference's TEST numbers from the design: ±10 % 15/46, ±20 % 28/46 (information only).

Rule (decided on VAL): ±10 % passes ≥ 3.9 and ±20 % ≥ 17.6 (the reference's val numbers 4 / 18 of 48 groups, scaled to 47 groups) and median |bias| ≤ 10 %: **FAIL**

#### 2. Bound hugging (u ≤ 0.01 or ≥ 0.99), % of fitted dimensions

| stage | family | dims | hugging | % | low / high |
|---|---|---|---|---|---|
| Sim A | ANPP_ratio | 61 | 15 | 24.6 | 2 / 13 |
| Sim A | B_MAX | 61 | 14 | 23.0 | 11 / 3 |
| Sim A | D | 41 | 4 | 9.8 | 0 / 4 |
| Sim A | S | 61 | 5 | 8.2 | 4 / 1 |
| Sim B | PROB_ESTAB_SPP | 504 | 14 | 2.8 | 8 / 6 |
| Sim B | PROB_MORT_SPP | 504 | 206 | 40.9 | 202 / 4 |

Rule: ≤ 5 % per family: **FAIL**

#### 3. Mortality: implied vs empirical natural-death rate (train)

Implied annual rate p from Sim B final params; empirical = per-cell MLE of the survive/die likelihood over the same cohort-intervals (cells with ≥ 20 intervals: 374).

- cells within ×2 of the empirical rate: 31 % (rule ≥ 80 %)
- pooled implied 0.0098 vs empirical 0.0145 per year: bias -33 % (rule |bias| ≤ 20 %)

**FAIL**

| eco | species | intervals | deaths | p_emp | p_implied |
|---|---|---|---|---|---|
| 8.5.3|lu=C | NYBI | 3106 | 175 | 0.0100 | 0.0068 |
| 8.5.3|lu=C | PIEL | 2876 | 336 | 0.0206 | 0.0268 |
| 8.3.5|lu=B | PITA | 2793 | 294 | 0.0213 | 0.0000 |
| 8.3.5|lu=C | NYBI | 2750 | 137 | 0.0093 | 0.0000 |
| 8.3.5|lu=B | LIST2 | 2640 | 194 | 0.0143 | 0.0000 |
| 8.4.4|lu=X | _GRP_43_HST | 2492 | 84 | 0.0067 | 0.0061 |
| 8.3.5|lu=C | PITA | 2432 | 217 | 0.0175 | 0.0117 |
| 8.3.5|lu=C | QUNI | 2326 | 209 | 0.0169 | 0.0345 |
| 8.5.3|lu=C | QULA3 | 2294 | 247 | 0.0186 | 0.0000 |
| 8.4.4|lu=X | ACRU | 2272 | 47 | 0.0041 | 0.0293 |
| 8.3.4|lu=B | LIST2 | 2260 | 155 | 0.0139 | 0.0000 |
| 8.5.3|lu=C | ACRU | 2208 | 145 | 0.0118 | 0.0000 |
| 8.3.5|lu=B | QUNI | 1970 | 167 | 0.0169 | 0.0000 |
| 8.3.4|lu=C | LIST2 | 1937 | 113 | 0.0118 | 0.0086 |
| 8.3.4|lu=B | PITA | 1777 | 170 | 0.0199 | 0.0203 |

(15 largest cells shown.)

#### 4. Establishment (Sim B PROB_ESTAB)

- cells with a data value: 81 of 504 (the table covers only the L3s it lists); inside the window [mult × data, 1]: 81; pinned at the lower bound: 3 (4 %, rule ≤ 25 %)

**PASS**

#### 5. Sim B held-out (plot-cluster bootstrap on log Σsim/Σobs): VAL decides, TEST reported

| split | groups | ±5 % | ±10 % | ±20 % | median abs bias % | cycle bias % |
|---|---|---|---|---|---|---|
| val | 48 | 0 | 8 | 19 | 10.6 | c1: -4.3, c2: -7.3 |
| test (information only) | 47 | 0 | 5 | 15 | 14.0 | c1: -5.7, c2: -6.3 |

Rule (decided on VAL): cycle-1 and cycle-2 total AGB bias within ±15 % (val: -4.3, -7.3): **PASS**; the ±20 % pass counts are the held-out score (no fixed threshold in the design).

#### 6. Identifiability (archive spread, reported)

Behavioural points: ref-normalised F ≤ 1.5 × knee F (0.960 at the knee).

| family | points | dims | median max/min | share of dims > 5× | flag |
|---|---|---|---|---|---|
| ANPP | 251 | 508 | 2.67 | 2 % |  |
| B_MAX | 251 | 508 | 2.52 | 0 % |  |
| D | 251 | 41 | 4.63 | 0 % |  |
| PROB_MORT | 251 | 508 | 200.00 | 100 % | not identified |
| S | 251 | 508 | 100.00 | 95 % | not identified |

#### Verdict

| criterion | 1 growth | 2 bounds | 3 mortality | 4 establishment | 5 Sim B |
|---|---|---|---|---|---|
| result | FAIL | FAIL | FAIL | PASS | PASS |

Proper parameterization (1–5 all pass): **NO**

## Variant V2f

### Sim A/B evaluation: V2f

Sim A config `gafl_A.yml`, params `/workspace/landis_mogd/runs_local/simab_V2f/refine/calibrated.jld2`; Sim B config `simab_B.yml`, params `/workspace/landis_mogd/runs_local/simab_V2f/simB_params.jld2`

#### 1. Growth (Sim A held-out TOST): VAL decides, TEST reported

| split | groups | ±5 % | ±10 % | ±20 % | median abs bias % |
|---|---|---|---|---|---|
| val | 47 | 1 | 2 | 4 | 42.1 |
| test (information only) | 46 | 1 | 3 | 3 | 41.8 |
| reference (TOST-targeted BS coordinate set), val | 48 | 0 | 4 | 18 | 14.2 |
| reference (TOST-targeted BS coordinate set), test | 46 | 1 | 8 | 22 | 13.2 |

The reference's TEST numbers from the design: ±10 % 15/46, ±20 % 28/46 (information only).

Rule (decided on VAL): ±10 % passes ≥ 3.9 and ±20 % ≥ 17.6 (the reference's val numbers 4 / 18 of 48 groups, scaled to 47 groups) and median |bias| ≤ 10 %: **FAIL**

#### 2. Bound hugging (u ≤ 0.01 or ≥ 0.99), % of fitted dimensions

| stage | family | dims | hugging | % | low / high |
|---|---|---|---|---|---|
| Sim A | ANPP_ratio | 61 | 33 | 54.1 | 14 / 19 |
| Sim A | B_MAX | 61 | 13 | 21.3 | 9 / 4 |
| Sim A | D | 41 | 6 | 14.6 | 4 / 2 |
| Sim A | S | 61 | 26 | 42.6 | 18 / 8 |
| Sim B | PROB_ESTAB_SPP | 504 | 53 | 10.5 | 36 / 17 |
| Sim B | PROB_MORT_SPP | 504 | 11 | 2.2 | 7 / 4 |

Rule: ≤ 5 % per family: **FAIL**

#### 3. Mortality: implied vs empirical natural-death rate (train)

Implied annual rate p from Sim B final params; empirical = per-cell MLE of the survive/die likelihood over the same cohort-intervals (cells with ≥ 20 intervals: 374).

- cells within ×2 of the empirical rate: 71 % (rule ≥ 80 %)
- pooled implied 0.0163 vs empirical 0.0145 per year: bias +12 % (rule |bias| ≤ 20 %)

**FAIL**

| eco | species | intervals | deaths | p_emp | p_implied |
|---|---|---|---|---|---|
| 8.5.3|lu=C | NYBI | 3106 | 175 | 0.0100 | 0.0030 |
| 8.5.3|lu=C | PIEL | 2876 | 336 | 0.0206 | 0.0200 |
| 8.3.5|lu=B | PITA | 2793 | 294 | 0.0213 | 0.0219 |
| 8.3.5|lu=C | NYBI | 2750 | 137 | 0.0093 | 0.0030 |
| 8.3.5|lu=B | LIST2 | 2640 | 194 | 0.0143 | 0.0132 |
| 8.4.4|lu=X | _GRP_43_HST | 2492 | 84 | 0.0067 | 0.0088 |
| 8.3.5|lu=C | PITA | 2432 | 217 | 0.0175 | 0.0174 |
| 8.3.5|lu=C | QUNI | 2326 | 209 | 0.0169 | 0.0160 |
| 8.5.3|lu=C | QULA3 | 2294 | 247 | 0.0186 | 0.0200 |
| 8.4.4|lu=X | ACRU | 2272 | 47 | 0.0041 | 0.0108 |
| 8.3.4|lu=B | LIST2 | 2260 | 155 | 0.0139 | 0.0132 |
| 8.5.3|lu=C | ACRU | 2208 | 145 | 0.0118 | 0.0108 |
| 8.3.5|lu=B | QUNI | 1970 | 167 | 0.0169 | 0.0160 |
| 8.3.4|lu=C | LIST2 | 1937 | 113 | 0.0118 | 0.0132 |
| 8.3.4|lu=B | PITA | 1777 | 170 | 0.0199 | 0.0200 |

(15 largest cells shown.)

#### 4. Establishment (Sim B PROB_ESTAB)

- cells with a data value: 81 of 504 (the table covers only the L3s it lists); inside the window [mult × data, 1]: 81; pinned at the lower bound: 9 (11 %, rule ≤ 25 %)

**PASS**

#### 5. Sim B held-out (plot-cluster bootstrap on log Σsim/Σobs): VAL decides, TEST reported

| split | groups | ±5 % | ±10 % | ±20 % | median abs bias % | cycle bias % |
|---|---|---|---|---|---|---|
| val | 48 | 0 | 4 | 14 | 18.0 | c1: -0.6, c2: -4.9 |
| test (information only) | 47 | 0 | 3 | 12 | 16.6 | c1: -2.1, c2: -4.2 |

Rule (decided on VAL): cycle-1 and cycle-2 total AGB bias within ±15 % (val: -0.6, -4.9): **PASS**; the ±20 % pass counts are the held-out score (no fixed threshold in the design).

#### 6. Identifiability (archive spread, reported)

Behavioural points: ref-normalised F ≤ 1.5 × knee F (1.059 at the knee).

| family | points | dims | median max/min | share of dims > 5× | flag |
|---|---|---|---|---|---|
| ANPP | 30 | 508 | 1.99 | 0 % |  |
| B_MAX | 30 | 508 | 1.85 | 0 % |  |
| D | 30 | 41 | 2.75 | 0 % |  |
| PROB_MORT | 30 | 508 | 37.68 | 96 % | not identified |
| S | 30 | 508 | 94.88 | 93 % | not identified |

#### Verdict

| criterion | 1 growth | 2 bounds | 3 mortality | 4 establishment | 5 Sim B |
|---|---|---|---|---|---|
| result | FAIL | FAIL | FAIL | PASS | PASS |

Proper parameterization (1–5 all pass): **NO**

## Variant V3

### Sim A/B evaluation: V3

Sim A config `gafl_A.yml`, params `/workspace/landis_mogd/runs_local/simab_V3/refine/calibrated.jld2`; Sim B config `simab_B.yml`, params `/workspace/landis_mogd/runs_local/simab_V3/simB_params.jld2`

#### 1. Growth (Sim A held-out TOST): VAL decides, TEST reported

| split | groups | ±5 % | ±10 % | ±20 % | median abs bias % |
|---|---|---|---|---|---|
| val | 48 | 1 | 4 | 10 | 27.7 |
| test (information only) | 46 | 2 | 5 | 11 | 25.9 |
| reference (TOST-targeted BS coordinate set), val | 48 | 0 | 4 | 18 | 14.2 |
| reference (TOST-targeted BS coordinate set), test | 46 | 1 | 8 | 22 | 13.2 |

The reference's TEST numbers from the design: ±10 % 15/46, ±20 % 28/46 (information only).

Rule (decided on VAL): ±10 % passes ≥ 4.0 and ±20 % ≥ 18.0 (the reference's val numbers 4 / 18 of 48 groups, scaled to 48 groups) and median |bias| ≤ 10 %: **FAIL**

#### 2. Bound hugging (u ≤ 0.01 or ≥ 0.99), % of fitted dimensions

| stage | family | dims | hugging | % | low / high |
|---|---|---|---|---|---|
| Sim A | ANPP_ratio | 61 | 37 | 60.7 | 12 / 25 |
| Sim A | B_MAX | 61 | 23 | 37.7 | 14 / 9 |
| Sim A | D | 41 | 8 | 19.5 | 0 / 8 |
| Sim A | S | 61 | 15 | 24.6 | 11 / 4 |
| Sim B | PROB_ESTAB_SPP | 504 | 13 | 2.6 | 4 / 9 |
| Sim B | PROB_MORT_SPP | 504 | 504 | 100.0 | 504 / 0 |

Rule: ≤ 5 % per family: **FAIL**

#### 3. Mortality: implied vs empirical natural-death rate (train)

Implied annual rate p from Sim B final params; empirical = per-cell MLE of the survive/die likelihood over the same cohort-intervals (cells with ≥ 20 intervals: 374).

- cells within ×2 of the empirical rate: 2 % (rule ≥ 80 %)
- pooled implied 0.0000 vs empirical 0.0145 per year: bias -100 % (rule |bias| ≤ 20 %)

**FAIL**

| eco | species | intervals | deaths | p_emp | p_implied |
|---|---|---|---|---|---|
| 8.5.3|lu=C | NYBI | 3106 | 175 | 0.0100 | 0.0000 |
| 8.5.3|lu=C | PIEL | 2876 | 336 | 0.0206 | 0.0000 |
| 8.3.5|lu=B | PITA | 2793 | 294 | 0.0213 | 0.0000 |
| 8.3.5|lu=C | NYBI | 2750 | 137 | 0.0093 | 0.0000 |
| 8.3.5|lu=B | LIST2 | 2640 | 194 | 0.0143 | 0.0000 |
| 8.4.4|lu=X | _GRP_43_HST | 2492 | 84 | 0.0067 | 0.0000 |
| 8.3.5|lu=C | PITA | 2432 | 217 | 0.0175 | 0.0000 |
| 8.3.5|lu=C | QUNI | 2326 | 209 | 0.0169 | 0.0000 |
| 8.5.3|lu=C | QULA3 | 2294 | 247 | 0.0186 | 0.0000 |
| 8.4.4|lu=X | ACRU | 2272 | 47 | 0.0041 | 0.0000 |
| 8.3.4|lu=B | LIST2 | 2260 | 155 | 0.0139 | 0.0000 |
| 8.5.3|lu=C | ACRU | 2208 | 145 | 0.0118 | 0.0000 |
| 8.3.5|lu=B | QUNI | 1970 | 167 | 0.0169 | 0.0000 |
| 8.3.4|lu=C | LIST2 | 1937 | 113 | 0.0118 | 0.0000 |
| 8.3.4|lu=B | PITA | 1777 | 170 | 0.0199 | 0.0000 |

(15 largest cells shown.)

#### 4. Establishment (Sim B PROB_ESTAB)

- cells with a data value: 81 of 504 (the table covers only the L3s it lists); inside the window [mult × data, 1]: 81; pinned at the lower bound: 0 (0 %, rule ≤ 25 %)

**PASS**

#### 5. Sim B held-out (plot-cluster bootstrap on log Σsim/Σobs): VAL decides, TEST reported

| split | groups | ±5 % | ±10 % | ±20 % | median abs bias % | cycle bias % |
|---|---|---|---|---|---|---|
| val | 48 | 0 | 3 | 14 | 18.8 | c1: +8.1, c2: +15.7 |
| test (information only) | 47 | 0 | 4 | 10 | 20.8 | c1: +7.2, c2: +16.9 |

Rule (decided on VAL): cycle-1 and cycle-2 total AGB bias within ±15 % (val: +8.1, +15.7): **FAIL**; the ±20 % pass counts are the held-out score (no fixed threshold in the design).

#### 6. Identifiability (archive spread, reported)

Behavioural points: ref-normalised F ≤ 1.5 × knee F (0.946 at the knee).

| family | points | dims | median max/min | share of dims > 5× | flag |
|---|---|---|---|---|---|
| ANPP | 87 | 508 | 2.17 | 0 % |  |
| B_MAX | 87 | 508 | 2.10 | 0 % |  |
| D | 87 | 41 | 3.35 | 0 % |  |
| S | 87 | 508 | 100.00 | 93 % | not identified |

#### Verdict

| criterion | 1 growth | 2 bounds | 3 mortality | 4 establishment | 5 Sim B |
|---|---|---|---|---|---|
| result | FAIL | FAIL | FAIL | PASS | FAIL |

Proper parameterization (1–5 all pass): **NO**

## Variant V4

### Sim A/B evaluation: V4

Sim A config `gafl_A.yml`, params `/workspace/landis_mogd/runs_local/simab_V4/refine/calibrated.jld2`; Sim B config `simab_B.yml`, params `/workspace/landis_mogd/runs_local/simab_V4/simB_params.jld2`

#### 1. Growth (Sim A held-out TOST): VAL decides, TEST reported

| split | groups | ±5 % | ±10 % | ±20 % | median abs bias % |
|---|---|---|---|---|---|
| val | 48 | 0 | 3 | 8 | 38.8 |
| test (information only) | 46 | 0 | 1 | 10 | 36.9 |
| reference (TOST-targeted BS coordinate set), val | 48 | 0 | 4 | 18 | 14.2 |
| reference (TOST-targeted BS coordinate set), test | 46 | 1 | 8 | 22 | 13.2 |

The reference's TEST numbers from the design: ±10 % 15/46, ±20 % 28/46 (information only).

Rule (decided on VAL): ±10 % passes ≥ 4.0 and ±20 % ≥ 18.0 (the reference's val numbers 4 / 18 of 48 groups, scaled to 48 groups) and median |bias| ≤ 10 %: **FAIL**

#### 2. Bound hugging (u ≤ 0.01 or ≥ 0.99), % of fitted dimensions

| stage | family | dims | hugging | % | low / high |
|---|---|---|---|---|---|
| Sim A | ANPP_ratio | 61 | 10 | 16.4 | 6 / 4 |
| Sim A | B_MAX | 61 | 11 | 18.0 | 5 / 6 |
| Sim A | D | 41 | 3 | 7.3 | 3 / 0 |
| Sim A | S | 61 | 10 | 16.4 | 8 / 2 |
| Sim B | PROB_ESTAB_SPP | 504 | 57 | 11.3 | 35 / 22 |
| Sim B | PROB_MORT_SPP | 504 | 7 | 1.4 | 3 / 4 |

Rule: ≤ 5 % per family: **FAIL**

#### 3. Mortality: implied vs empirical natural-death rate (train)

Implied annual rate p from Sim B final params; empirical = per-cell MLE of the survive/die likelihood over the same cohort-intervals (cells with ≥ 20 intervals: 374).

- cells within ×2 of the empirical rate: 32 % (rule ≥ 80 %)
- pooled implied 0.0074 vs empirical 0.0145 per year: bias -49 % (rule |bias| ≤ 20 %)

**FAIL**

| eco | species | intervals | deaths | p_emp | p_implied |
|---|---|---|---|---|---|
| 8.5.3|lu=C | NYBI | 3106 | 175 | 0.0100 | 0.0030 |
| 8.5.3|lu=C | PIEL | 2876 | 336 | 0.0206 | 0.0205 |
| 8.3.5|lu=B | PITA | 2793 | 294 | 0.0213 | 0.0213 |
| 8.3.5|lu=C | NYBI | 2750 | 137 | 0.0093 | 0.0030 |
| 8.3.5|lu=B | LIST2 | 2640 | 194 | 0.0143 | 0.0030 |
| 8.4.4|lu=X | _GRP_43_HST | 2492 | 84 | 0.0067 | 0.0120 |
| 8.3.5|lu=C | PITA | 2432 | 217 | 0.0175 | 0.0070 |
| 8.3.5|lu=C | QUNI | 2326 | 209 | 0.0169 | 0.0030 |
| 8.5.3|lu=C | QULA3 | 2294 | 247 | 0.0186 | 0.0120 |
| 8.4.4|lu=X | ACRU | 2272 | 47 | 0.0041 | 0.0030 |
| 8.3.4|lu=B | LIST2 | 2260 | 155 | 0.0139 | 0.0030 |
| 8.5.3|lu=C | ACRU | 2208 | 145 | 0.0118 | 0.0030 |
| 8.3.5|lu=B | QUNI | 1970 | 167 | 0.0169 | 0.0030 |
| 8.3.4|lu=C | LIST2 | 1937 | 113 | 0.0118 | 0.0030 |
| 8.3.4|lu=B | PITA | 1777 | 170 | 0.0199 | 0.0070 |

(15 largest cells shown.)

#### 4. Establishment (Sim B PROB_ESTAB)

- cells with a data value: 81 of 504 (the table covers only the L3s it lists); inside the window [mult × data, 1]: 81; pinned at the lower bound: 9 (11 %, rule ≤ 25 %)

**PASS**

#### 5. Sim B held-out (plot-cluster bootstrap on log Σsim/Σobs): VAL decides, TEST reported

| split | groups | ±5 % | ±10 % | ±20 % | median abs bias % | cycle bias % |
|---|---|---|---|---|---|---|
| val | 48 | 0 | 3 | 8 | 21.2 | c1: -0.0, c2: +0.8 |
| test (information only) | 47 | 0 | 2 | 8 | 22.9 | c1: -1.1, c2: +1.9 |

Rule (decided on VAL): cycle-1 and cycle-2 total AGB bias within ±15 % (val: -0.0, +0.8): **PASS**; the ±20 % pass counts are the held-out score (no fixed threshold in the design).

#### 6. Identifiability (archive spread, reported)

Behavioural points: ref-normalised F ≤ 1.5 × knee F (0.949 at the knee).

| family | points | dims | median max/min | share of dims > 5× | flag |
|---|---|---|---|---|---|
| ANPP | 287 | 508 | 2.94 | 0 % |  |
| B_MAX | 287 | 508 | 2.46 | 0 % |  |
| D | 287 | 41 | 4.33 | 0 % |  |
| PROB_MORT | 287 | 508 | 200.00 | 100 % | not identified |
| S | 287 | 508 | 100.00 | 98 % | not identified |

#### Verdict

| criterion | 1 growth | 2 bounds | 3 mortality | 4 establishment | 5 Sim B |
|---|---|---|---|---|---|
| result | FAIL | FAIL | FAIL | PASS | PASS |

Proper parameterization (1–5 all pass): **NO**

## Variant V1

### Sim A/B evaluation: V1

Sim A config `gafl_A.yml`, params `/workspace/landis_mogd/runs_local/simab_V1/refine/calibrated.jld2`; Sim B config `simab_B.yml`, params `/workspace/landis_mogd/runs_local/simab_V1/simB_params.jld2`

#### 1. Growth (Sim A held-out TOST): VAL decides, TEST reported

| split | groups | ±5 % | ±10 % | ±20 % | median abs bias % |
|---|---|---|---|---|---|
| val | 48 | 0 | 1 | 5 | 31.9 |
| test (information only) | 46 | 0 | 2 | 7 | 30.9 |
| reference (TOST-targeted BS coordinate set), val | 48 | 0 | 4 | 18 | 14.2 |
| reference (TOST-targeted BS coordinate set), test | 46 | 1 | 8 | 22 | 13.2 |

The reference's TEST numbers from the design: ±10 % 15/46, ±20 % 28/46 (information only).

Rule (decided on VAL): ±10 % passes ≥ 4.0 and ±20 % ≥ 18.0 (the reference's val numbers 4 / 18 of 48 groups, scaled to 48 groups) and median |bias| ≤ 10 %: **FAIL**

#### 2. Bound hugging (u ≤ 0.01 or ≥ 0.99), % of fitted dimensions

| stage | family | dims | hugging | % | low / high |
|---|---|---|---|---|---|
| Sim A | ANPP_ratio | 61 | 37 | 60.7 | 15 / 22 |
| Sim A | B_MAX | 61 | 21 | 34.4 | 15 / 6 |
| Sim A | D | 41 | 6 | 14.6 | 2 / 4 |
| Sim A | S | 61 | 21 | 34.4 | 17 / 4 |
| Sim B | PROB_ESTAB_SPP | 504 | 13 | 2.6 | 4 / 9 |
| Sim B | PROB_MORT_SPP | 504 | 504 | 100.0 | 504 / 0 |

Rule: ≤ 5 % per family: **FAIL**

#### 3. Mortality: implied vs empirical natural-death rate (train)

Implied annual rate p from Sim B final params; empirical = per-cell MLE of the survive/die likelihood over the same cohort-intervals (cells with ≥ 20 intervals: 374).

- cells within ×2 of the empirical rate: 2 % (rule ≥ 80 %)
- pooled implied 0.0000 vs empirical 0.0145 per year: bias -100 % (rule |bias| ≤ 20 %)

**FAIL**

| eco | species | intervals | deaths | p_emp | p_implied |
|---|---|---|---|---|---|
| 8.5.3|lu=C | NYBI | 3106 | 175 | 0.0100 | 0.0000 |
| 8.5.3|lu=C | PIEL | 2876 | 336 | 0.0206 | 0.0000 |
| 8.3.5|lu=B | PITA | 2793 | 294 | 0.0213 | 0.0000 |
| 8.3.5|lu=C | NYBI | 2750 | 137 | 0.0093 | 0.0000 |
| 8.3.5|lu=B | LIST2 | 2640 | 194 | 0.0143 | 0.0000 |
| 8.4.4|lu=X | _GRP_43_HST | 2492 | 84 | 0.0067 | 0.0000 |
| 8.3.5|lu=C | PITA | 2432 | 217 | 0.0175 | 0.0000 |
| 8.3.5|lu=C | QUNI | 2326 | 209 | 0.0169 | 0.0000 |
| 8.5.3|lu=C | QULA3 | 2294 | 247 | 0.0186 | 0.0000 |
| 8.4.4|lu=X | ACRU | 2272 | 47 | 0.0041 | 0.0000 |
| 8.3.4|lu=B | LIST2 | 2260 | 155 | 0.0139 | 0.0000 |
| 8.5.3|lu=C | ACRU | 2208 | 145 | 0.0118 | 0.0000 |
| 8.3.5|lu=B | QUNI | 1970 | 167 | 0.0169 | 0.0000 |
| 8.3.4|lu=C | LIST2 | 1937 | 113 | 0.0118 | 0.0000 |
| 8.3.4|lu=B | PITA | 1777 | 170 | 0.0199 | 0.0000 |

(15 largest cells shown.)

#### 4. Establishment (Sim B PROB_ESTAB)

- cells with a data value: 81 of 504 (the table covers only the L3s it lists); inside the window [mult × data, 1]: 81; pinned at the lower bound: 0 (0 %, rule ≤ 25 %)

**PASS**

#### 5. Sim B held-out (plot-cluster bootstrap on log Σsim/Σobs): VAL decides, TEST reported

| split | groups | ±5 % | ±10 % | ±20 % | median abs bias % | cycle bias % |
|---|---|---|---|---|---|---|
| val | 48 | 0 | 6 | 18 | 12.4 | c1: +2.5, c2: +6.0 |
| test (information only) | 47 | 0 | 6 | 20 | 10.6 | c1: +1.4, c2: +7.1 |

Rule (decided on VAL): cycle-1 and cycle-2 total AGB bias within ±15 % (val: +2.5, +6.0): **PASS**; the ±20 % pass counts are the held-out score (no fixed threshold in the design).

#### 6. Identifiability (archive spread, reported)

Behavioural points: ref-normalised F ≤ 1.5 × knee F (0.911 at the knee).

| family | points | dims | median max/min | share of dims > 5× | flag |
|---|---|---|---|---|---|
| ANPP | 41 | 508 | 1.78 | 0 % |  |
| B_MAX | 41 | 508 | 1.79 | 0 % |  |
| D | 41 | 41 | 2.92 | 0 % |  |
| S | 41 | 508 | 36.69 | 89 % | not identified |

#### Verdict

| criterion | 1 growth | 2 bounds | 3 mortality | 4 establishment | 5 Sim B |
|---|---|---|---|---|---|
| result | FAIL | FAIL | FAIL | PASS | PASS |

Proper parameterization (1–5 all pass): **NO**
