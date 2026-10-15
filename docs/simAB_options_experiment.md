# Experiment: Sim A death/age-structure scoring options → Sim B (2026-10-15)

User request: "try all three options in combinations (and adjustments if necessary) to Sim B. report whose runs end up with
proper parameterization for all params". Model: **Biomass Succession** (Sim B needs establishment, so PnET is out; HBAD has
no whole-cohort death parameter). Region: GA+FL, the same train/val/test split as all GA+FL work. TOST is evaluation only.

## Background (verified 2026-10-14/15)
With scoring before injection, the disturbance shock and cohort-level exclusion (commits bccb338 and c184e00), the Sim A
age-structure term W splits into:
- a matched-set shape part, about 65 %;
- a natural-death part, about 35 %. These are cohorts the data lost but the simulation keeps. The part is constant across
  parameter sets (13,770 cohorts every time), because Sim A runs with `PROB_MORT = 0` and nothing else can kill a cohort
  between visits.

## Sim A scoring options
- **O1 survivors-only.** W is computed on cohorts present in both the simulation and the data at T. Natural deaths are a
  diagnostic only. `PROB_MORT` is not in Sim A; Sim B fits it.
- **O2 death likelihood.** W on survivors, plus a per-group death term `d_g`. `d_g` is the mean negative log-likelihood of each
  cohort's survive/die outcome over the interval Δt, given `PROB_MORT` (eco × species):
  - survivor: −Δt·log(1−p);
  - natural death: −log(1−(1−p)^Δt).

  Disturbance-removed and excluded cohorts are not counted. `PROB_MORT` is fitted in Sim A but is not applied stochastically
  there, so Sim A stays deterministic. It carries over to Sim B.
- **O3 as is.** W covers all cohorts (natural deaths form a constant floor); `PROB_MORT = 0` in Sim A.

## Variants (Sim A multi-objective gdigel → coordinate refinement of the knee → Sim B)
| id | Sim A objectives | PROB_MORT | Sim B fits |
|---|---|---|---|
| V1 | (A, W_surv) | not in Sim A | PROB_ESTAB, PROB_MORT |
| V2 | (A, W_surv, D) | Sim A | PROB_ESTAB (PROB_MORT frozen from Sim A) |
| V2r | same Sim A as V2 | Sim A, then re-fitted | PROB_ESTAB, PROB_MORT (started from Sim A's) |
| V2f | (A, W_surv + D) folded into 2 objectives | Sim A | PROB_ESTAB |
| V3 | (A, W_all) | not in Sim A | PROB_ESTAB, PROB_MORT |
| V4 | (A, W_all, D) | Sim A | PROB_ESTAB |

Growth parameters (S, B_MAX, ANPP ratio, D) are frozen from Sim A in Sim B. Sim B uses the existing `dual_mode: b`,
`simB_disturb_only` path, which now also applies full disturbance removals, with the igelmo Sim-B search and its seeds.

## "Proper parameterization for all params": pass criteria per variant (VAL decides pass/fail; TEST is reported alongside, never used to rank, select or decide)
1. **Growth (Sim A):**
   - held-out TOST ±10 % and ±20 % on val are at least the TOST-targeted reference's val numbers (computed by the evaluation tool from the reference parameter set; its test numbers, 15/46 and 28/46, are listed for information only);
   - median |bias| on val is at most 10 %.
2. **Bounds:** at most 5 % of fitted dimensions within 1 % of a box bound (u ≤ 0.01 or ≥ 0.99), for every parameter family
   (S, B_MAX, ANPP ratio, D, PROB_MORT, PROB_ESTAB).
3. **Mortality:** the implied annual whole-cohort death rate (1−(1−p)^1) per (eco, species) is within ×2 of the empirical
   natural-death rate on train, measured on the same cohorts and intervals. The bias of the pooled rate is within ±20 %.
4. **Establishment:** PROB_ESTAB lies within the data window ([0.8×data, 1.0]) and is not pinned at its lower bound for more
   than 25 % of cells.
5. **Sim B held-out:** plot-cluster bootstrap of log Σsim/Σobs per group per cycle on val (decides) and test (reported), ±20 % pass count; the
   cycle-1 and cycle-2 total AGB bias on val is within ±15 %.
6. **Identifiability:** for each family, the max/min spread across the archive's behavioural points (at most 1.5× the knee's
   F) is reported. A family whose spread exceeds 5× is flagged "not identified".

A variant has proper parameterization when 1–5 all pass; 6 is reported.

## Compute (8 cores)
- Sim A multi-objective gdigel: about 1 h per variant (μ 16, ~6,400 evaluations), plus refinement of about 20 min.
- Sim B igelmo: a few hours per variant.
- Total about one day.
- Outputs go to `runs_local/simab_<id>/`; summary in `docs/simAB_options_results.md`.
