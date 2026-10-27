# HBAD4P decision log

Every decision point, with the options considered. Since 2026-10-26 the agent takes the recommended option without asking (user: "go with your
recommendations, but document the decision point and the different options"). The audit history is in hbad4p_design.md §7–9.

## H1 — Model scope (2026-10-26, user)
- **Cohort:** species × FIA 2-inch diameter class, tracking N, D, H and age.
- **Calibration data:** **stand tables only**. The alternative was tracked trees, which amplified the stand-level bias of HBAD3P.
- **Climate drivers:** **water + temperature + CO2**, with βC fixed at 0.58 (the alternatives were temperature only, or none).
- **Order of work:** **GA+FL nested CV first**, Region 4 after.
- **Lanes:** **tripling 0.5 / 0.25 / 0.25** (FVS SN uses 0.60 / 0.25 / 0.15, available as the :fvs option).

## H2 — Likelihood for the remeasurement stand tables (round 2 I4, agent; open since round 2)
- **Options:** (a) **independent NB counts per class**, plus log QMD and log H; (b) a transition likelihood of tree fates.
- **Chosen:** (a). (b) would need tracked-tree data, which conflicts with H1's "stand tables only".

## H3 — Atoms per cell (round 3 B1, agent)
- **Evidence:** NLL error against a 64-atom reference, in nats/site at κ = 0 / κ = 0.1:

  | NATOM | κ = 0 | κ = 0.1 | Cost |
  |---|---|---|---|
  | 4 (old) | — | +0.27 | — |
  | 8 | −0.002 | +0.029 | — |
  | 12 | — | +0.017 | about 0.6× that of 16 |
  | 16 | 0.000 | +0.004 | κ = 0.1 costs 8–9× κ = 0 (round 4 correction; it is 1.4–1.6× the round-2 kernel at the same κ) |

- **Chosen:** **16**. Accuracy of the κ effect matters for calibration: with 4 atoms the apparent effect was 17× the true one. At the warm start (κ = 0) the new kernel is faster than before.
- **To reverse:** set `NATOM = 12` in HBAD4P.jl if run time dominates.

## H4 — Lane-mixing parameter at the warm start (round 3 I2d, agent)
- **Options:** a log-encoding with a 1e-3 floor (the old one); **linear on [0, 1] starting at 0**; a mapping from HBAD3P's vigour.
- **Chosen:** linear [0, 1] from 0. HBAD3P's vigour nodes are fully persistent and have no mixing analogue.

## H5 — Baseline for "the model works" (round 3 I3, agent)
- **Options:** persistence; **the train-estimated transition-matrix null**.
- **Chosen:** the transition-matrix null. A calibrated HBAD4P must beat it on val counts. The uncalibrated warm start does not yet (+8,622 nats on k1 val counts).

## H6 — DBH measurement-error blur (round 3 I1b; agent / implementer, under audit in round 4)
- **Options:** no blur; **normal with sd 0.2 cm, in the likelihood only**; a blur in the dynamics.
- **Chosen:** blur in the likelihood only, with sd 0.2 cm.
- **Open sub-decision:** the lower bound of class 1 is NOT blurred, so recruits entering at exactly 2.54 cm stay tallied. Round 4 audits this.

## H7 — Scoring the size and height terms of trace cells (round 4 I1, agent)
- **Evidence:** the hard gate Np > 1e-9 makes O(10)-nat jumps in the objective; the scored sets also differ between models.
- **Options:**
  - (a) keep the gate;
  - (b) **a mixture of the model's Gaussian, weight Np / (Np + b), with a fixed data-derived background density, scored for every observed cell**;
  - (c) drop the size and height terms of cells with low N.
- **Chosen:** (b). It is the auditor's tested variant (3× less roughness) and keeps model and null totals comparable.
- **Implementation (2026-10-26, agent).**
  - **Pseudo-count b = bg**, the fitted count background density (bgM / bgS). Under the count model, μ = (N + bg) f a, a tallied tree of the cell is the
    model's with probability N / (N + bg). The weight is therefore that model's own posterior share, not a new parameter.
    - Alternatives: a fixed number of pseudo-trees (b · f · a = 1 tree), which halves the size information of every 1-tree cell; a fixed 1e-3 /ha, which
      gives almost no smoothing.
  - **Background densities:** TRAIN data constants (`obs_background`).
    - log QMD: per class, the mean / sd over all train cells.
    - log H: per key × class, the mean (≥ 10 cells, else the class mean), with the per-class sd around the key means.
    - Alternative: the auditor's uniform log QMD over the class and sd 0.3 for log H.
  - **Added (evidence below):** the predicted height of a class uses each atom's allometric height H_m = H + H*(D_m) − H*(Dq), not the cell mean. With the
    cell mean, an atom crossing a class bound carried its new cell's height into the blurred tail of its old class. The height term's step sd at 1e-3 fell
    from 1.96 to 0.94.
- **Evidence (full train, ±20 steps of ln ε PITA, auditor OBX):**

  | | step sd at 1e-4 | max jump at 1e-4 | step sd at 1e-3 |
  |---|---|---|---|
  | Gate (round 3) | 2.79 | 12.4 | 8.15 |
  | Mixture with the cell-mean heights | 0.58 | 1.82 | 3.22 |
  | Mixture + atom heights | 0.48 | 1.82 | 2.67 (2.66 detrended) |

  - κ ±20 steps around 0.1: 0.12 / 0.70 at 1e-4 / 1e-3 (max jump 0.55 / 2.42).
  - The count term keeps 0.16 / 2.58 (M2: exact transfer across class bounds, accepted).
- **Variants tested and NOT taken** (full train, observation parameters fitted at the warm start — σD 0.010, σH 0.066, sharper than the acceptance scripts'
  0.019 / 0.087; detrended step sd at 1e-4 / 1e-3, max jump at 1e-4):
  - **The atom's allometric height in the dynamics too** (dD_m = dV / (2 D_m H_m), hydraulics at H_m): 1.24 / 3.45, jump 6.5. Worse: the count term rose to
    0.98 at 1e-4.
  - **The height residual interpolated in D between the key-lane's cells** (continuous in D whichever cell holds the atom): 0.58 / 2.81, jump 2.32. The
    kernel with atom heights gives 0.60 / 2.78, jump 2.71: the residual interpolation lowers the jump but leaves the 1e-3 sd as it is, at more complexity.
- **Not met with the fitted observation parameters:** the step sd at 1e-3 is 2.78 (bar 2.7) and the largest jump at 1e-4 is 2.71 (bar 2). With the acceptance
  scripts' fixed observation parameters both bars are met (2.67, 1.82).
  - What remains at 1e-3 is mainly the count term: sd 2.46–2.51 alone. That is the M2 roughness of exact transfer across class bounds plus one height and
    age per cell in the dynamics, accepted in round 4.
  - Making cell membership smooth in the dynamics is the lever left; it is a model change for a later round.
  - The residual noise CMA-ES will see is about 2.5–2.8 nats per 1e-3 step of a coordinate on full train. Check 3.3 was split by term (H14).
- **To reverse:** `nll4` in data4.jl (the mixture) and `finish_site!` in HBAD4P.jl (H_m → the cell's H).

## H8 — Fecundity of the keys split from the _H / _S groups (round 4 I2, agent)
- **Options:**
  - (a) the whole group lnF, as now (6–24× too many saplings);
  - (b) **lnF_group + ln(the key's share of the group's train ingrowth)**;
  - (c) refit lnF per key before calibration.
- **Chosen:** (b). It is data-derived and cheap; calibration refits lnF anyway.
- **Implementation (2026-10-26, agent).** The shares are each key's TRAIN v2 sapling ingrowth density (D < 12.7 cm, w / f_micr), as a fraction of its HBAD3P
  group's. The remaining group key keeps its own share, so the group's total recruitment is kept.
  - Shares: OXAR 0.022 and QUPR2 0.003 of _H (_H keeps 0.975); PIEC2 0.037 and PIPA2 0.206 of _S (_S keeps 0.757).
- **Recruitment light at the subplot vs plot scale (the auditor's suspected convexity).** Rk = Σ (seed ? 1 : ε0) mean over subplots of e^{−λk RD_subplot} /
  Σ (seed ? 1 : ε0) e^{−λk RD_plot}, at the train v1 state, = **1.116–1.183** across the keys. It is more than 5 % away from 1, so lnF += ln Rk, as ε was
  rescaled for light.
- **Result:** predicted / observed v2 sapling tally on train at the warm start (verify 8.3; it was 6.1 / 6.6 / 2.1 / 23.9 for the four keys):
  - with Rk: 0.872 (OXAR) to 1.141 (LITU) for every key;
  - without Rk: 0.858–1.104.
  - Both are inside the 0.8–1.25 bar.
- **To reverse:** in params4.jl `warm_start4`, the share loop, and `rec && (f3 .+= log.(Rk))`.

## H9 — Lane-mixing κ in calibration (round 4 I3, agent)
- **Evidence:** simulate! on full train takes 1.46 s at κ = 0 and 13.2 s at κ = 0.1, about 9×.
- **Options:**
  - (a) κ free throughout, about 5× the wall time;
  - (b) κ free with 12 atoms when κ > 0;
  - (c) **two stages: stage 1 with κ = 0, stage 2 a κ profile {0.025, 0.05, 0.1} warm-started from stage 1 and judged on val counts; κ > 0 is kept only if val count NLL improves by ≥ 20 nats**;
  - (d) κ = 0 permanently.
- **Chosen:** (c). It costs about the price of κ = 0 and keeps the tripling hypothesis testable.
- **Implementation (2026-10-26, agent).**
  - Fixing κ, or any global, at a value: `HBAD4P_FIX="κ:0"` (env, as before), or `space4(…; fix_extra=Dict(:κ => v))`. κ starts at 0 anyway (H4).
  - **Stage 2 is `tools/hbad4/kappa_profile4.jl`.** It reads the stage-1 optimum (`save_z4` CSV). For **κ = 0 and each κ** of {0.025, 0.05, 0.1}, it runs the
    same short CMA-ES on all other free coordinates plus a short refit of the observation coordinates, then scores val counts. κ > 0 is kept only if it
    gains ≥ 20 nats.
  - κ = 0 is re-run with the same budget; scoring the stage-1 point as it stands would hand κ > 0 a free extra optimisation.
  - Tested only in a dry run (`HBAD4P_PROFILE_SUB=200`, 24 evaluations); no profile has been run.
  - Cost measured on the full train set: simulate! 1.5–2.8 s at κ = 0 vs 15–18 s at κ = 0.1 with 2 threads (4.0 s vs 33 s serial), about 8–9×.
- **To reverse:** run a single-stage fit with κ free (unset `HBAD4P_FIX`).

## H10 — Intervals whose v1 visit has no forested subplot (round 4 M1, agent, 2026-10-26)
- **Evidence:** 24 intervals (18 train / 3 val / 3 test at k1) have every subplot known and non-forest at v1. FIA tallies no tree there, so the v1 stand is
  unobserved: a non-forest → forest transition. `plotf` gave them f = 1.0, an empty stand scored as observed.
- **Options:**
  - (a) keep f = 1.0, i.e. an "observed" empty stand;
  - (b) keep them with zero exposure: an empty subplot layer, the sapling prior mean;
  - (c) **drop them, as round 3 I6 dropped the 102 intervals with no forested subplot at v2**.
- **Chosen:** (c). The model has no observed initial state to start from. The intervals are land-use changes, which HBAD4P's natural-dynamics scope excludes.
- **To reverse:** data4.jl `load4`, the `nf1` filter.

## H11 — Unobserved tallies (round 4 M1, agent, 2026-10-26)
- **Evidence:** in the intervals kept, every microplot was non-forest for 26 / 22 / 10 train / val / test intervals at v2, and for 34 / 23 / 10 at v1.
  - `plotf` set f = 1.0 for them.
  - At v2 they were scored as 4 empty microplots: 68.7 / 59.3 / 26.0 nats of count NLL at the warm start.
  - At v1 their sapling layer was the posterior of a zero count on a full microplot (≈ 100–108 /ha) instead of the prior mean (≈ 380–414 /ha). The sapling
    prior's fit counted them as zeros.
- **Options:** (a) f = 1.0 (old); (b) the FMIN floor 0.25; (c) **zero exposure, f = 0**.
- **Chosen:** (c).
  - At v2: no count term and no size / height term for that tally.
  - At v1: e = 0, so the sapling cells take the prior mean m.
  - The prior's NB fit excludes those plots.
  - A tree on a tally with f = 0 would be skipped (counted, `nzt`); there are none.
- **To reverse:** data4.jl `plotf` (`m <= 0 ? 0.0`).

## H12 — The roughness metric of verify 3.3 (round 4 V, agent, 2026-10-26)
- **Options:** the sd of the step differences around their mean (the auditor's exp2 / exp8), or **around their linear trend**, i.e. the slope of a quadratic.
- **Chosen:** the linear trend, which is "around the smooth slope". The curvature of a smooth objective is not roughness. The mean-centred value is reported
  alongside and also passes: 2.67 vs the 2.7 bar on full train at 1e-3.
- **To reverse:** verify4p.jl 3.3 (`rd`).

## H13 — The κ-effect criterion of verify 3.2c (round 4, agent, 2026-10-26)
- **Evidence:** with the round-4 likelihood and warm start, the 64-atom κ effect (κ = 0.1 vs 0) at the warm start is −0.0145 nats/site; in round 3 it was
  +0.170. NATOM 16 gives −0.0093.
  - The ratio 0.64 fails ±30 %, but the absolute difference (0.005 nats/site) is the merge error itself, already within 0.02.
- **Options:** the ratio only; **the ratio or an absolute floor of 0.01 nats/site**; the absolute difference only.
- **Chosen:** ±30 % or 0.01 nats/site, whichever is larger. A relative criterion is ill-conditioned for an effect near 0. The floor is half the 0.02
  nats/site merge-error bar.
- **To reverse:** verify4p.jl 3.2c.

## H14 — The roughness check after round 4 (coordinator, 2026-10-27)
- **Evidence:** with the observation parameters fitted at the warm start, the total objective's step sd at 1e-3 is 2.78 (bar 2.7) and its largest 1e-4 jump
  is 2.71 (bar 2). Of that, the count term alone has sd 2.46–2.51 at 1e-3. That is the M2 roughness, accepted. The I1 bar was written for the size / height
  scoring defect.
- **Options:**
  - (a) amend the bar on the total;
  - (b) a smooth cell-membership model change now;
  - (c) **split check 3.3 by likelihood term**.
- **Chosen:** (c), coordinator decision.
  - verify 3.3 enforces the I1 bar on size + height combined: sd ≤ 0.93 / 2.7 at 1e-4 / 1e-3, no jump > 2 nats at 1e-4.
  - 3.3b enforces a regression bar on the count term: no worse than the values measured on 2026-10-27 + 10 %, recorded in `COUNT_REF`.
  - 3.3c reports the total as INFO.
- **Open lever:** smooth cell membership in the dynamics (an atom's rates and height continuous across a class bound) is the remaining way to lower the
  count-term roughness. It is a model change, for a later round.
- **Residual noise scale CMA-ES will see:** about 2.5–2.8 nats per 1e-3 step of a coordinate on full train, mostly the count term.
- **Result (2026-10-27, k1, runtime-fitted observation parameters; step sd / largest jump):**

  | Series | size + height | size | height | count | total |
  |---|---|---|---|---|---|
  | ln ε PITA at 1e-4 (full train) | 0.549 / **2.62** | 0.219 / 0.83 | 0.494 / 2.59 | 0.159 / 0.55 | 0.603 / 2.71 |
  | ln ε PITA at 1e-3 (full train) | 1.619 / 4.87 | 1.616 / 4.35 | 1.162 / 4.16 | 2.509 / 8.37 | 2.780 / 10.24 |
  | κ at 1e-4 (5000 sites) | 0.045 / 0.13 | — | — | 0.048 / 0.14 | 0.064 / 0.17 |
  | κ at 1e-3 (5000 sites) | 0.421 / 1.94 | — | — | 0.355 / 1.69 | 0.553 / 2.32 |

  - The size + height sd bars are met (0.55 ≤ 0.93, 1.62 ≤ 2.7). **The no-jump-over-2-nats bar at 1e-4 is NOT met** (2.62, from the height term, 2.59).
  - **Not tuned, as instructed.** 3.3 is reported as FAIL. The count values are recorded as `COUNT_REF`.
- **To reverse:** verify4p.jl 3.3 / 3.3b / 3.3c.

## H15 — The single size + height jump of verify 3.3 (coordinator, 2026-10-27)
- **Evidence (H14 run):**
  - Both size + height sd bars pass: 0.549 ≤ 0.93 at 1e-4 and 1.619 ≤ 2.7 at 1e-3 (ln ε PITA, full train).
  - One 1e-4 jump of 2.62 nats exceeds the 2-nat bar; 2.59 of it is the height term.
  - Its source is the one-height-per-cell dynamics: an atom that crosses a class bound takes its new cell's height.
  - The jump is below the ≈ 2.5–2.8-nat per-1e-3-step noise that CMA-ES sees anyway.
- **Options:**
  - (a) widen the jump bar;
  - (b) the smooth cell-membership model change now;
  - (c) **a named, bounded known exception**.
- **Chosen:** (c), coordinator decision, the same mechanism as curation D6.
  - verify4p.jl `KNOWN_EXCEPTIONS` has one entry: the single-jump bar of size + height on ln ε PITA at 1e-4, recorded value 2.62, slack 1.0.
  - That cell prints KNOWN while the jump is ≤ 3.62 and FAILS beyond.
  - Every other bar stays strict: both sd bars, κ, the 1e-3 series, the count regression 3.3b.
- **To reverse:** remove the entry; (b) is the lever that would make it unnecessary.

## H16–H21 — Round-5 audit fixes (2026-10-27, agent; the implementer records the details)
- **H16, ρ encoding:** log-encoded on [1e-7, 0.05], as in HBAD3P, plus check 8.7 auditing every coordinate's encoding.
  - Option rejected: a linear encoding with a smaller σ for ρ only.
- **H17, atom heights in the observation:** centred so that ΣNH_m = ΣNH, the same invariant the upgrowth rule keeps.
  - Option rejected: leave the bias for calibration to absorb into η / λ / μ.
- **H18, recruitment convexity:** Rk weighted by the forested fraction, as HBAD3P recruits per forested subplot; 8.3 tests recruits rather than the whole sapling tally.
- **H19, stage-1 tool:** calib4.jl, and save / load refuse a mismatched parameter space.
  - Option rejected: rely on matching environment variables.
- **H20, roughness on every coordinate:** every global coordinate plus 3 per-key ones; per-coordinate KNOWN entries with +20 % slack; references keyed by (role, cache key), and an unreferenced combination is INFO.
  - Option rejected: k1 numbers applied everywhere.
- **H21, κ decision:** 3 seeds per κ; keep κ > 0 only if min(κ > 0) beats max(κ = 0) by ≥ max(20, 2 × (replicate sd + merge error)).
  - Options rejected: a single run with a 20-nat margin; no stage 2.

## H22 — Generalisation beyond Region 4 (2026-10-27, agent; recorded for the expansion phase)
- **Generic:** the kernel, the likelihood, the encodings and the κ plan.
- **Must be re-derived per region:**
  - the tallies assume a subplot / microplot design; PNW / CA need a macroplot tier (16.6 % of live trees at TPA 0.999);
  - the HBAD3P warm start exists for GA+FL only;
  - Tref and climate ranges would be extrapolated.
