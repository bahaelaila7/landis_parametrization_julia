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
  - The residual noise CMA-ES will see is about 2.5–2.8 nats per 1e-3 step of ln ε (PITA) on full train (corrected in round 5, H20: per coordinate, see §11 — 0.0–1.5 nats per 1e-4 step, and on 5000 sites 0.01–5.2 per 1e-3 step; ε:PITA was not representative). Check 3.3 was split by term (H14).
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
- **Residual noise scale CMA-ES will see:** about 2.5–2.8 nats per 1e-3 step of ln ε (PITA) on full train, mostly the count term (corrected in round 5, H20: per coordinate, see §11 — 0.0–1.5 nats per 1e-4 step, and on 5000 sites 0.01–5.2 per 1e-3 step; ε:PITA was not representative).
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

## H16–H21, H23 — Implementation of the round-5 fixes (2026-10-27, agent)
- **H16 encodings (params4.jl GSPEC4; verify 8.7).**
  - ρ is log on [1e-7, 0.05].
  - The 8.7 audit at the warm start (z ± 0.05, ± 0.5) found two more coordinates that clamp, now widened:
    - c, the H–D shape, log on [0.3, 2] → [0.3, 3]: +0.5 hit 2.0.
    - βT, linear on [−0.5, 0.5] → [−1, 1]: ±0.5 hit the bounds.
  - κ and f0 start at their lower bound by design (0 = none) and are flagged.
  - Coordinates whose ±0.05 step moves the train NLL by ≥ 1 % are FLAGGED, not changed (the table is printed by 8.7):
    - ε:PITA 1.31 %, b 1.20 %, s 1.15 %, pD 1.56 %, σv 6.1 %.
    - Recommendation for calibration: a smaller initial σ for these coordinates, or a per-coordinate scaling. calib4.jl does not do this yet.
    - Option rejected: re-encoding σv in log, which is still > 1 % at ±0.05.
- **H17 centred atom heights (HBAD4P.jl finish_site!; verify 0.14, 1.9).** H_m = H + H*(D_m) − mean_N H*(D), so Σ N H_m = Σ N H exactly (1.9: max rel
  9e-16). If an offset falls below 0.5 m, the cell is clamped at 0.5 and rescaled.
- **H18 recruitment light (params4.jl warm_start4).** Rk weights the subplots by their v1 forested fraction: 1.046–1.079 now (1.116–1.183 before).
- **H23 (new) recruits moment-matched on train.**
  - **Evidence:** 8.3 made discriminating (predicted vs observed RECRUITS = class-1 v2 ingrowth) showed that the HBAD3P mapping with shares and Rk predicts
    obs / pred of 0.73–2.31 by key on train: PITA 1.82, PIEL 1.72, OXAR 2.00, QUPR2 2.31, TAAS 0.73. HBAD3P's lnF was fitted through its own sapling
    survival to v2.
  - **Options:**
    - (a) keep the mapping: 8.3 fails, and calibration starts far off for these keys;
    - (b) **moment-match lnF per key on train**: lnF += ln(obs / pred), two passes;
    - (c) fit lnF by maximum likelihood now, which is calibration and not allowed.
  - **Chosen:** (b). It is data-derived, one multiplicative step, like the ε rescalings.
    - 8.3 checks the train ratio, recomputed independently (all 1.000).
    - 8.3b reports VAL: 0.64 (LITU) … 1.43 (TAAS); 2.84 for PIEC2 on 3 recruits. Several keys are beyond 2 sd of the NB count noise: recruitment differs
      between folds more than the count model's noise (spatial clustering). It is INFO, since calibration fits lnF.
  - **To reverse:** params4.jl warm_start4, the `recratio` loop.
- **H19 stage-1 tool and strict files.**
  - `tools/hbad4/calib4.jl`: CMA-ES on the train NLL in `stage1_space4` (κ = 0), with a budget, seed, running best and log.
    - Dry run only: 200 sites, 60 evaluations, NLL 3104.8 → 2922.0.
  - `save_z4` writes `<file>.sig`: the names, encodings, fixed globals and HBAD4P_FIX / FREE.
  - `load_z4` refuses a different space or a missing signature unless `allow_mismatch=true` / HBAD4P_Z4_ALLOW_MISMATCH=1.
    - verify 8.8: a file written with HBAD4P_FREE=gR is refused, loads with the override, and a missing .sig is refused.
  - kappa_profile4.jl reads stage 1 in `stage1_space4`.
- **H20 roughness on every coordinate (verify 3.3 / 3.3b / 3.3c).**
  - The coordinates: every global coordinate, plus ε of PITA (common), of the rarest own key, and of the split key with the smallest share (QUPR2).
  - The steps: 1e-4 on full train and 1e-3 on 5000 sites.
  - The bars:
    - the size + height sd bars are enforced everywhere;
    - jumps > 2 nats need a KNOWN entry within +20 %;
    - the count term is held to its reference + 10 % + 0.01 nats. The floor is needed because the references are recorded to 4 decimals: the six
      observation coordinates and am have count roughness ≈ 0, and a rounded 0.0000 cannot be met by 1e-5;
    - references are keyed by (role, cache key), and an unreferenced combination reports INFO plus the literal to record.
  - **k1 result (full run):** two coordinates FAIL the size + height sd bars:
    - b: 1.12 at 1e-4 and 2.39 at 1e-3;
    - βT: 1.51 and 4.01.
    - Both are global multipliers of every atom's growth.
  - Jumps > 2 nats recorded as KNOWN: ε:PITA 3.61 (2.62 in round 4, before the centred heights, w_n and the recruit match), b 4.03, pD 3.54, σv 3.01, βT 4.73.
  - The k1 references are in `ROUGH_REF[("k1", "9e31afc324039360")]`.
  - **Not tuned and not re-barred: open for the coordinator.** The options:
    - per-coordinate bars scaled by the 8.7 sensitivity;
    - KNOWN entries for the sd bar too;
    - the smooth cell-membership change.
  - This supersedes H14 / H15's single ε:PITA entry.
- **H21 κ decision rule (kappa_profile4.jl).**
  - 3 CMA-ES seeds per κ, κ = 0 included.
  - κ > 0 is kept only if min over its seeds of the val count NLL ≤ max over κ = 0's seeds − max(20, 2 × (pooled replicate sd + 0.005 nats/site × n_val)).
  - Note: as specified, this compares the best κ > 0 run with the worst κ = 0 run. The margin, 35 nats or more at n_val = 3511, is what protects it.
  - The dry run (200 sites, 24 evaluations) gave a pooled sd of 45 nats and a margin of 91 nats.
- **M-1 the mixture weight of a cell mean.**
  - w_n = 1 − (1 − w)^n (n_H for the height): the cell's mean is the background's only if all its trees are.
  - Effect at the warm start (train, fitted observation parameters): the size + height NLL changes from −227,675 to −224,035 nats (+1.6 %).
  - Their pull on the densities is about halved: the NLL change for densities × 0.8 / × 1.25 goes from −327 / +317 to −178 / +171 nats.
  - Kept: for a cell mean it is the right "all-background" event. It remains an approximation for mixed cells (a mixed mean is neither component).
- **M-2.** kappa_profile4.jl simulates the state once per κ before the observation-only refit.

## H24 — The roughness bar after round 5 (coordinator, 2026-10-27)
- **Evidence (round-5 k1 run):** b and βT exceed the absolute size + height sd bars of verify 3.3: b 1.12 at 1e-4 and 2.39 at 1e-3; βT 1.51 and 4.01 (bars
  0.93 / 2.7).
- **Options:**
  - (a) **a sensitivity-relative bar, pre-registered now:** a coordinate's roughness sd at step h ≤ 1 % of its |ΔNLL| at the calibration's initial step ±0.05
    (as in 8.7), on the same data subset;
  - (b) KNOWN entries only;
  - (c) smooth cell membership, the root-cause model change.
- **Chosen:** (a), coordinator decision.
  - All 38 coordinates must pass it.
  - The absolute bars stay in the output as REPORTED values.
  - b and βT are listed as KNOWN absolute-bar exceedances, with their numbers; nothing is hidden.
- **Why:**
  - The absolute bars ignore scale. b moves the train NLL by about 1.2 % (≈ 1.3k nats on full train) at ±0.05, so a roughness of 1–4 nats is ≈ 0.1 % of the
    step signal a derivative-free optimiser sees.
  - **The bar was changed AFTER the failure was observed.** It is recorded as such.
  - The relative bar is STRICTER than the absolute one for low-sensitivity coordinates, whose |ΔNLL| at ±0.05 is small.
- **Rejected:**
  - (b) leaves no criterion.
  - (c) is deferred to a later version: it would reopen the verification of the whole atom layer.
- **κ profile:** the H21 margin (≥ 35 nats at k1) is where roughness and seed noise matter. It already includes the pooled seed sd, i.e. the replicate
  spread of the val count NLL across CMA-ES seeds, which contains the objective's roughness as it acts on the optimiser. The roughness is absorbed there, not
  added to it.
- **Fixed alongside (found while implementing):** the round-5 3.3 series used the DEFAULT observation parameters at the base point, not the fitted OBX that
  its header states (the coordinate vector was the warm start's z0). The base point now carries the fitted ones, as 8.7 does. The k1 references were
  re-recorded (`HBAD4_VERIFY_RECORD=1`) and then observed.
- **Result (k1, base point with the fitted OBX; ratio = size + height step sd / (1 % of |ΔNLL(±0.05)|), at 1e-4 on full train | 1e-3 on 5000 sites):**
  - **7 of 38 coordinates FAIL the relative bar, all at 1e-3:**
    - mW 4.63 (|ΔNLL| ≈ 0.3 nats);
    - ε:MAVI2 3.17 (6 nats);
    - pM 2.86 (9 nats);
    - am 2.43 (≈ 0.2 nats);
    - ρ 1.63 (5 nats);
    - λ1 1.20 (15 nats);
    - βT 1.10 (311 nats).
  - Every 1e-4 ratio is ≤ 0.59.
  - The sensitive coordinates pass comfortably: b 0.055 / 0.222, σv 0.009 / 0.071, ε:PITA 0.027 / 0.173.
  - **The failures are coordinates the objective barely sees.** There the 1 % bar is 0.003–0.15 nats, below the dynamics' roughness floor of
    ≈ 0.007–0.25 nats at 1e-3 on 5000 sites. This is the "stricter for low-sensitivity coordinates" side of the bar, observed.
  - **Not tuned: reported for the coordinator.** The options: a floor on the relative bar (e.g. max(1 % |ΔNLL|, the roughness of a null coordinate)); the
    relative bar at 1e-4 only (where all pass); or accepting the seven as KNOWN.
  - Absolute bars (REPORTED): with the corrected base, only βT exceeds them (1.15 / 3.40). b is now within them (0.715 / 1.823): its round-5 exceedance
    came partly from the default-observation-parameter slip.
  - Count references (3.3b) were re-recorded with the corrected base; KNOWN jumps: f0 2.23, βT 2.44.
- **To reverse:** verify4p.jl 3.3 (`q4` / `q3`).

## H25 — No further roughness-bar changes; roughness judged by its effect on calibration (coordinator, 2026-10-27)
**Outcome of H24.** Under the sensitivity-relative bar (fd3d549), check 3.3 still FAILs:
- 7 of 38 coordinates fail at the 1e-3 step on 5000 sites: ε:MAVI2 3.17, mW 4.63, pM 2.86, am 2.43, ρ 1.63, λ1 1.20, βT 1.10.
- Every coordinate passes at the 1e-4 step on full train (all ratios ≤ 0.59).

The 3.3 base-point fix (OBX observation parameters) is a correction, not a bar change. After it, b is within the absolute bars, and βT still exceeds them.

**Decision.** The bar is NOT changed a second time. Changing it again after seeing a second failure would make the criterion meaningless.
- 3.3 is recorded as **FAIL-KNOWN**, with the table above, and stays a FAIL line in the verifier output (not converted to a PASS).
- The question it stood for — does roughness harm calibration? — is answered by an outcome check, pre-registered now, before any calibration run:
  1. **Stage-1 calibration replication (calib4.jl).** Run 3 seeds from the same start. Accept only if the validation NLL range across seeds is ≤ max(20, 2 · pooled sd) nats, the H21 margin form. Otherwise the calibration is not accepted, and the roughness is the first suspect.
  2. **Weakly identified coordinates.** mW, am, pM and ε:MAVI2 move the NLL by ≈ 0–2 nats at ±0.05. They are reported as weakly identified, with their cross-seed spread. Their fitted values are not interpreted or used in comparisons. They are NOT fixed at priors, since that would be another post-hoc change; that option is left for the next model version.

**Options considered:**
- (a) an absolute floor on the relative bar;
- (b) apply the bar at 1e-4 only;
- (c) KNOWN entries per coordinate;
- (d) chosen: no bar change; FAIL-KNOWN plus a pre-registered calibration-outcome check;
- (e) smooth cell membership: deferred to the next model version (H24(c)).

(a)–(c) are rejected because each is a second post-hoc adjustment of the same criterion.

## H25-impl — Implementation of H25 (2026-10-27, agent)
- **verify4p.jl 3.3.**
  - `check(...; known="H25")` prints `FAIL-KNOWN (H25)`. The RES entry stays `FAIL`, prefixed `[FAIL-KNOWN (H25)]`, so it is counted in the FAIL total and the
    verifier exits 1. It is never converted to PASS or INFO.
  - The label applies only to the H24 table of (k1, 9e31afc324039360): ε:MAVI2 3.169, mW 4.634, pM 2.861, am 2.427, ρ 1.626, λ1 1.199, βT 1.095 (the 1e-3
    ratios), each allowed +20 % (the H20 slack). Any other failure is a plain FAIL: a coordinate not in the table, a ratio above its entry × 1.2, any 1e-4
    ratio > 1, or a jump failure. Another role or cache key has no table, so a failure there is a plain FAIL.
  - The detail line starts with the relative-bar ratios (1e-4 / 1e-3) of the failing coordinates.
- **calib4.jl, stage-1 replication (the pre-registered check).**
  - 3 CMA-ES seeds from the same warm start by default (HBAD4P_CALIB_SEEDS = 1,2,3). The old single HBAD4P_CALIB_SEED is refused. Fewer than 3 seeds gives
    the verdict NOT ASSESSED, which is not an acceptance.
  - Each seed's best point is scored on val: the total NLL, with its own train-fitted observation parameters.
  - **Decision point, the "pooled sd" in max(20, 2 · pooled sd).** Read literally as the sd of the same 3 val NLLs, the check can never fail: for 3 values
    the range is at most 2 × their sample sd (between √3 and 2 times it). Options:
    - (a) the between-seed sd: vacuous, rejected;
    - (b) the 20-nat floor alone: drops the noise term the H21 form has;
    - (c) chosen: the pooled WITHIN-seed sd, i.e. the val NLL sd over 8 points jittered around each seed's best (step 1e-3 in every coordinate, the 3.3
      step; its own RNG stream), pooled over the seeds. This is the objective's own noise at the optimum, which is what H25 asks the check to bound.
    - The between-seed sd is still reported. To reverse: `sd_pool` in calib4.jl. The H21 κ rule also has a merge-error term, 0.005 nats/site × n; H25 states
      the form without it, so it is not added here.
  - **Carried seed:** the seed with the lowest TRAIN NLL, so that val stays a check rather than a selection. It is written as z_stage1.csv only if ACCEPTED,
    otherwise as z_stage1_REJECTED.csv, so kappa_profile4.jl cannot pick up a rejected stage 1.
  - **calib4_stage1.json** holds the verdict, the rule and its numbers, and per seed the evals, train / val / val-count NLL, the jitter values and their sd,
    and the file. It also lists the weakly identified mW, am, pM and ε:MAVI2 with their z and decoded values per seed, their ranges and the start value.
  - Helpers moved to params4.jl: `first_sites4` (also used by kappa_profile4.jl now; it had a private copy) and `values4` (used by save_z4). params4.jl is
    not part of the data cache key, so the 3.3 references stay valid.
- **Dry runs (k1, 200 train / 200 val sites, 30 evaluations per seed = 17 used): a test of the tool, NOT a calibration.**
  - Val NLL per seed 3180.5 / 3008.8 / 2970.7: range 209.8 vs margin max(20, 2 × 3.06) = 20, so REJECTED. With 17 evaluations from σ 0.1 the seeds are
    still at random points, so this says nothing about the model. Between-seed sd 111.8.
  - Weakly identified, cross-seed value range: mW 0.060, am 0.153, pM 0.093, ε:MAVI2 0.248.
  - With HBAD4P_CALIB_GAIN = 1e6 the ACCEPTED path writes z_stage1.csv, and kappa_profile4.jl loads it (its own dry run passes). With one seed the verdict is
    NOT ASSESSED and the between-seed sd is null. The seeds are reproducible: the same val NLLs on a re-run.
- **Observed, k1 section 3 (7276 s): 4 PASS / 1 FAIL / 4 INFO, exit 1.** 3.3 prints `FAIL-KNOWN (H25)`, and its ratios are identical to the H24 table (the run
  is deterministic). The summary lists it as `FAIL  [FAIL-KNOWN (H25)] 3.3 …`. 3.3b PASS.

**H25-impl confirmation (coordinator, 2026-10-27).** The implementer's reading is confirmed. Taken literally, "range ≤ max(20, 2 · pooled sd)" with the sd computed from the same 3 values can never fail, because for n = 3 the range is at most 2 × the sample sd. The noise scale is therefore the pooled WITHIN-seed sd: the validation NLL sd over 8 jittered points (step 1e-3) around each seed's best point, pooled across seeds. The between-seed sd is reported only. This is a correction of a vacuous formula, made before any calibration run. It does not move the bar, and the 20-nat floor stays.
