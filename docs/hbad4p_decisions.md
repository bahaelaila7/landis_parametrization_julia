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
  | 16 | 0.000 | +0.004 | at κ = 0.1, 1.4–1.6× slower than the old kernel |

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

## H8 — Fecundity of the keys split from the _H / _S groups (round 4 I2, agent)
- **Options:**
  - (a) the whole group lnF, as now (6–24× too many saplings);
  - (b) **lnF_group + ln(the key's share of the group's train ingrowth)**;
  - (c) refit lnF per key before calibration.
- **Chosen:** (b). It is data-derived and cheap; calibration refits lnF anyway.

## H9 — Lane-mixing κ in calibration (round 4 I3, agent)
- **Evidence:** simulate! on full train takes 1.46 s at κ = 0 and 13.2 s at κ = 0.1, about 9×.
- **Options:**
  - (a) κ free throughout, about 5× the wall time;
  - (b) κ free with 12 atoms when κ > 0;
  - (c) **two stages: stage 1 with κ = 0, stage 2 a κ profile {0.025, 0.05, 0.1} warm-started from stage 1 and judged on val counts; κ > 0 is kept only if val count NLL improves by ≥ 20 nats**;
  - (d) κ = 0 permanently.
- **Chosen:** (c). It costs about the price of κ = 0 and keeps the tripling hypothesis testable.
