# HBAD3P pipeline — one place to run, document and check the model for any region

Driver: `tools/hbad3/pipeline3p.sh <REGION> <TAG> [stage ...]`, which runs the stages below in this order. Configuration: `tools/hbad3/pipeline3p.env`. It is the
only place for switches and input paths, and every path derives from `WORKSPACE`, the parent folder of this repo. Shared code: `tools/hbad3/lib3p.jl`. Kernel:
`src/plugins/HBAD3/HBAD3P.jl`; its header has the full equations. Logs: `results/hbad3/pipeline_<TAG>/<stage>.log` and `pipeline.log`.

```bash
source wt_env.sh
tools/hbad3/pipeline3p.sh GAFL P3_GAFL                         # everything, one region
tools/hbad3/pipeline3p.sh L3:8.3.5 P3_L3_835 fit recruit freerun # some stages; REGION = GAFL | APP | IP | WSP | NCSC | L3:x.y.z
HBAD3P_FIX=gR:0.0 tools/hbad3/pipeline3p.sh GAFL P3gr0_GAFL fit  # optional fit switches pass through the environment
```

## Stages

| stage | script | inputs | outputs |
|---|---|---|---|
| `data` | Pan R4 builder `src/data/hbad2_intervals.jl` in `$HBAD3_PAN_R4`; this stage copies the cache and checks it byte for byte | slim_r4.duckdb `curated_trees`, R4 split | `results/hbad2_r4/{intervals,interval_trees,interval_ingrowth}.arrow` |
| `climate` | `scripts/hbad3/climate_join.py` | PRISM L4 monthly, plot L4, plot soil | `results/hbad3/{plot,interval}_climate_r4.csv` |
| `water` | `scripts/hbad3/water_balance.py` | PRISM, AWC | `results/hbad3/interval_water_r4.csv` (period AET/PET `W`, plot normal `Wn`) |
| `forest` | `scripts/hbad3/subplot_forest.py` | FIADB `SUBP_COND × COND × PLOT` | `results/hbad3/subplot_forest_r4.csv` (forested fraction of each subplot / microplot visit; NaN = unknown) |
| `vdata` | `tools/hbad3/verify3p.jl data` | the products above | PASS/FAIL list (D1–D6) |
| `fit` | `tools/hbad3/fit3p.jl <TAG>` | tracked trees of clean intervals, region `HBAD2_REGION` | `results/hbad3/fit3p_<TAG>/` with `species.csv`, `globals.csv`, `z.csv` (named), `meta.txt`, `log.txt` |
| `recruit` | `tools/hbad3/recruit3p.jl <TAG>` | fit, sapling ingrowth, forest table | `fit3p_<TAG>/recruitment.csv`, `recruit_log.txt` |
| `freerun` | `tools/hbad3/freerun3p.jl <TAG> val` | fit + recruitment | `fit3p_<TAG>/freerun_val.csv` and a summary table in the log |
| `crossing` | `tools/hbad3/crossing3p.jl <TAG>` | fit | in-growth diagnostic: survival and growth into 12.7 cm |
| `vmodel` | `tools/hbad3/verify3p.jl model <TAG>` | everything of the fit | PASS/FAIL list (K, L, F, R, FR) |

The R4-wide products (`data` … `forest`) are built once and skipped when present (`FORCE=1` rebuilds them). Every stage after `fit` reads the fit's `meta.txt`:
growth form, size split, site unit, vigour quadrature, size-class rule `dclass` and density convention `farea`. A stage refuses to run when the environment
disagrees with the meta file. Evaluate an older fit with its own rules, for example `HBAD2_DCLASS=legacy` for fits without a `dclass` line. `farea` is
read from the meta file, so no switch is needed for it.

## Data conventions (verified by `vdata`)

- **Intervals.** Consecutive visits of an FIA plot. Fits and free runs use only clean intervals: no treatment, disturbance or harvest code at v2, and
  dt > 2 yr. The split is by plot: every plot is in exactly one of train / val / test.
- **Tracked trees.** Live at v1 and followed to v2 as alive, dead, harvest or missing. Annual FIA design: D < 12.7 cm is tallied on the microplot
  (74.965 TPA); 12.7 cm and above on the subplot (6.018 TPA). Sites are subplots, so densities ×4.
- **Cohorts.** One cohort per subplot × species key × birth decade × diameter class. Diameter classes are 5 / 12.7 / 25 / 40 cm and **closed below**, so a
  12.7 cm tree is never lumped with microplot saplings (`dclass=lower`). Each cohort is simulated as 5 vigour sub-cohorts (Gauss–Hermite nodes).
- **Forested area.** FIA tallies trees only on forest conditions. With `farea=1` a subplot's stand density is per hectare of its **forested** part:
  N / a, with a = max(f_subp, 0.25). If f is unknown, or 0 although trees were tallied, a = 1. Survival ratios do not change; competition sees the real density.
- **Water.** W is the growing-season AET/PET of the interval's years; Wn is its 1991–2020 plot normal.

## Likelihood (Sim A: `fit`)

- Binomial survival of the tracked trees: `n1` counted (not missing), `s` alive; p = ΣN / N0 of the cohort's sub-cohorts.
- Gaussian error of the cohort-mean ln D2 and ln H2 of survivors (QMD and mean height of the survivors), with precision `s`. σ is profiled, which gives
  0.5 c ln(Σ s e² / c).
- Null baseline: per-key annual survival from train, with no growth.
- Fit by CMA-ES. Per key only ln ε is fitted; everything else is global (30 coordinates; `HBAD3P_FIX` holds some fixed). Warm starts (`HBAD3P_START`)
  match coordinates **by name**.

## Recruitment (`recruit`), consistent with the free run

The observations: Y = saplings (2.54 ≤ D < 12.7) of key k first tallied at v2 on the forested microplot. One row per train natural clean interval × subplot
(microplot status known, not nonforest-and-empty at v1) × key. In the free run a recruit enters at 2.54 cm at the end of every year: the step comes
first, then recruitment, and a final partial year counts with weight Δ. After entry the recruit lives under the fitted kernel. Hence

  E[Y] = e^{lF_k} · f_micr(v2) · Σ_y Δ_y · e^{−λ0 e^{λ1 (ST_k − 3)} RD_y} · (seed ? 1 : ε0) · s_ky,

- RD_y and seed_ky describe the stand at entry: RD after the year's step; seed = a conspecific tree ≥ 12.7 cm on the plot at v1, or a simulated cohort
  that reaches 12.7 cm.
- s_ky is the fraction of key-k recruits entering in year y that are alive with D < 12.7 at v2. It comes from **probe recruits**: negligible density,
  simulated inside the observed v1 stand.
- ln F (per ha) = lF + ln 4, because one microplot sapling of a subplot site is 4 × RSCALE = 4 × 185.2 trees/ha.
- The fit is Poisson, by CMA-ES. Its log reports the val deviance against a per-key constant rate and observed/expected counts by key, RD class and seed.
- The earlier version used E[Y] = e^{lF} dt e^{−λ RD1}. It treated the survivors at v2 as the entry rate, so the free run killed them a second time
  (mean probe survival 0.92). It also included empty subplots only when they had ingrowth.

## Free runs (`freerun`)

- Every clean natural plot of the set with ≥ 2 intervals runs from its first to its last visit (whole years plus the fraction), with 4 subplot sites in
  the fit's density convention.
- Recruitment area: rmult = f/a (farea=1), f (farea=0), or 0 for a subplot that is nonforest and empty at the first visit.
- Plot values = Σ sites N·a/4: BA, stems ≥ 12.7 cm (a lognormal within-cohort spread σ 0.16 with median QMD·e^{−σ²}), their QMD and Lorey height.
  They are compared with the observed last visit and with persistence (first-visit values). Plots whose simulation has no stem ≥ 12.7 cm are counted
  when they drop out of the QMD and Lorey-height rows.

## Verification

**`vdata`:**
- D1: identifiers, dates, outcomes, FIA design weights, RSCALE.
- D2: clean intervals and split.
- D3: cohorts recomputed from trees; the quadrature; contiguous sites; no cohort mixing microplot and subplot trees.
- D4: water coverage and range.
- D5: forest status known; no sapling on a nonforest microplot; area scaling leaves the weights unchanged.
- D6: traits.

**`vmodel`:**
- K1: `step_site!` equals an independent reference implementation of the documented equations, on 300 random sites with ties, planted stands, partial
  years and self-thinning.
- K2: invariants on the val sites; bitwise determinism; self-thinning convergence; the whole-year and partial-year paths agree.
- L1: the profiled σ equals a numerical optimum.
- L2: the survival term is recomputed.
- F1: the saved parameters reproduce the logged FIT totals.
- F2: `z.csv` coordinates are named.
- R0–R2: recruitment log complete; probe exposure × fitted rate = survivors under `run_free!` itself; probe survival < 1.
- FR1: at t = 0 the free-run sites reproduce the observed plot BA and stem density exactly.
- FR2: the non-recruiting sites are exactly the nonforest empty subplots, counted independently.

## Known limitations (documented, not bugs)

- The likelihood is on cohort means. The spread of D within a cohort carries no information, so σv (vigour) is weakly identified and partly
  confounded with ε.
- Free-run forest status is frozen at the first visit. Plots without any tracked tree at v1 are in neither the recruitment fit nor the free runs.
- Saplings are scaled by f_subp although they are tallied on f_micr; the two differ rarely.
- Recruitment uses seed **presence**, not seed-source abundance. Val calibration by key is poor (e.g. PITA observed 828 vs expected 307 in GA+FL),
  and the light window e^{−λ RD} under-predicts open stands.
- 122.98 in the growth equation is a fixed unit constant of ε. Older docs called it 20^1.605, but 20^1.605 = 122.50.
