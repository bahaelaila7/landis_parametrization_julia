# Why CMA-ES stalled on the calibration, and the Group-Decomposed Igel CMA-ES (`search_mode: gdigel`)

Date 2026-10-13. Branch `cmaes-gafl` (on top of `hbad`: all three succession plugins). Config family: GA+FL, Sim A
(`configs/region4/gafl_A.yml`, `gafl_pnet.yml`, `gafl_hbad.yml`).

## 1. Diagnosis — "too noisy" is not simulation noise
`tools/cmaes_diag.jl <config> <params> <out>` prepares the training context exactly like a search (run_from_yaml + its post_search_hook)
and, around the calibrated parameter set of each plugin, measures (A) repeat-evaluation noise, (B) 1-D slices of the
objectives along random u-directions at coarse (|t| ≤ 0.08) and fine (|t| ≤ 0.004) scale, (C) offspring clouds at
σ ∈ {0.08, 0.02, 0.005} — Δ of the training objectives [ΣW, ΣAGB] vs Δ of the per-species bias loss (what the TOST tables
measure), (D) dead u-dimensions.

| | Biomass Succession (224 dims) | PnET (282) | HBAD (163) |
|---|---|---|---|
| A. different seeds | identical (deterministic Sim A) | ΣAGB ±7e-4 (stochastic) | identical |
| B. fine-scale residual SD / range of ΣAGB (t ≤ 0.004) | 1.5e-4 / 7.5e-4 (rough: ≈ signal) | 2.5–4e-4 / 1.5–3.8e-3 | 0.5–3e-4 / 0.3–1.1e-3 |
| B. coarse-scale residual / range (t ≤ 0.08) | 2–5e-4 / 0.9–2.5e-2 | 5–10e-3 / 7–8e-2 | 3–6e-4 / 1.4–2.5e-2 |
| C. offspring dominating the parent (σ = 0.08 / 0.02 / 0.005) | 1/12, 1/12, 1/12 | 0/12, 1/12, 1/12 | 1/12, 2/12, 2/12 |
| C. corr(ΔΣAGB, Δ per-species bias loss) | **−0.66, −0.26, −0.73** | +0.27, +0.39, +0.28 | +0.31, +0.54, +0.03 |
| C. corr(ΔΣW, ΔΣAGB) at σ = 0.08 | −0.35 | +0.57 | +0.76 |
| D. dead dims (±0.02 in u changes nothing) | 1 | 0 | 0 |

What actually stalls the search (in order of importance):
1. **Aggregation of ~50 independent sub-problems.** Each species group's AGB bias depends mostly on its own 1–4
   parameters (shown by the coordinate calibration). A CMA-ES offspring perturbs all of them at once and is accepted only if
   the AGGREGATE improves; the per-group changes have random signs, so the aggregate change is a sum of ~50 ±terms —
   it behaves like noise, success rates are ~1/12 even at σ = 0.005, and σ collapses before anything is learned.
2. **Igel/(1+1)-CMA constants are computed from the full dimension** (`n = 224`): damping d = 1 + n/2 = 113,
   c_cov = 2/(n²+6) ≈ 4·10⁻⁵ — even with block-diagonal covariance the step size adapts ~100× too slowly and the
   covariance practically never learns.
3. **Objective misalignment (Biomass Succession).** The training AGB term is a squared hinge on ABSOLUTE g/m² error with a
   ±5 % band capped at 10 g/m² — dominated by large cohorts — and still scores synthetic phantom cohorts. The TOST
   criterion is the mean per-pair LOG ratio per species group on real records — dominated by the many small/medium cohorts.
   Offspring that improve ΣAGB worsen the TOST bias (r = −0.66 … −0.73). PnET/HBAD are weakly aligned (r ≈ +0.3/+0.5).
4. Real noise and roughness exist but are secondary: PnET's ±7e-4 seed noise (removed by the run's fixed common random
   numbers) and the deterministic roughness below u-steps of ≈ 0.005 (truncated biomass increments, binned age CDFs) —
   which only matters once σ is already small (handled by a σ floor).

## 2. Group-Decomposed Igel (1+1)-CMA-ES — `src/search/GDIgel.jl`, driver `src/gdigel_driver.jl`
- **Objective = the validated quantity, per group.** For each TOST group g (species, or species·site-cell·L3 where the
  config splits it): pairs = Σ sim vs Σ obs AGB per (plot, year, eco-species) over matched cohorts of the pre-injection
  simulation, real records only, disturbance-overwritten records excluded — byte-for-byte the pairing of
  `tools/tost_common.jl` (verified: identical n and bias for every shared group, `tools/gd_check.jl`).
  f_g = m_g² + w·mean_c m_{g,c}², m_g = mean log1p-difference, m_{g,c} its value per observed-size class
  (< 100, 100–1,000, 1,000–5,000, ≥ 5,000 g/m², ≥ 20 pairs; w = `size_weight`, default 0.5 — keeps the size structure
  flat, which separates growth-curve shape from level).
- **Decomposition.** u-coordinates are grouped into BLOCKS by the set of groups they affect (Biomass Succession: one block
  of S/B_MAX/ANPP (+ D when the species has one group) per group; species-level and eco-level parameters form blocks judged
  by the sum over all groups they touch). Each lineage holds an independent (1+1)-CMA strategy (σ, success rate,
  covariance, evolution path) per block, with constants from the BLOCK size (Igel, Hansen & Roth 2007).
- **One generation of a lineage**: perturb every block, evaluate once; each block succeeds iff the losses of ITS groups
  improved; successful blocks are assembled into x′, which is evaluated (so cross-group interactions — competition on mixed
  plots — are measured, not assumed away); if x′ is worse in total than the parent, the better of parent/offspring is kept.
  ≤ 2 simulations per lineage per generation; lineages are evaluated candidate-parallel (`PAN_PARALLEL=candidate`).
- **Multiple parameterizations.** Lineages never compete. A lineage that converges (all σ_b ≤ `sigma_stop`, or no total
  improvement > `rel_tol` for `patience` generations) is recorded as a solution and restarted from a fresh Sobol point; every
  evaluated point passing ≥ `archive_min_pass_frac` of the groups at ±10 % enters a behavioural archive. Solutions are then
  scored on held-out splits with the standard TOST tooling and analysed for equifinality (`tools/gd_equifinality.jl`).
- yaml: `search_mode: gdigel`, `gdigel: {mu, sigma0 (0.15), sigma_stop (0.004), sigma_min (0.002), patience (40),
  size_weight (0.5), real_only (true), include_start (false), group_param, archive_cap, archive_min_pass_frac (0.8), seed}`.
  Sim A only (override injection); works for every succession plugin.

## 3. Results (GA+FL, fit on the train split only; 2026-10-14)
Every gdigel run starts from Sobol points in the parameter box (no calibrated seed). The comparator is the per-group
coordinate calibration run on the same train split (`tools/calib_coord.jl`, the schedules of D10 / P2 / HBAD). Scores are
the standard TOST tables (`tools/tost_common.jl`, real records, 46–47 groups); "gdigel" = median over its solutions
[range in brackets].

| plugin | method | budget | solutions | train ±5/10/20 | val ±5/10/20 | test ±5/10/20 |
|---|---|---|---|---|---|---|
| Biomass Succession | coordinate | 81 evals | 1 | 21 / 39 / 44 | 7 / 18 / 34 | 6 / 15 / 28 |
| Biomass Succession | gdigel μ 8 | 3,208 | 8 | 16 / 34 / 45 | 6 / 17 / 34 | 5 / 18 / 30 |
| Biomass Succession | gdigel μ 16, fast restarts | 6,420 | 23 | 16 / 34 / 44 [5–18 / 17–35 / 37–45] | 7 / 16 / 34 [0–8 / 12–18 / 27–35] | 5 / 18 / 30 [1–7 / 9–19 / 24–31] |
| Biomass Succession | **ablation: one block** (no decomposition, same objective) | 3,200 | 24 | 61 fine groups: 0–2 / 0–7 / 8–18, median abs bias ≈ 25 % | — | — |
| HBAD | coordinate | 37 | 1 | 37 / 43 / 45 | 13 / 30 / 41 | 14 / 32 / 39 |
| HBAD | gdigel μ 8 | 3,215 | 15 (7 converged) | 22 / 40 / 44 | 11 / 28 / 40 | 10 / 30 / 38 |
| PnET | coordinate | 31 | 1 | 15 / 36 / 42 | 5 / 16 / 30 | 2 / 8 / 32 |
| PnET | gdigel μ 8 | 2,411 | 11 (3 converged) | 4 / 23 / 38 | 3 / 13 / 31 [0–5 / 4–15 / 22–35] | 2 / 10 / 27 [0–4 / 8–15 / 15–33] |

Findings:
1. **Decomposition is what makes CMA-ES work.** With the identical aligned objective and budget, a single (1+1)-CMA over all
   dimensions ends at total loss 9–18 (median |bias| ≈ 25 %); the decomposed search reaches 0.36 (median |bias| < 1 %).
2. **Held-out skill ≈ the coordinate calibration** for all three plugins, from box-centre/Sobol starts. gdigel fits train
   less tightly (its objective also flattens bias across size classes) but loses nothing out of sample — less over-fitting.
3. **It returns many distinct passing parameterizations** (groups with ≥ 2 clearly different passing parameterizations on
   train: Biomass Succession 33–35 / 46, HBAD 42 / 46, PnET 38 / 46). Spread (max/min across solutions in which the group
   passes at ±10 %, median over groups):
   - Biomass Succession: S ×1.2–1.7 (identified), B_MAX ×1.2–1.5, ANPP_MAX ×1.3–1.6, D (age-mortality shape) ×3–4 (weakly identified).
   - HBAD: G_MULT ×2.0, M_MULT ×11, effective SDIMAX ×2.7 (= its whole allowed range): a growth–mortality trade-off that AGB
     alone cannot resolve; observed TPA/BA (which HBAD simulates) would identify it.
   - PnET: FolN ×1.9; species traits MaxFracFol ×9, TOwood ×9, SLWmax ×6, FrActWd ×4.5, PsnAgeRed ×3.4 — the species-level
     ecophysiology is barely constrained by one-interval AGB; FolN compensates.
4. Cost: 2 simulations per lineage per generation; 7.4 s/generation for Biomass Succession with μ 8 on 8 threads
   (≈ 25 min for 3,200 evals). Coordinate calibration is ~40–100× cheaper for a single answer; gdigel is the tool when the
   ensemble of acceptable parameterizations is wanted.

Known issues: the behavioural archive threshold (`archive_min_pass_frac`, default 0.8 of groups at ±10 %) was never reached
by Biomass Succession / PnET (their passing fraction tops out ≈ 0.6) — use ≈ 0.5; PnET lineages need more generations
(3 of 8 converged in 150 generations); HBAD's SDIMAX is sampled in a wide box and clamped per species at decode (flat
u-regions; `tools/gd_equifinality.jl` reports the effective value).

Files: `runs_local/gd_{bs,bs_div,bs_single,hbad,pnet}/` (git-excluded): gd_log.csv, gd_state@N.jld2, solutions/,
gd_solutions.csv, equifinality_{train,val,test}[_solutions].csv; comparators `runs_local/coord_{bs,hbad,pnet}_train/`.

## 4. Anchor-and-explore: meaningfully distinct plausible parameter sets (`gdigel: {mode: explore}`)
Coordinate calibration finds ONE behavioural parameter set fast; gdigel (§2) finds several, but from random starts each
lineage converges to whatever optimum it meets first. The marriage of the two (2026-10-14):
1. **Anchor** = the coordinate-calibrated set (`start_from`). The anchor's per-group losses define the behavioural set:
   group g is acceptable iff f_g ≤ τ_g = max(log1p(`explore_bias`)², `explore_slack` × f_g(anchor)) (defaults 7 % and 1.5),
   so the anchor is behavioural by construction and groups it fits well keep an absolute ±7 % tolerance.
2. **Explore**: every lineage starts at the anchor (+ `explore_start_sigma` noise). A block step is accepted when the
   groups it affects stay behavioural AND the block moves farther (u-space distance over the block's coordinates) from
   the nearest other lineage, the anchor and every recorded solution; a block whose groups are out of tolerance is first
   repaired by loss (as in optimize mode). Lineages therefore spread over the set of equally acceptable parameterizations.
   After `patience` generations a lineage is recorded and restarted from the anchor. Block σ is capped at 0.5.
3. **Select** (`tools/gd_select_distinct.jl <cfg> <run> <anchor> K <out> [max_groups_above_tau]`, run inside the same data
   preparation so the u-space layout matches the search): candidates = recorded solutions + final lineage points +
   behavioural archive; behavioural = no group above τ_g (default strict); K picks by farthest-point sampling from the
   anchor. Writes `alt_k.jld2` + `selection.csv` (distance to the anchor and to the nearest pick, per-family change).
4. **Check** that the picks are meaningfully distinct: held-out TOST (val, test) of v1 and the alternatives
   (`tools/tost_select.jl`), and 100-year grow-only projections on 60 GA/FL plots (`tools/compare_param_sets.jl`).
   Results: `results/explore_{hbad,biomass}/`.
Run: `tools/gd_run.jl <cfg> <out> 1600 mode=explore start_from=<anchor.jld2> sigma0=0.05 patience=30 explore_bias=0.07
explore_slack=1.5` (PAN_PARALLEL=candidate). Cost ≈ 1600 simulations (HBAD ≈ 40 min, Biomass ≈ 60 min, PnET ≈ 90 min on 6 threads).

## 5. Multi-objective variant (accuracy + age structure, no TOST in the objective)
`gdigel: {objective: mo, weights: k}` replaces the TOST-aligned bias objective by per-group accuracy a_g and Wasserstein age-structure w_g with Chebyshev-scalarized lineages and a non-dominated archive; `CALIB_MODE=minimize` is the matching refinement of `calib_coord.jl`. TOST is then used only on held-out splits. See `docs/mo_gdigel.md`.
