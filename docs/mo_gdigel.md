# Multi-objective gdigel (`gdigel: {objective: mo}`) and the minimize-mode refinement

Branch `mo-gdigel` (from `scenario-engine`). Code: `src/search/GDIgel.jl`, `src/gdigel_driver.jl`, `tools/calib_coord.jl`
(+ `tools/calib_minimize.jl`, `tools/calib_levers.jl`), `tools/mo_archive_pick.jl`; tests `test/mo_gdigel_unit.jl`;
full-run recipe `runs_local/mogd_full.sh` (git-excluded, not executed by the branch work).

## 1. Why
The original gdigel objective `f_g = m_g² + w·mean_c m_{g,c}²` (m_g = per-group mean log1p-difference) and `calib_coord.jl`
(root-finding each group's mean bias to zero) both optimise the quantity the TOST equivalence test evaluates (the signed mean
bias, shrunk to zero). Fitting a statistic and then testing that same statistic is a form of p-hacking: pass counts on the
fitting split say little, and a parameter set can pass by cancelling errors of opposite sign. The MO variant optimises
**accuracy** and **age structure** instead; **TOST is used only to evaluate held-out splits** (val/test) with the unchanged
`tools/tost_common.jl` tooling.

## 2. Per-group objectives (same groups, same pairs as the bias objective)
Groups and matched pairs are exactly gdigel's (`_gd_groups`, `_gd_index`: real records, disturbance-overwritten excluded,
train split, pre-injection simulation).
* **a_g = mean over the group's matched pairs of |log1p(sim) − log1p(obs)|** (absolute accuracy; `_gd_losses`, field `a`;
  NaN when the group has < 3 pairs). Unlike m_g it does not cancel opposite-sign errors; a constant log-bias ±δ gives a_g = δ for both
  signs while m_g flips (unit-tested).
* **w_g = Wasserstein-1 age-structure distance**, re-using the main loss machinery. `fit_params` returns
  `eco_losses[e].sp_w_loss[gsp]` = Σ over the scored (site, measurement-year) visits of the per-species W1 distance between the
  simulated and the reference biomass-by-age CDF (`calculate_species_loss_acc!`: `_w_finish(_w1_sum(...))`, i.e. including the
  config's count-balance weights, softplus smoothing, hinge forgiveness, unbinned per-year age grid). A *cell* is one
  (eco, eco-species) = (e, j) with gsp = `eco_species_ids[e][j]`; a group is the set of cells `cell_group` assigns to it.
  Aggregation (`_gd_cell_w`, `_gd_group_w`), a ratio of sums over the group's cells with the main loss's own normalisation:
  * `cell_normalize: true` (GA+FL configs): `w_g = _w_pow( Σ_cells sp_w_loss / Σ_cells W_SCALE_A[gsp,e] )`, W_SCALE_A = Σ over the
    cell's reference visits of the per-reference-max W1 (× `w_scale_factor`) — the divisor `_mo_objectives` applies per cell. This is the
    visit-weighted mean W1 relative to the mean attainable W1. The RANKW stratum weights are deliberately **not** applied (they weight
    the loss across species; they are not part of a group's accuracy).
  * otherwise: `w_g = Σ_cells sp_w_loss / (# scored visits with a reference record for those cells)` (plain visit mean).
  A cell where the species is absent from the simulation but present in the reference contributes the full reference CDF (as in the main loss).
  Caveat: with `override_injection_sync` the *cohort set* at a measurement year is set to the observed one (surviving cohorts keep
  their simulated biomass), so w_g measures the biomass-by-age shape of the survivors; the AGB/accuracy side (a_g) uses the
  pre-injection cache and is not circular.
* Block / lineage totals: `A_b = Σ_{g∈groups(b)} a_g`, `W_b = Σ w_g`; `A = Σ_g a_g`, `W = Σ_g w_g`.

## 3. Search (`objective: mo`)
yaml (`gdigel:` block; also `tools/gd_run.jl … objective=mo weights=8`):
`objective: bias | mo` (default `bias` = unchanged behaviour), `weights: k` (default 8), optional `aref`, `wref` overrides.
* Lineage i gets a fixed weight λ_i from the uniform grid `linspace(0,1,k)` (lineages beyond k cycle the grid; k=1 → 0.5). Lineages
  remain independent (MOEA/D-style decomposition without neighbour interaction); a restarted lineage keeps its slot's λ.
* Scalarization (Chebyshev): `F = max( λ·A/A_ref , (1−λ)·W/W_ref )`; for a block b the same with A_b, W_b. Block success, assembly, the
  parent-vs-offspring rule, convergence (σ or no-improvement patience) and restarts are those of the bias objective with
  `F` replacing the summed loss. λ = 1 optimises accuracy only, λ = 0 age structure only.
* A_ref, W_ref: totals of the start point — `start_from` evaluated if given, else the **median over the μ initial (Sobol) points**; fixed for the run and logged
  (`gdigel mo: A_ref = …`). Stored in the state (`gd_state@N.jld2`, field `aref/wref`).
* Archive: every evaluated point is offered to a **non-dominated archive on (ΣA, ΣW)** (no TOST gating); cap `archive_cap` (default 2000) removes the most crowded interior point
  (2-objective crowding distance, extremes never dropped). Written every `ckpt_every` generations and at the end:
  `mo_archive.csv` (sorted by ΣA: gen, lineage, `lambda_lineage`, ΣA, ΣW, A/A_ref, W/W_ref, `lambda_balance`, `role` ∈ {min_A, knee, min_W}),
  `mo_archive/arch_NNNN.jld2` (decoded parameter sets), `mo_archive_u.jld2` (u-points, per-group a, w, refs).
  Knee = archive point farthest below the chord between the two extremes in (A/A_ref, W/W_ref); `lambda_balance = (W/W_ref)/(A/A_ref + W/W_ref)`.
* `gd_solutions.csv` gets `lambda, sumA, sumW` columns (and `total` = F(λ)); `gd_log.csv` gets `lambda,sumA,sumW`. The approximate-TOST pass counts still logged
  there are **information only** and never used by the search. Mode `explore` is not available with `objective: mo`.
* Old (pre-branch) `gd_state` checkpoints are not loadable by this code (struct fields changed); start new runs.

## 4. Refinement: `CALIB_MODE=minimize`
`tools/calib_coord.jl <cfg> <start.jld2> <out_dir> <schedule> [minimize]` with `CALIB_MODE=minimize` (default `root` = the old bias root-finding,
unchanged), `CALIB_LAMBDA=<λ>`, `CALIB_AREF/CALIB_WREF` (default: totals of the start set; pass the archive's refs). The training context is prepared by `run_from_yaml` (all loss flags
active; groups, pairs and W identical to gdigel). For each scheduled parameter the shared grid (absolute, or relative factors for `S~`, `ratio~`, `bmax~`, `foln~`, `gmult~`, `mmult~`) is evaluated;
each group takes the value minimising `obj_g = max(λ a_g/a_ref_g, (1−λ) w_g/w_ref_g)` (per-group refs = values at the start set, floored at 10 % of the median) and accepts it only
if it strictly improves its current `obj_g`; the assembled set is re-evaluated and the round is reverted if the total F got worse (interactions). No bias root-finding.
Log per round: ΣA, ΣW, F, and (information only) train approximate-TOST ±5/10/20 counts. Size-regime splitting (`CALIB_SIZE_THR`) is not used in this mode.

## 5. Running it and evaluating
`runs_local/mogd_full.sh` (per plugin: MO gdigel μ 16 / weights 8 → `tools/mo_archive_pick.jl` → refine knee + both extremes with their λ →
held-out evaluation). Evaluation protocol (fixed in advance): **val selects** among the refined points (and the unrefined archive picks)
with `tools/tost_select.jl <cfg> <jld2 list> K val`; **test reports** the val-selected point(s) with `tools/tost_select.jl … test`. TOST on train is
never used for any choice. `tools/gd_equifinality.jl` still applies to the gdigel `solutions/`.
Budget guidance: see the smoke timings in the commit/report (s per generation per plugin; ≈ 2 simulations per lineage per generation).
