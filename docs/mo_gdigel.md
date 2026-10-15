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
  **Rule at each measurement year T** (`loss_before_injection: true`, default on this branch; `false` = previous inject-then-score order with
  the species-level exclusion, byte-identical to earlier runs, no shock step). Growth/decay step first, then per cohort (needs
  `override_injection` + `override_injection_sync`):
  1. FULLY removed by disturbance in the interval (export: `full_removal`, drop = 1): the matching sim cohort (species, age) is removed from the
     site BEFORE scoring at T (`_apply_disturbance_shock!`, swap-remove as in the sync's Pass A) — nothing to compare at >= T.
  2. PARTIALLY disturbed, mode `scale`: the matched sim cohort is scaled by (1 - drop) BEFORE scoring and IS scored; the sync does not scale again
     (`skip_scale`), so exactly once.
  3. PARTIALLY disturbed, mode `exclude_overwrite` / `exclude_noscale`: exclude ONLY THAT COHORT (species, age) at T, on BOTH sides of the loss: the sim
     cohort is skipped when building the species' age CDF / AGB bins (`PU._excl_perm!`, kwarg `excl_c` of `calculate_site_loss2(_acc!)`), and the
     observed reference of that site-year is rebuilt without the cohort (`_build_excl_plan`: alternative `SPDFGroundTruth` via the same
     `smoothen_ref_years`). A species left with no cohort on either side contributes nothing (no "missing species" penalty). The overwrite of
     `exclude_overwrite` happens after scoring, in the sync. (Legacy order: whole species excluded - unchanged.)
  4. FULLY removed NOT by disturbance (natural death): still in the sim at scoring, so it is scored and surplus survival is penalised.
  5. RECRUITS / in-growth: an observed cohort at T with no record of the same cohort (species, age - (T - T_prev)) at the previous visit of that plot
     cannot be produced by Sim A (no establishment): excluded from the reference (and the sim) at T, cohort-level as in 3; it enters the sim through
     the sync. yaml `exclude_recruits_at_entry` (default true).
  6. Then the existing sync injection for the next interval.
  The a_g pre-injection cache is taken after the shock (step 1/2), before scoring.
  Data: `tools/export_cohort_disturbance.py` writes `data/cohort_disturbance_full.parquet` (partial AND full removals per
  subplot cohort and visit; reuses the logic of FIA_DATA_PREP/cohort_disturbance.py, which drops full removals). yaml
  `disturbance_full_removals: <parquet>` (default `data/cohort_disturbance_full.parquet` of the repo if present; `false` disables;
  a warning if sync + new order are on and no file exists). `Data.make_splots` maps rows to plot level with the same species tiering /
  eco / plot_id / sim_year mapping as `get_injection_cohorts` (train, val and test): a plot-level (species, age) cohort is "fully
  removed" only if NO subplot part of it is observed at T, otherwise it is partial and its drop is replaced by the agb-weighted
  plot-level share (removed subplot parts converted from tree-trajectory weight to cohort agb by 1/4046.86). When the cohort table
  carries no `disturbance_drop_pct` (the GA+FL `curated_cohorts_landis_stdorg` does not), the partial drops are attached from the
  same parquet (otherwise the table's column is used). a_g pairs (`_gd_index`): drop>0 records are excluded for `exclude_overwrite`
  (as before) and, in the new order, also `exclude_noscale`; for `scale` the pair is kept and compared with the scaled sim.
  Sim B (dual_mode b / joint, simB_disturb_only): the same full removals feed `_build_dual_b`'s disturbance dict with drop = 1; at the disturbance year the
  matching free-sim cohort (species, age) is REMOVED (partial scaling unchanged), so harvested cohorts stop growing in the free run.
  Diagnostics: `tools/shock_stats.jl` (counts of shock entries / matches / cohort-level exclusions at a params set).
  Not covered: tools/tost_common.jl keeps its own pair logic (drop-based exclusion only).
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
