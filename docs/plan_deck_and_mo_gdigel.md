# Plan: "Tools for Forest Simulation" deck (A) + non-TOST recalibration of GA+FL models (B)

## Context
The user wants a ~45-min beamer talk on the whole body of work. The outline is `/workspace/PERSEUS_PRESENTATION.md`: their host `~/Documents/projects`
is the container's `/workspace`, and products go to **`/workspace/PERSEUS_PRESENTATION/`**.

User decisions:
- **Deck format:** title "Tools for Forest Simulation"; clean custom theme (metropolis-like, forest green, 16:9); speaker notes + backup + handout.
- **Model characterization:** fully characterize PnET and HBAD relative to Biomass Succession (the core model): where each interacts with or replaces
  Biomass Succession processes, and the parameters each needs.
- **TOST:** optimizing the TOST bias directly is a form of **p-hacking**. Both the coordinate calibration (`tools/calib_coord.jl:1–4`, roots of the
  per-group mean log-bias) and gdigel's f_g (`cmaes_gdigel.md` §2) target the statistic TOST tests. **Redo GA+FL for all three models** with:
  - **group-decomposed multi-objective gdigel** on per-group objectives — (1) cohort AGB accuracy = mean |log(sim/obs)| (absolute error, not
    the signed bias); (2) Wasserstein distance of the group's age-class biomass distribution;
  - then **local coordinate refinement of selected Pareto points** (knee + both extremes);
  - TOST is used **only for held-out evaluation** (val for selection, test for the report).
- **HBAD:** present the current model as **HBAD v1** and detail its process: competition is structural (BAL, RD, Reineke SDI self-thinning), but
  growth and mortality response functions are empirical FIA regressions. State that v2 is being moved toward process-based. Say that v1 currently
  performs best, and that empirical fitting may limit its applicability under climate change, novel management and other regions.
- **Timing: deck now, update later.** Result slides use current results, explicitly flagged as "TOST-targeted (being redone)". They are swapped
  in when workstream B finishes.

Presenter line: "BahaaEddin AlAila — University of Georgia (advisor: Dr. Khaled Rasheed)", from the thesis deck. It is editable in `main.tex`.

---------------------------------------------------------------------------------------------------------------------------------------
## A. The deck

### Inputs (inventory, verified paths)
- **Thesis deck** `/workspace/thesis_latex/presentation.tex` + `presentation.pdf`:
  - Reuse its crop macros. Its `imgs/` paths map to `/workspace/landis_parametrization_julia/{runs,tools,test}/`.
  - Five figures exist only in the PDF (`old_growth`, `LBSA_convergence`, `l1-cv-{fold,position}-fronts`, `SCP`). Extract them with PyMuPDF.
- **Ready figures:**
  - loss illustrations: `tools/{wasserstein_one_plot_simA,wasserstein_simA_diverge,hinge_softplus_illustration,spinup_processaa}.png`;
  - toy optimizer suite: `test/{toy_panel,toy_realistic_2d,toy_scales}.png`;
  - `runs/fl5_l4cover_sobol_Aonly_Sglobal_2k_v2_outputs/sobol_umap_*.png`;
  - MO-CMA-ES `convergence.png` + `pareto_tost_10pct_by_stratum_val.png`;
  - `*ccigel*_outputs/ccigel_sweep_val.png`;
  - candidate 14 and SCP cand6 scatter/MAPE/TOST;
  - `all_cohort_trajectories.png`, `fvs_vs_pan_AUTOES_*.png`;
  - FL5 maps, `estab_raw_scatter_vs_relbio_p95.png`, `scatter_biomass_strata.png`, `traj_preview_ageVbio.png`;
  - FIA-layer screenshots `<scratchpad>/shots/*.png`.
- **Numbers:**
  - `landis_scen/docs/`: `gafl_model_comparison.md`, `cmaes_gdigel.md`, `campaign_2026-10-14.md`, `hbad_plugin.md`, `region4_pnet.md`,
    `pnet_spec.md`, `volume_equations.md`, `treemap_donors.md`, `gafl_calibration_log.md`;
  - `landis_scen/results/explore_*`, `results/r4/`;
  - `/workspace/pan_climate/{README,BENCHMARK}.md`, `scripts/bench_main.csv`;
  - `/workspace/FIA/FIA_DATA_PREP/CLAUDE.md`, `thesis_latex/{methods,results}.tex`;
  - FVSjl `README.md`, `docs/CERTIFICATION_STATUS.md`, `docs/MODERNIZATION_AUDIT.md`;
  - memory notes.
- **Concept-only items** (no results exist): generative treelist, MNAR, predictive-coding age (`TreeAgeTransformer/tree_age_model.py`, untrained),
  WASM/ARTEMIS. Microstand segmentation = data + pipeline written, not run.

### Story (outline order, corrected to match the work)
- **Speed path:** LANDIS black box → numpy → **Julia** ("5.5 h → < 20 min", `methods.tex:12,15`).
- **Optimizer timeline:** SA/LBSA/MOLBSA (non-unique) → Wasserstein vs AGB trade-off → MO-CMA-ES → CCIgel → group-decomposed Igel
  (beats one-block CMA-ES: median |bias| ≈ 25 % → < 1 %, `cmaes_gdigel.md:74`).
- **Coordinate calibration:** shown as the local refiner (strong at zeroing in, does not explore) and gdigel as the explorer. The current deployed sets
  came from TOST-targeted coordinate calibration. One slide, "**Lesson: don't optimize the test statistic**", introduces the redo (B): MO gdigel
  (AGB error + Wasserstein) → coordinate refinement → held-out TOST.
- **Results:** current GA+FL/R4 numbers flagged "TOST-targeted, being redone"; held-out numbers are still shown. Plus Sim A vs Sim B, and equifinality.
- **Harvest:** "Non-biomass harvest" is shown as solved (BA/TPA/SDI targets through the HBAD translation; `fe_pan/.../PAN.md:50–53`).
- **Climate:** a statistical PRISM baseline (AutoETS temps, climatology precipitation) + user scenarios + SSP CO₂ (RCMIP). There are no NOAA/CMIP
  projections. Climate and soils drive only PnET.

### Model characterization part (after "Approach"; ~6 slides + backup) — user request
Sources:
- Pan structs `landis_scen/src/plugins/{BiomassSuccession,PnET,HBAD}/params.jl` and the `plugin.jl` kernels;
- `docs/{pnet_spec.md, hbad_plugin.md, region4_pnet.md, scenario_engine.md}`; `src/params_yaml.jl`;
- LANDIS guides in `/workspace/*.pdf` (Biomass Succession v7, PnET v6.0, PnET v5.1 description, Scheller & Mladenoff).

Slides:
1. **Shared skeleton** (TikZ): eco cell × species cohorts. The annual loop growth → mortality/senescence → establishment → disturbance, with
   each model's replacement colour-coded.
2. **Process matrix** (rows: growth, competition/light, age and growth-related mortality, self-thinning, longevity, establishment, seed
   source, shade, climate, soil water, CO₂, harvest interface, state variables; columns: Biomass Succession | PnET | HBAD v1). Examples:
   - growth: ANPP_MAX × competition index (Biomass Succession) | GPP from FolN/Amax, light, water, T, VPD, CO₂ (PnET) |
     FIA dD regression × G_MULT, with BAL competition (HBAD);
   - establishment: PROB_ESTAB × SUFFICIENT_LIGHT with shade class from relative biomass (Biomass Succession) | none in Pan's PnET
     (LANDIS PnET: own probability from light/water) | Biomass Succession block, but shade class from relative density (HBAD).
3. **Parameter tables** — fitted / fixed-from-data / default, with units and bounds:
   - Biomass Succession: LONGEVITY, MATURITY, SHADE_TOL, PROB_ESTAB, SUFFICIENT_LIGHT, ANPP_MAX, B_MAX, S, D, PROB_MORT, …;
   - PnET: PFT, LONGEVITY, FolN, MaxFracFol, FrActWd, SLWmax, TOwood, PsnAgeRed, RD_MULT + fixed physiology (AmaxA/B, HalfSat, k, PsnT*, H1–H4, …);
   - HBAD v1: LONGEVITY, SDIMAX, G_MULT, M_MULT + the Biomass Succession establishment block (PROB_ESTAB, SHADE_TOL, MATURITY, SUFFICIENT_LIGHT,
     MIN_REL_RD, ESTAB_TPH) + 28 fixed prior coefficients per species.

   Full rosters go to backup.
4. **Inputs per model:**
   - all three: species traits from FIA (longevity classes, shade tolerance from CCLCD, PROB_ESTAB);
   - PnET only: climate + soils;
   - HBAD only: FIA priors.
5. **Identifiability / equifinality per model** (Biomass Succession D ×3–4; HBAD M_MULT ×11, G/M trade-off; PnET FolN ×1.9, TOwood ×9) + cost per site-year.
6. **Scenario-engine interface:** cohort ↔ TPA/QMD/H translation, harvest, climate (PnET only).

### HBAD v1 part (4 slides)
1. **Concept:** QMD / BA / stocking charts → a cohort state of TPH, QMD, height, age. Includes the "not Houston Black American Democrats" joke.
2. **Process detail, from `hbad_plugin.md` §1–3, verified in `plugin.jl`.** The annual step:
   - longevity;
   - dD = G_MULT·exp(g(D, age) − k·BAL) — **size-asymmetric competition** through BAL;
   - height through a Chapman–Richards ratio;
   - hazard = M_MULT·exp(m(D, RD, BAL)) — **density-dependent mortality**;
   - **Reineke SDI self-thinning**, smallest cohorts first;
   - establishment with shade class from relative density;
   - AGB through the mass–QMD allometry.
   - Fitted: G_MULT, M_MULT (eco × species), SDIMAX (species). Fixed: FIA regression coefficients (val R² 0.05–0.29, AUC 0.61–0.72, `hbad_plugin.md:105`).
3. **Performance:** v1 is best held-out (≈17/34/41 vs Biomass Succession ≈8/20/36, PnET ≈3/16/33 — flagged TOST-targeted). Plus a ★QMD–TPA trajectory
   with Reineke lines.
4. **Caveat + v2:**
   - The response functions are fitted to FIA under past climate and management, in GA+FL only. There are no climate, CO₂ or soil drivers.
     Extrapolation risk under climate change, novel management (heavy thinning, new genetics, plantation regimes outside the data) and other regions.
   - v2 direction: process-based growth (light/carbon/water) keeping the HBAD state and the competition structure.

### Deck structure (~46 main slides; ★ = new figure)
0. Title, outline
1. Problem statement (3): why process-based; reproduction as a separate process; study regions
2. Approach — models (2)
3. Model characterization (6)
4. Inverse parameterization (12):
   - loop, curation pipeline, age, trajectories;
   - timeline ★TikZ; speed; SA/LBSA/MOLBSA; Wasserstein; MO-CMA-ES; CCIgel; gdigel;
   - coordinate refinement; "don't optimize the test statistic" → redo (B)
5. Results (5, flagged): example candidates; TOST explained; GA+FL comparison ★bars; Region 4 ★bars; equifinality ★fans
6. Limitations (3): reproduction (Sim A vs B); estimates on estimates (chaos); harvest (now solved)
7. HBAD v1 (4)
8. Soil (1–2): SOLUS + Saxton–Rawls, ★field-capacity map, coverage 99.4 %
9. Climate (2): PRISM → L4 ★map + limitation; baseline benchmark ★bars + SSP CO₂ ★lines + PnET scenario effect
10. Viewer demo (5): screenshots ★; FVSjl certification + scaling ★; site-from-location / donors / volume equations
11. In the works (2): generative treelist (concept); microstand inputs ★panel
12. Viewer next steps (1)
13. Next steps (2): add the B recalibration and HBAD v2 to the existing items
14. Acknowledgements, Questions
15. Backup (~14): parameter rosters, process-matrix details, TOST per group, optimizer settings, volume equations, donors, climate benchmark,
    HBAD 100-yr table, FVSjl certification, Sim B pitfalls, gdigel algorithm, Wasserstein/hinge, MO-gdigel design

Every slide gets a `\note{}` with talking points + sources. `SOURCES.md` maps every number to file:line.

### Layout / toolchain
`/workspace/PERSEUS_PRESENTATION/`:
- `main.tex`, `parts/*.tex`, `beamerthemeperseus.sty`, `handout.tex` (2-up), `notes.tex`;
- `figures/{reused,extracted,new,screens}`, `scripts/` (+ `build.sh`, `perseus.mplstyle`), `data/` (incl. `model_matrix.csv`, `param_roster.csv`);
- `SOURCES.md`, `README.md`;
- final PDFs `Tools_for_Forest_Simulation{,_handout,_notes}.pdf`.

Tools:
- Tectonic via micromamba (`/workspace/fe_pan/.uitools/bin/micromamba`) into `.tools/`;
- Python venv `.venv` (matplotlib, pandas, pymupdf, duckdb, geopandas, rasterio, pyproj, pypdf), run with `-I`.

### A steps
1. Scaffold + toolchain + theme; build an empty deck.
2. Reused/extracted figures (originals untouched).
3. New figures via parallel Sonnet subagents writing only in the deck folder (charts, maps, HBAD trajectory, Forest Explorer screenshots on a private
   port 8093 + private cache). I verify each figure and number.
4. Model characterization data (`model_matrix.csv`, `param_roster.csv`) drafted by a subagent; I verify every row against its cited line.
5. Write the parts with notes and `SOURCES.md`. Resolve: SCP plot count (3,159 vs 3,519), FVSjl IE status.
6. Build main / handout / notes and render-check.

---------------------------------------------------------------------------------------------------------------------------------------
## B. Recalibration: MO group-decomposed Igel (AGB error + Wasserstein) → coordinate refinement → held-out TOST

Branch: new Pan branch **`mo-gdigel`** from `scenario-engine`, in a new worktree `/workspace/landis_mogd` (layered depot like the others).
Local commits only.

1. **Per-group objectives** (`src/gdigel_driver.jl` + the loss code). For each group g on TRAIN real records, using the same pairing as now:
   - **a_g = mean |log1p(sim) − log1p(obs)|** over matched (plot, year, eco-species) pairs — accuracy, not the signed mean;
   - **w_g = Wasserstein-1 distance** between simulated and observed age-class biomass CDFs of g's cohorts. Reuse the `SiteLoss.sp_w_loss` machinery,
     aggregated from eco-species to group.
   - Unit test: a_g = 0 and w_g = 0 for sim = obs; a sign flip of the bias leaves a_g unchanged.
2. **MO gdigel** (`src/search/GDIgel.jl`):
   - Each lineage carries a weight λ ∈ [0, 1] on a uniform grid (MOEA/D-style decomposition, keeping lineages independent as now).
   - Per block, the scalar objective is the Chebyshev scalarization of (Σ_{g∈block} a_g / a_ref, Σ_{g∈block} w_g / w_ref), with references from the
     start set.
   - Block success, assembly and restart logic are unchanged. An archive keeps the non-dominated set on the total (Σ a, Σ w).
   - yaml `gdigel: {objective: mo, weights: k}`. The old TOST-bias objective stays available (`objective: bias`) for comparison only.
3. **Coordinate refinement** (`tools/calib_coord.jl` new mode `minimize`):
   - For a chosen Pareto point (knee + both extremes), per group, run a 1-D line minimization of that point's scalarized objective over the existing
     grids / relative factors (S, ratio, B_MAX | FolN | G_MULT, M_MULT).
   - A group's value changes only if its objective improves. Iterate the schedule. **No bias root-finding.**
4. **Runs** (GA+FL, Sim A, same train/val/test split as the current results; Biomass Succession, PnET, HBAD): MO gdigel μ ≈ 16 lineages (≈ 3–6k
   evaluations each), then refinement of 3 points per model. Logs go to `runs_local/mogd_*`.
5. **Evaluation:**
   - TOST ±5/10/20 % on **val** (selection among the refined points) and **test** (report).
   - Plus a_g / w_g held-out, the Pareto fronts, and a comparison with the TOST-targeted sets on the same splits.
   - Equifinality re-analysis from the MO archive.
   - Doc `docs/mo_gdigel.md`.
6. **Deck update:** swap the result slides (comparison bars, R4 note, equifinality, HBAD performance) to the new numbers and remove the "TOST-targeted"
   flags. Add a Pareto-front figure and a "before vs after" slide.
7. **Region 4 rerun:** queued after GA+FL, outside this deck's first version.

## Verification
- **Deck:**
  - Tectonic build without errors; overfull boxes ≤ 5 pt.
  - Every page rendered to PNG and inspected.
  - Handout and notes PDFs checked.
  - Every slide number cross-checked against `SOURCES.md` (scripted grep diff).
  - Model-matrix rows each verified against code/guide lines.
- **B:**
  - Unit tests for a_g / w_g and MO block acceptance.
  - The `objective: bias` path reproduces the current gdigel results (regression gate).
  - The default Biomass Succession regression gate (`tools/regress_gate.sh`) stays byte-identical.
  - Held-out TOST numbers recomputed with the unchanged `tools/tost_common.jl`.
- **Isolation:** no files outside `/workspace/PERSEUS_PRESENTATION/` and the new `mo-gdigel` worktree are modified. Servers and simulations write
  only to the scratchpad or those folders.
