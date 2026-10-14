# Pan scenario engine × Forest Explorer — campaign plan (2026-10-14)

Owner request (unattended run): multi-scenario simulation with Biomass Succession, PnET-Succession or HBAD parameter sets
inside Forest Explorer, LANDIS Biomass-Harvest-like harvesting, PnET climate-change scenarios, and tighter inverse
parameterization that yields meaningfully distinct plausible parameter sets. Upfront answers: parameter files are **YAML**
whose header names the plugin; scope GA+FL → Region 4 → CONUS but **parameters never block the other work**; **HBAD gets
establishment**; climate forecasting method is my choice (benchmarked, documented).

## Repositories / branches (all local, never pushed)
| repo | worktree | branch | base | contents |
|---|---|---|---|---|
| Pan | `/workspace/landis_scen` | `scenario-engine` | `cmaes-gafl` | YAML roster I/O, scenario runner + worker, harvest, climate scenarios, hybrid calibration |
| Pan | `/workspace/landis_hbadest` | `hbad-estab` | `scenario-engine` | HBAD establishment (merged back into scenario-engine) |
| FVSjl | `/workspace/fe_pan` | `forest-explorer-pan` | `forest-explorer` | Forest Explorer fork: engine selector, Pan adapter, harvest/climate UI, export/import |
| data | `/workspace/pan_climate/` | — | — | PRISM baseline forecasts (DuckDB), benchmark report |
Existing FVSjl branches/worktrees are untouched (other agents work there).

## Workstreams
- **W0 Parameters (Pan).** (a) YAML roster export/import for all three plugins (generic over the params structs, exact
  round trip) incl. establishment + fixed constants; (b) SUFFICIENT_LIGHT: corner (shade-tol 5 × shade class 5) = 0.5 — already
  0.5 in every Pan branch (LANDIS default is 1.0 there); made explicit + validated on import; (c) HBAD establishment;
  (d) hybrid calibration ("anchor-and-explore": coordinate calibration finds an anchor fast, gdigel lineages explore the
  feasible set around it with a diversity objective, then k-medoids picks K distinct sets that also differ in projected
  dynamics); (e) GA+FL first, Region 4 when compute allows, CONUS later.
- **W1 Pan engine in Forest Explorer.** Pan runs as a persistent worker process (own Julia project/depot) speaking JSON lines
  over stdin/stdout; FE spawns it lazily. Reason: FE and Pan have separate, incompatible Manifests (GDAL etc.), crash
  isolation, Pan keeps its own threads warm.
- **W2 Harvest.** LANDIS Biomass-Harvest-style prescriptions with the FE interface options (residual BA/TPA/SDI, from
  below/above, DBH range, species) + BH options (cohort age ranges, % biomass removal per cohort, stand ranking + area
  target). BA/TPA/DBH specs are translated to cohort biomass removals through the empirical HBAD translation
  (species, age, AGB) ↔ (TPA, QMD, BA, H).
- **W3 Climate.** Baseline extrapolation per EPA L4 and variable: forecast model (benchmarked), "repeat last T years" or
  "average of the last K windows of T years"; user changes = additive (temperature, °C) / multiplicative (precipitation, %)
  deltas, step or linear ramp, optional per-season shaping; CO₂ pathway. Applied to PnET's climate array per run.
- **W4 Export/import + samples.** Scenario bundle JSON (AOI patches, plans, engine + embedded params YAML, climate settings)
  and sample inputs (AOIs in GA/FL, parameter YAMLs incl. distinct alternative sets, climate scenarios, plans).

## Worker protocol (W1) — contract between FE (`src/PanEngine.jl`) and Pan (`tools/pan_worker.jl`)
One JSON object per line on stdin; one JSON object per line on stdout (`{"id":…, "ok":true, "result":…}` or
`{"id":…, "ok":false, "error":"…"}`); progress lines `{"id":…, "progress":{"done":n,"total":m}}`.
Requests:
- `{"op":"ping"}` → `{"version":…, "plugins":["biomass","pnet","hbad"]}`
- `{"op":"load_params","yaml":"<text>"}` → `{"params_id":"<sha1>", "plugin":…, "name":…, "species":[…], "ecos":[…], "region":{…}}`
  (validates; cached by hash). `{"op":"default_params","plugin":"hbad"}` → same, for the shipped default set.
- `{"op":"simulate","params_id":…, "plots":[{"cn":"<FIA PLT_CN>","acres":…},…], "plan":<plan>, "climate":<climate|null>}` →
  `{"plots": {"<cn>": {"status":"ok|out_of_region|no_data", "metrics":[CycleMetrics…], "dist": {"<year>": [[spcd, dbh_lo, tpa, ba],…]}}}, "warnings":[…]}`
- `{"op":"climate_preview","l4":["8.3.5a",…], "climate":<climate>, "start":2025, "years":100}` → per variable monthly
  series (baseline and modified) for the popup's waveform chart.
`<plan>` = the FE plan JSON (`name, cycles, period, actions[]`) extended with `harvest` actions (W2) — `grow`, `thin`, `plant`
keep their FE meaning. `CycleMetrics` = the FE struct (year, ba ft²/ac, tpa, qmd in, tcuft, bdft, rem_tcuft, rem_bdft,
carbon_total, carbon_live_ag, carbon_removed t C/ha); Pan-derived volumes are approximations (flagged in `warnings`).

## Order (parallel where independent)
1. Subagents in parallel: climate forecasts (W3 data), HBAD establishment (W0c), FE frontend against this contract (W1/W2/W3 UI).
2. Me: YAML roster (W0a/b) → scenario runner + worker (W1) → harvest translation (W2) → climate application (W3) →
   FE backend adapter → integration + samples (W4) → hybrid calibration (W0d) in the background → Region 4.
3. Every subagent result is verified (tests re-run, spot checks) before merging.
