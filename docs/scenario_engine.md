# Pan scenario engine (branch `scenario-engine`)

Projects FIA plots forward with **Biomass Succession, PnET-Succession or HBAD** under a parameter set (YAML), a management
plan (thin / harvest / plant) and — for PnET — a climate scenario. Used by the Forest Explorer fork
(`/workspace/fe_pan`, branch `forest-explorer-pan`) through a worker process. Plan of the campaign:
`docs/scenario_campaign_plan.md`.

## 1. Parameter files (YAML) — `src/params_yaml.jl`
- One self-contained file per parameter set. Header `pan_params: {plugin: biomass|pnet|hbad, format_version: 1, name,
  description, created, region, provenance}`; the **plugin named in the header decides which engine the file is for**.
- **Full roster**: every field of the plugin's params struct, mutable or fixed, laid out as
  `global:` (scalars, matrices such as `SUFFICIENT_LIGHT`), `species:` (one row per species), `ecoregions:` (per eco cell,
  with a `species:` list of eco×species values). Exact round trip (verified for all three calibrated sets).
- **Mapping metadata** used to place new plots: `species_map` (every FIA symbol in REF_SPECIES → parameter species, by
  the same tiering rule as training: exact species, else `_GRP_<spgrpcd>_<shade tier>`, else `_S_/_H_<tier>` catch-alls)
  and `eco_mapping` (EPA L3 + 4-cell site class A=SITECLCD 1–3, B=4, C=5, D=6–7 of the dominant forested condition in the
  pine L3s, one cell `X` elsewhere; fallback `NA` cell, then the L3's first cell).
- `SUFFICIENT_LIGHT[shade_class+1][shade_tolerance]` (6 site shade classes × 5 tolerance classes). The corner — the most
  shade-tolerant species under the deepest shade — is **0.5** (the LANDIS default has 1.0 there; in Pan it has been 0.5 since
  the data-derived taper). Import warns if a file sets it otherwise.
- Import is strict about structure; fields missing from a file fall back to the struct's defaults **with a warning**
  (e.g. HBAD files written before establishment existed).
- Export: `tools/export_params_yaml.jl <training config> <params.jld2> <out.yaml> "<name>" ["<description>"] ['<provenance json>']`.
  Shipped sets: `results/params_yaml/gafl_{biomass,pnet,hbad}_v1.yaml` (GA+FL calibrations).

## 2. Scenario runner — `src/scenario.jl`
- **Plots**: FIA `PLOT.CN` (what TreeMap pixels resolve to) → plot key + visit → curated cohorts of that visit
  (`curated_cohorts_landis_stdorg` in the national FIADB, phantom backfill excluded), summed over subplots per
  (species, age) — the same per-plot cohorts as training. Site class from COND; L3/L4 from the cohort table.
- Plots whose L3 is not in the parameter set are `out_of_region` (FVSjl can still simulate them); plots without curated
  cohorts are `no_data`. Cohorts of species the eco cell cannot represent are dropped (share of AGB reported).
- **Run**: annual steps of the plugin (establishment on for Biomass Succession and HBAD; PnET has none). Records every
  `period` years, at elapsed 0, P, 2P, … (start-of-cycle state, before that cycle's management — FVS `.sum` convention);
  management removals are reported on the row of the cycle they happen in.
- **Outputs per plot and record** (FE `CycleMetrics`): BA ft²/ac, TPA, QMD in, total cubic ft³/ac, board feet, removed volume,
  carbon (t C/ha: `carbon_live_ag` = 0.5 × AGB; `carbon_total` = live above + belowground with roots 0.25 × AG; removed),
  plus AGB g/m², and a species × 2-inch-DBH tally. Tree structure (TPA, QMD, height) comes from HBAD's state for HBAD and
  from the empirical translation (species, age, AGB) → (TPA, QMD, H) for Biomass Succession and PnET. **Approximations**:
  volume = BA × H × 0.42; board feet = 4 × cubic volume of cohorts with QMD ≥ 9 in; no dead-wood pools.

## 3. Management (LANDIS Biomass-Harvest style, FE vocabulary)
Plan actions (FE plan JSON): `thin` (FE/FVS form: residual BA/TPA/SDI, from below/above, DBH range, species — mapped to a
harvest), `plant` (species, trees/ac, survival %), and `harvest`:
```json
{"kind":"harvest","cycle":2,"repeat_every":0,
 "selection":{"species":"all|PITA,QUNI","age_lo":0,"age_hi":999,"dbh_lo":0,"dbh_hi":999},
 "removal":{"type":"percent_biomass|residual_ba|residual_tpa|residual_sdi|clearcut","value":50,"direction":"below|above|proportional"},
 "stands":{"rank":"all|max_biomass|oldest|random","area_pct":100},
 "plant_after":{"species":"PITA","tpa":600,"survival":80}}
```
- Cohort selection by species (FIA symbols or parameter species), age and DBH (cohort QMD).
- Removal: `percent_biomass` (that % of each selected cohort's biomass, LANDIS cohort-removal %), `clearcut`, or residual
  targets: `below` removes smallest-QMD cohorts first, `above` largest first, `proportional` the same fraction from every
  selected cohort; partial cohorts lose trees at constant size (biomass ∝ trees). Residual metrics use the translated
  structure, so BA/TPA/SDI specifications act on Biomass Succession and PnET biomass cohorts too.
- Stand selection (LANDIS site selection): `rank` + `area_pct` of the AOI acres (top-ranked plots until the area is met).
- Planting / `plant_after`: new cohort, age 1, QMD 1 cm, trees/ha = TPA × 2.471 × survival; biomass from the allometry.

## 4. Climate scenarios (PnET) — `src/climate_scenario.jl` (+ data in `/workspace/pan_climate`)
PRISM monthly by EPA L4: observed 1895–2024, baseline extrapolation 2025–2150 (`pan_climate.duckdb`, benchmark and method
choice in `/workspace/pan_climate/BENCHMARK.md`: AutoETS temperatures without trend, 1995–2024 precipitation climatology).
Scenario object:
```json
{"baseline":{"mode":"forecast|repeat|average","T":30,"K":3},
 "co2":{"mode":"constant|linear|ssp245|ssp585","rate_ppm_per_yr":2.0},
 "changes":[{"var":"tmean|tmin|tmax|ppt","kind":"add|mul","value":2.0,"shape":"step|ramp","start_year":2030,"end_year":2080,"months":null}]}
```
`repeat` = the last T observed years repeated cyclically; `average` = mean of the corresponding year across the last K
windows of T years; `mul` values are percent changes (−25 = 25 % less); `step` applies the full change from `start_year` on;
`ramp` grows linearly from 0 at `start_year` to the full value at `end_year` and holds; `tmean` shifts tmin and tmax.
Every PnET run builds its climate for the plots' L4s over its whole horizon (no scenario ⇒ forecast baseline + constant CO₂).

## 5. Worker — `tools/pan_worker.jl`
JSON lines on stdin/stdout (`ping`, `load_params {yaml}`, `default_params {plugin}`, `list_params`,
`simulate {params_id|plugin, plots:[{cn, acres}], plan, climate}`, `climate_preview {plots, climate, start_year, years}`,
`shutdown`); progress lines during `simulate`; Pan's own printing goes to stderr. Parameter sets are cached by a hash of
their text; the FE adapter re-sends uploaded YAML if the worker restarted.

## 6. Verification (2026-10-14)
- YAML round trip exact for the three calibrated sets; older HBAD file loads with defaults + warning.
- 25 GA/FL plots × 5–8 cycles: all three plugins run, thin/clearcut/plant/harvest act as specified (removed volume and
  carbon on the management row, regeneration after clearcut); PnET responds to climate (hot-dry SSP5-8.5 lowers late-century
  biomass below the baseline after CO₂ fertilization early on).
- Worker protocol test (ping, params, simulate with progress, unknown plot → `no_data`, climate preview, error handling).
- HBAD unit tests 428/428 and the regression gate IDENTICAL after merging HBAD establishment.
