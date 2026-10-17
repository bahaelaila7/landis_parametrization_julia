# Pan scenario engine (branch `scenario-engine`)

Projects FIA plots forward with **Biomass Succession, PnET-Succession, HBAD or HBAD2** under a parameter set (YAML), a management
plan (thin / harvest / plant) and — for PnET — a climate scenario. Used by the Forest Explorer fork
(`/workspace/fe_pan`, branch `forest-explorer-pan`) through a worker process. Plan of the campaign:
`docs/scenario_campaign_plan.md`.

## 1. Parameter files (YAML) — `src/params_yaml.jl`
- One self-contained file per parameter set. Header `pan_params: {plugin: biomass|pnet|hbad|hbad2, format_version: 1, name,
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
- **Plots = vegetation only**: FIA `PLOT.CN` (what TreeMap pixels resolve to) → plot key + visit → curated cohorts of that
  visit (`curated_cohorts_landis_stdorg` in the national FIADB, phantom backfill excluded), summed over subplots per
  (species, age) — the same per-plot cohorts as training.
- **Site = the AOI location** (`src/scenario_site.jl`, 2026-10-14). TreeMap imputes plots from anywhere (e.g. Oklahoma plots
  in a Georgia AOI), so everything that belongs to the place comes from where the pixels are: the caller (Forest Explorer)
  sends per plot the majority EPA L4 of its AOI pixels and their centroid (`l4`, `lon`, `lat`); then
  - EPA L3/L4 → parameter eco cell and PnET climate (L4 series);
  - site class (eco-cell split) = inverse-distance vote of the 7 nearest FIA plots of the same L3 (actual plot coordinates);
  - PnET soils = SOLUS100 rooting-zone hydraulics at the centroid (same procedure as `tools/build_plot_soil.jl`; `PAN_SOLUS`),
    falling back to the FIA plot's soil row, then loam.
  Without a location (older callers) the FIA plot's own L3/L4, site class and soil are used, with a warning.
- **Plots without curated cohorts** (8,541 of 65,043 TreeMap plots, mostly pre-2008 visits) take the cohorts of a TreeMap
  **donor**: same forest type, nearest in BA/QMD/height/TPA/canopy/softwood share (+ geography), from
  `data/treemap_donors.csv` (`tools/build_treemap_donors.py`, `docs/treemap_donors.md`; `PAN_DONORS`). Remaining
  failures: `out_of_region` = the location's L3 is not in the parameter set; `no_data` = no cohorts and no donor.
  Cohorts of species the eco cell cannot represent fall back to the `_H*`/`_S*` catch-alls, else are dropped (AGB share reported).
- **Run**: annual steps of the plugin (establishment on for Biomass Succession and HBAD; PnET has none). Records every
  `period` years, at elapsed 0, P, 2P, … (start-of-cycle state, before that cycle's management — FVS `.sum` convention);
  management removals are reported on the row of the cycle they happen in.
- **Outputs per plot and record** (FE `CycleMetrics`): BA ft²/ac, TPA, QMD in, total cubic ft³/ac, board feet, removed volume,
  carbon (t C/ha: `carbon_live_ag` = 0.5 × AGB; `carbon_total` = live above + belowground with roots 0.25 × AG; removed),
  plus AGB g/m², and a species × 2-inch-DBH tally. Tree structure (TPA, QMD, height) comes from HBAD's state for HBAD and
  from the empirical translation (species, age, AGB) → (TPA, QMD, H) for Biomass Succession and PnET. **Volumes**
  (`src/scenario_volume.jl`): FIA Region-4 per-tree equations v = cf·exp(b0)·ba^b1·H^b2 by species (→ FIA species group →
  softwood/hardwood), evaluated at each cohort's QMD and height × TPA: total-stem cubic ft (VOLTSGRS) and board feet
  **International ¼-inch** (VOLBFGRS — the Scribner column is empty in this FIADB) for cohorts with QMD ≥ 9 in (softwood) /
  11 in (hardwood) × a fitted sawtimber-qualifying probability. Held-out FIA plots: cubic plot R² 0.997, bias −0.02 %
  (old BA × H × 0.42: +6 %); board feet plot R² 0.93, total bias +1.4 % (old 4 × cubic: +14 %). `docs/volume_equations.md`,
  `data/volume_eqs_r4.csv` (`PAN_VOLUME_EQS`). No dead-wood pools.

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

## 7. HBAD2 (HBAD v2, model v2g) in the scenario engine (stage 2)
`succession_plugin: hbad2`; docs of the plugin: `docs/hbad2_plugin.md`. Same plan JSON, same `SCMetrics` fields, same worker (`default_params {plugin: "hbad2"}` -> `results/params_yaml/r4_hbad2_v2g.yaml`).
- **Roster** (`params_to_yaml(::HBAD2Params)` / `_hbad2_from_yaml`): `global` (mD2 mA pH pD pM form mSD sigma_v), `species` = one row per v2 KEY (the 10 own species + `_H`, `_S`) with the 15 per-key parameters and rho0 rhoS rhoL D0 H0 A0,
  `ecoregions` (the 29 eco cells of the Region-4 sets, each listing all keys in order: the parameters do not depend on the cell, the cells only carry `eco_mapping`), `species_map` (every REF_SPECIES symbol -> its own key, else `_S`/`_H` by
  SFTWD_HRDWD). Unknown globals / species parameters, missing ones, bad `species_map` targets or ecoregions with another species list are errors. Equal (exactly) to the parameters built from `results/hbad2/*.csv`.
  Produced by `tools/export_params_yaml.jl --hbad2 <base roster> <params_dir> <out> "<name>"`.
- **Initial state = the trees** of the plot visit (not the curated cohort table): for each requested CN, `Data.load_tree_cohorts` on the scenario's `fia_db` (table `curated_trees`, READ_ONLY, bound parameters, `ORDER BY` the full
  key) with the visit (state, unit, county, plot, measdate) of the plot (or of its TreeMap donor), species -> key by `species_map`. **Each FIA plot = its 4 SUBPLOT sites** (per-ha densities: subplot trees x 4 as in the loader; empty subplots
  are empty sites). All 4 sites share the plot's ecoregion, acres, management and plot id (`mapcode`). A visit without a live measured tree with estimated age starts empty on the 4 sites (regeneration only, warning);
  a plot without curated cohorts takes its donor exactly as for the other plugins (status `no_data`/`out_of_region` unchanged).
- **Metrics** (`_plot_summary`): per-site summary from the native state (tph, dq, h of every cohort incl. vigour sub-cohorts) -> per-ha MEAN over the plot's 4 sites; BA, TPA, QMD (from mean BA and TPA), volumes by `cohort_volume` at each
  cohort's (tph, dq, h), the species x 2-inch DBH tally (divided by 4 sites), AGB and carbon from the plugin's reported allometric `c_bio` (reporting only). The volume equation is that of the cohort's KEY: the 10 own species use their species
  equation, every other species shares `_H` / `_S` and gets the hardwood / softwood equation; the tally reports `_H`/`_S` with synthetic codes -1/-2 (`spsym`).
- **Management** acts on the STAND = the plot's 4 sites together: stand selection (rank/area) per plot; cohort selection (species, age, DBH) over all 4 sites; residual BA / TPA / SDI targets are plot means (the removal order `below` /
  `above` runs over all cohorts of the 4 sites by QMD, `proportional` takes the same fraction everywhere), so the target is met exactly by the plot mean (tests: 1e-9 relative). `percent_biomass` scales every selected cohort by (1 - value);
  `clearcut` empties the sites. Removal is by trees at constant size: `c_tph` x keep, dq/h/age unchanged, reported AGB recomputed; a cohort leaves only when ALL its trees are removed (no 0.999 / 1 g/m2 rule). Removed volume and carbon
  are plot means. A species filter that names a species mapped to a group key (e.g. QUAL -> `_H`) selects the whole group key.
- **Planting** (`plant` and `plant_after`): `tpa x survival / 0.4047` trees/ha on EACH of the 4 sites -> 3 vigour sub-cohorts (D 2.54 cm, H 3 m, age 1, shares 1/6, 2/3, 1/6), merging into existing planted sub-cohorts of the same species
  and age. **Planted-stand rule:** applying a plant action to a plot sets `planted = true` on its 4 sites; from then on the v2g planted multipliers (pH, pD, pM) act on ALL cohorts of those sites (the flag is a stand attribute in the model,
  not per cohort), also on the pre-existing trees. It is never reset.
- **Cleanup**: the negligible-cohort cleanup (`hbad2_cleanup`, default 1e-3 trees/ha in scenario runs, 0 = off) is described in `docs/hbad2_plugin.md`.
- **Determinism**: no random numbers; the annual step is independent of the thread count (`tools/scenario_plugin_check.jl` prints the hash for `r4_hbad2_v2g`; identical at 1 and 2 threads).
- Limits: stage 1 has no calibration (fixed v2g parameters; the single-species-key volume equations above); carbon/volume are reporting quantities of the cohort state, not dynamics; cost grows with the cohort count (see the cleanup table).
