# Region 4 generalization, species-balanced training, and the PnET plugin

Status: October 2026. Branches `plugin-iface` → `region4` → `pnet` (each stacked on the previous one, all cut from
`split-productive-species`). Code lives in this repository; region data lives outside git in `data_r4/` (see §6).

## 1. What changed and why

| Goal | Problem it addresses | Where |
|---|---|---|
| Fit the whole southern region (EPA Region 4) instead of one L3 ecoregion (8.5.3) | params were valid only for the Southern Coastal Plain | `region4` |
| Species-balanced TRAINING set | commercial pines dominate the data (≈56 % of 8.5.3 plots and ≈40 % of R4 plots are PITA/PIEL/PIPA2/PIEC2-dominated; 76 % of cohort rows in planted plots), which biased the fit (cand1 on 8.5.3: PIEL +81…91 %, PITA +36…49 %, PIPA2 −38 %) and the data-derived floors | `region4` |
| Keep the dimensionality tractable with 14 L3s | per-(L3 × cell × species) growth params would be thousands of dims | `region4` (`param_pooling`) |
| Second succession model (PnET-Succession ecophysiology), fitted with the same loss + search | BiomassSuccession is a phenomenological growth curve; PnET is process-based (climate, soil water, light) | `plugin-iface`, `pnet` |
| One source of truth for data-prep in analysis scripts | ~28 scripts hand-copied the training kwargs and drifted (e.g. `scatter_sim_obs.jl` ignored `siteclass_scheme`) | `region4` |

Every commit was checked against a fixed-seed **regression gate** (§7): with none of the new yaml keys set, the 8.5.3
production config reproduces the `split-productive-species` search archive byte-for-byte.

## 2. The region

Shapefile (user-provided): `/workspace/EPA_ECOREGIONS/reg4_eco_l4/reg4_eco_l4.shp` — EPA Region 4 Level-IV ecoregions,
8 states: AL FL GA KY MS NC SC TN (FIPS 1, 12, 13, 21, 28, 37, 45, 47). Its 14 Level-III ecoregions (area inside R4 from
the DBF; remeasured FIA plots from `tools/region_inventory.jl`; pine-dom = share of plots whose AGB-dominant species at
the last visit is a commercial pine):

| NA_L3 | Name | km² in R4 | plots | pine-dom | site cells used |
|---|---|---|---|---|---|
| 8.3.5 | Southeastern Plains | 304,360 | 10,429 | 0.54 | A/B/C/D |
| 8.5.3 | Southern Coastal Plain | 138,942 | 3,620 | 0.54 | A/B/C/D |
| 8.3.4 | Piedmont | 132,397 | 4,416 | 0.44 | A/B/C/D |
| 8.3.3 | Interior Plateau | 102,956 | 2,198 | 0.04 | X |
| 8.5.1 | Middle Atlantic Coastal Plain | 54,577 | 1,833 | 0.57 | A/B/C/D |
| 8.3.6 | Mississippi Valley Loess Plains | 44,195 | 1,162 | 0.36 | A/B/C/D |
| 8.4.1 | Ridge and Valley | 38,850 | 1,107 | 0.26 | X |
| 8.4.9 | Southwestern Appalachians | 37,995 | 1,321 | 0.19 | X |
| 8.4.4 | Blue Ridge | 35,902 | 1,794 | 0.04 | X |
| 8.5.2 | Mississippi Alluvial Plain | 22,800 | 286 | 0.01 | X |
| 15.4.1 | Southern Florida Coastal Plain | 22,533 | 137 | 0.14 | X |
| 8.4.2 | Central Appalachians | 18,125 | 676 | 0.01 | X |
| 8.3.2 | Interior River Valleys and Hills | 14,199 | 278 | 0.03 | X |
| 8.4.3 | Western Allegheny Plateau | 10,325 | 335 | 0.01 | X |

Region membership is **state ∩ L3** (`filter_states` + `filter_ecos`): an L3 contributes only its plots inside the 8
states. Consequence: Louisiana plots of 8.5.3 (and other non-R4 states) drop out, so R4 runs are **not directly
comparable** to the older `fl853_*` runs.

## 3. New yaml keys (all optional; absent ⇒ previous behaviour)

### Data selection and stratification (`region4`)
| key | default | meaning |
|---|---|---|
| `filter_states` | `[]` | FIPS state codes; adds `AND statecd IN (...)`. Combined with `filter_ecos` ⇒ state ∩ eco. |
| `siteclass_l3s` | `[]` (all) | with `site_class_strata`, A/B/C/D site cells only in these ecos; every other eco collapses to one cell `X`. Avoids thin cells in hardwood L3s. |
| `max_exact`, `max_group` | 12, 4 | species-tiering caps (own-param species / pooled groups). Region runs: 30 / 8 with `min_agb_frac: 0.005` (at 0.02 only ~11 species qualify region-wide). |

### Species-balanced training (`region4`)
| key | default | meaning |
|---|---|---|
| `train_cap` | unset | cap TRAINING plots per (eco cell × dominant species). Int, or `"auto"`. |
| `train_cap_k` | 1.5 | `"auto"` cap = max(`train_cap_min`, ceil(k × median stratum size within that eco cell)). |
| `train_cap_min` | 20 | lower bound of the auto cap. |
| `split_only` | false | write `split_plots.csv`, `species_map.csv`, `pool_map.csv` to `output_dir` and stop (floor-building pass). |

Algorithm (`Data._cap_train_sites`, runs after the existing proportional split, so val/test selection of rare species is
untouched): dominant species of a plot = largest summed AGB at its last visit (ties by name); train plots are grouped by
(eco cell, dominant species); within a group plots are ordered by their normalized position in the existing seeded
per-stratum shuffle (no new random draws, and no bias toward low FIPS codes); plots beyond the cap are moved
alternately to val and test (2-way split ⇒ all to val). Nothing is discarded. Deterministic and resume-safe.

**Chosen setting (user decision): `train_cap: auto`, `train_cap_k: 4.0`** ("softer all-species"). Full R4: train
14,635 → 8,756 plots; PITA-dominated 4,045 → 558, PIEL 981 → 312; hardwoods trimmed modestly (e.g. QUAL 604 → 336).
Alternatives evaluated: k = 1.5 (train 6,412; PITA → 430; QUAL/LITU/LIST2 cut by ~60 %), pine-only cap at the largest
non-pine stratum (train 11,267; PITA → 1,279). Side effect: val/test become pine-heavy — report metrics per species.

### Parameter pooling and covariance (`region4`)
```yaml
param_pooling:
  min_plots: 150        # TRAIN plots an (L3, species) needs to own a dim at level l3 (else fall back)
  floor_pooled: max     # pooled slot's B_MAX/PROB_ESTAB floor = max over members (first = legacy)
  params:
    B_MAX_SPP:    {level: l3, fallback: [l2, region], cell_split: [PITA, PIEL]}
    ANPP_MAX_SPP: {level: l2, fallback: [region],     cell_split: [PITA, PIEL]}
    S:            {level: l2, fallback: [region],     cell_split: [PITA, PIEL]}
  cell_merge: {PIEL: {A: AB, B: AB}}
  overrides:  {B_MAX_SPP: {"8.4.4|QURU": region}}   # optional: force a level for "L3|SPECIES"
cmaes_blocks: l3        # eco (legacy, per cell) | l3
```
- Levels: `cell` (one dim per eco cell), `l3`, `l2` (first two code components: 8.3.5 → 8.3, 15.4.1 → 15.4), `region`.
  A target falls back along `[level, fallback...]` until its TRAIN-plot count at that level ≥ `min_plots`.
- `cell_split` species get one dim per site cell (merged by `cell_merge`) wherever their eco has real cells (not `X`).
- Species traits (D, LONGEVITY, SHADE_TOL, MATURITY) are Species-sampled and therefore already shared region-wide.
- Labels are computed from TRAIN counts only (no leakage), are deterministic given the split (resume-safe), and are
  written to `output_dir/pool_map.csv` (param, eco, species, n_train_plots, label).
- `cmaes_blocks: l3`: one covariance block per (param group, L3); pooled dims confined to one L2 get an L2 block; dims
  spanning several L2s go to a region tail block. Full R4: 19 blocks, sizes ≤ 100 (was a single 325-dim tail before
  the L2 split).
- Dimension budget (full R4, `min_plots 150`): B_MAX 297 + ANPP 153 + S 153 + ~44 species traits ≈ **650**.

### Plugin selection and PnET (`pnet`)
| key | default | meaning |
|---|---|---|
| `succession_plugin` | `biomass` | `biomass` (BiomassSuccession) or `pnet`. |
| `pnet_climate_cache` | — | JLD2 from `tools/build_pnet_climate.jl` (required for pnet). |
| `pnet_co2_csv` | — | `year,co2_ppm` (required for pnet). |
| `pnet_soil_texture` | `{}` | `{L3: TEXTURE}` per-L3 soil class (default LOAM) — used when `pnet_soil_csv` is unset. |
| `pnet_soil_csv` | — | per-plot SOLUS soil hydraulics from `tools/build_plot_soil.jl` (§5.4). |
| `pnet_generic` | `{}` | overrides of `PnETConst` fields (e.g. `{init_biomass_death: false}`); unknown names error. |

## 4. Region workflow (BiomassSuccession)

```bash
# 0. data (once): slim DB for state ∩ L3 (+ PLOT coords) and the inventory
./julia_gdal.sh --project=. tools/build_slim_db_region.jl --out data_r4/slim_r4.duckdb \
    --states 1,12,13,21,28,37,45,47 --ecos 8.3.5,8.5.3,8.3.4,8.3.3,8.5.1,8.3.6,8.4.1,8.4.9,8.4.4,8.5.2,15.4.1,8.4.2,8.3.2,8.4.3
./julia_gdal.sh --project=. tools/region_inventory.jl data_r4/slim_r4.duckdb
# 1. split pass (writes split_plots.csv / species_map.csv / pool_map.csv)
PAN_DATA=$PWD/data_r4 ./julia_gdal.sh --project=. -e 'using Pan; Pan.run_from_yaml("configs/region4/r4_igelmo_full.yml";
    overrides=Dict{String,Any}("split_only"=>true,"output_dir"=>"runs_local/r4_split","bmax_floor_from_data"=>false,"anpp_floor_from_data"=>false))'
# 2. floors from the TRAIN plots of that split, with the run's exact species categories
./julia_gdal.sh --project=. tools/build_floors_region.jl --run runs_local/r4_split --db data_r4/slim_r4.duckdb --out data_r4/floor_r4
# 3. the fit
PAN_DATA=$PWD/data_r4 PAN_OUT=... PAN_PARALLEL=candidate ./julia_gdal.sh --project=. --threads=N -e 'using Pan; Pan.run_from_yaml("configs/region4/r4_igelmo_full.yml")'
```
Floors (`tools/build_floors_region.jl`, a Julia port of `FIA_DATA_PREP/build_floors_4cell_shade.py` +
`build_anpp_floor_4cell_shade.py`): `bmax_floor = max(12000, p99 category biomass per plot-visit, 20 × p99 ΔAGB/Δt
[3–15 yr gaps])`, `anpp_floor = p99 plot-wide species increment (sum-then-diff)`, and the invariant
`bmax_floor ≥ 20 × anpp_floor` is enforced directly (replaces the hand-tuned "anppfix" +5 % rows). Cell-split species
get per-cell groups inside each L3; other categories pool the L3's cells; groups with < `--min-n` (30) values fall back
L3 → L2 → region (`level` column). Pan's loaders key on (category, l3, lu) — unchanged. On the 2-L3 smoke, the capped
training set lowered the binding PIEL AB B_MAX floor 28,830 → 23,175 and ANPP 1,382 → 1,159.

Analysis scripts (`test/scatter_sim_obs*.jl`, `test/tost_sim_obs.jl`, `test/smape_agebin.jl`, `test/sweep_archive_tost.jl`,
`tools/scatter_archive.jl`, `tools/check_rankw.jl`) now call `Pan.prepare_from_cfg(cfg)`, which derives the data-prep
kwargs exactly as training does (`Pan.data_config`). Behaviour changes: their `filter_eco_field` default is now
`epa_l3` (training default); `tools/scatter_archive.jl` follows the config's split (it used to force no split).

Configs: `configs/region4/r4_smoke_2l3.yml` (plumbing), `configs/region4/r4_igelmo_full.yml` (production, 14 L3),
`configs/region4/r4_gafl_test.yml` (local test, the 7 L3s touching GA/FL — §8).

## 5. PnET plugin

Science spec with `file:line` citations into the LANDIS-II C# reference (`Library-PnET-Cohort`, v2, 2026-09):
`docs/pnet_spec.md`. The C# was used as a **science reference only**; the Pan implementation follows Pan's design:
Structure-of-Arrays state, static dispatch, allocation-free threaded loops, Float32 state.

### 5.1 Plugin interface (`plugin-iface`, `src/succession_api.jl`)
`succession_plugin_of(soa)` (pure type computation) and per-plugin methods `params_type`, `make_param_dists`,
`generate_params`, `generate_eco_params`, `make_ctx`, `step!`, `spinup!`, `add_cohort!`, `per_eco_groups`,
`set_cohort_bio!`, `init_sites!`. `fit_params`, all search drivers, injection and disturbance code dispatch through
these. `PanCore.swap_cohort!` (generated) moves every `:cohort` CSR array on swap-remove, so plugins with extra
per-cohort state cannot desync. Spatial/export code (`simulate_spatial_*`, LANDIS export) is still BiomassSuccession-only.

### 5.2 Model (`src/plugins/PnET/`)
- Monthly PnET-II physiology inside the annual `process_plugin!`: each cohort has IMAX = 5 foliage sublayers; cohorts
  form up to 3 canopy layers by BiomassMax (LayerThreshRatio 0.5); layers are processed top → bottom and sublayers in a
  per-site-year random order (site RNG) with sequential soil-water depletion; snow, interception, Saxton–Rawls water
  retention; December turnover, NSC → structure, age + 1; deaths: NSC fraction ≤ 0.01, and (optional) TotalBiomass <
  InitBiomass.
- Pools per cohort: TotalBiomass (wood + root), Fol, NSC (gC). Reported cohort biomass (what the loss sees):
  `c_bio = ((1 − FracBelowG)·TotalBiomass + Fol) · CanopyLayerProp` — matching the reference's reported biomass, without
  the reference's rounding bug at `Cohort.cs:142` (which rounds `1 − FracBelowG` to 0 or 1).
- Observed-cohort injection: `add_cohort!` stores the observed AGB and marks the cohort pending; at the start of the next
  year the spec §8 circular solve (bisection) finds TotalBiomass such that reported AGB = observed, using the candidate's
  own fitted parameters. `set_cohort_bio!` rescales pools proportionally for sync/disturbance writes.
- Climate (`tools/build_pnet_climate.jl` → `data_r4/pnet_climate_r4_l4.jld2`): PRISM monthly means per EPA L4 polygon,
  1980–2025 (`/workspace/PRISM_LT/PRISM_LT_EPA/prism_lt_epa.duckdb`; values are °C, mm, hPa — the README's °F/inches is
  wrong; values are stored as text and must be CAST). Mar–Dec 2025 (PRISM ends Feb 2025) filled with each month's
  30-year climatology. PAR (not in PRISM) = FAO-56 extraterrestrial radiation × Bristow–Campbell transmissivity
  0.7·(1 − exp(−0.004·ΔT^2.4)) → daytime mean µmol m⁻² s⁻¹; CO₂ = NOAA Mauna Loa annual means (`co2_annual.csv`).
  Calendar year of sim year t = plot's first-measurement year + t, clamped to 1980–2025.
- Threading: site loop `Threads.@threads :static` (or candidate-level `:static` in `PAN_PARALLEL=candidate`), per-thread
  scratch, zero allocations per simulated year (unit-tested).

### 5.3 Parameters
Fitted (bounds):
| param | sampler | bounds | group |
|---|---|---|---|
| FolN | EcoSpecies (poolable via `param_pooling`) | 0.6–3.0 % | `:growth_pnet` |
| MaxFracFol | Species | 0.01–0.40 | `:pnet_species` |
| FrActWd | Species | 2e-5–1.2e-4 | `:pnet_species` |
| SLWmax | Species | 30–300 g m⁻² | `:pnet_species` |
| TOwood | Species | 0.002–0.04 yr⁻¹ | `:pnet_species` |
| PsnAgeRed | Species | 2–8 | `:pnet_species` |
| RD_MULT | Eco | 0.4–2.0 (× rooting depth) | `:soil` |

Fixed, by plant functional type (needle-leaf evergreen, broadleaf deciduous, broadleaf evergreen, deciduous conifer;
`pft_of` in `params.jl`), with sources documented next to each value in `src/plugins/PnET/params.jl` (reference test
inputs `PnETSpeciesParameters.txt` rows pinustro/pinuresi/querrubr, and `PnETGenericDefaultParameters.txt`). Judgment
calls: warm-adjusted PsnTOpt/PsnTMax for needle-leaf (25/33 °C instead of 23/30) so Southeast summer Tday does not zero
photosynthesis; broadleaf-evergreen and deciduous-conifer rows have no reference row and are interpolated. Global
constants: `PnETConst` (IMAX 5, MaxCanopyLayers 3, CFracBiomass 0.45, DNSC 0.05, MaintResp 0.002, …), overridable via
`pnet_generic`.

### 5.4 Soil (SOLUS, per plot)
`tools/build_plot_soil.jl` → `data_r4/plot_soil_r4.csv`; enabled with `pnet_soil_csv` (unset ⇒ the legacy per-L3
texture path `pnet_soil_texture`, default LOAM, bit-identical).
- Inputs: `/workspace/SOLUS/SOLUS100/hydro/derived/{theta_s,theta_33,theta_1500,awc,ksat}_<d>_cm.tif` (Saxton–Rawls 2006
  from SOLUS sand/clay/SOC by `hydro/hydro4.sh`; 100 m, Float32, nodata −9999; d = 0,5,15,30,60,100,150 cm; ksat mm h⁻¹)
  and restriction depth `min(resdept_all_cm_p, anylithicdpt_cm_p)` (UInt8 cm, nodata 255; 201 = no restriction within 2 m).
- Per plot (mean LAT/LON over visits): rooting depth D = clamp(10 × restriction_cm, 100, 1500) mm (1000 mm if missing);
  porosity = θs, fieldcap = θ33, wiltpnt = θ1500 averaged over 0–D on the piecewise-linear depth profile (trapezoid);
  awc_mm = ∫awc over 0–D. Missing plots → L4 median → L3 median → LOAM (`fill_level`).
- Coverage (R4): 30,346 / 30,542 plots (99.4 %) own values, 196 L4-filled; 0 invariant violations
  (wiltpnt < fieldcap < porosity). Sanity: 8.5.3 (sandy coastal plain) median fieldcap 0.077, wiltpnt 0.039,
  ksat 98 mm/h vs 8.4.4 (Blue Ridge) 0.226, 0.117, 12 mm/h.
- Water-retention curve exactly as `PressureHeadSaxton_Rawls.cs:150`: B = (ln 1500 − ln 33)/(ln θ33 − ln θ1500),
  A = exp(ln 33 + B ln θ33), ψ[kPa] = A θ^−B, head[m] = 0.1019977·ψ (0 above porosity); repairs (counted) keep
  wp ∈ [0.005, 0.6], fc ≥ wp + 0.01, por ≥ fc + 0.01, B ∈ [1, 12]. The fitted `RD_MULT` multiplies the plot's D.

### 5.5 Known gaps
No spinup, establishment, ozone, soil ice/permafrost, dead-pool decomposition, cold kill or refoliation. Simplified
hydrology (no surface-water/runoff capture). The months between a plot's first measurement date and January are not
simulated. PAR is derived from the monthly diurnal range (Bristow–Campbell is a daily model). VPD uses Tmin as the
dewpoint proxy (PRISM vpd/tdmean unused). The InitBiomass death rule (on by default, as in the reference) would kill
≈5.8 % of observed cohorts at their first December (0.2 % of AGB) — toggle with `pnet_generic: {init_biomass_death: false}`.

## 6. Data and tools

| path | what | produced by |
|---|---|---|
| `data_r4/slim_r4.duckdb` | 30,542 R4 plots: cohort tables, curated_trees, COND, **PLOT** (coords), REF_SPECIES, longevity | `tools/build_slim_db_region.jl` |
| `data_r4/slim_r4_inventory.csv` | per-L3 plots, site cells, pine share, top dominants | `tools/region_inventory.jl` |
| `data_r4/floor_r4_{bmax,anpp}.csv` | full-R4 floors (from the final split's train plots) | `tools/build_floors_region.jl` |
| `data_r4/floor_gafl_{bmax,anpp}.csv` | floors for the GA/FL test config | same |
| `data_r4/pnet_climate_r4_l4.jld2` | 99 L4 × 1980–2025 × 12 months × 7 vars | `tools/build_pnet_climate.jl` |
| `data_r4/co2_annual.csv` | 1959–2025 annual CO₂ (NOAA MLO) | downloaded |
| `data_r4/plot_soil_r4.csv` | per-plot SOLUS soil hydraulics | `tools/build_plot_soil.jl` |
| `data_r4/shadetol_all_species_ext.csv` | symlink to the main `runs/` file | — |

All source databases are opened READ-ONLY. `data_r4/` and `runs_local/` are git-excluded; copy `data_r4/` to the
cluster data dir (`$RUNSDIR/data`) before running there.

## 7. Verification protocol

1. **Regression gate** (`tools/regress_gate.sh <repo> <out>`, baseline `docs/regress/baseline_853_split-productive-species.txt`): 40-evaluation Igel search (igel_mu 6, n_reps 1, no Sobol init) on
   `fl853_igelmo_4cell_shade_n1000_s_anpp.yml`, dumps the sorted archive objectives at full precision; must be
   byte-identical (`cmp`) to the `split-productive-species` baseline. Baseline determinism confirmed by two independent
   runs. Passed after every commit on all three branches.
2. **Unit tests**: `test/split_cap_test.jl` (determinism, partition, cap, unset ⇒ legacy), `test/pnet_unit.jl` (monthly
   fluxes vs an independent transcription ≤ 1.6e-7 rel; carbon closure over 2 years ≤ 5.6e-7; injection solve within
   2 g/m² for 50…20000; determinism; zero allocations per simulated year; (f) soil-curve derivation/repairs/legacy
   equivalence; (g) solve beyond the reported-AGB peak). Run: `./julia_gdal.sh --project=. test/pnet_unit.jl`.
3. **Script equivalence**: converted `tost_sim_obs.jl` output == pre-conversion script on identical inputs.
4. **Smokes**: 2-L3 region smoke, full-R4 smoke (floors loaded for all 1,033 eco×species cells; 6.8 GB peak; ~41 s/gen
   for 10 candidates × 1 rep on 6 threads), PnET 8.5.3 smoke (median simulated/observed site AGB 0.88 before, 0.95 after
   injection).

## 8. Local GA/FL test run (BiomassSuccession, 2026-10-13)

Config `configs/region4/r4_gafl_test.yml`: the 7 L3 ecoregions with plots in Georgia or Florida — 8.3.5, 8.3.4, 8.5.3,
8.4.4, 8.4.9, 8.4.1, 15.4.1 — taken WHOLE within Region 4 (22,824 remeasured plots; restricting to GA+FL plots only would
be `filter_states: [12, 13]`, 8,985 plots); site cells in 8.3.5/8.3.4/8.5.3; species-balanced sampled training set
(`train_cap: auto`, k = 4); production loss/search settings except igel_mu 30, n_reps 2, 900 evaluations (20 gens).
- Split pass: 242 species → 30 exact + 12 groups; train 11,343 → **6,220** (cap) ; val 8,325; test 8,279.
  Dims: B_MAX 194 + ANPP 132 + S 132 + ~42 traits ≈ 500; 11 covariance blocks (max 108). Floors: 656 rows from train plots.
- Run: 40 min on 6 threads (~1 min/gen, initial Sobol screen of 300 candidates first); peak RSS 7.0 GB.
- Best-aggregate candidate (archive of 15): objectives W 0.0945, AGB 2.0746; train loss 2.169, val loss 3.910.
  **The best aggregate came from the initial Sobol screen and did not improve over generations 3–20** (the archive grew
  6 → 15) — expected for a ~500-dim (1+1)-CMA-ES population after 20 generations; this is a pipeline shake-down,
  not a converged fit.
- Per-species AGB bias (TOST, ±10 % margin; geometric-mean sim/obs, val = held-out plots):

| species | train | val | (8.5.3 cand1, old fit) |
|---|---|---|---|
| PITA | −49 % | −40 % | +36…+49 % |
| PIEL | −20 % | −15 % | +81…+91 % |
| PIPA2 | −4 % | −10 % | −38 % |
| QUPR2 | −10 % | −7 % | — |
| NYBI | +6 % | +3 % | — |
| ACRU | −22 % | −23 % | — |
| QUAL | −24 % | −29 % | — |
| LIST2 | −34 % | −37 % | — |
| LITU | −37 % | −38 % | — |
| QUNI | −47 % | −48 % | — |

  Median |bias| 30 % (train) / 32 % (val); 2 groups equivalent at ±10 %. Reading: the commercial-pine over-prediction
  is gone and train ≈ val (the fit generalizes to the pine-heavy val set), but nearly every species is under-predicted —
  consistent with an under-converged search. A production-length run (cluster) is needed before judging the
  balanced-training effect on accuracy.
- Outputs: `runs_local/r4_gafl_test_outputs/` (checkpoints, `analysis/` from `tools/analyze_run.jl`, `bestagg/` TOST).

### 8b. Georgia + Florida calibration (follow-up)
The GA+FL-only BiomassSuccession calibration (per-species TOST-bias calibration, real records only, 3-fold CV) is
documented step by step in `docs/gafl_calibration_log.md` (decisions D1–D10); final parameters and per-species results are
in `results/gafl/`. Headline: final parameters pass TOST ±10 % for 42/47 species groups (±20 %: 45/47) on all GA+FL plots;
out-of-sample (CV) ≈ 20/46 at ±10 % and ≈ 36/46 at ±20 %, with 14 groups passing ±10 % in every held-out fold.

## 9. Review (2026-10-13) and fixes

An independent code review of the full branch diff found no high-severity defects. Fixed:
| finding | severity | fix |
|---|---|---|
| PnET injection solve: observed cohort AGB above the maximum reportable AGB (reported AGB peaks because foliage/LAI saturates) returned a runaway TotalBiomass ≈ 8e6 → reported 0 → cohort deleted in year 1 (loss cliff inside the search bounds) | medium | detect on the rising-branch bracket, start at the peak (golden-section); counter `PnETPlugin.UNREACHABLE_INIT`; test (g) |
| Sobol `n_output_plots` branch, `_run_and_plot`, `simulate_and_test` built BiomassSuccession sites in PnET runs | medium-low | `make_sites_like` / `succession_plugin_for` / `SITE_ATTRS_FN` |
| CC-Igel `species_slot_map` dropped pooled (Vector-target) slots | low | map to the slot's species |
| `train_cap` silently ignored without a split; stale split-dominance table across calls | low | warning + reset per call |
| `pnet_generic.max_layers` > MAXL ⇒ out-of-bounds scratch write | low | validated in `configure!` |
| floor builder lumped tree species absent from `species_map.csv` into `_H_NA` | low | excluded (0 rows change on current data) |
Accepted as documented behaviour: floor increments skip visits where a category is absent (same as the original Python
recipe); unknown-L4/no-latitude PnET sites fall back (L3 climate / 33° N) with a logged count; analysis scripts that
previously forced "no split" now follow the config's split (and `train_cap`).

## 10. Open issues / not done
- `test/tost_sim_obs.jl` groups by species × cell only for `param_split_species` species; with `param_pooling` it reports
  per species (no per-cell breakdown yet).
- PnET runs cannot yet be analysed by the BiomassSuccession-only scripts (`scatter_archive.jl`, `check_rankw.jl`,
  `sweep_archive_tost.jl` call `make_sites`/`generate_eco_params` for BiomassSuccession).
- Archived analysis `fl853_igelmo_4cell_shade_n1000_s_anpp_no_niche/analysis/candidates/cand1/tost_sim_obs_10pct.csv`
  does not reproduce locally (per-species nplots/n differ) although the data query, row count, exclusions, split sizes and
  code commit are identical — pre-existing local-vs-cluster difference (probably the cluster's `slim_853.duckdb`), not
  caused by this work.
- `search_mode: plot_sample` / `plot_sample_sobol` crash when `common_kw` is splatted into them (they don't accept the
  stratification keys) — pre-existing, untouched.
- `FIA_DATA_PREP` repo (Python floor/slim builders) was not modified because it has foreign uncommitted changes; the
  Julia tools here supersede them for region runs.
