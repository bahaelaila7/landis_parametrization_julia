# Pan

Pan parametrizes forest-succession models against FIA remeasurement data. It simulates every inventory plot with a plugin model
(LANDIS-II Biomass Succession, a cohort PnET, or the empirical HBAD model), scores the simulated cohorts against the observed
age and biomass distributions (Wasserstein distance on age CDFs plus an AGB term), and searches the parameter space with
(multi-objective) evolutionary / annealing optimizers. The same simulation core serves the scenario engine (harvest / planting
projections) and the spatial LANDIS export.

Architecture, data flow, scoring rule, plugin interface and determinism rules: [docs/architecture.md](docs/architecture.md).
Tools: [docs/tools.md](docs/tools.md). Design notes: `docs/`.

## Layout

```
src/Pan.jl            module: sub-modules, then the flat files in dependency order
src/PanCore.jl        SoA engine (SiteSoA, SiteView, plugin hooks)       src/plugins/   BiomassSuccession, PnET, HBAD
src/config/           strict YAML reader, option structs, plugin / param-space / data-config builders
src/data/, sites.jl, reference.jl    data layer, initial sites, scoring reference
src/scoring/, loss/, sim/            injection + disturbance rule, loss kernels, fit_params + tier accumulators
src/search/, drivers/, io/           optimizers, search drivers (run_search_loop), checkpoint / output writers
src/entry.jl          parametrize, run_from_yaml, main, julia_main
src/scenario*.jl, climate_scenario.jl   scenario engine
configs/region4, configs/pnet           run configurations (YAML)
tools/                active command-line tools (status: docs/tools.md);  tools/legacy/  retired tools (tag refactor-step7)
test/                 unit suites, runtests.jl, golden harness (test/golden)
scripts/              precompile_exec.jl (sysimage / app build)
docs/                 architecture and design notes
```

## Setup and running

All commands run from the repository root and use `./julia_gdal.sh` instead of `julia` (it sets `GDAL_DRIVER_PATH`
from `GDAL_jll`). Dependencies: `./julia_gdal.sh --project=. -e 'using Pkg; Pkg.instantiate()'` (`make prepare` also runs
`Pkg.update()`). `Manifest.toml` is not tracked.

A run is one YAML config. Data and output locations are environment variables expanded inside the YAML (`${PAN_DATA}`,
`${PAN_OUT}`); the region-4 configs read `${PAN_DATA}/slim_r4.duckdb` (built by `tools/build_slim_db_region.jl`) and further CSV
tables from the same directory.

```bash
export PAN_DATA=/path/to/data_r4 PAN_OUT=/path/to/outputs
./julia_gdal.sh --project=. --threads=auto -e 'using Pan; Pan.run_from_yaml("configs/region4/gafl_A.yml")'
# same thing:  make yaml RUN=configs/region4/gafl_A.yml
# override keys for a short test:
./julia_gdal.sh --project=. --threads=2 -e 'using Pan; Pan.run_from_yaml("configs/region4/hbad_gafl_smoke.yml"; overrides=Dict{String,Any}("trials"=>16, "output_dir"=>"/tmp/out"))'
```

* Unknown or retired keys are errors (with a spelling suggestion); `Pan.config_docs()` prints the key table.
* `n_folds > 1` writes each fold to `<output_dir>/fold_<k>`; `resume_from` continues from a `search_state@N.jld2` checkpoint.
* Outputs per run: `best_params@N.{jld2,json}`, `search_state@N.jld2` (+ `search_state_latest.jld2`), `losses.duckdb`,
  `split_plots.csv`, `species_map.csv`, and for multi-objective drivers `metrics.csv` and the archives (see architecture.md, section 2).
* Threads: results do not depend on the thread count (per-site RNG, ordered queries). `PAN_PARALLEL=soa|candidate` selects
  site-parallel or candidate-parallel evaluation; `PAN_TIMING`, `PAN_SKIP_VAL`, `PAN_NEWBEST_STATS` are read once per run.

Other Makefile targets: `make run` (`Pan.main()`), `make debug` (`JULIA_DEBUG=Pan`), `make spatial`, `make landis`,
`make export-landis`, and the image builds below.

### Fast start (sysimage / app)

```bash
PAN_DATA=/path/to/data_r4 make sysimage        # Pan.so, traced by scripts/precompile_exec.jl (golden-size GA+FL fit)
make sysimage-deps  /  make sysimage-fit       # dependency-only images (Pan recompiles on top)
PAN_DATA=/path/to/data_r4 make exec            # standalone app in build/ (entry: Pan.julia_main)
```

`scripts/precompile_exec.jl` traces config parsing for every shipped config and, when `PAN_DATA` holds `slim_r4.duckdb`, a two-evaluation
Igel run on the 80 golden plots; without data it warns and bakes the config parser only.

## Configs

| Config | Model | Use |
|---|---|---|
| `configs/region4/gafl_A.yml`, `gafl_A2.yml` | Biomass Succession, Sim A (tier 3) | GA+FL calibration (stage A, widened boxes) |
| `configs/region4/gafl_B.yml`, `simab_B.yml` | Biomass Succession, Sim B (`dual_mode: b`) | free-simulation calibration seeded from Sim A |
| `configs/region4/gafl_hbad.yml`, `gafl_hbad_B.yml`, `r4_hbad.yml` | HBAD | Sim A / Sim B / region 4 |
| `configs/region4/gafl_pnet.yml`, `r4_pnet.yml`, `configs/pnet/r4_pnet_igelmo.yml` | PnET | Sim A / region 4 |
| `configs/region4/r4_A.yml`, `r4_igelmo_full.yml`, `gafl_states.yml`, `r4_gafl_test.yml` | Biomass Succession | region 4 and shake-down runs |
| `configs/region4/r4_smoke_2l3.yml`, `hbad_gafl_smoke.yml`, `configs/pnet/pnet_smoke_853*.yml` | BS / HBAD / PnET | smoke runs (plumbing, not calibrations) |

## Tests

```bash
./julia_gdal.sh --project=. --threads=2 test/runtests.jl      # unit suites + subprocess scripts; prints the final "Pan | N N" line
```

Golden harness (46 scenarios on 80 fixed GA+FL plots, every search driver, tier, injection mode and plugin; see `test/golden/README.md`):

```bash
export PAN_DATA=/workspace/landis_region4/data_r4
bash test/golden/run.sh <repo_root> <out_root> 2 [scenario ...]
python -I test/golden/compare.py <reference_out_root> <candidate_out_root> [scenario ...]
```

References are recorded outside the repository (`/workspace/pan_refactor/golden_out`). Scenario-engine regression hashes:
`PAN_DATA=<data_r4> ./julia_gdal.sh --project=. --threads=2 tools/scenario_plugin_check.jl` (one line per plugin).

## History

The refactor (module layout, explicit state, strict config, shared search loop) is documented step by step in
`/workspace/pan_refactor/WORKLOG.md`; the pre-refactor tree is the tag `pre-refactor`, the thesis state is `thesis-2026-07`,
the last tree before the tool clean-up is `refactor-step7`.
