# HBAD2 calibration in Pan

`succession_plugin: hbad2` plus a `hbad2_calib:` block in the yaml calibrates the HBAD v2 forest model (plugin `src/plugins/HBAD2`, `docs/hbad2_plugin.md`) inside Pan: data from the
tree table, the two likelihoods of the prototype harness (`tools/hbad2` on branch `hbad-v2`: Sim A and Sim B, `docs/hbad_v2.md` there, sections 4, 4.1, 5.8-5.24), the prototype's parameter
vector, and the search through Pan's shared machinery (`run_search_loop`, background checkpoint writer, `losses.duckdb`, resume). The semantics and the numbers are those of the prototype:
`tools/hbad2_calib_parity.jl` reproduces the prototype's own results to 1.3e-14 (see "Parity evidence").

```
PAN_DATA=/workspace/landis_region4/data_r4 PAN_OUT=/workspace/out ./julia_gdal.sh --project=. --threads=6 -e 'using Pan; Pan.run_from_yaml("configs/region4/hbad2_calib_joint.yml")'
```

Shipped configs: `configs/region4/hbad2_calib_joint.yml` (the prototype's joint fit j15: model v2j) and `configs/region4/hbad2_calib_simA.yml` (the Sim A fit v2fv5). Both use the prototype's split file.
A run without a `hbad2_calib:` block keeps the previous behaviour (fixed v2g parameters, no search).

## 1. Data (`Data.build_hbad2_intervals`, `src/data/hbad2_intervals.jl`)

The REMEASUREMENT-INTERVAL PRODUCT (port of `tools/hbad2/build_intervals.jl`), built from `curated_trees` (opened READ_ONLY; the split and the state list are registered tables, no value is
interpolated; the tree query has the total `ORDER BY STATECD, UNITCD, COUNTYCD, PLOT, MEASDATE, SUBP, TREE, CN`):

* `intervals`: one row per pair of consecutive visits of a plot (`iid st u c p d1 d2 dt k trt dst harv planted eco set l3`). Flags at v2: any treatment code (`TRTCD1 > 0`), any disturbance code
  (`DSTRBCD1 > 0`), any tree whose fate is harvest; `planted` = stand origin at v1 (`cond_stdorgcd == 1`); `dt` = years between the visits.
* `interval_trees`: every live tree of v1 with its tracked outcome at v2 (`out`: alive with DIA2/HT2/TPA2, dead, harvest, missing), linked by (plot, SUBP, TREE); sizes in the v2 units
  (cm, m, trees/ha = TPA_UNADJ x 2.47105).
* `interval_ingrowth`: trees first tallied at a later visit, with their SUBP.

`iid` is a canonical enumeration (plots in key order, intervals by date); the prototype's ids came from Dict iteration order, so the two products are compared through the natural key
(`Data.compare_interval_products`, order independent). `tools/hbad2_intervals_check.jl` rebuilds the product for the prototype's split and compares with `results/hbad2/*.arrow`:
30,703 intervals, 766,365 tracked trees, 183,698 ingrowth rows, **identical content**.

The split is a plot-level train/val/test assignment: `hbad2_calib.split_csv` = the `split_plots.csv` of a Pan run (the prototype uses
`/workspace/landis_region4/runs_local/gaflst_split/split_plots.csv`), else a deterministic plot-level shuffle from the top-level `val_frac`, `test_frac`, `split_seed`
(`Data.draw_plot_split`); the split used is written to `<output_dir>/split_plots.csv`. The product is cached as Arrow under `hbad2_calib.intervals_cache` (default
`<output_dir>/hbad2_intervals`; `cache_key.txt` records database, table, states and the split, a mismatching cache is refused; `rebuild_intervals: true` forces a rebuild).

## 2. Objectives (`src/hbad2_calib/`)

The simulation of both objectives runs the plugin kernel on the plugin's SoA (`HBAD2Plugin.run_interval!` and `site_year_free!`, new entry points that use `step_cohorts!`,
`recruit_rates!`, `recruit_apply_cap!` -- the same functions as `process_plugin!`; they skip the reported state, which the likelihoods do not use). Site loops are threaded over sites
(`@maybe_threads soa.parallel`, `:static`), each site touches only its slice; all sums over cohorts / cells are sequential in a fixed order, so the numbers do not depend on the thread
count (checked by `test/hbad2_calib_e2e.jl`: threaded == serial, bit for bit). The entry points allocate nothing.

**Tracked trees** (`load_tracked_trees`): trees of intervals with no treatment / disturbance / harvest flag and `dt > 2` yr, age, height, diameter > 0. Species key = the species symbol when it has
>= `min_sp_trees` (3000) TRAIN trees, else `_S` / `_H` (REF_SPECIES softwood / hardwood). Site = one FIA subplot of an interval (`iid*10 + SUBP`, densities x4 per hectare), or the whole plot
(`site: plot`).

**Sim A** (`build_simA`, `eval_simA!`): the cohorts of a site are (species key, 10-yr birth class `floor((y1 - age)/birth_w)`, initial size class by `size_split` 5/12.7/25/40 cm); N = sum of the v1
expansion weights, D the weighted quadratic mean, H and A weighted means (A >= 1). Each cohort is three vigour sub-cohorts (Gauss-Hermite quantiles -sqrt3, 0, sqrt3; shares 1/6, 2/3, 1/6). A
site is integrated `floor(dt)` whole years plus a FRACTIONAL last step of length `dt - floor(dt)` (`step_site_frac!`: growth increments, hazard and ageing scaled by the step, the prototype's
`step_site!(..., Delta, ...)`). The sub-cohorts are aggregated back to the observed cohort (`cohort_view!`: sum of N, quadratic mean D, N-weighted H). Objective `fA` =
joint negative log-likelihood `nll_parts`: binomial survival of the tracked tree counts (p = N2/N1, clamped to [1e-9, 1-1e-9]) + Gaussian ln D2 and ln H2 of the survivors (weight = survivors,
sigma profiled out): `total = mort + 0.5 n ln(sD/n) + 0.5 n ln(sH/n)`. With `sizecls: true` the survivors' size-class multinomial (`size_nll`, classes <5/5-12.7/12.7-25/25-40/>40 cm, lognormal spread
sigma_w around each sub-cohort QMD) is added.

**Sim B** (`build_simB`, `eval_simB!`): the clean natural plots of the set (no flag in any interval, not planted) with >= 2 intervals. Each of the 4 subplot sites (empty ones included) starts from the
first visit's cohorts (as above) and runs FREE for the number of years to the last visit: annual kernel step, then recruitment (per species rate `185.2 exp(rho0 + rhoS ln(1 + BA_sp(D >= 12.7)) + rhoL SDI/600)`,
a recruit class of 3 sub-cohorts per species per 5 years; the site has `cap_extra` (330 with vigour) free cohort slots, classes that do not fit are dropped and counted). At every later visit the
expected number of TALLIED trees per (key, class [2.54,5)/[5,12.7)/[12.7,25)/[25,40)/[40,inf) cm) is the model N through a lognormal within-cohort diameter operator (spread `simb_sigma` 0.16, normal cdf by
the Numerical-Recipes erfc approximation of the prototype) divided by the FIA expansion (185.2 microplot saplings, 14.87 subplot trees, x4 sites); the last visit is the tracked survivors + first-tallied
ingrowth. Terms (`simB_parts`): `counts` (Poisson deviance per site x visit x key x class), `H` (Gaussian ln Lorey height, sigma profiled), `size` (species-pooled class totals per plot and visit),
`species` (per plot and visit, saplings 2.54-12.7 cm and trees >= 12.7 cm of each key), `plotstand` (key x class per plot; `plotstand_trees` the classes >= 12.7 cm). `fb` selects `fB`:
`stand` = counts + H, `size` = size + H, `margins` = size + species + H (the j10-j15 objective), `plotstand` = plotstand + H, `both` = counts + H + size.

**Joint objective** (`objective: joint`): the two objectives share plots and are not summed; each weight `lambda` of `hbad2_calib.lambda` gets its own search minimising the augmented Chebyshev
scalarisation `max(l x, (1-l) y) + rho (l x + (1-l) y)`, `x = (fA - ideal_A)/(nadir_A - ideal_A)`, `y = (fB - ideal_B)/(nadir_B - ideal_B)`, with FIXED `ideal` / `nadir` (yaml; empty = start - 5 %|start| /
the start, only meaningful at lambda 0 or 1; nadir > ideal is required) -- a moving ideal made the prototype's scalarisation non-stationary (docs/hbad_v2.md 5.12).

## 3. Parameter space (`spec.jl`)

The search vector (prototype layout, `tools/hbad2/common.jl`):

```
theta = [ 14 per species key (FREE order: Hmax rH cH Dmax rD cD gOD gRD gOH gRH mbg mD msup sdimax) | 7 globals: mD2 pH pD pM mSD sigma_v ln(sigma_w) ] ++ recruitment coordinates
```
Species parameters are log-coordinates for `Hmax rH cH Dmax rD cD msup sdimax` (form 2: `cD`, `cH` linear), decoded and clamped to a box: form 1 `Hmax 8-70, rH 1e-3-0.5, cH 0.3-4, Dmax 15-250, rD 1e-3-0.5,
cD 0.3-4`; form 2 `Hmax 2-200, rH 0.005-3, cH -1.5-3, Dmax 5-300, rD 0.01-3, cD -1.5-3`; both `gOD -1-8, gRD -3-3, gOH -1-8, gRH -3-3, mbg -9..-1, mD -4-4, msup 1e-4-1, sdimax 250-2000` (`hbad2_calib.bounds`
overrides single boxes). LONGEVITY per key is fixed data (median over the key's symbols of `longevity_csv`, 300 yr default). Globals are clamped on decode: `mD2` [-3,3], `pH pD` [-1,1], `pM` [-2,2],
`mSD` [-4,2], `sigma_v = min(|.|, sigv_max)` (0 without vigour), `sigma_w = clamp(exp(.), 0.02, 0.6)`; `mA = 10`. Recruitment (`recruitment`): `free` = 3 coordinates per key (rho0 rhoS rhoL, clamps
[-12,2] [-1,4] [-10,2]) appended to theta; `fixed` = the data fit `recruitment_csv` (the Poisson fit on observed sapling ingrowth); `kappa` = the data fit with `rho0` shifted by one fitted coordinate
`ln kappa` (clamped to [-3,1]: an overall recruitment level, the nuisance parameter of docs/hbad_v2.md 5.19). `freeze` keeps coordinates at their start values (species parameters by name,
globals as `G:mD2 G:pH G:pD G:pM G:mSD G:sigma_v`; HBAD2_FREEZE semantics), the search runs over the remaining coordinates only. Start: `start_theta` (csv column `theta`; the prototype's shorter
layouts of earlier fits are padded) or the data-based start (`init_theta`, needs `priors_csv`).

## 4. Search (`src/drivers/hbad2_calib.jl`, `src/search/PlainCMA.jl`)

* `search_mode: cmaes` -- **the prototype's CMA-ES** (`Search.PlainCMA`: (mu/mu_w, lambda), full covariance, Hansen 2016 defaults, `Random.Xoshiro(cma_seed)`, the same stopping rules: budget,
  stagnation over `cma_stall` generations below `cma_ftol`, step-size collapse), NOT Pan's `CMAES` module: that one works in the [0,1]^d u-space of the Biomass-Succession `ParamSpace`, whereas
  the HBAD2 parameters live in their own decoded space. Population `cmaes_lambda`, initial step `cmaes_sigma0`, budget `trials` (evaluations, the start counts as one). The engine implements
  the search interface (`ask`, `tell!`, `state`, `save_state`, `load_state`): a checkpoint continues exactly. Objective `sim_a` (Sim A only) or `joint` (scalarisation; one search per `lambda`,
  in `<output_dir>/lam_<lambda>/` when there are several, `front.csv` lists train and val values).
* `search_mode: igelmo` -- the population-based MO-CMA-ES of Pan (`Search.IgelMOCMAES`, `igel_mu`, `igel_sigma0`, `cmaes_archive_cap`) over the two objectives (fA, fB) of `objective: joint`, both
  normalised by ideal / nadir; the engine works in the box `box` (half width, raw coordinate units) around the start's searched coordinates, the population is the start plus Gaussian perturbations.
  Output: Pareto archive in the checkpoint, `metrics.csv`, `front.csv` (train and val values of every archive member), `best/` = the representative (minimum scalarisation).
* Life cycle (both): `run_search_loop` (interrupt handling, writer drain, exit checkpoint), the background writer (`search_state@<gen>.jld2` and `best_params@<gen>.{json,jld2}` at a new best and every
  `ckpt_every` generations; the JSON/JLD2 hold the SEARCHED coordinates), `search_state_latest.jld2`, `losses.duckdb` (`total_loss` row per new best), `resume_from: <search_state*.jld2>` (the budget
  `trials` may be raised). `best/` is a plugin-ready parameter directory (`params_species.csv params_global.csv params_recruitment.csv meta.txt`, loadable as `hbad2_params_dir`, plus the
  prototype-format `theta.csv`, `recr.csv`), rewritten at every new best.

`trials: 1` evaluates the start only (START / FIT lines, `best/` = the start, exit checkpoint): the way to score a parameter vector (`start_theta`, `start_recr`) on train / val / test.

Log lines: `START`/`FIT` lines with `fA` (total, mort, D, H, sigma_D, sigma_H) and `fB` terms for train / val / test; `CMA gen ... evals ... fbest ... sigma ... cond` every 25 generations (the
prototype's format).

## 5. YAML keys

Top-level keys reused: `succession_plugin: hbad2`, `search_mode` (`cmaes` | `igelmo`), `cohorts_db_path` (the tree database), `output_dir`, `trials`, `cmaes_lambda`, `cmaes_sigma0`, `resume_from`,
`val_frac` / `test_frac` / `split_seed` (only without `split_csv`), `igel_mu`, `igel_sigma0`, `cmaes_archive_cap`. Everything else is under `hbad2_calib:` (own strict schema `HBAD2_CALIB_LIST`; unknown
keys are errors with suggestions; `Pan.config_docs()` lists them):

| key | default | meaning |
|---|---|---|
| `states` | `[12, 13]` | FIPS states of the plots |
| `trees_table` | `curated_trees` | tree table of `cohorts_db_path` |
| `split_csv` | `""` | split file; empty = drawn from `val_frac`/`test_frac`/`split_seed` |
| `intervals_cache`, `rebuild_intervals` | `<output_dir>/hbad2_intervals`, false | Arrow cache of the interval product |
| `longevity_csv` | `/workspace/species_longevity_ref.csv` | LONGEVITY table (`species_symbol`, `longevity_yr`) |
| `recruitment_csv` | `results/hbad2/params_recruitment.csv` | recruitment data fit (`key rho0 rhoS rhoL`) |
| `priors_csv` | `results/hbad/priors.csv` | HBAD v1 priors for the data-based start (`key`, `sdimax_p95_all`) |
| `form` | 2 | growth form (1 Chapman-Richards, 2 power-exponential) |
| `size_split` | `[5, 12.7, 25, 40]` | initial diameter-class breaks (cm) |
| `site` | `subplot` | `subplot` (4 sites per plot, x4) or `plot` |
| `vigour`, `sigv_max` | true, 0.7 | vigour sub-cohorts and the bound of sigma_v |
| `birth_w`, `min_sp_trees` | 10, 3000 | birth-class width; train trees for an own species key |
| `sizecls` | false | add the survivors' size-class term to fA |
| `bounds` | `{}` | species-parameter box overrides `{Hmax: [lo, hi]}` |
| `objective` | `sim_a` | `sim_a` or `joint` |
| `fb`, `simb_sigma` | `margins`, 0.16 | Sim B objective mode; observation spread |
| `lambda`, `rho`, `ideal`, `nadir` | `[0.5]`, 0.05, `[]`, `[]` | scalarisation weights (one search each), augmentation, fixed reference points `[fA, fB]` |
| `recruitment` | `fixed` | `free`, `fixed`, `kappa` |
| `freeze` | `[]` | coordinates kept at the start |
| `start_theta`, `start_recr` | `""` | start vectors (csv column `theta`); `start_recr` only with `recruitment: free` |
| `cma_seed`, `cma_stall`, `cma_ftol` | 1, 150, 1e-9 | CMA-ES RNG seed and stagnation stop |
| `ckpt_every` | 10 | checkpoint every n generations (and at every new best) |
| `box` | 0.3 | igelmo: half width of the u-space box |
| `test_set` | false | also evaluate TEST at the end (information only) |

## 6. The prototype setups as configs

* **j15 / model v2j** = `configs/region4/hbad2_calib_joint.yml`: `objective: joint`, `fb: margins`, `lambda: [0.75]`, `recruitment: kappa`, `freeze: [mbg, mD, msup, G:mD2, G:mSD, G:pM]`, `sigv_max: 0.7`,
  `cmaes_lambda: 24`, `cmaes_sigma0: 0.03`, ideal `[-272975.3, 14619.3]`, nadir `[-236466.5, 24102.3]` (the values printed in the prototype's log; the prototype read them from `HBAD2_IDEAL` /
  `HBAD2_NADIR`), start = the prototype's `fit_v2fv/theta.csv` (copied to `results/hbad2/calib/start_v2fv_theta.csv`).
* **v2fv5** = `configs/region4/hbad2_calib_simA.yml`: Sim A only, `sigv_max: 0.5`, `cmaes_lambda: 32`.

## 7. Parity evidence (`tools/hbad2_calib_parity.jl`)

The prototype harness (`tools/hbad2/common.jl` and `simB_lik.jl`, branch `hbad-v2` at /workspace/landis_hbad2, read-only) runs in the same process as Pan, on its own Arrow files; Pan
runs on the interval product rebuilt from the database. Parameter vectors: `fit_v2j` (joint fit j15: theta.csv + recr.csv, `HBAD2_VIGOUR=1 HBAD2_SIGV_MAX=0.7 HBAD2_SITE=subplot HBAD2_FORM=2
HBAD2_SIZE_SPLIT=5,12.7,25,40`) and `fit_v2fv5` (sigma_v <= 0.5, Poisson recruitment). 81 terms compared (the tables below plus the size-class term at sigma_w 0.110 and 0.083, a threaded-vs-serial
check and the data-based start vector); **max relative difference 1.3e-14** (tolerance 1e-9); site and cohort counts are equal (train: 17,984 Sim A sites / 245,040 sub-cohorts, 2,128 Sim B sites in
532 plots; val: 19,184 / 245,157, 1,860 in 465 plots); no Sim B recruit class was dropped. Pan agrees with the numbers logged by the prototype runs (v2fv5 `log.txt`: train total -273425.4,
val -272655.4; `joint_j15/log.txt`: train fA -270648.8 fB 16462.8, VAL fA -269744.7 fB 18317.9).

**v2j (j15)**

| term | set | Pan | prototype | rel. diff |
|---|---|---|---|---|
| simA total | train | -270648.7574456 | -270648.7574456 | 3.01e-15 |
| simA mort | train | 32053.69283369 | 32053.69283369 | 9.42e-15 |
| simA D | train | -162512.2714303 | -162512.2714303 | 1.79e-16 |
| simA H | train | -140190.178849 | -140190.178849 | 3.94e-15 |
| simA sigD | train | 0.1153908016 | 0.1153908016 | 7.22e-16 |
| simA sigH | train | 0.1552343879 | 0.1552343879 | 7.33e-15 |
| simA ncoh | train | 75257 | 75257 | 0.00e+00 |
| simA ntrees | train | 128185 | 128185 | 0.00e+00 |
| simA size_nll sigma_w=0.110 | train | 30530.00467159 | 30530.00467159 | 1.05e-14 |
| simA size_nll sigma_w=0.083 | train | 29154.49459155 | 29154.49459155 | 6.49e-15 |
| simB total | train | 39651.02589985 | 39651.02589985 | 3.67e-15 |
| simB counts | train | 49530.59396707 | 49530.59396707 | 2.94e-15 |
| simB H | train | -9879.568067215 | -9879.568067215 | 1.84e-16 |
| simB size | train | 8192.224503805 | 8192.224503805 | 2.22e-16 |
| simB species | train | 18150.14698675 | 18150.14698675 | 6.01e-16 |
| simB plotstand | train | 25284.11367488 | 25284.11367488 | 0.00e+00 |
| simB plotstand_trees | train | 18456.38607354 | 18456.38607354 | 0.00e+00 |
| simB margins (fB) | train | 16462.80342334 | 16462.80342334 | 6.63e-16 |
| simB dropped | train | 0 | 0 | 0.00e+00 |
| simA total | val | -269744.6567662 | -269744.6567662 | 1.29e-15 |
| simA mort | val | 33015.3084463 | 33015.3084463 | 1.32e-15 |
| simA D | val | -161625.4110022 | -161625.4110022 | 5.40e-16 |
| simA H | val | -141134.5542103 | -141134.5542103 | 1.86e-15 |
| simA sigD | val | 0.1182953844 | 0.1182953844 | 1.29e-15 |
| simA sigH | val | 0.1550589607 | 0.1550589607 | 3.40e-15 |
| simA ncoh | val | 75718 | 75718 | 0.00e+00 |
| simA ntrees | val | 139374 | 139374 | 0.00e+00 |
| simA size_nll sigma_w=0.110 | val | 34915.54807072 | 34915.54807072 | 1.88e-15 |
| simA size_nll sigma_w=0.083 | val | 33436.46916242 | 33436.46916242 | 3.05e-15 |
| simB total | val | 41603.60604897 | 41603.60604897 | 3.50e-16 |
| simB counts | val | 51006.75428252 | 51006.75428252 | 2.85e-16 |
| simB H | val | -9403.148233552 | -9403.148233552 | 0.00e+00 |
| simB size | val | 9148.647210443 | 9148.647210443 | 1.19e-15 |
| simB species | val | 18572.40511172 | 18572.40511172 | 1.96e-16 |
| simB plotstand | val | 25144.53448297 | 25144.53448297 | 5.79e-16 |
| simB plotstand_trees | val | 17976.6339716 | 17976.6339716 | 1.01e-15 |
| simB margins (fB) | val | 18317.90408861 | 18317.90408861 | 3.97e-16 |
| simB dropped | val | 0 | 0 | 0.00e+00 |

**v2fv5**

| term | set | Pan | prototype | rel. diff |
|---|---|---|---|---|
| simA total | train | -273425.4398662 | -273425.4398662 | 0.00e+00 |
| simA mort | train | 31941.94692637 | 31941.94692637 | 1.32e-14 |
| simA D | train | -164983.2135576 | -164983.2135576 | 1.59e-15 |
| simA H | train | -140384.1732349 | -140384.1732349 | 8.29e-16 |
| simA sigD | train | 0.1116636532 | 0.1116636532 | 3.48e-15 |
| simA sigH | train | 0.1548347464 | 0.1548347464 | 1.43e-15 |
| simA ncoh | train | 75257 | 75257 | 0.00e+00 |
| simA ntrees | train | 128185 | 128185 | 0.00e+00 |
| simA size_nll sigma_w=0.110 | train | 29726.19061672 | 29726.19061672 | 2.45e-15 |
| simA size_nll sigma_w=0.083 | train | 29052.43277692 | 29052.43277692 | 8.52e-15 |
| simB total | train | 50183.67250852 | 50183.67250852 | 1.30e-15 |
| simB counts | train | 59932.79944714 | 59932.79944714 | 1.09e-15 |
| simB H | train | -9749.126938615 | -9749.126938615 | 0.00e+00 |
| simB size | train | 9163.289245732 | 9163.289245732 | 7.94e-16 |
| simB species | train | 23471.42579256 | 23471.42579256 | 3.10e-16 |
| simB plotstand | train | 30992.28818624 | 30992.28818624 | 0.00e+00 |
| simB plotstand_trees | train | 23218.5363896 | 23218.5363896 | 3.13e-16 |
| simB margins (fB) | train | 22885.58809968 | 22885.58809968 | 6.36e-16 |
| simB dropped | train | 0 | 0 | 0.00e+00 |
| simA total | val | -272655.3795513 | -272655.3795513 | 1.28e-15 |
| simA mort | val | 32960.92917823 | 32960.92917823 | 1.04e-14 |
| simA D | val | -163997.2750087 | -163997.2750087 | 3.90e-15 |
| simA H | val | -141619.0337209 | -141619.0337209 | 4.32e-15 |
| simA sigD | val | 0.1146472227 | 0.1146472227 | 8.72e-15 |
| simA sigH | val | 0.1540699876 | 0.1540699876 | 8.29e-15 |
| simA ncoh | val | 75718 | 75718 | 0.00e+00 |
| simA ntrees | val | 139374 | 139374 | 0.00e+00 |
| simA size_nll sigma_w=0.110 | val | 33452.3252807 | 33452.3252807 | 2.18e-16 |
| simA size_nll sigma_w=0.083 | val | 32751.99457331 | 32751.99457331 | 3.44e-15 |
| simB total | val | 49909.16231983 | 49909.16231983 | 5.54e-15 |
| simB counts | val | 59111.44254736 | 59111.44254736 | 4.68e-15 |
| simB H | val | -9202.280227528 | -9202.280227528 | 0.00e+00 |
| simB size | val | 9746.988457536 | 9746.988457536 | 5.60e-16 |
| simB species | val | 22177.03486059 | 22177.03486059 | 1.64e-16 |
| simB plotstand | val | 28822.62027695 | 28822.62027695 | 1.26e-16 |
| simB plotstand_trees | val | 20907.33819365 | 20907.33819365 | 3.48e-16 |
| simB margins (fB) | val | 22721.74309059 | 22721.74309059 | 1.60e-16 |
| simB dropped | val | 0 | 0 | 0.00e+00 |

Not part of the table (printed by the tool): the interval product comparison (identical), threaded vs serial site loops (bit-identical for Sim A total and Sim B margins, train and val), and the
data-based start vector against the prototype's `init_theta` (max relative difference 6.4e-15 over 175 coordinates).

## 8. Further verification

* `test/unit_hbad2_calib.jl` (in `runtests.jl`): interval product on a synthetic tree table (outcomes alive / dead / harvest / missing, ingrowth, flags, cache key), the kernel entry points against the
  vendored prototype kernel (`run_interval!` with fractional steps and `site_year_free!` with a capacity that forces dropped classes: states and the dropped count identical), theta layout / decode /
  freeze, the config schema, `PlainCMA` (resume == uninterrupted; bit-identical to the prototype's `cmaes`).
* `test/hbad2_calib_e2e.jl` (subprocess in `runtests.jl`; every 15th plot of the prototype split, 599 plots): Sim A (200 evaluations) and joint (kappa + frozen mortality, 200 evaluations) runs improve on the start,
  write `search_state@N.jld2`, `best_params@N`, the plugin-ready `best/` directory (loaded back through `HBAD2Context(params_dir=)` it equals the decoded parameters) and `losses.duckdb`; a run of 100
  evaluations resumed to 200 ends in exactly the state of the uninterrupted 200-evaluation run (mean, covariance, step size, RNG, best, history); `trials: 1` evaluates the start only; a lambda list
  gives one sub-run per weight; igelmo (mu 4) 3 + 2 generations == 5 generations; threaded and serial site loops give identical objective values; the kernel entry point allocates 0 bytes.
* Speed (train sets, 2 threads, measured back to back on the shared machine): Sim A evaluation 164 ms (prototype 346 ms), Sim B evaluation 1022 ms (prototype 1282 ms); the annual kernel entry points do not allocate
  (the per-evaluation allocations are the setup arrays of the likelihood sums).
* Search trajectory vs the prototype logs: the scalar objective of the j15 setup (frozen mortality, kappa, ideal / nadir, lambda 0.75) equals the prototype's `fit_joint.jl` formulas (typed verbatim) to
  6e-15 at five random points around the start, and `PlainCMA` equals the prototype's `cmaes` bit for bit on a 136-dimensional toy problem; the START line of a j15 run reproduces the prototype's
  (`fA -273387.6 fB 23153.1`, val `-272544.7 / 23001.3`). The generation lines do NOT reproduce exactly: Pan, config as shipped, gen 25 `fbest 0.0811 sigma 0.0295 cond 1.2`, gen 50
  `0.0668 / 0.0397 / 1.7` against the prototype log `0.0797 / 0.0287 / 1.2` and `0.0657 / 0.0376 / 1.6` (the Pan trajectory is insensitive to the rounding of the printed ideal / nadir:
  shifting the ideal by 0.04 changes nothing at the printed precision). The launch environment of the historical run (`HBAD2_SIGMA`, Julia / BLAS version) is not recorded, so the cause is open; the
  objective and the engine are each verified equal.

## 9. Files

`src/data/hbad2_intervals.jl` (interval product), `src/hbad2_calib/{spec,data,objective,problem,export}.jl` (module `Pan.HBAD2Calib`), `src/search/PlainCMA.jl`, `src/drivers/hbad2_calib.jl`,
`src/config/schema.jl` (`hbad2_calib` schema), plugin entry points in `src/plugins/HBAD2/{kernel,plugin}.jl`, `make_sites_hbad2_arrays(...; extra_capacity)` in `src/sites.jl`.
Tools: `tools/hbad2_intervals_check.jl`, `tools/hbad2_calib_parity.jl`. Tests: `test/unit_hbad2_calib.jl` (in-process), `test/hbad2_calib_e2e.jl` (subprocess, real-data subset).
