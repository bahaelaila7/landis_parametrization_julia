# HBAD succession plugin (Height / Basal Area / Density)

`succession_plugin: hbad` (alias `hbad-succession`). A third succession model next to BiomassSuccession and PnET, behind the same
interface (`src/succession_api.jl`). Cohorts carry (species, age, trees/ha, quadratic mean diameter, mean height) and evolve by
**fixed empirical priors** fitted on GA+FL FIA (`results/hbad/`, see `results/hbad/README.md`) times a few **fitted multipliers**.
Reported cohort biomass is always the allometric AGB of the state, so the loss, the override injection and the TOST code work unchanged.

Code: `src/plugins/HBAD/{HBADPlugin,priors,params,plugin}.jl`, `src/parametrize/HBAD.jl`, wiring in `src/succession_api.jl`, `src/Pan.jl`,
`src/Plugins.jl`, `src/Parametrization.jl`. Tests: `test/hbad_unit.jl`. Smoke config: `configs/region4/hbad_gafl_smoke.yml`.

## 1. State

Per cohort (CSR `:cohort` arrays, all moved together by `PanCore.swap_cohort!`):

| field | type | unit | meaning |
|---|---|---|---|
| `c_species` | UInt32 | - | eco-LOCAL species index (BiomassSuccession convention) |
| `c_age` | Float32 | yr | cohort age |
| `c_bio` | Float32 | g/m2 | REPORTED aboveground biomass = `agb_of(tph, dq)`; exactly the observed value right after an injection |
| `c_tph` | Float32 | trees/ha | density, plot-scale accounting of the FIA cohort table (TPA_UNADJ x 2.47105) |
| `c_dq` | Float32 | cm | quadratic mean diameter |
| `c_h` | Float32 | m | mean tree height |
| `c_bal` | Float32 | m2/ha | BA of larger-dq cohorts at the start of the last step (scratch + diagnostic) |

Derived: `BA = pi/4 (dq/100)^2 tph` (m2/ha). Per site (scalars): `B = sum c_bio` (g/m2), `BA` (m2/ha), `SDI = sum tph (dq/25.4)^1.605`
(Reineke, metric, trees/ha at 25.4 cm), `RD = SDI / SDImax(site)`, `thin_tph` (trees/ha removed by self-thinning in the last step).
`sp_mature` (species CSR) is kept only because the shared loss code reads its length as the eco species count.

## 2. Fixed priors (not fitted)

`HBADPlugin.HBADContext(species_list, eco_species_ids; priors_csv, coefs_csv, ...)` (an immutable per-run context, since Step 3d; it replaced the module Refs and `configure!`) resolves, per GLOBAL species, one `SpPrior` (isbits, 28 Float64) from
`results/hbad/priors.csv` and `results/hbad/translation_coefs.csv` (+ `translation_selection.csv`, `species_group.csv`). Resolution (same rule as
`tools/hbad_translate_lib.jl::hb_key`): exact species symbol if it has a row; else the pooled group row `_S` / `_H`; else `_ALL`. Pan catch-all names:
`_S*` -> `_S`; `_H*` -> `_H`; `_GRP_<n>_*` -> `_S` if n <= 24 else `_H` (FIA SPGRPCD); other `_*` -> `_ALL`; plain symbols use `species_group.csv`
(unknown symbols -> hardwood, as the lib). The mapping is logged at configure time (`@info`; per species at `JULIA_DEBUG=Pan`).

Blocks taken from the species' own row unless the **sanity guard** replaces them by the group row (deviation from "exact symbol if present", documented):
* growth block (g_b0..g_k_bal) if missing, or if the increment at D = 5/10/30/60 cm, age 40, BAL 0 is not in (0.005, 3] cm/yr (SAPA: 4369 cm/yr at 5 cm -> `_H` growth);
* mortality block (m_c0..m_c5) if missing (CAAL27, TADI2 have none) or if the hazard on a grid of D/RD/BAL leaves (1e-4, 0.3] per year (SAPA -> `_H`).

Equations (D = dq in cm, age in yr, BAL in m2/ha, RD relative density):

```
dD   = G_MULT * exp(g_b0 + g_b1 ln(D/20) + g_b2 D/20 + g_b3 ln(age/30) - g_k_bal BAL/10)      [cm/yr], clamped to [0, 3]
haz  = M_MULT * exp(m_c0 + m_c1 ln(D/20) + m_c2 D/20 + m_c3 RDc + m_c4 BAL/10 + m_c5 RDc ln(D/20))   [1/yr], clamped <= 0.5, RDc = min(RD, 1.5)
CR(D) = hd_a (1 - exp(-hd_b D))^hd_c                                                            Chapman-Richards, H = 1.3 + CR(D)
```
Domain guard: D below 2.54 cm (FIA tallies trees >= 1 in) is evaluated at 2.54 cm in the two regressions (`dmin_cov`); the state itself keeps the real dq.

## 3. Annual step (`site_year!`, one call per active site, sites independent -> `@maybe_threads`)

1. Snapshot from the current state: site BA, SDI, SDImax (BA-weighted mean of the species values), RD, and per cohort BAL = sum BA of cohorts with strictly larger dq (O(n^2), n <~ 60, stored in `c_bal`).
2. Longevity: cohorts with `age >= LONGEVITY[sp]` are swap-removed; then `age += 1`.
3. Diameter growth `dD` (above) with pre-growth D, new age, BAL; `D' = D + dD`.
4. Height, ratio form so that each cohort keeps its own anomaly: `h' = 1.3 + (h - 1.3) CR(D')/CR(D)`; if `CR(D) < 0.1`: `h' = 1.3 + CR(D')`.
5. Mortality: `tph *= exp(-haz)` (expected value, deterministic) with pre-growth D, step-start BAL and RD.
6. Self-thinning: if SDI > site SDImax, trees are removed from the SMALLEST-dq cohort upward (reducing `tph` at constant dq) until SDI = SDImax. Because the BA-weighted SDImax shifts when small cohorts leave, the pass is repeated (<= 8 times, 2-3 in practice) so the final RD <= 1 + 1e-3.
7. Cohorts with `tph < 1` tree/ha are swap-removed.
8. `c_bio = tph/2.47105 * m_lb(dq) * 453.592/4046.86` with `ln m_lb = mq_a + mq_b ln(dq/2.54)` (cohort-level mass-QMD relation of the translation); site `B, BA, SDI, RD, live` updated.

Natural establishment (step g2) and planting: see section 9. With `no_establishment: true` (Sim A) steps (a)-(h) are exactly the kernel above and no random number is drawn. `spinup: true` still raises a clear error; `dual_mode` (Sim B) is supported (section 9).

Performance/design: sites are independent; per-site work touches only that site's CSR slice; no scratch buffers, no dynamic dispatch (eco parameters are a concrete struct of isbits `SpPrior` + Float64 vectors), no allocation (verified, section 8).

## 4. Initialisation / injection

`add_cohort!(ctx, site, esp, age, agb)` (priors and eco table from the `HBADContext` `ctx`) -> `translate(prior, age, agb)`, the selected translation of `tools/hbad_translate_lib.jl` (structure A, TPA-first, `mq`, Chapman-Richards height; no SI, natural):

```
x1 = ln(agb) - 6 ; x2 = ln(age) - 3.3
tph = max(exp(t0 + t1 x1 + t2 x2 + t3 x1^2 + t4 x1 x2 + t5 x2^2), 14.87)         # one macroplot tree = 6.01805*2.47105 trees/ha
m_lb = agb / (tph/2.47105 * 453.592/4046.86)
dq  = 2.54 * exp((ln m_lb - mq_a)/mq_b)  (cm) ;   h = 1.3 + hc_a (1 - exp(-hc_b dq))^hc_c
```
and `c_bio = agb` exactly. The translation is AGB-consistent by construction (round trip of allometric AGB: <= 2e-7 relative, Float32 storage).
Deviation at tiny AGB: the lib clamps `dq` to [2.54, 170] (then AGB is no longer reproduced); the plugin limits `dq` to [0.25, 170] and, when a limit binds, re-solves `tph` so that AGB stays exact (identical to the lib wherever the lib does not clamp, i.e. AGB >~ 0.5-3 g/m2 depending on species).

`set_cohort_bio!(site, k, newbio)` (override injection, disturbance drops, init perturbation): `c_tph *= newbio/oldbio` (trees removed/added at the same size; dq, h unchanged), `c_bio = newbio`; a cohort with no previous biomass is re-translated. All writers in `Pan.jl` already go through `set_cohort_bio!` (disturbance drop included), so nothing changed for BiomassSuccession.

## 5. Fitted parameters (`HBADParams`, `src/parametrize/HBAD.jl`)

| name | scope (sampler / applier) | prior / bounds | start | CMA-ES block group |
|---|---|---|---|---|
| `G_MULT` | EcoSpecies (`NestedIndexApplier`) | LogUniform[0.3, 3.0] (log-symmetric; default 1.0 at u ~ 0.52); sigma 0.1 | 1.0 | `:growth_hbad` (per eco / L3 block) |
| `M_MULT` | EcoSpecies | LogUniform[0.2, 5.0] (median exactly 1); sigma 0.2 | 1.0 | `:growth_hbad` |
| `SDIMAX` | Species (`IndexApplier`) | Uniform[200, 1500] global window, **clamped per species at decode to [0.7 p90, 1.3 p99]** of `sdimax_*_all` (`HBADPlugin.sdimax_range`); sigma 40 | p95 | `:hbad_species` (global block) |
| `LONGEVITY` | fixed | as BiomassSuccession: `BSTables.longevity` (yaml `longevity_from_data`, `longevity_overrides`) / `fix_longevity` / 400 | - | - |

Notes: the per-species data range is applied by a clamp rather than a rescaled prior (simplest mechanism; costs a flat region in u-space at the edges, and the
global window is ~2x wider than most species ranges). `HBAD_PER_ECO_GROUPS = {:growth_hbad}`. `param_pooling` works by parameter name (`G_MULT`, `M_MULT`), as for FolN/B_MAX.

`generate_hbad_params(species_list, eco_list, eco_species_ids; rng, mode)`: `mode = :default` (G = M = 1, SDIMAX = p95), `:jitter` (lognormal sd 0.25 around the default, SDIMAX sd 0.10, clipped to the bounds; **the default** because the Igel/CMA-ES drivers draw several initial candidates from this call) or `:prior` (draws from the search priors). yaml `hbad_init: default|jitter|prior` (`HBADPlugin.INIT_MODE`).

## 6. Wiring

yaml: `succession_plugin: hbad`, optional `hbad_priors_csv`, `hbad_coefs_csv` (default `results/hbad/priors.csv`, `results/hbad/translation_coefs.csv` relative to the repo), `hbad_init`. `parametrize` calls `HBADPlugin.configure!` right after the species/eco tables exist and before any `make_sites`. Checkpoints: `HBADParams` is a plain struct of vectors, so `best_params@N.json` (JSON3) and `.jld2`/`search_state@N.jld2` work like PnETParams (verified, section 8). `succession_plugin_for(::HBADParams)` re-resolves the priors for the params' species list if needed, so post-hoc tools (`make_sites_like`) work after a JLD2 load.

## 7. Design decisions and limitations

* Expected-value mortality: the cohort density is continuous (`tph *= exp(-haz)`); no stochastic death. A cohort disappears only through `tph < 1`, longevity, thinning, or injection removal.
* Plot-scale TPA accounting (14.87 trees/ha = one FIA macroplot tree): the density of a small cohort is a design artefact of the cohort table (discrete values), so site `BA`/`SDI`/`RD` are indicative stand statistics, not measurements. The `SDImax` priors were derived in the same accounting.
* Height is not in the loss; it only feeds the Chapman-Richards ratio (diagnostic).
* BAL/RD use only the cohorts present in the simulated site (the site unit of the cohort table), whereas the growth / mortality priors were fitted with plot-level BAL and RD of all live trees.
* Priors come from GA+FL only; other regions use the same coefficients (refit via `tools/hbad_*.jl`).
* Diameter-growth and mortality regressions are single-interval FIA fits (val R2 0.05-0.29; AUC 0.61-0.72). The multipliers absorb a constant bias per (eco, species) only.
* No spinup. Establishment is optional (section 9); Sim A calibration runs with it off.
* Sanity guard and dmin_cov are deviations from a literal use of the CSV rows (section 2).

## 8. Test results

Unit tests (`source wt_env.sh && ./julia_gdal.sh --project=. --threads=2 test/hbad_unit.jl`): **379 checks passed, 0 failed** (also with `--threads=1`).

1. Translation parity vs `tools/hbad_translate_lib.jl` (600 random species incl. `_S_HST`, `_H_LST`, `_GRP_41_LST`, SAPA, CAAL27..., age 1-300 (integer and fractional), AGB 0.05-25000 g/m2):
   372 cases where the lib does not clamp QMD: max relative error tph 1.8e-15, dq 1.1e-15, h 8.7e-16 (requirement 1e-3). The 228 cases where the lib clamps QMD to
   [2.54, 170] cm (tiny AGB) deviate by design; the plugin stays AGB-consistent there (max AGB error 9.6e-16).
2. `add_cohort!` round trip over 337 random cohorts: `c_bio == Float32(agb)` exactly; allometric AGB of the stored (tph, dq) within 1.7e-7 relative (limit 0.5 %).
   `set_cohort_bio!` halving: dq and h unchanged, allometric AGB equals the new value.
3. Allocation: `Pan.step!` over 40 sites (serial site loop, after warm-up) allocates 0 bytes; `site_year!` 0 bytes. The threaded loop allocates ~0.9 kB (task spawn), independent of problem size.
4. Single PITA cohort, age 5, AGB 300 g/m2, G_MULT = M_MULT = 1, 30 yr (BAL = 0, open-grown):

   | age | tph | dq cm | h m | BA m2/ha | SDI | AGB g/m2 | mortality %/yr |
   |---|---|---|---|---|---|---|---|
   | 5 | 105.8 | 12.06 | 10.89 | 1.21 | 32.0 | 300 | - |
   | 10 | 101.6 | 17.07 | 14.45 | 2.32 | 53.7 | 750 | 0.79 |
   | 20 | 93.6 | 26.22 | 19.92 | 5.05 | 98.5 | 2257 | 0.85 |
   | 30 | 85.2 | 34.46 | 23.80 | 7.94 | 139.0 | 4362 | 1.03 |
   | 35 | 80.6 | 38.31 | 25.32 | 9.29 | 155.8 | 5526 | 1.16 |

   Height equals the PITA Chapman-Richards curve at the final dq (25.32 m); SDI never approaches SDImax p95 = 675 (max 155.8); mean mortality 0.90 %/yr (README: 1.24 %/yr incl. disturbance).
5. Mixed site (PITA, QUNI, ACRU, PIEL): BAL snapshot equals the reference O(n^2) computation (3.786 / 0 / 15.385 / 16.923 m2/ha for dq 31.9 / 34.6 / 25.2 / 14.4 cm).
   Overstocked variant (SDI 771 > site SDImax 590, growth and mortality switched off): smallest cohort removed entirely (348.5 -> 0), next smallest thinned partially (150.6 -> 127.3), larger cohorts untouched; final RD = 1.000.
6. Swap-removal: 14 cohorts, 3 killed by longevity (species with LONGEVITY 30 / 60); the 11 survivors keep tph, dq, h, bal snapshot, bio consistent per cohort vs a reference loop; a cohort driven to < 1 tree/ha is removed; CSR capacity shrinks accordingly.
7. Determinism: 300 sites x 25 yr, 4 runs (serial, serial, threaded, threaded) all give the same hash of every state array; identical for `--threads=1` and `--threads=2`
   (hash 648ac254cad061f0). After 25 yr the maximum site RD is 1.0000.
   Throughput (informational, loaded machine): 3000 sites x 25 yr = 0.28 s serial = 3.8 us per site-year.

Smoke parametrization (`configs/region4/hbad_gafl_smoke.yml`, derived from gafl_A2: GA+FL, 7 L3, Sim A tier 3, override injection sync + `exclude_overwrite`, `train_cap: 15`, Igel MO-CMA-ES mu 8, 24 trials; 2 threads, `PAN_PARALLEL=candidate`):
runs end-to-end, writes `best_params@N.json/.jld2`, `search_state@N.jld2`, `best_val_params@N.*`, `metrics.csv`; JLD2 reload gives `HBADParams` (41 species, SDIMAX in [671, 1070], G_MULT in [0.61, 1.47], M_MULT in [0.52, 1.69]).
Losses are finite: train 2.5625 -> 2.5593, val 4.418 -> 4.411. Per generation of 8 candidates (eval phase, train set 2014 plots, 41 species, 16 eco cells): HBAD 6.1 s / 7.6 s vs BiomassSuccession twin (`gafl_A.yml` with the same cap/budget) 5.9 s / 5.8 s
(noisy: the machine ran at load 6-8 on 8 cores; the eval phase is dominated by the shared loss code, not by the succession step: HBAD step ~4 us per site-year).
Whole run wall time 3m37s (HBAD) vs 3m46s (biomass), ~145 s of it data preparation + compilation.

Regression gate for the default biomass plugin (`tools/regress_gate.sh`, 4 threads): REGRESS_ARCHIVE.txt byte-identical to `docs/regress/baseline_853_split-productive-species.txt` (`cmp` silent).

## GA+FL inverse parameterization (Sim A, 2026-10-13)
Same data, split, train cap, groups, real-only TOST criterion and 3-fold CV procedure as the BiomassSuccession GA+FL
calibration (`docs/gafl_calibration_log.md` D1–D11), so the three models are directly comparable.
- Config `configs/region4/gafl_hbad.yml` (= gafl_A.yml with `succession_plugin: hbad`; G_MULT/M_MULT pooled like B_MAX:
  region level, per site cell for PITA/PIEL). Tools now drive HBAD through `Pan.setup_plugin_from_cfg` (configures the
  priors with the run's species lists).
- Start `tools/seed_hbad_start.jl`: G_MULT = M_MULT = 1 (empirical priors as fitted), SDIMAX = species P95, LONGEVITY from
  the BS set. **With no calibration at all** this start already passes 7/16/33 of 46 groups at ±5/10/20 % on train
  (median |bias| 8.3 %) — BS defaults / PnET PFT defaults pass 0–2 at ±10 %.
- Levers (`tools/calib_coord.jl`): `gmult` (grid 0.4–2.3) and `mmult` (0.3–3.5), per TOST group; schedule
  `gmult,gmult~,mmult~,gmult~,mmult~,gmult~` (`tools/gafl_cv.sh`). SDIMAX is not a calibration lever (kept at P95).

| held-out | ±5 % | ±10 % | ±20 % | median abs bias |
|---|---|---|---|---|
| val (fit train+test) | 16/47 | 37/47 | 41/47 | 3.1 % |
| train (fit val+test) | 17/46 | 33/46 | 42/46 | 2.6 % |
| test (fit train+val) | 17/46 | 31/46 | 41/46 | 2.7 % |
| mean | ≈ 17 (36 %) | ≈ 34 (72 %) | ≈ 41 (89 %) | 2.8 % |
| BiomassSuccession v1, same protocol | ≈ 8 | ≈ 20 | ≈ 36 | 5.1 % |
| PnET, same protocol | ≈ 3 | ≈ 16 | ≈ 33 | 6.1 % |

Robust (held-out pass in all 3 folds): ±5 % 9 groups (LIST2, NYBI, PITA·B, QUAL, QULA3, QUNI, QUPR2, _H_HST, _H_LST);
±10 % 28; ±20 % 40 (BS v1: 14 at ±10 %, 29 at ±20 %).
Deployable set (all GA+FL plots): `results/hbad_gafl/params_gafl_hbad.jld2` — in-sample 42/47 ±5 %, 44/47 ±10 %, 47/47 ±20 %.
G_MULT median 1.17 (0.70–2.26), M_MULT median 1.00 (0.85–1.16); nothing at a bound.

Sanity check of the gain (held-out test, fold-3 parameters, same pairs as BS): pairs 36,175 (BS 37,386 — HBAD drops the
~3 % of cohorts it thins below 1 tree/ha; Sim A pairs are survivors-only for every model), per-pair log-error SD 0.51
(BS 0.78), log-log correlation 0.934 (BS 0.877), Σsim/Σobs 0.99 (BS 1.12); 101 pairs equal obs to 0.1 % (BS 110) ⇒ no leak.
Why it fits better: growth is a function of tree size and competition (BAL) taken from 600 k remeasured trees, and the
cohort's size comes from its own (age, AGB) via the translation — BiomassSuccession's ANPP curve sees only cohort biomass.
Residual size structure (test): obs < 100 g/m² +61 % (5 % of pairs), 100–1,000 +3 %, 1–5 kg −5 %, > 5 kg −8 %
(BS: +14/−2/+4/−2 %) — tiny cohorts grow too fast (translation clamps them to one macroplot tree with a small QMD;
the diameter-growth prior is weakest there). Candidate next step: a size-targeted lever (e.g. per-group small-tree growth
exponent) or a sapling-aware translation (README §4).
Files: `results/hbad_gafl/` (cvH_* held-out group tables, final in-sample table, rounds, cv_summary.log).

## 9. Establishment and planting (branch `hbad-estab`)

Natural recruitment and a planting API, so that HBAD can run the free Sim B (`dual_mode: b`, `no_establishment: false`) and scenario plant actions.
Sim A (`no_establishment: true`, every site carries `no_establish = true`) is untouched: no draw is consumed, the annual kernel and all outputs are byte-identical
to section 3 (verified, section 9.6).

### 9.1 Annual step with establishment (two phases)

`process_plugin!` = phase 1 (`site_prepare!`, threaded over sites) -> CSR capacity adjustment (serial, only when some site's cohort count != capacity) -> phase 2 (`site_commit!`, threaded).
Phase 1 is steps (a)-(g) of section 3 plus **(g2) establishment** (below); phase 2 appends the new cohorts and does step (h) (reported AGB, site totals). This is the
same two-phase pattern as BiomassSuccession (`_new_cohort_counts` + `readjust_soa!`); the steady state (no cohort added or removed) allocates nothing.

(g2), after self-thinning and removal and before the AGB update:

```
RD    = SDI / SDImax(site)                       (post-thinning state, 0 on an empty site)
class = #{ k : RD > MIN_REL_RD[eco][k] }  in 0..5                       (5 ascending thresholds per ecoregion)
for every species j of the site's ecoregion (fixed order):
   candidate  <=>  ESTAB_SEED_RAIN   or   a cohort of j with age >= MATURITY[j] exists
   if candidate:  u ~ U(0,1) from the SITE rng;   establish  <=>  u < PROB_ESTAB_SPP[eco][j] * SUFFICIENT_LIGHT[class][SHADE_TOL[j]]
```
One draw per candidate species and year, whatever the probabilities, so the RNG stream (and hence the result) is independent of the thread count (per-site `Xoshiro`, as the other plugins).
A success creates a cohort **age 1, tph = ESTAB_TPH[j], dq = 1.0 cm (`HBADConst.estab_dq`), h = 1.3 + CR(dq)** (Chapman-Richards of the species' growth prior, `hd_*`),
`c_bio` = allometric AGB of (tph, dq). At most one new cohort per species and year; if an age-1 cohort of that species already exists (injected or planted) the trees are
merged into it (densities add, dq = quadratic mean, h = tph-weighted mean), so there are never two age-1 cohorts of a species. New cohorts take part in the next year's BAL/RD
snapshot and are the smallest-dq cohorts, i.e. the first to be thinned when the stand is at SDImax: on a dense site recruits are removed again the following year, so RD can
exceed 1 by up to ~0.7 % per recruit cohort between steps (<= 1.02 observed) and is back at 1 after the next thinning. Two scratch species arrays are used: `sp_mature`
(mature seed source present) and `sp_estab` (success needs a new slot); a per-site scalar `shade_class` (0..5) is stored for diagnostics.

### 9.2 Parameters (`HBADParams`, all with defaults so old constructions keep working)

| name | scope | default and provenance | searched |
|---|---|---|---|
| `PROB_ESTAB_SPP` | eco x species, ANNUAL probability | `prob_estab_all_species.csv` (per-cycle occurrence of seedlings at the next remeasurement given a seed source, per (species, L3); 264 rows, only L3 8.3.5 and 8.5.3) converted `p_annual = 1-(1-p_cycle)^(1/cycle_years)`, `cycle_years` = yaml `cycle_years` (8). Missing (species, L3): species median over L3, else the table median 0.60 -> 0.108/yr | yes (Sim B only): Uniform[0,1], sigma 0.05, CMA-ES group `:estab_hbad` (per eco), poolable by name |
| `SHADE_TOL` | species, 1..5 | `BiomassSuccessionPlugin.SHADE_TOL_TABLE` (yaml `shade_tol_from_data` / `shade_tol_csv`, CCLCD-derived), else 3 | fixed |
| `MATURITY` | species, years | `MATURITY_TABLE` (yaml `maturity_from_data` / `maturity_csv`), else 20 (mode of the table: 236 of 264) | fixed |
| `SUFFICIENT_LIGHT` | 6 shade classes x 5 tolerances (`Vector{Vector}`) | the BiomassSuccession matrix (class 0: 0.75 0.5 .. ; corner tolerance 5 x class 5 = **0.50**); unit test compares to `generate_biomass_params` | fixed |
| `MIN_REL_RD` | per eco, 5 thresholds | `[0.15, 0.30, 0.45, 0.60, 0.75]` (defaults, see 9.4) | fixed |
| `ESTAB_TPH` | species, trees/ha | 750 (see 9.4) | fixed |
| `ESTAB_SEED_RAIN` | Bool | false (yaml `hbad_seed_rain`) | - |

yaml: `hbad_seed_rain`, `hbad_prob_estab_csv` (path of the (species, L3) table; empty = constant fallback), `cycle_years`. `HBADEcoParams` carries the resolved eco-local copies (concrete isbits/Float64 vectors; light matrix flattened), so the kernel does no dynamic dispatch.
JSON/JLD2 round trip works as for the other fields (plain struct of vectors); a JLD2 file written before this change (no establishment fields) loads and is upgraded with the defaults (`HBADPlugin.upgrade_params`, applied in `_load_params_from_path`).

### 9.3 Planting API

* `HBADPlugin.plant!(site, species_local, tph; age=1, dq=nothing, prior=nothing)` appends a cohort (or merges into an existing cohort of the same species and age) with the same state initialisation as establishment (`dq` default 1.0 cm; `age`/`dq` selectable); `c_bio` = allometric AGB; `site.B/BA/SDI/live/old` are updated incrementally (`RD` is refreshed by the next annual step). Returns the cohort index.
* **CSR capacity is the caller's job** (same pattern as `add_cohort!` in `_inject_observed_cohorts!`): set `site._new_cohort_counts = site.live + plant_slot_needed(site, sp, age)` (summed per site) for the planted sites, call `PanCore.readjust_soa!(soa, (cohort=soa.scalar._new_cohort_counts,))` once, re-fetch the `SiteView`s, then plant. `plant!` errors with a clear message when no slot is free. `Pan.plant_sites!(soa, [(site_index, species_local, tph), ...]; age, dq, eco_params)` does exactly this for any succession plugin.
* Generic hooks in `succession_api.jl`: `plant!(P, site, species_local, tph; kw...)` and `plant_slot_needed(P, site, sp, age)`. HBAD -> `HBADPlugin.plant!`. BiomassSuccession -> a new cohort of that species with the LANDIS initial biomass `calculate_initial_biomass(ANPP_MAX_SPP, site.B, B_MAX_ECO)` (`tph` is ignored, requires `eco_params=` of the site's ecoregion). PnET -> error "not supported yet".

### 9.4 Defaults: what is data and what is a choice

* **PROB_ESTAB**: data (per-cycle table) with the per-cycle -> annual conversion above. Applying the per-cycle values every year floods the plots (the BiomassSuccession Sim B finding); with the conversion a stand with a seed source has an expected occurrence `1-(1-p_annual)^8 = p_cycle` per 8-yr cycle (before the light factor, and ignoring that recruits can die before the remeasurement, so the table is a lower bound of the true annual rate).
* **ESTAB_TPH = 750 trees/ha**: a choice, bracketed by data. In `results/hbad/cohorts_gafl_head.csv` the 107 sapling cohorts (QMD < 12.7 cm) have mean density 293 tph, median 185 tph (one microplot sapling = 185.2 tph), 90th percentile 556 tph, and only 1.9 % reach >= 741 tph. Those are saplings that already passed 1 in; a newly established dq = 1 cm cohort loses density to HBAD mortality and thinning while it grows to the tally size, so 750 tph at birth was chosen as a few times the observed sapling density (not calibrated: the sapling-stratum density of the simulated stands was not compared with the data). The data give no per-species density: one value for all species.
* **MIN_REL_RD** = `[0.15, 0.30, 0.45, 0.60, 0.75]`: the spec defaults. Sanity check against data: relative density (SDI / BA-weighted species SDImax p95, plot-level, cohorts of the first 2000 rows of `cohorts_gafl_head.csv` = 88 plot-visits of one county, so indicative only) has quantiles p10/p30/p50/p70/p90/p95 = 0.17/0.43/0.66/0.82/0.94/1.03, mean 0.60: the thresholds span the observed range (about 10 % of plots are in class 0-1, 30 % above 0.75 = class 5). Quantile-matched thresholds (sixths of that sample) would be ~[0.26, 0.47, 0.66, 0.78, 0.86]; not used because the sample is one county. It is a fixed per-eco parameter and can be changed per ecoregion.
* **estab_dq = 1.0 cm**: below the plugin's other floor `HB_DQ_MIN` = 0.25 cm and below the FIA tally size; the growth/mortality regressions are evaluated at `dmin_cov` = 2.54 cm for smaller trees (section 2).

### 9.5 Sim B wiring (`dual_mode: b`)

* The `succession_plugin: hbad` / `dual_mode` error is removed. `_build_dual_b` previously built the free-run sites with the BiomassSuccession layout regardless of the plugin; it now uses `make_sites_like(ACTIVE_SUCCESSION[], ...)` (`ACTIVE_SUCCESSION` is set by `parametrize`, `setup_plugin_from_cfg` and `resim_simB_paired`), so BiomassSuccession stays byte-identical and HBAD/PnET get their own layout. `resim_simB_paired` sets the plugin from the params.
* `_seed_min_rel!` (yaml `fix_min_rel`) is a no-op for plugins without `MIN_REL_BIOMASS`; `_seed_maturity!` / `_seed_prob_estab!` work on the shared field names.
* `make_hbad_param_dists` adds `PROB_ESTAB_SPP` only when `!no_establishment || fit_establishment` (i.e. never in Sim A).
* Config: `configs/region4/gafl_hbad_B.yml` (smoke).

### 9.6 Tests and results

* `test/hbad_unit.jl`: **428 passed, 0 failed** (`--threads=2`; 425 + 3 before the params-upgrade checks, also run with `--threads=1`). The pre-existing 379 checks are unchanged and the old determinism hash is unchanged (`648ac254cad061f0`). New section (8): defaults (matrix == BiomassSuccession's, corner 0.50, annual conversion `1-(1-0.6)^(1/8)` = 0.1082),
  (a) 400 open sites with seed rain: all six species recruit (rates = PROB_ESTAB x light class 0: PITA tolerance 1 = 0.75 -> 298/400, the others 400/400), recruit state exact (age 1, 750 tph, dq 1.0, h = 1.3+CR(1), AGB allometric); without seed rain only the species with a mature cohort recruits,
  (b) 600 dense sites (RD 0.998-1.018 -> shade class 5): tolerance 1, 2, 3 -> 0 recruits, tolerance 4 -> 138/600 (0.25), tolerance 5 -> 308/600 and 287/600 (0.5), (c) a cohort aged into age 1 meets the same species' recruits -> exactly one age-1 cohort, tph 104.9 -> 852.8, dq 4.80 -> 2.19; no (species, age) duplicates in 30 years x 30 sites,
  (d) `Pan.step!` allocates **0 bytes** with establishment on (every species draws, nothing recruits) and `site_year!` 0 bytes; 300-site 30-yr run with recruitment gives the same state hash `8778b95a37671aac` serial, threaded, repeated, and with `--threads=1` and `--threads=2`,
  (e) `plant!` round trip incl. merge, explicit age/dq, capacity-error message, BiomassSuccession hook (initial biomass 498 g/m2 on an open site, 238 g/m2 at site B = 12000) and the PnET error, (f) the 100-yr trajectory (9.7), (g) JLD2 round trip and upgrade of `results/hbad_gafl/params_gafl_hbad.jld2`.
* Regression gate (`tools/regress_gate.sh`, 4 threads): `REGRESS_ARCHIVE.txt` byte-identical to `docs/regress/baseline_853_split-productive-species.txt`.
* HBAD Sim A with establishment off: `TOST_REAL_ONLY=1 tools/tost_select.jl configs/region4/gafl_hbad.yml results/hbad_gafl/params_gafl_hbad.jld2 1 train 0.05,0.1,0.2` before and after: **21/46, 39/46, 42/46** at +-5/10/20 %, median |bias| 1.8 % in both; the 46 per-group rows (n, pct_bias, CI, pass flags) are identical.
* Sim B smoke (`configs/region4/gafl_hbad_B.yml`, `dual_mode: b`, establishment on, 16 trials = initial population of 8 + 1 generation, 4 threads, train cap 15): runs end to end in 3 min; best train loss 16.287, val loss 11.362 (finite), checkpoints and `best_val_params@0.{json,jld2}` written; JSON and JLD2 reload give an equal `HBADParams`. Not a calibration.

### 9.7 100-year clear-cut trajectory (unit test (f))

One ecoregion (species PITA tol 1, PIEL 2, QUNI 3, ACRU 4, _S_HST 5, _H_LST 5), start = empty site, seed rain on (every species is a seed source), annual PROB_ESTAB = 0.108 (per-cycle 0.60), default HBAD priors (G_MULT = M_MULT = 1, SDImax = p95), mean of 200 replicate sites (independent site RNGs):

| yr | TPA /ha | BA m2/ha | QMD cm | BA share PITA/PIEL/QUNI/ACRU/_S_HST/_H_LST % | max RD | cohorts/site | mean shade class |
|---|---|---|---|---|---|---|---|
| 1 | 502 | 0.04 | 1.0 | 6.0/15.7/22.4/19.4/20.1/16.4 | 0.03 | 0.7 | 0.0 |
| 5 | 2104 | 1.28 | 2.8 | 28.7/30.4/7.4/5.3/22.4/5.9 | 0.34 | 3.0 | 0.1 |
| 10 | 3688 | 7.07 | 4.9 | 35.1/28.9/5.4/3.5/22.8/4.3 | 0.88 | 5.7 | 1.9 |
| 20 | 2663 | 24.27 | 10.8 | 40.4/28.5/3.5/1.7/23.3/2.6 | 1.007 | 5.3 | 4.9 |
| 30 | 941 | 32.05 | 20.8 | 44.9/29.6/1.4/0.3/23.1/0.7 | 1.007 | 2.2 | 5.0 |
| 40 | 549 | 36.06 | 28.9 | 41.5/32.8/0.6/0.1/25.0/0.1 | 1.014 | 1.6 | 5.0 |
| 50 | 418 | 38.98 | 34.5 | 40.5/32.9/0.1/0.0/26.5/0.0 | 1.013 | 1.2 | 5.0 |
| 60 | 350 | 41.16 | 38.7 | 40.6/32.9/0.0/0.0/26.5/0.0 | 1.013 | 1.1 | 5.0 |
| 70 | 315 | 43.00 | 41.7 | 40.7/32.8/0.0/0.0/26.5/0.0 | 1.012 | 1.1 | 5.0 |
| 80 | 282 | 44.59 | 44.9 | 40.8/32.7/0.0/0.0/26.5/0.0 | 1.014 | 1.1 | 5.0 |
| 90 | 467 | 45.55 | 35.3 | 40.6/32.6/0.0/0.0/26.7/0.0 | 1.011 | 1.9 | 5.0 |
| 100 | 780 | 44.76 | 27.0 | 38.9/33.0/0.0/0.0/27.9/0.2 | 1.006 | 2.9 | 5.0 |

Reading: dense natural regeneration (2000-3700 tph at 5-10 yr), canopy closure at RD ~ 1 by year ~15, self-thinning to ~300 large trees/ha, BA plateau 41-46 m2/ha (the SDImax-implied BA of 40-47 m2/ha for dq 30-45 cm; replicate 1: 47.1 at year 100). From year ~85 gaps (longevity/mortality of the large cohorts) are refilled by recruits (TPA rises, QMD falls). Limitations seen here: (i) under an SDImax-limited canopy the smallest cohorts are thinned first, so recruits of shade-tolerant species do not accumulate as an understory (the shade class only reduces the establishment probability, it does not protect small trees from thinning) and hardwoods (QUNI, ACRU; growth/mortality priors with no tolerance advantage in the kernel) vanish after ~50 yr; (ii) all species recruit at 750 tph.
