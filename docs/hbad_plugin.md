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

`HBADPlugin.configure!(; species_list, eco_species_ids, priors_csv, coefs_csv)` resolves, per GLOBAL species, one `SpPrior` (isbits, 28 Float64) from
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

`# ESTABLISHMENT HOOK` in `plugin.jl` marks where recruitment would append cohorts. Not implemented: `no_establishment: true` semantics (like PnET); `dual_mode`/Sim B and `spinup: true` raise clear errors.

Performance/design: sites are independent; per-site work touches only that site's CSR slice; no scratch buffers, no dynamic dispatch (eco parameters are a concrete struct of isbits `SpPrior` + Float64 vectors), no allocation (verified, section 8).

## 4. Initialisation / injection

`add_cohort!(site, esp, age, agb)` (module Refs `PRIORS[]`, `ECO_GIDS[]`, set by `configure!`/`set_eco_species!`) -> `translate(prior, age, agb)`, the selected translation of `tools/hbad_translate_lib.jl` (structure A, TPA-first, `mq`, Chapman-Richards height; no SI, natural):

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
| `LONGEVITY` | fixed | as BiomassSuccession: `LONGEVITY_TABLE` (yaml `longevity_from_data`, `longevity_overrides`) / `fix_longevity` / 400 | - | - |

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
* No establishment, no spinup, no dual mode. Establishment hook marked in `plugin.jl`.
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
