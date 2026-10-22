# HBAD3P plugin (process-grounded HBAD3 cohort model, fit P3_GAFL)

`succession_plugin: hbad3p`. The cohort model HBAD3P of the prototype tree (`landis_hbad3`: kernel `src/plugins/HBAD3/HBAD3P.jl` = the reference, fit pipeline `tools/hbad3/*`, `docs/hbad3p_pipeline.md`) as a Pan succession plugin, integrated the way
HBAD2 is (`docs/hbad2_plugin.md`): module `HBAD3PPlugin`, plugin type `HBAD3P <: AbstractPlugin` (`src/plugins/HBAD3P/`), fixed parameters (no search path in Pan), tree-based scenario sites. Parameters: the GA+FL fit
`fit3p_P3_GAFL` exported to `results/hbad3p/P3_GAFL`. Biomass never enters the dynamics; `c_bio` is a REPORTED allometric AGB (HBAD v1 priors, as HBAD2). The dynamics are deterministic expected values (no random numbers).

## State (SoA, `plugin.jl`)

Per cohort (CSR `:cohort`), Float64 unless noted: `c_species` (eco-local species, UIntType), `c_age` A (yr), `c_tph` N (trees/ha **of the forested subplot** when `farea = 1`), `c_dq` D (QMD, cm), `c_h` H (m),
`c_zq` vigour node (5-point Gauss-Hermite: z = -2.857, -1.356, 0, 1.356, 2.857; shares 0.0113, 0.2221, 0.5333, 0.2221, 0.0113), `c_cr` crown ratio CR, `c_bio` (FloatType) reported AGB g/m2 **of the plot**
(= tph x area / 2.47105 x exp(mq_a + mq_b ln(D/2.54)) x 453.592/4046.86), scratch `c_O`, `c_SDI`, `c_V` (= min(g, 1)), and `c_ropen` (run year in which the cohort's recruit class opened, 0 = not an open class; stored per cohort so that
it survives swap-removals). Per eco species (`:species`): `sp_seed` (plot-level seed presence, static), scratch `sp_estab`, `sp_rate`, `sp_oy`.
Per site: `W` (period water index AET/PET), `Wn` (its plot normal), `area` (a), `rmult` (recruitment area multiplier), `ryear` (recruitment clock = the reference's run year), `planted`, `no_establish`, and the reported totals
`B`, `BA`, `SDI` (per ha of the **plot**: state x area) and `RD` (of the stand itself, SDI / sdimax).

## Annual step (`kernel.jl`, equations identical to the reference header)

All rates come from the START-of-step state (explicit Euler); Δ = 1 in Pan (the entry points accept a final fraction of a year).

1. SDI_i = N_i (D_i/25.4)^1.605; RD = ΣSDI / sdimax; overtopping O_i = (Σ_{H_j>H_i} SDI_j + ½ Σ_{j≠i, H_j=H_i} SDI_j + ½ SDI_i) / sdimax. An empty stand (ΣSDI = 0) returns before the ageing.
2. Crowding / water / vigour / crown: cw = e^{gR RD} W^ω e^{σv z} (CR/0.4)^φ; light L = f0 + (1 - f0) e^{-a O}; g = L cw.
3. Growth: hyd = max(0, 1 - min(H/(1.1 Hmax), 1)^θ); dV = max(e^{lnε} (D/20)^b 122.98 g hyd fD - ρ D² H, 0) [cm² m/yr]; dD = dV Δ / (2 D max(H, 1.37)).
4. Height: H*(D) = 1.37 + (Hmax - 1.37)(1 - exp(-s D/(Hmax - 1.37)))^c; Lh = e^{-kh O}; dH = [H*(D+dD) - H*(D)](1 + η(1-Lh)) + λΔ(H*(D)(1 + μ(1-Lh)) - H); H += max(fH dH, 0).
5. Mortality: gm = e^{-am O} cw; h = fM [m0/LONGEVITY + ms e^{σST (ST-3)} e^{-β gm} (D/20)^ζ] + e^{mA (A/LONGEVITY - 1)} with fM = (planted ? e^{pM} : 1) e^{mW (Wn - W)/0.1} (the drought **anomaly** of the period against the plot normal, zero when W = Wn);
   N *= e^{-hΔ}. Planted stands: fD = e^{pD}, fH = e^{pH}.
6. Crown ratio: CR* = crlo + (crhi - crlo) e^{-kc O}; CR += (CR > CR* ? τr : τe)(CR* - CR)Δ, clamped to [0.05, 0.95].
7. Self-thinning (after the step): up to 6 passes while ΣSDI > sdimax: remove the excess SDI in proportion to SDI_i (1 - min(1, g_i)) + 1e-9 SDI_i. 8. A += Δ.
9. Recruitment (`no_establish` false): per species k, RD of the post-step state, rate = RSCALE e^{lnF_k - λ0 e^{λ1 (ST_k - 3)} RD} x (seed ? 1 : ε0) x rmult x Δ with RSCALE = 185.2 (one FIA microplot sapling per ha), seed = the plot-level
   seed presence OR a live cohort of the species with D >= 12.7 cm. A species with rate > 0 opens a class (5 vigour nodes at D = 2.54 cm, A = 5, H = H*(2.54), CR = crhi, N = rate x share) when none is open or the open one is >= 5 years old,
   else the year's recruits merge into the open class (quadratic-mean D, N-weighted H, densities add). Species are independent, so phase 1 (rates, `recruit_rates!`) before the CSR growth and phase 2 (`recruit_apply_cap!`) equal the reference's interleaved loop.
   CR of merged recruits stays (all recruits enter with crhi).

`process_plugin!` = phase 1 per site (threaded: step + rates) -> one `readjust_soa!` only when a class opens -> phase 2 (recruits + reporting). Steady state is allocation-free (tested), isbits parameter structs, no dynamic dispatch.

## Parameters (`params.jl`, `context.jl`)

`results/hbad3p/P3_GAFL/` (written by `tools/hbad3p_export_params.jl` from `landis_hbad3/results/hbad3/fit3p_P3_GAFL`): `params_species.csv` (key, eps, Hmax, ST, longevity), `params_global.csv` (the 32 `GlobPP` fields, unicode names of the fit),
`params_recruitment.csv` (`lnF:<key>`, λ0, λ1, ε0), `meta.txt` (the fit's meta: form=2, size_split 5,12.7,25,40, site=subplot, vigour=1, sigv_max=1.2, vig_n=5, dclass=lower, farea=1; + source, region=GAFL, fit_log),
`plot_normal_water.csv` (st, u, c, p, wn). 12 keys: ACRU LIST2 LITU MAVI2 NYBI PIEL PITA QULA3 QUNI TAAS + `_H`, `_S` (a Pan species maps to its own key if the set has a row, else the group key by SFTWD_HRDWD, exactly as HBAD2).
Per key: lnε = log(eps) (calibrated), lnF (fecundity, calibrated on sapling ingrowth), traits Hmax, ST, LONGEVITY. Globals fitted: b 1.76402, a 3.96213, gR 0.562937, θ 7.73305, ω 1.19688, s 1.09674, c 1.19793, η 0.481698, λ 0.00187067, μ 2 (at its bound),
kh 10.6215, m0 0.688957, ms 0.0855176, σST 0.236596, β 3.23648, ζ -0.632185, mW 0.463591, mA 10, pD 0.125486, pH 0.261156, pM 0.127673, σv 0.510187, sdimax 1500, am 0.75483, ρ 6.82521e-5, f0 0.181827, φ 0.690545, crhi 0.489797, crlo 0.384572,
kc 4.48621, τr 0.0719741, τe 0.0107857. Recruitment: λ0 3.50057, λ1 -0.240374, ε0 0.306922.
yaml roster `results/params_yaml/r4_hbad3p_gafl.yaml` (plugin `hbad3p`; `params_to_yaml` / `params_from_yaml`, exact round trip, unknown / missing keys are errors): `global` (ASCII names: b a gR theta omega s c eta lambda mu kh m0 ms sigma_ST beta zeta mW mA pD pH pM sigma_v sdimax am rho f0 phi crhi crlo kc
tau_r tau_e, lam0 lam1 eps0, `farea`, `vig_n` = 5), `species` per key (name, lneps, Hmax, ST, longevity, lnF), `ecoregions` (the R4 eco cells of `r4_hbad2_v2j.yaml`, each listing all keys), `species_map`, `eco_mapping`, and `data.plot_normal_water`
(path of the plot-normal table relative to the repository root). The parameters are those of a GA+FL fit; the roster covers the R4 eco cells so the set can be applied (extrapolated) anywhere in R4.
Config keys (`succession_plugin: hbad3p`, `hbad3p_params_dir`, `hbad3p_vigour`) select the plugin and its context like HBAD2; there is no search (`make_param_dists` raises). `μ` sits at its bound 2 in the fit (the prototype log lists it as AT BOUNDS).

## Data conventions (`src/data/tree_cohorts3p.jl`, `scenario.jl:_hbad3p_sites`)

* **Trees.** Live trees (STATUSCD 1, DIA > 0, HT > 0, estimated_age > 0, TPA_UNADJ > 0) of the plot visit from the scenario's FIA database (`curated_trees`, READ_ONLY, bound parameters, `ORDER BY` the full key); w = TPA_UNADJ x 2.47105 trees/ha, D = DIA x 2.54, H = HT x 0.3048, CR = FIA CR/100.
* **Sites.** 4 subplot sites per plot (empty subplots are empty sites). Cohort = (key, birth decade `floor((year(measdate) - age)/10)`, diameter class `searchsortedlast([5, 12.7, 25, 40], D) + 1`, i.e. classes **closed below** [b_i, b_i+1); a 12.7 cm tree is not
  lumped with microplot saplings) x 5 vigour nodes. N = 4 sum(w) x node weight / a, D = w-weighted quadratic mean, H = max(w-weighted mean, 1.37), A = w-weighted mean, CR = w-weighted mean over trees with a recorded CR > 0, divided by 100, else 0.4.
  This is the prototype's `subplot_cohorts` (lib3p.jl); the Sim A harness additionally uses the FIA CR of the first visit as the initial CR.
* **Forest area (farea = 1, the convention of the fit).** FIA tallies trees only on forest conditions, so a subplot's tree list covers the forested fraction f of its area. f = sum of SUBPCOND_PROP over the COND_STATUS_CD = 1 conditions of the subplot
  (SUBP_COND x COND x PLOT of the same database, visit date = PLOT MEASYEAR/MEASMON/MEASDAY; NaN if the proportions do not sum to 1). a = max(f, 0.25) (`FOREST_FMIN`), a = 1 when f is unknown or 0 (a stocked site that FIA records as nonforest).
  Stand densities are per ha of the **forested** subplot (N / a); competition sees the true density. Recruitment per ha of the plot is scaled by `rmult` = f / a (1 when f is unknown or 0 with trees tallied; 0 for an empty nonforest subplot: it does not recruit).
  If the fractions cannot be read from the database, a = 1 and rmult = 1 with a warning.
* **Plot-level values.** A plot's metrics are the mean of its 4 subplot sites per ha of the plot: state x a (`_cohort_struct` returns tph x area; `c_bio`, `B`, `BA`, `SDI` are already x a). Volumes, QMD and the species x DBH tally therefore use plot-ha densities.
* **Seed presence** `sp_seed[k]` = a tree of key k with D >= 12.7 cm on any subplot of the visit (plot-level, static, dispersal between subplots), as the prototype's `seedm`.
* **Water (climate).** `run_scenario` computes every plot's water index from the request's climate (`scenario.jl:_hbad3p_water!`, `HBAD3PPlugin.water_index`,
  a line-by-line port of the reference `scripts/hbad3/water_balance.py`):
  * **PET** (potential evapotranspiration, mm/month): Thornthwaite, from the monthly mean temperature (= (tmin + tmax)/2 of the scenario climate) and the
    day length at the plot's latitude. **AET** (actual evapotranspiration): what a monthly soil bucket of capacity AWC (= (field capacity − wilting point) ×
    rooting depth: SOLUS at the plot's location, else the FIA plot's soil table; floored at 5 mm) can supply: each month W += P, AET = min(PET, W),
    W = min(AWC, W − AET); spun up from 1980, full at the start.
  * annual index = April–October ΣAET / ΣPET (1 = no water limitation); **Wn** = its 1991–2020 mean under the scenario's baseline WITHOUT changes (the
    normal of the fit; replaces the table value); **W** of simulated year t (calendar year = visit year + t − 1) = trailing mean over
    `HBAD3P_WATER_WINDOW` = 5 years (the fit's W is a remeasurement-interval mean; median interval 5 yr).
  * a smooth baseline (the default `forecast`, or `average` with K > 1) has no interannual variability and its index sits above the normal (median +0.046
    on 300 R4 plots: −19 % drought mortality, +5.5 % growth with no climate change), so years after 2024 take Wn + (scenario − baseline); observed years
    and a replayed baseline (`repeat`, `average` K = 1) keep their own values. An unchanged baseline therefore gives exactly W = Wn after the window.
  * parity: `tools/hbad3p_water_parity.jl` recomputes the fit's plot normals from the reference inputs (PRISM tmean, L4-mean latitude, plot AWC):
    max |error| 4e-16 on 400 plots; the engine's own choices ((tmin+tmax)/2, the plot's latitude) move Wn by ≤ 0.005.
  * Climate acts through W only: growth × W^ω and mortality × e^{mW (Wn − W)/0.1}; temperature enters through PET, there is no CO₂ term.
  * Without a climate database (or for plots without an L4) W = Wn = the table's plot normal (regional median 0.9534 when missing), with a warning.
* **Management** (scenario): removal at constant tree size (`scale_cohort!`: N x keep; D, H, CR, A unchanged; a cohort leaves only when ALL its trees are removed), residual BA / TPA / SDI targets are met exactly by the plot mean. **Planting**: `tpa` x survival is the density per ha of the forested stand
  (the state density; the plot value is x a), 5 vigour nodes at D = 2.54, H = H*(2.54), A = 1, CR = crhi, `planted` set (fD, fH, fM apply from then on). `hbad2_cleanup` (trees/ha, default 1e-3 in scenario runs, 0 = off) removes negligible cohorts after every step.
* `add_cohort!` (AGB-driven injection used by the generic `make_sites` path) creates ONE mean-field cohort (zq = 0, CR 0.4) from the HBAD v1 translation; `init_sites!` sets W = Wn = the regional median there.

## Tests

* `test/unit_hbad3p.jl` (plugin unit tests: parameters / keys / roster round trip, quadrature moments, allocation-free step, serial = threaded hash, monotone N, add / swap / set_cohort_bio, plant!, forest-area reporting, cleanup).
* `test/unit_hbad3p_parity.jl` (in `runtests.jl`): the plugin vs the REFERENCE kernel `landis_hbad3/src/plugins/HBAD3/HBAD3P.jl` (included by a path relative to the worktree's parent; the test is skipped with a warning if the prototype tree is absent): 300 random sites
  (heights tied, planted, partial years, RD > 1 so the self-thinning loop acts, perturbed globals and traits); 20-year free runs with recruitment (class merging, seed presence, rmult incl. 0) with ample and with tight capacity (dropped classes), and through `process_plugin!`.
* `tools/hbad3p_parity_simA.jl` (end to end, needs the prototype tree and its data products; ~2.5 min): the plugin kernel on the prototype's own Sim A VAL (and TRAIN) site arrays reproduces the fit's validation NLL. Output: `results/hbad3p/parity_simA_P3_GAFL.txt`.
* `test/unit_scenario_hbad3p.jl` (needs the FIA database): roster, sites, forest area, water normals, climate (water index: normal vs table, baseline = no anomaly, drought scenario lowers W and BA, year mapping), constant-size removal with exactly met residual targets, planting, no-op and thinning scenario runs, determinism, thread independence.

Results (2 threads; full output in the session report):

```
unit_hbad3p_parity: step 300 sites / 1497 cohorts (241 sites over sdimax, 384 height ties, 130 planted, 149 partial years): max rel diff 0 (N, D, H, A, CR)
                    free run, ample capacity (40 sites x 20 yr, 101,100 cohort-states): max rel diff 0, 0 dropped
                    free run, tight capacity + partial last year (7038 dropped classes, same count as the reference): max rel diff 0
                    process_plugin! free run (101,100 cohort-states): max rel diff 0
hbad3p_parity_simA: VAL NLL plugin -272207.8631 (reference kernel identical, logged FIT -272207.9: rel. diff 1.4e-7); TRAIN -271807.4792 (logged -271807.5); state N, D, H, CR identical over 400,960 / 401,965 sub-cohorts
```

## Deviations from the reference / conventions of the port

* Cohort bookkeeping of the recruit classes is stored per cohort (`c_ropen`, as HBAD2) instead of the reference's `open_slot` / `open_year` matrices (survives swap-removal); results are identical (parity above).
* `W`, `Wn`, `area`, `rmult`, `sp_seed` are per-site arrays of the SoA (the reference keeps them in `SitesP` / `FreeSitesP`).
* Reported values (`c_bio`, `B`, `BA`, `SDI`) carry the forest-area factor a; the dynamics do not.
