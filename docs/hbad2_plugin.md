# HBAD2 plugin (HBAD v2, model v2g) -- stage 1 of the Pan integration

`succession_plugin: hbad2`. The biomass-free, process-based cohort model of the prototype branch `hbad-v2` (`docs/hbad_v2.md` 4.1 and 5.17 there, kernel
`src/plugins/HBAD2/HBAD2.jl` there) as a Pan succession plugin: module `HBAD2Plugin`, plugin type `HBAD2 <: AbstractPlugin` (`src/plugins/HBAD2/`).
Stage 1 = the kernel with the FIXED v2g parameters, validated bit for bit against the prototype. The CALIBRATION (data, Sim A / Sim B likelihoods, parameter vector, search) is `docs/hbad2_calibration.md`: a run with a `hbad2_calib:` block
calibrates the plugin (`search_mode: cmaes | igelmo`); `make_param_dists` still raises for the generic Biomass-Succession style search path, which hbad2 does not use.

## State (SoA, `plugin.jl`)

Per cohort (CSR `:cohort`), Float64 unless noted: `c_species` (eco-local species, UIntType), `c_age` A (yr), `c_tph` N (trees/ha), `c_dq` D (QMD, cm), `c_h` H (mean height, m),
`c_zq` vigour quantile, `c_bio` (FloatType) REPORTED allometric AGB = tph/2.47105 * exp(mq_a + mq_b ln(D/2.54)) * 453.592/4046.86 g/m2 (HBAD v1 priors, `kernel.jl:agb_of`; never used by the
dynamics), scratch `c_O` (overtopping), `c_SDI`, `c_V` (vigour), and `c_ropen` (run year in which the cohort's recruit class opened, 0 = not an open class).
Per eco species (`:species`): `sp_mature` (a D >= 12.7 cm seed tree is present; carries the species count for the shared loss), scratch `sp_estab`, `sp_rate`, `sp_ba`, `sp_oy`.
Per site: `B`, `BA`, `SDI`, `RD` (reported), `ryear` (recruitment clock = the prototype's run year), `planted` (stand origin: planted multipliers), `no_establish` (Sim A: no recruitment).
The prototype's per-site recruit bookkeeping (open slot + open year per species) is stored on the cohorts (`c_ropen`) so that it survives `swap_cohort!` / injection removals.
There are no random numbers: the dynamics are deterministic expected values (the per-site RNG of the SoA is unused), hence independent of the thread count.

## Annual step (`process_plugin!`, `plugin.jl`) -- equations with file:line

Phase 1 per site (`site_prepare!`, threaded over sites): `step_cohorts!` (`kernel.jl:35`, prototype `step_site!`), then `recruit_rates!` (`kernel.jl:89`) for sites with recruitment.
The CSR is grown once if a recruit class opens anywhere (`readjust_soa!`, steady state allocation free); phase 2 (`site_commit!`): `recruit_apply!` (`kernel.jl:123`) and `report_site!`.

1. Snapshot (`kernel.jl:36-48`): SDI_i = N_i (D_i/25.4)^1.605; SDImax_site = SDI-weighted mean of species SDIMAX; RD = sum SDI / SDImax_site; overtopping
   O_i = (sum_{H_j>H_i} SDI_j + 1/2 sum_{H_j=H_i, j!=i} SDI_j + 1/2 SDI_i) / SDImax_site. An empty stand (sum SDI = 0) returns before ageing (`:40`).
2. Growth (`kernel.jl:50-56`), form 2: dD = rD (D/20)^cD e^{-D/Dmax} e^{-gOD O} e^{gRD RD} e^{sigma_v zq} [e^{pD} if planted];
   dH = rH (H/15)^cH e^{-H/Hmax} e^{-gOH O + gRH RD} e^{sigma_v zq} [e^{pH} if planted] (form 1 = Chapman-Richards state form is kept).
3. Mortality (`kernel.jl:57-60`): h = fM [ e^{mbg + mD ln(D/20) + mD2 D/20} + msup (1 - min(1, e^{-gOD O} e^{sigma_v zq})) (D/20)^mSD ] + e^{mA (A/LONGEVITY - 1)}, fM = e^{pM} if planted; N <- N e^{-h};
   D += dD, H += dH (growth uses the snapshot, all cohorts updated in one pass).
4. Self-thinning cap (`kernel.jl:62-74`): up to 6 passes, while RD > 1 remove SDI in proportion to SDI_i (1 - v_i) (suppressed first). It never binds in the fitted model.
5. A += 1 (`kernel.jl:75`).
6. Recruitment (free runs, `no_establish` false): per species R = 185.2 exp(rho0 + rhoS ln(1 + BA_sp(D >= 12.7)) + rhoL SDI/600) trees/ha (`kernel.jl:107`; SDI of the post-step state, BA m2/ha).
   A species opens a new class when none is open or the open one is >= 5 years old (`:109`): 3 vigour sub-cohorts (zq = -sqrt3, 0, sqrt3; shares 1/6, 2/3, 1/6) at (D0 2.54 cm, H0 3 m, A0 5 yr);
   otherwise the year's recruits merge into the open sub-cohorts (D quadratic mean, H N-weighted, N adds). Species rates are independent of each other, so computing all rates first (phase 1) equals the prototype's interleaved loop.

Deliberate non-differences: cohorts are never removed by the kernel (as in the prototype), so a long free run accumulates 3 sub-cohorts per species per 5 years (cost O(n^2) in the overtopping loop).

## Parameters (`params.jl`, `context.jl`)

`results/hbad2/params_{species,global,recruitment}.csv` + `meta.txt` are copies of the prototype's `results/hbad2/fit_v2g/` (sha256 equal, checked by the parity tool): 12 keys (ACRU LIST2 LITU MAVI2 NYBI PIEL PITA QULA3 QUNI TAAS + `_H`, `_S`), 15 per-key parameters (incl. LONGEVITY from the csv),
8 globals (mD2, mA = 10, pH, pD, pM, form = 2, mSD, sigma_v = 0.7), recruitment rho0 rhoS rhoL D0 H0 A0 per key. `HBAD2Context(species_list, eco_species_ids; params_dir, vigour)` maps every Pan species to a key (logged):
own symbol if the set has a row for it, else the group key `_S` / `_H` by SFTWD_HRDWD (`results/hbad/species_group.csv`; `_S*`, `_H*`, `_GRP_n_*` catch-alls by group; unknown -> `_H`), and resolves the HBAD v1 priors for the species
(only the reported allometry and the AGB -> state translation of `add_cohort!`). `generate_params` returns the fixed set (`HBAD2Params`: keys, `SP`, `REC`, `G` per global species); `generate_eco_params` the per-eco `HBAD2EcoParams`.
yaml: `succession_plugin: hbad2`, `hbad2_params_dir`, `hbad2_vigour`; example `configs/region4/hbad2_v2g_example.yml`.

## API

`add_cohort!` (AGB-driven injection: ONE mean-field cohort from the HBAD v1 translation), `set_cohort_bio!` (scales N), `plant!(ctx, site, sp, tph; age=1, dq=2.54, h=3)` (3 vigour sub-cohorts, merges into existing planted sub-cohorts of the same species and age),
`plant_slot_needed` (0 or 3), generic `plant_slots_reserved` (1 for every other plugin, 3 for hbad2; `plant_sites!` reserves that many slots), `spinup!` errors (start from observed cohorts).
`plant!` itself does not set the site's `planted` flag (stand origin is a site attribute: `make_sites_hbad2*(...; planted=...)`); the scenario engine sets it (stage 2, below).
Stage 2 additions: `scale_cohort!(ctx, site, k, keep)` (management removal at constant size, updates the reported AGB), `refresh_totals!(ctx, site)` (B, BA, SDI after management), `identity_eco_params(ctx)`,
`HBAD2Context(...; v2tables=)` (context from the tables of a yaml roster), `tables_of(params)`, and the cleanup option below.

## Negligible-cohort cleanup (`HBAD2Const.cleanup_tph`, stage 2)

The kernel never removes cohorts (prototype behaviour; the parity gates rely on it), so a free run accumulates 3 vigour sub-cohorts per species per 5 years. `HBAD2Const(cleanup_tph=thr)`
(`HBAD2Context(...; fixed=...)` / `with(ctx; fixed=...)`) removes, at the end of every annual step of a site (`site_commit!`, after recruitment, before the reported state: `drop_small_cohorts!`), the cohorts with
fewer than `thr` trees/ha; the others keep their order (stable compaction, allocation-free; the CSR capacity shrinks at the next `process_plugin!` readjust, so only years in which something is removed allocate).
**Default 0 = OFF** (stage-1 parity tests and `tools/hbad2_parity.jl` run with it off and stay exact); the scenario engine turns it ON (1e-3 trees/ha, `run_scenario(...; hbad2_cleanup=)`, 0 disables).
Effect on a 100-year free run (4 plots x 4 subplot sites = 16 sites, no management, recruitment on; cohorts summed over the sites, metrics = max relative difference to cleanup off over the 4 plots and 11 records):

| threshold (trees/ha) | cohorts at 25 / 50 / 75 / 100 yr | max rel. diff BA | TPA | cubic vol | AGB |
|---|---|---|---|---|---|
| 0 (off) | 3021 / 5901 / 8781 / 11661 | 0 | 0 | 0 | 0 |
| 1e-3 (scenario default) | 2894 / 5536 / 7964 / 10290 | 1.7e-5 | 2.3e-5 | 1.8e-5 | 1.5e-5 |
| 1e-2 | 2801 / 5283 / 7428 / 9257 | 1.7e-4 | 9.3e-4 | 1.6e-4 | 1.0e-4 |
| 1e-1 | 2402 / 4094 / 5322 / 6326 | 1.4e-2 | 2.0e-2 | 1.3e-2 | 1.1e-2 |
| 1 | 1081 / 1409 / 1525 / 1534 | 0.11 | 0.26 | 0.095 | 0.089 |

At 1e-3 the metrics change by < 3e-5 but the cohort count falls only ~12 % after 100 years (most recruit classes are suppressed but above 1e-3 trees/ha); 1e-2 is still < 1e-3 relative. Growth of the cohort count is
therefore not stopped by a small threshold; a larger one trades accuracy for speed (the overtopping loop is O(n^2) per site).


## Tree loader (`src/data/tree_cohorts.jl`)

`Data.load_tree_cohorts(db_path, visits, keys; birth_w=10, size_breaks=[5, 12.7, 25, 40], vigour=true, subplots=1:4, site_mult=4.0)`: database opened READ_ONLY; `visits` (statecd, unitcd, countycd, plot, measdate) joined as a registered table, filters are bound parameters,
`ORDER BY` STATECD, UNITCD, COUNTYCD, PLOT, MEASDATE, SUBP, TREE, CN. Live trees (STATUSCD 1, DIA > 0, HT > 0, estimated_age > 0, TPA_UNADJ > 0). Site = (visit, SUBP), empty subplots included. Tree: w = TPA_UNADJ x 2.47105, D = DIA x 2.54, H = HT x 0.3048.
Cohort = (key, floor(-age/birth_w), size class `searchsortedfirst(size_breaks, D)`); N = sum(w) x 4, D = w-weighted quadratic mean, H and A w-weighted means (A >= 1); sorted by (key, birth class, size class); each cohort -> 3 consecutive vigour sub-cohorts.
`make_sites_hbad2(plugin, tc; ...)` / `make_sites_hbad2_arrays` build the SoA (key index = eco-local species index; the context's species list must equal `tc.keys`).

## Validation

* `tools/hbad2_parity.jl` (full gate, needs the prototype tree read-only + `slim_r4.duckdb`) and `test/unit_hbad2_parity.jl` (fast; vendored verbatim prototype kernel `test/ref_hbad2_proto.jl`, synthetic sites, in `runtests.jl`). Shared code `test/hbad2_parity_lib.jl`.
* `test/unit_hbad2.jl`: parameters / key mapping, zero allocations of the annual step in the steady state (serial site loop, with and without recruitment), identical state hash for repeated runs and serial vs threaded site loops,
  monotone N (except open recruit classes), add / swap / set_cohort_bio consistency, plant!.
* Results of the full gate: see the section below.

### Results of `tools/hbad2_parity.jl` (2 threads; prototype branch hbad-v2 @ 10befe5, read-only)

```
files: params_{species,global,recruitment}.csv, meta.txt  prototype == pan (sha256 identical); kernel HBAD2.jl == test/ref_hbad2_proto.jl (f3bc3edd4810ee55)
B  (Sim B VAL, 1860 subplot sites, 25 years, recruitment + vigour, 5,548,425 cohort-states): max rel diff = 0 (N 0 D 0 H 0 A 0), structural mismatches 0  (dropped recruits 0)
B' (Pan initial state from Data.load_tree_cohorts, in the prototype's cohort order): max rel diff = 7.19e-15 (N 7.19e-15 D 7.7e-16 H 1.0e-15 A 4.4e-16), mismatches 0
    loader vs prototype build_simB initial state: 465 visits -> 1860 sites, 21057 cohorts, 0 sites with a different cohort count, 0 unmatched, max rel diff 4.3e-16 (summation order only)
A  (Sim A style, 200 VAL sites incl. 55 planted stands, 969 site-years, no recruitment, 12,336 cohort-states): max rel diff = 0, mismatches 0
PARITY PASS (max relative difference <= 1e-12)
```
With identical initial arrays the plugin is bit-identical to the prototype (same operations in the same order). B' differs only by the loader's summation order of the initial N/D/H/A (<= 4.3e-16), amplified to 7e-15 over 25 years.
