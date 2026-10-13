# GA+FL PnET-Succession calibration (Sim A) — decision log

Same data, split, train cap, TOST groups, real-only criterion and 3-fold CV protocol as the BiomassSuccession GA+FL
calibration (`docs/gafl_calibration_log.md` D1–D11) so the models compare directly. Date: 2026-10-13.

## P1. Setup
- Config `configs/region4/gafl_pnet.yml` = `configs/region4/gafl_A.yml` + `succession_plugin: pnet`, PRISM L4 climate cache,
  NOAA CO₂, per-plot SOLUS soil (`pnet_soil_csv`); FolN pooled like B_MAX (region level, per site cell for PITA/PIEL).
- Tools: `Pan.setup_plugin_from_cfg` (configure the plugin + site attributes from a yaml exactly as `parametrize` does) so
  `tools/tost_common.jl` / `calib_coord.jl` / `tost_select.jl` / `gafl_cv.sh` run any plugin.
- Start `tools/seed_pnet_start.jl`: plant-functional-type defaults of `params.jl` without jitter (FolN, MaxFracFol,
  FrActWd, SLWmax, PsnAgeRed), TOwood 0.01, RD_MULT 1, LONGEVITY from the BS set. Start on train: 0/2/5 of 46 groups at
  ±5/10/20 %, median |bias| 32.5 %, 40/46 groups OVER-predicted (NYBI +141 %, PIPA2 +93 %, PITA·D +90 %).
- Cost: one GA+FL train evaluation ≈ 15–20 s with 4 threads (data prep + compile ≈ 2 min once), i.e. the monthly
  ecophysiology is not the bottleneck at this scale.

## P2. Lever and procedure
FolN (foliar N %, EcoSpecies, the PnET productivity knob — Amax is linear in FolN) per TOST group, root-found with the
same coordinate method (`calib_coord.jl` lever `foln`, grid 0.7–3.0, bounds 0.6–3.0), schedule `foln,foln~,foln~,foln~,foln~`.
Species-level traits (MaxFracFol, SLWmax, TOwood, PsnAgeRed) were left at PFT defaults: a group-level lever cannot set a
species trait shared by several cell groups, and FolN alone removed the level bias. Residual size structure after FolN
(train): obs < 100 g/m² +21 % (4 % of pairs), 100–1,000 −4 %, 1–5 kg +4 %, > 5 kg −7 % — no second lever needed.

## P3. Result
| held-out | ±5 % | ±10 % | ±20 % | median abs bias |
|---|---|---|---|---|
| val (fit train+test) | 3/46 | 22/46 | 33/46 | 4.2 % |
| train (fit val+test) | 1/46 | 13/46 | 34/46 | 7.5 % |
| test (fit train+val) | 4/46 | 13/46 | 33/46 | 6.5 % |
| mean | ≈ 3 | ≈ 16 (35 %) | ≈ 33 (72 %) | 6.1 % |
| BiomassSuccession v1 (same protocol) | ≈ 8 | ≈ 20 | ≈ 36 | 5.1 % |
Robust in all 3 folds: ±10 % 10 groups (QUNI, NYBI, QUAL, _H_LST, PIEL·AB, QULA3, _H_HST, QUST, LITU, PIPA2); ±20 % 30.
Deployable set (all GA+FL plots): `results/pnet_gafl/params_gafl_pnet.jld2` — in-sample 30/46 ±5 %, 42/46 ±10 %, 45/46 ±20 %.
FolN median 1.75 % (0.6–3.0); 9/508 targets at 3.0 and 15 at 0.6 (PIPA2 at 0.6 = still slightly over-productive).
**TADI2 (bald cypress) cannot be fitted** (−36 % at FolN 3.0): its deciduous-conifer PFT row has no reference values
(interpolated, `params.jl`) — the PFT photosynthesis/allocation constants, not FolN, limit it. TAAS (same PFT) fits at 2.77.
In-sample PnET equals BS v1 (42/46 vs 42/47 at ±10 %) but generalises slightly worse (≈16 vs 20 held-out at ±10 %): one
productivity scalar per group cannot reproduce how growth changes with cohort size/age across plot samples as well as
BS's three-parameter curve, and climate/soil drivers do not add held-out skill at this plot-interval scale.
Files: `results/pnet_gafl/` (cvP_* held-out group tables, final in-sample table, rounds, cv_summary.log).
