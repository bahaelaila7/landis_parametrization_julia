# GA+FL: BiomassSuccession vs PnET vs HBAD (Sim A) and BiomassSuccession Sim B — summary (2026-10-13)

Common protocol: Georgia + Florida FIA plots (8,985), species-balanced train cap, 47 TOST groups (species; PITA/PIEL per
site cell), equivalence test of the per-group mean log AGB bias at ±5/10/20 % (α = 0.05, real records only), per-group
coordinate calibration (`tools/calib_coord.jl`), 3-fold CV over the train/val/test splits + all-data fit
(`tools/gafl_cv.sh`). Groups passing (equivalent) out of ≈46:

| model (lever) | held-out ±5 % | ±10 % | ±20 % | robust ±10 / ±20 (all 3 folds) | in-sample all data ±5/10/20 | deployable params | details |
|---|---|---|---|---|---|---|---|
| **HBAD** (G_MULT, M_MULT) | **≈17** | **≈34** | **≈41** | **28 / 40** | 42 / 44 / 47 | results/hbad_gafl/params_gafl_hbad.jld2 | docs/hbad_plugin.md |
| BiomassSuccession v1 (S, ratio, B_MAX) | ≈8 | ≈20 | ≈36 | 14 / 29 | 34 / 42 / 45 | results/gafl/params_gafl_biomass_real.jld2 | docs/gafl_calibration_log.md D10–D11 |
| BiomassSuccession v2, wide box | ≈7 | ≈18 | ≈34 | 12 / 26 | 33 / 41 / 44 | results/gafl/A2_bounds/ (not deployable) | D11 |
| PnET (FolN) | ≈3 | ≈16 | ≈33 | 10 / 30 | 30 / 42 / 45 | results/pnet_gafl/params_gafl_pnet.jld2 | docs/gafl_pnet_calibration.md |
| BiomassSuccession **Sim B** (free run; PROB_MORT, B_MAX, ratio) | 0 | ≈4 | ≈17 | 1 / 8 | 7 / 26 / 40 (of 48) | results/gafl_simB/params_gafl_simB_3lever.jld2 | D12–D15 |

Sim B uses a different statistic (plot-cluster bootstrap of log Σsim/Σobs; D14) because a free stochastic run cannot predict
which cohorts die; it is not directly comparable to the Sim A rows.
