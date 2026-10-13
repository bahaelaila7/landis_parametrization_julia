# HBAD empirical translation and priors (Georgia + Florida FIA)

Everything here is derived from `curated_trees` in `slim_r4.duckdb` (read-only), STATECD 12 and 13, restricted to
the plots of `runs_local/gaflst_split/split_plots.csv` (8,985 plots; train/val/test by plot).
Regenerate with (from the worktree root, `source wt_env.sh`, `--threads=2`):

    tools/hbad_cohorts.jl      -> results/hbad/cohorts_gafl.arrow (zstd Arrow) + cohorts_gafl_head.csv
    tools/hbad_translation.jl  -> translation_{coefs,coefs_all_variants,selection,metrics}.csv, species_group.csv
    tools/hbad_priors.jl       -> priors.csv, priors_dgrowth_by_dbh.csv, priors_mortality_table.csv, priors_reineke_slope.csv
    tools/hbad_translate_lib.jl  (stand-alone `hbad_translate`, after `hb_init()`); tools/hbad_common.jl (loaders, OLS, LM)

## 1. Cohort table
Live trees (STATUSCD 1) with DIA>0, HT>0, estimated_age>0 (988,187 tree-visits). Cohort = (plot, SUBP, species symbol,
round(estimated_age) (min 1), INVYR). 731,403 cohort-visits: train 231,217 / val 252,484 / test 247,702. 205 species.
Units: `agb_g_m2` = sum(DRYBIO_AG lb x TPA_UNADJ) x 453.592/4046.86 (Pan's factor); `tpa_ha` = sum TPA_UNADJ x 2.47105;
`ba_m2ha` = sum(pi/4 DIA_m^2) x TPA_ha; `qmd_cm` = sqrt(4 BA/(pi TPA)); `h_mean_m` = TPA-weighted mean HT (m);
`h_dom_m` = TPA-weighted mean height of the tallest 40 % of the cohort's stems (partial weight at the boundary tree);
`si_m` = mean cond_sicond (ft x 0.3048; missing for ~46 % of cohorts; base age/species of SI vary); `planted` = any tree on a
condition with cond_stdorgcd==1; `sap_frac` = fraction of TPA from DIA<5 in trees; `n_trees`; `epa_l3`; `set`.

**Design caveats that matter for every number below.**
* TPA_UNADJ takes only two values: 6.018 (macroplot trees, DIA>=5 in; already expanded over the 4 subplots) and 74.965
  (microplot saplings, DIA<5 in). A subplot-level cohort therefore has a *discrete* density: 82 % of cohorts are single trees
  (TPA_ha = 14.87 exactly), 11.8 % contain saplings (185 trees/ha per sapling). "Density" is the TPA_UNADJ sum, i.e. Pan's
  plot-scale accounting of a subplot cohort, NOT the density of the subplot area itself (x4 for macroplot trees).
* Saplings (DIA<5 in) are kept in all headline numbers ("all trees"); "no_saplings" / "with_saplings" subsets are reported in
  `translation_metrics.csv`. Sapling cohorts are poorly predicted (TPA R2 -8, MdAPE 70 %) because (AGB, age) cannot tell a
  microplot sapling cluster from a larger macroplot tree.
* estimated_age is itself modelled (age_source estage/model), so age-dependent effects are partly circular.

## 2. Translation (species, age, AGB[, SI, planted]) -> (TPA, QMD, BA, H_mean, H_dom)
Fit on TRAIN plots; model forms chosen on VAL; TEST untouched. Per-species models for the 30 species with most train cohorts
(PITA, PIEL, LIST2, QULA3, QUNI, ...), pooled `_S`/`_H` (SFTWD_HRDWD) for the rest, `_ALL` last resort.
Selected structure (translation_selection.csv): **A, "TPA-first"**
1. `ln TPA_ha = b0 + b1 x1 + b2 x2 + b3 x1^2 + b4 x1 x2 + b5 x2^2` (x1 = ln AGB - 6, x2 = ln age - 3.3; clipped at one macroplot tree, 14.87/ha);
2. mean tree mass `m_lb = AGB/(TPA_ac x 0.112)`; `QMD_in = exp((ln m - a)/b)` with the cohort-level mass-QMD relation `ln m_lb = a + b ln QMD_in`
   (`mq`; absorbs the Jensen gap between the arithmetic mean tree and the tree-level allometry; tree-level alpha/beta alternative = `tree`);
3. `BA = pi/4 QMD^2 TPA` (so AGB = TPA x m(QMD) holds by construction);
4. `H_mean` = Chapman-Richards on QMD: `H = 1.3 + a (1-exp(-b D_cm))^c` (per-species tree fit, `hcr`); with SI supplied, `ln H = a + b ln QMD + c ln age + d (SI-25)` (`hllsi`);
5. `H_dom = H_mean x ratio` (median h_dom/h_mean of multi-tree train cohorts).
Alternatives compared on VAL (MdAPE of TPA/BA/QMD, `alt_structure_*` rows): A/mq 26.4/11.6/10.8, A/tree 26.4/11.7/10.9, B (QMD regression first, TPA from AGB/m) 27.8/12.9/11.8,
C (independent direct regressions of TPA, BA, QMD) 26.4/11.1/10.6 but AGB-inconsistent (4 % MdAPE) and only 5 % better, so the consistent A was kept (rule: C wins only if >10 % better).
Variants V0 (1,x1,x2), V2 (+planted), V3 (quadratic), V4 (V3+planted): V3 best for TPA, V0 for QMD, V4 for BA/H when used directly. Site index helped H (-0.8 pts MdAPE) and direct BA, not TPA/QMD.
Planted flag did not help the chosen path. API: `hb_init(); hbad_translate("PITA", 25, 1500.0; si=24.0)` -> NamedTuple (tpa_ha, qmd_cm, ba_m2ha, h_mean_m, h_dom_m, agb_check_g_m2).

### Validation (cohort level; R2 on log scale; MdAPE = median absolute % error)
| variable | VAL R2 | VAL MdAPE % | TEST R2 | TEST MdAPE % |
|---|---|---|---|---|
| TPA_ha | 0.563 | 26.4 | 0.560 | 27.1 |
| BA m2/ha | 0.938 | 11.6 | 0.936 | 11.7 |
| QMD cm | 0.814 | 10.8 | 0.813 | 11.0 |
| H_mean m | 0.740 | 12.3 | 0.739 | 12.4 |
| H_dom m | 0.730 | 11.8 | 0.728 | 11.9 |
| AGB recomputed from predicted TPA, QMD with tree-level allometry vs input | - | 0.53 | - | 0.52 |

With site index (rows that have it, n=145,666 val / 140,097 test): H R2 0.779 / 0.784, MdAPE 10.6 / 10.6 (no-SI on same rows: 0.765, 11.4).
Subsets (test): multi-tree cohorts TPA R2 0.29, MdAPE 37.8, BA 12.6, H 15.1; single-tree cohorts TPA 23.4, BA 11.4, H 11.9;
no-saplings BA 10.6, H 11.2, QMD 9.7; with-saplings BA 23.2, H 41.6, TPA 70.4. Planted: TPA 31.2, BA 12.1, H 12.2; natural: 24.8, 11.5, 12.4.
Per-species test MdAPE % (TPA / BA / H): PITA 29.6/13.8/11.5, PIEL 26.3/11.4/11.7, LIST2 31.0/10.7/11.9, QUNI 30.5/12.5/12.4,
ACRU 38.6/10.2/14.1, NYBI 35.3/7.2/14.1, QULA3 32.8/11.5/14.9, LITU 14.6/7.2/9.8, PIPA2 21.8/11.8/11.7, QUAL 9.6/9.5/8.8; hardwood fallback `_H` 37.0/18.7/17.6.
Full table: translation_metrics.csv (`subset` = overall, sp:<key>, ...). Individual-tree checks (VAL): ln(DRYBIO) vs ln(DIA) R2 0.981 (MdAPE 11.7 %); Chapman-Richards H-D R2 0.768 (MdAPE 13.1 %).
Notes: TPA log-R2 is limited by the discreteness above (TPA of a one-tree cohort is a constant); predictions are geometric-mean (median-type), not conditional means; no smearing correction.

## 3. Priors (priors.csv, one row per species + `_S`, `_H`, `_ALL`)
Fitted on TRAIN plots unless noted. Plot-visit quantities use all live trees of the plot (TPA_UNADJ, DIA>0).
(a) Tree biomass `ln(DRYBIO_AG lb) = alpha + beta ln(DIA in)`; (b) Chapman-Richards `H_m = 1.3 + a (1-exp(-b D_cm))^c`.
| key | alloc_alpha | alloc_beta | alloc_r2 | hd_a | hd_b | hd_c | hd_rmse_m |
|---|---|---|---|---|---|---|---|
| PITA | -0.123 | 2.74 | 0.965 | 36.2 | 0.0314 | 1.15 | 3.07 |
| PIEL | 0.0671 | 2.75 | 0.972 | 29.4 | 0.053 | 1.48 | 3.01 |
| PIPA2 | -0.0738 | 2.81 | 0.982 | 28.4 | 0.05 | 1.42 | 3.03 |
| LIST2 | 0.141 | 2.69 | 0.994 | 35.9 | 0.033 | 0.987 | 3.02 |
| QUNI | 0.953 | 2.47 | 0.987 | 29.2 | 0.0319 | 0.818 | 3.01 |
| ACRU | 0.977 | 2.34 | 0.989 | 29.9 | 0.0276 | 0.743 | 2.9 |
| LITU | 0.316 | 2.61 | 0.992 | 39.3 | 0.0293 | 0.897 | 3.32 |
| QUAL | 0.546 | 2.67 | 0.983 | 34.2 | 0.0345 | 1.01 | 3.2 |
| NYBI | 0.465 | 2.5 | 0.997 | 28.0 | 0.0368 | 0.883 | 3.13 |
| QULA3 | 0.887 | 2.49 | 0.993 | 30.7 | 0.0241 | 0.78 | 3.21 |
| _S | 0.19 | 2.63 | 0.962 | 33.3 | 0.0355 | 1.15 | 3.24 |
| _H | 0.634 | 2.5 | 0.968 | 32.8 | 0.0234 | 0.778 | 4.34 |
| _ALL | 0.521 | 2.52 | 0.963 | 33.3 | 0.0264 | 0.873 | 3.97 |

(c) Reineke SDI = TPA_ha x (QMD_cm/25.4)^1.605 (metric per hectare, 10 in reference; /2.471 for imperial per acre), computed on trees >= 5 in per plot-visit with >= 5 such trees;
`sdimax_*` = quantiles over plot-visits where the species has >= 60 % of BA (pooled S/H: >= 80 % group BA); `nvisits` counts plot-visits (not independent plots).
P95 is the suggested SDImax; the observed maxima are inflated by plot-expansion artefacts. Relative density RD = plot SDI / BA-share-weighted P95. `sdimax_ratio_to_all_p95` = species P95 / P95 of all plots.
A pooled empirical boundary fit (binned P95 of ln TPA vs ln QMD) gives slope -0.98, much shallower than -1.605 (small plots, discrete trees, mixed stands) -- the fixed Reineke slope is retained, but treat SDImax as a soft, low bound.
| key | sdimax_p90_all | sdimax_p95_all | sdimax_p99_all | sdimax_max_all | sdimax_nvisits_all | sdimax_ratio_to_all_p95 |
|---|---|---|---|---|---|---|
| PITA | 597.0 | 675.0 | 806.0 | 1010.0 | 6600.0 | 0.963 |
| PIEL | 529.0 | 610.0 | 744.0 | 980.0 | 4950.0 | 0.87 |
| PIPA2 | 395.0 | 451.0 | 596.0 | 756.0 | 866.0 | 0.644 |
| LIST2 | 459.0 | 554.0 | 799.0 | 872.0 | 281.0 | 0.791 |
| QUNI | 382.0 | 480.0 | 661.0 | 810.0 | 313.0 | 0.684 |
| ACRU | 420.0 | 492.0 | 603.0 | 626.0 | 173.0 | 0.701 |
| LITU | 720.0 | 871.0 | 969.0 | 1000.0 | 235.0 | 1.24 |
| QUAL | 464.0 | 527.0 | 622.0 | 648.0 | 123.0 | 0.751 |
| NYBI | 805.0 | 948.0 | 1120.0 | 1280.0 | 399.0 | 1.35 |
| QULA3 | 438.0 | 491.0 | 662.0 | 801.0 | 419.0 | 0.7 |
| _S | 570.0 | 656.0 | 827.0 | 1320.0 | 12200.0 |  |
| _H | 617.0 | 730.0 | 998.0 | 1690.0 | 10500.0 |  |
| _ALL | 603.0 | 701.0 | 932.0 | 1810.0 | 33700.0 |  |

(d) Diameter increment (remeasured live trees, consecutive visits, 2-12 yr, same SPCD; 613,796 pairs incl. deaths; shrinkage clamped to 0, |dD| outliers dropped):
`dD [cm/yr] = exp(b0 + b1 ln(D/20) + b2 D/20 + b3 ln(age/30) - k_bal x BAL/10)` (D in cm, BAL = m2/ha of larger live trees on the plot; NLS on increments, LM, <=150k pairs).
Adding stand BA gave a collinear, unstable coefficient and no gain (see `g_full_*`), so it is dropped. `g_sigma_ln_pos` = SD of ln(observed/predicted) among positive increments (multiplicative scatter);
`g_zero_frac` = share of non-growing trees. DBH-class increment percentiles: priors_dgrowth_by_dbh.csv. Val R2 is low (0.05-0.29) as is typical for single-interval FIA increments.
| key | g_b0 | g_b1_lnD | g_b2_D | g_b3_lnage | g_k_bal | g_sigma_ln_pos | g_zero_frac | g_mean_cm_yr | g_val_r2 | g_val_n |
|---|---|---|---|---|---|---|---|---|---|---|
| PITA | 0.179 | -0.287 | -0.157 | 0.127 | 0.417 | 0.692 | 0.0288 | 0.675 | 0.244 | 52000.0 |
| PIEL | 0.226 | -0.19 | -0.447 | 0.14 | 0.444 | 0.737 | 0.0495 | 0.542 | 0.251 | 34000.0 |
| PIPA2 | 0.77 | 0.404 | -1.21 | -0.0615 | 0.56 | 0.782 | 0.0567 | 0.424 | 0.285 | 3760.0 |
| LIST2 | -0.734 | -0.0634 | -0.0462 | 0.107 | 0.286 | 0.862 | 0.14 | 0.301 | 0.0687 | 12500.0 |
| QUNI | -0.351 | 0.127 | -0.139 | 0.208 | 0.2 | 0.77 | 0.0392 | 0.434 | 0.136 | 7370.0 |
| ACRU | -0.435 | 0.32 | -0.297 | 0.134 | 0.259 | 0.837 | 0.134 | 0.283 | 0.0953 | 8080.0 |
| LITU | -0.413 | -0.213 | -0.0451 | 0.27 | 0.255 | 0.863 | 0.0521 | 0.422 | 0.0958 | 4430.0 |
| QUAL | -0.768 | 0.0795 | -0.0808 | 0.236 | 0.336 | 0.693 | 0.0692 | 0.307 | 0.145 | 3760.0 |
| NYBI | -1.66 | -0.196 | 0.181 | 0.2 | 0.11 | 0.76 | 0.209 | 0.169 | 0.052 | 10400.0 |
| QULA3 | -0.531 | -0.000333 | 0.0144 | 0.251 | 0.156 | 0.794 | 0.0469 | 0.449 | 0.139 | 5170.0 |
| _S | 0.259 | 0.117 | -0.506 | 0.0313 | 0.46 | 0.849 | 0.0856 | 0.475 | 0.178 | 105000.0 |
| _H | -0.848 | -0.0542 | 0.0191 | 0.177 | 0.247 | 0.844 | 0.142 | 0.287 | 0.093 | 91600.0 |
| _ALL | -0.0848 | 0.153 | -0.363 | 0.101 | 0.406 | 0.88 | 0.121 | 0.356 | 0.153 | 196000.0 |

(e) Mortality: tree alive at one visit -> dead (`tree_fate == dead`, harvest excluded) at the next; binomial regression with complementary log-log link and offset ln(dt):
`annual hazard h = exp(c0 + c1 ln(D/20) + c2 D/20 + c3 RD + c4 BAL/10 + c5 RD ln(D/20))`, `P(die in dt) = 1 - exp(-dt h)`; annual p = 1 - exp(-h) (= logistic-type model made interval-consistent).
Natural + disturbance (fire, storm, insects) deaths are mixed; discrimination is modest (AUC 0.61-0.72) but calibration is good (obs/pred ~ 1). The empirical table priors_mortality_table.csv
(hazard by RD class x DBH class per species) is a more faithful prior than the parametric RD terms, whose signs are confounded with BAL and size (hazard is highest for small trees at high RD, ~2 % -> 7 % /yr for PITA < 12.7 cm).
| key | m_c0 | m_c1_lnD | m_c2_D | m_c3_RD | m_c4_BAL | m_c5_RDxlnD | mort_annual_rate_train | mort_val_auc | mort_val_obs_over_pred |
|---|---|---|---|---|---|---|---|---|---|
| PITA | -6.22 | -1.15 | 1.47 | -0.817 | 0.62 | -0.556 | 0.0124 | 0.712 | 0.957 |
| PIEL | -5.58 | -1.01 | 0.83 | 0.412 | 0.212 | -0.447 | 0.0148 | 0.654 | 0.958 |
| PIPA2 | -6.39 | -0.873 | 1.47 | -0.8 | 0.456 | -2.32 | 0.0081 | 0.648 | 1.12 |
| LIST2 | -5.16 | -0.9 | 0.908 | -1.07 | 0.319 | -0.0225 | 0.0147 | 0.658 | 0.96 |
| QUNI | -5.09 | -0.805 | 0.82 | -0.0483 | 0.146 | -0.222 | 0.0196 | 0.637 | 0.956 |
| ACRU | -4.83 | -0.67 | 0.833 | -0.285 | 0.132 | 0.0562 | 0.0206 | 0.612 | 1.03 |
| LITU | -5.11 | -0.765 | 0.895 | -2.28 | 0.618 | -0.432 | 0.012 | 0.705 | 0.942 |
| QUAL | -6.87 | -1.35 | 1.25 | -0.761 | 0.56 | -0.332 | 0.00665 | 0.718 | 1.18 |
| NYBI | -5.19 | -0.928 | 0.92 | -0.976 | 0.136 | -0.508 | 0.0119 | 0.614 | 0.845 |
| QULA3 | -5.4 | -0.917 | 0.731 | 0.727 | -0.0239 | -0.149 | 0.0172 | 0.644 | 1.06 |
| _S | -5.26 | -0.668 | 0.633 | 0.253 | 0.0822 | -0.683 | 0.0135 | 0.668 | 0.994 |
| _H | -4.46 | -0.659 | 0.383 | -0.434 | 0.0854 | -0.143 | 0.017 | 0.644 | 0.965 |
| _ALL | -4.73 | -0.706 | 0.449 | -0.203 | 0.089 | -0.196 | 0.0157 | 0.651 | 0.961 |

(f) Height increment (|dH| <= 2-3 m/yr filter; HT is measured to 1 ft so very noisy): `dH [m/yr] = exp(h0 + h1 ln(H/15) + h2 ln(age/30) - k_bal BAL/10)`.
`hg_val_r2_via_HD` = R2 of dH implied by the Chapman-Richards derivative times the observed dD (alternative using the H-D curve).
| key | hg_h0 | hg_h1_lnH | hg_h2_lnage | hg_k_bal | hg_mean_m_yr | hg_val_r2 | hg_val_r2_via_HD |
|---|---|---|---|---|---|---|---|
| PITA | -0.38 | -0.763 | 0.183 | 0.0571 | 0.625 | 0.224 | 0.0439 |
| PIEL | -0.503 | -0.893 | 0.221 | 0.0609 | 0.558 | 0.241 | 0.176 |
| PIPA2 | -0.991 | -0.977 | 0.2 | -0.0141 | 0.363 | 0.18 | 0.293 |
| LIST2 | -0.848 | -1.05 | 0.385 | 0.164 | 0.27 | 0.0535 | 0.123 |
| QUNI | -1.03 | -1.13 | 0.437 | 0.0735 | 0.274 | 0.0464 | 0.069 |
| ACRU | -1.25 | -1.16 | 0.43 | 0.18 | 0.181 | 0.0383 | 0.0499 |
| LITU | -0.44 | -1.55 | 0.71 | 0.144 | 0.305 | 0.0938 | 0.105 |
| QUAL | -1.25 | -1.25 | 0.893 | 0.168 | 0.236 | 0.0653 | -0.00499 |
| NYBI | -1.51 | -1.44 | 0.771 | 0.00354 | 0.146 | 0.00778 | -0.0208 |
| QULA3 | -1.17 | -0.812 | 0.318 | 0.111 | 0.248 | 0.0398 | 0.0598 |
| _S | -0.577 | -0.72 | 0.146 | 0.191 | 0.446 | 0.139 | 0.166 |
| _H | -1.35 | -0.475 | 0.178 | 0.118 | 0.205 | 0.0201 | 0.0459 |
| _ALL | -0.873 | -0.507 | 0.106 | 0.247 | 0.294 | 0.0591 | 0.167 |

## 4. Caveats
* DRYBIO_AG is itself a model of DIA (and HT); the allometry R2 reflects FIA's equations, not independent measurement.
* Cohort TPA is design-discrete (see section 1); quote TPA errors with that in mind. A sapling-aware split (macroplot vs microplot cohorts) would be the next improvement.
* SI is available for ~54 % of cohorts only; SI units/base ages are mixed (cond_sibase/cond_sisp ignored).
* Mortality includes harvest-free natural + disturbance; thinned plots appear as `harvest_*` and are excluded from the mortality pairs but still enter BAL/RD/SDI as measured.
* SDImax uses all plots (train+val+test) for the headline `_all` columns (`_train` columns also given); growth/mortality/height fits use train only and report val.
* No git commit was made.
