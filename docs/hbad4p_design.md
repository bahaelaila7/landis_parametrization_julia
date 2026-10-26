# HBAD4P — stand-level, process-based, climate-sensitive cohort model (design)

User request 2026-10-26: HBAD3P is calibrated on tracked trees, so stand-level quantities (stems, BA, ingrowth) are never in its objective and
their errors accumulate in free runs (nested CV: N≥12.7 cm −14.8 %; small-tree crossing of 12.7 cm 0.71 × observed; recruitment from seed presence
only). Build a model whose cohort is **species × diameter class**, tracking **height, basal area and density**, still process-based and climate
sensitive, with **FVS-style tripling** (0.5 of a cohort at the mean dynamics, 0.25 damped, 0.25 amplified). User choices: FIA 2-inch classes;
calibrated on **stand tables only**; climate = **water + temperature + CO2**; built and validated on **GA+FL with the campaign's 5-fold nested CV**
first, Region 4 after. Branch `hbad4p` (from hbad3-ncv 8d78254), worktree `landis_hbad4`.

## 1. State
Site = **plot** (the 4 subplots pooled): the model is applied to stands, so it is calibrated at the stand scale it is used at (HBAD3P competes
per subplot; applying a subplot-calibrated competition to a stand average is a Jensen-type aggregation bias).
Per site, species key k (as HBAD3P: the region's top keys + `_H` / `_S`), diameter class j, vigour lane v:

| | |
|---|---|
| classes j = 1…16 | FIA 2-inch classes: [1, 3) in = [2.54, 7.62) cm, [3, 5), …, [29, 31) in, last ≥ 31 in (78.74 cm). 12.7 cm (5 in, the microplot / subplot tally boundary) is a class boundary |
| lanes v | mid (weight 0.5, z = 0), damped (0.25, z = −√2), amplified (0.25, z = +√2): same mean and variance of the growth deviation z σv as N(0, σv²). Option `HBAD4P_TRIPLE=fvs`: FVS SN weights 0.60 / 0.25 / 0.15 at z = −0.142, +1.271, −1.549 (sn/dgdriv.f) |
| per (k, j, v) | N (trees / forested ha), B (basal area m² / forested ha; → class QMD Dq = √(B / (π/4 · N)) · 100), H (mean height m, N-weighted), A (mean age yr, carried for senescence and the BS / PnET translation) |

## 2. Annual step (all rates from the start-of-year state; reuses HBAD3P's process forms at cohort level)
1. **Competition**: SDI_c = N (Dq / 25.4)^1.605; RD = ΣSDI / sdimax; overtopping O_c = (Σ_{H > H_c} SDI + ½ Σ_{H = H_c} SDI) / sdimax.
2. **Growth** of the cohort's mean tree (HBAD3P): light L = f0 + (1 − f0) e^{−a O}; climate/vigour cw = W^ω · f_T · f_CO2 · e^{z σv};
   hydraulic hyd = max(0, 1 − min(H / 1.1 Hmax, 1)^θ); dV = max(ε (Dq/20)^b · 122.98 · L cw hyd − ρ Dq² H, 0); dD = dV / (2 Dq max(H, 1.37)).
   * water W = period AET / PET (HBAD3 water balance); f_T = exp(−((T_gs − T_opt,k) / wT)²), T_opt,k = a species TRAIT (the key's
     tree-weighted mean growing-season temperature over its Region 4 plots), wT global; f_CO2 = 1 + βC ln(C / C_ref), βC FIXED (not
     identifiable from 2000–2024 FIA: ≈ +15 % CO2, collinear with time) at 0.58 (FACE: +23 % NPP at 550 vs 370 ppm, Norby et al. 2005), C_ref = 2010.
3. **Upgrowth between classes (two moments)**: within a class the trees are distributed linearly on [L_j, U_j) with the mean matching the
   cohort's mean diameter (from N and B); all trees of a lane grow by the same dD (diameter growth of the lane's mean tree); the trees with
   D > U_j − dD move to class j+1 carrying their N, their own B (exact for the linear density), H and A. dD ≤ class width is asserted (≈ 5 cm/yr is
   far above any observed annual growth).
4. **Height** (HBAD3P): toward the H–D allometry H*(D) with the shade response (η, λ, μ, kh); movers keep their height and are N-weighted into the
   receiving cohort.
5. **Mortality** (HBAD3P hazard): carbon-starvation stress of realised relative growth, size, age / LONGEVITY senescence, drought anomaly
   (Wn − W); N and B scale together (mortality is size-neutral within a class); self-thinning when ΣSDI > sdimax.
6. **Lane mixing**: a fraction κ of each lane is re-split 0.5 / 0.25 / 0.25 per year (κ = 0: fully persistent lanes as FVS triples; κ is
   calibrated — HBAD2 found a persistent vigour correlation of 0.49 between remeasurements).
7. **Recruitment** into class 1 at D0 = 2.54 cm (+ the in-year growth): rate e^{lnF_k} × seed source (a cohort of k with D ≥ 12.7 cm in the
   site, else ε0) × understory light e^{−λ0 e^{λ1 (ST_k − 3)} RD} × W^ωR; split over the lanes 0.5 / 0.25 / 0.25. **Ingrowth is scored**
   (it is part of the class-1 / class-2 stand table), unlike HBAD3P, whose fecundity was fitted separately.
8. Age += 1.

## 3. Objective: stand tables only (user)
Per natural interval (FIA remeasurement pair, harvested intervals excluded; trees MISSING at v2 removed from v1 and v2) the v1 plot stand table
initialises the site (each observed tree → its class; per class N, B, H, A from the trees, split over the lanes), the model runs to v2
(fractional final year), and the v2 stand table is scored, per (k, j):
* **counts**: observed tally trees n_kj (subplot trees for D ≥ 12.7 cm, microplot saplings below) ~ NegBin(μ_kj = N_kj · area_j · f, r_j):
  area_j = 4 subplots of 1/24 ac (D ≥ 12.7) or 4 microplots of 1/300 ac, f = the forested fraction (HBAD3 forest table);
* **size**: log QMD of the observed trees of the cell vs the cohort's Dq, Gaussian, sd σD / √n (σD fitted);
* **height**: log mean measured height vs H, sd σH / √n.
The negative-binomial dispersion r_j (by tally) and σD, σH are global. The objective is the summed negative log-likelihood over the train
intervals (CMA-ES, `tools/hbad2/cmaes.jl`); val selects (best-val checkpoint); test is reported once per fold.

## 4. Evaluation (nested CV, comparable with the campaign)
Folds / splits = the campaign's (`landis_ncv/runs_local/ncv/prep/<role>/product/intervals.arrow` `set`). Per test fold:
one-interval stand-table NLL vs a persistence null (v1 table carried unchanged) and vs HBAD3P (same intervals); stand-level bias of N (all,
≥ 12.7 cm), BA, QMD, Lorey height, ingrowth into 12.7 cm; by species group and by size class; TOST (±5/10/20 %) on BA and N per group; free runs
along the plot's chained intervals. Pooled out-of-fold report like `tools/ncv/report.py`.

## 5. Verification (independent recomputation, as gv_verify / verify3p)
Unit checks: zero growth and zero mortality leave every cohort unchanged; N and B are conserved by the upgrowth transfer; the two-moment transfer
is exact for a linear density (vs a direct integration); the lane weights / nodes preserve mean and variance; the stand-table builder against
an SQL aggregation of curated_trees; the likelihood recomputed from its definition on sample intervals; determinism and threading parity.

## 6. Pan integration (after the GA+FL validation)
Plugin `hbad4p` on the fe-engine stack like hbad3p (water W from the scenario climate, plus T_gs and CO2 from `climate_scenario.jl`);
initial state from FIA plots or from the generative-vegetation stand cohorts (species × age slots with N, QMD, H → classes by QMD).
