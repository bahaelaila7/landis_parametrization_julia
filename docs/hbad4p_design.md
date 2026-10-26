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
| lanes v | mid (weight 0.5, z = 0), damped (0.25, z = −√2), amplified (0.25, z = +√2): same mean and variance of the growth deviation z σv as N(0, σv²). Option `HBAD4P_TRIPLE=fvs`: FVS SN weights 0.60 / 0.25 / 0.15 at z = −0.142, +1.271, −1.549 (sn/dgdriv.f), standardised to variance 1 (raw 0.776) so σv means the same; the store records the lanes it was built with and simulate! uses them |
| per (k, j, v) | N (trees / forested ha), B (basal area m² / forested ha; → class QMD Dq = √(B / (π/4 · N)) · 100), H (mean height m, N-weighted), A (mean age yr, carried for senescence and the BS / PnET translation) |

## 2. Annual step (all rates from the start-of-year state; reuses HBAD3P's process forms at cohort level)
1. **Competition**: SDI_c = N (Dq / 25.4)^1.605; RD = ΣSDI / sdimax; overtopping O_c = Σ SDI_c' r(H_c' − H_c) / sdimax with the height ramp
   r(Δ) = clamp(½ + Δ / 2δH, 0, 1), δH = 1 m (continuous in the heights; δH → 0 is HBAD3P's rule: taller count fully, equal heights and the own half).
   The ramp replaces the step at exact ties (heights recorded in whole feet gave identical recruit heights to several keys: a 1-ulp flip moved O by half a cohort).
2. **Growth** of the cohort's mean tree (HBAD3P): light L = f0 + (1 − f0) e^{−a O}; climate/vigour cw = W^ω · f_T · f_CO2 · e^{z σv};
   hydraulic hyd = max(0, 1 − min(H / 1.1 Hmax, 1)^θ); dV = max(ε (Dq/20)^b · 122.98 · L cw hyd − ρ Dq² H, 0); dD = dV / (2 Dq max(H, 1.37)).
   * water W = period AET / PET (HBAD3 water balance).
   * temperature f_T = exp(−[(T − Topt_k)² − (Tref_k − Topt_k)²] / wT²): T = the period mean ANNUAL temperature (EPA L4 mean, PRISM), Topt_k = the key's
     range optimum (tree-weighted annual normal over its plots: a named species over the whole product, a group key over the region's train plots),
     Tref_k = the key's mean period temperature over its train intervals. f_T = 1 at the key's typical regional climate, so ε keeps its meaning
     (growth at that climate) and f_T carries only deviations — spatial within the region, temporal, and scenario warming.
   * CO2 f_CO2 = 1 + βC ln(C / C_ref), βC FIXED at 0.58 (FACE: +23 % NPP at 550 vs 370 ppm, Norby et al. 2005; not identifiable from 2000–2024 FIA),
     C_ref = 390.1 ppm (2010, the Mauna Loa series PnET reads).
   * cw (incl. f_T, f_CO2) also enters the carbon-balance stress of mortality (gm), as W^ω does in HBAD3P: the realised carbon balance.
3. **Upgrowth between classes (two moments, truncated-exponential closure)**: within [L_j, U_j) the trees are distributed ∝ e^{a ξ}, ξ = (D − L_j)/w,
   with a matched to the cohort's mean square diameter (from N and B). Any QMD in (L_j, U_j) is representable (a linear density with a non-negative
   slope bound spans only the central third of a class: 55 % of the real v1 cells needed more; audit round 1, B1). All trees of a lane grow by the same dD;
   each ξ segment lands in the class t ≥ j it reaches — any number of classes — carrying its N and its exact basal area; N and B are conserved.
4. **Height** (HBAD3P): toward the H–D allometry H*(D) with the shade response (η, λ, μ, kh). Movers and stayers get the allometric offset
   H*(D_group) − H*(D_cohort), rescaled to conserve Σ N·H (the bigger trees of a cohort are the taller ones).
5. **Mortality** (HBAD3P hazard): carbon-starvation stress of realised relative growth, size, age / LONGEVITY senescence, drought anomaly
   (Wn − W); N and B scale together (mortality is size-neutral within a class); self-thinning when ΣSDI > sdimax.
6. **Lane mixing**: a fraction κ of each lane is re-split 0.5 / 0.25 / 0.25 per year (κ = 0: fully persistent lanes as FVS triples; κ is
   calibrated — HBAD2 found a persistent vigour correlation of 0.49 between remeasurements).
7. **Recruitment** into class 1 at D0 = 2.54 cm at the END of the year (no growth in its entry year; aged 4.5 then): rate e^{lnF_k} × seed source ×
   understory light e^{−λ0 e^{λ1 (ST_k − 3)} RD} (RD of the start of the year) × W^ωR; split over the lanes. Seed source: a tree of the key ≥ 12.7 cm on
   the plot at v1, or ≥ 1 tree / forested ha of the key ≥ 12.7 cm now (else ε0) — a physical threshold, so that thin tails of the closure are no seed
   trees. Ingrowth is scored (it is part of the class-1 / class-2 stand table), unlike HBAD3P, whose fecundity was fitted separately.
8. Age += 1.

## 3. Objective: stand tables only (user)
Per natural interval (FIA remeasurement pair, harvested intervals excluded; trees MISSING at v2 removed from v1 and v2) the v1 plot stand table
initialises the site and the model runs to v2 (fractional final year); the v2 stand table is scored per (k, j).
* **v1 state**: each tree → its class; N per forested ha of its tally (subplot fraction for subplot trees, microplot fraction for saplings, by the
  tree's expansion weight); B, H, A from the trees; split over the lanes. **Sapling layer (classes 1–2)**: the microplots sample 8 % of the subplot area,
  so a v1 sapling count is a noisy sample of the stand's sapling density (46 % of the train ingrowth ≥ 12.7 cm came from keys without a v1 microplot
  sapling; audit B3). The state takes the empirical-Bayes posterior mean density N̂ = (n + α) / (e + α / m), e = microplot area × forested fraction,
  with a negative-binomial prior per (key, class, adult of the key on the plot) fitted on the TRAIN plots (m = Σn/Σe, α by profile likelihood); an
  unobserved cell gets the key's train mean QMD / height / age of saplings.
* **counts**: v2 tally trees n_kj (classes 1–2 on the microplots, ≥ 3 on the subplots; each tree in the class of its v2 diameter, as FIA re-tallies a tree
  that shrank below 5 in as a sapling) ~ NegBin(μ_kj = (N_kj + bg_j) · area_j · f, r_j). area_j from FIA's expansion factors (TPA_UNADJ 6.018046 /
  74.965282 per ac); f = the forested fraction of that tally at v2. **bg_j** = background densities (microplot / subplot tallies; fitted): immigration,
  misidentification, recording error — what the model leaves empty — instead of an arbitrary floor.
* **size**: log QMD of the cell's observed trees vs the cohort's Dq, Gaussian, variance sD_j² / n + σD² (sD_j = the TRAIN within-cell sd of log D of
  class j: 0.29 in class 1 to 0.02 in class 13 — a single σ was off by a factor of 12 across classes; σD = model error, fitted).
* **height**: log mean measured height vs H, variance sH_j² / n_H + σH².
The objective is the summed negative log-likelihood over the train intervals (CMA-ES, `tools/hbad2/cmaes.jl`); val selects; test is reported once per fold.
The persistence null (v1 table with the same sapling layer carried unchanged) gets its own fitted observation parameters.

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

## 7. Audit round 1 (2026-10-26, independent Opus review, 42 PASS / 23 FAIL; fixed)
Blocking: the linear two-moment closure (B1, replaced by the truncated exponential: 0 of 1.1 M cells now outside their class vs 10,843), seed-source "dust"
(B2, physical threshold), the likelihood floor and the unobserved sapling layer (B3: empirical-Bayes sapling prior, fitted background; the persistence null on
the same basis). Important: height-tie discontinuity (I1, ramp), dD > class width (I2, multi-class transfer; ln ε ≤ 4), one σD for all classes (I3, per-class
sampling spread), the temperature response confounded with ε (I4, centred at Tref; group-key Topt from the region), the warm start losing HBAD3P's gR / φ
(I5, ε rescaled × 1.08 at k1), movers' heights and climate in gm (I6 / I7, documented above). Minor: dclass at exact bounds, shrunk trees' tally, tally areas
from TPA, recruit age, :fvs lanes standardised + one lane switch recorded in the store, train-only median age, the cache key (code + switches + input files).
Known, not changed: 98 trees recorded dead at v1 but alive at v2 are in neither table (product); the open last class grows B with E[D] ≈ QMD; recruitment
uses RD of the start of the step but the seed status after it; HBAD4P's val is the role's inner val + surplus train plots of the campaign split (clean of
test, but spatially interleaved with train: best-val selection is optimistic — test is the report).
