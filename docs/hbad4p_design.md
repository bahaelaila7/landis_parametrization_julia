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
| per (k, j, v) | up to NATOM = 16 diameter ATOMS (N_m trees / forested ha at D_m): N = Σ N_m, B = Σ N_m π/4 D_m² (→ class QMD Dq), H (mean height m, N-weighted), A (mean age yr, for senescence and the BS / PnET translation) |
| storage | every (k, j, v) has its cell totals (N, B, H, A); the initial atoms are stored only for the cells occupied at v1 (CSR per site), the working atoms of a site live in the per-thread scratch while the site runs, and a step visits the occupied cells only (audit round 3 B1 / I5: 16 atoms in a dense layout would be 4.4 GB for the train set) |

## 2. Annual step (all rates from the start-of-year state; reuses HBAD3P's process forms at cohort level)
1. **Competition**: SDI_c = N (Dq / 25.4)^1.605; RD = ΣSDI / sdimax; overtopping O_c = Σ SDI_c' r(H_c' − H_c) / sdimax with the height ramp
   r(Δ) = clamp(½ + Δ / 2δH, 0, 1), δH = 1 m (continuous in the heights; δH → 0 is HBAD3P's rule: taller count fully, equal heights and the own half).
   The ramp replaces the step at exact ties (heights recorded in whole feet gave identical recruit heights to several keys: a 1-ulp flip moved O by half a cohort).
2. **Growth** (HBAD3P forms): light L = f0 + (1 − f0) e^{−a O}; climate/vigour cw = W^ω · f_T · f_CO2 · e^{z σv}; hydraulic hyd = max(0, 1 − min(H / 1.1 Hmax, 1)^θ)
   per cell (cell height and competition); the size terms per ATOM (audit round 3 I4: per cell from Dq the within-class size dependence was lost, +9 % ingrowth
   into 12.7 cm vs per-atom rates): dV_m = max(ε (D_m/20)^b · 122.98 · L cw hyd − ρ D_m² H, 0); dD_m = dV_m / (2 D_m max(H, 1.37)) (one table lookup per atom).
   * water W = period AET / PET (HBAD3 water balance).
   * temperature f_T = exp(βT (T − Tref_k)): T = the period mean ANNUAL temperature (EPA L4 mean, PRISM), Tref_k = the key's tree-weighted mean of the
     same variable over its train intervals; βT global, either sign (audit round 2 B2: a quadratic around a Region-4 "optimum" could only slow growth on
     warmer sites). f_T = 1 at the key's typical regional climate, so ε keeps its meaning and f_T carries only deviations (spatial, temporal, scenarios).
   * CO2 f_CO2 = 1 + βC ln(C / C_ref), βC FIXED at 0.58 (FACE: +23 % NPP at 550 vs 370 ppm, Norby et al. 2005; not identifiable from 2000–2024 FIA),
     C_ref = 390.1 ppm (2010, the Mauna Loa series PnET reads).
   * f_CO2 multiplies the growth only (the FACE constraint is on NPP); the carbon-balance stress of mortality (gm) and the self-thinning weight use the
     realised relative growth without it (light, water, temperature, vigour).
3. **Upgrowth between classes (cohorts in bins)**: every atom of a cell grows by the cell's dD and lands in the class of its new diameter (any number of
   classes); the transfer is EXACT for the atoms — no within-class density is assumed, so a cell of 1–3 tallied trees does not diffuse (audit round 2 B1: any
   closure smeared sparse cells over the class, +19 % crossings into 12.7 cm over 5 yr). A cell with more than NATOM atoms merges its two closest (N and N·D²
   conserved: N and B are conserved exactly). Merging (audit round 3 B1 / I1a): the atoms are sorted by D and the closest ADJACENT pair merges (in 1-D the
   closest pair is adjacent) while the cell holds more than NATOM; distances are binned at 1e-6 cm and ties go to the first pair in D order (a merge choice
   decided by rounding made the objective rough); atoms closer than 1e-6 cm (duplicates: recruits at 2.54 cm, lanes mixed back) always merge. Every atom
   grows by its own dD_m and dies at its own rate (§2.2, §2.5); light, climate, vigour, height and the thinning weight stay per cell.
4. **Height** (HBAD3P): toward the H–D allometry H*(D) with the shade response (η, λ, μ, kh); the allometric increment H*(Dq') − H*(Dq) with Dq' the QMD of the
   cell's grown atoms (start-of-year N). Movers and stayers get the allometric offset
   H*(D_group) − H*(D_cohort), rescaled to conserve Σ N·H (the bigger trees of a cohort are the taller ones).
5. **Mortality** (HBAD3P hazard): h_m = fM [m0/LONGEVITY + ms e^{σST (ST − 3)} e^{−β gm} (D_m/20)^ζ] + e^{mA (A/LONGEVITY − 1)}: carbon-starvation stress of the
   realised relative growth gm (per cell), size (per ATOM, audit round 3 I4), age / LONGEVITY senescence, drought anomaly (Wn − W) in fM; self-thinning when
   ΣSDI > sdimax (per cell, the excess removed ∝ SDI (1 − g)).
6. **Lane mixing**: a fraction κ of each lane is re-split 0.5 / 0.25 / 0.25 per year (κ = 0: fully persistent lanes as FVS triples; κ is
   calibrated — HBAD2 found a persistent vigour correlation of 0.49 between remeasurements). Lane v's atoms = its own atoms ONCE with weight 1 − κΔ + κΔ w_v
   + the other lanes' atoms with κΔ w_v (audit round 3 B1: adding the own atoms twice, as (1 − κΔ) and κΔ w_v, doubled every lane's atoms each year and the
   merges dominated κ's apparent cost). κ is linear on [0, 1] (κ = 0 reachable) and starts at 0 (§7 warm start).
7. **Recruitment** into class 1 at D0 = 2.54 cm at the END of the year (no growth in its entry year; aged 4.5 then): rate e^{lnF_k} × seed source ×
   understory light e^{−λ0 e^{λ1 (ST_k − 3)} RD} (RD of the start of the year) × W^ωR; split over the lanes. Seed source: a tree of the key ≥ 12.7 cm on
   the plot at v1, or ≥ 1 tree / forested ha of the key ≥ 12.7 cm now (else ε0) — a physical threshold, so that a trickle of survivors (a dying cohort, the
   lanes' tails, background recruits grown on) below one tree per hectare is no seed source. Ingrowth is scored (it is part of the class-1 / class-2 stand
   table), unlike HBAD3P, whose fecundity was fitted separately.
8. Age += 1.
9. **Planted stands** (the product's `planted`, FIA STDORGCD = 1): growth × e^{pD} (fD in dV), height increment × e^{pH} (fH), the non-senescence hazard × e^{pM}
   (in fM) — HBAD3P's planted factors, carried over by the warm start.

## 3. Objective: stand tables only (user)
Per natural interval (FIA remeasurement pair, harvested intervals excluded; trees MISSING at v2 removed from v1 and v2; intervals whose v2 visit has NO forested
subplot — a land-use change inside a "natural" interval, FIA tallies no tree there — excluded, audit round 3 I6: 102 intervals, 67 of them train at k1; likewise those whose v1 visit has no forested subplot — no observed v1 stand, audit round 4
M1 / decision H10: 24 intervals, 18 train) the v1 plot stand table
initialises the site and the model runs to v2 (fractional final year); the v2 stand table is scored per (k, j).
* **v1 state**: each tree → its class; N per forested ha of its tally (subplot fraction for subplot trees, microplot fraction for saplings, by the
  tree's expansion weight); B, H, A from the trees; split over the lanes. **Sapling layer (classes 1–2)**: the microplots sample 8 % of the subplot area,
  so a v1 sapling count is a noisy sample of the stand's sapling density (46 % of the train ingrowth ≥ 12.7 cm came from keys without a v1 microplot
  sapling; audit B3). The state takes the empirical-Bayes posterior mean density N̂ = (n + α) / (e + α / m), e = microplot area × forested fraction,
  with a negative-binomial prior per (key, class, adult of the key on the plot) fitted on the TRAIN plots ((m, α) by maximum likelihood, α by profile
  likelihood over (α, m); round 2 had m = Σn/Σe, the Poisson MLE); an
  unobserved cell gets the key's train mean QMD / height / age of saplings. A plot whose four microplots are all non-forest has no sapling tally: e = 0, its
  sapling cells take the prior mean m, and it is not a zero count in the prior's fit (audit round 4 M1: f was set to 1.0, i.e. an unobserved microplot was
  scored as an empty full one).
* **counts**: v2 tally trees n_kj (classes 1–2 on the microplots, ≥ 3 on the subplots; each tree in the class of its v2 diameter, as FIA re-tallies a tree
  that shrank below 5 in as a sapling) ~ NegBin(μ_kj = (N_kj + bg_j) · area_j · f, r_j). area_j from FIA's expansion factors (TPA_UNADJ 6.018046 /
  74.965282 per ac); f = the forested fraction of that tally at v2 (0 when every known microplot / subplot is non-forest: that tally is unobserved and
  scores nothing, round 4 M1). **bg_j** = background densities (microplot / subplot tallies; fitted): immigration, misidentification, recording error — what
  the model leaves empty — instead of an arbitrary floor.
* **observation space** (audit round 3 I1b; the LIKELIHOOD only, the dynamics are unchanged): every atom is measured with the FIA DBH error N(0, sdM²),
  sdM = 0.2 cm fixed (`SD_DBH`, params4.jl): it contributes to class t with probability Φ((ub_t − D)/sdM) − Φ((lb_t − D)/sdM) (lb of class 1 = −∞: the 1-in
  tally threshold is the model's entry definition — recruits enter AT 2.54 cm and would otherwise be half untallied; ub of class 16 = ∞). The predicted
  table per (k, j) is N_kj = Σ p N, B_kj = Σ p N π/4 D², H_kj = Σ p N H_m / N_kj over the key's atoms (all lanes), H_m = H + H*(D_m) − H*(Dq) the atom's
  allometric height around its cell's height (round 4 I1: with the cell mean an atom crossing a class bound carried its new cell's height into the blurred
  tail of its old class); an atom approaching a class bound moves into the next class's prediction smoothly instead of in one jump. The nulls are tables in
  observation space already and are scored unblurred.
* **size** (every observed cell, n ≥ 1, of a tallied area): log QMD of the cell's observed trees ~ the MIXTURE w · N(log of the predicted (blurred) QMD,
  sD_j² / n + σD²) + (1 − w) · N(mD_j, sDb_j²), w = N_kj / (N_kj + bg_j) (sD_j = the TRAIN within-cell sd of log D of class j: 0.29 in class 1 to 0.02 in
  class 13 — a single σ was off by a factor of 12 across classes; classes with < 30 df the uniform-spread value; σD = model error, fitted).
* **height** (every observed cell with a measured height): log mean measured height ~ w · N(log H_kj, sH_j² / n_H + σH²) + (1 − w) · N(mH_kj, sHb_j²) (sH_j
  likewise; classes with < 30 df pool the neighbouring classes, audit round 3 — was a flat 0.1).
* **the mixture** (audit round 4 I1, decision H7): under the count model a tallied tree of the cell is the model's with probability N / (N + bg), so the
  fitted count background bg is the pseudo-count of background trees and w = N / (N + bg) the model's share; the background density is a fixed TRAIN data
  constant (`obs_background`, ObsBG: per class the mean / sd of log QMD over all train cells; per key × class the mean log height, its per-class sd). Before,
  only cells with N > 1e-9 were scored: a cell's first trace prediction (a blur tail at a class bound) switched a −1.5 … +13-nat term on, and every model or
  null scored its own set of cells (val: 1,477 observed cells unscored for the warm start, 72 for the TM null), so totals were not comparable between them.
  Now every model and null scores the same cells, and its totals are comparable.
The objective is the summed negative log-likelihood over the train intervals (CMA-ES, `tools/hbad2/cmaes.jl`); val selects; test is reported once per fold.
Baselines, each with its own fitted observation parameters: the **transition-matrix null** (audit round 3 I3; data4.jl `tm_null` / `tm_state!`: the TRAIN
fates per key × v1 class × dt tercile — the fraction of the v1 density alive at v2 in each v2 class and its basal area — plus the train ingrowth density per
key × class × v1 seed presence per year; heights the train mean per key × class) is the bar HBAD4P must beat on val COUNTS; the persistence null (the v1
table with the same sapling layer carried unchanged) is reported but is not the bar (the transition matrix gains 60 % of its val NLL at k1).

## 4. Evaluation (nested CV, comparable with the campaign)
Folds / splits = the campaign's (`landis_ncv/runs_local/ncv/prep/<role>/product/intervals.arrow` `set`). Per test fold:
one-interval stand-table NLL vs the train-estimated transition-matrix null (§3; the bar on val / test counts), the persistence null (reported) and HBAD3P
(same intervals); stand-level bias of N (all,
≥ 12.7 cm), BA, QMD, Lorey height, ingrowth into 12.7 cm; by species group and by size class; TOST (±5/10/20 %) on BA and N per group; free runs
along the plot's chained intervals. Pooled out-of-fold report like `tools/ncv/report.py`.

## 5. Verification (independent recomputation, as gv_verify / verify3p)
Unit checks: zero growth and zero mortality leave every cohort unchanged; N and B are conserved by the upgrowth transfer; the atom transfer is exact (a
tree-level replay of the real v1 trees) and the merges are checked against 64 atoms (incl. their κ dependence) and the objective's roughness; the lane
weights / nodes preserve mean and variance; the stand-table builder against
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
Known, not changed: 98 trees recorded dead at v1 but alive at v2 are in neither table (product); recruitment uses RD of the start of the step but the seed
status after it. (Round-3 corrections of this paragraph: "the open last class grows B with E[D] ≈ QMD" is obsolete — since round 2 the atoms of the open class
grow exactly; "HBAD4P's val is the role's inner val + surplus train plots" is obsolete — since round 2 I5 the sets are the role split's whole outer folds,
at k1 val = fold 2 only, train = folds 3–5, test = fold 1. The HBAD3P warm start, however, was SELECTED on the product's own val set, i.e. on HBAD4P's val
fold plus surplus train plots: a mild leak into the choice of the warm start only — HBAD4P's own fit and selection do not see it; test is the report.)

## 8. Audit round 2 (2026-10-26, independent Opus review: 56 PASS / 9 FAIL; fixed)
B1 closure diffusion on sparse real cells → cohorts in bins (atoms); B2 temperature sign → βT (T − Tref), Tref on the kernel's variable; I1 movers escaped
self-thinning → inflows carry the source cell's realised growth; I2 warm start → tree-weighted gR / φ factor at the subplot RD (×1.18 at k1); I5 train = the role's
3 folds (HBAD4_SPLIT_CSV = folds/split_<role>.csv; the product's `set` had moved the train cap's surplus to val): k1 train 10,475 intervals; I6 CO2 on growth
only; minor: inactive cells keep their trees on inflow, allocation-free overtopping sort, physical within-class spread for classes with < 30 df, tree-weighted
LONGEVITY of group keys, traits from the product dir. Open: I3 — report against a train-estimated transition-matrix null (verify4p `tm_state!`); I4 — 77 % of
v2 tally trees are remeasured individuals, the NegBin / per-cell-mean observation model ignores it (kept: the user chose stand tables only); I7 — the curated
tree table misses ~half the saplings in AL/GA/NC/SC/TN (RECONCILECD exclusion) — fixed upstream (FIA_DATA_PREP branch curation-fix), HBAD4P to be rebuilt on it.

## 9. Audit round 3 (2026-10-26, independent Opus review of 1479df8: 60 PASS / 6 FAIL; fixed — the updated verifier: 69 PASS / 0 FAIL / 20 INFO on k1)
Verifier `tools/hbad4/verify4p.jl` (+ `verify4p_ref.jl`, updated with the fixes), fold k1 (train 10,408 intervals after I6). Numbers at the HBAD3P warm start
unless stated.
* **B1 (blocking) merge error depended on κ.** Lane mixing copied the atoms of all three lanes into each lane (own atoms twice) and merged 16 → 4 every year:
  NATOM 4 − 64-atom NLL +0.265 nats/site at κ = 0.1 vs −0.005 at κ = 0, the apparent κ effect 17× the true one. Fixed: (C) the own lane's atoms enter once with
  weight 1 − κΔ + κΔ w_v (§2.6); sorted merging — atoms sorted by D, the closest ADJACENT pair merged, N and N·D² (and Σ N H, Σ N A) conserved; NATOM 4 → 16
  (scouted on the verifier's 120 sites, NATOM − 64 NLL per site at κ = 0 / 0.1: 8: −0.002 / +0.029, 10: 0.000 / +0.015, 12: 0.000 / +0.017, 16: 0.000 / +0.004);
  a compact store (initial atoms of the occupied cells only; the working atoms in the per-thread scratch; steps visit the occupied cells only — 16 dense atoms
  would be 4.4 GB). Acceptance (3.2 / 3.2c): NATOM 16 vs 64 at κ = 0: NLL +0.000 nats/site, N≥12.7 / BA / ingrowth / classes 2–3 +0.00 %; at κ = 0.1: +0.004
  nats/site; the κ effect (κ = 0.1 vs 0) +0.174 vs +0.170 nats/site with 64 atoms (ratio 1.02).
* **I1 roughness.** (a) merge distances binned at 1e-6 cm, ties by D order (and duplicates < 1e-6 cm always merged); (b) the LIKELIHOOD blurs every atom's class
  with the FIA DBH error N(0, 0.2²) cm (§3; fixed `SD_DBH`). 3.3 on 1500 train sites, ±5 steps: ln ε (PITA) at 1e-9 range 4e-5 nats (= the slope; residual of a
  quadratic 2e-8) vs 0.28–0.33 before; at 1e-4 residual 0.055 nats vs 3.5 (range 4.2 nats = the slope, was 14.7 with jumps); κ at 1e-9 range 3e-5 nats.
* **I4 per-atom rates.** Growth dD and the size term of the hazard per atom from its own D (one table lookup each); light, climate, vigour, height, thinning stay
  per cell; the height increment uses the QMD of the grown atoms. 3.2b (64 atoms): per-atom − per-cell ingrowth into 12.7 cm +5.9 %, N≥12.7 +0.7 %, NLL −0.34
  nats/site. verify4p_ref.jl follows (section 2 PASS).
* **I5 performance.** compact! keeps the binned distances as integers over a linked list of the slots (no shifting), recomputes only the two next to a merge and
  finds the minimum by a vectorised pass (bitwise identical to a full rescan, which in turn ends at the first zero distance); the lane mixing builds each
  lane's buffer by a 3-way merge of the sorted lanes; every loop visits the occupied cells only. simulate! of the k1 train set (serial / 2 threads, wall, a
  shared 8-core box): round-2 kernel at its warm start (κ = 0.1) 23.0 s / 6.6–11.7 s; now at the warm start (κ = 0) 4.0–4.7 s / 1.5–2.8 s. At κ = 0.1 the new
  kernel takes 33 s / 15–18 s — **8–9× its own κ = 0 cost** (round-4 correction: the "1.4–1.6×" is relative to the round-2 kernel at the same κ, not to
  κ = 0): with lane mixing 1.43 M of 1.51 M live cells hold 16 atoms and the 48 → 16 merges of every lane every year dominate (NATOM 12 would cost ~0.6× of
  that at +0.017 nats/site of merge error at κ = 0.1). Calibration therefore runs in two stages (decisions H9).
* **I2 warm start (params4.jl `warm_start4`).** (a) OXAR, PIEC2, PIPA2 (← _S / _H) and QUPR2 take their HBAD3P group key's ε / lnF; (e) the crowding / crown factor
  F uses HBAD3P's RD per FORESTED subplot ha: 1.1847 (was 1.1815); (c) ε × the subplot / plot light ratio 0.9389 (tree-weighted e^{−aO}, all train sites) →
  ε × 1.1124 in all; (b) ms × 0.768 so that the N-weighted v1 hazard equals HBAD3P's (gm × F × the mortality-light ratio 0.969; it was 1.29×); (d) κ linear on
  [0, 1], warm start κ = 0 (HBAD3P's vigour nodes are persistent and its σv was fitted so; no HBAD3P parameter maps to a mixing rate). 8.2 / 8.2b / 8.3 / 8.4 /
  8.6 PASS.
* **I3 baseline.** The train-estimated transition-matrix null (data4.jl `tm_null` / `tm_state!`, = the verifier's) is reported by smoke4 and is the bar on val
  counts (§3–4). Warm start (no calibration), NLL train / val: HBAD4P 124,816 / 43,537 (val count 111,419), transition matrix 130,142 / 48,636 (val count
  102,796), persistence 357,947 / 120,825: the warm start beats the TM null in total but NOT yet on val counts.
* **I6.** Intervals whose v2 visit has no forested subplot are dropped in load4 (logged): 102 (train 67, val 21, test 14).
* Minor: prof4.jl uses warm_start4 (the space4 MethodError is gone); sH of classes with < 30 df pools the neighbouring classes (classes 13–16: 0.088 / 0.074 /
  0.079 / 0.079, was 0.1); the sapling prior's (m, α) are fitted jointly (profile likelihood over (α, m), `nb_joint`; +1.6 nats over the Poisson-m rule);
  equal-diameter atoms merge; stale §2.7 / §5 / §7 text corrected; planted factors documented (§2.9); the warm start's selection on the product's val (§7).

## 10. Audit round 4 (2026-10-26, independent Opus review of 0217126 on k1; fixed — decisions H7–H13 in hbad4p_decisions.md)
* **I1 the size / height terms were gated at N > 1e-9** (data4.jl). A cell's first trace prediction (a blur tail) switched on a term of −1.5 to +13 nats, and the
  scored set differed between models and nulls (val: 1,477 unscored observed cells for the warm start, 72 for the TM null).
  - Now every observed cell of a tallied area is scored as the mixture w · model + (1 − w) · background, w = N / (N + bg), with bg the count background as
    pseudo-count (§3, H7).
  - The predicted class height uses each atom's allometric height H + H*(D_m) − H*(Dq).
  - Step tests, full train, ±20 steps of ln ε (PITA), the acceptance scripts' observation parameters, step sd around the smooth slope (H12):

    | | sd at 1e-4 | max jump at 1e-4 | sd at 1e-3 |
    |---|---|---|---|
    | Gate (before) | 2.79 | 12.4 | 8.15 |
    | Now | 0.47 | 1.82 | 2.66 |

  - κ (±20 steps around 0.1): 0.11 / 0.70 at 1e-4 / 1e-3.
  - With the observation parameters fitted at the warm start (sharper σD 0.010, σH 0.066), the sd at 1e-3 is 2.78 and the largest jump at 1e-4 is 2.71:
    just over the bars (H7).
  - Totals of the model and of the nulls are now computed over the SAME scored cells. Before this fix they were not comparable.
* **M2 the count term keeps jumps of about 1 nat at 1e-4** (sd 0.16; 2.5 at 1e-3 steps): exact transfer across class bounds, and one height and age per cell in
  the dynamics. Accepted; it is most of what remains at 1e-3.
  - The verifier therefore splits the roughness check by term (H14): 3.3 enforces the I1 bar on size + height, 3.3b holds the count term to its 2026-10-27
    values + 10 %, and 3.3c reports the total as INFO.
  - The residual noise CMA-ES will see is about 2.5–2.8 nats per 1e-3 step of ln ε (PITA) on full train (corrected in round 5, H20: per coordinate, see §11 — 0.0–1.5 nats per 1e-4 step, and on 5000 sites 0.01–5.2 per 1e-3 step; ε:PITA was not representative).
  - The open lever is smooth cell membership in the dynamics (an atom's rates and height continuous across a class bound), a model change for a later round.
* **I2 split keys took their group's whole fecundity** (predicted / observed v2 sapling tally 6.1 / 6.6 / 2.1 / 23.9 for OXAR / PIEC2 / PIPA2 / QUPR2).
  - Now lnF = the group's + ln(the key's share of the group's train sapling ingrowth); the group keeps its own share (H8).
  - HBAD3P's recruitment light was evaluated per subplot. The subplot / plot ratio of e^{−λ RD} is 1.12–1.18 at the train v1 state, so it is applied to lnF.
  - Every key is now at 0.87–1.14 (verify 8.3).
* **I3 κ > 0 costs 8–9× κ = 0** (corrects §9 I5's "1.4–1.6×", which was relative to the old kernel). Calibration is in two stages (H9): `HBAD4P_FIX="κ:0"` /
  `space4(fix_extra=…)`, and `tools/hbad4/kappa_profile4.jl` for the κ profile on val counts. Only a dry run was made; no calibration.
* **M1 unobserved tallies.**
  - Plots whose four microplots are non-forest have f = 0: no exposure, no count term, the v1 sapling layer at its prior mean, and they are out of the
    prior's fit (H11).
  - Before, they were scored as empty full microplots: 68.7 / 59.3 / 26.0 nats on train / val / test, from 26 / 22 / 10 intervals at v2. At v1 the 34 / 23 /
    10 intervals had sapling layers of ≈ 100 instead of ≈ 400 /ha.
  - Intervals whose v1 visit has no forested subplot are dropped (24; 18 train; H10).
* **V verifier.**
  - 3.3 is the step test above, with jump detection.
  - 7.2 recomputes the mixture independently (Distributions, the background recomputed — 7.4c).
  - 8.3 is the per-key sapling-tally check (it replaces the circular group check).
  - 8.4 computes the overtopping and hazard independently.
  - 9.3 checks the tools' transition-matrix null against a re-implementation written from its definition.
  - The observation parameters are fitted at runtime.
  - 3.2c gets an absolute floor (H13): the κ effect is now ≈ −0.015 nats/site.
  - Found on the way: the cache key hashed the code as byte vectors, and Base.hash SAMPLES a vector. 56 % of one-byte edits of data4.jl kept the old key,
    i.e. a stale cache. It now hashes Strings (5.11 tests 300 edits).
* **Noted (one height per cell):** recruits enter class 1 at H*(2.54 cm) and pull the cell's mean height down. The existing saplings of the cell then grow
  slightly faster: dD ∝ 1 / max(H, 1.37), a higher hydraulic factor, and they are overtopped less. This is a consequence of a single height per cell; it is
  small but systematic in class 1.

## 11. Audit round 5 (2026-10-27, independent Opus review of 3e4f6a7 on k1; fixed — decisions H16–H21, H23 in hbad4p_decisions.md)
* **I-1 encodings (H16).**
  - ρ is log on [1e-7, 0.05]: linear on [0, 0.01], a CMA-ES step hit the clamp (+17,510 nats).
  - Verify 8.7 audits every coordinate (z ± 0.05 / ± 0.5 at the warm start), which also widened c (log [0.3, 3]) and βT (linear [−1, 1]).
  - Flagged: κ and f0, at a bound by design; ε:PITA, b, s, pD and σv, sensitive (± 0.05 moves the train NLL by 1.1–6.1 %).
* **I-2 observation heights (H17):** H_m = H + H*(D_m) − mean_N H*(D), so Σ N H_m = Σ N H (class 1 was −2.2 %). Verify 0.14 is the hand check, 1.9 the
  invariant (9e-16).
* **I-3 recruitment.**
  - Rk weighted by the subplots' forested fraction: 1.05–1.08 (H18).
  - The discriminating check (predicted vs observed RECRUITS) showed the HBAD3P mapping off by 0.73–2.31 (obs / pred) per key. lnF is now moment-matched on
    train (H23): train 1.000; val 0.64–1.43, 2.8 for PIEC2 on 3 recruits, and 6 of 17 keys beyond 2 sd of the NB noise (8.3b, INFO).
* **I-4 stage 1 (H19):** `tools/hbad4/calib4.jl` (dry run only). `save_z4` writes the space signature (`.sig`), and `load_z4` refuses a mismatch (verify 8.8).
* **I-5 κ decision (H21):** 3 seeds per κ; κ > 0 must beat κ = 0 by ≥ max(20, 2 (pooled sd + 0.005 n_val)) nats.
* **I-6 roughness on every coordinate (H20).** Verify 3.3 covers 38 coordinates: 1e-4 steps on full train, 1e-3 on 5000 sites. Size + height step sd / jump:

  | Coordinate | sd at 1e-4 | sd at 1e-3 | jump at 1e-4 | Within the sd bars (0.93 / 2.7)? |
  |---|---|---|---|---|
  | b | **1.12** | 2.39 | 4.03 | **no** |
  | βT | **1.51** | **4.01** | 4.73 | **no** |
  | σv | 0.88 | 2.37 | 3.01 | yes |
  | pD | 0.83 | 2.20 | 3.54 | yes |
  | ε:PITA | 0.77 | 1.34 | 3.61 | yes |
  | f0 (at 0.05) | 0.56 | 1.72 | 1.75 | yes |
  | every other coordinate | ≤ 0.39 | ≤ 0.97 | | yes |

  - **H24 (after this run):** the bar became relative, sd ≤ 1 % of the coordinate's |ΔNLL| at ±0.05. The base point now carries the fitted observation
    parameters: the table above used the default ones by a slip.
    - With both changes, 31 of 38 coordinates pass.
    - 7 fail at 1e-3 only: mW, ε:MAVI2, pM, am, ρ, λ1, βT, with ratios 1.1–4.6. They are coordinates the objective barely sees (|ΔNLL| 0.2–311 nats).
    - Of the absolute bars, only βT exceeds them (1.15 / 3.40); b is within them (0.715 / 1.823). See decisions H24.
  - **H25:** no further bar change. 3.3 prints `FAIL-KNOWN (H25)` with the per-coordinate ratios and still counts as a FAIL (the verifier exits 1). It is
    FAIL-KNOWN only while the failures are exactly within the H24 table of k1 (+20 % on each ratio); anything else is a plain FAIL. Whether the roughness harms
    calibration is answered by calib4.jl's 3-seed replication check (decisions H25, H25-impl).
  - (Round-5 text, before H24:) b and βT FAIL the sd bars. Both are global multipliers of every tree's growth (the size exponent, the temperature response), so a 1e-4 step moves every
    atom, and the one-height-per-cell crossings add up. I did not tune; this is reported as open (§9 lever: smooth cell membership).
  - The jumps > 2 nats are KNOWN entries of k1 (+20 %).
  - The count term is held to its k1 values + 10 % (3.3b); the references are keyed by (role, cache key).
  - **The noise CMA-ES sees** (total sd per step, 3.3c): up to 1.5 nats per 1e-4 step on full train (βT 1.47, b 1.18, σv 1.04, pD 0.91, ε:PITA 0.79) and
    up to 5.2 per 1e-3 step on 5000 sites (βT 5.18, σv 3.33, b 3.12, pD 2.72). Most coordinates are below 0.3 / 1.0.
* **M-1:** the size / height mixture weight of a cell mean of n trees is w_n = 1 − (1 − w)^n. The size + height NLL changes by +1.6 %, and their pull on the
  densities is about halved.
* **M-2:** the κ profile's observation refit uses the cached state.
