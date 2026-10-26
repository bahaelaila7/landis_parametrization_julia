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
subplot — a land-use change inside a "natural" interval, FIA tallies no tree there — excluded, audit round 3 I6: 102 intervals, 67 of them train at k1) the v1 plot stand table
initialises the site and the model runs to v2 (fractional final year); the v2 stand table is scored per (k, j).
* **v1 state**: each tree → its class; N per forested ha of its tally (subplot fraction for subplot trees, microplot fraction for saplings, by the
  tree's expansion weight); B, H, A from the trees; split over the lanes. **Sapling layer (classes 1–2)**: the microplots sample 8 % of the subplot area,
  so a v1 sapling count is a noisy sample of the stand's sapling density (46 % of the train ingrowth ≥ 12.7 cm came from keys without a v1 microplot
  sapling; audit B3). The state takes the empirical-Bayes posterior mean density N̂ = (n + α) / (e + α / m), e = microplot area × forested fraction,
  with a negative-binomial prior per (key, class, adult of the key on the plot) fitted on the TRAIN plots ((m, α) by maximum likelihood, α by profile
  likelihood over (α, m); round 2 had m = Σn/Σe, the Poisson MLE); an
  unobserved cell gets the key's train mean QMD / height / age of saplings.
* **counts**: v2 tally trees n_kj (classes 1–2 on the microplots, ≥ 3 on the subplots; each tree in the class of its v2 diameter, as FIA re-tallies a tree
  that shrank below 5 in as a sapling) ~ NegBin(μ_kj = (N_kj + bg_j) · area_j · f, r_j). area_j from FIA's expansion factors (TPA_UNADJ 6.018046 /
  74.965282 per ac); f = the forested fraction of that tally at v2. **bg_j** = background densities (microplot / subplot tallies; fitted): immigration,
  misidentification, recording error — what the model leaves empty — instead of an arbitrary floor.
* **observation space** (audit round 3 I1b; the LIKELIHOOD only, the dynamics are unchanged): every atom is measured with the FIA DBH error N(0, sdM²),
  sdM = 0.2 cm fixed (`SD_DBH`, params4.jl): it contributes to class t with probability Φ((ub_t − D)/sdM) − Φ((lb_t − D)/sdM) (lb of class 1 = −∞: the 1-in
  tally threshold is the model's entry definition — recruits enter AT 2.54 cm and would otherwise be half untallied; ub of class 16 = ∞). The predicted
  table per (k, j) is N_kj = Σ p N, B_kj = Σ p N π/4 D², H_kj = Σ p N H / N_kj over the key's atoms (all lanes): an atom approaching a class bound moves into
  the next class's prediction smoothly instead of in one jump. The nulls are tables in observation space already and are scored unblurred.
* **size**: log QMD of the cell's observed trees vs the predicted (blurred) QMD, Gaussian, variance sD_j² / n + σD² (sD_j = the TRAIN within-cell sd of log D
  of class j: 0.29 in class 1 to 0.02 in class 13 — a single σ was off by a factor of 12 across classes; classes with < 30 df the uniform-spread value; σD =
  model error, fitted).
* **height**: log mean measured height vs H, variance sH_j² / n_H + σH² (sH_j likewise; classes with < 30 df pool the neighbouring classes, audit round 3 —
  was a flat 0.1).
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
  kernel takes 33 s / 15–18 s, 1.4–1.6× the round-2 kernel at the same κ: with lane mixing 1.43 M of 1.51 M live cells hold 16 atoms and the 48 → 16 merges
  of every lane every year dominate (NATOM 12 would cost ~0.6× of that at +0.017 nats/site of merge error at κ = 0.1).
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
