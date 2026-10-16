# HBAD v2 — biomass-free, process-based cohort model (branch `hbad-v2`, worktree /workspace/landis_hbad2)

Goal (user, 2026-10-16): leave Biomass Succession aside; build HBAD v2 that is **completely biomass independent**:
Biomass-Succession-inspired structure, but **mean height, QMD and TPA drive competition** (QMD–stocking charts: BA, TPA and
stocking are constrained); each species has its own curves of **height vs diameter growth over cohort age**; **TPA only decreases**
(new trees = new cohorts). Work autonomously, iterate on the concept, verify empirically.

Why biomass-free: FIA AGB is not measured, it is an allometric estimate from DIA (+HT) (model on model; it changed with the NSVB
switch), so a biomass-scored model is fitted to an estimate of an estimate. DIA (0.1 in), tree counts and remeasured survival are
measured; HT is measured but noisy (see §1).

## 1. Data audit (tools/hbad2/audit_trees.jl; GA+FL, slim_r4.duckdb read-only)
* Tree-visits: 868,568 live macro trees (TPA_UNADJ 6.018, DIA ≥ 5 in), 124,152 live microplot saplings (74.965, DIA 1–4.9 in).
* Tree identity (plot, SUBP, TREE) is stable across visits: 581,401 live→live, 168,532 live→dead/removed consecutive pairs (dt ≈ 5.4 yr).
* `estimated_age` advances exactly with elapsed time for every tracked tree (age2 − age1 − dt: mean −7e−17), i.e. each tree has a
  fixed birth year → cohort identity by (species, birth-year class) is stable.
* **TPA of an observed cohort is not monotone, although the true cohort TPA is**:
  - 8,118 live saplings cross 5 in and switch from TPA 74.965 to 6.018 (a −92 % "drop" that is a sampling-design change);
  - 143,661 trees are first tallied at a later visit with DIA ≥ 5 in, median estimated age **20 yr** (q10–q90 12.7–30) — ingrowth
    trees join birth-year classes that already exist ("newer trees = younger cohorts" does not hold at the tally threshold).
  ⇒ Sim A scores **closed sets of tracked trees**: a cohort = the trees alive at v1; at v2 its survivors keep their v1 expansion
  weight, so observed N can only decrease and mortality is a binomial count of tree fates. Ingrowth is a separate process.
* Height: HT present for all live trees, = ACTUALHT for 98.5 %; **16 % of remeasured live trees get shorter** (sd ln(H2/H1) 0.10–0.31
  by size) → measured H is noisy (see E1b).
* Plots by number of visits: 1:131, 2:1212, 3:1354, 4:2458, 5:1127, 6:3021.

Data product (tools/hbad2/build_intervals.jl → results/hbad2/, not committed): `intervals.arrow` (30,703 plot intervals; flags trt/dst/harv
at v2, planted at v1), `interval_trees.arrow` (766,365 tracked trees: alive 581,359, dead 51,606, harvest 114,146, missing 19,254),
`interval_ingrowth.arrow` (183,698). Undisturbed intervals (no TRTCD1, DSTRBCD1, harvest fate) with dt > 2 yr are used for fitting.

## 2. Concept checks on trees (tools/hbad2/empirical_checks{,2}.jl; fit TRAIN, score VAL; 403,800 trees)
E1 — competition index (species + ln D, D, ln age always in; VAL):

| index set | R² ln dD | R² dH | mort NLL | mort AUC |
|---|---|---|---|---|
| none | 0.207 | 0.186 | 0.2352 | 0.678 |
| BAL (v1) | 0.266 | 0.200 | 0.2347 | 0.682 |
| SDIH (stand density of TALLER trees) | 0.270 | 0.185 | 0.2337 | 0.690 |
| BAL+RD | 0.266 | 0.205 | 0.2343 | 0.684 |
| SDIH+RD | 0.270 | 0.186 | **0.2322** | **0.696** |
| SDIĤ+SDI (taller by species H–D curve height) | 0.269 | **0.204** | | |

E1b — measured-height ranking creates a regression-to-the-mean artefact: corr(ln H/Ĥ at v1, dH) = −0.28 (trees measured too short
"grow" more). Ranked by Ĥ (species Chapman–Richards height at the tree's D) the effects are: overtopping −30 % ln dD per SD and
−0.072 m/yr per SD of dH; **symmetric crowding (stand SDI) ≈ 0 on dD (+2 %) but +0.040 m/yr (+14 %) on dH** ⇒ crowding shifts growth
from diameter to height (H/D rises with density), suppression reduces both. Implication for the model: rank cohorts by MEAN height
(cohort means average the noise), height growth needs its own process (not an H–D curve).

E3 — species timing (free-growing CCLCD 1–3; mean dH m/yr / dD cm/yr by age class; results/hbad2/timing_free_growing.csv):
pines (PITA 0.86/0.86 at 5 yr → 0.31/0.56 at 70 yr) grow height and diameter together early and height growth collapses with age;
oaks (QUNI 0.42/0.37 → 0.24/0.65; QULA3 similar) keep **increasing** diameter growth while height growth falls (dH:dD 1.1 → 0.4);
LITU/LIST2 height-first; NYBI/TAAS slow, diameter growth rises with age. ⇒ species-specific H and D curves with different timing.

## 3. Model (src/plugins/HBAD2/HBAD2.jl; stand-alone prototype kernel)
See the module header for the equations: cohort state (N, D, H, A); overtopping O from SDI of taller cohorts (+½ own),
Chapman–Richards free-growth curves for H and D in state form, competition modifiers exp(−gO·O + gR·RD) per variable, hazard =
background(size) + suppression (1 − e^{−gOD·O}) + Biomass-Succession age term, self-thinning to SDImax removing suppressed cohorts first.
14 parameters per species (13 fitted + LONGEVITY from species_longevity_ref), 1 global (size slope of background mortality).

## 4. Sim A harness and likelihood (tools/hbad2/common.jl, fit_simA.jl, cmaes.jl)
Cohort = (plot, species key, 10-yr birth class) of the tracked trees; species keys = species with ≥ 3000 train trees (10) + _S/_H.
Each interval is an independent site initialised from the observed (N, D, H, A) and integrated dt years (whole years + one fractional step).
Objective = joint NLL: binomial survival of tree counts (p = N2/N1) + Gaussian ln D2 and ln H2 of the survivors (weight = survivors, σ profiled out).

## 5. Results log

### 5.1 v2a (fit_simA.jl v2a: 13 params/species + 1 global, CMA-ES λ 32, 20k evals from the data-based start)
Joint NLL (lower is better; mortality = binomial tree counts, D/H = profiled Gaussian on ln of survivor QMD / mean H):

| model (same tracked cohorts) | val total | mort | ln D | ln H | val deaths sim/obs |
|---|---|---|---|---|---|
| null (no growth, per-species constant survival) | −57,281 | 34,324 | −50,027 | −41,578 | |
| HBAD v1 kernel as designed (FIA regressions, SDImax p95, smallest-first thinning), init from observed N/D/H | −35,976 | 107,035 | −79,611 | −63,400 | 16,151 / 9,570 |
| HBAD v1 kernel without self-thinning | −109,570 | 33,493 | −79,661 | −63,402 | |
| **v2a** | **−120,438** | 33,688 | **−81,265** | **−72,861** | 9,7xx / 9,570 |

(train v2a −128,582; σ_D 0.158, σ_H 0.192 on val.) v1 findings on real tracked trees: its smallest-first self-thinning kills
1.7× the observed trees; its H–D-ratio height rule under-predicts height growth by 35 % (0.25 vs 0.39 m/yr) — exactly the
"height needs its own process" result of E1b/E2. v2a residuals (val): mortality by size U-shaped wrong (saplings −15 %, 13–20 cm
+24 %, > 45 cm −32 %), planted stands taller/less mortality than predicted → v2b adds a per-species ln-size mortality slope, a
global linear size term, and planted multipliers (pH, pD, pM).

### 5.2 Recruitment (tools/hbad2/recruitment_check.jl → results/hbad2/recruitment_coefs.csv)
Sapling ingrowth into the 1-in tally (microplot, 185.2 trees/ha per tree), NATURAL undisturbed intervals, Poisson per species:
E[count] = dt · exp(ρ0 + ρS ln(1 + BA_sp ≥ 5 in) + ρL SDI/600). The light coefficient reproduces the Biomass-Succession shade
logic from data: ρL pines −3.5/−3.6, LITU −3.0, oaks −2.0/−2.3, LIST2 −2.0, tolerant ACRU −1.5, NYBI −1.3, MAVI2 −0.8, _H −0.8.
Seed term helps for most species (VAL deviance). A conspecific-sapling term ("advance regeneration") lowers deviance a lot more
but would make recruitment self-reinforcing in the kernel (not used). PITA/LIST2 are under-predicted on VAL (353 vs 828): the
recruitment surface is very heterogeneous (and planting-affected; fitted on natural stands only).

### 5.3 Long run from bare ground (tools/hbad2/longrun.jl, intermediate v2a, recruitment incl. planted plots — superseded)
BA plateau ~45 m²/ha at 90–110 yr, top height 37 m at 150 yr, QMD(≥5 in) 32 cm, RD ≈ 0.6 (density regulated by suppression
mortality; self-thinning cap never binds); PITA ~65 % of BA throughout (recruitment then fitted including planted plots).

### 5.4 Sim B — free stand runs on real plots (tools/hbad2/simB.jl; VAL, 465 clean natural plots, 1,723 later visits)
Whole observed stand (all live trees ≥ 1 in, ingrowth included) vs a free run from the first visit with recruitment; baseline =
persistence. Median |ln(sim/obs)| (bias), all gaps; v2a parameters:

| variant | BA | N ≥ 5 in | N all | QMD | Lorey H | composition (½Σ|Δshare|) |
|---|---|---|---|---|---|---|
| persistence | 25.6 % (−40) | 21.5 % (−37) | 33.4 % (−24) | 19.0 % (−7) | 13.6 % (−17) | 0.134 |
| v2a, 10-yr birth cohorts, RD-units bug in recruitment | 12.7 % (+7) | 26.2 % (−6) | 60.6 % (+74) | 30.5 % (−33) | 7.5 % (−4) | 0.145 |
| + size-split initial cohorts (5/12.7/25/40 cm) + recruitment light index fixed to SDI/600 | 12.8 % (+4) | 24.1 % (−6) | 42.0 % (+51) | 21.3 % (−23) | 6.4 % (−2) | 0.141 |
| + lognormal within-cohort D observation operator σ 0.16 (data: 0.16 lumped / 0.11 split) | 12.8 % (+4) | 21.5 % (0) | 38.5 % (+45) | 19.8 % (−20) | 6.5 % (−2) | 0.140 |

BA change: corr(sim Δ, obs Δ) 0.62; Lorey-H change 0.69. Bug found and fixed: the kernel's recruitment light index used the
SDImax-relative RD of the growth kernel while the coefficients were fitted on SDI/600 → up to 2× too many pine recruits.
Remaining failure = dynamics of small trees: after > 21 yr the 5–12.7 cm class holds 284 trees/ha (obs 94) and 12.7–25 cm 161
(obs 244) — recruits stall as suppressed poles instead of dying or growing through (Sim A also under-predicts small-tree mortality).

### 5.5 v2b (per-species ln-size mortality slope, global linear size term, planted multipliers; warm start v2a, σ 0.03, 25k evals)
VAL total **−122,232** (mort 33,477 · ln D −81,914 · ln H −73,795; σ_D 0.156, σ_H 0.188); train −130,067. Every part improves on v2a;
mortality is now better than the best v1 variant (33,493). Planted multipliers (fitted): height-growth rate ×e^0.353 = 1.42,
diameter-growth rate ×1.32, background+suppression hazard ×0.78 (genetic improvement / site preparation — management, not
succession). (First v2b attempt with σ 0.15 from the warm start stopped after 1,953 evals without a better sample: in 172
dimensions the v2a optimum is sharp; σ 0.03 + a 150-generation stall window works.)
Sim B (σ_obs 0.16, size split): BA 12.9 % (+0 bias), N ≥ 5 in 22.3 % (−4), N all 37.9 % (+44), QMD 20.0 %, Lorey H 6.7 % — the pole stall
persists (> 21 yr: 5–12.7 cm 275 vs obs 94 trees/ha).
Small tracked trees (D < 12.7 cm) by overtopping (VAL): deaths obs/sim O 0.2–0.35 1,028/839, 0.35–0.5 618/467, 0.5–0.75 271/193 —
suppressed small trees die 20–40 % more than the size-independent suppression hazard allows, and their survivors grow 1–5 % more
(the slowest die: within-cohort selection). → v2d: suppression hazard × (D/20)^mSD.

Species timing implied by the fitted curves (tools/hbad2/timing_curves.jl; model increments at the E3 free-growing class means, O 0.15,
RD 0.4): corr with observed dH 0.94, dD 0.77; the dH:dD allocation decline is reproduced for oaks (QUNI obs 1.1→0.4, model 1.1→0.3),
pines, MAVI2, NYBI, ACRU. Failure at large sizes for LIST2 / LITU / TAAS / PIEL: the Chapman–Richards diameter asymptote is low
(LIST2 Dmax 38 cm) so diameter growth stalls at old ages; several Chapman–Richards shape parameters sit at their bound (cD = 4,
SDIMAX = 2000, ACRU Hmax = 70) → v2c tests a power-exponential potential r (X/X0)^c e^{−X/Xmax} (its free-growth age curve follows by integration).
