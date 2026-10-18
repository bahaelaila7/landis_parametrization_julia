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

## 4.1 Current model specification (v2f / v2g; `HBAD2_FORM=2 HBAD2_SIZE_SPLIT=5,12.7,25,40 HBAD2_SITE=subplot [HBAD2_VIGOUR=1]`)

**Site** = one FIA subplot (7.3 m radius; densities expanded ×4 to per hectare). Competition (light, crowding) is computed within the
site; the four subplots of a plot are simulated independently (E4: local competition explains growth and small-tree mortality better).

**Cohort state** (SoA arrays, CSR over sites): species `sp`, trees/ha `N` (only decreases after a cohort's recruitment class closes),
quadratic mean diameter `D` (cm), mean height `H` (m), age `A` (yr), vigour quantile `zq` (0 = mean-field cohort). No biomass.
Initial cohorts from data: (species key × 10-yr birth class × initial D class 5/12.7/25/40 cm), N = Σ TPA_UNADJ·2.47105·4, D = QMD,
H = TPA-weighted mean height, A = mean estimated age (ages of saplings are unreliable; growth does not depend on age).

**Annual step per site** (Δ = 1, or the fractional remainder of a remeasurement interval):
1. Snapshot: SDI_i = N_i (D_i/25.4)^1.605; SDImax_site = SDI-weighted mean of species SDIMAX; RD = ΣSDI / SDImax_site;
   overtopping O_i = (Σ_{H_j > H_i} SDI_j + ½ Σ_{H_j = H_i, j≠i} SDI_j + ½ SDI_i) / SDImax_site.
2. Growth (power-exponential potential in the current size; its integral is the species' free-growth age curve):
   dD = rD (D/20)^cD e^{−D/Dmax} · e^{−gOD·O} · e^{gRD·RD} · e^{σv·zq} · (planted ? e^{pD} : 1)
   dH = rH (H/15)^cH e^{−H/Hmax} · e^{−gOH·O + gRH·RD} · e^{σv·zq} · (planted ? e^{pH} : 1)
3. Mortality hazard (N ← N e^{−hΔ}): h = fM [ e^{mbg + mD ln(D/20) + mD2 D/20} + msup (1 − min(1, e^{−gOD·O} e^{σv·zq})) (D/20)^{mSD} ]
   + e^{mA (A/LONGEVITY − 1)},  fM = planted ? e^{pM} : 1, mA = 10 (Biomass Succession "d").
   = background (U-shaped in size) + growth-suppression (light-limited, size-dependent, vigour-dependent) + age-related.
4. Self-thinning safety cap: while RD > 1, SDI is removed in proportion to SDI_i (1 − v_i) (suppressed first). In the fitted
   models the cap never binds: the Reineke boundary emerges from suppression mortality (stocking chart, 5.10).
5. A ← A + Δ.
6. Free runs only — recruitment into the 1-in tally per species: R = 185.2 · exp(ρ0 + ρS ln(1 + BA_sp(D ≥ 12.7)) + ρL · SDI/600)
   trees/ha/yr; recruits enter at D 2.54 cm, H 3 m, A 5 and accumulate in one cohort per species and 5-yr class (×3 vigour sub-cohorts).

**Parameters**: per species key (12 = 10 species with ≥ 3000 train trees + _S/_H): Hmax rH cH Dmax rD cD (growth curves), gOD gRD gOH gRH
(competition responses), mbg mD msup (mortality), SDIMAX (cap), LONGEVITY (fixed, species_longevity_ref) = 14 fitted; global mD2, pH, pD,
pM, mSD, σv; recruitment ρ0 ρS ρL per species (start = Poisson fit on natural sapling ingrowth).

**Observation operators and likelihoods** (no biomass anywhere):
* Sim A (tracked trees, each undisturbed remeasurement interval an independent site initialised from the observed stand): per cohort,
  binomial survival of the tracked tree COUNTS (p = N2/N1) + Gaussian ln QMD and ln mean H of the survivors (σ profiled). This is the
  calibration target (`fit_simA.jl`).
* Sim B (free runs from the first visit of clean natural plots, recruitment on): expected TALLIED trees per size class through a
  lognormal within-cohort D distribution (σ_obs = data-based 0.11) ÷ FIA expansion (microplot 185.2·4, subplot 14.87·4 per site), Poisson
  NLL against the observed tally; `HBAD2_FB=stand` per (species × class × subplot), `size` per (plot × class, species pooled); + Gaussian
  ln Lorey height. Second objective of the joint calibration (`fit_joint.jl`).
* Evaluation: val selects among structures/fits, test is reported once (5.10).

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

### 5.6 v2c (power-exponential potential growth, cold start) and v2d (+ size exponent of the suppression hazard)
Fits record their structural switches in `fit_<tag>/meta.txt` (form, size split); every tool refuses a fit evaluated with a different
growth form (a silent mismatch would decode the parameters with the wrong equations).

| fit | form | VAL total | mort | ln D | ln H | note |
|---|---|---|---|---|---|---|
| v2b | Chapman–Richards state form | −122,232 | 33,477 | −81,914 | −73,795 | |
| v2c | power-exponential r (X/X0)^c e^{−X/Xmax} | −122,365 | 33,530 | −82,054 | −73,841 | cold start, 30k evals |
| v2d | v2c + msup (1−v)(D/20)^mSD | **−122,600** | **33,431** | −82,097 | −73,934 | mSD = −0.39 |

Species timing (timing_curves.jl): v2c corr with the observed free-growing increments dD 0.86 (v2b 0.77), dH 0.955, dH:dD ratio 0.82;
no diameter stall at large sizes any more (figures `results/hbad2/figures/v2c_{species_curves,allocation_vs_obs}.png`). The
E3 trees are train+val plots, so this is a consistency check of the curves, not a held-out test. Planted multipliers are stable
across fits (v2d: height rate ×1.42, diameter ×1.30, hazard ×0.79).

Sim B (v2c, v2d): unchanged from v2b (BA 12.9 % / bias +1 %, N≥5 in 21.9 %, Lorey H 6.4 %, composition 0.140 vs persistence 0.134).
Diagnosis of the remaining stand-structure error:
* tree-level annual mortality declines smoothly with D (2.5 cm 3.8 %, 5 cm 2.6 %, 10 cm 1.5 %, 12.7 cm 1.4 %, 20 cm 1.0 %, 25–35 cm 0.9 %):
  no step at the 5-in sampling threshold; but by cohort-QMD class the model under-predicts deaths at 5–12.7 cm (−18 %) and over-
  predicts at 12.7–20 cm (+17 %) → aggregation: cohorts that mix microplot saplings (185/ha each) and subplot trees (14.9/ha) get one
  QMD and the convex size hazard at the mean is wrong for both parts.
* recruitment scaling (diagnostic only, ×0.5 / ×0.25) cannot fix the size profile: poles match only at ×0.25 when saplings collapse
  (87 vs 264), and 12.7–25 cm stays low (144 vs 244) → the error is the flow through the mid sizes (graduation from poles, mortality
  of 12.7–20 cm), not the recruitment level.
* v2d evaluated on size-split cohorts (5/12.7/25/40 cm): mortality NLL 33,094 (vs 33,431 lumped) — splitting alone helps.
→ v2e: fit on size-split Sim A cohorts (warm start v2d).

### 5.7 v2e — fitted on size-split Sim A cohorts (5/12.7/25/40 cm; warm start v2d)
VAL total −170,903 on the split cohorts (v2d on the same cohorts −170,604); mortality NLL 33,022 (v2d lumped-fit 33,094 on split,
33,431 on lumped). Globals: mSD −0.68 (stronger size dependence of suppression mortality once cohorts are not size-mixed),
planted ×1.43 height / ×1.33 diameter / ×0.79 hazard. Size-class residuals (VAL deaths obs/sim): < 5 cm 1,757/1,536, 5–12.7 1,185/1,096,
12.7–20 4,063/4,315 (+6 %, was +17 %), 20–30 1,554/1,508, 30–45 757/800, > 45 254/207.
Sim B (σ_obs 0.11 = the within-cohort spread of split cohorts): unchanged at the stand level (BA 12.9 %, bias +0.1 %; N ≥ 5 in 22.0 %;
Lorey H 6.4 %) and the pole stall persists (> 21 yr: 5–12.7 cm 246 vs obs 94; 12.7–25 cm 160 vs 244).
⇒ the tracked-tree likelihood does not identify the recruit → pole → mid-size pathway (Sim A has no recruits and no ingrowth).

### 5.8 Joint calibration (tools/hbad2/simB_lik.jl, fit_joint.jl)
Sim B stand-table likelihood: free runs (train: 532 clean natural plots) scored at every later visit by a Poisson NLL of the TALLIED
tree counts per (species key × size class [2.54,5) [5,12.7) [12.7,25) [25,40) [40,∞)) — expected count = model N in the class through
the lognormal within-cohort operator ÷ FIA expansion (185.2 microplot, 14.87 subplot) — plus a Gaussian ln Lorey-height term.
0.2 s per evaluation; no recruitment doubles it (val 33.6k → 67.1k). Two objectives (fA = Sim A NLL, fB = Sim B NLL) because the
two share plots (summing would double count); augmented Chebyshev per λ, one CMA-ES run per λ; recruitment (3 per species) free.
Run j1: λ 0.5, 0, 1 from v2e + Poisson recruitment (start train fA −180,553 fB 36,561; val fA −170,903 fB 34,012).
j1 (plot sites) λ = 0.5 after 10k evals: train fA −180,513 fB 35,506; VAL fA −170,913 (start −170,903) fB 33,341 (start 34,012) — Sim B
improved without any loss in Sim A (no trade-off at this point). Sim B metrics: saplings fixed (< 5 cm at > 21 yr 291 vs obs 264) but
poles (237 vs 94) and mid trees (158 vs 244) unchanged → structural, not a parameter problem. j1 stopped after λ = 0.5 (superseded by 5.9).

### 5.9 Competition is local: subplot sites (E4, v2f)
E4 (tools/hbad2/empirical_checks3.jl; Ĥ-ranked overtopping + stand SDI, computed over the tree's SUBPLOT (×4 → per ha) vs the PLOT):

| indices | R² ln dD | R² dH | mort NLL | mort AUC |
|---|---|---|---|---|
| plot | 0.269 | 0.204 | 0.2335 | 0.687 |
| subplot | **0.281** | 0.206 | 0.2334 | 0.688 |
| plot + subplot | 0.288 | 0.209 | 0.2331 | 0.689 |
| small trees (< 12.7 cm), plot | 0.317 | | 0.4192 | 0.648 |
| small trees, subplot | **0.338** | | **0.4162** | **0.658** |

An FIA plot is 4 subplots of 7.3 m radius spread over ~0.4 ha: a sapling on subplot 2 is not shaded by a large tree on subplot 4, and
a plot-level mean-field canopy suppresses every recruit equally (no gaps). ⇒ `HBAD2_SITE=subplot`: every subplot (empty ones too in
Sim B) is a site, densities ×4 → per ha, recruitment refitted at subplot scale (recruitment_coefs_subplot.csv; light coefficients
PIEL −10.1, PITA −5.1, LITU −3.1, LIST2 −2.6, QULA3 −2.4, QUNI −2.3, _S −2.0, ACRU −1.6, NYBI −1.3, _H −1.2, MAVI2 −0.4).
v2f (Sim A on subplot sites, warm start v2e, 20k evals): VAL mortality NLL **32,872** (v2e 33,022 plot sites; v1 33,493; null 34,321 —
tree-count likelihood, comparable across site units); σ ln D 0.116 / σ ln H 0.155 (plot sites 0.132 / 0.170 — partly smaller, more
homogeneous cohorts). Globals: mSD −0.80, planted ×1.43 / ×1.28 / ×0.78. Sim B with v2f: BA 12.7 % (+5.5), N ≥ 5 in 22.3 % (−0.4 bias),
Lorey H 6.9 % — but more saplings/poles with gaps and plot-scale-fitted small-tree dynamics (> 21 yr: < 5 cm 396 vs 264, 5–12.7 cm 365 vs 94)
→ joint fit j2 on subplot sites (recruitment free).

### 5.10 Joint j2 (subplot), stocking chart, FIA design check, fair v1 comparison (val selects; test reported once)
* j2 λ = 0.5 (8k evals from v2f + subplot recruitment): VAL fA −271,772 (start −271,797) fB 61,853 (start 63,454): Sim B −2.5 % at no Sim A
  cost; stand metrics nearly unchanged (BA 12.8 %, N ≥ 5 in 22.4 %, Lorey H 6.9 %); the size profile still over-produces saplings/poles
  (> 21 yr 356/364 vs 264/94) and under-produces 12.7–25 cm (181 vs 244). CMA-ES in 209 dimensions moves slowly here.
* Stocking chart (`results/hbad2/figures/v2f_stocking.png`; tools/hbad2/{obs_stocking.jl,plot_stocking.py}): from bare ground the v2f
  stand recruits to ~1,600 trees/ha by 30 yr (QMD 10 cm), then self-thins PARALLEL to the Reineke −1.605 line (60–150 yr), max SDI 737 at
  95 yr — observed SDI p95 730 (plot) / 909 (subplot). The SDImax cap never binds (RD ≤ 0.46): the boundary of the QMD–TPA chart emerges
  from height-based suppression mortality.
* FIA design check (tools/hbad2/ingrowth_design_check.jl, 17,476 undisturbed intervals): first-tallied macro trees (≥ 12.7 cm) = 85.1
  trees/ha per interval, tracked microplot saplings crossing 12.7 cm × 185.2 = 60.4 → observed 5-in ingrowth is 1.41× what the microplot
  sample implies, and 5.3 % of "ingrowth" trees are already ≥ 20 cm (missed / procedural trees). ≈ 30 % of the observed ingrowth into the
  ≥ 5 in tally is not reproducible from the sapling sample by any model; part of the Sim B N ≥ 5 in deficit is this, the rest is
  mean-field threshold crossing.
* Fair head-to-head with v1 (same size-split cohorts, same site unit, v1 WITHOUT its self-thinning = its best variant; `v1_baseline.jl`):

| site | model | VAL total | VAL mort | TEST total | TEST mort | TEST σ ln D / σ ln H |
|---|---|---|---|---|---|---|
| plot | v1 (FIA regressions) | −153,794 | 33,003 | −152,684 | 32,512 | 0.146 / 0.216 |
| plot | **v2e** | **−170,903** | 33,022 | **−169,004** | 32,489 | 0.137 / 0.170 |
| subplot | v1 | −248,805 | 32,998 | −246,242 | 32,533 | 0.128 / 0.191 |
| subplot | **v2f** | **−271,797** | **32,872** | **−267,776** | **32,350** | 0.119 / 0.155 |

  Mortality: tie at plot sites, v2 better at subplot sites; diameter: v2 better; height: v2 much better (v1's H follows its H–D curve).
  v1's regressions were fitted on 600 k remeasured TRAIN trees, so this is a strong empirical baseline; v2 is the process model.

### 5.11 j2 front, pole diagnostics, size-profile objective
j2 (subplot; VAL): λ 1 = v2f (fA −271,797, fB 63,454; v2f is already a local optimum of fA: the λ = 1 run found nothing better);
λ 0.5: −271,772 / 61,853; λ 0: −270,969 / 59,833 → a real trade-off appears beyond λ 0.5. But even the Sim-B-only point (λ 0) does not
fix the size profile (> 21 yr poles 387 vs 94) — the stand-table likelihood (species × class per SUBPLOT, mostly zero cells) is dominated
by species presence/absence and gives little weight to the size profile.
Pole/sapling demography (tools/hbad2/pole_demography.jl, VAL clean natural plots; model cohort-years in free runs vs tracked trees, by
subplot overtopping SDI class <100 / 100–250 / 250–450 / ≥450 per ha):
* poles 5–12.7 cm, mortality %/yr model (recruit-origin) 1.8 / 2.3 / 2.7 / 3.2 vs data 1.5 / 1.8 / 2.7 / 3.3; dD cm/yr 0.62 / 0.37 / 0.28 / 0.21
  vs 0.57 / 0.33 / 0.21 / 0.13;
* saplings 2.54–5 cm, mortality 3.1 / 3.5 / 4.1 / 4.8 vs 2.7 / 3.1 / 4.6 / 5.4; dD 0.33 / 0.24 / 0.18 / 0.11 vs 0.42 / 0.26 / 0.16 / 0.11;
* recruit inflow model 51.2 trees/ha/yr (gross) vs observed sapling ingrowth 43.5 (net of in-interval deaths) — consistent;
  sapling stock model 469 /ha (run mean) vs 291 at v1.
Per-capita rates match within ~10–20 % in every light class; the pole excess is systematic (median plot-visit excess +14 / +108 / +207 /
+258 trees/ha at ≤7 / 8–14 / 15–21 / > 21 yr, not driven by outliers) → many small biases (shade mortality of saplings a bit low,
open-grown sapling growth a bit slow) compounding over decades.
j3: Sim B objective = species-pooled size-class totals per PLOT and visit (Poisson) + Lorey H (`HBAD2_FB=size`), λ 0 then 0.5 from v2f.

### 5.12 Persistent growth vigour (E5) → vigour sub-cohorts (v2g); scalarisation bug in fit_joint.jl
E5 (tools/hbad2/vigour_persistence.jl): 151,042 trees with two consecutive undisturbed growth intervals; residuals of ln annual dD
(species + size + subplot overtopping + subplot SDI) correlate **0.49** between interval k and k+1 (residual sd 0.975) → about half of the
unexplained growth variation is a persistent tree trait (≈ 2× growth between trees ±1 sd apart). A mean-field cohort cannot have fast
trees that graduate and slow trees that stall and die.
Kernel (σv, `HBAD2_VIGOUR=1`): every cohort (initial and each recruit class) is simulated as 3 Gauss–Hermite sub-cohorts, quantiles
z = −√3, 0, √3 with shares 1/6, 2/3, 1/6; growth (D and H) × e^{σv z}; the suppression hazard uses the vigour-adjusted relative growth
min(1, v e^{σv z}) (slow trees die — growth-dependent mortality, which also produces the within-cohort selection seen in 5.5). Sim A scores
the sub-cohorts aggregated back to the observed cohort (ΣN, quadratic-mean D, N-weighted H). With σv = 0 / vigour off every output is
bit-identical to before (unit tests 13/13; v2f val −271,797.3 reproduced). v2f parameters with σv = 0.3 and no refit: Sim A val −271,735
(slightly worse), Sim B size-profile objective val 614 vs 1,754 without vigour (lower = better) — vigour alone moves the size profile.
Bug found and fixed in tools/hbad2/fit_joint.jl: the Chebyshev ideal point was re-based whenever a candidate improved an objective;
at λ = 0 or 1 the best scalarised value is then pinned at 0 and CMA's best-so-far never updates (the returned point is the FIRST
improvement), and at other λ the objective drifts during the run. Now fixed reference points (start, ideal = start − 5 %). j2's λ = 0 / 0.5
points were produced with the moving ideal → treat them as indicative only (rerun pending). Plot-level species-pooled size-class objective
(`HBAD2_FB=size`): runs j3 (no vigour) and j4 (vigour) at λ 0 then 0.5.
j3 λ = 0 (no vigour, Sim B size objective only, fixed normalisation; 8k evals from v2f): VAL size objective 1,754 → −1,119, but
Sim A VAL −271,797 → **−226,188** (−45k NLL) and composition worse (0.192 vs 0.134). The size profile is then almost right (> 21 yr:
< 5 cm 253 vs 264, 5–12.7 cm 123 vs 94, 12.7–25 cm 224 vs 244, ≥ 25 cm 144 vs 162). ⇒ with mean-field cohorts, matching the stand's size
structure requires distorting the tracked-tree dynamics: a structural trade-off (model inadequacy), not a calibration problem. j4 (vigour
sub-cohorts, same objective) tests whether within-cohort heterogeneity removes it. (j3's λ = 0.5 leg was stopped: it used the start − 5 %
normalisation, which is arbitrary for a near-zero objective; fit_joint.jl now takes HBAD2_IDEAL from the end-point runs.)
j4 λ = 0 (vigour, size objective only): VAL size objective −1,249 (j3 −1,119), Sim A VAL −218,010; size profile > 21 yr 240 / 137 / 219 / 149
vs obs 264 / 94 / 244 / 162. As expected at λ = 0 the Sim A fit is sacrificed; the question is the compromise → j5 (no vigour) and j6
(vigour) at λ = 0.5 with ideal points fA = v2f train −272,975, fB = the λ = 0 train optima (−2,919 / −2,944).
Second scalarisation bug (found 11:55): with ideal fA = the start's own fA (v2f is the Sim A optimum) the normaliser (start − ideal) is
0 (j5) or tiny (j6, 306 NLL units) → invalid runs, moved to results/hbad2/aborted/. fit_joint.jl now normalises by ideal → NADIR
(env HBAD2_NADIR; nadir = the other end point) and refuses nadir ≤ ideal. j5 (no vigour) and j6 (vigour) relaunched at λ 0.5 with the
SAME constants so their points are directly comparable: ideal (fA −272,975.3, fB −2,944.0), nadir (fA −237,456.7, fB 635.0).

### 5.13 Balanced joint fits (λ 0.5, same ideal/nadir): vigour vs no vigour
| VAL | Sim A NLL | Sim B size obj. | BA err (bias) | N≥5in err (bias) | QMD err | Lorey H err | composition | > 21 yr classes (<5 / 5–12.7 / 12.7–25 / ≥25; obs 264 / 94 / 244 / 162) |
|---|---|---|---|---|---|---|---|---|
| v2f (Sim A optimum) | −271,797 | 1,754 | 12.7 % (+5.5) | 22.3 % (−0.4) | 23.6 % | 6.9 % | 0.157 | 396 / 365 / 181 / 147 |
| j5 no vigour | −262,905 | −1,213 | 13.1 % (+8.5) | 19.9 % (+15.1) | 17.6 % | 6.6 % | 0.172 | 247 / 163 / 203 / 159 |
| **j6 vigour** | **−265,212** | **−1,420** | 13.0 % (+9.3) | 19.2 % (+16.8) | 17.3 % | 6.8 % | 0.174 | 234 / 166 / 203 / 154 |

The vigour model dominates the mean-field model at the balanced point on VAL (better on BOTH objectives) → keep the vigour sub-cohorts.
j6 costs 2.4 % of the Sim A NLL vs v2f (6.6k) for most of the stand-structure gain; its VAL Sim B value (−1,420) is better than the
Sim-B-only point j4 (−1,249) → the λ = 0 end overfits TRAIN. Remaining Sim B errors: N ≥ 5 in over-predicted (+17 %), poles still
1.8× at > 21 yr, composition drifts (0.174 vs persistence 0.134). Selection (user rule): VAL decides; TEST is reported once at the end.
Vigour front (same ideal/nadir; VAL): λ 0.25 (j7): fA −260,213, fB −1,431 vs λ 0.5 (j6): −265,212 / −1,420 → moving weight to Sim B
costs 5.0k Sim A NLL for no held-out Sim B gain (−1,431 vs −1,420): λ 0.5 is at/near the knee on VAL. λ 0.75 running.

### 5.14 Composition drift — diagnosis
Sim B per-species BA (tools/hbad2/simB.jl now also writes simB_species_<set>.csv; VAL, j6): PITA BA share initial / obs / sim at ≤ 7 yr
0.118 / 0.127 / 0.177, at > 21 yr 0.164 / 0.209 / **0.321**; mean BA change per plot-visit PITA obs +1.61 vs sim +2.07 m²/ha, every
other species gains less than observed (_H +0.97 vs +0.65, _S +0.48 vs +0.18). First-interval decomposition (composition_decomp.jl;
survivor growth, mortality loss, recruit BA): in j6 the PITA RECRUITS hold 0.255 m²/ha after one interval vs 0.002–0.023 for every
other species (v2f with the data-based recruitment: PITA 0.050). Cause: the joint fit had recruitment FREE and the species-pooled size
objective carries no composition information → it cut ρ0 of nearly all species (−0.6 to −1.4; _H −1.42, LIST2 −1.16, PIEL −1.01) to
remove the sapling/pole excess and RAISED PITA (ρ0 −2.60 → −2.32, ρL −5.12 → −4.70): recruitment was used to compensate structural
error elsewhere (equifinality), handing the regeneration niche to pine.
Count-based Sim B likelihoods do not see this: j6 beats v2f on every one of them (val: species-margin 22,310 vs 27,333; plot-scale
species × class 29,117 vs 35,768; its ≥ 12.7 cm part 21,696 vs 27,565) — the drift is in BA share (few, large trees), the counts are
dominated by small trees, where j6's gain is large.
Recruitment is observed directly (sapling ingrowth; Poisson NLL maximised by the data fit), so it should stay tied to its own data:
j8 = j6 with recruitment FIXED at the Poisson fit (HBAD2_RECR_FIXED=1), same ideal/nadir, λ 0.5.
Sim A per species (v2f, VAL): PITA dD 0.596 vs obs 0.669 (−11 %), dH 0.586 vs 0.649; _H fine (0.246 vs 0.251); all species
dD −5 %, dH −7 % (survivor selection, which the mean-field cohort cannot produce; vigour sub-cohorts can).
Vigour front complete (VAL; same ideal/nadir; recruitment free): start (v2f + σv 0.3) −271,735 / 614; λ 0.75 −268,525 / −1,230;
λ 0.5 −265,212 / −1,420; λ 0.25 −260,213 / −1,431. λ 0.75 gets ~90 % of the held-out Sim B gain for < half the Sim A cost of λ 0.5;
its stand metrics equal λ 0.5 (BA 12.8 %, N≥5in 19.4 %, QMD 17.1 %, Lorey H 6.6 %, > 21 yr poles 176 vs obs 94) → knee near λ 0.75.
Both show the same pine drift (PITA BA share > 21 yr 0.314 / 0.321 vs obs 0.209): the free-recruitment problem of 5.14, not the weight.

### 5.15 Composition drift — refined diagnosis (j8, small-tree growth, ingrowth data)
* j8 (j6 with recruitment FIXED at the Poisson fit, λ 0.5): VAL fA −261,362 / fB −1,217 — dominated by j6 (−265,212 / −1,420);
  PITA BA share > 21 yr 0.291 (j6 0.321, obs 0.209), composition dissimilarity 0.226 (j6 0.228). Fixing recruitment does NOT remove
  the drift; in j8 the PITA recruits still hold 0.232 m²/ha after one interval (v2f 0.050) with the same recruit numbers as v2f.
* Cause = small-tree GROWTH of PITA: diameter increment at D 3 cm, O 0.3, RD 0.4 (mean-field cohort) — v2f PITA 0.36 cm/yr (others
  0.11–0.19), j6 1.19, j8 **1.77** (others 0.06–0.13); with σv 0.58 the fast sub-cohort grows ×2.7 more (≈ 3–5 cm/yr; observed open-grown
  saplings ≈ 0.4 cm/yr, free-growing young PITA 0.86). The species-pooled Sim B size objective asks for more ingrowth into ≥ 12.7 cm,
  and the cheapest way is to accelerate the most abundant recruiter: equifinality across species, which hands BA to pine.
* Why the data ask for more ingrowth: first-tallied macro trees vs tracked microplot saplings crossing 12.7 cm = 1.41 (5.3).
  RECONCILECD does not separate it (141,814 first-tallied macro trees coded 1 = ingrowth, 299 = 2, 1,548 missing code with mean D 24 cm).
  "Missing" tracked saplings (1,731) are mostly the SMALLEST (2.5–5 cm 4.1 % missing vs 10–12.7 cm 1.4 %), so they are not re-tallied
  crossers; the 1.41 excess stays unexplained by tracking artefacts.
⇒ Next: (i) j9 (recruitment fixed, λ 0.75) — does less weight on Sim B keep PITA small-tree growth plausible? (ii) a species-aware
  observation term for the macro ingrowth, or a constraint that keeps growth tied to Sim A (growth is observed directly on tracked trees).
* j9 (recruitment fixed, λ 0.75): VAL −267,266 / −1,061 — dominated by j7 (free recruitment, λ 0.75: −268,525 / −1,230); PITA BA share
  > 21 yr 0.281 (j7 0.314, obs 0.209). PITA small-tree growth (D 3 cm, O 0.3) j7 0.89, j9 0.93 cm/yr vs v2f 0.36 (others 0.08–0.15):
  the inflation is present at every λ with the species-pooled size objective, with or without free recruitment. Fixing recruitment is
  NOT the remedy; the objective is (→ j10: species-aware `margins` objective, λ 0 first for its ideal point).

### 5.16 Species-aware Sim B objective (`HBAD2_FB=margins`) fixes the composition drift
`margins` = plot-level size-class totals (species pooled) + plot-level species × {saplings 2.54–12.7 cm, trees ≥ 12.7 cm} totals
(both Poisson on tallied counts) + Lorey H. j10 (λ 0, vigour, recruitment fixed at the data fit): VAL BA share > 21 yr obs / sim —
PITA 0.209 / **0.182**, _H 0.351 / 0.361, LIST2 0.081 / 0.072, LITU 0.065 / 0.068, ACRU 0.040 / 0.066, QULA3 0.024 / 0.040: the pine
take-over is gone (j6 0.321, j9 0.281). Size profile > 21 yr 253 / 171 / 217 / 158 vs obs 264 / 94 / 244 / 162. BUT σv → 1.14 (fast sub-cohort
×7.2), far above the measured persistent vigour (E5: sd of ln dD ≈ 0.68), and Sim A collapses (λ 0: VAL −223,904).
→ j11 (λ 0.75) and j12 (λ 0.5) on `margins`, recruitment fixed, σv ≤ 0.7 (HBAD2_SIGV_MAX, data-based bound; recorded in meta.txt and
checked), ideal fA −272,975 (v2f train) / fB 14,619 (j10 train), nadir fA −236,467 (j10 train) / fB 24,102 (start).

### 5.17 Compromise on the species-aware objective; candidate selection (VAL) and the one TEST report
j11 (`margins`, λ 0.75, vigour σv ≤ 0.7 [binds at 0.70], recruitment fixed at the Poisson fit) and j12 (same, λ 0.5), VAL:
| | Sim A | Sim B margins | PITA BA share > 21 yr (obs 0.209) | PITA dD at 3 cm | composition | > 21 yr classes (obs 264/94/244/162) |
|---|---|---|---|---|---|---|
| j11 λ 0.75 | **−269,252** | 19,034 | 0.196 | 0.24 cm/yr | **0.165** | 341 / 251 / 183 / 163 |
| j12 λ 0.5 | −266,081 | 18,548 | 0.191 | 0.33 cm/yr | 0.169 | 324 / 251 / 191 / 161 |
(start v2f + vigour −271,735 / 23,389.) j11: Sim A only 2.5k NLL below the Sim A optimum and better than every earlier compromise
(j6 −265,212, j7 −268,525); composition close to observed for all species (_H 0.327 vs 0.351, ACRU 0.048 vs 0.040, LIST2 0.087 vs 0.081);
small-tree growth plausible for every species (0.09–0.24 cm/yr at 3 cm). The pole excess partly returns (251 vs 94; j6 166): plausible
growth + correct composition cannot also clear the poles — the remaining structural trade-off (part of it the FIA ingrowth inconsistency, 5.10).
**Selected on VAL + physical plausibility: v2g = j11** (results/hbad2/fit_v2g: theta.csv, recr.csv = the Poisson fit, meta.txt).

TEST (reported once; not used for any choice):
* Sim A (same tracked-tree cohorts, subplot sites): v2g total −265,505 (mort 32,918, ln D −158,459, ln H −139,964; σ_D 0.121, σ_H 0.155);
  v2f −267,776 (mort 32,350); v1 (FIA regressions, best variant) −246,242 (mort 32,533; σ_D 0.128, σ_H 0.191). v2g beats v1 overall,
  in diameter and in height, but NOT in mortality (v2f did; the joint compromise costs mortality accuracy).
* Sim B (free runs, 439 clean natural test plots, 1,552 later visits; median |error| (bias)): BA 11.6 % (+14.5) [persistence 24.3 %,
  v2f 12.6 %], N ≥ 5 in 18.3 % (+12) [18.2 %, 20.2 %], QMD 18.5 % [16.0 %, 20.5 %], Lorey H 6.6 % [13.1 %, 7.1 %], composition 0.159
  [0.138, 0.155]; > 21 yr poles 240 vs obs 73 (v2f 351). Persistence still wins on QMD and composition and ties N ≥ 5 in.
v2g figures: results/hbad2/figures/v2g_{species_curves,allocation_vs_obs,simB_scatter}.png. Timing check (timing_curves.jl): corr with
observed free-growing increments dD 0.81, dH 0.94, dH:dD 0.80 (v2c 0.86 / 0.955 / 0.82 — the joint compromise constrains the curves a little).
Decoded v2g export (tools/hbad2/export_params.jl): results/hbad2/fit_v2g/params_{species,global,recruitment}.csv. At a bound: LITU rH = 3.0 (form-2 upper bound), LIST2 sdimax = 2000 (upper bound; the cap never binds anyway).

## 6. Integration into Pan (2026-10-17, user-approved tree-based loader)
* Stage 1 (merged into `refactor` 669e54b): plugin src/plugins/HBAD2 (`succession_plugin: hbad2`), Data.load_tree_cohorts (curated_trees →
  subplot sites, cohorts by species key × birth class × size class, 3 vigour sub-cohorts), exact parity with this prototype (Sim B 1860 sites ×
  25 yr: max rel diff 0; from the loader 7.2e-15; Sim A 0) — tools/hbad2_parity.jl, docs/hbad2_plugin.md (in the Pan repo).
* Stage 2 (merged, dc5b082): scenario engine runs hbad2 — roster results/params_yaml/r4_hbad2_v2g.yaml, each FIA plot = 4 subplot sites,
  plot-level metrics/removal (residual BA/TPA/SDI met exactly), planting with a planted flag, cleanup of cohorts < 1e-3 trees/ha in scenario
  runs; scenario hash r4_hbad2_v2g 87b47235… (identical t=1/t=2). Timing (2 threads): 10 plots × 100 yr in 2.6 s (HBAD v1 0.9 s).
* Calibration stays in tools/hbad2 (this prototype harness); Pan's generic fit_params path is not used for hbad2.

### 5.18 v2g mortality shape and the frozen-mortality test (j13)
v2g VAL deaths obs / sim by D class: ≤ 5 cm 1,757 / 2,172 (+24 %), 5–12.7 1,185 / 1,347 (+14 %), 12.7–20 4,011 / 4,021, 20–30 1,632 / 1,325
(−19 %), 30–45 739 / 519 (−30 %), > 45 cm 246 / 108 (−56 %); small trees in the open (O ≤ 0.2) 693 / 1,091. Total right (9,570 / 9,491),
size shape wrong: the joint compromise uses SAPLING mortality as the lever to clear the free-run pole excess and lets large trees live (few
large-tree deaths → little weight in the Sim A binomial), which also fits the + 12–15 % free-run BA bias and the worse TEST mortality NLL
(v2g 32,918 vs v1 32,533 vs v2f 32,350). j13 = j11 settings with the mortality parameters FROZEN at the Sim A optimum v2f
(`HBAD2_FREEZE=mbg,mD,msup,G:mD2,G:mSD,G:pM`, 39 coordinates frozen, 135 searched): can Sim B be served by growth/competition/vigour alone?
j13 result (VAL): Sim A total −268,472 (v2g −269,252) but mortality NLL **33,014** (v2g 33,441; v2f 32,872); deaths by D class obs / v2h
(v2g): ≤ 5 cm 1,757 / 1,711 (2,172), 5–12.7 1,185 / 1,256 (1,347), 12.7–20 4,011 / 4,494 (4,021), 20–30 1,632 / 1,725 (1,325), 30–45 739 / 814
(519), > 45 cm 246 / 208 (108); total 9,570 / 10,208 (9,491). Sim B margins objective VAL 18,748 (v2g 19,034); stand medians BA 13.6 % (+11.0),
N ≥ 5 in 20.0 %, QMD 22.2 %, Lorey H 6.7 %, composition 0.171; > 21 yr poles 295 (v2g 251).
DECISION (made on VAL + plausibility, before any TEST look at j13): **v2h = j13 replaces v2g as the candidate.** Neither dominates on VAL
(fA −780, fB +286 in v2h's favour); v2h has the observed size shape of mortality (large-tree mortality restored: > 45 cm 208 vs 246, v2g 108),
which governs long-run projections, while v2g reached its Sim B fit by killing saplings and sparing big trees — a plausibility criterion that
should have been checked before v2g was selected. v2g's TEST report (5.17) stays as written; TEST is now reported once for v2h.
TEST (reported once for v2h): Sim A total −264,319 (mort **32,441**, ln D −157,558, ln H −139,202; σ_D 0.123, σ_H 0.157) — v1 −246,242
(mort 32,533), v2g −265,505 (mort 32,918), v2f −267,776 (mort 32,350): v2h beats v1 overall, in diameter, height AND mortality. Sim B
(439 test plots, 1,552 visits): BA 12.2 % (+12.7) [persistence 24.3 %], N ≥ 5 in 18.7 % (+11.7) [18.2 %], QMD 20.1 % [16.0 %], Lorey H 6.7 %
[13.1 %], composition 0.163 [0.138]; > 21 yr poles 283 vs obs 73. Persistence still wins QMD and composition and ties N ≥ 5 in.

### 5.19 Recruitment level (v2h sensitivity) and a fitted global level κ (j14)
v2h free runs (VAL) with all recruitment rates × m: m 1 / 0.75 / 0.5 — BA bias +11.0 / +7.7 / +3.4 %, N ≥ 5 in bias +10.2 / +7.1 / +3.2 %,
QMD error 22.2 / 20.0 / **17.6 %** (persistence 19.0 %), composition 0.171 / 0.163 / **0.154**; > 21 yr classes at m 0.5: 203 / 193 / 164 / 147
(obs 264 / 94 / 244 / 162): halving recruitment fixes most stand-level biases and halves the pole excess, but saplings drop below observed and
the 12.7–25 cm deficit remains (pole graduation). → one global level κ (ln κ shifts every species' ρ0; species / seed / light structure stays
the data fit — no equifinality across species) as a nuisance parameter for the mismatch between recruits entering at exactly 2.54 cm and the
observed ingrowth (median 3.6 cm, net of in-interval deaths): `HBAD2_RECR_KAPPA=1`. j14 = j13 settings (mortality frozen) + κ, warm start j13.
simB.jl now uses a fit directory's own recr.csv (v2g/v2h recr.csv == the Poisson fit exactly, so their evaluations are unchanged).
