# HBAD3 — design study: fewer, process-bound parameters for climate sensitivity

Branch `hbad3` (worktree `/workspace/landis_hbad3`, from `hbad-v2` 230662b). Started 2026-10-18.
The user's goal: "think about how to reduce the number of these parameters so that the model (probably hbad3) is amenable to climate
sensitivity (more process-bound, not too empirical)". The v2 kernel, harness and data stay as they are; HBAD3 is designed and tested here.

## 1. Starting point: where v2's ~210 numbers come from
v2j has 12 species keys × 15 per-species parameters, 8 globals and 12 × 6 recruitment numbers. Of these, 12 × 14 + 7 = 175 are
calibrated in Sim A and 36 more (or one κ) in the joint fit. They are all empirical shape coefficients:
- 6 growth-curve shapes per species (rD cD Dmax rH cH Hmax).
- 4 competition responses (gOD gRD gOH gRH).
- 3 mortality coefficients (mbg mD msup).
- An SDImax.
- 3 recruitment coefficients.

None of them is tied to a resource or a physiological rate, so there is no place for temperature, water or CO₂ to act. Any climate
response would have to be bolted on as more empirical multipliers per species.

## 2. Design principles for HBAD3
1. **Keep what v2 established empirically.** The state stays N, D, H, A plus vigour sub-cohorts, with no biomass. Subplot sites keep
   competition local. Height-structured light competition (overtopping) stays the main competition axis. Recruitment enters as size-class
   cohorts.
2. **Every rate is resource × efficiency × modifier.** Growth is light capture × an efficiency. Climate enters as modifiers of that
   capture (temperature, water balance, CO₂), never as free per-species fudge factors.
3. **Species differ through measurable traits**, mapped to process parameters by a few global coefficients:
   - shade-tolerance class (FIA crown class), wood density (REF_SPECIES), observed maximum H and D, longevity, softwood/hardwood;
   - later, drought tolerance and thermal envelope (species climate range).
   Only a few per-species parameters stay free: a growth efficiency and a fecundity.
4. **One stress pathway links growth and mortality.** Mortality responds to the realised growth relative to the open-grown potential, so
   light, water and temperature stress kill through the same pathway. It gets its own sensitivity, not the growth coefficient itself;
   see S2d.
5. **Climate response is tested out of sample**: on drought periods and climate blocks held out, not only on random plots.

## 3. Empirical findings so far (GA+FL; v2j / v2fv5 residuals; train and val agree)
**S1 — covariates.** For each of the 8,985 plots: L4 → PRISM 1991–2020 normals (°C / mm / hPa), Thornthwaite PET, climatic water
deficit, and soil AWC. Every interval gets its own mean climate and anomaly. Tool: `scripts/hbad3/climate_join.py`; outputs
`results/hbad3/{plot,interval}_climate.csv`.
- The spatial gradient is wide: annual T 12.4–24.7 °C, CWD 53–291 mm, AWC 6–229 mm.
- Interval anomalies are small: T ±1 °C, CWD ±33 mm sd.

**S2 — climate signal in v2j residuals** (`scripts/hbad3/climate_signal.py`, `climate_period_check.py`). Ratios are observed/simulated
increment of survivors and observed/expected deaths.
- **Water-balance anomalies → mortality.** This is the robust signal. Within each measurement period, the wet → dry terciles of the
  growing-season CWD anomaly give mortality O/E 0.85 → 0.95 → 1.06 (train) and 0.92 → 0.94 → 1.04 (val).
- **Growth vs climate.** Within periods the growth signal is small: about 2–3 % between wet and dry terciles. Between periods it is
  large: diameter growth in 2007–10, the Southeast drought, runs about 15 % below ≥ 2015. That difference cannot be separated from
  time trends (CO₂, protocol) with FIA alone.
- **Normals (the spatial gradient)** show no monotone growth or mortality signal once v2's species and region structure is in place. The
  exception is soil AWC: the driest AWC tercile grows 5–12 % less.
- **Consequence:** in GA+FL, climate parameters are identifiable only as a few *global* sensitivities (a water-stress mortality slope, a
  water modifier on growth). They are not identifiable per species. The temperature response needs Region 4's wider gradient (direction 3)
  or external trait data.

**S2c — is crowding a water-sharing effect?** (`scripts/hbad3/crowding_water.py`) The model's crowding error does not change between
water classes (AWC, CWD normal, CWD anomaly). There is no evidence that symmetric crowding is mediated by water in this humid region.
HBAD3 therefore keeps water as a separate modifier, not as a belowground competition process.

**S2d — the shape of the competition response** (`scripts/hbad3/growth_surface.py`; all-survivor cohorts, so no selection effect). Mean
annual diameter increment in cm/yr, observed / v2fv5:

| O \ RD | ≤ .25 | .25–.5 | .5–.75 | > .75 |
|---|---|---|---|---|
| ≤ 0.1 | 0.570 / 0.540 | 0.494 / 0.475 | 0.444 / 0.408 | 0.373 / 0.334 |
| 0.1–0.3 | 0.357 / 0.370 | 0.333 / 0.323 | 0.335 / 0.287 | 0.255 / 0.225 |
| 0.3–0.6 | — | 0.185 / 0.175 | 0.181 / 0.144 | 0.158 / 0.126 |
| > 0.6 | — | — | 0.121 / 0.081 | 0.117 / 0.064 |

- The observed response to overtopping **saturates**: 0.57 → 0.36 → 0.19 → 0.12. v2's exponential e^{−gOD·O} keeps falling, so the most
  suppressed cohorts are under-predicted by 30–45 %.
- The crowding (RD) effect is weaker than the model's.
- A likely cause is that v2 couples suppression mortality to the *growth* coefficient gOD (v = e^{−gOD·O}). Mortality data want a strong
  shade signal, and the shared coefficient forces growth to follow it.
- HBAD3 therefore needs a saturating light response with a shade-tolerance-dependent floor, and its own stress-mortality sensitivity.

**S3 — information content of the per-species parameters** (`tools/hbad3/fit_tied.jl`, ladder `tools/hbad3/run_ladder.sh`). In a Sim A
fit, each per-species parameter is either species-specific, one global value, or a linear function of standardised traits. Val NLL and
BIC are compared against v2fv5 (175 coordinates; val −272,655.4). Results: §5.

## 4. HBAD3 process formulation (draft, to be tested piece by piece)
**State.** Per cohort: N, D, H, A and vigour quantile z, as in v2. Per site: L4 climate (annual, from PRISM or a scenario), soil AWC.

**Light.**
- Cohort leaf area per tree is proportional to the Reineke size D^1.605. This is the SDI exponent, consistent with the pipe model, so a
  site's SDI is its leaf-area proxy.
- Relative light at a cohort's crown is L_i = exp(−k · SDI_above,i / S₀), with S₀ a fixed scale (e.g. 1000).
- The light response is f_L = L / (L + K_ST), with K_ST = K₀ · e^{κ_ST (ST − 3)}. That is 2 globals plus the data-derived shade
  tolerance ST, replacing gOD and gOH (24 coordinates).

**Potential growth.**
- Diameter: dD_pot = ε_D,sp · φ(D) with φ the v2 power-exponential, its shape global or trait-based. The scale ε_D is per species, or
  WD/ST-based (S3).
- Height follows a species H–D allometry fitted to FIA directly (H* (D), fixed, not calibrated). Shade elongation gives
  dH = (dH*/dD) · dD · (1 + η (1 − L)), with η global.
- The relaxation toward H* is global. This replaces rH, cH and Hmax (36 coordinates) with 1–2 globals plus fixed allometries, and
  reproduces "crowding shifts growth from diameter to height" through light rather than RD.

**Climate modifiers** (multiplicative on growth, as 3-PG / PnET do):
- **Water.** f_W = 1 − ω · max(0, CWD_rel), where CWD_rel is the interval CWD over the soil-water capacity (AWC). ω is global and
  identifiable from the drought period and the AWC gradient (S2).
- **Temperature.** f_T is a species thermal-envelope parabola, its optimum and width from the species' FIA occurrence × PRISM normals
  (data-derived, not fitted). It is calibrated in Region 4 at most as one global width multiplier.
- **CO₂.** f_C = 1 + β ln(CO₂/350). β is fixed from the literature (e.g. 0.3–0.6): FIA cannot identify it, so it is stated as a
  scenario assumption.

**Mortality.**
- Intrinsic: from longevity (fixed).
- Size: the global mD and mD2 U-shape.
- Stress: h_s = m_s · (1 − g_rel)^γ, where g_rel = realised dD / open-grown dD_pot of the same cohort. This includes the light, water and
  temperature modifiers, and vigour z enters through the realised growth. m_s and γ are global, possibly with an ST slope.
- Self-thinning: the species SDImax cap, data-derived or one global.
- Drought kills through g_rel, which addresses the S2 mortality signal without a separate drought-mortality term. A test of HBAD3 is that
  the O/E pattern by CWD anomaly disappears.

**Recruitment.** R = F_sp · BA_seed^{ρS} · f_L(L; ST) · f_W · f_T, with F_sp per species (fecundity). The light response uses the same
f_L as growth; ρS is global.

**Planting.** pD and pH as in v2: genetic and silvicultural gain, 2 globals.

**Parameter budget (12 keys).** About 2–3 per species (ε_D, F_sp, possibly a mortality scale) plus about 15 globals: k, K₀, κ_ST, η,
relaxation, φ shapes (2), ω, m_s, γ, mD, mD2, ρS, pD, pH, σv. That is ≈ 40–50 calibrated numbers instead of 211. Every per-species
process quantity that is not fitted comes from a measurable trait or a data table: ST, WD, H–D allometry, longevity, thermal envelope,
SDImax.

## 5. Results log
(appended as the experiments finish)

**S4 — height from diameter through the H–D allometry** (`scripts/hbad3/hd_allocation.py`; allometries in `results/hbad3/hd_allometry.csv`).
- Per key, a Chapman–Richards H*(D) is fitted to the train trees: 3 numbers per key, taken from the data, not calibrated.
- Rule, given the observed D2: H2 = H1 + ΔH*(1 + η(1−L)) + λ·dt·(H*(D1)(1 + μ(1−L)) − H1), with L = e^{−kO}.
- Fitted globals: η 0.149, λ 0.0148/yr, k 17.3, μ 0.49 (shaded trees 49 % taller at the same D).
- Val sd of ln(H2pred/H2obs):

  | predictor | val sd |
  |---|---|
  | no growth | 0.229 |
  | allometry only | 0.133 |
  | **rule** | **0.108** |
  | v2fv5, unconditional | 0.118 |

- There is no bias by overtopping class.
- This is conditional on the observed D2, so the true test is the kernel fit. Still, height looks derivable from diameter with 4 globals
  instead of 5 × 12 per-species parameters.

**S1b — process water index** (`scripts/hbad3/water_balance.py` → `results/hbad3/interval_water.csv`).
- A monthly soil bucket (capacity = plot AWC) with PRISM ppt and Thornthwaite PET gives the growing-season (Apr–Oct) AET/PET ratio.
- Interval mean 0.923; the minimum year averages 0.82.
- It is planned as f_W = (AET/PET)^ω on growth, reaching mortality through g_rel.

## 6. Checkpoint / resume (2026-10-18, container restart)
- **Committed:** scripts, tools, the design doc, and the small outputs.
- **Not committed — regenerate:**
  - `resid_*.csv`: `tools/hbad3/export_resid.jl v2j|v2fv5` with the v2 env of each fit (v2j: HBAD2_SIGV_MAX=0.7; v2fv5: 0.5).
  - `interval_climate.csv`: `climate_join.py`.
  - `results/hbad2/*.arrow`: copied from /workspace/landis_hbad2/results/hbad2.
- **Environment:** `source wt_env.sh` before Julia. For Python, use a scratch venv with numpy, pandas, pyarrow, scipy and duckdb (the
  deck's venv has no scipy). Create it with `python3 -m venv <scratch>/h3venv && <scratch>/h3venv/bin/pip install numpy pandas pyarrow scipy duckdb`.
- **The S3 ladder was stopped** during T1, at 1,200 evaluations with no improvement on the start yet. It takes about 230 ms per
  evaluation at 4 threads, so the whole ladder is ≈ 2 h. Rerun with `bash tools/hbad3/run_ladder.sh` (EVALS / THREADS env).
- **Next steps:**
  1. Rerun the ladder.
  2. Write the HBAD3 kernel `src/plugins/HBAD3/HBAD3.jl` per §4: saturating light response, H from the H–D allometry, g_rel stress
     mortality, f_W from interval_water. Use the same Sites SoA, so it fits the v2 harness, and add per-site fW.
  3. Write `tools/hbad3/fit3.jl` reusing the tie machinery of fit_tied.jl.
  4. Compare val against v2fv5 (−272,655.4) and against T5.
  5. Run the climate holdout tests: the 2007–10 drought period, and AWC/CWD blocks.

## 7. Results (2026-10-18/19)

### 7.1 GA+FL in-sample (Sim A val; v2fv5 = −272,655 with 175 coordinates)
| fit | coordinates | val | note |
|---|---|---|---|
| H3a (stress (1−g)^γ) | 36 | −269,796 | 97.3 % of v2's gain over null; diameter is the main loss (+2,034) |
| H3b (Kobe on relative growth) | 36 | −269,788 | same as H3a; light response ≈ linear (large K); SDImax at its bound |
| H3c (diagnostic: mortality decoupled from growth, SDImax fixed 1,500) | 36 | −270,087 | k halves (6.5 → 3.2); shaded growth 2–3× low (was ~100×) |
| H3d (Kobe on absolute growth) | 35 | −269,520 | worse |

v2 ladder (v2 kernel, tied coordinates):

| ladder step | coordinates | val | change vs full fit |
|---|---|---|---|
| T1 (competition tied to ST) | 135 | −271,906 | |
| T2 (mortality tied to traits) | 137 | −271,474 | |
| T3 (shapes global) | 131 | −270,796 | |
| T4 (only rD and rH per species) | 55 | −271,132 | +1,524 |
| T5 (growth rates from traits) | 39 | −265,531 | the species growth rate is NOT derivable from traits |

### 7.2 VIABILITY OUTSIDE GA+FL (the decisive test; user: "quickly determine viability elsewhere")
GA+FL-fitted models are scored unchanged on Region 4 plots outside GA+FL: R4 split val, 340,642 tracked trees, 13 L3s. Tools:
`tools/hbad3/transfer_eval.jl`, `scripts/hbad3/transfer_climate.py`, `tools/hbad3/fit_climate.jl`.

- **Neither GA+FL calibration transfers cleanly.** Growth ratio (obs/sim) is 1.05 in GA+FL vs 1.14 (v2fv5) / 1.16 (H3b) outside. Mortality O/E is
  1.00 overall, but 0.78–1.36 by L3.
- **Climate-structured residuals are the same for both models.**
  - Mortality O/E vs drought anomaly: 0.86 → 1.11 from wet to dry.
  - Mortality O/E vs AET/PET: 1.08 → 0.82 as the water index rises.
  - Cold L3s (< 14 °C): O/E 0.82.
  - Growth ratio vs soil AWC: 1.07 → 1.23.
  - HBAD3's GA+FL water term (ω 0.27) does not capture any of this. HBAD3 is no more transferable than v2 on any axis.
- **4-coordinate refits on R4 outside GA+FL, from H3b:**

  | refit | R4 val gain | GA+FL val | finding |
  |---|---|---|---|
  | climate (ω, drought mW, temperature τg / τm on spatial normals) | −775 | **+437** | removes the climate-structured mortality bias (0.10 → 0.02–0.05), but hurts GA+FL |
  | control (shifts of eD, mbg, ln ms, ln k; no climate) | −3,483 | −117 | most of the transfer error is the mortality split (stress ms ×7.5, background −0.23), which GA+FL alone does not identify |
  | both | −3,761 | **+781** | the climate terms cost GA+FL ≈ 900: coefficients fitted on spatial normals absorb regional confounds and do not transfer |
  | anomaly-driven climate (W / W_normal) + control | see 7.3 | | |

- **Consequences:**
  1. Calibrate region-wide (R4 incl. GA+FL), not on GA+FL alone. The mortality split and the growth level need the region's variety.
  2. Climate terms on spatial normals are not viable as climate sensitivity.

### 7.3 Anomaly-driven climate (the form climate scenarios use)
Site W = interval AET/PET ÷ the plot's own 1991–2020 normal; no temperature terms. The run is warm-started at the control optimum, so the
change is what the 2 anomaly terms add on top of the regional re-levelling.

| | R4 train | R4 val | GA+FL val |
|---|---|---|---|
| change | −157 | +316 (mortality −132, D/H +449) | −87 (mortality −7) |

- Fitted: ω → 0 (no growth response); mW 0.39 (hazard ×1.47 per 10 % relative AET/PET drop).
- **Drought-anomaly mortality is the one climate effect that is consistent in and out of GA+FL.** It is small at the likelihood level (≈ 0.02 %).
- The water response of growth is not identifiable in these remeasurements.

### 7.4 Viability verdict (2026-10-19)
1. **GA+FL-only calibrations (v2 or HBAD3) are not viable elsewhere.**
   - Growth is under-predicted by ≈ 15 % out of region, against 5 % in region.
   - Mortality is biased by −22 % to +36 % by L3.
   - Most of the error is a mortality split (stress vs background) and a growth level that GA+FL alone does not identify. A 4-coordinate
     regional re-levelling fixes most of it, and also slightly improves GA+FL.
   - → **Calibrate region-wide (R4 incl. GA+FL), then vet in GA+FL.**
2. **HBAD3 transfers no better than v2.**
   - Its advantages are 5× fewer coordinates (cheaper region-wide calibration) and explicit hooks for climate drivers.
   - In-sample it is 2,600–2,900 worse than v2 in GA+FL.
3. **Climate sensitivity.**
   - Spatial-normal climate terms absorb regional confounds and do not transfer.
   - Anomaly-driven drought mortality transfers but is small.
   - FIA remeasurements identify little climate response of growth. The model's climate sensitivity would rest mostly on the drought-
     mortality term plus external (literature / physiological) priors.

### 7.5 Process grounding per region (the user's meaning of "viability": the model's processes exist in data from other regions)
The user's correction (2026-10-19): viability means *calibratable elsewhere*: each process must be grounded in data from other regions. It
does NOT mean that a GA+FL calibration transfers (that is §7.2–7.4, kept for the record).

Data-only checks (`scripts/hbad3/process_checks.py`, 40 s; `results/hbad3/process_checks.txt`) in 5 regions: GA+FL, APP (Appalachian L3s
8.4.x), IP (8.3.3), WSP (AL/MS plains + Mississippi alluvium), NCSC (NC/SC Piedmont and plains).

| process | GA+FL | APP | IP | WSP | NCSC |
|---|---|---|---|---|---|
| P1 light: relative growth at O .1–.3 / .3–.6 / .6–1 | .69/.50/.41 | .68/.48/.35 | .65/.46/.43 | .66/.43/.32 | .67/.44/.34 |
| P2 crowding: open-grown growth, RD < .25 → .75–1 | −8 % | −12 % | −6 % | −5 % | −16 % |
| P3 height from D (sd): none → allometry → +4 globals | .182→.137→.122 | .157→.122→.113 | .155→.120→.110 | .226→.170→.150 | .204→.141→.124 |
| P4 mortality, slowest → fastest growth quintile (prev. interval) | 3.6→0.46 % | 3.4→0.41 % | 3.6→0.40 % | 4.6→0.67 % | 4.0→0.46 % |
| P5 drought-anomaly mortality, wet → dry | +15 % | +16 % | +14 % | +22 % | +5 % |
| P6 recruitment: seed source present / RD < .25 vs .5–.75 | ×3.4 / ×2.8 | ×2.1 / ×3.5 | ×3.1 / ×1.7 | ×1.9 / ×3.2 | ×2.1 / ×3.6 |

- **Every process is grounded in every region, with a similar shape.**
- **P1 corrects §3 S2d.** At tree level a plain exponential e^{−aO} (a 1.7–2.1) fits as well as the saturating form in every region, so the
  half-saturation K is not needed. The cohort-level "saturation" was an artefact of the model's own O.
- **P3:** η ≈ 0 in 4 of 5 regions, so 3 globals suffice.
- **Next (layer 2): local calibration of one HBAD3 structure in GA+FL, APP and WSP** (fit3 R_<region>). The structure follows these
  checks: exponential light (lnK const), Kobe stress mortality, σv ≤ 0.7, SDImax fixed, H–D fitted to each region. The checks are: share of
  the gain over null, parameters off their bounds, and residual diagnostics.

### 7.6 Process replacements of the empirical equations (data only, 5 regions, held-out half of the plots)
`scripts/hbad3/process_replace.py`, `process_replace2.py`; `results/hbad3/process_replace*.txt`.

**R1 potential growth.** Empirical r(D/20)^c e^{−D/Dmax} (3/key) vs a stem-volume balance with one efficiency ε per key:
dV = ε (D/20)^b e^{−aO}, dD = dV/(2DH).
- Held-out sd of ln D2: process ≤ empirical in 4 of 5 regions (tie in IP).
- Globals agree across regions: b 1.84–1.93 (pipe model), a 2.4–2.9; maintenance ρ ≈ 0, so it is dropped.
- With a hydraulic limit ×(1 − (H/1.1 Hmax)^θ), large-tree growth is unbiased: > 40 cm, GA+FL .424 vs .421 obs, NCSC .415 vs .422,
  WSP .532 vs .523 (without the limit, 10–25 % over).

**R2 background mortality.** Empirical U-shape (2/key + 1) + stress vs h = m0/LONGEVITY + ms e^{−β g}(D/20)^ζ (+ ST slope).
- 6–8 globals and 0 per key capture 78–93 % of the per-key model's deviance gain.
- Consistent across regions: ζ −0.56…−0.79 (small trees have small reserves); β 2.0–3.4; m0 1.0–1.6; σST < 0 in all regions
  (tolerant species survive low growth).
- The large-tree height risk is not needed (→ 0).

**R3 recruitment.** Per-key Poisson (3/key) vs F_k × light window e^{−λ0 e^{λ1(ST−3)} RD} × seed presence.
- Held-out deviance ≤ empirical in 3 regions, ≈ in NCSC, slightly worse in IP.
- λ0 3.8–5.3; λ1 < 0 everywhere.
- The seed-source effect is presence only (ρS ≈ 0.02–0.08).

**R4 height–diameter curve.** Per-key curve (3/key) vs a trait curve: Hmax_k (99.5 % height) + initial slenderness s + shape c (2 globals).
- Held-out sd equal (≤ 0.0004) in all regions.
- s 0.8–1.14 m/cm, c 0.78–0.93.

**Local calibrations of HBAD3** (structure of §7.5, `fit3_R_<region>`):

| region | coordinates | val gain over null per tracked tree | at bounds |
|---|---|---|---|
| GA+FL | 33 | 0.750 | none |
| APP | 30 | 0.663 | ω = 0 (no growth–water effect in the humid mountains) |
| WSP | 27 | 0.763 | none |

## 8. HBAD3P — the process-grounded model (`src/plugins/HBAD3/HBAD3P.jl`, `tools/hbad3/fit3p.jl`)
User goal (2026-10-19):
- Informative for harvest: height, BA, density.
- Responsive to climate.
- Minimise empirical equations, which carry implicit conditions, in favour of process grounding.
- Finally calibrate HBAD3 on every R4 ecoregion.

**Equations** (header of HBAD3P.jl):
- **growth:** dV = ε_k (D/20)^b · 20^1.605 · e^{−aO} · e^{gR·RD} · max(0, 1 − (H/1.1 Hmax_k)^θ) · W^ω · e^{σv z}; dD = dV/(2DH)
- **height:** trait curve H*(D) from Hmax_k with s, c; light rule with η, λ, μ, kh
- **mortality:** e^{pM}[m0/LONGEVITY + ms e^{σST(ST−3)} e^{−β g}(D/20)^ζ] e^{mW(Wn − W)/0.1} + age term
- **self-thinning:** at a constant SDImax
- **recruitment** (free runs; to be added to the kernel): F_k × light window × seed presence

**Parameter count:**
- Per key: ε_k (+ F_k for recruitment).
- Traits per key: Hmax, ST, LONGEVITY.
- Globals: b a gR θ ω | s c η λ μ kh | m0 ms σST β ζ mW | pD pH pM σv = 21 (+ λ0 λ1 ε0 for recruitment).
- An R4 region with 30 keys has ≈ 30 + 30 + 24 calibrated numbers, against ≈ 450 for v2.

**Climate pathways:**
- AET/PET (monthly bucket with soil AWC) on growth (ω).
- Drought anomaly on mortality (mW).
- Shared through the growth-stress term.
- CO₂ / temperature: from the literature, as scenario assumptions; the FIA data do not identify them.

### 8.1 HBAD3P results (2026-10-19) — `fit3p_P_<region>`, `freerun_P_<region>.out`
**Sim A (val)**

| region | model | coordinates | val | gain / tree | at bounds |
|---|---|---|---|---|---|
| GA+FL | v2fv5 | 175 | −272,655 | 0.770 | — |
| GA+FL | HBAD3 | 33 | −269,774 | 0.750 | — |
| GA+FL | **H3P** | **33** | **−273,469** | **0.777** | none |
| APP | HBAD3 | 30 | −217,762 | 0.663 | ω |
| APP | **H3P** | 30 | **−218,694** | **0.675** | ω = 0 |
| WSP | HBAD3 | 27 | −185,459 | 0.763 | — |
| WSP | **H3P** | 27 | **−188,445** | **0.791** | σv = 0.7 (cap) |

H3P beats v2 in GA+FL with 1/5 of the coordinates; the null baselines are identical. It beats HBAD3 everywhere.

**Free runs** (first → last visit of val plots, ≈ 10–15 yr, process recruitment fitted on train ingrowth). Bias / median |rel err|; persistence in brackets.

| region | plots | BA | stems ≥ 12.7 cm | QMD | Lorey H |
|---|---|---|---|---|---|
| GA+FL | 434 | −0.7 % / 16.8 % (−30.5 / 32.4) | −10.8 % / 24.7 % (−24.5 / 28.0) | −1.1 % / 8.6 % (−10.8 / 12.5) | −2.0 % / 7.5 % (−18.4 / 16.9) |
| APP | 434 | +3.4 % / 12.7 % (−24.5 / 23.4) | −10.2 % / 21.5 % (−13.9 / 17.9) | +2.3 % / 8.7 % (−9.8 / 11.1) | 0.0 % / 6.1 % (−13.0 / 12.1) |
| WSP | 518 | +2.7 % / 18.9 % (−31.3 / 34.0) | −6.1 % / 24.4 % (−27.7 / 27.3) | −0.9 % / 8.4 % (−10.9 / 13.5) | +2.3 % / 7.4 % (−17.9 / 19.2) |

- **BA, QMD and Lorey H are within ±3.4 % bias in all 3 regions**, and the error is ≈ half that of persistence.
- **Known weakness:** stems ≥ 12.7 cm are under-predicted by 6–11 % (in-growth into the merchantable class). In APP the median error is
  worse than persistence (21.5 vs 17.9 %).
- No recruit classes were dropped.
- Recruitment globals agree across regions: λ0 5.0–6.4, λ1 −0.25…−0.63, ε0 0.34–0.74.

### 8.2 CHECKPOINT / resume (2026-10-19, container restart)
- Branch hbad3 @ this commit.
- **Environment:**
  - `source wt_env.sh`
  - Region env: `HBAD2_FORM=2 HBAD2_SIZE_SPLIT=5,12.7,25,40 HBAD2_SITE=subplot HBAD2_VIGOUR=1 HBAD2_SIGV_MAX=0.7 HBAD2_DATA_DIR=results/hbad2_r4 HBAD3_WATER_CSV=results/hbad3/interval_water_r4.csv HBAD2_REGION=<GAFL|APP|IP|WSP|NCSC|L3:x.y.z>`
- **Regenerate untracked inputs if missing:**
  - `results/hbad2_r4/*.arrow`: copy from /workspace/landis_hbad2_r4/results/hbad2_r4/intervals/ (the Pan R4 interval product, branch hbad2-r4).
  - `results/hbad3/{plot,interval}_climate_r4.csv`: `scripts/hbad3/climate_join.py <repo> results/hbad2_r4/intervals.arrow _r4`
  - `interval_water_r4.csv`: `water_balance.py`, same args.
  - Python: a scratch venv with numpy pandas pyarrow scipy duckdb.
- **Tools:**
  - `tools/hbad3/fit3p.jl <tag>`: Sim A fit of one region.
  - `tools/hbad3/freerun3p.jl <tag>`: free runs + recruitment fit.
  - `scripts/hbad3/process_checks.py`, `process_replace*.py`: data checks.
- **Next steps (user goal: HBAD3 calibrated on EVERY R4 ecoregion, as process-grounded as possible; informative for harvest H/BA/density;
  responsive to climate):**
  1. Look at the in-growth weakness into 12.7 cm (recruit entry size / sapling growth) and the σv cap in WSP; then freeze the H3P structure.
  2. Port H3P to Pan: plugin + calibration (Sim A + recruitment fit + free-run check), with parity gates, on a new branch from refactor
     (hbad2-r4 holds the R4 split + recruitment tool).
  3. Calibrate every R4 L3 (14; the small ones 15.4.1, 8.3.2, 8.5.2 need pooling or a lower key threshold); report val per L3 + free runs.

### 8.3 Crown-ratio state + ingrowth diagnosis (2026-10-22)
- **Fit Pcr_GAFL** (crown-ratio state, 5-point vigour quadrature, maintenance ρ, diffuse-light floor f0, own survival light coefficient am;
  warm start from Prf): **val −278,579** (train −276,772), 42 coordinates (12 per key + 30 global), no parameters at bounds. This is the best
  GA+FL fit (v2fv5 −272,655 with 175).
- **Ingrowth diagnosis** (`tools/hbad3/crossing3p.jl Pcr_GAFL`; tracked val cohorts starting below 12.7 cm, one interval):
  - Survival is right (sim / obs 1.026). Crossing 12.7 cm is too low (0.71): small trees grow too slowly.
  - The deficit is concentrated in OPEN-grown cohorts (overtopping O < 0.1): diameter growth sim / obs 0.66–0.84 in every size class, natural
    and planted alike. Partly and fully shaded cohorts are mostly within 10–15 % (0.73–0.99).
- **Suspect: gR = +0.43.** The crowding term e^{gR·RD} increases growth with stand density, which is not physical; it likely absorbs site
  quality (dense stands on good sites). Its size (≈ 27 % between RD 0.2 and 0.8) matches the open-grown deficit of young, low-density stands.
  - Test: refit with gR fixed at 0 (`HBAD3P_FIX=gR:0.0`, warm start from Pcr) → `fit3p_Pgr0_GAFL`, then rerun crossing3p.
  - If val holds and crossing improves, keep gR ≤ 0 and add site quality as an explicit site input.
