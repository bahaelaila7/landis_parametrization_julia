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
