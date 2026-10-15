# PnET parameter containers + the FIXED (non-fitted) parameter tables.
#
# Fitted (see parametrize/PnET.jl):  FolN [EcoSpecies], MaxFracFol / FrActWd / SLWmax / TOwood / PsnAgeRed [Species],
#                                    RD_MULT [Eco] (rooting-depth / water-holding multiplier).
# Everything else is FIXED per plant functional type (PFT), sourced from the reference example parameter files
#   E1 = /workspace/LandisII/ref_pnet/Extension-PnET-Succession/test/BiomassInput/PnETSpeciesParameters.txt
#   E2 = .../deploy/Defaults/PnETGenericDefaultParameters.txt   (line cites in docs/pnet_spec.md)
# and documented one-by-one in PFT_TABLE below.

"""PFT codes: 1 = needle-leaf evergreen (NLE), 2 = broadleaf deciduous (BLD), 3 = broadleaf evergreen (BLE), 4 = deciduous conifer (DC)."""
const PFT_NLE = UInt8(1)
const PFT_BLD = UInt8(2)
const PFT_BLE = UInt8(3)
const PFT_DC = UInt8(4)

"""One row of fixed (and prior-median) species parameters for a PFT. All Float64 literals; converted to FloatType in the eco params."""
struct PFTSpec
  folN::Float64        # prior median of the FITTED FolN
  slwmax::Float64      # prior median of the FITTED SLWmax
  slwdel::Float64
  tofol::Float64
  amaxa::Float64
  amaxb::Float64
  halfsat::Float64
  h1::Float64
  h2::Float64
  h3::Float64
  h4::Float64
  k::Float64
  fracfol::Float64
  maxfracfol::Float64  # prior median of the FITTED MaxFracFol
  fracfolshape::Float64
  fractwd::Float64     # prior median of the FITTED FrActWd
  psntmin::Float64
  psntopt::Float64
  psntmax::Float64
  leafonmint::Float64
  psnagered::Float64   # prior median of the FITTED PsnAgeRed
  fracbelowg::Float64
end

# Source of every number (E1 row names; "judgment" = no reference row exists):
#  NLE  : E1 pinustro/pinuresi rows. FolN 1.5 (pinustro), SLWmax 205 (pinustro), SLWDel 0, TOfol 0.4 (pinustro/pinuresi),
#         AmaxA/B 5.3/21.5 (evergreen pair, all conifer rows), HalfSat 300 (pinustro), H1..H4 0/4/111/152 (pinuresi),
#         k 0.5, FracFol 0.11 / MaxFracFol 0.15 / shape 6 (pinustro), FrActWd 4.4e-5 (pinustro), LeafOnMinT 2.3 (pinustro),
#         PsnAgeRed 5, FracBelowG 0.33 (pinustro; = E2 default).
#         PsnTMin 1.4 (pinustro); PsnTOpt/PsnTMax 25/33 = JUDGMENT (warm-adjusted: the northern rows 23/30 would zero
#         summer photosynthesis at Southeast Tday ~ 30-32 C).
#  BLD  : E1 querrubr row (the warm-end deciduous row): FolN 2.4, SLWmax 70, SLWDel 0.2, TOfol 1, AmaxA/B -46/71.9,
#         HalfSat 300, H 0/4/111/152, PsnAgeRed 5, LeafOnMinT 2.5, PsnTMin 1.25, PsnTOpt 27.1, PsnTMax 34.1, k 0.58,
#         FracFol 0.021, MaxFracFol 0.025, shape 6, FrActWd 3.5e-5, FracBelowG 0.33.
#  BLE  : JUDGMENT (no evergreen-broadleaf row in E1): NLE-like water/light parameters, evergreen Amax pair 5.3/21.5,
#         FolN 1.5, SLWmax 110 (between NLE 205 and BLD 70), SLWDel 0, TOfol 0.5 (E1 evergreen range 0.25-0.5),
#         k 0.5, FracFol 0.05 / MaxFracFol 0.06 (between NLE and BLD), shape 6, FrActWd 4e-5 (= E2 default),
#         PsnTMin 1.5 / Opt 26 / Max 34, LeafOnMinT 2.3.
#  DC   : JUDGMENT (deciduous conifer, TAAS/TADI2/LA**): NLE row with TOfol = 1 (deciduous), FolN 1.3, SLWmax 150, LeafOnMinT 2.3.
const PFT_TABLE = (
  PFTSpec(1.5, 205.0, 0.0, 0.4, 5.3, 21.5, 300.0, 0.0, 4.0, 111.0, 152.0, 0.5, 0.11, 0.15, 6.0, 4.4e-5, 1.4, 25.0, 33.0, 2.3, 5.0, 0.33),   # NLE
  PFTSpec(2.4, 70.0, 0.2, 1.0, -46.0, 71.9, 300.0, 0.0, 4.0, 111.0, 152.0, 0.58, 0.021, 0.025, 6.0, 3.5e-5, 1.25, 27.1, 34.1, 2.5, 5.0, 0.33), # BLD
  PFTSpec(1.5, 110.0, 0.0, 0.5, 5.3, 21.5, 300.0, 0.0, 4.0, 111.0, 152.0, 0.5, 0.05, 0.06, 6.0, 4.0e-5, 1.5, 26.0, 34.0, 2.3, 5.0, 0.33),   # BLE
  PFTSpec(1.3, 150.0, 0.0, 1.0, 5.3, 21.5, 300.0, 0.0, 4.0, 111.0, 152.0, 0.5, 0.11, 0.15, 6.0, 4.4e-5, 1.4, 25.0, 33.0, 2.3, 5.0, 0.33),   # DC
)
@inline pft_spec(pft::UInt8) = @inbounds PFT_TABLE[Int(pft)]

# Global (non-PFT) fixed constants — E2 defaults unless noted.
Base.@kwdef struct PnETConst
  precip_events::Int32 = Int32(11)     # E2 PrecipEvents
  prec_int_const::Float64 = 0.1        # E2 PrecIntConst
  prec_loss_frac::Float64 = 0.0        # ex EcoregionParameters.txt (PrecLossFrac 0.0)
  leakage_frac::Float64 = 1.0          # E2 LeakageFrac
  snow_sublim::Float64 = 0.15          # E2 SnowSublimFrac
  et_ext::Float64 = 0.65               # E2 ETExtCoeff
  ret_crop::Float64 = 0.3              # E2 RETCropCoeff
  evap_depth::Float64 = 500.0          # E2 EvapDepth (mm)
  layer_thresh::Float64 = 0.5          # E2 LayerThreshRatio
  max_layers::Int32 = Int32(3)         # E2 MaxCanopyLayers (<= MAXL)
  canopy_sum_scale::Float64 = 0.206    # E2 CanopySumScale
  cfrac::Float64 = 0.45                # E2 CFracBiomass
  dnsc::Float64 = 0.05                 # E2 DNSC
  maint_resp::Float64 = 0.002          # E2 MaintResp
  amax_frac::Float64 = 0.75            # E2 AmaxFrac
  dvpd1::Float64 = 0.05                # E2 DVPD1
  dvpd2::Float64 = 2.0                 # E2 DVPD2
  toroot::Float64 = 0.02               # E2 TOroot
  initial_nsc::Float64 = 7.0           # E2 InitialNSC
  init_biomass_death::Bool = true      # spec 6.5(2): TotalBiomass < InitBiomass at December (year>0) -> cohort dies
  root_depth_base::Float64 = 1000.0    # ex EcoregionParameters.txt (RootingDepth 1000 mm)
end

"""PnET parameters, indexed like BiomassSuccessionParams (global species ids; ECO_SPECIES_IDS maps eco-local -> global)."""
Base.@kwdef struct PnETParams
  ECO_LIST::Vector{String}
  SPECIES_LIST::Vector{String}
  ECO_SPECIES_IDS::Vector{Vector{UIntType}}
  # fixed metadata (per global species)
  PFT::Vector{UInt8}
  LONGEVITY::Vector{FloatType}
  # fitted, species-level (global species index)
  MaxFracFol::Vector{FloatType}
  FrActWd::Vector{FloatType}
  SLWmax::Vector{FloatType}
  TOwood::Vector{FloatType}
  PsnAgeRed::Vector{FloatType}
  # fitted, eco x species (eco-local species index)
  FolN::Vector{Vector{FloatType}}
  # fitted, eco level
  RD_MULT::Vector{FloatType}
end

"""Per-ecoregion, per-candidate resolved parameters: concrete eco-local Float64 vectors (no abstract fields)."""
struct PnETEcoParams
  folN::Vector{Float64}
  maxfracfol::Vector{Float64}
  fracfol::Vector{Float64}
  fracfolshape::Vector{Float64}
  fractwd::Vector{Float64}
  slwmax::Vector{Float64}
  slwdel::Vector{Float64}
  tofol::Vector{Float64}
  towood::Vector{Float64}
  amaxa::Vector{Float64}
  amaxb::Vector{Float64}
  halfsat::Vector{Float64}
  k::Vector{Float64}
  psnagered::Vector{Float64}
  psntmin::Vector{Float64}
  psntopt::Vector{Float64}
  psntmax::Vector{Float64}
  leafonmint::Vector{Float64}
  h1::Vector{Float64}
  h2::Vector{Float64}
  h3::Vector{Float64}
  h4::Vector{Float64}
  fracbelowg::Vector{Float64}
  longevity::Vector{Float64}
  maxlai::Vector{Float64}
  initbiomass::Vector{Float64}
  rd_mult::Float64
end

# ---------------------------------------------------------------------------------------------------------------------
# Plant functional type from the species symbol (+ FIA SFTWD_HRDWD when known)
# ---------------------------------------------------------------------------------------------------------------------
# Evergreen / semi-evergreen broadleaves of the Southeast (treated as BLE per the task brief).
const BROADLEAF_EVERGREEN = Set([
  "QUVI", "QULA3", "QUGE2", "QUHE2", "QUCH2", "MAGR4", "MAVI2", "PEBO", "PEPA37", "ILOP", "SAPA", "OSAM", "GOLA", "MYCE"])
# Deciduous conifers (TAAS pond cypress, TADI2 baldcypress, larches).
const DECIDUOUS_CONIFER = Set(["TAAS", "TADI2", "LALA", "LALY", "LAOC", "METAS"])
# FIA SFTWD_HRDWD per species symbol (S/H) comes from REF_SPECIES (`read_species_sftwd`, kept in `PnETContext.species_sftwd`); empty => genus-prefix heuristic.
const _SOFTWOOD_PREFIX = ("PI", "JU", "TS", "AB", "PC", "CH", "TH", "TA", "LA", "TX", "CU", "SE", "PS", "ME")

function pft_of(sym::AbstractString, sftwd::AbstractDict=Dict{String,String}())::UInt8
  s = String(sym)
  sym_u = uppercase(s)
  if startswith(s, "_")
    # catch-all / group names: "_S_*" softwood, "_H_*" hardwood, "_GRP_<spgrpcd>[_tier]" (FIA species group)
    startswith(s, "_S") && return PFT_NLE
    startswith(s, "_H") && return PFT_BLD
    m = match(r"^_GRP_(\d+)", s)
    if m !== nothing
      g = parse(Int, m.captures[1])
      g == 8 && return PFT_DC          # FIA SPGRPCD 8 = cypress
      return g <= 10 ? PFT_NLE : PFT_BLD   # SPGRPCD 1-10 softwood groups, >= 11 hardwood groups
    end
    return PFT_BLD
  end
  sym_u in DECIDUOUS_CONIFER && return PFT_DC
  sym_u in BROADLEAF_EVERGREEN && return PFT_BLE
  sw = get(sftwd, sym_u, "")
  if sw == "S"
    return PFT_NLE
  elseif sw == "H"
    return PFT_BLD
  end
  return (length(sym_u) >= 2 && sym_u[1:2] in _SOFTWOOD_PREFIX) ? PFT_NLE : PFT_BLD
end

# ---------------------------------------------------------------------------------------------------------------------
# Saxton & Rawls soil hydraulics (spec 5.9) and the soil table
# ---------------------------------------------------------------------------------------------------------------------
# SaxtonAndRawlsParameters.txt (sand, clay, pctOM); densFactor = 1, gravel = 0 for all but BEDR.
const SOIL_TABLE = Dict{String,NTuple{3,Float64}}(
  "SAND" => (0.85, 0.04, 2.08), "LOSA" => (0.80, 0.05, 2.33), "SALO" => (0.63, 0.10, 2.52), "LOAM" => (0.41, 0.19, 3.06),
  "SILO" => (0.15, 0.18, 3.05), "SILT" => (0.05, 0.10, 1.89), "SNCL" => (0.53, 0.26, 2.29), "CLLO" => (0.29, 0.32, 3.51),
  "SLCL" => (0.09, 0.32, 4.06), "SACL" => (0.50, 0.40, 2.50), "SICL" => (0.08, 0.47, 3.59), "CLAY" => (0.13, 0.55, 2.88),
  "BEDR" => (0.10, 0.10, 1.00))

"""Saxton-Rawls constants for a texture code -> (porosity, fieldcap, wiltpnt, A, B) (volumetric m3/m3; tension = A*theta^-B kPa)."""
function saxton_rawls(texture::AbstractString)
  haskey(SOIL_TABLE, uppercase(String(texture))) || error("unknown PnET soil texture '$texture' (valid: $(sort(collect(keys(SOIL_TABLE)))))")
  sand, clay, om = SOIL_TABLE[uppercase(String(texture))]
  pm1500 = -0.024 * sand + 0.487 * clay + 0.006 * om + 0.005 * sand * om - 0.013 * clay * om + 0.068 * sand * clay + 0.031
  pm1500a = pm1500 + 0.14 * pm1500 - 0.02
  pm33 = -0.251 * sand + 0.195 * clay + 0.011 * om + 0.006 * sand * om - 0.027 * clay * om + 0.452 * sand * clay + 0.299
  pm33a = pm33 + (1.283 * pm33^2 - 0.374 * pm33 - 0.015)
  por33 = 0.278 * sand + 0.034 * clay + 0.022 * om - 0.018 * sand * om - 0.027 * clay * om - 0.584 * sand * clay + 0.078
  por33a = por33 + (0.636 * por33 - 0.107)
  satpor33 = por33a + pm33a
  sandadjsat = satpor33 + (-0.097 * sand + 0.043)
  dens_om = (1.0 - sandadjsat) * 2.65
  dens_comp = dens_om * 1.0
  porosity = 1.0 - dens_comp / 2.65
  moist33 = pm33a + 0.2 * ((1.0 - dens_comp / 2.65) - (1.0 - dens_om / 2.65))
  B = (log(1500.0) - log(33.0)) / (log(moist33) - log(pm1500a))
  A = exp(log(33.0) + B * log(moist33))
  fc = (33.0 / A)^(1.0 / -B)
  wp = (1500.0 / A)^(1.0 / -B)
  return (porosity, fc, wp, A, B)
end

"""
    saxton_rawls_theta(porosity, fieldcap, wiltpnt) -> (por, fc, wp, A, B, nrepair)

Pressure-head curve of one site from MEASURED/derived volumetric contents (e.g. per-plot SOLUS rooting-zone means, tools/build_plot_soil.jl)
instead of a texture class. Same two-point Saxton & Rawls (2006) fit the texture path ends with (`saxton_rawls` above, and the C# reference
PressureHeadSaxton_Rawls.cs:116-117), with theta33 = `fieldcap` and theta1500 = `wiltpnt`:

    B = (ln 1500 - ln 33) / (ln theta33 - ln theta1500)         [C# tensionB, line 116]
    A = exp(ln 33 + B * ln theta33)                               [C# tensionA, line 117]
    tension [kPa] = A * theta^-B   =>  tension(theta33) = 33 kPa, tension(theta1500) = 1500 kPa

Same convention as the plugin (`press_head`, plugin.jl): tension in kPa, head [m H2O] = tension * 0.1019977334 (C# CalculateWaterPressure,
lines 56-64: 0 above porosity, rounded to 0.01 m); the Feddes wilting point uses h4*9.804139432/A (kPa) the same way, and `fc`/`wp` here play
the role of `(33/A)^(-1/B)` / `(1500/A)^(-1/B)` (identical by construction).

Invariant wiltpnt < fieldcap < porosity is enforced by repair (counted in `nrepair`): wp in [0.005, 0.60], fc >= wp + 0.01, por >= fc + 0.01,
and B limited to [1, 12] (the SOLUS lambda_bc raster clips 1/B to [0.1, 30]; B > 12 would underflow `A` in Float32) by lowering wp / raising wp.
"""
function saxton_rawls_theta(porosity::Real, fieldcap::Real, wiltpnt::Real)
  por = Float64(porosity); fc = Float64(fieldcap); wp = Float64(wiltpnt)
  nrep = 0
  w0 = wp
  wp = clamp(wp, 0.005, 0.60)
  fc = max(fc, wp + 0.01)
  # B bounds <=> ln(fc/wp) in [ln(1500/33)/12, ln(1500/33)/1]
  lr = log(1500.0) - log(33.0)
  r = log(fc) - log(wp)
  if r < lr / 12
    wp = exp(log(fc) - lr / 12)
  elseif r > lr
    wp = exp(log(fc) - lr)
  end
  por = max(por, fc + 0.01)
  (wp != w0 || fc != Float64(fieldcap) || por != Float64(porosity)) && (nrep = 1)
  B = lr / (log(fc) - log(wp))
  A = exp(log(33.0) + B * log(fc))
  return (por, fc, wp, A, B, nrep)
end

# ---------------------------------------------------------------------------------------------------------------------
# Param generation / eco-param resolution
# ---------------------------------------------------------------------------------------------------------------------
# Search bounds (kept here so params + parametrize agree).
const BOUNDS_FOLN = (0.6f0, 3.0f0)
const BOUNDS_MAXFRACFOL = (0.01f0, 0.40f0)
const BOUNDS_FRACTWD = (2.0f-5, 1.2f-4)
const BOUNDS_SLWMAX = (30.0f0, 300.0f0)
const BOUNDS_TOWOOD = (0.002f0, 0.04f0)
const BOUNDS_PSNAGERED = (2.0f0, 8.0f0)
const BOUNDS_RDMULT = (0.4f0, 2.0f0)

@inline _jit(rng, med, lo, hi, sd=0.25) = clamp(med * exp(sd * randn(rng)), lo, hi)

function generate_pnet_params(tables::BiomassSuccessionPlugin.BSTables, sftwd::AbstractDict, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}; rng::Random.AbstractRNG, no_establishment::Bool=false)
  n_species = length(species_list)
  pft = UInt8[pft_of(s, sftwd) for s in species_list]
  LONG = BiomassSuccessionPlugin.default_longevity(tables, species_list)
  spec(i) = pft_spec(pft[i])
  mk(f, b) = FloatType[_jit(rng, f(spec(i)), b[1], b[2]) for i in 1:n_species]
  MaxFracFol = mk(s -> s.maxfracfol, BOUNDS_MAXFRACFOL)
  FrActWd = mk(s -> s.fractwd, BOUNDS_FRACTWD)
  SLWmax = mk(s -> s.slwmax, BOUNDS_SLWMAX)
  TOwood = FloatType[_jit(rng, 0.01, BOUNDS_TOWOOD[1], BOUNDS_TOWOOD[2]) for _ in 1:n_species]
  PsnAgeRed = mk(s -> s.psnagered, BOUNDS_PSNAGERED)
  FolN = [FloatType[_jit(rng, spec(Int(g)).folN, BOUNDS_FOLN[1], BOUNDS_FOLN[2]) for g in sp_ids] for sp_ids in eco_species_ids]
  RD_MULT = FloatType[_jit(rng, 1.0, BOUNDS_RDMULT[1], BOUNDS_RDMULT[2], 0.15) for _ in eco_list]
  PnETParams(ECO_LIST=eco_list, SPECIES_LIST=species_list, ECO_SPECIES_IDS=eco_species_ids, PFT=pft, LONGEVITY=LONG,
    MaxFracFol=MaxFracFol, FrActWd=FrActWd, SLWmax=SLWmax, TOwood=TOwood, PsnAgeRed=PsnAgeRed, FolN=FolN, RD_MULT=RD_MULT)
end

"""MaxLAI of a species (spec 4.2, SpeciesPnET.cs:236-246): LAI of the peak-foliage cohort, summed over the IMAX sublayers."""
function max_lai(fracfol::Float64, fractwd::Float64, slwmax::Float64, slwdel::Float64)
  peakb = 1.0 / fractwd
  peakfol = peakb * fracfol * exp(-fractwd * peakb)
  sub = peakfol / IMAX
  s = 0.0
  for i in 0:IMAX-1
    s += max(0.01, sub / (slwmax - slwdel * i * sub))
  end
  return s
end

function generate_pnet_eco_params(params::PnETParams, cst::PnETConst)::Vector{PnETEcoParams}
  out = Vector{PnETEcoParams}(undef, length(params.ECO_SPECIES_IDS))
  ib = Float64(trunc(Int, cst.initial_nsc / (cst.dnsc * cst.cfrac)))
  for (e, gids) in enumerate(params.ECO_SPECIES_IDS)
    n = length(gids)
    v() = Vector{Float64}(undef, n)
    folN, maxff, ff, ffs, fawd, slw, slwdel, tofol, towood = v(), v(), v(), v(), v(), v(), v(), v(), v()
    amaxa, amaxb, halfsat, kk, par, pmin, popt, pmax, lom = v(), v(), v(), v(), v(), v(), v(), v(), v()
    h1, h2, h3, h4, fbg, lng, mlai, ibio = v(), v(), v(), v(), v(), v(), v(), v()
    for j in 1:n
      g = Int(gids[j]); s = pft_spec(params.PFT[g])
      folN[j] = params.FolN[e][j]
      maxff[j] = params.MaxFracFol[g]
      ff[j] = min(s.fracfol, maxff[j])           # FracFol is fixed per PFT but never above the fitted MaxFracFol
      ffs[j] = s.fracfolshape
      fawd[j] = params.FrActWd[g]; slw[j] = params.SLWmax[g]; slwdel[j] = s.slwdel; tofol[j] = s.tofol
      towood[j] = params.TOwood[g]
      amaxa[j] = s.amaxa; amaxb[j] = s.amaxb; halfsat[j] = s.halfsat; kk[j] = s.k; par[j] = params.PsnAgeRed[g]
      pmin[j] = s.psntmin; popt[j] = s.psntopt; pmax[j] = s.psntmax; lom[j] = s.leafonmint
      h1[j] = s.h1; h2[j] = s.h2; h3[j] = s.h3; h4[j] = s.h4; fbg[j] = s.fracbelowg; lng[j] = params.LONGEVITY[g]
      mlai[j] = max_lai(ff[j], fawd[j], slw[j], slwdel[j])
      # SpeciesPnET.cs:410-411 (uint truncations)
      ibio[j] = trunc(ib - trunc(fbg[j] * ib) * cst.toroot - trunc((1 - fbg[j]) * ib) * towood[j])
    end
    out[e] = PnETEcoParams(folN, maxff, ff, ffs, fawd, slw, slwdel, tofol, towood, amaxa, amaxb, halfsat, kk, par, pmin, popt, pmax, lom,
      h1, h2, h3, h4, fbg, lng, mlai, ibio, Float64(params.RD_MULT[e]))
  end
  return out
end
