struct PnET <: AbstractPlugin end
const PluginType = PnET

# ---------------------------------------------------------------------------------------------------------------------
# SoA layout
# ---------------------------------------------------------------------------------------------------------------------
PanCore.scalar_arrays(::Type{PluginType}, n::Int) =
  (
    B=Vector{FloatType}(undef, n),                # total REPORTED aboveground biomass (sum of c_bio), g/m2
    soil_water=Vector{FloatType}(undef, n),       # volumetric soil water (m3/m3); init = field capacity
    snowpack=Vector{FloatType}(undef, n),         # mm water equivalent
    canopy_lai_prev=Vector{FloatType}(undef, n),  # layer-weighted canopy LAI of the previous month (interception, ground PET)
    subcanopy_par=Vector{FloatType}(undef, n),    # PAR below the canopy, last month processed (diagnostic)
    clim_id=Vector{Int32}(undef, n),              # index into PnETClimate.ids
    cal_year0=Vector{Int16}(undef, n),            # calendar year of the plot's sim year 0
    lat=Vector{FloatType}(undef, n),              # degrees N
    soil_por=Vector{FloatType}(undef, n),         # Saxton-Rawls porosity / field capacity / wilting point (m3/m3), A, B (tension = A*theta^-B kPa)
    soil_fc=Vector{FloatType}(undef, n),
    soil_wp=Vector{FloatType}(undef, n),
    soil_A=Vector{FloatType}(undef, n),
    soil_B=Vector{FloatType}(undef, n),
    rooting_depth=Vector{FloatType}(undef, n),    # base rooting depth (mm); effective = rooting_depth * eco RD_MULT
  )

PanCore.csr_fields(::Type{PluginType}) =
  (
    c_species=:cohort,      # eco-LOCAL species index (BiomassSuccession convention)
    c_age=:cohort,
    c_bio=:cohort,          # REPORTED aboveground biomass g/m2 = ((1-FracBelowG)*TotalBiomass + Fol) * CanopyLayerProp
    c_tbio=:cohort,         # TotalBiomass (wood+root) g/m2 of the cohort's canopy; 0 => pending (initialise from c_bio at next step)
    c_bmax=:cohort,         # BiomassMax (lifetime max TotalBiomass)
    c_fol=:cohort,          # foliage g/m2
    c_nsc=:cohort,          # non-structural carbon gC/m2
    c_clp=:cohort,          # CanopyLayerProp
    c_cgs=:cohort,          # CanopyGrowingSpace
    c_lastlai=:cohort,      # seasonal max LAI
    c_adjff=:cohort,        # adjFracFol
    c_fradsum=:cohort,      # sum over leaf-on months of the monthly mean sublayer fRad (last season)
    c_fradn=:cohort,        # number of such months
    c_growm=:cohort,        # growMonth: <=0 leaf off, 1 first leaf-on month, saturating at 100
    # --- per-month scratch (reset every month) ---
    c_idx=:cohort,          # next sublayer index 0..IMAX-1
    c_laisum=:cohort,       # running sum of this month's sublayer LAI
    c_fradm=:cohort,        # sum of this month's sublayer fRad
    c_gpp_m=:cohort,        # monthly sums over sublayers (diagnostics; the unit tests read these)
    c_folresp_m=:cohort,
    c_netpsn_m=:cohort,
    c_transp_m=:cohort,
    # --- annual accumulators (reset every year; carbon-closure diagnostics) ---
    c_acc_net=:cohort,      # sum NetPsn (gC)
    c_acc_maint=:cohort,    # sum maintenance respiration (gC)
    c_acc_folsen=:cohort,   # foliage senescence (gDW)
    c_acc_turn=:cohort,     # wood+root turnover (gDW)
    sp_mature=:species,     # unused by PnET (no establishment); its length carries the eco species count for the shared loss
  )

PanCore.csr_arrays(::Type{PluginType}, nnz::NamedTuple) =
  (
    c_species=Vector{UIntType}(undef, nnz.cohort),
    c_age=Vector{FloatType}(undef, nnz.cohort),
    c_bio=Vector{FloatType}(undef, nnz.cohort),
    c_tbio=Vector{FloatType}(undef, nnz.cohort),
    c_bmax=Vector{FloatType}(undef, nnz.cohort),
    c_fol=Vector{FloatType}(undef, nnz.cohort),
    c_nsc=Vector{FloatType}(undef, nnz.cohort),
    c_clp=Vector{FloatType}(undef, nnz.cohort),
    c_cgs=Vector{FloatType}(undef, nnz.cohort),
    c_lastlai=Vector{FloatType}(undef, nnz.cohort),
    c_adjff=Vector{FloatType}(undef, nnz.cohort),
    c_fradsum=Vector{FloatType}(undef, nnz.cohort),
    c_fradn=Vector{FloatType}(undef, nnz.cohort),
    c_growm=Vector{Int8}(undef, nnz.cohort),
    c_idx=Vector{Int8}(undef, nnz.cohort),
    c_laisum=Vector{FloatType}(undef, nnz.cohort),
    c_fradm=Vector{FloatType}(undef, nnz.cohort),
    c_gpp_m=Vector{FloatType}(undef, nnz.cohort),
    c_folresp_m=Vector{FloatType}(undef, nnz.cohort),
    c_netpsn_m=Vector{FloatType}(undef, nnz.cohort),
    c_transp_m=Vector{FloatType}(undef, nnz.cohort),
    c_acc_net=Vector{FloatType}(undef, nnz.cohort),
    c_acc_maint=Vector{FloatType}(undef, nnz.cohort),
    c_acc_folsen=Vector{FloatType}(undef, nnz.cohort),
    c_acc_turn=Vector{FloatType}(undef, nnz.cohort),
    sp_mature=Vector{Bool}(undef, nnz.species),
  )

# Bundle of the per-cohort CSR arrays (concrete, pointer-only struct: free to pass around, no allocation).
struct CohArrays
  species::Vector{UIntType}
  age::Vector{FloatType}
  bio::Vector{FloatType}
  tbio::Vector{FloatType}
  bmax::Vector{FloatType}
  fol::Vector{FloatType}
  nsc::Vector{FloatType}
  clp::Vector{FloatType}
  cgs::Vector{FloatType}
  lastlai::Vector{FloatType}
  adjff::Vector{FloatType}
  fradsum::Vector{FloatType}
  fradn::Vector{FloatType}
  growm::Vector{Int8}
  idx::Vector{Int8}
  laisum::Vector{FloatType}
  fradm::Vector{FloatType}
  gpp_m::Vector{FloatType}
  folresp_m::Vector{FloatType}
  netpsn_m::Vector{FloatType}
  transp_m::Vector{FloatType}
  acc_net::Vector{FloatType}
  acc_maint::Vector{FloatType}
  acc_folsen::Vector{FloatType}
  acc_turn::Vector{FloatType}
end
CohArrays(c::NamedTuple) = CohArrays(c.c_species, c.c_age, c.c_bio, c.c_tbio, c.c_bmax, c.c_fol, c.c_nsc, c.c_clp, c.c_cgs, c.c_lastlai,
  c.c_adjff, c.c_fradsum, c.c_fradn, c.c_growm, c.c_idx, c.c_laisum, c.c_fradm, c.c_gpp_m, c.c_folresp_m, c.c_netpsn_m, c.c_transp_m,
  c.c_acc_net, c.c_acc_maint, c.c_acc_folsen, c.c_acc_turn)

# ---------------------------------------------------------------------------------------------------------------------
# Scratch (per thread)
# ---------------------------------------------------------------------------------------------------------------------
"""Species x month derived quantities (spec 3), computed once per cohort-month."""
struct SpMonth
  ftemp::Float64      # FTempPSN
  dvpd::Float64       # DVPD
  jh2o::Float64       # JH2O
  basefrac::Float64   # BaseFolRespFrac (Wythers)
  q10f::Float64       # Q10Factor at Tave (foliar respiration)
  maintf::Float64     # MaintRespFTempResp
end
const SPM0 = SpMonth(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

"""Site-month working state (soil bucket + current climate); a mutable struct so the sublayer kernel updates it without allocation."""
mutable struct MonthWork
  water::Float64; rd::Float64; por::Float64; fc::Float64; wp::Float64; tA::Float64; tB::Float64
  melt_in::Float64; pe::Float64; gpe::Float64; snowpack::Float64
  par_layer::Float64
  tmin::Float64; tave::Float64; tday::Float64; co2::Float64; vpd::Float64; dayl::Float64; nightl::Float64; dayspan::Float64
  month::Int
end
MonthWork() = MonthWork(0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1)

mutable struct PnETScratch
  ord::Vector{Int32}       # cohort local indices sorted by BiomassMax (desc)
  seq::Vector{Int32}       # processing order of cohort-sublayer calls (cohort index repeated IMAX times, shuffled within layer)
  evc::Vector{Int32}       # precipitation events per processing position
  spm::Vector{SpMonth}     # per cohort
  lay_c::Vector{Int}       # layer l covers ord[lay_c[l] : lay_c[l+1]-1]
  lay_p::Vector{Int}       # layer l covers seq[lay_p[l] : lay_p[l+1]-1]
  hr::Vector{Float64}      # daylength hours per month (this site)
  ra::Vector{Float64}      # extraterrestrial radiation per month (this site)
  w::MonthWork
end
PnETScratch() = PnETScratch(Int32[], Int32[], Int32[], SpMonth[], zeros(Int, MAXL + 2), zeros(Int, MAXL + 2), zeros(12), zeros(12), MonthWork())

const _SCRATCH_POOL = PnETScratch[]
"""Global per-thread scratch pool (indexed by Threads.threadid()); grown lazily, never shrunk."""
const _POOL_LOCK = ReentrantLock()
function scratch_pool()
  nt = Threads.maxthreadid()
  if length(_SCRATCH_POOL) < nt
    lock(_POOL_LOCK) do
      while length(_SCRATCH_POOL) < nt
        push!(_SCRATCH_POOL, PnETScratch())
      end
    end
  end
  return _SCRATCH_POOL
end

function ensure_capacity!(scr::PnETScratch, n::Int)
  length(scr.ord) < n && (resize!(scr.ord, n); resize!(scr.spm, n))
  length(scr.seq) < n * IMAX && (resize!(scr.seq, n * IMAX); resize!(scr.evc, n * IMAX))
  return nothing
end

# ---------------------------------------------------------------------------------------------------------------------
# Equations (spec section refs)
# ---------------------------------------------------------------------------------------------------------------------
"""Two half-parabola temperature response, peak 1 at PsnTOpt (spec 3, EPV.cs:218-241)."""
@inline function dtemp_response(tday::Float64, topt::Float64, tmin::Float64, tmax::Float64)
  (tday < tmin || tday > tmax) && return 0.0
  if tday <= topt
    tmx = topt + (topt - tmin)
    return max(0.0, (tmx - tday) * (tday - tmin) / ((tmx - tmin) / 2)^2)
  else
    tmn = topt + (topt - tmax)
    return max(0.0, (tmax - tday) * (tday - tmn) / ((tmax - tmn) / 2)^2)
  end
end

@inline function sp_month(ep::PnETEcoParams, cst::PnETConst, s::Int, tmin::Float64, tave::Float64, tday::Float64, dayl::Float64, nightl::Float64, vpd::Float64)
  topt = ep.psntopt[s]
  dvpd = max(0.0, 1.0 - cst.dvpd1 * vpd^cst.dvpd2)
  jh2o = 0.239 * vpd / (8314.47 * (tmin + 273.0))
  ftemp = dtemp_response(tday, topt, ep.psntmin[s], ep.psntmax[s])
  basefrac = 0.138071 - 0.0024519 * tave                       # Wythers
  q10base = 3.22 - 0.046 * 0.5 * (tave + topt)
  q10f = q10base^((tave - topt) / 10)
  fday = q10base^((tday - topt) / 10)
  fnight = q10base^((tmin - topt) / 10)
  wtd = min(1.0, (fday * dayl + fnight * nightl) / (dayl + nightl))
  return SpMonth(ftemp, dvpd, jh2o, basefrac, q10f, cst.maint_resp * wtd)
end

"""Sublayer i (0-based) LAI given foliage and the LAI of the sublayers above (spec 4.4, Cohort.cs:1864-1904)."""
@inline function lai_sub(fol::Float64, slwmax::Float64, slwdel::Float64, i::Int, laisum_above::Float64)
  laimax = max(0.01, 25.0 - laisum_above)
  l = (1.0 / IMAX) * fol / (slwmax - slwdel * i * (1.0 / IMAX) * fol)
  (fol > 0 && l <= 0) && return laimax / (IMAX - i)
  return min(laimax, l)
end

"""Soil-water pressure head in m (spec 5.9, SR.cs:63-85): 0 above porosity, +Inf when dry."""
@inline function press_head(water::Float64, por::Float64, tA::Float64, tB::Float64)
  water > por && return 0.0
  water <= 0 && return Inf
  ph = tA * water^(-tB) * 0.1019977334
  return round(ph * 100.0) / 100.0
end

"""Feddes water-stress factor (spec 5.6)."""
@inline function fwater_of(h1::Float64, h2::Float64, h3::Float64, h4::Float64, ph::Float64)
  ph <= h1 && return 0.0
  (ph < min(h1, h2) || ph >= h4) && return 0.0
  ph > h3 && return 1.0 - (ph - h3) / (h4 - h3)
  ph < h2 && return ph / (h2 - h1) - h1 / (h2 - h1)
  return 1.0
end

# ---------------------------------------------------------------------------------------------------------------------
# Cohort initialisation from an observed reported AGB (spec section 8, bounded bisection)
# ---------------------------------------------------------------------------------------------------------------------
@inline function reported_of_tb(tb::Float64, ep::PnETEcoParams, s::Int)
  fbg = ep.fracbelowg[s]
  fol = ep.maxfracfol[s] * exp(-ep.fractwd[s] * tb) * tb                 # adjFracFol = MaxFracFol, BiomassMax = TB
  lai = 0.0
  for i in 0:IMAX-1
    lai += lai_sub(fol, ep.slwmax[s], ep.slwdel[s], i, lai)
  end
  lai = min(lai, ep.maxlai[s])
  clp = min(lai / ep.maxlai[s], 1.0)
  return ((1 - fbg) * tb + fol) * clp
end

"""Find TotalBiomass such that the reported AGB equals `target` (tolerance 2 g/m2 in the reference; bisection reaches ~1e-6)."""
function solve_total_biomass(target::Float64, ep::PnETEcoParams, s::Int)
  target <= 0 && return 1.0
  lo = 1.0
  reported_of_tb(lo, ep, s) >= target && return lo
  hi = 2.0
  while reported_of_tb(hi, ep, s) < target
    lo = hi
    hi *= 2
    hi > 1e7 && return lo       # unreachable target (beyond the rising branch): keep the largest bracketed value
  end
  for _ in 1:60
    mid = 0.5 * (lo + hi)
    reported_of_tb(mid, ep, s) < target ? (lo = mid) : (hi = mid)
    hi - lo < 1e-4 && break
  end
  return 0.5 * (lo + hi)
end

"""Initialise the pools of cohort k from its stored observed AGB (c_bio). Winter state: leaf off, foliage = IdealFol*(1-TOfol)."""
function materialize_cohort!(ca::CohArrays, o::Int, k::Int, ep::PnETEcoParams, cst::PnETConst)
  j = o + k
  s = Int(ca.species[j])
  target = Float64(ca.bio[j])
  tb = solve_total_biomass(target, ep, s)
  fbg = ep.fracbelowg[s]
  fol_ideal = ep.maxfracfol[s] * exp(-ep.fractwd[s] * tb) * tb
  lai = 0.0
  for i in 0:IMAX-1
    lai += lai_sub(fol_ideal, ep.slwmax[s], ep.slwdel[s], i, lai)
  end
  lai = min(lai, ep.maxlai[s])
  clp = min(lai / ep.maxlai[s], 1.0)
  fol = fol_ideal * (1 - ep.tofol[s])
  factive = exp(-ep.fractwd[s] * tb)
  ca.tbio[j] = tb
  ca.bmax[j] = tb
  ca.fol[j] = fol
  ca.nsc[j] = cst.dnsc * factive * (tb + fol) * cst.cfrac
  ca.clp[j] = clp
  ca.cgs[j] = 1.0
  ca.lastlai[j] = lai
  ca.adjff[j] = ep.maxfracfol[s]
  ca.fradsum[j] = 0.0
  ca.fradn[j] = 0.0
  ca.growm[j] = 0
  ca.bio[j] = ((1 - fbg) * tb + fol_ideal) * clp
  return nothing
end

@inline function materialize_pending!(ca::CohArrays, o::Int, n::Int, ep::PnETEcoParams, cst::PnETConst)
  @inbounds for k in 1:n
    ca.tbio[o+k] <= 0 && materialize_cohort!(ca, o, k, ep, cst)
  end
  return nothing
end

# ---------------------------------------------------------------------------------------------------------------------
# Sublayer kernel (spec 5.1-5.8). One call = one cohort-sublayer in the layered processing order.
# ---------------------------------------------------------------------------------------------------------------------
function sublayer!(w::MonthWork, ca::CohArrays, o::Int, k::Int, nev::Int, ep::PnETEcoParams, cst::PnETConst, spm::SpMonth)
  j = o + k
  s = Int(ca.species[j])
  idx = Int(ca.idx[j])
  rd = w.rd
  water = w.water
  cfrac = cst.cfrac

  # ---- 5.1 water inputs: this sublayer's share of snowmelt + its precipitation events
  melt = w.melt_in
  if melt > 0
    cap = w.por * rd
    ro = min(melt, max(water * rd + melt - cap, 0.0))
    water = max(0.0, water + (melt - ro) / rd)
  end
  if nev > 0 && w.pe > 0
    for _ in 1:nev
      cap = w.por * rd
      rr = min(w.pe, max(water * rd + w.pe - cap, 0.0))
      water = max(0.0, water + (w.pe - rr) / rd)
      leak = max(cst.leakage_frac * (water - w.fc), 0.0) * rd
      water = max(0.0, water - leak / rd)
    end
  elseif melt > 0
    leak = max(cst.leakage_frac * (water - w.fc), 0.0) * rd
    water = max(0.0, water - leak / rd)
  end

  tb = Float64(ca.tbio[j]); fol = Float64(ca.fol[j]); nsc = Float64(ca.nsc[j]); bmax = Float64(ca.bmax[j])
  age = Float64(ca.age[j]); growm = Int(ca.growm[j])
  fractwd = ep.fractwd[s]; fbg = ep.fracbelowg[s]

  # ---- 5.2 maintenance respiration
  maint = (1.0 / IMAX) * min(nsc, spm.maintf * tb * cfrac)
  nsc = max(0.0, nsc - maint)
  ca.acc_maint[j] += maint

  # ---- 5.3 December events (last sublayer only)
  if w.month == 12 && idx == IMAX - 1
    factive = exp(-fractwd * bmax)
    nscfrac = nsc / (factive * (tb + fol) * cfrac)
    if !(nscfrac > 0.01)
      nsc = 0.0
    elseif cst.init_biomass_death && tb < ep.initbiomass[s]
      nsc = 0.0
      growm = -1
      ca.acc_folsen[j] += fol               # FoliageSenescence(): NSCfrac = 0 -> all foliage lost
      fol = 0.0
    end
    root = round(fbg * tb); wood = round((1 - fbg) * tb)
    sen = root * cst.toroot + wood * ep.towood[s]
    tb -= sen
    ca.acc_turn[j] += sen
    bmax = max(bmax, tb)
    factive = exp(-fractwd * bmax)
    alloc = max(nsc - cst.dnsc * factive * tb * cfrac, 0.0)
    tb += alloc / cfrac
    bmax = max(bmax, tb)
    nsc = max(nsc - alloc, 0.0)
    age += 1.0
  end

  # ---- 5.4 phenology and foliage allocation (first sublayer only)
  if idx == 0
    if w.tmin <= ep.leafonmint[s]
      if growm >= 1
        factive = exp(-fractwd * bmax)
        nscfrac = nsc / (factive * (tb + fol) * cfrac)
        lit = nscfrac <= 0.01 ? fol : ep.tofol[s] * fol
        fol -= lit
        ca.acc_folsen[j] += lit
      end
      growm = -1
    else
      growm = growm >= 1 ? min(growm + 1, 100) : 1
    end
    if growm >= 1 && growm != 2
      factive = exp(-fractwd * bmax)
      idealfol = Float64(ca.adjff[j]) * factive * tb
      folcost = 0.0; foltent = 0.0
      if growm < 2
        if idealfol > fol
          folcost = max(0.0, min(nsc, cfrac * (idealfol - fol)))
          foltent = folcost / cfrac
        end
        ca.lastlai[j] = 0.0
      end
      foltent > 0.01 && (fol += foltent)
      nsc = max(0.0, nsc - folcost)
    end
  end

  # ---- 5.5 light
  laisum_above = Float64(ca.laisum[j])
  lai_i = lai_sub(fol, ep.slwmax[s], ep.slwdel[s], idx, laisum_above)
  layerpar = w.par_layer * exp(-ep.k[s] * laisum_above)
  frad = 1.0 - exp(-layerpar * 0.6931471805599453 / ep.halfsat[s])
  ca.laisum[j] = laisum_above + lai_i
  ca.fradm[j] += frad

  # ---- 5.6-5.8 water stress, photosynthesis, transpiration, respiration
  gpp = 0.0; transp = 0.0; folresp = 0.0; netpsn = 0.0
  if growm >= 1
    ph = press_head(water, w.por, w.tA, w.tB)
    fw = fwater_of(ep.h1[s], ep.h2[s], ep.h3[s], ep.h4[s], ph)
    folN = ep.folN[s]
    cica = -0.075 * folN + 0.875
    ci_elev = w.co2 * cica
    ca_ci = w.co2 - ci_elev
    delamax = max(0.0, (ci_elev - 40.0) / (ci_elev + 80.0) * (350.0 + 80.0) / (350.0 - 40.0))
    vmol = 8314.47 * (w.tmin + 273.0) / 101.3
    jco2 = 0.139 * (ca_ci / vmol) * 1e-6
    amax = delamax * (ep.amaxa[s] + ep.amaxb[s] * folN)                    # CO2AMaxBEff = 1 -> AmaxB_CO2 = AmaxB
    basefolresp = spm.basefrac * amax
    grossamax = amax * cst.amax_frac + basefolresp
    refgross = w.dayspan * (grossamax * spm.dvpd * w.dayl * 12.0) / 1e9
    fage = max(0.0, 1.0 - (age / ep.longevity[s])^ep.psnagered[s])
    gpp_pot = (1.0 / IMAX) * spm.ftemp * frad * fage * refgross * fol
    clp = Float64(ca.clp[j])
    wr = spm.jh2o / jco2                                                   # JH2O/JCO2
    pot_tr = 0.0015 * gpp_pot * wr * clp
    wiltpt = (ep.h4[s] * 9.804139432 / w.tA)^(1.0 / -w.tB)
    avail = (water - wiltpt) * rd
    if pot_tr > avail
      transp = max(avail, 0.0)
      gpp = (clp > 0 && wr > 0) ? (transp / 0.0015) / wr / clp : 0.0
    else
      gpp = gpp_pot * fw
      transp = pot_tr * fw
    end
    water = max(0.0, water - transp / rd)
    refresp = basefolresp * spm.q10f * w.dayspan * (3600.0 * 24.0) * 12.0 / 1e9
    folresp = refresp * fol / IMAX
    netpsn = gpp - folresp
    nsc = max(0.0, nsc + netpsn)
  end

  # ---- write back
  w.water = water
  ca.tbio[j] = tb; ca.fol[j] = fol; ca.nsc[j] = nsc; ca.bmax[j] = bmax; ca.age[j] = age; ca.growm[j] = growm
  ca.idx[j] = idx + 1
  ca.gpp_m[j] += gpp; ca.folresp_m[j] += folresp; ca.netpsn_m[j] += netpsn; ca.transp_m[j] += transp
  ca.acc_net[j] += netpsn
  return nothing
end

"""Ground evaporation for `nev` events assigned to the sublayer just processed (spec 5.10)."""
@inline function ground_evap!(w::MonthWork, nev::Int, cst::PnETConst)
  (w.gpe > 0 && w.snowpack == 0) || return nothing
  evdepth = min(w.rd, cst.evap_depth)
  for _ in 1:nev
    ev = max(0.0, min(w.gpe, (w.water - w.wp) * evdepth))
    w.water = max(0.0, w.water - ev / w.rd)
  end
  return nothing
end

"""After all sublayers of a layer: LastLAI, CanopyLayerProp (+ crowding rescale), PAR transmission (spec 4.2, 4.3). Returns (par_below, layer_wtLAI)."""
function layer_post!(w::MonthWork, ca::CohArrays, o::Int, ord::Vector{Int32}, c0::Int, c1::Int, ep::PnETEcoParams, cst::PnETConst)
  par_in = w.par_layer
  wsum = 0.0; lwsum = 0.0; agbsum = 0.0; propsum = 0.0
  for q in c0:c1
    j = o + Int(ord[q]); s = Int(ca.species[j])
    laisum = Float64(ca.laisum[j])
    ll = Float64(ca.lastlai[j])
    if ca.growm[j] >= 1 && laisum > ll
      ll = laisum
      ca.lastlai[j] = ll
    end
    cp = min(ll / ep.maxlai[s], Float64(ca.cgs[j]))
    wsum += exp(-ep.k[s] * laisum) * cp
    lwsum += laisum * cp
    propsum += cp
    agbsum += (1 - ep.fracbelowg[s]) * Float64(ca.tbio[j]) + Float64(ca.fol[j])
  end
  crowded = propsum > 1.0
  prodexp = 0.0
  for q in c0:c1
    j = o + Int(ord[q]); s = Int(ca.species[j])
    laisum = Float64(ca.laisum[j])
    cp = min(Float64(ca.lastlai[j]) / ep.maxlai[s], Float64(ca.cgs[j]))
    if crowded && ca.growm[j] == 1
      adj = cp / propsum
      cp = (adj - cp) * cst.canopy_sum_scale + cp
      ca.cgs[j] = min(Float64(ca.cgs[j]), cp)
    end
    ca.clp[j] = cp
    prodexp += -ep.k[s] * laisum * cp                 # log of prod frac^cp
  end
  if agbsum > 0
    return (crowded ? par_in * exp(prodexp) : par_in * (wsum + (1.0 - propsum)), lwsum)
  end
  return (par_in, 0.0)
end

# ---------------------------------------------------------------------------------------------------------------------
# Site-year
# ---------------------------------------------------------------------------------------------------------------------
function build_layers!(scr::PnETScratch, ca::CohArrays, o::Int, n::Int, cst::PnETConst, rng::R) where {R<:Random.AbstractRNG}
  ord = scr.ord
  @inbounds for k in 1:n
    ord[k] = Int32(k)
  end
  # insertion sort, BiomassMax descending (stable)
  @inbounds for a in 2:n
    key = ord[a]; kb = ca.bmax[o+Int(key)]
    b = a - 1
    while b >= 1 && ca.bmax[o+Int(ord[b])] < kb
      ord[b+1] = ord[b]; b -= 1
    end
    ord[b+1] = key
  end
  # layer split (spec 4.1)
  nl = 1
  scr.lay_c[1] = 1
  ref = ca.bmax[o+Int(ord[1])]
  @inbounds for a in 2:n
    b = ca.bmax[o+Int(ord[a])]
    if b / ref < cst.layer_thresh && nl < Int(cst.max_layers)
      nl += 1
      scr.lay_c[nl] = a
      ref = b
    end
  end
  scr.lay_c[nl+1] = n + 1
  # processing sequence: cohort index repeated IMAX times, Fisher-Yates shuffled within each layer
  seq = scr.seq
  p = 0
  for l in 1:nl
    scr.lay_p[l] = p + 1
    for q in scr.lay_c[l]:scr.lay_c[l+1]-1
      kk = ord[q]
      for _ in 1:IMAX
        p += 1; seq[p] = kk
      end
    end
    lo = scr.lay_p[l]
    for a in p:-1:lo+1
      b = lo + rand(rng, 0:a-lo)
      seq[a], seq[b] = seq[b], seq[a]
    end
  end
  scr.lay_p[nl+1] = p + 1
  return nl
end

"""One simulation year (12 months) for site i. `rng` is the site RNG (type-specialised by the caller). `cb(m, ca, o, n, w)` is an optional per-month hook."""
function site_year!(soa, i::Int, year::Int, ep::PnETEcoParams, cst::PnETConst, clim::PnETClimate, scr::PnETScratch, rng::R; cb::F=nothing) where {R<:Random.AbstractRNG,F}
  sc = soa.scalar
  ca = CohArrays(soa.csr)
  o = Int(soa.refs.cohort[i]) - 1
  n = Int(sc.live[i])
  if n == 0
    sc.B[i] = 0
    return nothing
  end
  ensure_capacity!(scr, n)
  materialize_pending!(ca, o, n, ep, cst)
  @inbounds for k in 1:n
    j = o + k
    ca.acc_net[j] = 0; ca.acc_maint[j] = 0; ca.acc_folsen[j] = 0; ca.acc_turn[j] = 0
  end
  nl = build_layers!(scr, ca, o, n, cst, rng)
  Ntot = n * IMAX

  cid = Int(sc.clim_id[i])
  yi = clamp(Int(sc.cal_year0[i]) + year, clim.y1, clim.y2) - clim.y1 + 1
  daylength_ra!(scr.hr, scr.ra, Float64(sc.lat[i]))
  D = clim.D
  w = scr.w
  w.por = sc.soil_por[i]; w.fc = sc.soil_fc[i]; w.wp = sc.soil_wp[i]; w.tA = sc.soil_A[i]; w.tB = sc.soil_B[i]
  w.rd = Float64(sc.rooting_depth[i]) * ep.rd_mult
  w.water = sc.soil_water[i]
  snowpack = Float64(sc.snowpack[i])
  canopy_lai_prev = Float64(sc.canopy_lai_prev[i])
  nevents = Int(cst.precip_events)
  evc = scr.evc

  for m in 1:12
    tmin = Float64(D[DV_TMIN, m, yi, cid]); tmax = Float64(D[DV_TMAX, m, yi, cid])
    tave = Float64(D[DV_TAVE, m, yi, cid]); tday = Float64(D[DV_TDAY, m, yi, cid])
    prec = Float64(D[DV_PREC, m, yi, cid]); co2 = Float64(D[DV_CO2, m, yi, cid])
    vpd = Float64(D[DV_VPD, m, yi, cid]); dtr = Float64(D[DV_DTR, m, yi, cid])
    dayspan = Float64(DAYSPAN[m])
    hr = scr.hr[m]
    dayl = 3600.0 * hr; nightl = 3600.0 * (24.0 - hr)
    par0 = par_daytime(scr.ra[m], dtr, dayl)
    w.tmin = tmin; w.tave = tave; w.tday = tday; w.co2 = co2; w.vpd = vpd; w.dayl = dayl; w.nightl = nightl
    w.dayspan = dayspan; w.month = m

    @inbounds for k in 1:n
      j = o + k
      if m == 1     # CalcAdjFracFol + ClearFRad (Jan, spec 6.3)
        s = Int(ca.species[j])
        if ca.fradn[j] > 0
          mf = Float64(ca.fradsum[j]) / Float64(ca.fradn[j])
          ca.adjff[j] = ep.fracfol[s] + (ep.maxfracfol[s] - ep.fracfol[s]) * mf^ep.fracfolshape[s]
        else
          ca.adjff[j] = ep.fracfol[s]
        end
        ca.fradsum[j] = 0; ca.fradn[j] = 0
      end
      ca.idx[j] = 0; ca.laisum[j] = 0; ca.fradm[j] = 0
      ca.gpp_m[j] = 0; ca.folresp_m[j] = 0; ca.netpsn_m[j] = 0; ca.transp_m[j] = 0
    end

    # ---- 6.1 snow (melt from the pack BEFORE adding this month's snow)
    maxmelt = 2.74 * max(0.0, tave) * dayspan
    snowmelt = min(snowpack, maxmelt)
    newsnow = clamp((tave - 2.0) / (-7.0), 0.0, 1.0) * prec
    snowpack += newsnow * (1.0 - cst.snow_sublim) - snowmelt
    # ---- 6.2 throughfall, events
    newrain = prec - newsnow
    interception = newrain * (1.0 - exp(-cst.prec_int_const * canopy_lai_prev))
    surf = newrain - interception
    precin = surf * (1.0 - cst.prec_loss_frac)
    pe = nevents > 0 ? precin / nevents : 0.0
    # Hamon RET (mm/day) and ground PET per event (spec 5.10)
    ret = if tave < 0
      0.0
    else
      es = 6.108 * exp(17.27 * tave / (tave + 237.3))
      1.2 * 0.165 * 216.7 * (dayl / 3600.0 / 12.0) * (es / (tave + 273.3))
    end
    gpet = cst.ret_crop * ret * exp(-cst.et_ext * canopy_lai_prev) * dayspan
    w.melt_in = snowmelt / Ntot; w.pe = pe; w.gpe = nevents > 0 ? gpet / nevents : 0.0; w.snowpack = snowpack
    @inbounds for p in 1:Ntot
      evc[p] = 0
    end
    @inbounds for _ in 1:nevents
      evc[rand(rng, 1:Ntot)] += 1
    end

    # ---- layers top -> bottom
    w.par_layer = par0
    lwtot = 0.0
    for l in 1:nl
      @inbounds for p in scr.lay_p[l]:scr.lay_p[l+1]-1
        k = Int(scr.seq[p])
        j = o + k
        if ca.idx[j] == 0
          s = Int(ca.species[j])
          scr.spm[k] = sp_month(ep, cst, s, tmin, tave, tday, dayl, nightl, vpd)
        end
        nev = Int(evc[p])
        sublayer!(w, ca, o, k, nev, ep, cst, scr.spm[k])
        nev > 0 && ground_evap!(w, nev, cst)
      end
      par_below, lw = layer_post!(w, ca, o, scr.ord, scr.lay_c[l], scr.lay_c[l+1] - 1, ep, cst)
      lwtot += lw
      w.par_layer = par_below
    end
    canopy_lai_prev = lwtot
    # StoreFRad (leaf-on cohorts only)
    @inbounds for k in 1:n
      j = o + k
      if ca.growm[j] >= 1
        ca.fradsum[j] += ca.fradm[j] / IMAX
        ca.fradn[j] += 1
      end
    end
    cb === nothing || cb(m, ca, o, n, w)      # test/diagnostic hook (compiled away when `nothing`)
  end

  # ---- write site state, remove dead cohorts (NSCfrac <= 0.01), refresh reported biomass
  sc.soil_water[i] = w.water
  sc.snowpack[i] = snowpack
  sc.canopy_lai_prev[i] = canopy_lai_prev
  sc.subcanopy_par[i] = w.par_layer
  site = SiteView(soa, i)
  last = n
  k = 1
  while k <= last
    j = o + k
    s = Int(ca.species[j])
    factive = exp(-ep.fractwd[s] * Float64(ca.bmax[j]))
    nscfrac = Float64(ca.nsc[j]) / (factive * (Float64(ca.tbio[j]) + Float64(ca.fol[j])) * cst.cfrac)
    alive = nscfrac > 0.01 && isfinite(nscfrac) && ca.tbio[j] > 0
    if alive
      k += 1
    else
      k != last && PanCore.swap_cohort!(site, k, last)
      last -= 1
    end
  end
  B = 0.0
  @inbounds for k in 1:last
    j = o + k
    s = Int(ca.species[j])
    b = ((1 - ep.fracbelowg[s]) * Float64(ca.tbio[j]) + Float64(ca.fol[j])) * Float64(ca.clp[j])
    ca.bio[j] = b
    B += Float64(ca.bio[j])
  end
  sc.live[i] = last
  sc.old[i] = last
  sc.B[i] = B
  return nothing
end

function PanCore.process_plugin!(soa::PanCore.AnySoA, ::Type{PluginType}, current_time::Int; ctx::NamedTuple)
  eco = ctx.eco; cst = ctx.cst; clim = ctx.clim; pool = ctx.scratch
  PanCore.@maybe_threads PanCore.PARALLEL_SITES[] for i in 1:soa.n
    @inbounds begin
      if soa.scalar.active[i]
        scr = pool[Threads.threadid()]
        ep = eco[soa.scalar.eco_id[i]]
        rng = soa.scalar.rng[i]
        if rng isa Random.Xoshiro            # static dispatch for the production RNG (keeps the loop allocation-free)
          site_year!(soa, i, current_time, ep, cst, clim, scr, rng)
        else
          site_year!(soa, i, current_time, ep, cst, clim, scr, rng)
        end
      end
      soa.scalar._new_cohort_counts[i] = soa.scalar.live[i]
    end
  end
  # shrink CSR capacity only when a cohort actually died (keeps the steady state allocation-free); the injection code relies on
  # _new_cohort_counts == live for every site, which holds either way.
  refs = soa.refs.cohort; live = soa.scalar.live
  shrink = false
  @inbounds for i in 1:soa.n
    if live[i] != Int(refs[i+1]) - Int(refs[i])
      shrink = true
      break
    end
  end
  shrink && PanCore.readjust_soa!(soa, (cohort=soa.scalar._new_cohort_counts,))
  return nothing
end

# ---------------------------------------------------------------------------------------------------------------------
# Cohort add / external biomass writes
# ---------------------------------------------------------------------------------------------------------------------
"""
    add_cohort!(site, species_local, age, agb; eco=nothing, cst=FIXED[])

Append a cohort with OBSERVED reported aboveground biomass `agb` (g/m2). The pools (TotalBiomass, Fol, NSC, ...) depend on the candidate's
parameters, so they are initialised lazily from `agb` by the spec-section-8 solve at the start of the next `process_plugin!` year
(`c_tbio == 0` marks a pending cohort). Passing `eco` (a PnETEcoParams) initialises immediately.
"""
function add_cohort!(site::SiteView, species::Integer, age::Real, agb::Real; eco::Union{Nothing,PnETEcoParams}=nothing, cst::PnETConst=FIXED[])
  site.live += 1
  age > 1 && (site.old += 1)
  k = Int(site.live)
  site.c_species[k] = UIntType(species)
  site.c_age[k] = FloatType(age)
  site.c_bio[k] = FloatType(agb)
  site.c_tbio[k] = 0
  site.c_bmax[k] = 0; site.c_fol[k] = 0; site.c_nsc[k] = 0; site.c_clp[k] = 0; site.c_cgs[k] = 1; site.c_lastlai[k] = 0
  site.c_adjff[k] = 0; site.c_fradsum[k] = 0; site.c_fradn[k] = 0; site.c_growm[k] = 0
  site.c_idx[k] = 0; site.c_laisum[k] = 0; site.c_fradm[k] = 0
  site.c_gpp_m[k] = 0; site.c_folresp_m[k] = 0; site.c_netpsn_m[k] = 0; site.c_transp_m[k] = 0
  site.c_acc_net[k] = 0; site.c_acc_maint[k] = 0; site.c_acc_folsen[k] = 0; site.c_acc_turn[k] = 0
  if eco !== nothing
    soa = site.soa
    materialize_cohort!(CohArrays(soa.csr), Int(soa.refs.cohort[site.i]) - 1, k, eco, cst)
  end
  return nothing
end

"""
    set_cohort_bio!(site, k, newbio)

Set the REPORTED aboveground biomass of cohort k. A pending cohort just stores the value; an initialised cohort has TotalBiomass, Fol and NSC
rescaled proportionally (reported = ((1-FBG)*TB + Fol)*CLP scales exactly; BiomassMax only grows).
"""
function set_cohort_bio!(site::SiteView, k::Integer, newbio::Real)
  cb = site.c_bio
  tb = site.c_tbio
  old = cb[k]
  if tb[k] <= 0 || !(old > 0)
    cb[k] = FloatType(newbio)
    tb[k] = 0
    return nothing
  end
  if !(newbio > 0)
    cb[k] = FloatType(newbio); tb[k] = 0
    return nothing
  end
  sc = FloatType(newbio) / old
  tb[k] *= sc
  site.c_fol[k] *= sc
  site.c_nsc[k] *= sc
  site.c_bmax[k] = max(site.c_bmax[k], tb[k])
  cb[k] = FloatType(newbio)
  return nothing
end
