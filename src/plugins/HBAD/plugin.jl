struct HBAD <: AbstractPlugin end
const PluginType = HBAD

# ---------------------------------------------------------------------------------------------------------------------
# SoA layout
# ---------------------------------------------------------------------------------------------------------------------
PanCore.scalar_arrays(::Type{PluginType}, n::Int) =
  (
    B=Vector{FloatType}(undef, n),          # total REPORTED aboveground biomass (sum of c_bio), g/m2
    BA=Vector{FloatType}(undef, n),         # stand basal area, m2/ha
    SDI=Vector{FloatType}(undef, n),        # Reineke stand density index (metric: trees/ha at 25.4 cm)
    RD=Vector{FloatType}(undef, n),         # relative density SDI / site SDImax
    thin_tph=Vector{FloatType}(undef, n),   # trees/ha removed by self-thinning in the last step (diagnostic)
  )

PanCore.csr_fields(::Type{PluginType}) =
  (
    c_species=:cohort,   # eco-LOCAL species index (BiomassSuccession convention)
    c_age=:cohort,
    c_bio=:cohort,       # REPORTED aboveground biomass g/m2 = allometric AGB of (c_tph, c_dq); exact observed value right after an injection
    c_tph=:cohort,       # trees per hectare (plot-scale TPA_UNADJ x 2.47105 accounting of the FIA cohort table)
    c_dq=:cohort,        # quadratic mean diameter, cm
    c_h=:cohort,         # mean tree height, m
    c_bal=:cohort,       # basal area (m2/ha) of larger-dq cohorts at the start of the last step (scratch / diagnostic)
    sp_mature=:species,  # unused (no establishment); its length carries the eco species count for the shared loss
  )

PanCore.csr_arrays(::Type{PluginType}, nnz::NamedTuple) =
  (
    c_species=Vector{UIntType}(undef, nnz.cohort),
    c_age=Vector{FloatType}(undef, nnz.cohort),
    c_bio=Vector{FloatType}(undef, nnz.cohort),
    c_tph=Vector{FloatType}(undef, nnz.cohort),
    c_dq=Vector{FloatType}(undef, nnz.cohort),
    c_h=Vector{FloatType}(undef, nnz.cohort),
    c_bal=Vector{FloatType}(undef, nnz.cohort),
    sp_mature=Vector{Bool}(undef, nnz.species),
  )

# ---------------------------------------------------------------------------------------------------------------------
# Stand statistics
# ---------------------------------------------------------------------------------------------------------------------
const _PI4E4 = pi / 4 / 1.0e4        # BA(m2/ha) = _PI4E4 * dq(cm)^2 * tph

"""(BA, SDI, SDImax) of cohorts `o+1 : o+n` with tph > 0. SDImax is the BA-weighted mean of the species values. 0 when empty."""
@inline function stand_stats(species::Vector{UIntType}, tph::Vector{FloatType}, dq::Vector{FloatType}, o::Int, n::Int, sdimax::Vector{Float64})
  ba = 0.0; sdi = 0.0; sw = 0.0
  @inbounds for k in 1:n
    j = o + k
    t = Float64(tph[j]); d = Float64(dq[j])
    b = _PI4E4 * d * d * t
    ba += b
    sdi += t * (d / HB_REINEKE_REF)^HB_REINEKE
    sw += b * sdimax[Int(species[j])]
  end
  return ba, sdi, (ba > 0 ? sw / ba : 0.0)
end

# ---------------------------------------------------------------------------------------------------------------------
# One site, one year
# ---------------------------------------------------------------------------------------------------------------------
"""
    site_year!(soa, i, ep, cst)

Annual update of site `i` (see docs/hbad_plugin.md): (a) BA/SDI/RD + BAL snapshot, (b) longevity removal then age+1, (c) diameter growth,
(d) height by Chapman-Richards ratio, (e) expected-value mortality, (f) self-thinning from the smallest cohort, (g) removal of cohorts < 1 tree/ha,
(h) reported AGB = allometric AGB, site totals. Allocation-free; reads/writes only site i. Establishment hook: see `# ESTABLISHMENT HOOK`.
"""
function site_year!(soa::PanCore.AnySoA, i::Int, ep::HBADEcoParams, cst::HBADConst)
  csr = soa.csr
  species = csr.c_species; age = csr.c_age; bio = csr.c_bio; tph = csr.c_tph; dq = csr.c_dq; hh = csr.c_h; bal = csr.c_bal
  sc = soa.scalar
  o = Int(soa.refs.cohort[i]) - 1
  n = Int(sc.live[i])
  if n == 0
    sc.B[i] = 0; sc.BA[i] = 0; sc.SDI[i] = 0; sc.RD[i] = 0; sc.thin_tph[i] = 0; sc.old[i] = 0
    return nothing
  end
  site = SiteView(soa, i)

  # (a) snapshot: stand BA / SDI / RD and per-cohort BAL (BA of cohorts with strictly larger dq)
  ba0, sdi0, sdm0 = stand_stats(species, tph, dq, o, n, ep.sdimax)
  rd0 = sdm0 > 0 ? sdi0 / sdm0 : 0.0
  rdc = min(rd0, cst.rd_max)
  @inbounds for k in 1:n
    dk = dq[o+k]
    s = 0.0
    for m in 1:n
      dm = Float64(dq[o+m])
      if dm > Float64(dk)
        s += _PI4E4 * dm * dm * Float64(tph[o+m])
      end
    end
    bal[o+k] = s
  end

  # (b) longevity removal (swap-remove moves every per-cohort field incl. the BAL snapshot), then age + 1
  last = n
  k = 1
  @inbounds while k <= last
    if Float64(age[o+k]) >= ep.longevity[Int(species[o+k])]
      k != last && PanCore.swap_cohort!(site, k, last)
      last -= 1
    else
      k += 1
    end
  end
  @inbounds for k in 1:last
    age[o+k] += one(FloatType)
  end

  # (c) diameter growth, (d) height, (e) mortality -- all from the pre-growth diameter and the step-start BAL / RD
  @inbounds for k in 1:last
    j = o + k
    s = Int(species[j])
    p = ep.pri[s]
    D = Float64(dq[j])
    Dc = max(D, cst.dmin_cov)
    lnD = log(Dc / 20.0)
    b10 = Float64(bal[j]) / 10.0
    a = max(Float64(age[j]), 1.0)
    dD = ep.gmult[s] * exp(p.g_b0 + p.g_b1 * lnD + p.g_b2 * Dc / 20.0 + p.g_b3 * log(a / 30.0) - p.g_k * b10)
    dD = clamp(dD, 0.0, cst.gmax)
    Dn = min(D + dD, HB_DQ_MAX)
    h = Float64(hh[j])
    cr0 = chapman_richards(p.hd_a, p.hd_b, p.hd_c, D)
    cr1 = chapman_richards(p.hd_a, p.hd_b, p.hd_c, Dn)
    hn = cr0 < 0.1 ? 1.3 + cr1 : 1.3 + (h - 1.3) * cr1 / cr0
    hn = max(hn, 1.3)
    haz = ep.mmult[s] * exp(p.m_c0 + p.m_c1 * lnD + p.m_c2 * Dc / 20.0 + p.m_c3 * rdc + p.m_c4 * b10 + p.m_c5 * rdc * lnD)
    haz = min(haz, cst.haz_max)
    dq[j] = FloatType(Dn)
    hh[j] = FloatType(hn)
    tph[j] = FloatType(Float64(tph[j]) * exp(-haz))
  end

  # (f) self-thinning toward the site SDImax, smallest dq first (reduces tph at constant dq). SDImax is the BA-weighted species mean and
  # shifts slightly as small cohorts are removed, so the thinning pass is repeated (converges in 2-3 passes) until RD <= 1.
  thinned = 0.0
  @inbounds for _ in 1:8
    _, sdi1, sdm1 = stand_stats(species, tph, dq, o, last, ep.sdimax)
    excess = sdi1 - sdm1
    (sdm1 > 0 && excess > 1e-6 * sdm1) || break
    while excess > 1e-9 * sdm1
      km = 0; dmin = Inf
      for k in 1:last
        t = tph[o+k]
        if t > 0
          d = Float64(dq[o+k])
          if d < dmin
            dmin = d; km = k
          end
        end
      end
      km == 0 && break
      j = o + km
      u = (Float64(dq[j]) / HB_REINEKE_REF)^HB_REINEKE
      t = Float64(tph[j])
      c = t * u
      if c <= excess
        tph[j] = zero(FloatType); excess -= c; thinned += t
      else
        r = excess / u
        tph[j] = FloatType(t - r); thinned += r; excess = 0.0
      end
    end
  end

  # (g) remove cohorts below the minimum density
  k = 1
  @inbounds while k <= last
    if Float64(tph[o+k]) < cst.min_tph
      k != last && PanCore.swap_cohort!(site, k, last)
      last -= 1
    else
      k += 1
    end
  end

  # ESTABLISHMENT HOOK: new cohorts (seedlings/saplings) would be appended here (needs CSR capacity growth via readjust_soa!, see
  # BiomassSuccession reproduction_commit_step!); not implemented in this version.

  # (h) reported AGB = allometric AGB of the state; site totals
  B = 0.0
  @inbounds for k in 1:last
    j = o + k
    p = ep.pri[Int(species[j])]
    b = agb_of(p, Float64(tph[j]), Float64(dq[j]))
    bio[j] = FloatType(b)
    B += Float64(bio[j])
  end
  ba2, sdi2, sdm2 = stand_stats(species, tph, dq, o, last, ep.sdimax)
  sc.live[i] = last
  sc.old[i] = last
  sc.B[i] = FloatType(B)
  sc.BA[i] = FloatType(ba2)
  sc.SDI[i] = FloatType(sdi2)
  sc.RD[i] = FloatType(sdm2 > 0 ? sdi2 / sdm2 : 0.0)
  sc.thin_tph[i] = FloatType(thinned)
  return nothing
end

function PanCore.process_plugin!(soa::PanCore.AnySoA, ::Type{PluginType}, current_time::Int; ctx::NamedTuple)
  eco = ctx.eco; cst = ctx.cst
  PanCore.@maybe_threads PanCore.PARALLEL_SITES[] for i in 1:soa.n
    @inbounds begin
      if soa.scalar.active[i]
        site_year!(soa, i, eco[soa.scalar.eco_id[i]], cst)
      end
      soa.scalar._new_cohort_counts[i] = soa.scalar.live[i]
    end
  end
  # shrink CSR capacity only when a cohort actually died (keeps the steady state allocation-free)
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
    add_cohort!(site, species_local, age, agb)

Append a cohort with OBSERVED aboveground biomass `agb` (g/m2): (tph, dq, h) from the empirical translation (`translate`), `c_bio = agb` exactly.
Reads the module-level `PRIORS[]` / `ECO_GIDS[]` (set by `configure!`). `prior=` overrides the species prior (tests).
"""
function add_cohort!(site::SiteView, species::Integer, age::Real, agb::Real; prior::Union{Nothing,SpPrior}=nothing)
  p = prior === nothing ? PRIORS[].sp[global_species(Int(site.eco_id), species)] : prior
  site.live += 1
  age > 1 && (site.old += 1)
  k = Int(site.live)
  tph, dq, h = translate(p, Float64(age), Float64(agb))
  site.c_species[k] = UIntType(species)
  site.c_age[k] = FloatType(age)
  site.c_bio[k] = FloatType(agb)
  site.c_tph[k] = FloatType(tph)
  site.c_dq[k] = FloatType(dq)
  site.c_h[k] = FloatType(h)
  site.c_bal[k] = zero(FloatType)
  return nothing
end

"""
    set_cohort_bio!(site, k, newbio)

Set the reported AGB of cohort k. Trees are removed / added at the same size: `c_tph` is scaled by newbio/oldbio (dq and h unchanged), so the
allometric AGB equals `newbio`. A cohort with no previous biomass is re-translated from (species, age, newbio).
"""
function set_cohort_bio!(site::SiteView, k::Integer, newbio::Real)
  old = site.c_bio[k]
  nb = max(Float64(newbio), 0.0)
  if old > 0 && site.c_tph[k] > 0
    site.c_tph[k] = FloatType(Float64(site.c_tph[k]) * nb / Float64(old))
  elseif nb > 0
    p = PRIORS[].sp[global_species(Int(site.eco_id), site.c_species[k])]
    tph, dq, h = translate(p, Float64(site.c_age[k]), nb)
    site.c_tph[k] = FloatType(tph); site.c_dq[k] = FloatType(dq); site.c_h[k] = FloatType(h)
  else
    site.c_tph[k] = zero(FloatType)
  end
  site.c_bio[k] = FloatType(nb)
  return nothing
end
