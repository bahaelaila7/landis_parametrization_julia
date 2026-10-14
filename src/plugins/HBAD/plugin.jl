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
    shade_class=Vector{UIntType}(undef, n), # 0..5 from RD and MIN_REL_RD (set by the establishment check; 0 when establishment is off)
    no_establish=Vector{Bool}(undef, n),    # true: no natural establishment on this site (Sim A / injection-driven runs)
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
    sp_mature=:species,  # per eco species: a cohort of age >= MATURITY is present (written by the establishment check only); its length carries the eco species count for the shared loss
    sp_estab=:species,   # per eco species: establishment succeeded this year and needs a new cohort slot (scratch between the two phases)
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
    sp_estab=Vector{Bool}(undef, nnz.species),
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
    site_prepare!(soa, i, ep, cst)

Phase 1 of the annual update of site `i` (see docs/hbad_plugin.md): (a) BA/SDI/RD + BAL snapshot, (b) longevity removal then age+1, (c) diameter growth,
(d) height by Chapman-Richards ratio, (e) expected-value mortality, (f) self-thinning from the smallest cohort, (g) removal of cohorts < 1 tree/ha,
(g2) establishment check (only when the site is not flagged `no_establish`): shade class from RD, one uniform draw per candidate species; a success either
merges into an existing age-1 cohort or flags `sp_estab` (a new cohort slot is needed). Sets `live` and `_new_cohort_counts = live + #new cohorts`.
Allocation-free; reads/writes only site i. The CSR capacity is adjusted by `process_plugin!` between the phases; `site_commit!` is phase 2.
"""
function site_prepare!(soa::PanCore.AnySoA, i::Int, ep::HBADEcoParams, cst::HBADConst)
  csr = soa.csr
  species = csr.c_species; age = csr.c_age; bio = csr.c_bio; tph = csr.c_tph; dq = csr.c_dq; hh = csr.c_h; bal = csr.c_bal
  sc = soa.scalar
  o = Int(soa.refs.cohort[i]) - 1
  n = Int(sc.live[i])
  if n == 0 && sc.no_establish[i]
    sc.B[i] = 0; sc.BA[i] = 0; sc.SDI[i] = 0; sc.RD[i] = 0; sc.thin_tph[i] = 0; sc.old[i] = 0; sc.shade_class[i] = 0
    sc._new_cohort_counts[i] = 0
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

  # (g2) establishment (after self-thinning / removal, before the AGB update)
  nnew = 0
  if sc.no_establish[i]
    sc.shade_class[i] = 0
  else
    nnew = establish_check!(soa, i, ep, cst, o, last)
  end
  sc.live[i] = last
  sc.thin_tph[i] = FloatType(thinned)
  sc._new_cohort_counts[i] = last + nnew
  return nothing
end

"""
    site_commit!(soa, i, ep, cst)

Phase 2 of the annual update: append the cohorts flagged by `establish_check!` (age 1, `ESTAB_TPH`, dq = `estab_dq`, h = 1.3 + CR(dq)) into the slots made
available by `process_plugin!`, then (h) reported AGB = allometric AGB of every cohort and the site totals. Errors if the site has no free slot (cannot
happen through `process_plugin!`; guards direct callers against writing into the neighbouring site).
"""
function site_commit!(soa::PanCore.AnySoA, i::Int, ep::HBADEcoParams, cst::HBADConst)
  csr = soa.csr
  species = csr.c_species; age = csr.c_age; bio = csr.c_bio; tph = csr.c_tph; dq = csr.c_dq; hh = csr.c_h; bal = csr.c_bal
  sc = soa.scalar
  o = Int(soa.refs.cohort[i]) - 1
  cap = Int(soa.refs.cohort[i+1]) - Int(soa.refs.cohort[i])
  last = Int(sc.live[i])
  if !sc.no_establish[i]
    so = Int(soa.refs.species[i]) - 1
    nsp = Int(soa.refs.species[i+1]) - Int(soa.refs.species[i])
    sest = csr.sp_estab
    @inbounds for j in 1:nsp
      if sest[so+j]
        last >= cap && error("HBAD site_commit!: site $i has no free cohort slot (capacity $cap) - the caller must readjust_soa! first")
        last += 1
        k = o + last
        d0 = cst.estab_dq
        species[k] = UIntType(j); age[k] = one(FloatType); tph[k] = FloatType(ep.estab_tph[j]); dq[k] = FloatType(d0)
        hh[k] = FloatType(1.3 + chapman_richards(ep.pri[j].hd_a, ep.pri[j].hd_b, ep.pri[j].hd_c, d0)); bal[k] = zero(FloatType)
        sest[so+j] = false
      end
    end
  end

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
  return nothing
end

"""Both phases for one site (tests, single-site tools). Valid only when no new cohort slot is needed (`_new_cohort_counts[i] <= capacity`)."""
function site_year!(soa::PanCore.AnySoA, i::Int, ep::HBADEcoParams, cst::HBADConst)
  site_prepare!(soa, i, ep, cst)
  site_commit!(soa, i, ep, cst)
end

# Union-split so that the per-site RNG (stored as AbstractRNG) is called statically in the production case (Xoshiro = Pan.RNGType).
@inline _draw(rng::Random.AbstractRNG) = rng isa Random.Xoshiro ? rand(rng::Random.Xoshiro) : rand(rng)

"""Shade class 0..5 = number of ascending RD thresholds exceeded by the stand's relative density."""
@inline shade_class_of(rd::Float64, t::NTuple{5,Float64}) = Int(rd > t[1]) + Int(rd > t[2]) + Int(rd > t[3]) + Int(rd > t[4]) + Int(rd > t[5])

"""Merge `rtph` trees of size (`rdq`, `rh`) into cohort k: densities add, dq = quadratic mean, h = tph-weighted mean."""
@inline function merge_cohort!(tph, dq, hh, k::Int, rtph::Float64, rdq::Float64, rh::Float64)
  t0 = Float64(tph[k]); d0 = Float64(dq[k]); h0 = Float64(hh[k])
  tt = t0 + rtph
  dq[k] = FloatType(sqrt((t0 * d0 * d0 + rtph * rdq * rdq) / tt))
  hh[k] = FloatType((t0 * h0 + rtph * rh) / tt)
  tph[k] = FloatType(tt)
  return nothing
end

"""
    establish_check!(soa, i, ep, cst, o, last) -> Int

Natural establishment of site `i` (cohorts `o+1:o+last` after thinning/removal). Writes `sp_mature` (cohort of age >= MATURITY present), the site
`shade_class` (RD vs `MIN_REL_RD`) and `sp_estab`. For every candidate species (mature, or all with `seed_rain`) ONE uniform draw `u` of the site RNG
is consumed (fixed draw count -> deterministic for any thread count); success iff `u < PROB_ESTAB * SUFFICIENT_LIGHT[shade class][SHADE_TOL]`.
A success merges into an existing age-1 cohort of the species (no duplicates) or flags `sp_estab`. Returns the number of new slots needed.
"""
function establish_check!(soa::PanCore.AnySoA, i::Int, ep::HBADEcoParams, cst::HBADConst, o::Int, last::Int)::Int
  csr = soa.csr
  species = csr.c_species; age = csr.c_age; tph = csr.c_tph; dq = csr.c_dq; hh = csr.c_h
  sc = soa.scalar
  so = Int(soa.refs.species[i]) - 1
  nsp = Int(soa.refs.species[i+1]) - Int(soa.refs.species[i])
  smat = csr.sp_mature; sest = csr.sp_estab
  @inbounds for j in 1:nsp
    smat[so+j] = false; sest[so+j] = false
  end
  @inbounds for k in 1:last
    j = Int(species[o+k])
    Float64(age[o+k]) >= ep.maturity[j] && (smat[so+j] = true)
  end
  _, sdi, sdm = stand_stats(species, tph, dq, o, last, ep.sdimax)
  rd = sdm > 0 ? sdi / sdm : 0.0
  cls = shade_class_of(rd, ep.min_rel_rd)
  sc.shade_class[i] = UIntType(cls)
  rng = sc.rng[i]
  nnew = 0
  @inbounds for j in 1:nsp
    (ep.seed_rain || smat[so+j]) || continue
    u = _draw(rng)
    u < ep.prob_estab[j] * ep.light[cls*5+ep.shade_tol[j]] || continue
    kmatch = 0
    for k in 1:last
      if Int(species[o+k]) == j && age[o+k] == one(FloatType)
        kmatch = k; break
      end
    end
    if kmatch > 0
      p = ep.pri[j]
      merge_cohort!(tph, dq, hh, o + kmatch, ep.estab_tph[j], cst.estab_dq, 1.3 + chapman_richards(p.hd_a, p.hd_b, p.hd_c, cst.estab_dq))
    else
      sest[so+j] = true; nnew += 1
    end
  end
  return nnew
end

function PanCore.process_plugin!(soa::PanCore.AnySoA, ::Type{PluginType}, current_time::Int; ctx::NamedTuple)
  eco = ctx.eco; cst = ctx.cst
  PanCore.@maybe_threads PanCore.PARALLEL_SITES[] for i in 1:soa.n
    @inbounds begin
      if soa.scalar.active[i]
        site_prepare!(soa, i, eco[soa.scalar.eco_id[i]], cst)
      else
        soa.scalar._new_cohort_counts[i] = soa.scalar.live[i]
      end
    end
  end
  # resize the CSR only when a cohort died or was established (keeps the steady state allocation-free)
  refs = soa.refs.cohort; cnt = soa.scalar._new_cohort_counts
  resize = false
  @inbounds for i in 1:soa.n
    if max(Int(cnt[i]), 1) != Int(refs[i+1]) - Int(refs[i])
      resize = true
      break
    end
  end
  resize && PanCore.readjust_soa!(soa, (cohort=cnt,))
  PanCore.@maybe_threads PanCore.PARALLEL_SITES[] for i in 1:soa.n
    @inbounds if soa.scalar.active[i]
      site_commit!(soa, i, eco[soa.scalar.eco_id[i]], cst)
    end
  end
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

# ---------------------------------------------------------------------------------------------------------------------
# Planting API (scenario harvest / plant actions; also usable by tests)
# ---------------------------------------------------------------------------------------------------------------------
"""
    plant_slot_needed(site, species_local, age) -> Int

1 if `plant!` has to append a cohort (no cohort of this species and age exists on the site), 0 if it merges into an existing one.
"""
function plant_slot_needed(site::SiteView, species::Integer, age::Real=1)::Int
  n = Int(site.live)
  cs = site.c_species; ca = site.c_age
  a = FloatType(age)
  @inbounds for k in 1:n
    Int(cs[k]) == Int(species) && ca[k] == a && return 0
  end
  return 1
end

"""
    plant!(site, species_local, tph; age=1, dq=nothing, prior=nothing, cst=FIXED[]) -> Int

Plant / add `tph` trees/ha of eco-local species `species_local` at `age` (years) and quadratic mean diameter `dq` (cm; default `cst.estab_dq` = the
establishment state, clamped to [0.25, 170]). Same state initialisation as natural establishment: `h = 1.3 + CR(dq)` (Chapman-Richards of the
species prior), `c_bio` = allometric AGB of (tph, dq). If a cohort of the same species and age exists the trees are merged into it (densities add,
dq = quadratic mean, h = tph-weighted mean); otherwise a cohort is appended. Site totals `B`, `BA`, `SDI` are updated incrementally (`RD` is refreshed by
the next annual step, which recomputes it from the state). Returns the index of the cohort (1-based within the site).

CSR capacity is the CALLER's job, exactly as for `add_cohort!` in `_inject_observed_cohorts!`: before appending, set
`site._new_cohort_counts = site.live + plant_slot_needed(site, sp, age)` (summed over all plantings of that site) for every planted site and call
`PanCore.readjust_soa!(soa, (cohort=soa.scalar._new_cohort_counts,))` once (note the SiteView must be re-fetched afterwards). `Pan.plant_sites!`
does this for a list of plantings. Throws a descriptive error when no free slot exists. Reads the module-level `PRIORS[]` / `ECO_GIDS[]`.
"""
function plant!(site::SiteView, species::Integer, tph::Real; age::Real=1, dq::Union{Nothing,Real}=nothing,
                prior::Union{Nothing,SpPrior}=nothing, cst::HBADConst=FIXED[])
  t = Float64(tph)
  (isfinite(t) && t > 0) || error("HBAD plant!: tph must be positive and finite, got $tph")
  age >= 0 || error("HBAD plant!: age must be >= 0, got $age")
  p = prior === nothing ? PRIORS[].sp[global_species(Int(site.eco_id), species)] : prior
  d = clamp(dq === nothing ? cst.estab_dq : Float64(dq), HB_DQ_MIN, HB_DQ_MAX)
  h = 1.3 + chapman_richards(p.hd_a, p.hd_b, p.hd_c, d)
  n = Int(site.live)
  cs = site.c_species; ca = site.c_age
  a = FloatType(age)
  kk = 0
  @inbounds for k in 1:n
    if Int(cs[k]) == Int(species) && ca[k] == a
      kk = k; break
    end
  end
  if kk > 0
    ob = Float64(site.c_bio[kk])
    oba = _PI4E4 * Float64(site.c_dq[kk])^2 * Float64(site.c_tph[kk]); osd = Float64(site.c_tph[kk]) * (Float64(site.c_dq[kk]) / HB_REINEKE_REF)^HB_REINEKE
    merge_cohort!(site.c_tph, site.c_dq, site.c_h, kk, t, d, h)
    nb = agb_of(p, Float64(site.c_tph[kk]), Float64(site.c_dq[kk]))
    site.c_bio[kk] = FloatType(nb)
    site.B += FloatType(nb - ob)
    site.BA += FloatType(_PI4E4 * Float64(site.c_dq[kk])^2 * Float64(site.c_tph[kk]) - oba)
    site.SDI += FloatType(Float64(site.c_tph[kk]) * (Float64(site.c_dq[kk]) / HB_REINEKE_REF)^HB_REINEKE - osd)
    return kk
  end
  n < length(cs) || error("HBAD plant!: site has no free cohort slot (capacity $(length(cs)), live $n); set _new_cohort_counts = live + plant_slot_needed(...) and PanCore.readjust_soa! first")
  k = n + 1
  site.live = k
  age > 1 && (site.old += 1)
  cs[k] = UIntType(species); ca[k] = a
  site.c_tph[k] = FloatType(t); site.c_dq[k] = FloatType(d); site.c_h[k] = FloatType(h); site.c_bal[k] = zero(FloatType)
  b = agb_of(p, t, d)
  site.c_bio[k] = FloatType(b)
  site.B += FloatType(b)
  site.BA += FloatType(_PI4E4 * d * d * t)
  site.SDI += FloatType(t * (d / HB_REINEKE_REF)^HB_REINEKE)
  return k
end
