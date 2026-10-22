struct HBAD3P <: AbstractPlugin end
const PluginType = HBAD3P

# ---------------------------------------------------------------------------------------------------------------------
# SoA layout
# ---------------------------------------------------------------------------------------------------------------------
PanCore.scalar_arrays(::Type{PluginType}, n::Int) =
  (
    B=Vector{FloatType}(undef, n),          # total REPORTED aboveground biomass (sum of c_bio), g/m2 of the PLOT -- never used by the dynamics
    BA=Vector{FloatType}(undef, n),         # stand basal area, m2 per ha of the PLOT (state x area)
    SDI=Vector{FloatType}(undef, n),        # Reineke stand density index per ha of the PLOT (metric: trees/ha at 25.4 cm)
    RD=Vector{FloatType}(undef, n),         # relative density SDI / sdimax of the stand itself (per forested ha)
    ryear=Vector{Int32}(undef, n),          # recruitment clock: number of recruitment steps this site has made (the reference's run year)
    planted=Vector{Bool}(undef, n),         # planted stand (stand origin): the global planted multipliers pH / pD / pM apply
    no_establish=Vector{Bool}(undef, n),    # true: no recruitment on this site
    W=Vector{Float64}(undef, n),            # period water index AET/PET of the site
    Wn=Vector{Float64}(undef, n),           # plot normal of the water index (W = Wn: no drought anomaly)
    area=Vector{Float64}(undef, n),         # a: forested fraction convention of the site (farea=1: state N is per ha of the forested subplot, plot value = N x a; 1 otherwise)
    rmult=Vector{Float64}(undef, n),        # recruitment area multiplier (farea=1: forested fraction / a; 0 = nonforest subplot, no recruitment)
  )

PanCore.csr_fields(::Type{PluginType}) =
  (
    c_species=:cohort,   # eco-LOCAL species index (BiomassSuccession convention)
    c_age=:cohort,       # A, yr (Float64)
    c_tph=:cohort,       # N, trees per hectare of the forested subplot (Float64)
    c_dq=:cohort,        # D, quadratic mean diameter, cm (Float64)
    c_h=:cohort,         # H, mean height, m (Float64)
    c_zq=:cohort,        # vigour node (0 = mean-field cohort; Float64)
    c_cr=:cohort,        # crown ratio (fraction of the height with live crown; Float64)
    c_bio=:cohort,       # REPORTED allometric AGB g/m2 of the PLOT of (c_tph x area, c_dq) (FloatType)
    c_O=:cohort,         # scratch: overtopping
    c_SDI=:cohort,       # scratch: cohort SDI
    c_V=:cohort,         # scratch: growth vigour min(g, 1)
    c_ropen=:cohort,     # recruit class bookkeeping: run year in which this (recruit) cohort's class opened, 0 = not an open recruit class
    sp_seed=:species,    # per eco species: a conspecific seed tree on the PLOT at the start (static; dispersal between subplots); its length carries the eco species count for the shared loss
    sp_estab=:species,   # scratch: a new recruit class opens this year
    sp_rate=:species,    # scratch: recruit rate of this year (trees/ha)
    sp_oy=:species,      # scratch: opening year of the species' open recruit class (0 = none)
  )

PanCore.csr_arrays(::Type{PluginType}, nnz::NamedTuple) =
  (
    c_species=Vector{UIntType}(undef, nnz.cohort),
    c_age=Vector{Float64}(undef, nnz.cohort),
    c_tph=Vector{Float64}(undef, nnz.cohort),
    c_dq=Vector{Float64}(undef, nnz.cohort),
    c_h=Vector{Float64}(undef, nnz.cohort),
    c_zq=Vector{Float64}(undef, nnz.cohort),
    c_cr=Vector{Float64}(undef, nnz.cohort),
    c_bio=Vector{FloatType}(undef, nnz.cohort),
    c_O=Vector{Float64}(undef, nnz.cohort),
    c_SDI=Vector{Float64}(undef, nnz.cohort),
    c_V=Vector{Float64}(undef, nnz.cohort),
    c_ropen=Vector{Int32}(undef, nnz.cohort),
    sp_seed=Vector{Bool}(undef, nnz.species),
    sp_estab=Vector{Bool}(undef, nnz.species),
    sp_rate=Vector{Float64}(undef, nnz.species),
    sp_oy=Vector{Int32}(undef, nnz.species),
  )

# ---------------------------------------------------------------------------------------------------------------------
# One site, one year: phase 1 = step + recruitment rates (decides how many cohort slots are needed), phase 2 = recruits + reporting
# ---------------------------------------------------------------------------------------------------------------------
@inline _nq(cst::HBAD3PConst) = cst.vigour ? length(VIG_Z) : 1

"""
    site_prepare!(soa, i, ep, cst)

Phase 1 of the annual update of site `i`: the reference `step_site!` on the site's cohorts (Δ = 1), then -- unless the site is `no_establish` -- the recruit rates and class decisions
(`recruit_rates!`). Sets `_new_cohort_counts = live + new slots`. Allocation-free; touches only site `i`.
"""
function site_prepare!(soa::PanCore.AnySoA, i::Int, ep::HBAD3PEcoParams, cst::HBAD3PConst)
  csr = soa.csr; sc = soa.scalar
  o = Int(soa.refs.cohort[i]) - 1
  n = Int(sc.live[i])
  if n == 0 && sc.no_establish[i]
    sc.B[i] = 0; sc.BA[i] = 0; sc.SDI[i] = 0; sc.RD[i] = 0; sc.old[i] = 0
    sc._new_cohort_counts[i] = 0
    return nothing
  end
  n > 0 && step_cohorts!(csr.c_species, csr.c_tph, csr.c_dq, csr.c_h, csr.c_age, csr.c_zq, csr.c_cr, csr.c_O, csr.c_SDI, csr.c_V, ep.sp, ep.G, o + 1, o + n, 1.0, sc.planted[i], sc.W[i], sc.Wn[i])
  nnew = 0
  if !sc.no_establish[i]
    so = Int(soa.refs.species[i]) - 1
    nsp = Int(soa.refs.species[i+1]) - Int(soa.refs.species[i])
    yr = Int(sc.ryear[i]) + 1
    sc.ryear[i] = Int32(yr)
    nnew = recruit_rates!(csr.c_species, csr.c_tph, csr.c_dq, csr.c_ropen, csr.sp_rate, csr.sp_oy, csr.sp_seed, csr.sp_estab, ep.sp, ep.rec, ep.G, sc.rmult[i], 1.0,
                          o + 1, o + n, so, nsp, yr, _nq(cst))
  end
  sc._new_cohort_counts[i] = n + nnew
  return nothing
end

"""
    site_commit!(soa, i, ep, cst)

Phase 2: append / merge this year's recruits into the slots made available by `process_plugin!`, then the REPORTED state (`report_site!`). Errors if the site has no free slot (cannot happen through `process_plugin!`).
"""
function site_commit!(soa::PanCore.AnySoA, i::Int, ep::HBAD3PEcoParams, cst::HBAD3PConst)
  csr = soa.csr; sc = soa.scalar
  o = Int(soa.refs.cohort[i]) - 1
  cap = Int(soa.refs.cohort[i+1]) - Int(soa.refs.cohort[i])
  n = Int(sc.live[i])
  if !sc.no_establish[i]
    so = Int(soa.refs.species[i]) - 1
    nsp = Int(soa.refs.species[i+1]) - Int(soa.refs.species[i])
    nq = _nq(cst)
    need = 0
    @inbounds for k in 1:nsp
      csr.sp_estab[so+k] && (need += nq)
    end
    n + need <= cap || error("HBAD3P site_commit!: site $i has no free cohort slot (capacity $cap, live $n, need $need) - the caller must readjust_soa! first")
    n = recruit_apply_cap!(csr.c_species, csr.c_age, csr.c_tph, csr.c_dq, csr.c_h, csr.c_zq, csr.c_cr, csr.c_ropen, csr.sp_rate, csr.sp_estab, ep.sp, ep.G,
                           o, n, so, nsp, Int(sc.ryear[i]), nq, cap)[1]
  end
  cst.cleanup_tph > 0 && (n = drop_small_cohorts!(csr, o, n, cst.cleanup_tph))
  report_site!(csr, sc, ep, i, o, n)
  return nothing
end

"""
    drop_small_cohorts!(csr, o, n, thr) -> Int

Remove the cohorts `o+1:o+n` of a site with fewer than `thr` trees/ha, shifting the others down (their order is kept). Returns the new live count. Allocation-free. Scratch arrays are recomputed
every step and `c_bio` by `report_site!`, so only the state arrays move.
"""
function drop_small_cohorts!(csr, o::Int, n::Int, thr::Float64)::Int
  m = 0
  @inbounds for k in 1:n
    j = o + k
    csr.c_tph[j] < thr && continue                      # (NaN is kept: it is a bug elsewhere, not negligible)
    m += 1
    if m != k
      jj = o + m
      csr.c_species[jj] = csr.c_species[j]; csr.c_age[jj] = csr.c_age[j]; csr.c_tph[jj] = csr.c_tph[j]; csr.c_dq[jj] = csr.c_dq[j]
      csr.c_h[jj] = csr.c_h[j]; csr.c_zq[jj] = csr.c_zq[j]; csr.c_cr[jj] = csr.c_cr[j]; csr.c_ropen[jj] = csr.c_ropen[j]
    end
  end
  return m
end

"""Reported allometric AGB of every cohort and the site totals (per ha of the PLOT: state x area; RD of the stand itself); sets `live` and `old`."""
function report_site!(csr, sc, ep::HBAD3PEcoParams, i::Int, o::Int, n::Int)
  B = 0.0; ba = 0.0; sdit = 0.0; a = sc.area[i]
  @inbounds for k in 1:n
    j = o + k
    s = Int(csr.c_species[j])
    N = csr.c_tph[j]; D = csr.c_dq[j]
    b = agb_of(ep.allo[s], N * a, D)
    csr.c_bio[j] = FloatType(b)
    B += Float64(csr.c_bio[j])
    ba += _PI4E4 * D * D * N * a
    sdit += sdi(N, D)
  end
  sc.live[i] = n
  sc.old[i] = n
  sc.B[i] = FloatType(B)
  sc.BA[i] = FloatType(ba)
  sc.SDI[i] = FloatType(sdit * a)
  sc.RD[i] = FloatType(sdit / ep.G.sdimax)
  return nothing
end

"""Both phases for one site (tests, single-site tools). Valid only when no new cohort slot is needed (`_new_cohort_counts[i] <= capacity`)."""
function site_year!(soa::PanCore.AnySoA, i::Int, ep::HBAD3PEcoParams, cst::HBAD3PConst)
  site_prepare!(soa, i, ep, cst)
  site_commit!(soa, i, ep, cst)
end

# ---------------------------------------------------------------------------------------------------------------------
# Calibration / parity entry points: whole years + a fractional last step (Sim A), free years with recruitment on a fixed capacity (reference run_free!).
# They run the SAME kernel functions as `process_plugin!` on one site's slice of the SoA arrays; they only skip the reported state. Allocation-free; touch only site `i`.
# ---------------------------------------------------------------------------------------------------------------------
"""One kernel step of length `delta` on the live cohorts of site `i` (reference `step_site!(..., Δ, ...)`), with the site's planted flag and water indices."""
@inline function step_site_frac!(soa::PanCore.AnySoA, i::Int, ep::HBAD3PEcoParams, delta::Float64)
  csr = soa.csr; sc = soa.scalar
  o = Int(soa.refs.cohort[i]) - 1
  n = Int(sc.live[i])
  n > 0 && step_cohorts!(csr.c_species, csr.c_tph, csr.c_dq, csr.c_h, csr.c_age, csr.c_zq, csr.c_cr, csr.c_O, csr.c_SDI, csr.c_V, ep.sp, ep.G, o + 1, o + n, delta, sc.planted[i], sc.W[i], sc.Wn[i])
  return nothing
end

"""Integrate site `i` for `nyr` whole years and then one fractional step of length `frac` (if > 0), no recruitment (reference `simulate!`)."""
function run_interval!(soa::PanCore.AnySoA, i::Int, ep::HBAD3PEcoParams, nyr::Int, frac::Float64)
  for _ in 1:nyr
    step_site_frac!(soa, i, ep, 1.0)
  end
  frac > 0 && step_site_frac!(soa, i, ep, frac)
  return nothing
end

"""
    site_year_free!(soa, i, ep, nq, delta=1.0) -> Int

One annual step of a free run of site `i` on its fixed capacity (reference `run_free!` inner step): the kernel step of length `delta`, then recruitment (`nq` vigour sub-cohorts per class, or 1; rates x `delta`).
Advances the site's recruitment clock `ryear`, sets `live`. Returns the number of recruit classes dropped for lack of a free slot (0 when the capacity suffices).
"""
function site_year_free!(soa::PanCore.AnySoA, i::Int, ep::HBAD3PEcoParams, nq::Int, delta::Float64=1.0)::Int
  csr = soa.csr; sc = soa.scalar
  o = Int(soa.refs.cohort[i]) - 1
  cap = Int(soa.refs.cohort[i+1]) - Int(soa.refs.cohort[i])
  n = Int(sc.live[i])
  n > 0 && step_cohorts!(csr.c_species, csr.c_tph, csr.c_dq, csr.c_h, csr.c_age, csr.c_zq, csr.c_cr, csr.c_O, csr.c_SDI, csr.c_V, ep.sp, ep.G, o + 1, o + n, delta, sc.planted[i], sc.W[i], sc.Wn[i])
  so = Int(soa.refs.species[i]) - 1
  nsp = Int(soa.refs.species[i+1]) - Int(soa.refs.species[i])
  yr = Int(sc.ryear[i]) + 1
  sc.ryear[i] = Int32(yr)
  recruit_rates!(csr.c_species, csr.c_tph, csr.c_dq, csr.c_ropen, csr.sp_rate, csr.sp_oy, csr.sp_seed, csr.sp_estab, ep.sp, ep.rec, ep.G, sc.rmult[i], delta, o + 1, o + n, so, nsp, yr, nq)
  n, dropped = recruit_apply_cap!(csr.c_species, csr.c_age, csr.c_tph, csr.c_dq, csr.c_h, csr.c_zq, csr.c_cr, csr.c_ropen, csr.sp_rate, csr.sp_estab, ep.sp, ep.G, o, n, so, nsp, yr, nq, cap)
  sc.live[i] = n
  return dropped
end

function PanCore.process_plugin!(soa::PanCore.AnySoA, ::Type{PluginType}, current_time::Int; ctx::NamedTuple)
  eco = ctx.eco; cst = ctx.cst
  PanCore.@maybe_threads soa.parallel for i in 1:soa.n
    @inbounds begin
      if soa.scalar.active[i]
        site_prepare!(soa, i, eco[soa.scalar.eco_id[i]], cst)
      else
        soa.scalar._new_cohort_counts[i] = soa.scalar.live[i]
      end
    end
  end
  # resize the CSR only when a recruit class opened (keeps the steady state allocation-free)
  refs = soa.refs.cohort; cnt = soa.scalar._new_cohort_counts
  resize = false
  @inbounds for i in 1:soa.n
    if max(Int(cnt[i]), 1) != Int(refs[i+1]) - Int(refs[i])
      resize = true
      break
    end
  end
  resize && PanCore.readjust_soa!(soa, (cohort=cnt,))
  PanCore.@maybe_threads soa.parallel for i in 1:soa.n
    @inbounds if soa.scalar.active[i]
      site_commit!(soa, i, eco[soa.scalar.eco_id[i]], cst)
    end
  end
  return nothing
end

# ---------------------------------------------------------------------------------------------------------------------
# Cohort add / external biomass writes / planting / management
# ---------------------------------------------------------------------------------------------------------------------
@inline _prior_of(ctx::HBAD3PContext, eco::Integer, esp::Integer) = @inbounds ctx.priors.sp[HBADPlugin.global_species(ctx.eco_gids, eco, esp)]
@inline _allo_of(ctx::HBAD3PContext, eco::Integer, esp::Integer) = (p = _prior_of(ctx, eco, esp); Allo(p.mq_a, p.mq_b))
@inline _spp_of(ctx::HBAD3PContext, eco::Integer, esp::Integer) = (g = HBADPlugin.global_species(ctx.eco_gids, eco, esp); ctx.sp[findfirst(==(ctx.species_key[g]), ctx.keys)])

function _set_cohort!(site, k::Int, species::Integer, age::Real, tph::Real, dq::Real, h::Real, zq::Real, cr::Real, bio::Real)
  site.c_species[k] = UIntType(species)
  site.c_age[k] = Float64(age); site.c_tph[k] = Float64(tph); site.c_dq[k] = Float64(dq); site.c_h[k] = Float64(h); site.c_zq[k] = Float64(zq); site.c_cr[k] = Float64(cr)
  site.c_bio[k] = FloatType(bio)
  site.c_O[k] = 0.0; site.c_SDI[k] = 0.0; site.c_V[k] = 0.0; site.c_ropen[k] = Int32(0)
  return nothing
end

"""
    add_cohort!(ctx, site, species_local, age, agb)

Append a cohort with OBSERVED aboveground biomass `agb` (g/m2 of the plot) as ONE mean-field cohort (zq = 0, CR 0.4): (tph, dq, h) from the HBAD v1 empirical translation (`HBADPlugin.translate`; tph is
per ha of the plot, taken as the state density: the site's `area` convention does not apply to this biomass-driven path), `c_bio = agb`. Tree-based initial states come from `Data.load_tree_cohorts3p`.
"""
function add_cohort!(ctx::HBAD3PContext, site::SiteView, species::Integer, age::Real, agb::Real)
  p = _prior_of(ctx, Int(site.eco_id), species)
  site.live += 1
  age > 1 && (site.old += 1)
  k = Int(site.live)
  tph, dq, h = HBADPlugin.translate(p, Float64(age), Float64(agb))
  _set_cohort!(site, k, species, age, tph, dq, h, 0.0, 0.4, agb)
  return nothing
end

"""Set the reported AGB of cohort k: trees are removed / added at the same size (`c_tph` scaled by newbio/oldbio); a cohort without previous biomass is re-translated."""
function set_cohort_bio!(ctx::HBAD3PContext, site::SiteView, k::Integer, newbio::Real)
  old = site.c_bio[k]
  nb = max(Float64(newbio), 0.0)
  if old > 0 && site.c_tph[k] > 0
    site.c_tph[k] = site.c_tph[k] * nb / Float64(old)
  elseif nb > 0
    p = _prior_of(ctx, Int(site.eco_id), site.c_species[k])
    tph, dq, h = HBADPlugin.translate(p, Float64(site.c_age[k]), nb)
    site.c_tph[k] = tph; site.c_dq[k] = dq; site.c_h[k] = h
  else
    site.c_tph[k] = 0.0
  end
  site.c_bio[k] = FloatType(nb)
  return nothing
end

"""Cohort slots `plant!` may need per planting (the scenario harness reserves this many): 5 vigour sub-cohorts, or 1."""
plant_slots_reserved(cst::HBAD3PConst) = _nq(cst)

"""
    plant_slot_needed(ctx, site, species_local, age) -> Int

Slots `plant!` appends for trees of this species and age: 0 when the site already holds its planted sub-cohorts (age equal, vigour sub-cohorts present) and the trees merge, else the number of vigour sub-cohorts (5, or 1).
"""
function plant_slot_needed(ctx::HBAD3PContext, site::SiteView, species::Integer, age::Real=1)::Int
  nq = _nq(ctx.fixed); need = 0
  n = Int(site.live)
  cs = site.c_species; ca = site.c_age; cz = site.c_zq
  @inbounds for q in 1:nq
    zqq = nq == 1 ? 0.0 : VIG_Z[q]
    found = false
    for k in 1:n
      if Int(cs[k]) == Int(species) && ca[k] == Float64(age) && cz[k] == zqq && site.c_ropen[k] == 0
        found = true; break
      end
    end
    found || (need += 1)
  end
  return need
end

"""
    plant!(ctx, site, species_local, tph; age=1, dq=2.54, h=NaN) -> Int

Plant `tph` trees per ha of the forested stand (the state density) of eco-local species `species_local` (age `age`, D = `dq` 2.54 cm, H = `h`, default the recruit height H*(dq) of the species, CR = crhi) as the
vigour sub-cohorts (Gauss-Hermite nodes and shares; one mean-field cohort when the context has `vigour=false`). Trees merge into an existing planted sub-cohort of the same species, age and vigour (densities add,
D = quadratic mean, H, CR = N-weighted mean), else a cohort is appended. Reported totals B, BA, SDI are updated incrementally (x area; RD at the next annual step). CSR capacity is the CALLER's job (`plant_sites!`
reserves `plant_slots_reserved` slots per planting). Returns the index (1-based within the site) of the first sub-cohort.
"""
function plant!(ctx::HBAD3PContext, site::SiteView, species::Integer, tph::Real; age::Real=1, dq::Real=D_REC, h::Real=NaN)
  t = Float64(tph)
  (isfinite(t) && t > 0) || error("HBAD3P plant!: tph must be positive and finite, got $tph")
  age >= 0 || error("HBAD3P plant!: age must be >= 0, got $age")
  al = _allo_of(ctx, Int(site.eco_id), species)
  spp = _spp_of(ctx, Int(site.eco_id), species)
  hh = isnan(Float64(h)) ? hstar(spp, ctx.glob, Float64(dq)) : Float64(h)
  a = Float64(site.area)
  nq = _nq(ctx.fixed); first_ = 0
  for q in 1:nq
    zqq = nq == 1 ? 0.0 : VIG_Z[q]; add = t * (nq == 1 ? 1.0 : VIG_W[q])
    n = Int(site.live); kk = 0
    cs = site.c_species; ca = site.c_age; cz = site.c_zq
    @inbounds for k in 1:n
      if Int(cs[k]) == Int(species) && ca[k] == Float64(age) && cz[k] == zqq && site.c_ropen[k] == 0
        kk = k; break
      end
    end
    if kk > 0
      ob = Float64(site.c_bio[kk])
      m0 = site.c_tph[kk]; m = m0 + add
      site.c_dq[kk] = sqrt((m0 * site.c_dq[kk]^2 + add * Float64(dq)^2) / m)
      site.c_h[kk] = (m0 * site.c_h[kk] + add * hh) / m
      site.c_cr[kk] = (m0 * site.c_cr[kk] + add * ctx.glob.crhi) / m
      site.c_tph[kk] = m
      nb = agb_of(al, m * a, site.c_dq[kk]); site.c_bio[kk] = FloatType(nb)
      site.B += FloatType(nb - ob)
    else
      n < length(cs) || error("HBAD3P plant!: site has no free cohort slot (capacity $(length(cs)), live $n); set _new_cohort_counts = live + plant_slots_reserved and PanCore.readjust_soa! first (plant_sites! does)")
      kk = n + 1
      site.live = kk
      age > 1 && (site.old += 1)
      b = agb_of(al, add * a, Float64(dq))
      _set_cohort!(site, kk, species, age, add, dq, hh, zqq, ctx.glob.crhi, b)
      site.B += FloatType(b)
    end
    site.BA += FloatType(_PI4E4 * Float64(dq)^2 * add * a)
    site.SDI += FloatType(sdi(add, Float64(dq)) * a)
    first_ == 0 && (first_ = kk)
  end
  return first_
end

"""
    scale_cohort!(ctx, site, k, keep)

Management removal: keep the fraction `keep` (0..1) of the trees of cohort k at constant size (`c_tph` x keep; dq, h, cr, age unchanged) and update its reported AGB.
Site totals are refreshed by `refresh_totals!` (call it once after a series of removals).
"""
function scale_cohort!(ctx::HBAD3PContext, site::SiteView, k::Integer, keep::Real)
  al = _allo_of(ctx, Int(site.eco_id), site.c_species[k])
  t = site.c_tph[k] * Float64(keep)
  site.c_tph[k] = t
  site.c_bio[k] = FloatType(agb_of(al, t * Float64(site.area), site.c_dq[k]))
  return nothing
end

"""Recompute the REPORTED site totals B, BA, SDI (per ha of the plot) from the cohorts after management (RD is recomputed by the next annual step)."""
function refresh_totals!(ctx::HBAD3PContext, site::SiteView)
  B = 0.0; ba = 0.0; sd = 0.0; a = Float64(site.area)
  @inbounds for k in 1:Int(site.live)
    B += Float64(site.c_bio[k]); D = site.c_dq[k]; N = site.c_tph[k]
    ba += _PI4E4 * D * D * N * a; sd += sdi(N, D) * a
  end
  site.B = FloatType(B); site.BA = FloatType(ba); site.SDI = FloatType(sd)
  return nothing
end
