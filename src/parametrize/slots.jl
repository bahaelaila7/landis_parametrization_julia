# Flat enumeration of every tunable scalar dimension as (param_idx, target_idx), where
# target_idx is nothing (Global), an Int (Species/Eco), or an (eco,sp) tuple (EcoSpecies).
# Built once per ParamSpace from the species count, the eco list and `eco_species_ids`; `length(space.slots)`
# is the search-space dimensionality `d` shared by sobol_samples and the CMA-ES bridge.
_eco_lu(s) = (p = split(String(s), "|lu="); length(p) == 2 ? String(p[2]) : "")  # eco "l3|lu=CELL" → "CELL"

function _build_slots(param_dists::ParamDists, tying::ParamTying, n_species::Int, eco_list::AbstractVector{<:AbstractString}, eco_species_ids::AbstractVector{<:AbstractVector{<:Integer}})::Vector{Tuple{Int,Any}}
  slots = Tuple{Int,Any}[]
  for (pi, param) in enumerate(param_dists.params)
    if param.sampler isa GlobalSampler
      push!(slots, (pi, nothing))
    elseif param.sampler isa SpeciesSampler
      for i in 1:n_species
        push!(slots, (pi, i))
      end
    elseif param.sampler isa EcoSampler
      for i in 1:length(eco_list)
        push!(slots, (pi, i))
      end
    elseif param.sampler isa EcoSpeciesSampler
      pl = get(tying.labels, param.name, nothing)          # param_pooling: one u-dim per pooling label
      ss = get(tying.split_sets, param.name, nothing)   # global-species ids kept per-eco; others shared
      if pl !== nothing
        groups = Dict{String,Vector{Tuple{Int,Int}}}()
        for (eco_id, sp_ids) in enumerate(eco_species_ids), sp_local in eachindex(sp_ids)
          push!(get!(groups, pl[(eco_id, sp_local)], Tuple{Int,Int}[]), (eco_id, sp_local))
        end
        for lbl in sort!(collect(keys(groups)))
          g = groups[lbl]
          push!(slots, (pi, length(g) == 1 ? g[1] : g))
        end
      elseif ss === nothing
        for (eco_id, sp_ids) in enumerate(eco_species_ids)
          for sp_id in eachindex(sp_ids)
            push!(slots, (pi, (eco_id, sp_id)))
          end
        end
      else
        pos = Dict{Int,Vector{Tuple{Int,Int}}}()          # global species → all its (eco, sp_local) positions
        for (eco_id, sp_ids) in enumerate(eco_species_ids), sp_local in eachindex(sp_ids)
          push!(get!(pos, Int(sp_ids[sp_local]), Tuple{Int,Int}[]), (eco_id, sp_local))
        end
        mp = get(tying.tier_merge, param.name, nothing)
        for gsp in sort!(collect(keys(pos)))
          if gsp in ss
            mg = mp === nothing ? nothing : get(mp, gsp, nothing)
            if mg === nothing
              for t in pos[gsp]; push!(slots, (pi, t)); end   # split: one u-dim per (eco, species)
            else
              groups = Dict{String,Vector{Tuple{Int,Int}}}()  # merge the site-cells that share a label
              for t in pos[gsp]
                cell = _eco_lu(eco_list[t[1]])
                push!(get!(groups, get(mg, cell, cell), Tuple{Int,Int}[]), t)
              end
              for lbl in sort!(collect(keys(groups)))
                g = groups[lbl]
                push!(slots, (pi, length(g) == 1 ? g[1] : g))  # merged cells → one broadcast u-dim
              end
            end
          else
            push!(slots, (pi, pos[gsp]))                    # shared: one u-dim broadcast across its ecos
          end
        end
      end
    end
  end
  return slots
end

"The slot layout of `space` re-derived for the lists of `template` (a params struct); equals `space.slots` when the template belongs to the same data."
build_slots(space::ParamSpace{T}, template::T) where {T} =
  _build_slots(space.dists, space.tying, length(template.SPECIES_LIST), template.ECO_LIST, template.ECO_SPECIES_IDS)
"The CMA-ES block-diagonal covariance groups over `slots` (default `space.slots`): a partition of the u-coordinates, see `CmaBlocks`."
build_groups(space::ParamSpace, slots::Vector{Tuple{Int,Any}}=space.slots) = _build_groups(space.dists, slots, space.blocks)

_slot_floor(fl, tgts, dflt, pooled_max::Bool) = pooled_max ? maximum(Float64(get(fl, t, dflt)) for t in tgts) : Float64(get(fl, first(tgts), dflt))

# Map a u-vector in [0,1]^d to a params struct via each prior's quantile, then bounds-clip and
# cast to the param's type — identical to the sobol_samples inner mapping. Bounds and discrete
# priors (DiscreteUniform) are handled automatically by quantile, so any u is representable.
function u_to_params(u::AbstractVector{<:Real}, space::ParamSpace{T}, slots::Vector{Tuple{Int,Any}}, template::T)::T where T
  param_dists = space.dists
  p = template
  bf = space.floors.bmax
  af = space.floors.anpp
  pef = space.floors.prob_estab
  pmax = space.tying.floor_pooled_max
  for (dim, (pi, idx)) in enumerate(slots)
    param = param_dists.params[pi]
    min_, max_ = param.bounds
    tgts = idx isa Vector ? idx : (idx,)                 # >1 target ⇒ shared u-dim broadcast across ecos
    rep = first(tgts)                                    # representative target for floor / B_MAX lookup
    if param.name === :ANPP_MAX_SPP   # ratio reparam: u → ratio ∈ [20,35]; ANPP_MAX_SPP stores derived ANPP = B_MAX/ratio per eco (B_MAX_SPP decoded first). Shared ⇒ one ratio, ANPP re-derived per eco from that eco's B_MAX.
      ratio = Dists.quantile(param.dist, clamp(Float64(u[dim]), 1e-10, 1 - 1e-10))
      f = getproperty(p, :ANPP_MAX_SPP)
      for t in tgts
        bmax = Float64(get_field_val(param.applier, getproperty(p, :B_MAX_SPP), t))
        anpp = bmax / ratio
        af !== nothing && (anpp = max(anpp, Float64(get(af, t, 0.0))))   # data floor: ANPP ≥ p99(agb/age)
        f = apply_mutation(param.applier, f, t, param.type(anpp))
      end
      p = Setfield.@set p.ANPP_MAX_SPP = f
      continue
    end
    lo_eff = min_   # effective lower bound for the clamp (raised to a per-cell data floor for B_MAX / PROB_ESTAB)
    raw = if bf !== nothing && param.name === :B_MAX_SPP
      lo = _slot_floor(bf, tgts, 12000.0, pmax); lo_eff = lo; hi = Float64(max_)   # rescale uniform u onto [floor, upper]
      lo + clamp(Float64(u[dim]), 0.0, 1.0) * (hi - lo)
    elseif pef !== nothing && param.name === :PROB_ESTAB_SPP
      lo = _slot_floor(pef, tgts, Float64(min_), pmax); lo_eff = lo; hi = Float64(max_)   # data lower bound → [floor, 1.0]
      lo + clamp(Float64(u[dim]), 0.0, 1.0) * (hi - lo)
    else
      Dists.quantile(param.dist, clamp(Float64(u[dim]), 1e-10, 1 - 1e-10))
    end
    # clip to [lo_eff, max_]; a data floor above the box upper bound wins (B18: the slot is pinned at its floor for every u —
    # previously u ≤ 0 decoded to the upper bound, BELOW the floor, and every u > 0 to the floor)
    v = raw
    !isnothing(max_) && v > max_ && (v = max_)
    !isnothing(lo_eff) && v < lo_eff && (v = lo_eff)
    val = param.type(v)
    val = _quantize(val, param.quantum, param.type)   # snap to the param's grid (e.g. B_MAX by 100)
    f = getproperty(p, param.name)
    for t in tgts                                     # single target ⇒ one write (identical to before)
      f = apply_mutation(param.applier, f, t, val)
    end
    p = Setfield.@set p.$(param.name) = f
  end
  return p
end

"`u_to_params(u, space, space.slots, template)`."
u_to_params(u::AbstractVector{<:Real}, space::ParamSpace{T}, template::T) where {T} = u_to_params(u, space, space.slots, template)

# Inverse of u_to_params (on the prior-grid): map each tunable value to u = cdf(prior, value).
# Used once to seed the CMA-ES mean from an initial params struct. Clamped off the open
# endpoints so the round-trip through quantile stays inside the bounds.
function params_to_u(params::T, space::ParamSpace{T}, slots::Vector{Tuple{Int,Any}})::Vector{Float64} where T
  param_dists = space.dists
  u = Vector{Float64}(undef, length(slots))
  bf = space.floors.bmax
  pef = space.floors.prob_estab
  pmax = space.tying.floor_pooled_max
  for (dim, (pi, idx)) in enumerate(slots)
    param = param_dists.params[pi]
    rep = idx isa Vector ? idx[1] : idx                 # shared slot ⇒ read the representative eco (all equal)
    tgts = idx isa Vector ? idx : (idx,)
    cur = get_field_val(param.applier, getproperty(params, param.name), rep)
    u[dim] = if param.name === :ANPP_MAX_SPP   # ratio reparam inverse: recover ratio = B_MAX/ANPP, map via the ratio prior. Handles legacy seeds (real ANPP) by clamping their implied ratio into [20,35].
      anpp = Float64(cur); bmax = Float64(get_field_val(param.applier, getproperty(params, :B_MAX_SPP), rep))
      ratio = anpp > 0 ? bmax / anpp : Float64(Dists.mean(param.dist))
      clamp(Float64(Dists.cdf(param.dist, ratio)), 1e-6, 1 - 1e-6)
    elseif bf !== nothing && param.name === :B_MAX_SPP
      lo = _slot_floor(bf, tgts, 12000.0, pmax); hi = Float64(param.bounds[2])   # inverse of the u_to_params rescale
      clamp((Float64(cur) - lo) / (hi - lo), 1e-6, 1 - 1e-6)
    elseif pef !== nothing && param.name === :PROB_ESTAB_SPP
      lo = _slot_floor(pef, tgts, Float64(param.bounds[1]), pmax); hi = Float64(param.bounds[2])   # inverse of the [floor,1] rescale
      clamp((Float64(cur) - lo) / (hi - lo), 1e-6, 1 - 1e-6)
    elseif param.dist isa Dists.DiscreteUnivariateDistribution
      # integer prior: encode to the MIDDLE of level k's mass interval (cdf(k-1), cdf(k)]. cdf(k) itself sits on the quantile
      # boundary, and floating point decoded it to k+1 for some levels (SHADE_TOL 3, MATURITY 7/14/28) — B17.
      k = Float64(cur)
      clamp(0.5 * (Float64(Dists.cdf(param.dist, k - 1)) + Float64(Dists.cdf(param.dist, k))), 1e-6, 1 - 1e-6)
    else
      clamp(Float64(Dists.cdf(param.dist, Float64(cur))), 1e-6, 1 - 1e-6)
    end
  end
  return u
end

"`params_to_u(params, space, space.slots)`."
params_to_u(params::T, space::ParamSpace{T}) where {T} = params_to_u(params, space, space.slots)

# native parameter VALUE per slot — same layout as `params_to_u`, but WITHOUT the prior-CDF transform.
# Used by parameter-space niching so decision distances are measured in native units (and thus in the
# per-parameter `quantum`s), not in the CDF-warped u-space. Reads only slot dims, so it is identical
# regardless of the `template`/`base` used to decode (frozen-growth-safe).
function params_to_nativevec(params::T, space::ParamSpace{T}, slots::Vector{Tuple{Int,Any}})::Vector{Float64} where T
  param_dists = space.dists
  v = Vector{Float64}(undef, length(slots))
  for (dim, (pi, idx)) in enumerate(slots)
    param = param_dists.params[pi]
    rep = idx isa Vector ? idx[1] : idx                 # shared slot ⇒ representative target (all equal)
    v[dim] = Float64(get_field_val(param.applier, getproperty(params, param.name), rep))
  end
  return v
end

# per-slot semantic quantum in NATIVE units: the parameter's `quantum` if set, else its Gaussian `sigma`
# (quantum-first, sigma-fallback). The scale below which a difference is deemed semantically irrelevant.
slot_quanta(space::ParamSpace, slots::Vector{Tuple{Int,Any}}=space.slots) =
  Float64[(p = space.dists.params[pi]; Float64(p.quantum > zero(FloatType) ? p.quantum : p.sigma)) for (pi, _) in slots]

# Hansen-style mixed-integer handling: a per-u-coordinate lower bound on the CMA-ES sampling
# std-dev, so discrete coordinates keep flipping integers even as σ shrinks (otherwise a coarse
# discrete like SHADE_TOL freezes once σ·√C_ii drops below its plateau width and all offspring
# round to the same value). Each DiscreteUniform prior with L levels has a u-space step width of
# 1/L; the floor is `factor/L`. Continuous priors get 0 (no floor). The floor self-targets coarse
# discretes — for fine ones (large L) it is tiny and effectively never binds. Returns a length-d
# vector aligned with `slots` (used to top up the marginal std in CMAES.ask).
function integer_u_min_std(space::ParamSpace, slots::Vector{Tuple{Int,Any}}=space.slots, factor::Float64=0.3)::Vector{Float64}
  s = zeros(Float64, length(slots))
  for (dim, (pi, _)) in enumerate(slots)
    p = space.dists.params[pi]
    if p.dist isa Dists.DiscreteUniform
      span = Float64(Dists.maximum(p.dist) - Dists.minimum(p.dist))   # value-range width
      step = p.quantum > zero(FloatType) ? Float64(p.quantum) : 1.0   # effective grid step
      L = floor(span / step) + 1                                      # number of reachable levels
      L > 1 && (s[dim] = factor / L)                                  # u-width of one step ≈ 1/L
    end
  end
  return s
end

# Generate N quasi-random Sobol samples covering the full parameter space.
# `space` is the run's ParamSpace (see Pan.build_param_space); `initial_params` is the decode template.
function sobol_samples(space::ParamSpace{T}, initial_params::T, N::Int)::Vector{T} where T
  slots = space.slots
  d = length(slots)
  seq = Sobol.SobolSeq(d)
  u = zeros(Float64, d)
  results = Vector{T}(undef, N)
  for n in 1:N
    Sobol.next!(seq, u)
    results[n] = u_to_params(u, space, slots, initial_params)
  end
  return results
end

# Saltelli A/B/ABₖ design for GROUPED (param-type) Sobol sensitivity. Factors = parameter TYPES: one
# group per MutableParam name, whose eco/species/eco-species u-dims move together. Returns the N(K+2)
# param structs in evaluation order plus a matching tag vector ("A", "B", "AB:<name>"), so the existing
# sobol eval loop yields f(A), f(B), f(ABₖ) — enough for first-order Sᵢ and total-effect Sₜᵢ per type.
function saltelli_design(space::ParamSpace{T}, initial_params::T, N::Int) where T
  slots = space.slots
  d = length(slots)
  groups = Tuple{Symbol,Vector{Int}}[]      # (param-name, u-dim indices), first-seen order
  gpos = Dict{Symbol,Int}()
  for (dim, (pi, _)) in enumerate(slots)
    nm = space.dists.params[pi].name
    haskey(gpos, nm) || (push!(groups, (nm, Int[])); gpos[nm] = length(groups))
    push!(groups[gpos[nm]][2], dim)
  end
  seq = Sobol.SobolSeq(d)                    # one d-dim sequence split: A = draws 1:N, B = draws N+1:2N
  A = Matrix{Float64}(undef, N, d); B = Matrix{Float64}(undef, N, d)
  u = zeros(Float64, d)
  for n in 1:N; Sobol.next!(seq, u); @views A[n, :] .= u; end
  for n in 1:N; Sobol.next!(seq, u); @views B[n, :] .= u; end
  samples = Vector{T}(undef, N * (length(groups) + 2))
  tags = Vector{String}(undef, length(samples))
  k = 0
  for n in 1:N; k += 1; samples[k] = u_to_params(@view(A[n, :]), space, slots, initial_params); tags[k] = "A"; end
  for n in 1:N; k += 1; samples[k] = u_to_params(@view(B[n, :]), space, slots, initial_params); tags[k] = "B"; end
  for (nm, cols) in groups
    for n in 1:N
      u2 = Vector{Float64}(A[n, :]); @views u2[cols] .= B[n, cols]   # ABₖ: A with group-k cols from B
      k += 1; samples[k] = u_to_params(u2, space, slots, initial_params); tags[k] = "AB:" * String(nm)
    end
  end
  return samples, tags
end
