# search_mode: gdigel — Group-Decomposed Igel (1+1)-CMA-ES driver (algorithm: src/search/GDIgel.jl; docs/cmaes_gdigel.md).
# Works for every succession plugin (dispatch via succession_plugin_of / make_param_dists / generate_params).
#
# Objective (per species GROUP g, TOST-aligned — the quantity the equivalence tests evaluate):
#   pairs = Σ simulated vs Σ observed AGB per (plot, year, eco-species) over matched cohorts (pre-injection simulation,
#           observed real records whose interval started from a real record when real_only; exclude_overwrite drops
#           disturbance-overwritten records) — identical to tools/tost_common.jl (Sim A);
#   d     = log1p(sim) − log1p(obs);  m_g = mean d over the group's pairs;  m_{g,c} = mean d per observed-size class c;
#   f_g   = m_g² + size_weight · mean_c m_{g,c}²   (classes < 100, 100–1,000, 1,000–5,000, ≥ 5,000 g/m² with ≥ min_pairs).
# Groups = the pooling slots of the anchor EcoSpecies parameter (first EcoSpecies param, or gdigel.group_param), i.e. a
# species, or species·site-cell where the config splits it — the same groups the TOST tables report.

struct GDGroupIndex
  key2rec::Dict{NTuple{4,Int},Int}     # (plot, sim_year, eco_species, age) → record
  rec_pair::Vector{Int}                # record → pair (plot, sim_year, eco_species)
  rec_obs::Vector{Float64}
  pair_group::Vector{Int}
  pair_class::Vector{Int}              # filled per evaluation (size class by matched observed Σ)
  ngroups::Int
  labels::Vector{String}
  cell_group::Dict{Tuple{Int,Int},Int} # (eco, eco_species) → group
end

const GD_SIZE_EDGES = (100.0, 1000.0, 5000.0)
@inline _gd_class(o::Float64) = o < GD_SIZE_EDGES[1] ? 1 : o < GD_SIZE_EDGES[2] ? 2 : o < GD_SIZE_EDGES[3] ? 3 : 4

# cells (eco, eco-local species) touched by a slot target
function _gd_slot_cells(sampler, target, eco_species_ids)
  cells = Tuple{Int,Int}[]
  if sampler isa PU.GlobalSampler
    for (e, ids) in enumerate(eco_species_ids), j in eachindex(ids); push!(cells, (e, j)); end
  elseif sampler isa PU.SpeciesSampler
    for (e, ids) in enumerate(eco_species_ids), j in eachindex(ids); Int(ids[j]) == Int(target) && push!(cells, (e, j)); end
  elseif sampler isa PU.EcoSampler
    for j in eachindex(eco_species_ids[Int(target)]); push!(cells, (Int(target), j)); end
  elseif target isa Tuple
    push!(cells, (Int(target[1]), Int(target[2])))
  else
    for t in target; push!(cells, (Int(t[1]), Int(t[2]))); end
  end
  cells
end

function _gd_groups(param_dists, slots, eco_species_ids, species_list, eco_list; group_param="")
  anchor = isempty(group_param) ? findfirst(p -> p.sampler isa PU.EcoSpeciesSampler, param_dists.params) :
           findfirst(p -> String(p.name) == group_param, param_dists.params)
  anchor === nothing && error("gdigel: no EcoSpecies parameter to define groups")
  cell_group = Dict{Tuple{Int,Int},Int}(); labels = String[]
  aslots = [(pi, t) for (pi, t) in slots if pi == anchor]
  acells = [_gd_slot_cells(param_dists.params[pi].sampler, t, eco_species_ids) for (pi, t) in aslots]
  asp = [Int(eco_species_ids[c[1][1]][c[1][2]]) for c in acells]
  nper = Dict(s => count(==(s), asp) for s in unique(asp))
  for (k, cells) in enumerate(acells)
    sp = species_list[asp[k]]
    cellnames = sort(unique(PU._eco_lu(eco_list[c[1]]) for c in cells))
    l3s = sort(unique(String(first(split(eco_list[c[1]], "|lu="))) for c in cells))
    push!(labels, nper[asp[k]] > 1 ? "$(sp)·$(join(cellnames, ""))" * (length(l3s) == 1 ? "@$(l3s[1])" : "") : sp)   # split species: site cells (+ L3 when per-L3)
    for c in cells; cell_group[c] = length(labels); end
  end
  # cells the anchor never covered (should not happen) get their own species group
  for (e, ids) in enumerate(eco_species_ids), j in eachindex(ids)
    haskey(cell_group, (e, j)) && continue
    push!(labels, species_list[Int(ids[j])] * "·" * PU._eco_lu(eco_list[e])); cell_group[(e, j)] = length(labels)
  end
  cell_group, labels
end

function _gd_index(injection_cohorts, splots, cell_group, labels; real_only::Bool=true)
  inj = injection_cohorts
  ph = hasproperty(inj, :phantom_frac) ? Float64.(inj.phantom_frac) : zeros(nrow(inj))
  dr = hasproperty(inj, :disturbance_drop_pct) ? Float64.(coalesce.(inj.disturbance_drop_pct, 0.0)) : zeros(nrow(inj))
  pid = Int.(inj.plot_id); sy = Int.(inj.sim_year); esp = Int.(inj.eco_species_id); age = Int.(inj.age_calc); ob = Float64.(inj.agb_sum)
  phmap = Dict{NTuple{4,Int},Float64}(); obs = Dict{NTuple{4,Int},Float64}(); drop = Dict{NTuple{4,Int},Float64}()
  years = Dict{Int,Vector{Int}}()
  for i in eachindex(pid)
    k = (pid[i], sy[i], esp[i], age[i])
    phmap[k] = max(get(phmap, k, 0.0), ph[i])
    push!(get!(years, pid[i], Int[]), sy[i])
    sy[i] > 0 || continue
    obs[k] = get(obs, k, 0.0) + ob[i]; drop[k] = max(get(drop, k, 0.0), dr[i])
  end
  prevyr = Dict{Tuple{Int,Int},Int}()
  for (p, ys) in years; u = sort(unique(ys)); for k in 2:length(u); prevyr[(p, u[k])] = u[k-1]; end; end
  plot2eco = Dict(Int(r.plot_id) => Int(r.eco_id) for r in eachrow(unique(select(splots, [:plot_id, :eco_id]))))
  excl = OVERRIDE_INJECTION_DISTURBANCE[] == :exclude_overwrite
  key2rec = Dict{NTuple{4,Int},Int}(); rec_pair = Int[]; rec_obs = Float64[]; pairid = Dict{NTuple{3,Int},Int}(); pair_group = Int[]
  for (k, o) in obs
    (excl && drop[k] > 0) && continue
    if real_only
      phmap[k] >= 0.5 && continue
      py = get(prevyr, (k[1], k[2]), nothing)
      py !== nothing && get(phmap, (k[1], py, k[3], k[4] - (k[2] - py)), 0.0) >= 0.5 && continue
    end
    e = get(plot2eco, k[1], 0); e == 0 && continue
    g = get(cell_group, (e, k[3]), 0); g == 0 && continue
    pk = (k[1], k[2], k[3])
    pi = get!(pairid, pk) do; push!(pair_group, g); length(pair_group); end
    push!(rec_pair, pi); push!(rec_obs, o); key2rec[k] = length(rec_obs)
  end
  GDGroupIndex(key2rec, rec_pair, rec_obs, pair_group, zeros(Int, length(pair_group)), length(labels), labels, cell_group)
end

"""Per-group losses + bias summaries from a pre-injection cached simulation (fit_params(...; cache_preinject=true))."""
function _gd_losses(ix::GDGroupIndex, cached; size_weight::Float64=0.5, min_pairs::Int=20)
  np = length(ix.pair_group); ps = zeros(np); po = zeros(np); seen = falses(length(ix.rec_obs))
  for (pid, sy, esp, age, bio) in cached
    sy > 0 || continue
    r = get(ix.key2rec, (Int(pid), Int(sy), Int(esp), Int(age)), 0); r == 0 && continue
    p = ix.rec_pair[r]; ps[p] += Float64(bio)
    seen[r] || (po[p] += ix.rec_obs[r]; seen[r] = true)
  end
  G = ix.ngroups
  s = zeros(G); s2 = zeros(G); n = zeros(Int, G); sc = zeros(G, 4); nc = zeros(Int, G, 4)
  for p in 1:np
    po[p] > 0 || continue
    d = log1p(ps[p]) - log1p(po[p]); g = ix.pair_group[p]; c = _gd_class(po[p])
    s[g] += d; s2[g] += d * d; n[g] += 1; sc[g, c] += d; nc[g, c] += 1
  end
  f = fill(NaN, G); m = fill(NaN, G); se = fill(NaN, G)
  for g in 1:G
    n[g] >= 3 || continue
    m[g] = s[g] / n[g]; se[g] = sqrt(max(s2[g] / n[g] - m[g]^2, 0.0) * n[g] / (n[g] - 1) / n[g])
    cl = 0.0; k = 0
    for c in 1:4; nc[g, c] >= min_pairs || continue; cl += (sc[g, c] / nc[g, c])^2; k += 1; end
    f[g] = m[g]^2 + (k > 0 ? size_weight * cl / k : 0.0)
  end
  (f=f, m=m, se=se, n=n)
end

# approximate TOST (normal CI, α = 0.05) at margin M: equivalent iff m ± 1.645·se inside ±log1p(M)
_gd_pass(m, se, M) = count(g -> !isnan(m[g]) && abs(m[g]) + 1.645 * se[g] < log1p(M), eachindex(m))

function parametrize_GDIgel(; ref_soa::SiteSoA, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years,
    species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, spinup::Bool,
    TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=3, resume_from::Union{Nothing,String}=nothing,
    start_from::Union{Nothing,String}=nothing, n_reps::Int=1, no_establishment::Bool=false, injection_cohorts=nothing,
    gd_opts::AbstractDict=Dict{String,Any}(), kw...)
  o(k, d) = get(gd_opts, k, d)
  mu = Int(o("mu", 8)); σ0 = Float64(o("sigma0", 0.15)); σstop = Float64(o("sigma_stop", 0.004)); σmin = Float64(o("sigma_min", 0.002))
  patience = Int(o("patience", 40)); sw = Float64(o("size_weight", 0.5)); real_only = Bool(o("real_only", true))
  include_start = Bool(o("include_start", false)); ckpt_every = Int(o("ckpt_every", 10)); min_pairs = Int(o("min_pairs", 20))
  arch_frac = Float64(o("archive_min_pass_frac", 0.8))
  DUAL_MODE[] == :off || error("gdigel: Sim A only (dual_mode must be off)")
  isnothing(injection_cohorts) && error("gdigel needs override injection cohorts (Sim A)")
  mkpath(output_dir)
  SP = succession_plugin_of(ref_soa)
  pd = make_param_dists(SP, length(species_list), length(eco_list), eco_species_ids; no_establishment=no_establishment, fit_establishment=false)
  template = !isnothing(start_from) ? _load_params_from_path(start_from) : generate_params(SP, species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
  slots = PU.build_slots(pd, template); nd = length(slots)
  cell_group, labels = _gd_groups(pd, slots, eco_species_ids, species_list, eco_list; group_param=String(o("group_param", "")))
  ix = _gd_index(injection_cohorts, splots, cell_group, labels; real_only=real_only)
  # blocks: u-coordinates grouped by the set of groups they affect
  bmap = Dict{Vector{Int},Vector{Int}}()
  for (i, (pi, t)) in enumerate(slots)
    resp = sort!(unique(cell_group[c] for c in _gd_slot_cells(pd.params[pi].sampler, t, eco_species_ids)))
    push!(get!(bmap, resp, Int[]), i)
  end
  bdefs = [GDIgel.BlockDef(idx, resp) for (resp, idx) in sort!(collect(bmap); by=kv -> first(kv[2]))]
  if Bool(o("single_block", false))   # ablation: no decomposition — one (1+1)-CMA over all u-dims judged by the TOTAL loss
    bdefs = [GDIgel.BlockDef(collect(1:nd), collect(1:ix.ngroups))]
  end
  @info "gdigel: $nd u-dims → $(length(bdefs)) blocks (sizes $(sort(length.(getfield.(bdefs, :idx)))) ); $(ix.ngroups) groups; $(length(ix.pair_group)) pairs ($(real_only ? "real-only" : "all"))"
  injection_dict = _build_injection_dict(injection_cohorts, ref_soa); injection_years = Set(Int.(injection_cohorts.sim_year))
  max_sim_year = site_sim_years.sim_years .|> maximum |> maximum
  n_species = length(species_list)
  fixed_seeds = _fixed_seeds(rng, n_reps)
  ws_t = [deepcopy(ref_soa) for _ in 1:Threads.maxthreadid()]
  dec(u) = PU.u_to_params(clamp.(u, 0.0, 1.0), pd, slots, template)
  function ev(u; ws=ws_t[Threads.threadid()])
    r = fit_params(ref_soa, dec(u), max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params;
      debug=false, search_tier=search_tier, seeds=fixed_seeds, injection_dict=injection_dict, injection_years=injection_years, work_soa=ws, cache_preinject=true)
    _, _, k = _agg_reps(r)
    _gd_losses(ix, r[k][2]; size_weight=sw, min_pairs=min_pairs)
  end
  # evaluate a batch: candidate-parallel (each eval sites-serial) when PAN_PARALLEL=candidate, else sequential sites-parallel
  function evbatch(us)
    out = Vector{Any}(undef, length(us))
    if PARALLEL_MODE[] == :candidate
      prev = PARALLEL_SITES[]; PARALLEL_SITES[] = false
      try
        Threads.@threads :dynamic for k in eachindex(us); out[k] = ev(us[k]; ws=ws_t[Threads.threadid()]); end
      finally; PARALLEL_SITES[] = prev; end
    else
      for k in eachindex(us); out[k] = ev(us[k]; ws=ws_t[1]); end
    end
    out
  end
  archive = NamedTuple[]; cap = Int(o("archive_cap", 2000))
  amin = Ref(typemax(Int))   # set from the first evaluation: archive_min_pass_frac × groups with data
  function offer!(u, r, gen, lid)
    amin[] == typemax(Int) && (amin[] = ceil(Int, arch_frac * count(!isnan, r.m)))
    n10 = _gd_pass(r.m, r.se, 0.1)
    n10 >= amin[] || return
    push!(archive, (gen=gen, lineage=lid, u=copy(u), f=copy(r.f), m=copy(r.m), se=copy(r.se), n05=_gd_pass(r.m, r.se, 0.05), n10=n10, n20=_gd_pass(r.m, r.se, 0.2)))
    if length(archive) > cap   # keep the archive bounded: drop the worst-total entry
      deleteat!(archive, argmax([GDIgel._tot(a.f) for a in archive]))
    end
  end
  if resume_from !== nothing
    saved = JLD2.load_object(resume_from); st = saved.st; append!(archive, saved.archive)
    @info "gdigel: resumed $(basename(resume_from)) at gen $(st.gen), $(st.n_evals) evals, $(length(st.solutions)) solutions"
  else
    mode = Symbol(String(o("mode", "optimize")))
    # :explore (anchor-and-explore, docs/cmaes_gdigel.md §4): behavioural threshold tau from the tolerated bias
    u_anchor = start_from === nothing ? nothing : PU.params_to_u(template, pd, slots)
    mode === :explore && u_anchor === nothing && error("gdigel mode explore needs start_from (the anchor parameter set)")
    # per-group behavioural threshold: tau_g = max(log1p(explore_bias)², explore_slack × f_g(anchor)) — the anchor is
    # behavioural by construction, and groups the anchor already fits well get the absolute tolerance
    tau = Float64[]
    if mode === :explore
      fa = only(evbatch([u_anchor])).f; t0 = log1p(Float64(o("explore_bias", 0.07)))^2; sl = Float64(o("explore_slack", 1.5))
      tau = [isnan(v) ? Inf : max(t0, sl * v) for v in fa]
      @info "gdigel explore: anchor total $(round(GDIgel._tot(fa), sigdigits=4)); tau_abs $(round(t0, sigdigits=3)), slack $sl; groups at the absolute tolerance: $(count(==(t0), tau))/$(count(isfinite, tau))"
    end
    st = GDIgel.GDState(bdefs, nd; sigma0=σ0, sigma_stop=σstop, sigma_min=σmin, patience=patience, rel_tol=Float64(o("rel_tol", 1e-3)), seed=Int(o("seed", 1)),
                        mode=mode, tau=tau, anchors=(mode === :explore ? [u_anchor] : Vector{Float64}[]),
                        restart_sigma=Float64(o("explore_start_sigma", 0.02)))
    starts = Vector{Vector{Float64}}()
    if mode === :explore   # every lineage starts at the anchor, slightly perturbed
      while length(starts) < mu; push!(starts, clamp.(u_anchor .+ Float64(o("explore_start_sigma", 0.02)) .* randn(st.rng, nd), 0.0, 1.0)); end
    else
      include_start && start_from !== nothing && push!(starts, u_anchor)
    end
    while length(starts) < mu; push!(starts, GDIgel.next_start!(st)); end
    r0 = evbatch(starts)
    for k in 1:mu; GDIgel.add_lineage!(st, starts[k], r0[k].f); offer!(starts[k], r0[k], 0, k); end
    st.n_evals = mu
  end
  logf = joinpath(output_dir, "gd_log.csv")
  isfile(logf) || open(io -> println(io, "gen,n_evals,lineage,born,total,n05,n10,n20,median_abs_bias_pct,mean_sigma,min_sigma,max_sigma"), logf, "w")
  last_r = Dict{Int,Any}()
  t_start = time()
  while st.n_evals < TRIALS
    ys = GDIgel.ask(st)
    rys = evbatch(ys)
    xps = GDIgel.assemble!(st, ys, [r.f for r in rys])
    need = findall(!isnothing, xps)
    rxp = isempty(need) ? Any[] : evbatch([xps[k] for k in need])
    fxps = Vector{Any}(nothing, length(st.lineages)); rmap = Dict{Int,Any}()
    for (j, k) in enumerate(need); fxps[k] = rxp[j].f; rmap[k] = rxp[j]; end
    for (k, L) in enumerate(st.lineages); offer!(ys[k], rys[k], st.gen + 1, L.id); haskey(rmap, k) && offer!(xps[k], rmap[k], st.gen + 1, L.id); end
    # which evaluated result is the new parent (for logging): resolved after finish!
    cand = [(ys[k], rys[k], haskey(rmap, k) ? (xps[k], rmap[k]) : nothing) for k in eachindex(ys)]
    restart_eval(x) = (r = only(evbatch([x])); offer!(x, r, st.gen, -1); r.f)
    GDIgel.finish!(st, fxps, restart_eval)
    open(logf, "a") do io
      for (k, L) in enumerate(st.lineages)
        r = nothing
        c = k <= length(cand) ? cand[k] : nothing
        if c !== nothing
          L.x == c[1] && (r = c[2]); c[3] !== nothing && L.x == c[3][1] && (r = c[3][2])
        end
        r === nothing && (r = get(last_r, L.id, nothing))
        r === nothing || (last_r[L.id] = r)
        sig = [B.sigma for B in L.bs]
        if r === nothing
          println(io, "$(st.gen),$(st.n_evals),$(L.id),$(L.born),$(L.tot),,,,,$(sum(sig)/length(sig)),$(minimum(sig)),$(maximum(sig))")
        else
          mab = Statistics.median(filter(!isnan, abs.(expm1.(r.m)))) * 100
          println(io, "$(st.gen),$(st.n_evals),$(L.id),$(L.born),$(L.tot),$(_gd_pass(r.m, r.se, 0.05)),$(_gd_pass(r.m, r.se, 0.1)),$(_gd_pass(r.m, r.se, 0.2)),$(round(mab, digits=2)),$(sum(sig)/length(sig)),$(minimum(sig)),$(maximum(sig))")
        end
      end
    end
    best = minimum(L.tot for L in st.lineages)
    @info "gdigel gen $(st.gen): evals $(st.n_evals), best total $(round(best, sigdigits=4)), solutions $(length(st.solutions)), archive $(length(archive)), $(round((time() - t_start) / st.gen, digits=1)) s/gen"
    if st.gen % ckpt_every == 0
      JLD2.save_object(joinpath(output_dir, "gd_state@$(st.gen).jld2"), (st=st, archive=archive, labels=labels, slots=slots, template=template))
    end
  end
  JLD2.save_object(joinpath(output_dir, "gd_state@$(st.gen).jld2"), (st=st, archive=archive, labels=labels, slots=slots, template=template))
  # final solutions: converged lineages + the current lineages (as unconverged end points)
  sols = vcat([(kind="converged", lineage=s.lineage, gen=s.gen, x=s.x) for s in st.solutions],
              [(kind="final", lineage=L.id, gen=st.gen, x=L.x) for L in st.lineages])
  sdir = joinpath(output_dir, "solutions"); mkpath(sdir)
  rows = DataFrame(file=String[], kind=String[], lineage=Int[], gen=Int[], total=Float64[], n05=Int[], n10=Int[], n20=Int[], median_abs_bias_pct=Float64[])
  rs = evbatch([s.x for s in sols])
  for (i, s) in enumerate(sols)
    fn = "sol_$(lpad(i, 3, '0'))_$(s.kind)_L$(s.lineage).jld2"; JLD2.save_object(joinpath(sdir, fn), dec(s.x)); r = rs[i]
    push!(rows, (fn, s.kind, s.lineage, s.gen, GDIgel._tot(r.f), _gd_pass(r.m, r.se, 0.05), _gd_pass(r.m, r.se, 0.1), _gd_pass(r.m, r.se, 0.2),
      round(Statistics.median(filter(!isnan, abs.(expm1.(r.m)))) * 100, digits=2)))
  end
  CSV.write(joinpath(output_dir, "gd_solutions.csv"), rows)
  CSV.write(joinpath(output_dir, "gd_groups.csv"), DataFrame(group=1:ix.ngroups, label=labels))
  @info "gdigel done: $(st.n_evals) evals, $(nrow(rows)) solutions → $(sdir)"
  return st
end
