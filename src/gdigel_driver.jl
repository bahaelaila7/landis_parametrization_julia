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

function _gd_index(injection_cohorts, splots, cell_group, labels, scoring::ScoringOptions; real_only::Bool=true)
  inj = injection_cohorts
  ph = hasproperty(inj, :phantom_frac) ? Float64.(inj.phantom_frac) : zeros(nrow(inj))
  dr = hasproperty(inj, :disturbance_drop_pct) ? Float64.(coalesce.(inj.disturbance_drop_pct, 0.0)) : zeros(nrow(inj))
  pid = Int.(inj.plot_id); sy = Int.(inj.sim_year); esp = Int.(inj.eco_species_id); age = Int.(inj.age_calc); ob = Float64.(inj.agb_sum)
  phmap = Dict{NTuple{4,Int},Float64}(); obs = Dict{NTuple{4,Int},Float64}(); drop = Dict{NTuple{4,Int},Float64}()
  seen = Set{NTuple{4,Int}}((pid[i], sy[i], esp[i], age[i]) for i in eachindex(pid))   # every observed cohort incl. year 0 (recruit test)
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
  # legacy order: only :exclude_overwrite drops partially-disturbed pairs. New order (loss_before_injection): also :exclude_noscale (its
  # sim is NOT scaled/overwritten, so a disturbed observation is not comparable); :scale compares against the (1-drop)-scaled sim and
  # keeps the pair; fully removed cohorts have no observed record at all.
  excl = scoring.disturbance == :exclude_overwrite || (scoring.loss_before_injection && scoring.disturbance == :exclude_noscale)
  key2rec = Dict{NTuple{4,Int},Int}(); rec_pair = Int[]; rec_obs = Float64[]; pairid = Dict{NTuple{3,Int},Int}(); pair_group = Int[]
  for (k, o) in obs
    (excl && drop[k] > 0) && continue
    if scoring.loss_before_injection && scoring.exclude_recruits_at_entry   # recruits at entry have no sim counterpart (no establishment): not scored (as in the loss)
      py0 = get(prevyr, (k[1], k[2]), nothing)
      py0 !== nothing && !((k[1], py0, k[3], k[4] - (k[2] - py0)) in seen) && continue
    end
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
  s = zeros(G); s2 = zeros(G); n = zeros(Int, G); sc = zeros(G, 4); nc = zeros(Int, G, 4); sabs = zeros(G)
  for p in 1:np
    po[p] > 0 || continue
    d = log1p(ps[p]) - log1p(po[p]); g = ix.pair_group[p]; c = _gd_class(po[p])
    s[g] += d; s2[g] += d * d; n[g] += 1; sc[g, c] += d; nc[g, c] += 1; sabs[g] += abs(d)
  end
  f = fill(NaN, G); m = fill(NaN, G); se = fill(NaN, G); a = fill(NaN, G)
  for g in 1:G
    n[g] >= 3 || continue
    a[g] = sabs[g] / n[g]   # objective :mo — absolute accuracy a_g = mean |log1p(sim) − log1p(obs)|
    m[g] = s[g] / n[g]; se[g] = sqrt(max(s2[g] / n[g] - m[g]^2, 0.0) * n[g] / (n[g] - 1) / n[g])
    cl = 0.0; k = 0
    for c in 1:4; nc[g, c] >= min_pairs || continue; cl += (sc[g, c] / nc[g, c])^2; k += 1; end
    f[g] = m[g]^2 + (k > 0 ? size_weight * cl / k : 0.0)
  end
  (f=f, m=m, se=se, n=n, a=a)
end

# ---- objective :mo — Wasserstein-1 age-structure distance per group (docs/mo_gdigel.md §2) -------------------------------
# Re-uses the main loss: `fit_params` returns eco_losses[e].sp_w_loss[gsp] = Σ over the scored (site, measurement-year) visits
# of the per-species W1 age-CDF distance (calculate_species_loss_acc!: `_w_finish(_w1_sum(...))`). One (eco, eco-species) CELL
# is one (e, j) with gsp = eco_species_ids[e][j]; a group is a set of cells (`cell_group`). The group value is the
# ratio-of-sums with the SAME normalisation the main loss applies per cell:
#   cell_norm (cell_normalize: true; GA+FL configs): w_g = _w_pow( Σ_cells sp_w_loss / Σ_cells W_cell )   (LossScales.w_cell[gsp,e] =
#       Σ over the cell's reference visits of the per-reference-max W1 × w_scale_factor, i.e. what `_mo_objectives` divides by;
#       the RANKW stratum weights are NOT applied — they weight the loss, not the accuracy of a group)
#   otherwise: w_g = Σ_cells sp_w_loss / (# scored visits with a reference record for the cells)  (the plain visit mean)
# Per-cell numerator / denominator are returned so several splits can be pooled before the division.
"Number of scored (site, year) visits that carry a reference record, per cell: nvis[e][j]."
function _gd_cell_visits(spdf_plts, eco_species_ids, first_year::Int)
  nv = [zeros(Int, length(ids)) for ids in eco_species_ids]
  for ((_, e), yd) in spdf_plts, (yr, gt) in yd
    yr >= first_year || continue
    for j in eachindex(nv[e]); gt.keys[j] && (nv[e][j] += 1); end
  end
  nv
end
"Per-cell (numerator, denominator) of the W term for ONE evaluation: num[e][j] = sp_w_loss, den[e][j] = the reference's W cell scale or visit count."
function _gd_cell_w(eco_losses, eco_species_ids, nvis, lp::PU.LossParams, scales::PU.LossScales)
  ne = length(eco_species_ids); cell = lp.cell_norm; Ws = scales.w_cell
  num = [zeros(length(ids)) for ids in eco_species_ids]; den = [zeros(length(ids)) for ids in eco_species_ids]
  for e in 1:ne, (j, gsp) in enumerate(eco_species_ids[e])
    num[e][j] = Float64(eco_losses[e].sp_w_loss[gsp])
    den[e][j] = cell ? (size(Ws, 1) >= gsp && size(Ws, 2) >= e ? Float64(Ws[gsp, e]) : 0.0) : Float64(nvis[e][j])
  end
  num, den
end
"Group values w_g from per-cell (num, den) — `groupcells[g]` is the vector of (e, j) cells of group g; NaN when the group has no reference visits."
function _gd_group_w(num, den, groupcells::AbstractVector, lp::PU.LossParams)
  w = fill(NaN, length(groupcells))
  for (g, cs) in enumerate(groupcells)
    nn = 0.0; dd = 0.0
    for (e, j) in cs; nn += num[e][j]; dd += den[e][j]; end
    dd > 0 || continue
    r = nn / dd
    w[g] = lp.cell_norm ? Float64(PU._w_pow(lp, FloatType(r))) : r
  end
  w
end
"Group death term d_g = Σ_cells NLL / Σ_cells #cohorts (NaN when the group has no scored cohort). pmort[e][j] = the cell's PROB_MORT."
function _gd_group_d(acc, pmort, eco_species_ids, groupcells::AbstractVector)
  d = fill(NaN, length(groupcells))
  for (g, cs) in enumerate(groupcells)
    nn = 0.0; dd = 0
    for (e, j) in cs; s, n = death_cell(acc, e, j, pmort[e][j]); nn += s; dd += n; end
    dd > 0 && (d[g] = nn / dd)
  end
  d
end
_gd_groupcells(cell_group, G) = (gc = [Tuple{Int,Int}[] for _ in 1:G]; for (c, g) in cell_group; push!(gc[g], c); end; gc)

# approximate TOST (normal CI, α = 0.05) at margin M: equivalent iff m ± 1.645·se inside ±log1p(M)
_gd_pass(m, se, M) = count(g -> !isnan(m[g]) && abs(m[g]) + 1.645 * se[g] < log1p(M), eachindex(m))

# ---- objective :mo — non-dominated archive on (ΣA, ΣW) --------------------------------------------------------------------
# 2-objective crowding: index of the interior point with the smallest normalised neighbour gap (extremes are never dropped).
function _nd_most_crowded(nd, aref, wref)
  n = length(nd); n <= 2 && return 1
  o = sortperm([e.A for e in nd]); best = 0; bd = Inf
  for k in 2:n-1
    p, q, r = nd[o[k-1]], nd[o[k]], nd[o[k+1]]
    d = abs(r.A - p.A) / aref + abs(p.W - r.W) / wref
    d < bd && (bd = d; best = o[k])
  end
  best
end
# m=3: index of the point (never one that is the unique minimum of an objective) with the smallest nearest-neighbour distance in
# the ref-normalised (A, W, D) space.
function _nd_most_crowded3(nd, aref, wref, dref)
  n = length(nd); n <= 3 && return 1
  P = [(e.A / aref, e.W / wref, e.D / dref) for e in nd]
  keep = trues(n)
  for c in 1:3; keep[argmin([p[c] for p in P])] = false; end
  best = 0; bd = Inf
  for i in 1:n
    keep[i] || continue
    dmin = Inf
    for j in 1:n; j == i && continue; dmin = min(dmin, sqrt(sum(abs2, P[i] .- P[j]))); end
    dmin < bd && (bd = dmin; best = i)
  end
  best == 0 ? 1 : best
end
# m=3 archive: knee = the point minimising the Chebyshev distance to the ideal point (per-objective minima) after normalising each objective by
# its reference total:  max_i (S_i/ref_i − ideal_i/ref_i). Roles: min_A, min_W, min_D, knee. Written to the same files as the m=2 archive plus sumD/D_rel.
function _write_mo_archive3(dir, nd, st, labels, dec)
  isempty(nd) && return
  E = sort(nd; by=e -> e.A); n = length(E)
  x = [e.A / st.aref for e in E]; y = [e.W / st.wref for e in E]; z = [e.D / st.dref for e in E]
  ideal = (minimum(x), minimum(y), minimum(z))
  cheb = [max(x[i] - ideal[1], y[i] - ideal[2], z[i] - ideal[3]) for i in 1:n]
  knee = argmin(cheb); iA = argmin(x); iW = argmin(y); iD = argmin(z)
  adir = joinpath(dir, "mo_archive"); isdir(adir) && rm(adir; recursive=true); mkpath(adir)
  df = DataFrame(idx=Int[], file=String[], gen=Int[], lineage=Int[], weights=String[], sumA=Float64[], sumW=Float64[], sumD=Float64[],
                 A_rel=Float64[], W_rel=Float64[], D_rel=Float64[], cheb_to_ideal=Float64[], role=String[])
  for (i, e) in enumerate(E)
    fn = "arch_$(lpad(i, 4, '0')).jld2"; JLD2.save_object(joinpath(adir, fn), dec(e.u))
    roles = String[]; i == knee && push!(roles, "knee"); i == iA && push!(roles, "min_A"); i == iW && push!(roles, "min_W"); i == iD && push!(roles, "min_D")
    push!(df, (i, fn, e.gen, e.lineage, join(round.(e.lam, digits=3), "|"), e.A, e.W, e.D, x[i], y[i], z[i], cheb[i], join(roles, "+")))
  end
  CSV.write(joinpath(dir, "mo_archive.csv"), df)
  JLD2.save_object(joinpath(dir, "mo_archive_u.jld2"), (u=[e.u for e in E], A=[e.A for e in E], W=[e.W for e in E], D=[e.D for e in E], lam=[e.lam for e in E],
                   aref=st.aref, wref=st.wref, dref=st.dref, labels=labels, a=[e.a for e in E], w=[e.w for e in E], d=[e.d for e in E]))
  nothing
end

# Writes <dir>/mo_archive.csv (sorted by ΣA; knee + extremes flagged) and <dir>/mo_archive/arch_XXXX.jld2 (decoded params);
# mo_archive_u.jld2 holds the u-points. knee = point farthest below the chord between the two extremes in (A/A_ref, W/W_ref).
# lambda_balance = (W/Wref)/(A/Aref + W/Wref): the λ at which the Chebyshev terms are equal at that point.
function _write_mo_archive(dir, nd, st, labels, dec)
  isempty(nd) && return
  o = sortperm([e.A for e in nd]); E = nd[o]; n = length(E)
  x = [e.A / st.aref for e in E]; y = [e.W / st.wref for e in E]
  knee = 1
  if n >= 3
    x1, y1, x2, y2 = x[1], y[1], x[n], y[n]; L = hypot(x2 - x1, y2 - y1) + 1e-300
    dist = [((y2 - y1) * (x[i] - x1) - (x2 - x1) * (y[i] - y1)) / L for i in 1:n]   # signed distance to the chord
    knee = argmin(dist)    # chord runs from min-A (high W) to min-W (high A); convex fronts bulge to the lower-left ⇒ most negative
  else
    knee = argmin(max.(x, y))
  end
  adir = joinpath(dir, "mo_archive"); isdir(adir) && rm(adir; recursive=true); mkpath(adir)
  df = DataFrame(idx=Int[], file=String[], gen=Int[], lineage=Int[], lambda_lineage=Float64[], sumA=Float64[], sumW=Float64[],
                 A_rel=Float64[], W_rel=Float64[], lambda_balance=Float64[], role=String[])
  for (i, e) in enumerate(E)
    fn = "arch_$(lpad(i, 4, '0')).jld2"; JLD2.save_object(joinpath(adir, fn), dec(e.u))
    role = i == knee ? "knee" : i == 1 ? "min_A" : i == n ? "min_W" : ""
    i == knee && (i == 1 || i == n) && (role = "knee+" * (i == 1 ? "min_A" : "min_W"))
    push!(df, (i, fn, e.gen, e.lineage, e.lam, e.A, e.W, x[i], y[i], y[i] / (x[i] + y[i] + 1e-300), role))
  end
  CSV.write(joinpath(dir, "mo_archive.csv"), df)
  JLD2.save_object(joinpath(dir, "mo_archive_u.jld2"), (u=[e.u for e in E], A=[e.A for e in E], W=[e.W for e in E], lam=[e.lam for e in E],
                   aref=st.aref, wref=st.wref, labels=labels, a=[e.a for e in E], w=[e.w for e in E]))
  nothing
end

function parametrize_GDIgel(; ref_soa::SiteSoA, space::PU.ParamSpace, plugin::PluginConfig, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years,
    species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, scoring::ScoringOptions, simb::SimBOptions, spinup::Bool,
    TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=3, resume_from::Union{Nothing,String}=nothing,
    start_from::Union{Nothing,String}=nothing, n_reps::Int=1, no_establishment::Bool=false, injection_cohorts=nothing,
    gd_opts::AbstractDict=Dict{String,Any}(), scale_seed::PU.LossScales=PU.LossScales(), kw...)
  gd = resolve_gdigel(gd_opts)   # validated against GDIGEL_LIST, defaults filled (idempotent for a resolved dict)
  o(k) = gd[k]
  mu = o("mu"); σ0 = o("sigma0"); σstop = o("sigma_stop"); σmin = o("sigma_min")
  patience = o("patience"); sw = o("size_weight"); real_only = o("real_only")
  include_start = o("include_start"); ckpt_every = o("ckpt_every"); min_pairs = o("min_pairs")
  arch_frac = o("archive_min_pass_frac")
  objective = Symbol(o("objective")); objective in (:bias, :mo) || error("gdigel: objective must be bias | mo")
  MO = objective === :mo
  M_OBJ = o("m"); M_OBJ in (2, 3) || error("gdigel: m must be 2 or 3")
  DEATH_FOLD = o("death_fold")
  M3 = MO && M_OBJ == 3
  USE_D = M3 || DEATH_FOLD
  USE_D && scoring.death_term !== :likelihood && error("gdigel: m=3 / death_fold need yaml death_term: likelihood")
  USE_D && !(scoring.sync && scoring.loss_before_injection) && error("gdigel: the death term needs override_injection_sync + loss_before_injection")
  (M3 && DEATH_FOLD) && error("gdigel: death_fold (m=2) and m=3 are exclusive")
  kappa = Ref(0.0)   # death_fold: w'_g = w_g + κ d_g,  κ = W_ref/D_ref (fixed after the start points are evaluated)
  MO && o("mode") == "explore" && error("gdigel: objective mo is not available in mode explore")
  MO && o("single_block") && @warn "gdigel mo: single_block ablation uses the total A, W of all groups as the one block"
  simb.dual_mode == :off || error("gdigel: Sim A only (dual_mode must be off)")
  isnothing(injection_cohorts) && error("gdigel needs override injection cohorts (Sim A)")
  mkpath(output_dir)
  pd = space.dists   # gdigel is Sim A only (dual_mode off), so the run's space already has fit_establishment = false
  template = !isnothing(start_from) ? _load_params_from_path(start_from) : generate_params(plugin.ctx, species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
  slots = space.slots; nd = length(slots)
  cell_group, labels = _gd_groups(pd, slots, eco_species_ids, species_list, eco_list; group_param=o("group_param"))
  ix = _gd_index(injection_cohorts, splots, cell_group, labels, scoring; real_only=real_only)
  # blocks: u-coordinates grouped by the set of groups they affect
  bmap = Dict{Vector{Int},Vector{Int}}()
  for (i, (pi, t)) in enumerate(slots)
    resp = sort!(unique(cell_group[c] for c in _gd_slot_cells(pd.params[pi].sampler, t, eco_species_ids)))
    push!(get!(bmap, resp, Int[]), i)
  end
  bdefs = [GDIgel.BlockDef(idx, resp) for (resp, idx) in sort!(collect(bmap); by=kv -> first(kv[2]))]
  if o("single_block")   # ablation: no decomposition — one (1+1)-CMA over all u-dims judged by the TOTAL loss
    bdefs = [GDIgel.BlockDef(collect(1:nd), collect(1:ix.ngroups))]
  end
  @info "gdigel: $nd u-dims → $(length(bdefs)) blocks (sizes $(sort(length.(getfield.(bdefs, :idx)))) ); $(ix.ngroups) groups; $(length(ix.pair_group)) pairs ($(real_only ? "real-only" : "all"))"
  injection = build_injection_plan(injection_cohorts, ref_soa, eco_species_ids, loss_params, scoring)
  max_sim_year = site_sim_years.sim_years .|> maximum |> maximum
  n_species = length(species_list)
  fixed_seeds = _fixed_seeds(rng, n_reps, scoring)
  scales, _ = build_split_scales(loss_params, scale_seed, search_tier, spdf_plts, nothing, eco_species_ids, n_species; simb)   # the TRAIN (Sim A) reference scales
  groupcells = _gd_groupcells(ix.cell_group, ix.ngroups)
  nvis = MO ? _gd_cell_visits(spdf_plts, eco_species_ids, spinup ? 0 : 1) : nothing
  ws_t = [deepcopy(ref_soa) for _ in 1:Threads.maxthreadid()]
  dec(u) = PU.u_to_params(clamp.(u, 0.0, 1.0), space, slots, template)
  function ev(u; ws=ws_t[Threads.threadid()], sites_parallel::Bool=true)
    pp = dec(u)
    dacc = USE_D ? DeathAcc(eco_species_ids) : nothing
    r = fit_params(ref_soa, pp, max_sim_year, n_species, eco_species_ids, spdf_plts, site_sim_years, spinup, spinup_cohorts, loss_params;
      debug=false, search_tier=search_tier, seeds=fixed_seeds, injection=injection, work_soa=ws, cache_preinject=true, death_acc=dacc, scales=scales, scoring, simb, plugin=plugin, parallel_sites=sites_parallel)
    _, _, k = _agg_reps(r, loss_params, scoring)
    res = _gd_losses(ix, r[k][2]; size_weight=sw, min_pairs=min_pairs)
    MO || return res
    num, den = _gd_cell_w(r[k][3], eco_species_ids, nvis, loss_params, scales)
    w = _gd_group_w(num, den, groupcells, loss_params)
    if USE_D
      d = _gd_group_d(dacc, pp.PROB_MORT_SPP, eco_species_ids, groupcells)
      return merge(res, (w=w, d=d, fx=vcat(res.a, w)))
    end
    merge(res, (w=w, fx=vcat(res.a, w)))   # fx = per-group objective vector handed to GDIgel (a_g ; w_g)
  end
  # the objective vector handed to GDIgel: m=2 (a ; w), m=3 (a ; w ; d), death_fold (a ; w + κ d)
  fxof(r) = M3 ? vcat(r.a, r.w, r.d) : DEATH_FOLD ? vcat(r.a, [isnan(d) ? w : w + kappa[] * d for (w, d) in zip(r.w, r.d)]) : r.fx
  sums_of(r) = M3 ? GDIgel.mo_sums3(fxof(r), ix.ngroups) : GDIgel.mo_sums(fxof(r), ix.ngroups)
  fvec(r) = MO ? fxof(r) : r.f
  # evaluate a batch: candidate-parallel (each eval sites-serial) when PAN_PARALLEL=candidate, else sequential sites-parallel
  function evbatch(us)
    out = Vector{Any}(undef, length(us))
    if PARALLEL_MODE[] == :candidate
      # :static — the work SoA (and the PnET scratch pool) are indexed by threadid, which :dynamic tasks may not keep.
      # Each eval is sites-serial (`ev` passes parallel_sites=false in candidate mode).
      Threads.@threads :static for k in eachindex(us); out[k] = ev(us[k]; ws=ws_t[Threads.threadid()], sites_parallel=false); end
    else
      for k in eachindex(us); out[k] = ev(us[k]; ws=ws_t[1]); end
    end
    out
  end
  archive = NamedTuple[]; cap = o("archive_cap")
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
  # objective :mo — non-dominated archive on (ΣA, ΣW) of EVERY evaluated point (no TOST gating), capped by archive_cap
  ndarc = NamedTuple[]
  function offer_nd!(u, r, gen, lid, lam)
    sm = sums_of(r); A, W = sm[1], sm[2]; D = M3 ? sm[3] : 0.0
    if M3
      any(e -> e.A <= A && e.W <= W && e.D <= D, ndarc) && return
      filter!(e -> !(A <= e.A && W <= e.W && D <= e.D), ndarc)
    else
      any(e -> e.A <= A && e.W <= W, ndarc) && return
      filter!(e -> !(A <= e.A && W <= e.W), ndarc)
    end
    push!(ndarc, (gen=gen, lineage=lid, lam=lam, u=copy(u), A=A, W=W, D=D, a=copy(r.a), w=copy(r.w), d=(USE_D ? copy(r.d) : Float64[])))
    length(ndarc) > cap && deleteat!(ndarc, M3 ? _nd_most_crowded3(ndarc, st_ref[].aref, st_ref[].wref, st_ref[].dref) : _nd_most_crowded(ndarc, st_ref[].aref, st_ref[].wref))
  end
  st_ref = Ref{Any}(nothing)
  if resume_from !== nothing
    saved = JLD2.load_object(resume_from); st = saved.st; append!(archive, saved.archive)
    MO && hasproperty(saved, :nd) && append!(ndarc, saved.nd)
    st_ref[] = st
    @info "gdigel: resumed $(basename(resume_from)) at gen $(st.gen), $(st.n_evals) evals, $(length(st.solutions)) solutions"
  else
    mode = Symbol(o("mode"))
    lams = !MO ? Float64[] : M3 ? GDIgel.simplex_weights(o("weights")) : GDIgel.mo_lambdas(o("weights"))
    # :explore (anchor-and-explore, docs/cmaes_gdigel.md §4): behavioural threshold tau from the tolerated bias
    u_anchor = start_from === nothing ? nothing : PU.params_to_u(template, space, slots)
    mode === :explore && u_anchor === nothing && error("gdigel mode explore needs start_from (the anchor parameter set)")
    # per-group behavioural threshold: tau_g = max(log1p(explore_bias)², explore_slack × f_g(anchor)) — the anchor is
    # behavioural by construction, and groups the anchor already fits well get the absolute tolerance
    tau = Float64[]
    if mode === :explore
      fa = only(evbatch([u_anchor])).f; t0 = log1p(o("explore_bias"))^2; sl = o("explore_slack")
      tau = [isnan(v) ? Inf : max(t0, sl * v) for v in fa]
      @info "gdigel explore: anchor total $(round(GDIgel._tot(fa), sigdigits=4)); tau_abs $(round(t0, sigdigits=3)), slack $sl; groups at the absolute tolerance: $(count(==(t0), tau))/$(count(isfinite, tau))"
    end
    st = GDIgel.GDState(bdefs, nd; sigma0=σ0, sigma_stop=σstop, sigma_min=σmin, patience=patience, rel_tol=o("rel_tol"), seed=o("seed"),
                        mode=mode, tau=tau, anchors=(mode === :explore ? [u_anchor] : Vector{Float64}[]),
                        restart_sigma=o("explore_start_sigma"), obj=objective, G=ix.ngroups, lams=lams, m=(M3 ? 3 : 2))
    st_ref[] = st
    starts = Vector{Vector{Float64}}()
    if mode === :explore   # every lineage starts at the anchor, slightly perturbed
      while length(starts) < mu; push!(starts, clamp.(u_anchor .+ o("explore_start_sigma") .* randn(st.rng, nd), 0.0, 1.0)); end
    else
      include_start && start_from !== nothing && push!(starts, u_anchor)
    end
    while length(starts) < mu; push!(starts, GDIgel.next_start!(st)); end
    r0 = evbatch(starts)
    if MO   # reference totals A_ref, W_ref (fixed for the run): the start_from point if given, else the median of the μ start points
      if DEATH_FOLD   # κ = W_ref/D_ref from the raw (unfolded) sums of the start points (or the anchor), fixed for the run
        rr = start_from !== nothing ? [only(evbatch([u_anchor]))] : r0
        Wr = Statistics.median([GDIgel.mo_sums(vcat(r.a, r.w), ix.ngroups)[2] for r in rr]); Dr = Statistics.median([sum(filter(!isnan, r.d)) for r in rr])
        kappa[] = Wr / Dr; o("kappa") === nothing || (kappa[] = o("kappa"))
        write(joinpath(output_dir, "kappa.txt"), string(kappa[]))
        @info "gdigel mo (death_fold): κ = W_ref/D_ref = $(round(Wr, sigdigits=5))/$(round(Dr, sigdigits=5)) = $(round(kappa[], sigdigits=5))"
      end
      sums = [sums_of(r) for r in r0]
      if start_from !== nothing
        sa = sums_of(only(evbatch([u_anchor]))); st.aref, st.wref = sa[1], sa[2]; M3 && (st.dref = sa[3])
      else
        st.aref = Statistics.median([x[1] for x in sums]); st.wref = Statistics.median([x[2] for x in sums]); M3 && (st.dref = Statistics.median([x[3] for x in sums]))
      end
      o("aref") === nothing || (st.aref = o("aref")); o("wref") === nothing || (st.wref = o("wref")); o("dref") === nothing || (st.dref = o("dref"))
      @info "gdigel mo$(M3 ? " (m=3)" : DEATH_FOLD ? " (death folded)" : ""): A_ref = $(round(st.aref, sigdigits=5)), W_ref = $(round(st.wref, sigdigits=5))$(M3 ? ", D_ref = $(round(st.dref, sigdigits=5))" : ""); weight grid $(M3 ? "simplex lattice, $(length(st.lams)) points" : "λ " * string(round.(st.lams, digits=3))) over $mu lineages"
    end
    for k in 1:mu; GDIgel.add_lineage!(st, starts[k], fvec(r0[k])); offer!(starts[k], r0[k], 0, k); MO && offer_nd!(starts[k], r0[k], 0, k, st.lineages[k].lam); end
    st.n_evals = mu
  end
  logf = joinpath(output_dir, "gd_log.csv")
  isfile(logf) || open(io -> println(io, "gen,n_evals,lineage,born,total,n05,n10,n20,median_abs_bias_pct,mean_sigma,min_sigma,max_sigma" * (MO ? ",lambda,sumA,sumW" * (M3 ? ",sumD" : "") : "")), logf, "w")
  last_r = Dict{Int,Any}()
  t_start = time()
  while st.n_evals < TRIALS
    ys = GDIgel.ask(st)
    rys = evbatch(ys)
    xps = GDIgel.assemble!(st, ys, [fvec(r) for r in rys])
    need = findall(!isnothing, xps)
    rxp = isempty(need) ? Any[] : evbatch([xps[k] for k in need])
    fxps = Vector{Any}(nothing, length(st.lineages)); rmap = Dict{Int,Any}()
    for (j, k) in enumerate(need); fxps[k] = fvec(rxp[j]); rmap[k] = rxp[j]; end
    for (k, L) in enumerate(st.lineages)
      offer!(ys[k], rys[k], st.gen + 1, L.id); haskey(rmap, k) && offer!(xps[k], rmap[k], st.gen + 1, L.id)
      if MO
        offer_nd!(ys[k], rys[k], st.gen + 1, L.id, L.lam); haskey(rmap, k) && offer_nd!(xps[k], rmap[k], st.gen + 1, L.id, L.lam)
      end
    end
    # which evaluated result is the new parent (for logging): resolved after finish!
    cand = [(ys[k], rys[k], haskey(rmap, k) ? (xps[k], rmap[k]) : nothing) for k in eachindex(ys)]
    restart_eval(x, lam) = (r = only(evbatch([x])); offer!(x, r, st.gen, -1); MO && offer_nd!(x, r, st.gen, -1, lam); fvec(r))
    restart_eval(x) = restart_eval(x, M3 ? [1/3, 1/3, 1/3] : 0.5)
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
        mo_cols = MO ? (M3 ? (sa = GDIgel.mo_sums3(L.f, ix.ngroups); ",$(join(round.(L.lam, digits=4), "|")),$(sa[1]),$(sa[2]),$(sa[3])") : (sa = GDIgel.mo_sums(L.f, ix.ngroups); ",$(L.lam),$(sa[1]),$(sa[2])")) : ""
        if r === nothing
          println(io, "$(st.gen),$(st.n_evals),$(L.id),$(L.born),$(L.tot),,,,,$(sum(sig)/length(sig)),$(minimum(sig)),$(maximum(sig))$(mo_cols)")
        else
          mab = Statistics.median(filter(!isnan, abs.(expm1.(r.m)))) * 100   # (TOST pass counts here are INFORMATION only; never used by the search)
          println(io, "$(st.gen),$(st.n_evals),$(L.id),$(L.born),$(L.tot),$(_gd_pass(r.m, r.se, 0.05)),$(_gd_pass(r.m, r.se, 0.1)),$(_gd_pass(r.m, r.se, 0.2)),$(round(mab, digits=2)),$(sum(sig)/length(sig)),$(minimum(sig)),$(maximum(sig))$(mo_cols)")
        end
      end
    end
    best = minimum(L.tot for L in st.lineages)
    @info "gdigel gen $(st.gen): evals $(st.n_evals), best total $(round(best, sigdigits=4)), solutions $(length(st.solutions)), archive $(MO ? length(ndarc) : length(archive)), $(round((time() - t_start) / st.gen, digits=1)) s/gen"
    if st.gen % ckpt_every == 0
      JLD2.save_object(joinpath(output_dir, "gd_state@$(st.gen).jld2"), (st=st, archive=archive, labels=labels, slots=slots, template=template, nd=ndarc))
      MO && (M3 ? _write_mo_archive3(output_dir, ndarc, st, labels, dec) : _write_mo_archive(output_dir, ndarc, st, labels, dec))
    end
  end
  JLD2.save_object(joinpath(output_dir, "gd_state@$(st.gen).jld2"), (st=st, archive=archive, labels=labels, slots=slots, template=template, nd=ndarc))
  MO && (M3 ? _write_mo_archive3(output_dir, ndarc, st, labels, dec) : _write_mo_archive(output_dir, ndarc, st, labels, dec))
  # final solutions: converged lineages + the current lineages (as unconverged end points)
  sols = vcat([(kind="converged", lineage=s.lineage, gen=s.gen, x=s.x, lam=s.lam) for s in st.solutions],
              [(kind="final", lineage=L.id, gen=st.gen, x=L.x, lam=L.lam) for L in st.lineages])
  sdir = joinpath(output_dir, "solutions"); mkpath(sdir)
  rows = DataFrame(file=String[], kind=String[], lineage=Int[], gen=Int[], total=Float64[], n05=Int[], n10=Int[], n20=Int[], median_abs_bias_pct=Float64[])
  MO && (rows.lambda = (M3 ? String[] : Float64[]); rows.sumA = Float64[]; rows.sumW = Float64[]; M3 && (rows.sumD = Float64[]))
  rs = evbatch([s.x for s in sols])
  for (i, s) in enumerate(sols)
    fn = "sol_$(lpad(i, 3, '0'))_$(s.kind)_L$(s.lineage).jld2"; JLD2.save_object(joinpath(sdir, fn), dec(s.x)); r = rs[i]
    if MO   # total = F(λ) of the lineage; TOST counts are information only
      sa = sums_of(r)
      push!(rows, (fn, s.kind, s.lineage, s.gen, GDIgel.lineage_total(st, s.lam, fxof(r)), _gd_pass(r.m, r.se, 0.05), _gd_pass(r.m, r.se, 0.1), _gd_pass(r.m, r.se, 0.2),
        round(Statistics.median(filter(!isnan, abs.(expm1.(r.m)))) * 100, digits=2), (M3 ? join(round.(s.lam, digits=4), "|") : s.lam), sa[1], sa[2], (M3 ? (sa[3],) : ())...))
    else
      push!(rows, (fn, s.kind, s.lineage, s.gen, GDIgel._tot(r.f), _gd_pass(r.m, r.se, 0.05), _gd_pass(r.m, r.se, 0.1), _gd_pass(r.m, r.se, 0.2),
        round(Statistics.median(filter(!isnan, abs.(expm1.(r.m)))) * 100, digits=2)))
    end
  end
  CSV.write(joinpath(output_dir, "gd_solutions.csv"), rows)
  CSV.write(joinpath(output_dir, "gd_groups.csv"), DataFrame(group=1:ix.ngroups, label=labels))
  @info "gdigel done: $(st.n_evals) evals, $(nrow(rows)) solutions → $(sdir)"
  return st
end
