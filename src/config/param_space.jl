# The PARAMETER SPACE of a run (Step 3c): everything that decides which parameters are searched, how u-coordinates decode to them and how the
# covariance is blocked. It is built ONCE per run from (config, species/eco lists, eco_species_ids, data tables) as an immutable
# `PU.ParamSpace` and passed explicitly to every driver, decoder and tool — no module-level state (these used to be 12 `Ref`s in
# PU / BSP / Pan, written by `parametrize` and read implicitly by `u_to_params`).
#
#   param_space_options(cfg)  -> PU.ParamSpaceOptions     which parameters exist (fix_growth, ...) and their boxes
#   build_param_tying(...)    -> PU.ParamTying             pooling labels / split sets / tier merges (what shares a u-dimension)
#   cma_blocks(mode, ecos)    -> PU.CmaBlocks              block-diagonal covariance layout
#   build_param_floors(...)   -> PU.ParamFloors            data floors of B_MAX / ANPP / PROB_ESTAB
#   make_param_space(pc, ...) -> PU.ParamSpace             distributions + slots + all of the above (pc = the plugin context)

"""
    param_space_options(cfg) -> PU.ParamSpaceOptions

The ONE place the parameter-space switches are read from a loaded config (`run_from_yaml` and the analysis tools both call it, so a tool decodes
exactly the space the run searched). Pure: it touches no module state. `death_pmort_sima` (a score-only PROB_MORT in Sim A) follows
`death_term: likelihood`.
"""
function param_space_options(cfg0::AbstractDict)::PU.ParamSpaceOptions
  cfg = _as_config(cfg0)
  return PU.ParamSpaceOptions(;
    fix_growth=cfg["fix_growth"], fix_maturity=cfg["fix_maturity"], fix_min_rel=cfg["fix_min_rel"],
    death_pmort_sima=(cfg["death_term"] === :likelihood),
    fix_pmort=cfg["fix_pmort"], pmort_upper_a=Float32(cfg["pmort_upper_simA"]), pmort_upper_b=Float32(cfg["pmort_upper"]),
    bmax_bounds=Tuple{Int,Int}((12000, cfg["bmax_upper"])),                       # widen the B_MAX search box (lower stays 12000)
    anpp_ratio_bounds=Tuple{Float32,Float32}(Tuple(Float32.(cfg["anpp_ratio_bounds"]))),   # B_MAX/ANPP ratio box
    prob_estab_lower_bound=cfg["prob_estab_lower_bound"], prob_estab_lower_mult=cfg["prob_estab_lower_bound_mult"])
end

"""
    make_param_space(pc, species_list, eco_list, eco_species_ids; no_establishment, fit_establishment, opts, tying, floors, blocks) -> PU.ParamSpace

The search space of the succession plugin of context `pc` (`BSTables` | `PnETContext` | `HBADContext`) over the given data lists. `blocks` carries the covariance mode and L3/L2 indices (see `cma_blocks`);
the plugin's per-eco groups are filled in here. Defaults give the plain legacy space (no floors, no pooling, `:eco` blocks).
"""
function make_param_space(pc, species_list, eco_list, eco_species_ids::Vector{Vector{Int}}; no_establishment::Bool=false, fit_establishment::Bool=false,
    opts::PU.ParamSpaceOptions=PU.ParamSpaceOptions(), tying::PU.ParamTying=PU.ParamTying(), floors::PU.ParamFloors=PU.ParamFloors(),
    blocks::PU.CmaBlocks=PU.CmaBlocks())
  dists = make_param_dists(pc, length(species_list), length(eco_list), eco_species_ids; no_establishment=no_establishment, fit_establishment=fit_establishment, opts=opts)
  blocks = PU.CmaBlocks(; mode=blocks.mode, eco_l3=blocks.eco_l3, eco_l2=blocks.eco_l2, per_eco_groups=per_eco_groups(plugin_type(pc)))
  return PU.ParamSpace(dists, length(species_list), eco_list, eco_species_ids; opts=opts, tying=tying, floors=floors, blocks=blocks)
end

# eco string "8.3.5|lu=X" → L3 "8.3.5"; L3 "8.3.5" → L2 "8.3" ("15.4.1" → "15.4")
_l3_of(eco::AbstractString) = String(first(split(String(eco), "|lu=")))
_l2_of(l3::AbstractString) = (p = split(String(l3), "."); length(p) >= 2 ? join(p[1:2], ".") : String(l3))

"""
    cma_blocks(mode::AbstractString, eco_list) -> PU.CmaBlocks

CMA-ES covariance blocking (`cmaes_blocks`): "eco" (one block per ecoregion cell, legacy) or "l3" (one per L3, + L2 / region tail blocks for
pooled slots spanning several L3s).
"""
function cma_blocks(mode::AbstractString, eco_list)::PU.CmaBlocks
  m = Symbol(lowercase(mode))
  m === :l3 || return PU.CmaBlocks(; mode=m)
  l3 = [_l3_of(e) for e in eco_list]; u3 = unique(l3)
  l2 = [_l2_of(x) for x in l3]; u2 = unique(l2)
  b = PU.CmaBlocks(; mode=m, eco_l3=[findfirst(==(x), u3) for x in l3], eco_l2=[findfirst(==(x), u2) for x in l2])
  @info "CMA-ES blocks per L3: $(length(unique(b.eco_l3))) L3s (+ tail block for multi-L3 pooled dims)"
  return b
end

# param_pooling: per (param, eco cell, species) pooling label from TRAIN plot counts (no val/test leakage).
#   min_plots: an (level-group, species) needs ≥ this many train plots to own a dim, else fall back a level.
#   params: {P: {level: l3, fallback: [l2, region], cell_split: [PITA, PIEL]}}  levels: cell | l3 | l2 | region
#     cell_split species keep per-site-cell dims (merged by cell_merge) wherever the eco has real cells.
#   overrides: {P: {"8.4.4|QURU": region}}  force a level.   floor_pooled: max (default) | first.
# Deterministic given the split ⇒ resume-safe; written to output_dir/pool_map.csv for inspection.
# Returns (labels, floor_pooled_max); empty labels ⇒ legacy (no pooling).
function _build_pool_labels(pool_cfg::AbstractDict, splots, eco_list, species_list, eco_species_ids, output_dir::AbstractString)
  labels = Dict{Symbol,Dict{Tuple{Int,Int},String}}()
  isempty(pool_cfg) && return labels, false
  min_plots = Int(get(pool_cfg, "min_plots", 40))
  floor_pooled_max = String(get(pool_cfg, "floor_pooled", "max")) == "max"
  cnt = Dict{Tuple{Int,Int},Int}()                       # (eco_id, gsp) → # distinct TRAIN plots carrying gsp
  for g in groupby(unique(select(splots, [:plot_id, :eco_id, :species_id])), [:eco_id, :species_id])
    cnt[(Int(g.eco_id[1]), Int(g.species_id[1]))] = nrow(g)
  end
  l3s = [_l3_of(e) for e in eco_list]
  keyof(lv, e) = lv == "cell" ? String(eco_list[e]) : lv == "l3" ? l3s[e] : lv == "l2" ? _l2_of(l3s[e]) :
                 lv in ("region", "reg") ? "R" : error("param_pooling: unknown level '$lv' (cell|l3|l2|region)")
  count_at(lv, key, gsp) = sum((get(cnt, (e, gsp), 0) for e in eachindex(eco_list) if keyof(lv, e) == key); init=0)
  cell_merge = get(pool_cfg, "cell_merge", Dict())
  overrides = get(pool_cfg, "overrides", Dict())
  rows = NamedTuple[]
  for (pname, spec) in get(pool_cfg, "params", Dict())
    levels = vcat([String(spec["level"])], String.(get(spec, "fallback", String[])))
    csplit = Set(String.(get(spec, "cell_split", String[])))
    ov = get(overrides, String(pname), Dict())
    lab = Dict{Tuple{Int,Int},String}()
    for (e, sp_ids) in enumerate(eco_species_ids), sl in eachindex(sp_ids)
      gsp = Int(sp_ids[sl]); sp = species_list[gsp]; cell = PU._eco_lu(eco_list[e])
      l = if sp in csplit && !(cell in ("", "X"))
        "cell:$(l3s[e])|$(get(get(cell_merge, sp, Dict()), cell, cell))"
      else
        lv = get(ov, "$(l3s[e])|$sp", nothing)
        if lv === nothing
          i = findfirst(x -> count_at(x, keyof(x, e), gsp) >= min_plots, levels)
          lv = i === nothing ? last(levels) : levels[i]
        end
        "$(lv):$(keyof(String(lv), e))"
      end
      lab[(e, sl)] = "$l|$sp"
      push!(rows, (param=String(pname), eco=String(eco_list[e]), species=sp, n_train_plots=get(cnt, (e, gsp), 0), label=lab[(e, sl)]))
    end
    labels[Symbol(pname)] = lab
    lvc = Dict{String,Int}()
    for l in unique(values(lab)); k = first(split(l, ":")); lvc[k] = get(lvc, k, 0) + 1; end
    @info "param_pooling $(pname): $(length(unique(values(lab)))) dims over $(length(lab)) (eco,species) targets" by_level=lvc
  end
  mkpath(output_dir)
  CSV.write(joinpath(output_dir, "pool_map.csv"), DataFrame(rows))
  return labels, floor_pooled_max
end

"""
    build_param_tying(param_split_species, param_tier_merge, param_pooling, splots, eco_list, species_list, eco_species_ids, output_dir) -> PU.ParamTying

Which targets of an EcoSpecies parameter share one u-dimension, from the three config blocks:
  - `param_split_species` {param => [species]}: fit per-eco only for the listed species; every other species is tied across ecoregions (empty ⇒ all split).
  - `param_tier_merge` {param => {species => {cell => group}}}: split cells sharing a label are tied.
  - `param_pooling` (region runs): L3 → L2 → region pooling from the TRAIN plot counts (writes `pool_map.csv`).
"""
function build_param_tying(param_split_species::AbstractDict, param_tier_merge::AbstractDict, param_pooling::AbstractDict, splots, eco_list, species_list, eco_species_ids,
    output_dir::AbstractString)::PU.ParamTying
  split_sets = Dict{Symbol,Set{Int}}()
  for (pname, syms) in param_split_species
    ids = Set{Int}()
    for sym in syms
      i = findfirst(==(String(sym)), species_list)
      i === nothing ? (@warn "split-set species not in tiering — skipped" param=pname species=sym) : push!(ids, i)
    end
    isempty(ids) || (split_sets[Symbol(pname)] = ids)
  end
  isempty(split_sets) ||
    @info "Param split sets (species fit per-eco; others tied across ecos)" splits=Dict(string(k) => [species_list[i] for i in sort(collect(v))] for (k, v) in split_sets)
  tier_merge = Dict{Symbol,Dict{Int,Dict{String,String}}}()
  for (pname, spmap) in param_tier_merge
    m = Dict{Int,Dict{String,String}}()
    for (sym, cellmap) in spmap
      i = findfirst(==(String(sym)), species_list)
      i === nothing ? (@warn "tier-merge species not in tiering — skipped" param=pname species=sym) :
        (m[i] = Dict{String,String}(String(c) => String(g) for (c, g) in cellmap))
    end
    isempty(m) || (tier_merge[Symbol(pname)] = m)
  end
  isempty(tier_merge) ||
    @info "Param tier merges (split cells tied per species)" merges=Dict(string(k) => Dict(species_list[i] => cm for (i, cm) in v) for (k, v) in tier_merge)
  labels, floor_pooled_max = _build_pool_labels(param_pooling, splots, eco_list, species_list, eco_species_ids, output_dir)
  return PU.ParamTying(; labels=labels, floor_pooled_max=floor_pooled_max, split_sets=split_sets, tier_merge=tier_merge)
end

# Per-(eco_id, sp_local) B_MAX lower bound = max(12000, table[(species,l3,lu)]) (nothing when the feature is off). The table is
# `BSTables.bmax_floor`, passed in explicitly.
# A floor above the B_MAX upper bound pins its cell at the floor (B18): warned ONCE here, when the floors of the run are built.
function _build_bmax_floor(tables::BiomassSuccessionPlugin.BSTables, opts::PU.ParamSpaceOptions, eco_list, species_list, eco_species_ids)
  tbl = tables.bmax_floor
  isnothing(tbl) && return nothing
  d = Dict{Tuple{Int,Int},FloatType}()
  for (eco_id, sp_ids) in enumerate(eco_species_ids)
    l3, lu = _parse_eco_lu(eco_list[eco_id])
    for (j, gsp) in enumerate(sp_ids)
      d[(eco_id, j)] = FloatType(max(12000.0, get(tbl, (uppercase(species_list[gsp]), l3, lu), 12000.0)))
    end
  end
  lo, hi = opts.bmax_bounds
  @info "bmax_floor_from_data: B_MAX lower bound set for $(length(d)) eco×species cells ($(count(>(FloatType(lo)), values(d))) above the $(lo) default); window [$(lo),$(hi)]"
  pinned = sort!([(eco_list[e], species_list[eco_species_ids[e][j]], Float64(f)) for ((e, j), f) in d if f > hi]; by=x -> -x[3])
  isempty(pinned) || @warn "bmax_floor_from_data: $(length(pinned)) cells have a data floor above the B_MAX upper bound $(hi) — their B_MAX is pinned at the floor (no search freedom; raise `bmax_upper`)" cells = join(["$(c[1])/$(c[2])=$(round(Int, c[3]))" for c in first(pinned, 20)], ", ")
  return d
end

# Per-(eco_id, sp_local) PROB_ESTAB lower bound = mult × data value (yaml `prob_estab_lower_bound`: use the per-species data PROB_ESTAB as a LOWER
# bound on the search, window [data, 1.0], instead of only seeding it, so the optimizer can't suppress establishment below the empirically observed
# rate). Cells with no table entry keep the param's own lower bound. nothing when off or without the table (prob_estab_from_data).
function _build_prob_estab_floor(tables::BiomassSuccessionPlugin.BSTables, opts::PU.ParamSpaceOptions, eco_list, species_list, eco_species_ids)
  tbl = tables.prob_estab
  (!opts.prob_estab_lower_bound || isnothing(tbl)) && return nothing
  mult = opts.prob_estab_lower_mult
  d = Dict{Tuple{Int,Int},FloatType}()
  for (eco_id, sp_ids) in enumerate(eco_species_ids)
    l3, lu = _parse_eco_lu(eco_list[eco_id])
    lu in ("natural", "artificial") || (lu = "natural")   # site_class_strata puts a site-tier in |lu= ⇒ use natural
    for (j, gsp) in enumerate(sp_ids)
      v = get(tbl, (uppercase(species_list[gsp]), l3, lu), nothing)
      v === nothing || (d[(eco_id, j)] = FloatType(clamp(mult * Float64(v), 0.0, 1.0)))
    end
  end
  @info "prob_estab_lower_bound: PROB_ESTAB search floored at $(mult)×data for $(length(d)) eco×species cells (window [floor,1.0])"
  return isempty(d) ? nothing : d
end

# Per-(eco_id, sp_local) ANPP lower bound = table[(species,l3,lu)] (0 = no floor); nothing when off. Parallel to _build_bmax_floor.
function _build_anpp_floor(tables::BiomassSuccessionPlugin.BSTables, eco_list, species_list, eco_species_ids)
  tbl = tables.anpp_floor
  isnothing(tbl) && return nothing
  d = Dict{Tuple{Int,Int},FloatType}()
  for (eco_id, sp_ids) in enumerate(eco_species_ids)
    l3, lu = _parse_eco_lu(eco_list[eco_id])
    for (j, gsp) in enumerate(sp_ids)
      d[(eco_id, j)] = FloatType(get(tbl, (uppercase(species_list[gsp]), l3, lu), 0.0))
    end
  end
  @info "anpp_floor_from_data: ANPP lower bound set for $(length(d)) eco×species cells ($(count(>(FloatType(0)), values(d))) with a nonzero floor)"
  return d
end

"""
    build_param_floors(tables, opts, eco_list, species_list, eco_species_ids) -> PU.ParamFloors

The data floors of the decode (B_MAX, derived ANPP, PROB_ESTAB), one dict per (eco_id, sp_local); each `nothing` when its feature is off.
Built once per run (the B18 warning for floors above the B_MAX upper bound therefore fires once).
"""
build_param_floors(tables::BiomassSuccessionPlugin.BSTables, opts::PU.ParamSpaceOptions, eco_list, species_list, eco_species_ids) =
  PU.ParamFloors(; bmax=_build_bmax_floor(tables, opts, eco_list, species_list, eco_species_ids),
                   anpp=_build_anpp_floor(tables, eco_list, species_list, eco_species_ids),
                   prob_estab=_build_prob_estab_floor(tables, opts, eco_list, species_list, eco_species_ids))

# Raise any seed B_MAX below its floor up to the floor, so the initial candidate is feasible / consistent
# with the rescaled decode. No-op when the floor feature is off.
function _floor_bmax_seed!(params, space::PU.ParamSpace)
  d = space.floors.bmax
  isnothing(d) && return params
  for (eco_id, sp_ids) in enumerate(params.ECO_SPECIES_IDS)
    for j in eachindex(sp_ids)
      lo = get(d, (eco_id, j), FloatType(12000))
      params.B_MAX_SPP[eco_id][j] < lo && (params.B_MAX_SPP[eco_id][j] = lo)
    end
  end
  params
end

# PIN MIN_REL_BIOMASS out of the search: overwrite every ecoregion's thresholds with MIN_REL_PINNED IN PLACE (no-op unless fix_min_rel).
# next_candidate() draws a random gradient, so (like _apply_frozen_growth!) this must be re-applied on restart.
function _seed_min_rel!(params, eco_list, opts::PU.ParamSpaceOptions)
  opts.fix_min_rel || return params
  hasproperty(params, :MIN_REL_BIOMASS) || return params   # other succession plugins (HBAD has MIN_REL_RD, fixed by default)
  for e in eachindex(params.MIN_REL_BIOMASS)
    params.MIN_REL_BIOMASS[e] = copy(MIN_REL_PINNED)
  end
  @info "fix_min_rel: pinned MIN_REL_BIOMASS = $(MIN_REL_PINNED) for $(length(params.MIN_REL_BIOMASS)) ecoregions (out of search)"
  params
end

# Re-pin the fix_growth-frozen params {D,S,ANPP_MAX,B_MAX} to a reference (start_from / Sim A) IN PLACE. IPOP
# restart draws a fresh next_candidate() with RANDOM growth, which silently breaks fix_growth (frozen params
# must stay at the seed). No-op when fix_growth is off or no reference is given.
function _apply_frozen_growth!(cand, ref, opts::PU.ParamSpaceOptions)
  (opts.fix_growth && ref !== nothing) || return cand
  cand.D .= ref.D
  cand.S .= ref.S
  for e in eachindex(cand.ANPP_MAX_SPP); cand.ANPP_MAX_SPP[e] .= ref.ANPP_MAX_SPP[e]; end
  for e in eachindex(cand.B_MAX_SPP); cand.B_MAX_SPP[e] .= ref.B_MAX_SPP[e]; end
  cand
end

"""
    with_establishment(pc, space; no_establishment, fit_establishment) -> PU.ParamSpace

The run's `space` over the distributions of the same plugin (context `pc`) with other establishment switches (tools that fit another parameter subset than
the run did); options, tying, floors and blocks are kept.
"""
with_establishment(pc, space::PU.ParamSpace; no_establishment::Bool, fit_establishment::Bool=false) =
  PU.with_dists(space, make_param_dists(pc, space.n_species, length(space.eco_list), space.eco_species_ids;
                                         no_establishment=no_establishment, fit_establishment=fit_establishment, opts=space.opts))

"""
    param_space_from_cfg(cfg, pc, splots, eco_list, species_list, eco_species_ids; no_establishment, fit_establishment=false, floors=true, output_dir=mktempdir()) -> PU.ParamSpace

The space `parametrize` builds, re-derived from a loaded config and the prepared data (analysis tools that run no search, e.g. gd_equifinality):
the same options, split sets / tier merges / pooling (from the TRAIN plots `splots`; `pool_map.csv` goes to `output_dir`) and covariance blocks.
`floors` adds the data floors from the plugin context's `BSTables` (`bs_tables(pc)`; `nothing` entries unless the caller loaded the tables, see `bs_tables_from_cfg`).
"""
function param_space_from_cfg(cfg0::AbstractDict, pc, splots, eco_list, species_list, eco_species_ids; no_establishment::Bool, fit_establishment::Bool=false,
    floors::Bool=true, output_dir::AbstractString=mktempdir())
  cfg = _as_config(cfg0)
  opts = param_space_options(cfg)
  tying = build_param_tying(cfg["param_split_species"], cfg["param_tier_merge"], cfg["param_pooling"], splots, eco_list, species_list, eco_species_ids, output_dir)
  fl = floors ? build_param_floors(bs_tables(pc), opts, eco_list, species_list, eco_species_ids) : PU.ParamFloors()
  return make_param_space(pc, species_list, eco_list, eco_species_ids; no_establishment=no_establishment, fit_establishment=fit_establishment,
    opts=opts, tying=tying, floors=fl, blocks=cma_blocks(cfg["cmaes_blocks"], eco_list))
end
