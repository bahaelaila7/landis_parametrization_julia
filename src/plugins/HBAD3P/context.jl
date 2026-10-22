# The immutable per-run context of the HBAD3P plugin: the fixed tables, the species -> key mapping, the HBAD v1 priors (REPORTED allometric AGB and the AGB -> state translation of `add_cohort!` only),
# the eco -> species table and the plot-normal water index table of the parameter set. Built once per (species list, eco table).

"""
Kernel options. `vigour`: recruits / planted cohorts enter as 5 vigour sub-cohorts (Gauss-Hermite nodes), else as one mean-field cohort (zq = 0).
`cleanup_tph`: negligible-cohort cleanup (long scenario runs): at the end of every annual step (before the reported state is computed) cohorts with fewer than `cleanup_tph` trees/ha are removed, keeping
the order of the others. 0 (the default) = OFF: the kernel is then exactly the reference's (cohorts are never removed).
"""
Base.@kwdef struct HBAD3PConst
  vigour::Bool = true
  cleanup_tph::Float64 = 0.0
end

struct HBAD3PContext
  priors::HBADPlugin.HBADPriors          # v1 priors of the run's species (allometry mq_a / mq_b, translation)
  eco_gids::HBADPlugin.EcoGids
  fixed::HBAD3PConst
  keys::Vector{String}                   # keys in file order
  sp::Vector{SpPP}                       # per key
  rec::RecGP
  glob::GlobPP
  farea::Bool                            # density convention of the fit: per ha of the FORESTED subplot
  species_key::Vector{String}            # per Pan species: its key
  params_dir::String
  tables::BiomassSuccessionPlugin.BSTables
  wn::Dict{String,Float64}               # plot normal water index (AET/PET) by plot key "STATE_UNIT_COUNTY_PLOT" (the scenario places W = Wn = this value)
  wn_median::Float64                     # regional median, for plots without a normal
end

"""Read a plot-normal water table: csv with columns `st,u,c,p,wn` -> Dict "st_u_c_p" => wn, and the median over plots."""
function load_plot_normals(path::AbstractString)
  isfile(path) || error("HBAD3P: plot-normal water table not found: $path")
  d = CSV.read(path, DataFrame)
  for c in (:st, :u, :c, :p, :wn); hasproperty(d, c) || error("HBAD3P: $path needs the column '$c'"); end
  m = Dict{String,Float64}(string(r.st, "_", r.u, "_", r.c, "_", r.p) => Float64(r.wn) for r in eachrow(d))
  return m, (isempty(m) ? 0.9 : Statistics.median(collect(values(m))))
end

"""
    HBAD3PContext(species_list, eco_species_ids; params_dir, priors_csv, coefs_csv, vigour, verbose, tables, tables3, wn, wn_median)

Load the tables of `params_dir` (default `results/hbad3p/P3_GAFL`), map every species of `species_list` to its key (logged) and resolve the v1 priors. `tables3` = tables of a loaded yaml roster;
`wn` = plot-normal water table (key "STATE_UNIT_COUNTY_PLOT" => AET/PET; see `load_plot_normals`), `wn_median` = value of plots without a normal.
"""
function HBAD3PContext(species_list::AbstractVector{<:AbstractString}, eco_species_ids::AbstractVector;
                       params_dir::AbstractString=DEFAULT_PARAMS_DIR, priors_csv::AbstractString="", coefs_csv::AbstractString="",
                       vigour::Bool=true, verbose::Bool=true, fixed::HBAD3PConst=HBAD3PConst(vigour=vigour),
                       tables::BiomassSuccessionPlugin.BSTables=BiomassSuccessionPlugin.BSTables(), tables3=nothing,
                       wn::AbstractDict{String,Float64}=Dict{String,Float64}(), wn_median::Real=0.9)
  pdir = isempty(params_dir) ? DEFAULT_PARAMS_DIR : String(params_dir)
  t = tables3 === nothing ? load_3p_tables(pdir) : tables3
  grp_csv = joinpath(dirname(HBADPlugin.DEFAULT_COEFS_CSV), "species_group.csv")
  grp_map = Dict{String,String}(String(r.species) => String(r.grp) for r in eachrow(CSV.read(grp_csv, DataFrame)))
  skey = map_species_keys(species_list, t.keys; grp_map=grp_map)
  priors = HBADPlugin.load_priors(species_list; priors_csv=isempty(priors_csv) ? HBADPlugin.DEFAULT_PRIORS_CSV : priors_csv,
                                  coefs_csv=isempty(coefs_csv) ? HBADPlugin.DEFAULT_COEFS_CSV : coefs_csv)
  if verbose
    own = count(i -> skey[i] == species_list[i], eachindex(skey))
    @info "HBAD3P parameters ($(basename(pdir)), $(length(t.keys)) keys, farea=$(t.farea)): $(length(species_list)) species; own key=$own group key=$(length(species_list) - own)" params_dir=pdir
    @info "HBAD3P species -> keys" mapping = join(["$(s)=>$(skey[i])" for (i, s) in enumerate(species_list)], "  ")
  end
  return HBAD3PContext(priors, HBADPlugin.eco_gids(eco_species_ids), fixed, t.keys, t.sp, t.rec, t.glob, t.farea == 1, skey, pdir, tables, Dict{String,Float64}(wn), Float64(wn_median))
end

"""The tables (`keys, sp, rec, glob, farea, meta`, per key) of a parameter set whose species ARE the keys (the yaml roster); the input of `HBAD3PContext(...; tables3=)`."""
function tables_of(p::HBAD3PParams)
  p.KEYS == p.SPECIES_LIST || error("HBAD3P tables_of: the species of the set must be the keys")
  return (keys=copy(p.KEYS), sp=copy(p.SP), rec=p.REC, glob=p.G, farea=p.FAREA, meta=Dict{String,String}())
end

"""A copy of `ctx` with other field values (`with(ctx; fixed=...)`)."""
with(ctx::HBAD3PContext; kw...) = HBAD3PContext((get(kw, f, getfield(ctx, f)) for f in fieldnames(HBAD3PContext))...)

"""Throw unless `ctx` was built for exactly this species list and eco -> species table."""
function check_context(ctx::HBAD3PContext, species_list::AbstractVector{<:AbstractString}, eco_species_ids::AbstractVector)
  ctx.priors.species_list == species_list ||
    error("HBAD3P context was built for another species list ($(length(ctx.priors.species_list)) species) than the parameters' ($(length(species_list)) species): build one with HBAD3PContext(species_list, eco_species_ids)")
  ctx.eco_gids == HBADPlugin.eco_gids(eco_species_ids) ||
    error("HBAD3P context was built for another eco -> species table than the parameters' ECO_SPECIES_IDS: build one with HBAD3PContext(species_list, eco_species_ids)")
  return nothing
end

"""The fixed parameter set for the run's lists (no search: `rng`, `no_establishment` and the other keywords are accepted and ignored)."""
function generate_hbad3p_params(ctx::HBAD3PContext, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}; kw...)
  check_context(ctx, species_list, eco_species_ids)
  kix = Dict(k => i for (i, k) in enumerate(ctx.keys))
  return HBAD3PParams(eco_list, species_list, [UIntType.(ids) for ids in eco_species_ids], copy(ctx.species_key),
                      [ctx.sp[kix[k]] for k in ctx.species_key], ctx.rec, ctx.glob, ctx.farea ? 1 : 0)
end

"""Per-ecoregion resolved parameters of `params` under the context."""
function generate_hbad3p_eco_params(ctx::HBAD3PContext, params::HBAD3PParams)::Vector{HBAD3PEcoParams}
  check_context(ctx, params.SPECIES_LIST, params.ECO_SPECIES_IDS)
  out = Vector{HBAD3PEcoParams}(undef, length(params.ECO_SPECIES_IDS))
  for (e, gids) in enumerate(params.ECO_SPECIES_IDS)
    sp = SpPP[params.SP[Int(g)] for g in gids]
    allo = Allo[(p = ctx.priors.sp[Int(g)]; Allo(p.mq_a, p.mq_b)) for g in gids]
    out[e] = HBAD3PEcoParams(sp, allo, params.REC, params.G)
  end
  return out
end

"""Resolved parameters for the species of the context in order (eco-local index = global index): what a site sees in an ecoregion that lists all species in order. Independent of the eco table."""
function identity_eco_params(ctx::HBAD3PContext)::HBAD3PEcoParams
  n = length(ctx.priors.species_list); kix = Dict(k => i for (i, k) in enumerate(ctx.keys))
  return HBAD3PEcoParams(SpPP[ctx.sp[kix[ctx.species_key[g]]] for g in 1:n], Allo[(p = ctx.priors.sp[g]; Allo(p.mq_a, p.mq_b)) for g in 1:n], ctx.rec, ctx.glob)
end
