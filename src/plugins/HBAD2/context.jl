# The immutable per-run context of the HBAD2 plugin: the fixed v2 tables, the species -> key mapping, the HBAD v1 priors (REPORTED allometric AGB and the
# AGB -> state translation of `add_cohort!` only) and the eco -> species table. Built once per (species list, eco table).

"""Kernel options. `vigour`: recruits / planted cohorts enter as 3 vigour sub-cohorts (zq = -sqrt3, 0, sqrt3; shares 1/6, 2/3, 1/6), else as one mean-field cohort."""
Base.@kwdef struct HBAD2Const
  vigour::Bool = true
end

struct HBAD2Context
  priors::HBADPlugin.HBADPriors          # v1 priors of the run's species (allometry mq_a / mq_b, translation)
  eco_gids::HBADPlugin.EcoGids
  fixed::HBAD2Const
  keys::Vector{String}                   # v2 keys in file order
  sp::Vector{SpP}                        # per key
  rec::Vector{RecP}                      # per key
  glob::GlobP
  species_key::Vector{String}            # per Pan species: its v2 key
  params_dir::String
  tables::BiomassSuccessionPlugin.BSTables
end

"""
    HBAD2Context(species_list, eco_species_ids; params_dir, priors_csv, coefs_csv, vigour, verbose, tables)

Load the v2 parameter files of `params_dir` (default `results/hbad2`, the v2g export), map every species of `species_list` to its key (logged) and resolve the v1 priors.
"""
function HBAD2Context(species_list::AbstractVector{<:AbstractString}, eco_species_ids::AbstractVector;
                      params_dir::AbstractString=DEFAULT_PARAMS_DIR, priors_csv::AbstractString="", coefs_csv::AbstractString="",
                      vigour::Bool=true, verbose::Bool=true, fixed::HBAD2Const=HBAD2Const(vigour=vigour),
                      tables::BiomassSuccessionPlugin.BSTables=BiomassSuccessionPlugin.BSTables())
  pdir = isempty(params_dir) ? DEFAULT_PARAMS_DIR : String(params_dir)
  t = load_v2_tables(pdir)
  grp_csv = joinpath(dirname(HBADPlugin.DEFAULT_COEFS_CSV), "species_group.csv")
  grp_map = Dict{String,String}(String(r.species) => String(r.grp) for r in eachrow(CSV.read(grp_csv, DataFrame)))
  skey = map_species_keys(species_list, t.keys; grp_map=grp_map)
  priors = HBADPlugin.load_priors(species_list; priors_csv=isempty(priors_csv) ? HBADPlugin.DEFAULT_PRIORS_CSV : priors_csv,
                                  coefs_csv=isempty(coefs_csv) ? HBADPlugin.DEFAULT_COEFS_CSV : coefs_csv)
  if verbose
    own = count(i -> skey[i] == species_list[i], eachindex(skey))
    @info "HBAD2 parameters ($(basename(pdir)), form=$(t.glob.form), $(length(t.keys)) keys): $(length(species_list)) species; own key=$own group key=$(length(species_list) - own)" params_dir=pdir
    @info "HBAD2 species -> keys" mapping = join(["$(s)=>$(skey[i])" for (i, s) in enumerate(species_list)], "  ")
  end
  return HBAD2Context(priors, HBADPlugin.eco_gids(eco_species_ids), fixed, t.keys, t.sp, t.rec, t.glob, skey, pdir, tables)
end

"""A copy of `ctx` with other field values (`with(ctx; fixed=...)`)."""
with(ctx::HBAD2Context; kw...) = HBAD2Context((get(kw, f, getfield(ctx, f)) for f in fieldnames(HBAD2Context))...)

"""Throw unless `ctx` was built for exactly this species list and eco -> species table."""
function check_context(ctx::HBAD2Context, species_list::AbstractVector{<:AbstractString}, eco_species_ids::AbstractVector)
  ctx.priors.species_list == species_list ||
    error("HBAD2 context was built for another species list ($(length(ctx.priors.species_list)) species) than the parameters' ($(length(species_list)) species): build one with HBAD2Context(species_list, eco_species_ids)")
  ctx.eco_gids == HBADPlugin.eco_gids(eco_species_ids) ||
    error("HBAD2 context was built for another eco -> species table than the parameters' ECO_SPECIES_IDS: build one with HBAD2Context(species_list, eco_species_ids)")
  return nothing
end

"""The fixed v2g parameter set for the run's lists (no search in stage 1: `rng`, `no_establishment` and the other keywords are accepted and ignored)."""
function generate_hbad2_params(ctx::HBAD2Context, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}; kw...)
  check_context(ctx, species_list, eco_species_ids)
  kix = Dict(k => i for (i, k) in enumerate(ctx.keys))
  return HBAD2Params(eco_list, species_list, [UIntType.(ids) for ids in eco_species_ids], copy(ctx.species_key),
                     [ctx.sp[kix[k]] for k in ctx.species_key], [ctx.rec[kix[k]] for k in ctx.species_key], ctx.glob)
end

"""Per-ecoregion resolved parameters of `params` under the context."""
function generate_hbad2_eco_params(ctx::HBAD2Context, params::HBAD2Params)::Vector{HBAD2EcoParams}
  check_context(ctx, params.SPECIES_LIST, params.ECO_SPECIES_IDS)
  out = Vector{HBAD2EcoParams}(undef, length(params.ECO_SPECIES_IDS))
  for (e, gids) in enumerate(params.ECO_SPECIES_IDS)
    sp = SpP[params.SP[Int(g)] for g in gids]
    rec = RecP[params.REC[Int(g)] for g in gids]
    allo = Allo[(p = ctx.priors.sp[Int(g)]; Allo(p.mq_a, p.mq_b)) for g in gids]
    out[e] = HBAD2EcoParams(sp, rec, allo, params.G)
  end
  return out
end
