# ---------------------------------------------------------------------------
# Figure: forest species composition by ecoregion (stacked bars) from the
# coalesced cohorts.duckdb. One subplot per ecoregion; a stacked bar per year
# (default 0/25/50), stacked by species. Species labelled with FIA common names.
#
# The coalesced cohorts table stores integer (eco_id, species_id) where species_id
# is ECO-LOCAL; we rebuild the sim's params from the scenario dir to resolve
# species_id → ECO_SPECIES_IDS[eco_id][species_id] → SPECIES_LIST → symbol, then
# map symbols to common names (REF_SPECIES / REF_SPECIES_GROUP).
# ---------------------------------------------------------------------------
function plot_species_composition(;
  cohorts_db::String,
  scenario_dir::String,          # exported LANDIS files (CoreSpeciesData.txt, SpeciesData.csv, SppEcoregionData.csv)
  fia_db::String,                # DB holding REF_SPECIES (+ REF_SPECIES_GROUP)
  output_path::String,
  years::Vector{Int}=[0, 25, 50],
  spp_eco_year::Int=0,
  relative::Bool=false,          # normalize each bar to fractions of total biomass
  species_order::Vector{String}=String[],  # canonical symbol order for stable colors across plots
)
  # 1. Reconstruct the sim's id scheme.
  core_sp_df = SpatialData.load_landis_core_species(joinpath(scenario_dir, "CoreSpeciesData.txt"))
  species_df = CSV.read(joinpath(scenario_dir, "SpeciesData.csv"), DataFrame)
  spp_eco_df = SpatialData.load_landis_spp_ecoregion(joinpath(scenario_dir, "SppEcoregionData.csv"); year=spp_eco_year)
  params = SpatialData.make_landis_params(core_sp_df, species_df, spp_eco_df)
  SPECIES_LIST = params.SPECIES_LIST
  ECO_LIST = params.ECO_LIST
  eco_species_ids = params.ECO_SPECIES_IDS

  # 2. Aggregate biomass per (eco_id, species_id, year).
  db = DuckDB.DB(cohorts_db)
  con = DuckDB.connect(db)
  agg = DuckDB.execute(
    con,
    """
  SELECT year, eco_id, species_id, SUM(biomass) AS biomass
  FROM cohorts WHERE year IN ($(join(years, ",")))
  GROUP BY year, eco_id, species_id
  ORDER BY year, eco_id, species_id
"""
  ) |> DataFrame
  DuckDB.close(db)
  isempty(agg) && error("No cohorts for years $(years) in $(cohorts_db)")
  present_years = sort(unique(Int.(agg.year)))
  miss = setdiff(years, present_years)
  isempty(miss) || @warn "requested years absent in cohorts.duckdb (skipped)" absent = miss

  # 3. Common-name labels + softwood/hardwood class (REF_SPECIES_GROUP.CLASS).
  fdb = DuckDB.DB(fia_db)
  fcon = DuckDB.connect(fdb)
  ref_sp = DuckDB.execute(fcon, "SELECT r.SPECIES_SYMBOL AS sym, r.COMMON_NAME AS cn, r.SFTWD_HRDWD AS sh FROM REF_SPECIES r ORDER BY r.SPCD") |> DataFrame
  grp = DataFrame(spgrpcd=Int[], name=String[], class=String[])
  try
    grp = DuckDB.execute(fcon, "SELECT g.SPGRPCD AS spgrpcd, g.NAME AS name, g.CLASS AS class FROM REF_SPECIES_GROUP g ORDER BY g.SPGRPCD") |> DataFrame
  catch e
    @warn "REF_SPECIES_GROUP unavailable; _GRP_ labels stay raw and their soft/hard defaults to hardwood" exception = e
  end
  DuckDB.close(fdb)
  common = Dict(uppercase(strip(String(r.sym))) => String(r.cn) for r in eachrow(ref_sp))
  sym_sh = Dict(uppercase(strip(String(r.sym))) => uppercase(strip(String(r.sh))) for r in eachrow(ref_sp) if !ismissing(r.sh))
  grpname = Dict(Int(r.spgrpcd) => String(r.name) for r in eachrow(grp))
  grpclass = Dict(Int(r.spgrpcd) => String(r.class) for r in eachrow(grp))

  # softwood (blue–green) vs hardwood (yellow–red):
  #   real species → REF_SPECIES.SFTWD_HRDWD ; _GRP_<spgrpcd> → REF_SPECIES_GROUP.CLASS ; _S/_H explicit.
  function is_soft(sym::AbstractString)
    s = uppercase(strip(sym))
    s == "_S" && return true
    s == "_H" && return false
    if startswith(s, "_GRP_")
      g = tryparse(Int, s[6:end])
      return startswith(uppercase(isnothing(g) ? "" : get(grpclass, g, "")), "S")
    end
    return get(sym_sh, s, "H") == "S"   # default (unknown) → hardwood
  end

  function label_for(sym::AbstractString)
    if startswith(sym, "_GRP_")
      g = tryparse(Int, sym[6:end])
      return (!isnothing(g) && haskey(grpname, g)) ? "Other $(grpname[g])" : "Other group $(sym[6:end])"
    elseif sym == "_H"
      return "Other hardwood"
    elseif sym == "_S"
      return "Other softwood"
    else
      cn = get(common, uppercase(strip(sym)), nothing)
      return isnothing(cn) ? sym : "$(cn) ($(sym))"
    end
  end

  # 4. id (eco-local) → global symbol → label; eco_id → name. Drop any out-of-range ids.
  neco = length(ECO_LIST)
  keep = [1 <= Int(r.eco_id) <= neco && 1 <= Int(r.species_id) <= length(eco_species_ids[Int(r.eco_id)])
          for r in eachrow(agg)]
  agg = agg[keep, :]
  agg.symbol = [string(SPECIES_LIST[Int(eco_species_ids[Int(r.eco_id)][Int(r.species_id)])]) for r in eachrow(agg)]
  agg.label = label_for.(agg.symbol)
  agg.eco = [ECO_LIST[Int(r.eco_id)] for r in eachrow(agg)]
  agg = combine(groupby(agg, [:eco, :year, :symbol, :label]), :biomass => sum => :biomass)

  # 5. Color & stack by each species' CANONICAL position so the same species gets the same
  #    color across different plots. Default order = SPECIES_LIST (identical across runs of the
  #    same scenario); pass `species_order` (a shared symbol list) for cross-scenario consistency.
  #    Any present symbol not in the order is appended deterministically (sorted) at the end.
  order = String[string(s) for s in (isempty(species_order) ? SPECIES_LIST : species_order)]
  seen = Set(order)
  for s in sort(unique(agg.symbol))
    s in seen || (push!(order, s); push!(seen, s))
  end
  # Split canonically into softwoods (blue→green) and hardwoods (yellow→red); shade each
  # species by its position WITHIN its class (over the full order, so shades are plot-stable).
  soft_all = [s for s in order if is_soft(s)]
  hard_all = [s for s in order if !is_soft(s)]
  soft_grad = CairoMakie.cgrad([:navy, :dodgerblue, :darkturquoise, :seagreen, :limegreen])
  hard_grad = CairoMakie.cgrad([:gold, :orange, :orangered, :red, :darkred])
  shade(i, n) = n <= 1 ? 0.5 : (i - 1) / (n - 1)
  color_of = Dict{String,CairoMakie.RGBAf}()
  for (i, s) in enumerate(soft_all)
    color_of[s] = CairoMakie.RGBAf(soft_grad[shade(i, length(soft_all))])
  end
  for (i, s) in enumerate(hard_all)
    color_of[s] = CairoMakie.RGBAf(hard_grad[shade(i, length(hard_all))])
  end
  # stack order: softwoods (bottom) then hardwoods, each in canonical order.
  stack_order = vcat(soft_all, hard_all)
  sym_idx = Dict(s => i for (i, s) in enumerate(stack_order))

  # 6. figure: one axis per ecoregion + shared legend column.
  ecos = sort(unique(agg.eco))
  yx = Dict(y => i for (i, y) in enumerate(present_years))
  ncols = max(1, ceil(Int, sqrt(length(ecos))))   # square-ish grid (4 ecoregions → 2×2)
  nrows = cld(length(ecos), ncols)
  f = CairoMakie.Figure(size=(360 * ncols + 340, 80 + 320 * nrows))
  CairoMakie.Label(f[0, 1:(ncols+1)],
    "Forest species composition by ecoregion" * (relative ? " (relative)" : " (biomass g/m²)");
    fontsize=16, font=:bold, tellwidth=false)

  for (ei, eco) in enumerate(ecos)
    r = cld(ei, ncols)
    c = mod1(ei, ncols)
    ax = CairoMakie.Axis(f[r, c]; title="Ecoregion: $(eco)", xlabel="year",
      ylabel=relative ? "biomass fraction" : "biomass (g/m²)",
      xticks=(collect(1:length(present_years)), string.(present_years)))
    sub = agg[agg.eco.==eco, :]
    ymap = relative ? Dict(row.year => row.ytot for row in eachrow(combine(groupby(sub, :year), :biomass => sum => :ytot))) : Dict()
    xs = Int[]
    ys = Float64[]
    stk = Int[]
    cs = CairoMakie.RGBAf[]
    for row in eachrow(sub)
      haskey(yx, Int(row.year)) || continue
      h = relative ? Float64(row.biomass) / max(ymap[row.year], eps()) : Float64(row.biomass)
      push!(xs, yx[Int(row.year)])
      push!(ys, h)
      push!(stk, sym_idx[row.symbol])
      push!(cs, color_of[row.symbol])
    end
    isempty(xs) || CairoMakie.barplot!(ax, xs, ys; stack=stk, color=cs)
  end

  # Legend: present species in stack order (softwoods then hardwoods), each with its fixed color.
  present = sort(unique(agg.symbol); by=s -> sym_idx[s])
  label_of = Dict(r.symbol => r.label for r in eachrow(agg))
  elems = [CairoMakie.PolyElement(color=color_of[s]) for s in present]
  CairoMakie.Legend(f[1:nrows, ncols+1], elems, [label_of[s] for s in present],
    "Species  (softwood: blue–green, hardwood: yellow–red)"; framevisible=false, labelsize=10)

  mkpath(dirname(output_path))
  CairoMakie.save(output_path, f)
  println("Wrote $output_path  ($(length(ecos)) ecoregions, $(length(present)) species, years $(present_years))")
  return output_path
end

function species_composition_main()
  prefix = joinpath("../poster/fl5_strat4/landis_90_2")
  plot_species_composition(
    cohorts_db=joinpath("../poster/fl5_strat4_prob_spinup/pan_outputs_90_2", "cohorts.duckdb"),
    scenario_dir=prefix,
    fia_db="../FIA/truth/fia_truth_r4.duckdb",
    output_path=joinpath("../poster/fl5_strat4_prob_spinup/pan_outputs_90_2", "species_composition.png"),
    years=[0, 25, 50],
  )
end

function landis_main()
  prefix = joinpath("/home/bahaa/Downloads/", "Extension-Biomass-Succession-master/testings/CoreV8.0-BiomassSuccession7.0/")
  simulate_spatial_landis(
    output_dir="./outputs/landis_test",
    initial_communities_tif=joinpath(prefix, "initial-communities.tif"),
    ecoregion_tif=joinpath(prefix, "ecoregions.tif"),
    initial_communities_csv=joinpath(prefix, "biomass-succession_InitialCommunities.csv"),
    core_species_data=joinpath(prefix, "CoreSpeciesData.txt"),
    spp_ecoregion_data=joinpath(prefix, "SppEcoregionData.csv"),
    species_data=joinpath(prefix, "SpeciesData.csv"),
    eco_ecocode_mapping=joinpath(prefix, "eco_ecocode_mapping.csv"),
    spp_eco_year=0,
    timehorizon_years=50,
    output_every_years=1,
  )
end
