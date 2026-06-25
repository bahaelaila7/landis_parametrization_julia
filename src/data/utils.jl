using ..PanCore
export prepare_parametrization_data, get_site_sim_years, get_spinup_cohorts, make_spdf_dict, get_initial_cohorts, check_cohort_continuity, get_injection_cohorts, build_padded_sim_years, print_cycle_coverage, build_cycle_map
import DuckDB
import Random
import StatsBase
using DataFrames
import Dates
import ....Parametrization.SPDFGroundTruth
import ....Parametrization.SPDFRecord


function make_spdf_dict(spdf::DataFrame, eco_species_ids::Vector{Vector{Int}})::Dict{Tuple{UIntType,UIntType},Dict{Int,SPDFGroundTruth}}
  #n_species = length(unique(spdf.species_id))
  return Dict( #{Int, Dict{Int,DataFrame}}()
    (plt_key.plot_id, plt_key.eco_id) => Dict( #( {Int, DataFrame}(
      year_key.sim_year => SPDFGroundTruth(keys=begin
          n_species = length(eco_species_ids[first(year_df).eco_id])
          #println("----")
          #println(eco_species_ids)
          #println(n_species)
          #println(year_df)
          #println(unique(year_df.eco_species_id))
          #println("----")
          a = falses(n_species)
          a[unique(year_df.eco_species_id)] .= true
          a
        end,
        records=Dict(
          sp_key.eco_species_id => begin
            nrow(sp_df) > 1 && @warn "more than 1 sp $(sp_df)"
            rec = last(sp_df)
            SPDFRecord(sp_agb_sum=rec.data_agb_sum, sp_age_cdf=rec.data_agbs_cdf)
          end

          for (sp_key, sp_df) in pairs(groupby(year_df, :eco_species_id, sort=false))
        )
      )
      for (year_key, year_df) in pairs(groupby(plt_df, [:sim_year], sort=false))
    )
    for (plt_key, plt_df) in pairs(groupby(spdf, [:plot_id, :eco_id], sort=false))
  )
end
function get_initial_cohorts(df::DataFrame)
  #return df[df.year_deficit.==0, :]
  return df[df.sim_year.==0, :]
end
function get_site_sim_years(df::DataFrame)::DataFrame
  return combine(groupby(df, [:plot_id, :eco_id])) do rows
    (; sim_years=[unique(sort(rows.sim_year))])
  end

end
function mark_estab_year!(df::DataFrame)
  #println(minimum(df.age_calc))
  #show(df[df.age_calc .< 0,:]) #.age_calc .= 1
  year_estab = df.measdate .- (df.age_calc .|> Dates.Year)
  oldest = minimum(year_estab)
  #last = maximum(df.start_measdate)
  #println("Oldest cohort established: $oldest")
  #println("First measurement date: $last")
  #df.year_deficit .= Dates.value.(Dates.Day.(year_estab .- last)) ./ 365.25 .|> round .|> Int
  df.year_deficit .= Dates.value.(Dates.Day.(year_estab .- df.start_measdate)) ./ 365.25 .|> round .|> Int
end
function assign_tiered_species!(df::DataFrame;
  max_exact::Int=12, max_group::Int=4,
  min_trees::Int=100, min_agb_frac::Float64=0.05,
  tree_stats::Union{Nothing,DataFrame}=nothing)
  if isnothing(tree_stats)
    sp_stats = combine(groupby(df, [:species_symbol, :spgrpcd, :sftwd_hrdwd]),
      nrow => :n_rows,
      :agb => sum => :sp_agb)
  else
    # Tier on RAW per-distinct-tree stats from curated_trees (count + summed max DRYBIO_AG),
    # grouped by (species, spgrpcd, sftwd_hrdwd) — not cohort-row counts / cohort agb. Cohort
    # species absent from tree_stats fall to the H/S catchall (get(sp_map, s, "_H") below).
    sp_stats = DataFrame(
      species_symbol=tree_stats.species_symbol,
      spgrpcd=tree_stats.spgrpcd,
      sftwd_hrdwd=tree_stats.sftwd_hrdwd,
      n_rows=Int.(tree_stats.n_trees),
      sp_agb=Float64.(tree_stats.sp_biomass))
  end
  total_agb = sum(sp_stats.sp_agb)
  sp_stats.agb_frac = sp_stats.sp_agb ./ max(total_agb, 1e-9)
  sort!(sp_stats, :n_rows, rev=true)

  exact_set = Set{String}()
  for row in eachrow(sp_stats)
    length(exact_set) >= max_exact && break
    row.n_rows >= min_trees && row.agb_frac >= min_agb_frac || continue
    push!(exact_set, row.species_symbol)
  end

  grp_set = Set{Int}()
  remaining = filter(row -> row.species_symbol ∉ exact_set, sp_stats)
  if nrow(remaining) > 0
    grp_stats = combine(groupby(remaining, :spgrpcd),
      :n_rows => sum => :grp_n,
      :sp_agb => sum => :grp_agb)
    grp_stats.grp_agb_frac = grp_stats.grp_agb ./ max(total_agb, 1e-9)
    sort!(grp_stats, :grp_n, rev=true)
    for row in eachrow(grp_stats)
      length(grp_set) >= max_group && break
      row.grp_n >= min_trees && row.grp_agb_frac >= min_agb_frac || continue
      push!(grp_set, row.spgrpcd)
    end
  end

  sp_map = Dict{String,String}()
  for row in eachrow(sp_stats)
    sp = row.species_symbol
    sp_map[sp] = if sp in exact_set
      sp
    elseif row.spgrpcd in grp_set
      "_GRP_$(row.spgrpcd)"
    else
      "_" * coalesce(row.sftwd_hrdwd, "H")
    end
  end

  df.effective_species = [get(sp_map, s, "_H") for s in df.species_symbol]

  # Summary
  exact_list = sort([k for (k, v) in sp_map if v == k])
  group_list = sort(unique([v for (_, v) in sp_map if startswith(v, "_GRP_")]))
  n_H = count(v == "_H" for v in values(sp_map))
  n_S = count(v == "_S" for v in values(sp_map))
  @info "Species tiers  ($(length(sp_map)) total → $(length(exact_list)) exact / $(length(group_list)) groups / H=$n_H S=$n_S)  [min_trees=$min_trees, min_agb_frac=$min_agb_frac]" exact = join(exact_list, ", ") groups = join(group_list, ", ")
end

# Split raw cohort-level df by site (plot or subplot), stratified by species.
# Each site goes to exactly one stratum (its rarest species) so the total val
# fraction stays close to val_frac. Returns (df_train, df_val).
function _stratified_split_df(df::DataFrame, id_cols::Vector{Symbol},
  val_frac::Float64, rng::Random.AbstractRNG)
  site_sp = unique(select(df, vcat(id_cols, [:effective_species])))
  sp_count = Dict(r.effective_species => r.n_sites
                  for r in eachrow(combine(groupby(site_sp, :effective_species), nrow => :n_sites)))
  site_strata = combine(groupby(site_sp, id_cols)) do rows
    idx = argmin(i -> get(sp_count, rows.effective_species[i], typemax(Int)), 1:nrow(rows))
    (; stratum=rows.effective_species[idx])
  end
  val_sites = vcat([
    begin
      shuffled = gdf[Random.shuffle(rng, 1:nrow(gdf)), id_cols]
      n_val = max(1, round(Int, nrow(gdf) * val_frac))
      shuffled[1:n_val, :]
    end for gdf in groupby(site_strata, :stratum)
  ]...)
  n_total = nrow(site_strata)
  n_val_out = nrow(val_sites)
  @info "Train/val split" n_train = n_total - n_val_out n_val = n_val_out val_pct = round(100 * n_val_out / n_total, digits=1)
  df_train = antijoin(df, val_sites, on=id_cols)
  df_val = semijoin(df, val_sites, on=id_cols)
  return df_train, df_val
end

# Tag each plot with :mixed_plot over its entire (loaded) history. true = mixed stand:
# neither hardwood nor softwood makes up more than `threshold` of the plot's aggregate
# biomass. false = one type dominates (≥ threshold). Returns df with the :mixed_plot column.
function with_mixed_plot(df::DataFrame; threshold::Float64=0.7)
  ks = [:statecd, :unitcd, :countycd, :plot]
  m = combine(groupby(df, ks)) do rows
    tot = sum(rows.agb)
    if tot <= 0
      (; mixed_plot=true)
    else
      hw = sum((a for (a, s) in zip(rows.agb, rows.sftwd_hrdwd) if coalesce(s, "") == "H"); init=0.0)
      sw = sum((a for (a, s) in zip(rows.agb, rows.sftwd_hrdwd) if coalesce(s, "") == "S"); init=0.0)
      (; mixed_plot=(max(hw, sw) / tot <= threshold))
    end
  end
  return leftjoin(df, m, on=ks)
end

function make_splots(df::DataFrame; eco::String="epa_l4", filter_species::Vector{String}=String[],
  by_subplot::Bool=false,
  val_frac::Float64=0.0, split_rng::Union{Nothing,Random.AbstractRNG}=nothing,
  min_trees::Int=100, min_agb_frac::Float64=0.05,
  stratify_eco_mixed::Bool=false,
  tree_stats::Union{Nothing,DataFrame}=nothing)
  # eco_field can be any column of the loaded cohorts (epa_l4/epa_l3/ecosubcd, or a curated
  # stratifier like land_use). Falls back to ecosubcd only for the legacy unspecified case.
  eco_field = if eco in ("epa_l4", "epa_l3", "ecosubcd", "land_use")
    Symbol(eco)
  elseif hasproperty(df, Symbol(eco))
    Symbol(eco)
  else
    :ecosubcd
  end
  hasproperty(df, eco_field) || error("make_splots: eco_field :$eco_field is not a column of the cohorts table")

  assign_tiered_species!(df; min_trees=min_trees, min_agb_frac=min_agb_frac, tree_stats=tree_stats)
  if !isempty(filter_species)
    fs = Set(filter_species)
    filter!(row -> row.effective_species in fs, df)
  end

  # Optionally stratify the ecoregion by stand mixedness: :eco is the base ecoregion,
  # :eco_mixed splits it into "<eco>_mixed" / "<eco>_pure" per (plot-level) mixed_plot. When
  # enabled the model fits params per eco_mixed; otherwise eco_mixed == the base ecoregion.
  df.eco = string.(getproperty(df, eco_field))
  df.eco_mixed = stratify_eco_mixed ?
                 df.eco .* ifelse.(coalesce.(df.mixed_plot, true), "_mixed", "_pure") :
                 df.eco

  eco_vals = sort(unique(df.eco_mixed))
  eco_dict = Dict(eco => i for (i, eco) in enumerate(eco_vals))
  species_symbol_map_vals = sort(unique(df.effective_species))
  species_symbol_map_dict = Dict(ssm => i for (i, ssm) in enumerate(species_symbol_map_vals))

  df.species_id = getindex.(Ref(species_symbol_map_dict), df.effective_species)
  df.eco_id = getindex.(Ref(eco_dict), df.eco_mixed)

  # eco→species_ids built from full df so train and val share the same vocabulary
  ddf = combine(groupby(df, :eco_id, sort=true)) do rows
    (; species_ids=[sort(unique(rows.species_id))])
  end
  eco_species_id_map = Dict((eco_id, species_id) => eco_species_id
                            for (eco_id, species_ids) in enumerate(ddf.species_ids)
                            for (eco_species_id, species_id) in enumerate(species_ids))

  if !(df.measdate[1] isa Dates.Date)
    df.measdate = Dates.DateTime.(df.measdate, Dates.dateformat"yyyy-mm-dd")
  end
  df.age_calc = df.age_calc .|> UIntType
  # Per-cohort partial-disturbance biomass-drop fraction (from curation; 0 where absent/un-attached).
  # The override-sync scales matched cohorts by (1-drop) at the disturbance year.
  df.disturbance_drop_pct = hasproperty(df, :disturbance_drop_pct) ?
    Float32.(coalesce.(df.disturbance_drop_pct, 0.0)) : zeros(Float32, nrow(df))

  id_cols = by_subplot ? [:statecd, :unitcd, :countycd, :plot, :subp] :
            [:statecd, :unitcd, :countycd, :plot]
  # NB: sim_year is deliberately NOT a grouping key. In curated_cohorts_landis sim_year is
  # anchored to each SUBPLOT's first measurement, so two subplots at the same plot visit can
  # carry different sim_year for the same (plot-level) measdate — which would split otherwise
  # identical species×age cohorts when by_subplot=false. We group on measdate (plot-level) and
  # recompute sim_year per id_cols below.
  base_fields = [:plt_cn, :statecd, :unitcd, :countycd, :plot, :eco_id, :measdate,
    :species_id, :effective_species, :age_calc]
  fields = by_subplot ? vcat(base_fields, [:subp]) : base_fields

  # Split right here — after mapping, before aggregation — so each half gets its
  # own contiguous plot_ids and is fully self-contained.
  do_split = val_frac > 0.0 && !isnothing(split_rng)
  df_train, df_val_raw = do_split ? _stratified_split_df(df, id_cols, val_frac, split_rng) :
                         (df, nothing)

  # Aggregate a raw cohort df into a splots DataFrame with contiguous plot_ids.
  function _agg(df_sub::DataFrame)
    raw = combine(groupby(df_sub, fields, sort=false), nrow => :count, :agb => sum => :agb_sum,
      [:agb, :disturbance_drop_pct] => _weighted_cohort_drop => :disturbance_drop_pct)
    plots = if by_subplot
      raw
    else
      sc = combine(groupby(df_sub, :plt_cn), :subp => (x -> length(unique(x))) => :subp_count)
      p = leftjoin(raw, sc, on=:plt_cn)
      p.agb_sum ./= p.subp_count
      p
    end
    sm = combine(groupby(plots, id_cols, sort=false)) do rows
      (; start_measdate=[minimum(rows.measdate)])
    end
    sp = sort!(innerjoin(plots, sm, on=id_cols),
      [:measdate, :statecd, :unitcd, :countycd, :plot, :age_calc, :species_id])
    # sim_year anchored to id_cols' first measurement (per plot, or per subplot if by_subplot).
    sp.sim_year = round.(Int, Dates.value.(Dates.Day.(sp.measdate .- sp.start_measdate)) ./ 365.25)
    sp.plot_id .= groupindices(groupby(sp, id_cols)) .|> UIntType
    sp.eco_species_id .= getindex.(Ref(eco_species_id_map), zip(sp.eco_id, sp.species_id))
    sp
  end

  splots_train = _agg(df_train)
  splots_val = isnothing(df_val_raw) ? nothing : _agg(df_val_raw)
  return splots_train, eco_vals, species_symbol_map_vals, ddf.species_ids, splots_val
end

function build_padded_sim_years(splots_subset::DataFrame, n_plots_total::Int)
  # Returns a NamedTuple (sim_years = padded_vector) where padded_vector[plot_id]
  # gives the correct sim_years for that plot. Plots not in the subset get Int[].
  # This preserves the site_sim_years.sim_years[mapcode] indexing convention used
  # in fit_params, which relies on plot_id being a 1-based index into the vector.
  ssy = get_site_sim_years(splots_subset)
  padded = Vector{Vector{Int}}([Int[] for _ in 1:n_plots_total])
  for row in eachrow(ssy)
    padded[Int(row.plot_id)] = row.sim_years
  end
  (sim_years=padded,)
end

# Biomass-weighted aggregation of per-row cohort disturbance drop. would_be_i = agb_i/(1-drop_i);
# aggregated drop = 1 - Σagb / Σwould_be (a ratio → unaffected by later subplot normalization).
function _weighted_cohort_drop(agb, drop)
  wb = 0.0; s = 0.0
  @inbounds for k in eachindex(agb)
    a = Float64(agb[k]); s += a
    wb += a / (1.0 - clamp(Float64(drop[k]), 0.0, 0.999))
  end
  wb > 0.0 ? Float32(1.0 - s / wb) : 0.0f0
end

const _INJECT_COLS = [:plot_id, :sim_year, :eco_species_id, :age_calc, :agb_sum, :disturbance_drop_pct]

function get_injection_cohorts(splots::DataFrame; all_cohorts::Bool=false)::DataFrame
  # all_cohorts=true: return the FULL observed state at every measurement year
  # (one row per observed cohort), so the caller can override the simulator
  # entirely — replacing each measured site's cohorts with the empirical ones.
  # Used for the override sanity test and the injection-noise sensitivity test.
  if all_cohorts
    return select(splots, _INJECT_COLS)
  end
  # Cohorts with birth_sim_year = sim_year - age_calc > 0 were born after
  # simulation start and are not in the initial conditions. Return one row per
  # cohort at its first FIA measurement year so the caller can inject it into
  # the simulation at the right time with the observed age and biomass.
  birth_sym = Int.(splots.sim_year) .- Int.(splots.age_calc)
  inject = splots[birth_sym.>0, :]
  isempty(inject) && return select(inject, _INJECT_COLS)
  inject = transform(inject,
    [:sim_year, :age_calc] => ByRow((s, a) -> Int(s) - Int(a)) => :_birth_sym)
  first_app = combine(groupby(inject, [:plot_id, :eco_species_id, :_birth_sym]),
    :sim_year => minimum => :_inject_year)
  result = innerjoin(inject,
    rename(first_app, :_inject_year => :sim_year),
    on=[:plot_id, :eco_species_id, :_birth_sym, :sim_year])
  return select(result, _INJECT_COLS)
end

function get_spinup_cohorts(df::DataFrame)
  spinup_cohorts = df[df.year_deficit.<-1, :]
  spinup_cohorts = unique(select(spinup_cohorts, [:year_deficit, :plot_id, :eco_id, :eco_species_id]))
  spinup_cohorts = sort!(spinup_cohorts, [:year_deficit, :plot_id, :eco_id, :eco_species_id])
  return spinup_cohorts
end

# Diagnostic for the tier-4 (population-level) loss: bucket each [sub]plot measurement into
# fixed-width calendar cycles measured from the earliest measdate, and report how many land
# in each — overall and per ecoregion — so we can confirm the population snapshots are
# well-sampled and the first/last cycles aren't too thin to be representative.
function print_cycle_coverage(splots::DataFrame; cycle_years::Real=10, eco_list::Union{Nothing,Vector{String}}=nothing)
  meas = unique(select(splots, [:plot_id, :eco_id, :measdate]))
  epoch = minimum(meas.measdate)
  yrs = Dates.value.(Dates.Day.(meas.measdate .- epoch)) ./ 365.25
  meas.cycle = floor.(Int, yrs ./ cycle_years)
  ncyc = maximum(meas.cycle) + 1
  base_year = Dates.year(epoch)

  println("\n=== Cycle coverage: $(cycle_years)-yr buckets from $(epoch) ===")
  println("$(nrow(meas)) [sub]plot-measurements | $(length(unique(meas.plot_id))) plots | $(length(unique(meas.eco_id))) ecoregions | $(ncyc) cycles")

  overall = sort!(combine(groupby(meas, :cycle), nrow => :n), :cycle)
  mx = maximum(overall.n)
  thin = max(5.0, 0.25 * (sum(overall.n) / ncyc))   # flag cycles below 25% of the mean (or < 5)
  println("\nOverall (⚠ = thin, < $(round(Int, thin))):")
  for r in eachrow(overall)
    lo = base_year + Int(round(r.cycle * cycle_years))
    hi = lo + Int(round(cycle_years))
    bar = repeat("█", clamp(round(Int, 40 * r.n / mx), 0, 40))
    println("  cycle $(lpad(r.cycle, 2)) [$(lo)–$(hi)): $(lpad(r.n, 5))  $(bar)$(r.n < thin ? "  ⚠" : "")")
  end

  println("\nPer ecoregion × cycle (counts; 0 = unsampled):")
  ec = combine(groupby(meas, [:eco_id, :cycle]), nrow => :n)
  wide = sort!(unstack(ec, :eco_id, :cycle, :n; fill=0), :eco_id)
  if !isnothing(eco_list)
    wide.eco_name = [get(eco_list, Int(id), "?") for id in wide.eco_id]
    select!(wide, :eco_id, :eco_name, Not([:eco_id, :eco_name]))
  end
  show(wide; allrows=true, allcols=true)
  println("\n")
  return wide
end

# Maps each (plot_id, sim_year) measurement to a 1-based calendar cycle index, using the
# same epoch (earliest measdate) and width as print_cycle_coverage. Used by the tier-4
# population loss to bucket measurements into calendar snapshots. Returns (map, n_cycles).
function build_cycle_map(splots::DataFrame; cycle_years::Real=8)
  meas = unique(select(splots, [:plot_id, :sim_year, :measdate]))
  epoch = minimum(meas.measdate)
  cmap = Dict{Tuple{Int,Int},Int}()
  n_cycles = 0
  for r in eachrow(meas)
    yrs = Dates.value(Dates.Day(r.measdate - epoch)) / 365.25
    c = floor(Int, yrs / cycle_years) + 1            # 1-based cycle index
    cmap[(Int(r.plot_id), Int(r.sim_year))] = c
    c > n_cycles && (n_cycles = c)
  end
  return cmap, n_cycles
end

function check_cohort_continuity(splots::DataFrame; age_tol::Int=0)
  T_md = eltype(splots.measdate)

  # Previous measdate per (plot_id, measdate) — visits identified by measdate, not sim_year.
  plot_prev_md = Dict{Tuple{UIntType,T_md},T_md}()
  for gdf in groupby(sort(unique(select(splots, [:plot_id, :measdate])), [:plot_id, :measdate]), :plot_id)
    for i in 2:nrow(gdf)
      plot_prev_md[(gdf.plot_id[i], gdf.measdate[i])] = gdf.measdate[i-1]
    end
  end

  # sim_year per (plot_id, measdate) — used only for integer gap arithmetic.
  sy_map = Dict{Tuple{UIntType,T_md},Int}()
  for r in eachrow(unique(select(splots, [:plot_id, :measdate, :sim_year])))
    sy_map[(r.plot_id, r.measdate)] = Int(r.sim_year)
  end

  # Age set per (plot_id, effective_species, measdate) — join by measdate, not sim_year.
  age_sets = Dict{Tuple{UIntType,String,T_md},Set{Int}}()
  for r in eachrow(splots)
    key = (r.plot_id, r.effective_species, r.measdate)
    push!(get!(age_sets, key, Set{Int}()), Int(r.age_calc))
  end

  n_viol = 0
  for r in eachrow(splots)
    md = r.measdate
    age = Int(r.age_calc)
    # Condition 1: no prior plot visit → always OK
    haskey(plot_prev_md, (r.plot_id, md)) || continue
    prev_md = plot_prev_md[(r.plot_id, md)]
    sy = get(sy_map, (r.plot_id, md), 0)
    prev_sy = get(sy_map, (r.plot_id, prev_md), 0)
    gap = sy - prev_sy        # integer sim_year gap, exact
    # Condition 1b: gap=0 (two visits in same sim_year step) → OK
    gap == 0 && continue
    # Condition 2: cohort born within the interval → OK
    age <= gap && continue
    # Condition 3: matching cohort at previous measdate → OK
    expected = age - gap
    prev_ages = get(age_sets, (r.plot_id, r.effective_species, prev_md), Set{Int}())
    any(abs(a - expected) <= age_tol for a in prev_ages) && continue
    n_viol += 1
  end

  @info "check_cohort_continuity: $n_viol violations / $(nrow(splots)) rows  (age_tol=$(age_tol))"

  if n_viol > 0
    println("  sample violations (first 10):")
    shown = 0
    for r in eachrow(splots)
      shown >= 10 && break
      md = r.measdate
      age = Int(r.age_calc)
      haskey(plot_prev_md, (r.plot_id, md)) || continue
      prev_md = plot_prev_md[(r.plot_id, md)]
      sy = get(sy_map, (r.plot_id, md), 0)
      prev_sy = get(sy_map, (r.plot_id, prev_md), 0)
      gap = sy - prev_sy
      gap == 0 && continue
      age <= gap && continue
      expected = age - gap
      prev_ages = get(age_sets, (r.plot_id, r.effective_species, prev_md), Set{Int}())
      any(abs(a - expected) <= age_tol for a in prev_ages) && continue
      println("    plot_id=$(r.plot_id) sp=$(r.effective_species) md=$md sy=$sy age=$age gap=$gap expected=$expected prev_ages=$(sort(collect(prev_ages)))")
      shown += 1
    end
  end

  n_viol
end

# Keep only the cohort rows whose FIA plot location falls inside the supplied shapefile.
# Plot coordinates come from the PLOT table (LON/LAT, treated as EPSG:4326); the shapefile is
# reprojected to 4326 and a point-in-polygon test selects plots. Ecoregion assignment is
# untouched — this only restricts WHICH plots are used for calibration.
function filter_cohorts_by_extent(con, cohorts_df::DataFrame, shapefile_path::String; plot_table::String="PLOT")
  isfile(shapefile_path) || error("filter_extent shapefile not found: $shapefile_path")
  plot_keys = [:statecd, :unitcd, :countycd, :plot]
  plots = unique(select(cohorts_df, plot_keys))
  DuckDB.register_data_frame(con, plots, "extent_plots")
  coords = DuckDB.execute(con, """
    SELECT p.statecd AS statecd, p.unitcd AS unitcd, p.countycd AS countycd, p.plot AS plot,
           ANY_VALUE(p.LAT) AS lat, ANY_VALUE(p.LON) AS lon
    FROM $(plot_table) p
    JOIN extent_plots e
      ON e.statecd  = p.statecd AND e.unitcd = p.unitcd
     AND e.countycd = p.countycd AND e.plot  = p.plot
    WHERE p.LAT IS NOT NULL AND p.LON IS NOT NULL
    GROUP BY p.statecd, p.unitcd, p.countycd, p.plot
  """) |> DataFrame
  DuckDB.execute(con, "DROP VIEW IF EXISTS extent_plots")
  println("  filter_extent: $(nrow(plots)) loaded plots, $(nrow(coords)) with coords from $(plot_table)")
  if isempty(coords)
    error("filter_extent: no plot coordinates found — does $(plot_table) exist in this DB with LAT/LON for the loaded plots?")
  end
  println("  plot coords: lon [$(round(minimum(coords.lon),digits=4)), $(round(maximum(coords.lon),digits=4))], lat [$(round(minimum(coords.lat),digits=4)), $(round(maximum(coords.lat),digits=4))]")

  # Union all shapefile geometries, reprojected to EPSG:4326 (lon/lat) to match plot coords.
  geom = AG.read(shapefile_path) do ds
    layer = AG.getlayer(ds, 0)
    src_srs = AG.getspatialref(layer)
    acc = nothing
    for feat in layer
      g = AG.getgeom(feat)
      g === nothing && continue
      acc = isnothing(acc) ? AG.clone(g) : AG.union(acc, g)
    end
    isnothing(acc) && error("filter_extent shapefile has no geometries: $shapefile_path")
    # Only reproject when the shapefile is PROJECTED. A geographic shapefile already stores
    # coords as (X=lon, Y=lat), matching FIA LON/LAT; reprojecting 4326→4326 just triggers
    # GDAL's authority lat,lon axis order and silently swaps them. PROJ4 "+proj=longlat" target
    # forces traditional lon,lat output for the projected case.
    is_proj = false
    if !isnothing(src_srs)
      try
        is_proj = AG.isprojected(src_srs)
      catch
        is_proj = false
      end
    end
    if is_proj
      AG.createcoordtrans(src_srs, AG.importPROJ4("+proj=longlat +datum=WGS84 +no_defs")) do ct
        AG.transform!(acc, ct)
      end
    end
    acc
  end

  try
    env = AG.envelope(geom)
    println("  polygon bbox (lon/lat after reproject): lon [$(round(env.MinX,digits=4)), $(round(env.MaxX,digits=4))], lat [$(round(env.MinY,digits=4)), $(round(env.MaxY,digits=4))]")
  catch e
    @warn "could not compute polygon envelope" exception = e
  end

  inside = Set{NTuple{4,Int}}()
  for r in eachrow(coords)
    AG.contains(geom, AG.createpoint(Float64(r.lon), Float64(r.lat))) &&
      push!(inside, (Int(r.statecd), Int(r.unitcd), Int(r.countycd), Int(r.plot)))
  end
  println("  plots inside polygon: $(length(inside)) of $(nrow(coords))")

  keep = [(Int(r.statecd), Int(r.unitcd), Int(r.countycd), Int(r.plot)) in inside for r in eachrow(cohorts_df)]
  return cohorts_df[keep, :]
end

function prepare_parametrization_data(; cohorts_db_path::String, filter_eco_field::String, eco_field::String, tablename::String, output_dir::String, skip_disturbances=true, spinup=false, by_subplot::Bool=false, val_frac::Float64=0.0, split_rng::Union{Nothing,Random.AbstractRNG}=nothing, min_trees::Int=100, min_agb_frac::Float64=0.05, stratify_eco_mixed::Bool=false, filter_extent::Union{Nothing,String}=nothing, filter_ecos::Vector{String}=String[], filter_plots::Vector{NTuple{4,Int}}=NTuple{4,Int}[], filter_species::Vector{String}=String[], filter_planted::Bool=false, RNG::Union{Nothing,Random.AbstractRNG})
  println("Connecting to: $(cohorts_db_path) ")
  con = DuckDB.connect(DuckDB.DB(cohorts_db_path))
  println("Creating index if necessary")
  DuckDB.execute(con, "CREATE INDEX IF NOT EXISTS PLT_ECO_IDX_$(eco_field) ON $(tablename)($(eco_field));")
  sql = "SELECT * FROM $(tablename) WHERE true"
  if length(filter_ecos) > 0
    sql *= " AND $(filter_eco_field) in ('$(join(filter_ecos,"','"))')"
  end
  if skip_disturbances
    sql *= " AND subp_has_dstrb= false"
  end
  if !spinup
    if by_subplot
      sql *= " AND subp_meas_num > 1 "
    else
      sql *= " AND plot_meas_num > 1 "
    end
  end
  if length(filter_plots) > 0
    tuples_str = join(["($(s),$(u),$(c),$(p))" for (s, u, c, p) in filter_plots], ",")
    sql *= " AND (statecd, unitcd, countycd, plot) IN ($(tuples_str))"
  end
  if filter_planted
    sql *= " AND (statecd, unitcd, countycd, plot, subp) IN (SELECT statecd, unitcd, countycd, plot, subp FROM $(tablename) WHERE intro_type = 'planted')"
  end
  println(sql)
  cohorts_df = DuckDB.execute(con, sql) |> DataFrame
  println("$(nrow(cohorts_df)) cohort rows loaded.")

  # Restrict to plots whose location falls inside the supplied shapefile (eco_field still
  # decides each plot's ecoregion). Applied before any other per-plot processing.
  if !isnothing(filter_extent)
    cohorts_df = filter_cohorts_by_extent(con, cohorts_df, filter_extent)
    println("After filter_extent ($(basename(filter_extent))): $(nrow(cohorts_df)) rows, $(length(unique(zip(cohorts_df.statecd,cohorts_df.unitcd,cohorts_df.countycd,cohorts_df.plot)))) plots")
    isempty(cohorts_df) && error("filter_extent kept 0 plots — compare the 'plot coords' and 'polygon bbox' ranges printed above. If they don't overlap it's a CRS/axis issue: try pre-reprojecting the shapefile with `ogr2ogr -t_srs EPSG:4326 out.shp $(filter_extent)`.")
  end

  # Tag each plot mixed/pure over its entire loaded history (used to stratify ecoregions).
  if stratify_eco_mixed
    cohorts_df = with_mixed_plot(cohorts_df)
    println("Plots tagged mixed_plot: $(sum(unique(select(cohorts_df, [:statecd,:unitcd,:countycd,:plot,:mixed_plot])).mixed_plot)) mixed of $(length(unique(zip(cohorts_df.statecd,cohorts_df.unitcd,cohorts_df.countycd,cohorts_df.plot))))")
  end

  # Per-species RAW tree stats from curated_trees, restricted to the loaded subplots, for
  # species tiering (cohort-row counts under-count trees). Per DISTINCT tree (full TREE_ID)
  # take max(DRYBIO_AG) — a tree measured multiple times counts once, biomass is its max, not
  # a sum over visits — then per (species, spgrpcd, sftwd_hrdwd): COUNT distinct trees and SUM
  # the per-tree maxes. spgrpcd/sftwd_hrdwd are carried so assign_tiered_species! can group on them.
  subplots_df = unique(select(cohorts_df, [:statecd, :unitcd, :countycd, :plot, :subp]))
  DuckDB.register_data_frame(con, subplots_df, "loaded_subplots")
  tree_stats = DuckDB.execute(con, """
    WITH tree_max AS (
      SELECT t.STATECD, t.UNITCD, t.COUNTYCD, t.PLOT, t.SUBP, t.TREE,
             max(t.SPCD)      AS spcd,
             max(t.SPGRPCD)   AS spgrpcd,
             max(t.DRYBIO_AG) AS drybio_max
      FROM curated_trees t
      JOIN loaded_subplots ls
        ON ls.statecd = t.STATECD AND ls.unitcd = t.UNITCD AND ls.countycd = t.COUNTYCD
       AND ls.plot = t.PLOT AND ls.subp = t.SUBP
      WHERE t.STATUSCD = 1 AND t.DRYBIO_AG IS NOT NULL
      GROUP BY t.STATECD, t.UNITCD, t.COUNTYCD, t.PLOT, t.SUBP, t.TREE
    )
    SELECT r.SPECIES_SYMBOL AS species_symbol,
           tm.spgrpcd        AS spgrpcd,
           r.SFTWD_HRDWD     AS sftwd_hrdwd,
           COUNT(*)::BIGINT  AS n_trees,
           SUM(tm.drybio_max)::DOUBLE AS sp_biomass
    FROM tree_max tm
    JOIN REF_SPECIES r ON r.SPCD = tm.spcd
    GROUP BY r.SPECIES_SYMBOL, tm.spgrpcd, r.SFTWD_HRDWD
  """) |> DataFrame
  println("Tree stats: $(nrow(tree_stats)) (species,spgrpcd,sftwd) rows from curated_trees over $(nrow(subplots_df)) subplots.")

  @time splots, eco_list, species_list, eco_species_ids, splots_val =
    make_splots(cohorts_df, eco=eco_field, filter_species=filter_species,
      by_subplot=by_subplot, val_frac=val_frac, split_rng=split_rng,
      min_trees=min_trees, min_agb_frac=min_agb_frac,
      stratify_eco_mixed=stratify_eco_mixed, tree_stats=tree_stats)
  mark_estab_year!(splots)
  isnothing(splots_val) || mark_estab_year!(splots_val)

  n_viol = check_cohort_continuity(splots)
  println("Cohort continuity violations: $n_viol")

  return splots, eco_list, species_list, eco_species_ids, splots_val
end
