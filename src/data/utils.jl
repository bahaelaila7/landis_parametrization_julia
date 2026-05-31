using ..PanCore
export prepare_parametrization_data, get_site_sim_years, get_spinup_cohorts, make_spdf_dict, get_initial_cohorts, check_cohort_continuity, get_injection_cohorts
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
                                min_trees::Int=100, min_agb_frac::Float64=0.05)
  sp_stats = combine(groupby(df, [:species_symbol, :spgrpcd, :sftwd_hrdwd]),
                     nrow        => :n_rows,
                     :agb        => sum => :sp_agb)
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
      "GRP_$(row.spgrpcd)"
    else
      coalesce(row.sftwd_hrdwd, "H")
    end
  end

  df.effective_species = [get(sp_map, s, "H") for s in df.species_symbol]

  # Summary
  exact_list = sort([k for (k,v) in sp_map if v == k])
  group_list = sort(unique([v for (_,v) in sp_map if startswith(v,"GRP_")]))
  n_H = count(v == "H" for v in values(sp_map))
  n_S = count(v == "S" for v in values(sp_map))
  @info "Species tiers  ($(length(sp_map)) total → $(length(exact_list)) exact / $(length(group_list)) groups / H=$n_H S=$n_S)" exact=join(exact_list,", ") groups=join(group_list,", ")
end

function make_splots(df::DataFrame; eco::String="epa_l4", filter_species::Vector{String}=String[], by_subplot::Bool=false)::Tuple{DataFrame,Vector{String},Vector{String},Vector{Vector{Int}}}
  eco_field = begin
    if eco == "epa_l4"
      :epa_l4
    elseif eco == "epa_l3"
      :epa_l3
    else
      :ecosubcd
    end
  end

  assign_tiered_species!(df)
  if !isempty(filter_species)
    fs = Set(filter_species)
    filter!(row -> row.effective_species in fs, df)
  end
  species_field = :effective_species

  eco_vals = sort(unique(getproperty(df, eco_field)))
  eco_dict = Dict(eco => i for (i, eco) in enumerate(eco_vals))
  species_symbol_map_vals = sort(unique(getproperty(df, species_field)))
  species_symbol_map_dict = Dict(ssm => i for (i, ssm) in enumerate(species_symbol_map_vals))

  df.species_id = getindex.(Ref(species_symbol_map_dict), getproperty(df, species_field))
  df.eco_id = getindex.(Ref(eco_dict), getproperty(df, eco_field))
  #println(df)
  #
  #

  ddf = combine(groupby(df, :eco_id, sort=true)) do rows
    (; species_ids=[sort(unique(rows.species_id))])
  end

  #maps list species text
  #maps list eco_text
  #println(df.measdate)
  #println(typeof(df.measdate))

  if !(df.measdate[1] isa Dates.Date)
    df.measdate = Dates.DateTime.(df.measdate, Dates.dateformat"yyyy-mm-dd")
  end
  df.age_calc = df.age_calc .|> UIntType

  id_cols = by_subplot ? [:statecd, :unitcd, :countycd, :plot, :subp] : [:statecd, :unitcd, :countycd, :plot]
  base_fields = [:plt_cn, :statecd, :unitcd, :countycd, :plot, :eco_id, :measdate, :sim_year, :species_id, :effective_species, :age_calc]
  fields = by_subplot ? vcat(base_fields, [:subp]) : base_fields
  raw_plots = combine(groupby(df, fields, sort=false), nrow => :count, :agb => sum => :agb_sum)
  if by_subplot
    plots = raw_plots
  else
    subp_counts = combine(groupby(df, :plt_cn), :subp => (x -> length(unique(x))) => :subp_count)
    plots = leftjoin(raw_plots, subp_counts, on=:plt_cn)
    plots.agb_sum ./= plots.subp_count
  end

  start_measdates = combine(groupby(plots, id_cols, sort=false)) do rows
    (; start_measdate=[minimum(rows.measdate)])
  end
  plots_measdate = innerjoin(plots, start_measdates, on=id_cols)
  splots = sort!(plots_measdate, [:measdate, :statecd, :unitcd, :countycd, :plot, :age_calc, :species_id])

  splots.plot_id .= groupindices(groupby(splots, id_cols)) .|> UIntType

  eco_species_id_map = Dict((eco_id, species_id) => eco_species_id
                            for (eco_id, species_ids) in enumerate(ddf.species_ids)
                            for (eco_species_id, species_id) in enumerate(species_ids))
  splots.eco_species_id .= getindex.(Ref(eco_species_id_map), zip(splots.eco_id, splots.species_id))
  #println(effective_species_symbol_map_vals)
  #println(eco_vals)
  return splots, eco_vals, species_symbol_map_vals, ddf.species_ids

end
function get_injection_cohorts(splots::DataFrame)::DataFrame
  # Cohorts with birth_sim_year = sim_year - age_calc > 0 were born after
  # simulation start and are not in the initial conditions. Return one row per
  # cohort at its first FIA measurement year so the caller can inject it into
  # the simulation at the right time with the observed age and biomass.
  birth_sym = Int.(splots.sim_year) .- Int.(splots.age_calc)
  inject = splots[birth_sym .> 0, :]
  isempty(inject) && return select(inject, [:plot_id, :sim_year, :eco_species_id, :age_calc, :agb_sum])
  inject = transform(inject,
    [:sim_year, :age_calc] => ByRow((s, a) -> Int(s) - Int(a)) => :_birth_sym)
  first_app = combine(groupby(inject, [:plot_id, :eco_species_id, :_birth_sym]),
    :sim_year => minimum => :_inject_year)
  result = innerjoin(inject,
    rename(first_app, :_inject_year => :sim_year),
    on=[:plot_id, :eco_species_id, :_birth_sym, :sim_year])
  return select(result, [:plot_id, :sim_year, :eco_species_id, :age_calc, :agb_sum])
end

function get_spinup_cohorts(df::DataFrame)
  spinup_cohorts = df[df.year_deficit.<-1, :]
  spinup_cohorts = unique(select(spinup_cohorts, [:year_deficit, :plot_id, :eco_id, :eco_species_id]))
  spinup_cohorts = sort!(spinup_cohorts, [:year_deficit, :plot_id, :eco_id, :eco_species_id])
  return spinup_cohorts
end

function check_cohort_continuity(splots::DataFrame; age_tol::Int=0)
  T_md = eltype(splots.measdate)

  # Previous measdate per (plot_id, measdate) — visits identified by measdate, not sim_year.
  plot_prev_md = Dict{Tuple{UIntType, T_md}, T_md}()
  for gdf in groupby(sort(unique(select(splots, [:plot_id, :measdate])), [:plot_id, :measdate]), :plot_id)
    for i in 2:nrow(gdf)
      plot_prev_md[(gdf.plot_id[i], gdf.measdate[i])] = gdf.measdate[i-1]
    end
  end

  # sim_year per (plot_id, measdate) — used only for integer gap arithmetic.
  sy_map = Dict{Tuple{UIntType, T_md}, Int}()
  for r in eachrow(unique(select(splots, [:plot_id, :measdate, :sim_year])))
    sy_map[(r.plot_id, r.measdate)] = Int(r.sim_year)
  end

  # Age set per (plot_id, effective_species, measdate) — join by measdate, not sim_year.
  age_sets = Dict{Tuple{UIntType, String, T_md}, Set{Int}}()
  for r in eachrow(splots)
    key = (r.plot_id, r.effective_species, r.measdate)
    push!(get!(age_sets, key, Set{Int}()), Int(r.age_calc))
  end

  n_viol = 0
  for r in eachrow(splots)
    md  = r.measdate
    age = Int(r.age_calc)
    # Condition 1: no prior plot visit → always OK
    haskey(plot_prev_md, (r.plot_id, md)) || continue
    prev_md = plot_prev_md[(r.plot_id, md)]
    sy      = get(sy_map, (r.plot_id, md),      0)
    prev_sy = get(sy_map, (r.plot_id, prev_md), 0)
    gap     = sy - prev_sy        # integer sim_year gap, exact
    # Condition 1b: gap=0 (two visits in same sim_year step) → OK
    gap == 0 && continue
    # Condition 2: cohort born within the interval → OK
    age <= gap && continue
    # Condition 3: matching cohort at previous measdate → OK
    expected  = age - gap
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
      md  = r.measdate
      age = Int(r.age_calc)
      haskey(plot_prev_md, (r.plot_id, md)) || continue
      prev_md = plot_prev_md[(r.plot_id, md)]
      sy      = get(sy_map, (r.plot_id, md),      0)
      prev_sy = get(sy_map, (r.plot_id, prev_md), 0)
      gap     = sy - prev_sy
      gap == 0 && continue
      age <= gap && continue
      expected  = age - gap
      prev_ages = get(age_sets, (r.plot_id, r.effective_species, prev_md), Set{Int}())
      any(abs(a - expected) <= age_tol for a in prev_ages) && continue
      println("    plot_id=$(r.plot_id) sp=$(r.effective_species) md=$md sy=$sy age=$age gap=$gap expected=$expected prev_ages=$(sort(collect(prev_ages)))")
      shown += 1
    end
  end

  n_viol
end

function prepare_parametrization_data(; cohorts_db_path::String, filter_eco_field::String, eco_field::String, tablename::String, output_dir::String, skip_disturbances=true, spinup=false, by_subplot::Bool=false, filter_ecos::Vector{String}=String[], filter_plots::Vector{NTuple{4,Int}}=NTuple{4,Int}[], filter_species::Vector{String}=String[], RNG::Union{Nothing,Random.AbstractRNG})
  #cohorts_df = load_cohorts_sqlite(db_path, tablename; filter_ecos=filter_ecos)
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
	sql *= " AND plot_meas_num > 1 "
  end
  if length(filter_plots) > 0
    tuples_str = join(["($(s),$(u),$(c),$(p))" for (s,u,c,p) in filter_plots], ",")
    sql *= " AND (statecd, unitcd, countycd, plot) IN ($(tuples_str))"
  end
  println(sql)
  cohorts_df = DuckDB.execute(con, sql) |> DataFrame
  #DuckDB.close(con)
  println("Closing db. $(nrow(cohorts_df)) rows loaded.")

  @time splots, eco_list, species_list, eco_species_ids = make_splots(cohorts_df, eco=eco_field, filter_species=filter_species, by_subplot=by_subplot)
  mark_estab_year!(splots)

  n_viol = check_cohort_continuity(splots)
  println("Cohort continuity violations: $n_viol")

  return splots, eco_list, species_list, eco_species_ids

end
