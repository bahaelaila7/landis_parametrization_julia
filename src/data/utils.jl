using ..PanCore
export DataConfig, with_data, prepare_parametrization_data, get_site_sim_years, get_spinup_cohorts, make_spdf_dict, get_initial_cohorts, check_cohort_continuity, get_injection_cohorts, build_padded_sim_years, print_cycle_coverage, build_cycle_map
import DuckDB
import CSV
import ArchGDAL
const AG = ArchGDAL
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
            SPDFRecord(sp_agb_sum=rec.data_agb_sum, sp_age_cdf=rec.data_agbs_cdf, sp_age_agb=rec.data_age_agb)
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
# shade_tier (species_symbol → "LST"/"HST") splits the GROUPING key from (spgrpcd | sftwd) to
# (spgrpcd | sftwd) × tier, so `_GRP_<g>` → `_GRP_<g>_<tier>` and `_H`/`_S` → `_<sftwd>_<tier>`;
# the same size thresholds re-apply to the finer key (species shift down to the catch-all when a
# per-tier group is too small). nothing ⇒ old single-key behaviour. Exact species are unaffected.
function assign_tiered_species!(df::DataFrame;
  max_exact::Int=12, max_group::Int=4,
  min_trees::Int=100, min_agb_frac::Float64=0.05,
  tree_stats::Union{Nothing,DataFrame}=nothing,
  shade_tier::Union{Nothing,Dict{String,String}}=nothing,
  on_missing::Symbol=:warn)
  tierof(sp) = shade_tier === nothing ? "" : get(shade_tier, uppercase(strip(String(sp))), "NA")
  suffix(t) = isempty(t) ? "" : "_" * t
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

  grp_set = Set{Tuple{Int,String}}()             # qualifying (spgrpcd, tier) groups
  remaining = filter(row -> row.species_symbol ∉ exact_set, sp_stats)
  if nrow(remaining) > 0
    remaining.tier = [tierof(s) for s in remaining.species_symbol]
    grp_stats = combine(groupby(remaining, [:spgrpcd, :tier]),
      :n_rows => sum => :grp_n,
      :sp_agb => sum => :grp_agb)
    grp_stats.grp_agb_frac = grp_stats.grp_agb ./ max(total_agb, 1e-9)
    sort!(grp_stats, :grp_n, rev=true)
    for row in eachrow(grp_stats)
      length(grp_set) >= max_group && break
      row.grp_n >= min_trees && row.grp_agb_frac >= min_agb_frac || continue
      push!(grp_set, (row.spgrpcd, row.tier))
    end
  end

  shade_tier === nothing || _note_missing(on_missing, "species absent from the shade-tier table (tier \"NA\")",
    [sp for sp in sp_stats.species_symbol if !haskey(shade_tier, uppercase(strip(String(sp))))])
  _note_missing(on_missing, "species without a softwood/hardwood class (treated as \"H\")",
    [sp for (sp, sh, g) in zip(sp_stats.species_symbol, sp_stats.sftwd_hrdwd, sp_stats.spgrpcd) if ismissing(sh) && !(sp in exact_set)])
  sp_map = Dict{String,String}()
  for row in eachrow(sp_stats)
    sp = row.species_symbol; t = tierof(sp)
    sp_map[sp] = if sp in exact_set
      sp
    elseif (row.spgrpcd, t) in grp_set
      "_GRP_$(row.spgrpcd)$(suffix(t))"
    else
      "_" * coalesce(row.sftwd_hrdwd, "H") * suffix(t)
    end
  end

  _note_missing(on_missing, "cohort species absent from the tree statistics (mapped to the \"_H\" catch-all)",
    [s for s in unique(df.species_symbol) if !haskey(sp_map, s)])
  df.effective_species = [get(sp_map, s, "_H") for s in df.species_symbol]

  # Summary
  exact_list = sort([k for (k, v) in sp_map if v == k])
  group_list = sort(unique([v for (_, v) in sp_map if startswith(v, "_")]))   # _GRP_* and _H/_S catch-alls (± tier)
  @info "Species tiers  ($(length(sp_map)) total → $(length(exact_list)) exact / $(length(group_list)) groups$(shade_tier === nothing ? "" : " × shade-tier"))  [min_trees=$min_trees, min_agb_frac=$min_agb_frac]" exact = join(exact_list, ", ") groups = join(group_list, ", ")
end

# Split raw cohort-level df by site (plot or subplot), stratified by eco×lu × species.
# Each site goes to exactly one stratum — (its eco_id, its rarest species) — so the val fraction
# stays close to val_frac WITHIN every ecoregion×land-use, not just globally per species (the
# per-cell normalization needs each eco×lu×sp cell represented in both halves). Returns (df_train, df_val).
# Species-balanced TRAINING set. The proportional split hands train a pine-heavy sample (commercial pines dominate
# plantations), so cap the train plots per (eco cell × dominant species); the surplus is NOT discarded but moved to
# val/test (alternating, 2-way ⇒ all to val), so it still measures transfer. Dominant species = largest summed AGB at
# the plot's LAST visit (ties by name). Order = each site's normalized position in its split-stratum shuffle (no new
# RNG draws, no spatial bias toward low FIPS). cap: Int, or "auto" = max(train_cap_min, ceil(train_cap_k × median
# stratum size in that eco cell)). Returns (surplus_val_sites, surplus_test_sites).
_median(v) = (w = sort(v); n = length(w); isodd(n) ? Float64(w[(n+1)÷2]) : (w[n÷2] + w[n÷2+1]) / 2)

"""
    DataOutputs(split_dom, species_map)

The by-products of data preparation that the run writes next to its results (returned by `make_splots` / `prepare_parametrization_data` as the 7th value):
`split_dom` = id_cols + eco_id + dom of every plot when a species-balanced train cap ran (else `nothing`; for split_plots.csv), `species_map` =
species_symbol → effective_species of this run's tiering (species_map.csv).
"""
struct DataOutputs
  split_dom::Union{Nothing,DataFrame}
  species_map::DataFrame
end

"""
    CycleOptions(; use_fia_cycle=false, merge=3)

How measurements are bucketed into calendar cycles by `build_cycle_map`. `use_fia_cycle`: take cycles from the FIA inventory `cycle` column
verbatim (else computed `cycle_years`-wide buckets from the first visit); `merge`: merge this many consecutive FIA cycles into one group
(1 = none; e.g. 3 gives {7,8,9},{10,11,12}); a plot measured in more than one cycle of a group contributes a duplicate measurement to that group.
"""
Base.@kwdef struct CycleOptions
  use_fia_cycle::Bool = false
  merge::Int = 3
end

"""
    DataConfig(; cohorts_db_path, tablename, eco_field, ...)

EVERYTHING data preparation reads, as one concretely typed immutable struct (built from a yaml by `Pan.data_config`, the one place; tools build
it with keywords). `prepare_parametrization_data(dc)` is a function of this struct (+ the filesystem) only. Unknown keywords are an error.

Selection: `cohorts_db_path`/`tablename` (the cohort table; `curated_trees`, `REF_SPECIES` and, for `site_class_strata`, `COND` must be in the same DB),
`eco_field` (stratification column) / `filter_eco_field`+`filter_ecos`, `filter_plots`, `exclude_plots`, `filter_species` (effective species),
`filter_planted`, `filter_extent` (shapefile), `filter_states`, `skip_disturbances`, `spinup`, `by_subplot`, `drop_phantom`.
Split: `val_frac`, `test_frac`, `n_folds`, `fold_index`, `split_seed` (the split RNG is `SPLIT_RNG(split_seed)` whenever a split is active), `train_cap[_k|_min]`.
Strata / tiering: `stratify_eco_mixed`, `single_ecoregion`, `stratify_landuse`, `site_class_strata` (+`siteclass_hi_max|scheme|l3s`), `shade_tier_csv`,
`min_trees`, `min_agb_frac`, `max_exact`, `max_group`. `disturbance_path`: parquet of per-(subplot, species, visit, age) disturbance rows
(tools/export_cohort_disturbance.py); empty = off.

`on_missing` (:warn | :error) controls every place where preparation used to substitute a default for missing data: species absent from the shade-tier
table (tier "NA"), cohort species absent from the tree statistics (the "_H" catch-all), species without softwood/hardwood class ("H"), plots without a
site class ("NA"), NULL land_use ("NA"), listed filter values that match nothing. `:warn` (default, the behaviour so far) logs each case with its
count; `:error` stops.
"""
Base.@kwdef struct DataConfig
  cohorts_db_path::String
  tablename::String = "data_eco_cohorts"
  eco_field::String = "epa_l3"
  filter_eco_field::String = "epa_l3"
  filter_ecos::Vector{String} = String[]
  filter_plots::Vector{NTuple{4,Int}} = NTuple{4,Int}[]
  exclude_plots::Vector{NTuple{4,Int}} = NTuple{4,Int}[]
  filter_species::Vector{String} = String[]
  filter_planted::Bool = false
  filter_extent::Union{Nothing,String} = nothing
  filter_states::Vector{Int} = Int[]
  skip_disturbances::Bool = true
  spinup::Bool = false
  by_subplot::Bool = false
  drop_phantom::Bool = false
  min_trees::Int = 100
  min_agb_frac::Float64 = 0.05
  val_frac::Float64 = 0.0
  test_frac::Float64 = 0.0
  n_folds::Int = 1
  fold_index::Int = 1
  split_seed::Int = 42
  split_csv::String = ""
  train_cap::Union{Nothing,Int,String} = nothing
  train_cap_k::Float64 = 1.5
  train_cap_min::Int = 20
  stratify_eco_mixed::Bool = false
  single_ecoregion::Bool = false
  stratify_landuse::Bool = false
  site_class_strata::Bool = false
  siteclass_hi_max::Int = 4
  siteclass_scheme::String = "2way"
  siteclass_l3s::Vector{String} = String[]
  shade_tier_csv::Union{Nothing,String} = nothing
  max_exact::Int = 12
  max_group::Int = 4
  disturbance_path::String = ""
  on_missing::Symbol = :warn
end

"`dc` with some fields replaced (DataConfig is immutable)."
with_data(dc::DataConfig; kw...) = DataConfig(; (k => getfield(dc, k) for k in fieldnames(DataConfig))..., kw...)

# The split RNG type (== Pan.RNGType, checked in test/unit_data_layer.jl); seeded from `split_seed` alone so all folds share the shuffle.
const SPLIT_RNG = Random.Xoshiro
split_active(dc::DataConfig) = dc.val_frac > 0.0 || dc.n_folds > 1 || dc.test_frac > 0.0 || !isempty(dc.split_csv)

# The checks that used to be silent (or late, deep in the split code), done once up front.
function validate_data_config(dc::DataConfig)
  dc.on_missing in (:warn, :error) || error("DataConfig.on_missing must be :warn or :error, got :$(dc.on_missing)")
  dc.siteclass_scheme in ("2way", "4cell") || error("siteclass_scheme must be \"2way\" or \"4cell\", got $(repr(dc.siteclass_scheme))")
  0.0 <= dc.val_frac < 1.0 || error("val_frac must be in [0, 1), got $(dc.val_frac)")
  0.0 <= dc.test_frac < 1.0 || error("test_frac must be in [0, 1), got $(dc.test_frac)")
  dc.n_folds >= 1 || error("n_folds must be >= 1, got $(dc.n_folds)")
  1 <= dc.fold_index <= dc.n_folds || error("fold_index $(dc.fold_index) is outside 1..n_folds ($(dc.n_folds))")
  dc.test_frac > 0.0 && dc.n_folds > 1 && error("test_frac>0 (3-way train/val/test split) is incompatible with n_folds>1 (cross-validation) — pick one")
  dc.train_cap !== nothing && dc.n_folds > 1 && error("train_cap (species-balanced training set) is not supported with n_folds>1 cross-validation")
  if !isempty(dc.split_csv)
    isfile(dc.split_csv) || error("split_csv not found: $(dc.split_csv)")
    (dc.val_frac == 0.0 && dc.test_frac == 0.0 && dc.n_folds == 1) || error("split_csv fixes the split: val_frac, test_frac must be 0 and n_folds 1 (got $(dc.val_frac), $(dc.test_frac), $(dc.n_folds))")
    dc.by_subplot && error("split_csv assigns whole plots: by_subplot=true is not supported")
  end
  dc.train_cap isa AbstractString && dc.train_cap != "auto" && error("train_cap must be an integer or \"auto\", got $(repr(dc.train_cap))")
  dc.max_exact >= 0 && dc.max_group >= 0 || error("max_exact / max_group must be >= 0")
  dc.min_trees >= 0 || error("min_trees must be >= 0")
  isfile(dc.cohorts_db_path) || error("cohort database not found: $(dc.cohorts_db_path)")
  dc.shade_tier_csv === nothing || isfile(dc.shade_tier_csv) || error("shade_tier_csv not found: $(dc.shade_tier_csv)")
  isempty(dc.disturbance_path) || isfile(dc.disturbance_path) || error("disturbance parquet not found: $(dc.disturbance_path)")
  dc.filter_extent === nothing || isfile(dc.filter_extent) || error("filter_extent shapefile not found: $(dc.filter_extent)")
  dc
end

# ---- SQL helpers: values are always bound parameters; the few things that cannot be parameters (the attached file, table and
# column names) are validated here ---------------------------------------------------------------------------------------
const _IDENT_RE = r"^[A-Za-z_][A-Za-z0-9_]*$"
function _ident(s::AbstractString, what::AbstractString)::String
  occursin(_IDENT_RE, s) || error("$what: $(repr(s)) is not a valid SQL identifier (letters, digits, underscore; not starting with a digit)")
  String(s)
end
# ATTACH takes a string literal, not a parameter: escape it (and refuse control characters).
_sql_literal_path(p::AbstractString)::String = PanCore.sql_quote_path(p)
_params_placeholders(n::Int) = join(fill("?", n), ",")
# Names of the tables of the attached database `src` (lower-cased -> stored name).
function _src_tables(con)::Dict{String,String}
  df = DataFrame(DuckDB.execute(con, "SELECT table_name FROM information_schema.tables WHERE table_catalog = 'src' ORDER BY table_name"))
  Dict(lowercase(String(t)) => String(t) for t in df.table_name)
end
function _require_table(con, name::AbstractString, what::AbstractString)::String
  nm = _ident(name, what)
  tabs = _src_tables(con)
  haskey(tabs, lowercase(nm)) || error("$what: table '$nm' not found in the cohort database (tables: $(join(sort(collect(values(tabs)))[1:min(end, 25)], ", ")))")
  tabs[lowercase(nm)]
end
function _table_columns(con, table::AbstractString)::Dict{String,String}
  df = DataFrame(DuckDB.execute(con, "SELECT column_name FROM information_schema.columns WHERE table_catalog = 'src' AND table_name = ? ORDER BY ordinal_position", [table]))
  Dict(lowercase(String(c)) => String(c) for c in df.column_name)
end
function _require_column(cols::Dict{String,String}, name::AbstractString, table::AbstractString, what::AbstractString)::String
  nm = _ident(name, what)
  haskey(cols, lowercase(nm)) || error("$what: column '$nm' not found in table '$table' (columns: $(join(sort(collect(values(cols))), ", ")))")
  cols[lowercase(nm)]
end

# on_missing policy: log (or stop on) a substituted default. `what` names the substitution, `items` the affected keys.
function _note_missing(dc_policy::Symbol, what::AbstractString, items)
  isempty(items) && return
  shown = join(first(sort!(string.(collect(items))), 8), ", ")
  msg = "data preparation: $(length(items)) $(what) [$(shown)$(length(items) > 8 ? ", …" : "")]"
  dc_policy === :error && error(msg * " (on_missing: error)")
  @warn msg
end

# ---- disturbance shock data (tools/export_cohort_disturbance.py) ---------------------------------------------
# Parquet of per-(subplot, species, visit T, age) disturbance rows (partial AND full removals). Empty path = off.
# Path comes from `DataConfig.disturbance_path` (config `disturbance_full_removals`); only used with the new scoring order (loss_before_injection).
const _K0_W_TO_AGB = 1.0 / 4046.86     # tree_trajectories w = drybio_g*tpa  →  curated cohort `agb` (g/m²): agb = w / 4046.86

function _load_disturbance_rows(path::AbstractString)::DataFrame
  con = DuckDB.connect(DuckDB.DB())
  DuckDB.execute(con, "SELECT statecd, unitcd, countycd, plot, subp, species_symbol, CAST(measdate AS DATE) AS measdate, age_calc, disturbance_drop_pct, would_be, full_removal FROM read_parquet(?) ORDER BY ALL", [String(path)]) |> DataFrame   # fixed order: rows feed float accumulations (B34)
end

# Plot-level full removals + plot-level partial-drop update, for ONE split's aggregated `sp` (same plot_id / sim_year /
# eco_species_id / age_calc keys as get_injection_cohorts). A plot-level cohort (species, age) at visit T is FULLY removed
# only if no subplot part of it is observed at T in `sp`; if some parts survive, it is partial and `sp.disturbance_drop_pct`
# is replaced by the agb-weighted plot-level share  1 - Σagb / (Σ agb/(1-drop) + Σ removed would-be·K0).
function _plot_shocks!(sp::DataFrame, sh::DataFrame, sym2sid::Dict{String,Int}, esp_map)::DataFrame
  out = DataFrame(plot_id=UIntType[], sim_year=Int[], eco_species_id=UIntType[], age_calc=UIntType[])
  isempty(sh) && return out
  pinfo = Dict{NTuple{4,Int},Tuple{UIntType,Int,Dates.Date}}()
  for r in eachrow(unique(select(sp, [:statecd, :unitcd, :countycd, :plot, :plot_id, :eco_id, :start_measdate])))
    pinfo[(Int(r.statecd), Int(r.unitcd), Int(r.countycd), Int(r.plot))] = (UIntType(r.plot_id), Int(r.eco_id), Dates.Date(r.start_measdate))
  end
  acc = Dict{NTuple{4,Int},Float64}()
  for r in eachrow(sh)
    r.full_removal === true || continue
    pi = get(pinfo, (Int(r.statecd), Int(r.unitcd), Int(r.countycd), Int(r.plot)), nothing); pi === nothing && continue
    sid = get(sym2sid, String(r.species_symbol), 0); sid == 0 && continue
    esp = get(esp_map, (pi[2], sid), 0); esp == 0 && continue
    (ismissing(r.measdate) || ismissing(r.age_calc)) && continue
    sy = round(Int, Dates.value(Dates.Day(Dates.Date(r.measdate) - pi[3])) / 365.25)
    (sy > 0 && r.age_calc >= 0) || continue
    k = (Int(pi[1]), sy, Int(esp), Int(r.age_calc))
    acc[k] = get(acc, k, 0.0) + Float64(r.would_be) * _K0_W_TO_AGB
  end
  rows_by_key = Dict{NTuple{4,Int},Vector{Int}}()
  for i in 1:nrow(sp)
    k = (Int(sp.plot_id[i]), Int(sp.sim_year[i]), Int(sp.eco_species_id[i]), Int(sp.age_calc[i]))
    haskey(acc, k) && push!(get!(rows_by_key, k, Int[]), i)
  end
  nfull = 0; npart = 0
  for k in sort!(collect(keys(acc)))
    rem = acc[k]
    idx = get(rows_by_key, k, nothing)
    if idx === nothing
      push!(out, (UIntType(k[1]), k[2], UIntType(k[3]), UIntType(k[4]))); nfull += 1
    else
      a = sum(Float64(sp.agb_sum[i]) for i in idx)
      wb = sum(Float64(sp.agb_sum[i]) / (1.0 - clamp(Float64(sp.disturbance_drop_pct[i]), 0.0, 0.999)) for i in idx)
      nd = Float32(1.0 - a / (wb + rem))
      for i in idx; sp.disturbance_drop_pct[i] = nd; end
      npart += 1
    end
  end
  println("disturbance shocks: $(nfull) plot-level full removals; $(npart) plot-level cohorts partial (some subplot parts survive) → drop updated")
  out
end
function _cap_train_sites(df::DataFrame, id_cols::Vector{Symbol}, parts, three_way::Bool;
  train_cap::Union{Int,String}, train_cap_k::Float64, train_cap_min::Int)
  tr = vcat([let t = p[3], n = nrow(p[3])
    n == 0 ? hcat(t, DataFrame(_rk=Float64[])) : hcat(t, DataFrame(_rk=[(i - 0.5) / n for i in 1:n]))
  end for p in parts]...)
  last_d = combine(groupby(df, id_cols), :measdate => maximum => :_last)
  j = innerjoin(select(df, vcat(id_cols, [:measdate, :effective_species, :agb, :eco_id])), last_d, on=id_cols)
  j = j[j.measdate .== j._last, :]
  spa = combine(groupby(j, vcat(id_cols, [:effective_species])), :agb => sum => :_a, :eco_id => first => :eco_id)
  dom = combine(groupby(spa, id_cols)) do r
    i = argmin(k -> (-Float64(r._a[k]), String(r.effective_species[k])), 1:nrow(r))
    (; eco_id=r.eco_id[i], dom=String(r.effective_species[i]))
  end
  tr = innerjoin(tr, dom, on=id_cols)
  sort!(tr, vcat([:eco_id, :dom, :_rk], id_cols))
  sizes = combine(groupby(tr, [:eco_id, :dom]), nrow => :n)
  capof = if train_cap isa Int
    Dict(e => train_cap for e in unique(sizes.eco_id))
  else
    train_cap == "auto" || error("train_cap must be an Int or \"auto\" (got \"$train_cap\")")
    Dict(e => max(train_cap_min, ceil(Int, train_cap_k * _median(sizes.n[sizes.eco_id .== e]))) for e in unique(sizes.eco_id))
  end
  keep = falses(nrow(tr))
  for g in groupby(tr, [:eco_id, :dom]; sort=false)
    c = capof[g.eco_id[1]]
    rows = parentindices(g)[1]
    keep[rows[1:min(c, length(rows))]] .= true
  end
  surplus = tr[.!keep, :]
  sort!(surplus, vcat([:_rk, :eco_id, :dom], id_cols))          # interleave strata so val/test both get every stratum
  to_test = three_way ? isodd.(1:nrow(surplus)) .== false : falses(nrow(surplus))
  sv = surplus[.!to_test, id_cols]; st = surplus[to_test, id_cols]
  tab = combine(groupby(tr, [:eco_id, :dom]), nrow => :n_before)
  tab.n_after = [min(capof[e], n) for (e, n) in zip(tab.eco_id, tab.n_before)]
  @info "train_cap ($(train_cap)): train $(nrow(tr)) → $(count(keep)); surplus → val $(nrow(sv)), test $(nrow(st))"
  show(stdout, MIME("text/plain"), sort(tab[tab.n_before .> tab.n_after, :], :n_before, rev=true); allrows=true); println()
  dd = select(dom, vcat(id_cols, [:eco_id, :dom]))
  return sv, st, dd
end

function _stratified_split_df(df::DataFrame, id_cols::Vector{Symbol},
  val_frac::Float64, rng::Random.AbstractRNG; n_folds::Int=1, fold_index::Int=1, test_frac::Float64=0.0,
  train_cap::Union{Nothing,Int,String}=nothing, train_cap_k::Float64=1.5, train_cap_min::Int=20)
  three_way = test_frac > 0.0            # train/val/test single split — a SEPARATE mode from K-fold CV
  three_way && n_folds > 1 && error("test_frac>0 (3-way train/val/test split) is incompatible with n_folds>1 (cross-validation) — pick one")
  site_sp = unique(select(df, vcat(id_cols, [:eco_id, :effective_species])))
  sp_count = Dict(r.effective_species => r.n_sites
                  for r in eachrow(combine(groupby(site_sp, :effective_species), nrow => :n_sites)))
  site_strata = combine(groupby(site_sp, id_cols)) do rows
    # tie-break rarest species by NAME so the stratum is independent of DB row order (reproducible split)
    idx = argmin(i -> (get(sp_count, rows.effective_species[i], typemax(Int)), rows.effective_species[i]), 1:nrow(rows))
    (; stratum=(rows.eco_id[1], rows.effective_species[idx]))   # eco×lu × rarest species
  end
  sort!(site_strata, id_cols)   # canonical site order → the seeded per-stratum shuffle is reproducible regardless of DB row order
  # Three modes, all shuffling each stratum ONCE with the same rng (reproducible from split_seed):
  #   • three_way (test_frac>0): val = first val_frac, test = next test_frac, train = rest.
  #   • n_folds>1: K-fold CV — contiguous chunk `fold_index`/`n_folds` = val, rest = train.
  #   • n_folds<=1, test_frac=0: single val_frac draw (bit-for-bit the original train/val behaviour).
  parts = [
    begin
      shuffled = gdf[Random.shuffle(rng, 1:nrow(gdf)), id_cols]
      m = nrow(shuffled)
      if n_folds <= 1
        n_val = max(1, round(Int, m * val_frac))
        if three_way
          n_test = min(m - n_val, max(1, round(Int, m * test_frac)))
          (shuffled[1:n_val, :], n_test > 0 ? shuffled[n_val+1:n_val+n_test, :] : shuffled[1:0, :], shuffled[n_val+n_test+1:end, :])
        else
          (shuffled[1:n_val, :], shuffled[1:0, :], shuffled[n_val+1:end, :])   # test part empty in 2-way mode
        end
      else
        lo = div((fold_index - 1) * m, n_folds) + 1   # contiguous K-fold chunk (partitions the stratum)
        hi = div(fold_index * m, n_folds)
        (hi >= lo ? shuffled[lo:hi, :] : shuffled[1:0, :], shuffled[1:0, :], shuffled[1:0, :])
      end
    end for gdf in groupby(site_strata, :stratum)
  ]
  val_sites  = vcat([p[1] for p in parts]...)
  test_sites = three_way ? vcat([p[2] for p in parts]...) : nothing
  split_dom = nothing   # id_cols + eco_id + dom of every plot, set when the species-balanced train cap ran (for split_plots.csv)
  if train_cap !== nothing
    n_folds > 1 && error("train_cap (species-balanced training set) is not supported with n_folds>1 cross-validation")
    sv, st, split_dom = _cap_train_sites(df, id_cols, parts, three_way; train_cap=train_cap, train_cap_k=train_cap_k, train_cap_min=train_cap_min)
    val_sites = vcat(val_sites, sv)
    three_way && (test_sites = vcat(test_sites, st))
  end
  n_total = nrow(site_strata); n_val_out = nrow(val_sites); n_test_out = three_way ? nrow(test_sites) : 0
  @info "Split" mode = (three_way ? "train/val/test" : (n_folds > 1 ? "cv" : "train/val")) n_folds fold_index n_train = n_total - n_val_out - n_test_out n_val = n_val_out n_test = n_test_out
  df_val = semijoin(df, val_sites, on=id_cols)
  df_test = three_way ? semijoin(df, test_sites, on=id_cols) : nothing
  df_train = antijoin(df, val_sites, on=id_cols)
  three_way && (df_train = antijoin(df_train, test_sites, on=id_cols))
  return df_train, df_val, df_test, split_dom
end

"""
    _split_from_csv(df, id_cols, path, rng; train_cap, train_cap_k, train_cap_min, on_missing) -> (df_train, df_val, df_test, split_dom)

The split of `_stratified_split_df` with every plot's set read from `path` (`read_split_plots`: statecd, unitcd, countycd, plot, set) instead of drawn:
one role of a nested CV (tools/ncv). Plots of the cohort table that the file does not list are dropped (`on_missing`). The species-balanced train cap
(`train_cap`) ranks each (eco × rarest species) stratum's train plots by a seeded shuffle (`rng`, as the drawn split) and moves its surplus to VAL only, so the
test plots stay exactly the file's. `df_test` is `nothing` when the file has no test plot (e.g. the final fit of a nested CV).
"""
function _split_from_csv(df::DataFrame, id_cols::Vector{Symbol}, path::AbstractString, rng::Random.AbstractRNG;
  train_cap::Union{Nothing,Int,String}=nothing, train_cap_k::Float64=1.5, train_cap_min::Int=20, on_missing::Symbol=:warn)
  id_cols == [:statecd, :unitcd, :countycd, :plot] || error("split_csv assigns whole plots (id_cols $(id_cols))")
  sp = read_split_plots(path)
  nrow(unique(sp[:, id_cols])) == nrow(sp) || error("split_csv $(path): a plot is listed more than once")
  setof = Dict((r.statecd, r.unitcd, r.countycd, r.plot) => r.set for r in eachrow(sp))
  pkey(r) = (Int(r.statecd), Int(r.unitcd), Int(r.countycd), Int(r.plot))
  site_sp = unique(select(df, vcat(id_cols, [:eco_id, :effective_species])))
  sp_count = Dict(r.effective_species => r.n_sites for r in eachrow(combine(groupby(site_sp, :effective_species), nrow => :n_sites)))
  site_strata = combine(groupby(site_sp, id_cols)) do rows
    idx = argmin(i -> (get(sp_count, rows.effective_species[i], typemax(Int)), rows.effective_species[i]), 1:nrow(rows))
    (; stratum=(rows.eco_id[1], rows.effective_species[idx]))
  end
  sort!(site_strata, id_cols)
  site_strata.set = [get(setof, pkey(r), "") for r in eachrow(site_strata)]
  nmiss = count(isempty, site_strata.set)
  nmiss > 0 && _note_missing(on_missing, "plots of the cohort table absent from split_csv $(basename(path)) (dropped)", ["$(nmiss) of $(nrow(site_strata)) plots"])
  site_strata = site_strata[.!isempty.(site_strata.set), :]
  three_way = any(==("test"), site_strata.set)
  parts = [(g[g.set .== "val", id_cols], g[g.set .== "test", id_cols], (t = g[g.set .== "train", id_cols]; t[Random.shuffle(rng, 1:nrow(t)), :]))
           for g in groupby(site_strata, :stratum)]
  val_sites = vcat([p[1] for p in parts]...); test_sites = vcat([p[2] for p in parts]...); train_sites = vcat([p[3] for p in parts]...)
  split_dom = nothing
  if train_cap !== nothing
    sv, _, split_dom = _cap_train_sites(df, id_cols, parts, false; train_cap=train_cap, train_cap_k=train_cap_k, train_cap_min=train_cap_min)
    val_sites = vcat(val_sites, sv); train_sites = antijoin(train_sites, sv, on=id_cols)
  end
  @info "Split" mode = "split_csv" file = basename(path) n_train = nrow(train_sites) n_val = nrow(val_sites) n_test = nrow(test_sites) n_dropped = nmiss
  df_train = semijoin(df, train_sites, on=id_cols); df_val = semijoin(df, val_sites, on=id_cols)
  df_test = three_way ? semijoin(df, test_sites, on=id_cols) : nothing
  return df_train, df_val, df_test, split_dom
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

function make_splots(df::DataFrame, dc::DataConfig; split_rng::Union{Nothing,Random.AbstractRNG}=nothing,
  tree_stats::Union{Nothing,DataFrame}=nothing, shade_tier::Union{Nothing,Dict{String,String}}=nothing)
  eco = dc.eco_field; filter_species = dc.filter_species; by_subplot = dc.by_subplot
  val_frac = dc.val_frac; n_folds = dc.n_folds; fold_index = dc.fold_index; test_frac = dc.test_frac
  min_trees = dc.min_trees; min_agb_frac = dc.min_agb_frac
  stratify_eco_mixed = dc.stratify_eco_mixed; single_ecoregion = dc.single_ecoregion; stratify_landuse = dc.stratify_landuse
  max_exact = dc.max_exact; max_group = dc.max_group
  train_cap = dc.train_cap; train_cap_k = dc.train_cap_k; train_cap_min = dc.train_cap_min
  # eco_field can be any column of the loaded cohorts (epa_l4/epa_l3/ecosubcd, or a curated stratifier like land_use).
  # (Step 6: it used to fall back silently to :ecosubcd when the column was absent; now an error.)
  hasproperty(df, Symbol(eco)) || error("make_splots: eco_field '$eco' is not a column of the cohorts table (columns: $(join(names(df), ", ")))")
  eco_field = Symbol(eco)

  assign_tiered_species!(df; max_exact=max_exact, max_group=max_group, min_trees=min_trees, min_agb_frac=min_agb_frac, tree_stats=tree_stats, shade_tier=shade_tier, on_missing=dc.on_missing)
  species_map = sort!(unique(select(df, [:species_symbol, :effective_species])), :species_symbol)
  if !isempty(filter_species)
    fs = Set(filter_species)
    _note_missing(dc.on_missing, "filter_species entries matching no effective species", setdiff(fs, Set(df.effective_species)))
    filter!(row -> row.effective_species in fs, df)
    nrow(df) > 0 || error("make_splots: filter_species $(sort(collect(fs))) selects no cohort rows (effective species present: $(join(sort(unique(species_map.effective_species)), ", ")))")
  end

  # Optionally stratify the ecoregion by stand mixedness: :eco is the base ecoregion,
  # :eco_mixed splits it into "<eco>_mixed" / "<eco>_pure" per (plot-level) mixed_plot. When
  # enabled the model fits params per eco_mixed; otherwise eco_mixed == the base ecoregion.
  # single_ecoregion collapses the EPA ecoregion to one group ("ALL"); stratify_landuse then (or on its
  # own) re-splits by the land_use column. So: single only → 1 eco; single+landuse → land_use classes
  # (EPA ignored); neither → the eco_field grouping as before.
  df.eco = single_ecoregion ? fill("ALL", nrow(df)) : string.(getproperty(df, eco_field))
  if stratify_landuse && eco_field != :land_use
    hasproperty(df, :land_use) || error("stratify_landuse=true but the cohorts table has no :land_use column")
    n_lu_null = count(ismissing, df.land_use)
    n_lu_null > 0 && _note_missing(dc.on_missing, "cohort rows with NULL land_use (stratum \"NA\")", ["$n_lu_null rows"])
    df.eco = df.eco .* "|lu=" .* string.(coalesce.(df.land_use, "NA"))
  end
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
  shock_rows = isempty(dc.disturbance_path) ? nothing : _load_disturbance_rows(dc.disturbance_path)
  if hasproperty(df, :disturbance_drop_pct)
    df.disturbance_drop_pct = Float32.(coalesce.(df.disturbance_drop_pct, 0.0))
  elseif shock_rows !== nothing
    # the cohort table carries no partial drop: attach it from the export's partial rows (same key as cohort_disturbance.py --attach)
    pd = Dict{Tuple{Int,Int,Int,Int,Int,String,Dates.Date,Int},Float32}()
    for r in eachrow(shock_rows)
      r.full_removal === true && continue
      pd[(Int(r.statecd), Int(r.unitcd), Int(r.countycd), Int(r.plot), Int(r.subp), String(r.species_symbol), Dates.Date(r.measdate), Int(r.age_calc))] = Float32(r.disturbance_drop_pct)
    end
    df.disturbance_drop_pct = Float32[get(pd, (Int(a), Int(b), Int(c), Int(d), Int(e), String(f), Dates.Date(g), Int(h)), 0f0)
      for (a, b, c, d, e, f, g, h) in zip(df.statecd, df.unitcd, df.countycd, df.plot, df.subp, df.species_symbol, df.measdate, df.age_calc)]
    println("disturbance_drop_pct attached from $(basename(dc.disturbance_path)): $(sum(df.disturbance_drop_pct .> 0)) / $(nrow(df)) cohort rows > 0")
  else
    df.disturbance_drop_pct = zeros(Float32, nrow(df))
  end
  # phantom-backfill share (FIA_DATA_PREP cohorts_landis: synthetic pre-debut rows, intro_type = "phantom_backfill");
  # aggregated AGB-weighted into splots.phantom_frac — evaluation-only (not read by the simulator or the loss).
  df._phantom = hasproperty(df, :intro_type) ? Float32[coalesce(t, "") == "phantom_backfill" ? 1f0 : 0f0 for t in df.intro_type] : zeros(Float32, nrow(df))

  id_cols = by_subplot ? [:statecd, :unitcd, :countycd, :plot, :subp] :
            [:statecd, :unitcd, :countycd, :plot]
  # NB: sim_year is deliberately NOT a grouping key. In curated_cohorts_landis sim_year is
  # anchored to each SUBPLOT's first measurement, so two subplots at the same plot visit can
  # carry different sim_year for the same (plot-level) measdate — which would split otherwise
  # identical species×age cohorts when by_subplot=false. We group on measdate (plot-level) and
  # recompute sim_year per id_cols below.
  base_fields = [:plt_cn, :statecd, :unitcd, :countycd, :plot, :eco_id, :measdate,
    :species_id, :effective_species, :age_calc,
    (hasproperty(df, :cycle) ? [:cycle] : Symbol[])...]   # carry FIA inventory cycle when present
  fields = by_subplot ? vcat(base_fields, [:subp]) : base_fields

  # Split right here — after mapping, before aggregation — so each half gets its
  # own contiguous plot_ids and is fully self-contained.
  do_split = split_active(dc)
  do_split && isnothing(split_rng) && error("make_splots: a split is requested (val_frac/test_frac/n_folds) but no split RNG was given")   # used to skip the split silently
  train_cap !== nothing && !do_split &&
    @warn "train_cap=$(train_cap) is IGNORED: no train/val/test split is active (set val_frac/test_frac)"
  df_train, df_val_raw, df_test_raw, split_dom = !isempty(dc.split_csv) ?
                         _split_from_csv(df, id_cols, dc.split_csv, split_rng; train_cap=train_cap, train_cap_k=train_cap_k, train_cap_min=train_cap_min, on_missing=dc.on_missing) :
                         do_split ? _stratified_split_df(df, id_cols, val_frac, split_rng; n_folds=n_folds, fold_index=fold_index, test_frac=test_frac,
                           train_cap=train_cap, train_cap_k=train_cap_k, train_cap_min=train_cap_min) :
                         (df, nothing, nothing, nothing)

  # Aggregate a raw cohort df into a splots DataFrame with contiguous plot_ids.
  function _agg(df_sub::DataFrame)
    raw = combine(groupby(df_sub, fields, sort=false), nrow => :count, :agb => sum => :agb_sum,
      [:agb, :disturbance_drop_pct] => _weighted_cohort_drop => :disturbance_drop_pct,
      [:agb, :_phantom] => ((a, p) -> (t = sum(a); t > 0 ? Float32(sum(a .* p) / t) : Float32(maximum(p; init=0)))) => :phantom_frac)
    # NB: curated cohort `agb` already carries the plot-level TPA_UNADJ expansion (each subplot is
    # 1/nsub of the plot), so Step-1's `:agb => sum` over subplots IS the plot per-acre density
    # (verified against TreeMap DRYBIO_L). Do NOT divide by subplot count again — that deflated the
    # ground-truth AGB ~4× and pushed B_MAX/ANPP_MAX low.
    plots = raw
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

  nrow(df_train) > 0 || error("make_splots: the training split is empty ($(nrow(df)) cohort rows before the split; val_frac=$val_frac, test_frac=$test_frac, n_folds=$n_folds)")
  splots_train = _agg(df_train)
  splots_val = isnothing(df_val_raw) ? nothing : _agg(df_val_raw)
  splots_test = isnothing(df_test_raw) ? nothing : _agg(df_test_raw)
  if shock_rows !== nothing
    if by_subplot
      @warn "disturbance_full_removals: ignored with by_subplot=true (plot-level aggregation only)"
    else
      sym2sid = Dict{String,Int}(String(r.species_symbol) => species_symbol_map_dict[r.effective_species] for r in eachrow(species_map) if haskey(species_symbol_map_dict, r.effective_species))
      for (nm, spx) in (("train", splots_train), ("val", splots_val), ("test", splots_test))
        spx === nothing && continue
        println("  [$(nm)]")
        metadata!(spx, "disturbance_shock", _plot_shocks!(spx, shock_rows, sym2sid, eco_species_id_map); style=:note)
      end
    end
  end
  return splots_train, eco_vals, species_symbol_map_vals, ddf.species_ids, splots_val, splots_test, DataOutputs(split_dom, species_map)
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
_inject_cols(df) = hasproperty(df, :phantom_frac) ? vcat(_INJECT_COLS, [:phantom_frac]) : _INJECT_COLS   # + evaluation-only phantom share

function get_injection_cohorts(splots::DataFrame; all_cohorts::Bool=false)::DataFrame
  r = _get_injection_cohorts(splots; all_cohorts=all_cohorts)
  # carry the plot-level full-removal shock rows (Data._plot_shocks!) alongside the injection cohorts
  "disturbance_shock" in metadatakeys(splots) && metadata!(r, "disturbance_shock", metadata(splots, "disturbance_shock"); style=:note)
  r
end

function _get_injection_cohorts(splots::DataFrame; all_cohorts::Bool=false)::DataFrame
  # all_cohorts=true: return the FULL observed state at every measurement year
  # (one row per observed cohort), so the caller can override the simulator
  # entirely — replacing each measured site's cohorts with the empirical ones.
  # Used for the override sanity test and the injection-noise sensitivity test.
  if all_cohorts
    return select(splots, _inject_cols(splots))
  end
  # Cohorts with birth_sim_year = sim_year - age_calc > 0 were born after
  # simulation start and are not in the initial conditions. Return one row per
  # cohort at its first FIA measurement year so the caller can inject it into
  # the simulation at the right time with the observed age and biomass.
  birth_sym = Int.(splots.sim_year) .- Int.(splots.age_calc)
  inject = splots[birth_sym.>0, :]
  isempty(inject) && return select(inject, _inject_cols(inject))
  inject = transform(inject,
    [:sim_year, :age_calc] => ByRow((s, a) -> Int(s) - Int(a)) => :_birth_sym)
  first_app = combine(groupby(inject, [:plot_id, :eco_species_id, :_birth_sym]),
    :sim_year => minimum => :_inject_year)
  result = innerjoin(inject,
    rename(first_app, :_inject_year => :sim_year),
    on=[:plot_id, :eco_species_id, :_birth_sym, :sim_year])
  return select(result, _inject_cols(result))
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
# When `cycle.use_fia_cycle` is set, cycles come from the FIA inventory `cycle` column verbatim (raw value
# as the 1-based index, so train and val share the same mapping; empty low slots cost a little memory
# but add zero loss). Otherwise cycles are the calculated cycle_years-wide buckets from the first visit.
function build_cycle_map(splots::DataFrame; cycle_years::Real=8, cycle::CycleOptions=CycleOptions())
  if cycle.use_fia_cycle && hasproperty(splots, :cycle)
    meas = unique(select(splots, [:plot_id, :sim_year, :cycle]))
    vals = Int[Int(r.cycle) for r in eachrow(meas) if !ismissing(r.cycle)]
    isempty(vals) && return Dict{Tuple{Int,Int},Int}(), 0
    mn = minimum(vals); mrg = max(1, cycle.merge)   # merge mrg consecutive FIA cycles into one group
    cmap = Dict{Tuple{Int,Int},Int}()
    n_cycles = 0
    for r in eachrow(meas)
      ismissing(r.cycle) && continue
      g = (Int(r.cycle) - mn) ÷ mrg + 1                   # group index (1-based); mrg=1 ⇒ raw FIA cycle (shifted)
      cmap[(Int(r.plot_id), Int(r.sim_year))] = g
      g > n_cycles && (n_cycles = g)
    end
    return cmap, n_cycles
  end
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
  plot_table = _require_table(con, plot_table, "filter_extent")
  plot_keys = [:statecd, :unitcd, :countycd, :plot]
  plots = unique(select(cohorts_df, plot_keys))
  DuckDB.register_data_frame(con, plots, "extent_plots")
  coords = DuckDB.execute(con, """
    SELECT p.statecd AS statecd, p.unitcd AS unitcd, p.countycd AS countycd, p.plot AS plot,
           ANY_VALUE(p.LAT) AS lat, ANY_VALUE(p.LON) AS lon
    FROM src.$(plot_table) p
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

"""
    prepare_parametrization_data(dc::DataConfig; split_rng=nothing) -> (splots, eco_list, species_list, eco_species_ids, splots_val, splots_test, outs)

Load the cohorts selected by `dc`, tier the species, split train / val / test and aggregate. `split_rng` defaults to `SPLIT_RNG(split_seed)` when a split is
active (val_frac, test_frac or n_folds > 1), so all folds share the shuffle. Every SQL value is a bound parameter; the table and column names that come from
the config are validated against the database. Selections that come out empty are errors.
"""
function prepare_parametrization_data(dc::DataConfig; split_rng::Union{Nothing,Random.AbstractRNG}=nothing)
  validate_data_config(dc)
  if split_active(dc) && split_rng === nothing
    split_rng = SPLIT_RNG(UInt64(dc.split_seed))
  end
  println("Connecting to: $(dc.cohorts_db_path) ")
  # In-memory main DB + ATTACH the file READ-ONLY: a killed run can never corrupt the file (a mid-write
  # checkpoint was the corruption cause), yet df registrations (loaded_subplots) still land in the writable
  # in-memory catalog. DB tables are qualified `src.…`; registered views stay unqualified. No CREATE INDEX
  # on a read-only DB — DuckDB's columnar scans are fast enough for the one-shot load.
  con = DuckDB.connect(DuckDB.DB())
  DuckDB.execute(con, "ATTACH '$(_sql_literal_path(dc.cohorts_db_path))' AS src (READ_ONLY);")
  tbl = _require_table(con, dc.tablename, "tablename")
  cols = _table_columns(con, tbl)
  conds = String[]; params = Any[]
  if length(dc.filter_ecos) > 0
    fcol = _require_column(cols, dc.filter_eco_field, tbl, "filter_eco_field")
    push!(conds, "\"$(fcol)\" IN ($(_params_placeholders(length(dc.filter_ecos))))"); append!(params, dc.filter_ecos)
  end
  if dc.drop_phantom   # synthetic pre-debut cohorts (FIA_DATA_PREP phantom backfill) are not measurements
    _require_column(cols, "intro_type", tbl, "drop_phantom")
    push!(conds, "coalesce(intro_type, '') <> 'phantom_backfill'")
  end
  if length(dc.filter_states) > 0   # region = state ∩ eco (e.g. EPA Region 4 FIPS); empty ⇒ no state filter
    push!(conds, "statecd IN ($(_params_placeholders(length(dc.filter_states))))"); append!(params, dc.filter_states)
  end
  if dc.skip_disturbances
    _require_column(cols, "subp_has_dstrb", tbl, "skip_disturbances")
    push!(conds, "subp_has_dstrb = false")
  end
  if !dc.spinup
    mcol = dc.by_subplot ? "subp_meas_num" : "plot_meas_num"
    _require_column(cols, mcol, tbl, "spinup: false")
    push!(conds, "$(mcol) > 1")
  end
  plot_tuples(v) = DataFrame(statecd=[t[1] for t in v], unitcd=[t[2] for t in v], countycd=[t[3] for t in v], plot=[t[4] for t in v])
  if length(dc.filter_plots) > 0
    DuckDB.register_data_frame(con, plot_tuples(dc.filter_plots), "sel_plots")
    push!(conds, "(statecd, unitcd, countycd, plot) IN (SELECT statecd, unitcd, countycd, plot FROM sel_plots)")
  end
  if length(dc.exclude_plots) > 0
    DuckDB.register_data_frame(con, plot_tuples(dc.exclude_plots), "excl_plots")
    push!(conds, "(statecd, unitcd, countycd, plot) NOT IN (SELECT statecd, unitcd, countycd, plot FROM excl_plots)")
  end
  if dc.filter_planted
    _require_column(cols, "intro_type", tbl, "filter_planted")
    push!(conds, "(statecd, unitcd, countycd, plot, subp) IN (SELECT statecd, unitcd, countycd, plot, subp FROM src.$(tbl) WHERE intro_type = 'planted')")
  end
  sql = "SELECT * FROM src.$(tbl) WHERE true" * join(" AND " .* conds) * " ORDER BY ALL"   # total, deterministic row order (all columns) — a parallel scan is otherwise unordered, making the downstream aggregation/split non-reproducible even with a fixed seed
  println(sql, isempty(params) ? "" : "   -- params: $(length(params))")
  cohorts_df = DataFrame(DuckDB.execute(con, sql, params))
  println("$(nrow(cohorts_df)) cohort rows loaded.")
  nrow(cohorts_df) > 0 || error("prepare_parametrization_data: the cohort selection is empty (table $(tbl); filter_ecos=$(dc.filter_ecos) on $(dc.filter_eco_field), filter_states=$(dc.filter_states), " *
    "$(length(dc.filter_plots)) filter_plots, $(length(dc.exclude_plots)) exclude_plots, skip_disturbances=$(dc.skip_disturbances), spinup=$(dc.spinup), drop_phantom=$(dc.drop_phantom), filter_planted=$(dc.filter_planted))")
  if length(dc.filter_ecos) > 0
    fcol = _require_column(cols, dc.filter_eco_field, tbl, "filter_eco_field")
    _note_missing(dc.on_missing, "filter_ecos values matching no cohort row", setdiff(Set(dc.filter_ecos), Set(string.(skipmissing(cohorts_df[!, Symbol(fcol)])))))
  end
  if length(dc.filter_plots) > 0
    have = Set(zip(cohorts_df.statecd, cohorts_df.unitcd, cohorts_df.countycd, cohorts_df.plot))
    _note_missing(dc.on_missing, "filter_plots entries matching no cohort row", [t for t in dc.filter_plots if !(t in have)])
  end

  # Site-productivity stratification: replace the `land_use` column with a per-plot site-tier derived from
  # COND.SITECLCD (dominant forested condition, max CONDPROP_UNADJ), binned hi = SITECLCD ≤ siteclass_hi_max,
  # lo = above. FIADB stays read-only (computed here, not stored). `stratify_landuse` then appends |lu=<tier>,
  # so the ecoregion becomes epa_l3 × {hi,lo}. Order-preserving Dict map ⇒ deterministic.
  if dc.site_class_strata
    cond_tbl = _require_table(con, "COND", "site_class_strata")
    # 2way: hi = SITECLCD ≤ siteclass_hi_max, lo = above.  4cell: A=1-3, B=4, C=5, D=6-7 (per-species tiering).
    bin_expr, bin_params = dc.siteclass_scheme == "4cell" ?
      ("CASE WHEN sc<=3 THEN 'A' WHEN sc=4 THEN 'B' WHEN sc=5 THEN 'C' ELSE 'D' END", Any[]) :
      ("CASE WHEN sc <= ? THEN 'hi' ELSE 'lo' END", Any[dc.siteclass_hi_max])
    sc = DuckDB.execute(con, """
      WITH cr AS (SELECT co.STATECD s, co.UNITCD u, co.COUNTYCD c, co.PLOT p, co.SITECLCD sc, SUM(co.CONDPROP_UNADJ) w
                  FROM src.$(cond_tbl) co WHERE co.COND_STATUS_CD = 1 AND co.SITECLCD IS NOT NULL GROUP BY ALL),
      rk AS (SELECT s, u, c, p, sc, ROW_NUMBER() OVER (PARTITION BY s, u, c, p ORDER BY w DESC, sc) rn FROM cr)
      SELECT s AS statecd, u AS unitcd, c AS countycd, p AS plot,
             $(bin_expr) AS site_class
      FROM rk WHERE rn = 1
    """, bin_params) |> DataFrame
    scmap = Dict((r.statecd, r.unitcd, r.countycd, r.plot) => String(r.site_class) for r in eachrow(sc))
    plot_keys = unique(zip(cohorts_df.statecd, cohorts_df.unitcd, cohorts_df.countycd, cohorts_df.plot))
    _note_missing(dc.on_missing, "plots without a site class in COND (land_use \"NA\")", [t for t in plot_keys if !haskey(scmap, t)])
    cohorts_df.land_use = [get(scmap, (r.statecd, r.unitcd, r.countycd, r.plot), "NA") for r in eachrow(cohorts_df)]
    if !isempty(dc.siteclass_l3s)   # site cells only in the listed ecos; every other eco collapses to ONE cell "X"
      keep_sc = Set(dc.siteclass_l3s); ecol = Symbol(dc.eco_field)
      hasproperty(cohorts_df, ecol) || error("siteclass_l3s: eco_field '$(dc.eco_field)' is not a column of $(tbl)")
      cohorts_df.land_use = [string(e) in keep_sc ? lu : "X" for (e, lu) in zip(cohorts_df[!, ecol], cohorts_df.land_use)]
    end
    nplt(v) = length(unique(zip(cohorts_df.statecd[cohorts_df.land_use.==v], cohorts_df.unitcd[cohorts_df.land_use.==v], cohorts_df.countycd[cohorts_df.land_use.==v], cohorts_df.plot[cohorts_df.land_use.==v])))
    scheme_desc = dc.siteclass_scheme == "4cell" ? "4cell A=1-3,B=4,C=5,D=6-7" : "2way hi≤$(dc.siteclass_hi_max),lo=above"
    println("site_class_strata: land_use ← site-tier ($(scheme_desc)); plots: " *
            join(["$v=$(nplt(v))" for v in sort(unique(cohorts_df.land_use))], ", "))
  end

  # Restrict to plots whose location falls inside the supplied shapefile (eco_field still
  # decides each plot's ecoregion). Applied before any other per-plot processing.
  if !isnothing(dc.filter_extent)
    cohorts_df = filter_cohorts_by_extent(con, cohorts_df, dc.filter_extent)
    println("After filter_extent ($(basename(dc.filter_extent))): $(nrow(cohorts_df)) rows, $(length(unique(zip(cohorts_df.statecd,cohorts_df.unitcd,cohorts_df.countycd,cohorts_df.plot)))) plots")
    isempty(cohorts_df) && error("filter_extent kept 0 plots — compare the 'plot coords' and 'polygon bbox' ranges printed above. If they don't overlap it's a CRS/axis issue: try pre-reprojecting the shapefile with `ogr2ogr -t_srs EPSG:4326 out.shp $(dc.filter_extent)`.")
  end

  # Tag each plot mixed/pure over its entire loaded history (used to stratify ecoregions).
  if dc.stratify_eco_mixed
    cohorts_df = with_mixed_plot(cohorts_df)
    println("Plots tagged mixed_plot: $(sum(unique(select(cohorts_df, [:statecd,:unitcd,:countycd,:plot,:mixed_plot])).mixed_plot)) mixed of $(length(unique(zip(cohorts_df.statecd,cohorts_df.unitcd,cohorts_df.countycd,cohorts_df.plot))))")
  end

  # Per-species RAW tree stats from curated_trees, restricted to the loaded subplots, for
  # species tiering (cohort-row counts under-count trees). Per DISTINCT tree (full TREE_ID)
  # take max(DRYBIO_AG) — a tree measured multiple times counts once, biomass is its max, not
  # a sum over visits — then per (species, spgrpcd, sftwd_hrdwd): COUNT distinct trees and SUM
  # the per-tree maxes. spgrpcd/sftwd_hrdwd are carried so assign_tiered_species! can group on them.
  trees_tbl = _require_table(con, "curated_trees", "species tiering (tree statistics)")
  ref_tbl = _require_table(con, "REF_SPECIES", "species tiering (tree statistics)")
  subplots_df = unique(select(cohorts_df, [:statecd, :unitcd, :countycd, :plot, :subp]))
  DuckDB.register_data_frame(con, subplots_df, "loaded_subplots")
  tree_stats = DuckDB.execute(con, """
    WITH tree_max AS (
      SELECT t.STATECD, t.UNITCD, t.COUNTYCD, t.PLOT, t.SUBP, t.TREE,
             max(t.SPCD)      AS spcd,
             max(t.SPGRPCD)   AS spgrpcd,
             max(t.DRYBIO_AG) AS drybio_max
      FROM src.$(trees_tbl) t
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
    JOIN src.$(ref_tbl) r ON r.SPCD = tm.spcd
    GROUP BY r.SPECIES_SYMBOL, tm.spgrpcd, r.SFTWD_HRDWD
    ORDER BY ALL
  """) |> DataFrame
  println("Tree stats: $(nrow(tree_stats)) (species,spgrpcd,sftwd) rows from curated_trees over $(nrow(subplots_df)) subplots.")
  nrow(tree_stats) > 0 || error("prepare_parametrization_data: no live trees in $(trees_tbl) for the $(nrow(subplots_df)) loaded subplots — the species tiering has no statistics")

  # optional shade-tolerance tier (species → LST[1-3]/HST[4-5]) to split the species grouping key
  shade_tier = nothing
  if dc.shade_tier_csv !== nothing
    st = CSV.read(dc.shade_tier_csv, DataFrame)
    for c in ("sym", "shade_class")
      c in names(st) || error("shade_tier_csv $(dc.shade_tier_csv): column '$c' not found (columns: $(join(names(st), ", ")))")
    end
    shade_tier = Dict(uppercase(strip(String(r.sym))) => (Int(r.shade_class) <= 3 ? "LST" : "HST") for r in eachrow(st))
    println("shade_tier: $(length(shade_tier)) species → LST/HST from $(basename(dc.shade_tier_csv))")
  end

  @time splots, eco_list, species_list, eco_species_ids, splots_val, splots_test, outs =
    make_splots(cohorts_df, dc; split_rng=split_rng, tree_stats=tree_stats, shade_tier=shade_tier)
  mark_estab_year!(splots)
  isnothing(splots_val) || mark_estab_year!(splots_val)
  isnothing(splots_test) || mark_estab_year!(splots_test)

  n_viol = check_cohort_continuity(splots)
  println("Cohort continuity violations: $n_viol")

  return splots, eco_list, species_list, eco_species_ids, splots_val, splots_test, outs
end

# Keyword form (analysis tools): `prepare_parametrization_data(; cohorts_db_path=..., ...)` = `DataConfig(; ...)` + `split_rng`.
# Unknown keywords are an error; a requested split without `split_rng` uses RNGType(split_seed).
function prepare_parametrization_data(; split_rng::Union{Nothing,Random.AbstractRNG}=nothing, output_dir=nothing, RNG=nothing, eco_field=nothing, kw...)
  # `output_dir` / `RNG` were accepted and ignored by the old signature; `eco_field` may be a Symbol there.
  ef = eco_field === nothing ? () : (:eco_field => String(eco_field),)
  prepare_parametrization_data(DataConfig(; ef..., kw...); split_rng=split_rng)
end
