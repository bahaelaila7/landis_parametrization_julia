# Load the data-derived per-species LONGEVITY table (species_symbol -> years) from species_longevity_ref
# in the cohorts DB. Returns nothing (with a warning) if the table is absent. See species_longevity_ref_README.md.
function _load_longevity_table(db_path::String)
  # READ-ONLY attach (in-memory main + ATTACH … READ_ONLY) — never open the cohorts DB read-write just to
  # SELECT; a read-write open here risks corrupting the DB if the process is killed mid-load.
  con = DuckDB.connect(DuckDB.DB())
  isfile(db_path) || error("longevity_from_data: cohort database not found: $db_path")
  DuckDB.execute(con, "ATTACH '$(PanCore.sql_quote_path(db_path))' AS src (READ_ONLY);")
  has = (DuckDB.execute(con, "SELECT count(*) AS n FROM information_schema.tables WHERE table_catalog = 'src' AND table_name = 'species_longevity_ref'") |> DataFrame).n[1]
  has == 0 && error("longevity_from_data=true but table species_longevity_ref is not in $db_path (set longevity_from_data: false to keep LONGEVITY in the search)")   # used to warn and silently keep LONGEVITY in the search
  df = DuckDB.execute(con, "SELECT species_symbol, longevity FROM src.species_longevity_ref WHERE longevity IS NOT NULL ORDER BY species_symbol") |> DataFrame
  d = Dict{String,Float64}(String(r.species_symbol) => Float64(r.longevity) for r in eachrow(df))
  @info "longevity_from_data: pinned LONGEVITY for $(length(d)) species from species_longevity_ref (out of search)"
  d
end

# Load per-species SHADE_TOL (shade class 1-5) from a CSV (cols: sym, shade_class), e.g.
# runs/shadetol_all_species.csv. Pins SHADE_TOL out of the search. Missing/absent → SHADE_TOL_DEFAULT.
function _load_shade_tol_table(csv_path::String)
  isfile(csv_path) || error("shade_tol_from_data=true but $csv_path not found (set shade_tol_from_data: false to keep SHADE_TOL in the search)")   # used to warn and silently keep SHADE_TOL in the search
  df = CSV.read(csv_path, DataFrame)
  for c in ("sym", "shade_class")
    c in names(df) || error("shade_tol_csv $csv_path: column '$c' not found (columns: $(join(names(df), ", ")))")
  end
  d = Dict{String,Int}(uppercase(strip(String(r.sym))) => Int(r.shade_class) for r in eachrow(df))
  @info "shade_tol_from_data: pinned SHADE_TOL for $(length(d)) species from $csv_path (out of search)"
  d
end

# Load per-(category, L3, land_use) PROB_ESTAB from the natural + artificial rollup CSVs (cols: category,
# l3, prob_estab). natural CSV → natural strata, artificial CSV → artificial strata. Seeds (not pins) the
# initial PROB_ESTAB_SPP → the run becomes a calibration around the data-derived rates.
function _load_prob_estab_table(csv_nat::String, csv_art::String)
  d = Dict{Tuple{String,String,String},Float64}()
  for (path, lu) in ((csv_nat, "natural"), (csv_art, "artificial"))
    isempty(path) && continue
    isfile(path) || error("prob_estab_from_data=true but $path ($lu table) not found")   # used to warn (natural) or skip silently (artificial)
    df = CSV.read(path, DataFrame)
    for r in eachrow(df)
      d[(uppercase(strip(String(r.category))), String(r.l3), lu)] = Float64(r.prob_estab)
    end
  end
  isempty(d) && error("prob_estab_from_data=true but no rows were loaded from $csv_nat / $csv_art")   # used to warn and leave PROB_ESTAB as-is
  @info "prob_estab_from_data: loaded $(length(d)) (category,l3,lu) PROB_ESTAB values"
  d
end

_parse_eco_lu(s) = (p = split(String(s), "|lu="); length(p) == 2 ? (String(p[1]), String(p[2])) : (String(s), ""))

# Seed the initial candidate's PROB_ESTAB_SPP[eco][sp] from the data table (in place; stays in the search).
function _seed_prob_estab!(tables::BiomassSuccessionPlugin.BSTables, params, eco_list, species_list, eco_species_ids)
  tbl = tables.prob_estab
  isnothing(tbl) && return params
  n = 0
  for (eco_id, sp_ids) in enumerate(eco_species_ids)
    l3, lu = _parse_eco_lu(eco_list[eco_id])
    lu in ("natural", "artificial") || (lu = "natural")   # site_class_strata puts a site-tier (A/B/C/D) in |lu=; establishment prob is site-tier-independent → use the natural model input
    for (j, gsp) in enumerate(sp_ids)
      v = get(tbl, (uppercase(species_list[gsp]), l3, lu), nothing)
      v === nothing && continue
      params.PROB_ESTAB_SPP[eco_id][j] = FloatType(v); n += 1
    end
  end
  @info "prob_estab_from_data: seeded PROB_ESTAB for $n eco×species cells (calibration start)"
  params
end

# Per-(category, L3, land_use) B_MAX floor (g/m²) from runs/bmax_floor.csv (cols: category,l3,lu,bmax_floor).
function _load_bmax_floor_table(csv_path::String)
  if !isfile(csv_path)
    @warn "bmax_floor_from_data=true but $csv_path not found — B_MAX floor left flat [12000,35000]"; return nothing
  end
  df = CSV.read(csv_path, DataFrame)
  d = Dict{Tuple{String,String,String},Float64}()
  for r in eachrow(df)
    d[(uppercase(strip(String(r.category))), String(r.l3), String(r.lu))] = Float64(r.bmax_floor)
  end
  isempty(d) && (@warn "bmax_floor: no rows in $csv_path"; return nothing)
  @info "bmax_floor_from_data: loaded $(length(d)) (category,l3,lu) B_MAX floors from $csv_path"
  d
end

# ANPP_MAX data floor — parallel to _load_bmax_floor_table. CSV cols: category,l3,lu,anpp_floor (+ extras
# ignored). anpp_floor = p99 of per-cohort agb/age (a valid lower bound on ANPP_MAX; see plugin growth eq).
function _load_anpp_floor_table(csv_path::String)
  if !isfile(csv_path)
    @warn "anpp_floor_from_data=true but $csv_path not found — ANPP floor off"; return nothing
  end
  df = CSV.read(csv_path, DataFrame)
  col = "anpp_floor" in names(df) ? :anpp_floor : (:anpp_floor_pai in propertynames(df) ? :anpp_floor_pai : nothing)
  isnothing(col) && (@warn "anpp_floor: no anpp_floor column in $csv_path"; return nothing)
  d = Dict{Tuple{String,String,String},Float64}()
  for r in eachrow(df)
    v = getproperty(r, col)
    ismissing(v) && continue
    d[(uppercase(strip(String(r.category))), String(r.l3), String(r.lu))] = Float64(v)
  end
  isempty(d) && (@warn "anpp_floor: no rows in $csv_path"; return nothing)
  @info "anpp_floor_from_data: loaded $(length(d)) (category,l3,lu) ANPP floors from $csv_path (col=$col)"
  d
end

# Per-category MATURITY (SONA age) from prob_estab_all_species.csv (cols: category, maturity). One value per
# category (constant across L3); median if a category has several. Seeds (not pins) MATURITY.
function _load_maturity_table(csv_path::String)
  if !isfile(csv_path)
    @warn "maturity_from_data=true but $csv_path not found — MATURITY left as-is"; return nothing
  end
  df = CSV.read(csv_path, DataFrame)
  d = Dict{String,Int}()
  for g in groupby(df, :category)
    d[uppercase(strip(String(g.category[1])))] = round(Int, Statistics.median(Float64.(g.maturity)))
  end
  @info "maturity_from_data: loaded MATURITY for $(length(d)) categories from $csv_path"
  d
end

# Seed the initial candidate's per-species MATURITY from the table (in place; stays in the search).
function _seed_maturity!(tables::BiomassSuccessionPlugin.BSTables, params, species_list)
  tbl = tables.maturity
  isnothing(tbl) && return params
  n = 0
  for gsp in eachindex(species_list)
    v = get(tbl, uppercase(species_list[gsp]), nothing)
    v === nothing && continue
    params.MATURITY[gsp] = FloatType(v); n += 1
  end
  @info "maturity_from_data: seeded MATURITY for $n species (calibration start)"
  params
end

# Data-derived MIN_REL_BIOMASS default (base 0.10 + fixed 0.175 spacing; see FIA_DATA_PREP shade analysis).
const MIN_REL_PINNED = FloatType[0.10, 0.275, 0.45, 0.625, 0.80]
# ---------------------------------------------------------------------------------------------------------------
# Single source of truth for the DATA-PREP keyword arguments (train/val/test split, region filters, site-class
# tiers, species tiering, outlier exclusions, ...). Used by `run_from_yaml` (training) AND by the analysis scripts
# (`prepare_from_cfg`), so a script re-derives EXACTLY the plots/split the run trained on.
# ---------------------------------------------------------------------------------------------------------------
