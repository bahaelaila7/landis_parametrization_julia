# Scenario engine: project FIA plots (Forest Explorer / TreeMap) forward with any succession plugin and a YAML parameter
# set, under a management plan (thin / harvest / plant) and, for PnET, a climate scenario. docs/scenario_engine.md.
#
#   ps  = load_paramset(yaml_text)
#   out = run_scenario(ps, [(cn = "<FIA PLOT.CN>", acres = 12.3), …], plan; climate = nothing)
# Each plot: initial cohorts = the curated FIA cohorts of the plot visit that CN identifies (phantom backfill excluded),
# summed over subplots per (species, age) as in training; eco cell from (EPA L3, site class) via the YAML's eco_mapping;
# species via its species_map. Simulation is annual; metrics are recorded every `period` years (start-of-cycle state,
# before that cycle's management) and management actions fire at elapsed = cycle × period.
import Dates


# ---------------------------------------------------------------------------------------------------------------------
# Configuration: data sources and lazy caches (Step 3f). Nothing is read from the environment at module load (so nothing is
# baked into a sysimage); a `ScenarioConfig` is built at call time and passed through the scenario API (a `ParamSet` carries
# the one it was loaded with).
# ---------------------------------------------------------------------------------------------------------------------
# one volume equation: v = cf · exp(b0) · ba^b1 · H^b2 (see scenario_volume.jl)
struct VolEq
  b0::Float64; b1::Float64; b2::Float64; cf::Float64
  min_dbh::Float64; sq_a::Float64; sq_c::Float64
end

"""
    ScenarioCache()

The lazily loaded tables of ONE `ScenarioConfig`, guarded by a lock (the worker may serve requests from several tasks). The cache is
owned by its config, so a different config (other paths) never sees this config's tables and a config's tables die with it - no cache
key to get wrong. `get_cached!(f, cache, :slot)` builds a slot once under the lock.
"""
Base.@kwdef mutable struct ScenarioCache
  lock::ReentrantLock = ReentrantLock()
  species_codes::Union{Nothing,Dict{String,Int}} = nothing         # REF_SPECIES: symbol → SPCD
  donors::Union{Nothing,Dict{String,String}} = nothing             # TreeMap plot CN → donor CN
  pnet_base::Union{Nothing,PnETPlugin.PnETContext} = nothing       # PnET constants + 1980-2025 climate cache
  site_table::Union{Nothing,Dict{String,NamedTuple}} = nothing     # per EPA L3: FIA plot coordinates + site classes
  voleq::Union{Nothing,Dict{Tuple{String,String,String},VolEq}} = nothing   # volume-equation table
  spgrp::Union{Nothing,Dict{Int,Int}} = nothing                    # SPCD → FIA species group
  voleq_by_set::Dict{String,Vector{NTuple{2,VolEq}}} = Dict{String,Vector{NTuple{2,VolEq}}}()   # ParamSet.id → (tcuft, bdft) per global species
end

"The slot `slot` of the cache, built by `f()` under the cache lock the first time it is needed."
function get_cached!(f, cache::ScenarioCache, slot::Symbol)
  lock(cache.lock) do
    v = getfield(cache, slot)
    v === nothing || return v
    setfield!(cache, slot, f())
  end
end

"""
    ScenarioConfig(; fia_db, cohort_table, soil_csv, pnet_climate, co2_csv, donors, climate_db, solus, volume_eqs)

Where the scenario engine reads its data. Every default is resolved when the config is BUILT (env `PAN_FIA_DB`, `PAN_COHORT_TABLE`,
`PAN_SOIL_CSV`, `PAN_PNET_CLIMATE`, `PAN_CO2_CSV`, `PAN_DONORS`, `PAN_CLIMATE_DB`, `PAN_SOLUS`, `PAN_VOLUME_EQS`), never at module load.
`cache` holds the lazily loaded tables of this config. Build it once (tools/pan_worker.jl does) and pass it to `load_paramset`
(`run_scenario` uses the set's) and `climate_preview`.
"""
Base.@kwdef struct ScenarioConfig
  fia_db::String = get(ENV, "PAN_FIA_DB", "/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb")
  cohort_table::String = get(ENV, "PAN_COHORT_TABLE", "curated_cohorts_landis_stdorg")
  soil_csv::String = get(ENV, "PAN_SOIL_CSV", "/workspace/landis_region4/data_r4/plot_soil_r4.csv")
  pnet_climate::String = get(ENV, "PAN_PNET_CLIMATE", "/workspace/landis_region4/data_r4/pnet_climate_r4_l4.jld2")
  co2_csv::String = get(ENV, "PAN_CO2_CSV", "/workspace/landis_region4/data_r4/co2_annual.csv")
  donors::String = get(ENV, "PAN_DONORS", normpath(joinpath(@__DIR__, "..", "data", "treemap_donors.csv")))
  climate_db::String = get(ENV, "PAN_CLIMATE_DB", "/workspace/pan_climate/pan_climate.duckdb")
  solus::String = get(ENV, "PAN_SOLUS", "/workspace/SOLUS/SOLUS100")
  volume_eqs::String = get(ENV, "PAN_VOLUME_EQS", normpath(joinpath(@__DIR__, "..", "data", "volume_eqs_r4.csv")))
  cache::ScenarioCache = ScenarioCache()
end

# unit conversions
const SC_HA_PER_AC = 0.40468564224
const SC_M2HA_TO_FT2AC = 4.356            # m²/ha → ft²/ac
const SC_M3HA_TO_FT3AC = 14.2913          # m³/ha → ft³/ac
const SC_ROOT_FRAC = 0.25                 # belowground/aboveground live biomass (Cairns et al. 1997 mean) for carbon_total
const SC_CARBON = 0.5                     # carbon fraction of dry biomass

# ---------------------------------------------------------------------------------------------------------------------
# Parameter sets
# ---------------------------------------------------------------------------------------------------------------------
struct ParamSet
  id::String
  params::Any
  meta::Any
  plugin::Type
  sp_index::Dict{String,Int}
  eco_index::Dict{String,Int}
  local_index::Vector{Dict{Int,Int}}   # per eco: global species → eco-local index
  spcd::Dict{String,Int}               # parameter species → FIA SPCD (groups: negative synthetic codes)
  scen::ScenarioConfig                 # the data sources (and caches) this set was loaded against
  hb::HBADPlugin.HBADContext           # HBAD priors of THIS set's species list: cohort-structure translation (TPA/QMD/height) for every plugin; the HBAD run context
end

function load_paramset(text::AbstractString, scen::ScenarioConfig=ScenarioConfig())
  p, meta = params_from_yaml(text)
  SP = succession_plugin_for(p)
  sp_index = Dict(String(s) => k for (k, s) in enumerate(p.SPECIES_LIST))
  eco_index = Dict(String(e) => k for (k, e) in enumerate(p.ECO_LIST))
  local_index = [Dict(Int(g) => j for (j, g) in enumerate(ids)) for ids in p.ECO_SPECIES_IDS]
  # cohort-structure translation (TPA/QMD/height) for every plugin uses the HBAD priors of this species list (an immutable context owned by the set)
  hb = HBADPlugin.HBADContext(String.(p.SPECIES_LIST), [Int.(ids) for ids in p.ECO_SPECIES_IDS])
  spcd = Dict{String,Int}(); k = 0
  codes = _species_codes(scen)
  for s in p.SPECIES_LIST
    c = get(codes, uppercase(String(s)), 0)
    spcd[String(s)] = c > 0 ? c : (k -= 1)
  end
  ParamSet(string(hash(text); base=16, pad=16), p, meta, SP, sp_index, eco_index, local_index, spcd, scen, hb)
end

_species_codes(scen::ScenarioConfig) = get_cached!(scen.cache, :species_codes) do
  db = DuckDB.DB(scen.fia_db; readonly=true); con = DuckDB.DBInterface.connect(db)
  d = Dict{String,Int}()
  try
    for r in DuckDB.DBInterface.execute(con, "SELECT UPPER(TRIM(SPECIES_SYMBOL)) s, SPCD c FROM REF_SPECIES WHERE SPECIES_SYMBOL IS NOT NULL")
      d[String(r.s)] = Int(r.c)
    end
  finally
    DuckDB.DBInterface.close!(con); DuckDB.close(db)
  end
  d
end

# ---------------------------------------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------------------------------------
"""Initial state of FIA plots identified by PLOT.CN: one row per plot (meta) + per-plot cohort tables (symbol, age, agb)."""
function load_scenario_plots(scen::ScenarioConfig, cns::Vector{String})
  db = DuckDB.DB(); con = DuckDB.DBInterface.connect(db)   # in-memory (temp views allowed) + FIA DB attached read-only
  DuckDB.execute(con, "ATTACH '$(scen.fia_db)' AS src (READ_ONLY)")
  try
    DuckDB.register_data_frame(con, DataFrame(cn=unique(cns)), "req_cns")
    pl = DataFrame(DuckDB.DBInterface.execute(con, """
      SELECT p.CN cn, p.STATECD s, p.UNITCD u, p.COUNTYCD c, p.PLOT p, p.INVYR invyr, p.MEASYEAR my, p.MEASMON mm, p.MEASDAY md, p.LAT lat, p.LON lon
      FROM src.PLOT p JOIN req_cns r ON p.CN = r.cn
      ORDER BY p.CN"""))
    sc = DataFrame(DuckDB.DBInterface.execute(con, """
      WITH cr AS (SELECT co.PLT_CN cn, co.SITECLCD sc, SUM(co.CONDPROP_UNADJ) w FROM src.COND co JOIN req_cns r ON co.PLT_CN = r.cn
                  WHERE co.COND_STATUS_CD = 1 AND co.SITECLCD IS NOT NULL GROUP BY ALL),
           rk AS (SELECT cn, sc, ROW_NUMBER() OVER (PARTITION BY cn ORDER BY w DESC, sc) rn FROM cr)
      SELECT cn, sc FROM rk WHERE rn = 1"""))
    scmap = Dict(String(r.cn) => Int(r.sc) for r in eachrow(sc))
    pl.key = [string(r.s, "_", r.u, "_", r.c, "_", r.p) for r in eachrow(pl)]
    DuckDB.register_data_frame(con, DataFrame(rkey=unique(pl.key)), "req_keys")
    co = DataFrame(DuckDB.DBInterface.execute(con, """
      SELECT plt_cn AS pkey, measdate, UPPER(TRIM(species_symbol)) sym, age_calc age, SUM(agb) agb, any_value(epa_l3) l3, any_value(epa_l4) l4
      FROM src.$(scen.cohort_table) t JOIN req_keys k ON t.plt_cn = k.rkey
      WHERE coalesce(t.intro_type, '') <> 'phantom_backfill' AND t.age_calc > 0
      GROUP BY plt_cn, measdate, UPPER(TRIM(species_symbol)), age_calc
      ORDER BY plt_cn, measdate, sym, age"""))
    byk = Dict{String,DataFrame}(String(k.pkey) => DataFrame(g) for (k, g) in pairs(groupby(co, :pkey)))
    out = NamedTuple[]
    for r in eachrow(pl)
      want = Dates.Date(coalesce(r.my, r.invyr), Int(coalesce(r.mm, 6)), coalesce(r.md, 15) isa Number ? Int(coalesce(r.md, 15)) : 15)
      g = get(byk, r.key, nothing)
      if g === nothing || nrow(g) == 0
        push!(out, (cn=String(r.cn), key=r.key, status="no_data", measdate=want, l3="", l4="", siteclcd=get(scmap, String(r.cn), 0),
                    lat=Float64(coalesce(r.lat, NaN)), cohorts=DataFrame(sym=String[], age=Int[], agb=Float64[]))); continue
      end
      dates = unique(g.measdate); d = dates[argmin(abs.(Dates.value.(dates .- want)))]
      gg = g[g.measdate .== d, :]
      abs(Dates.value(d - want)) > 400 && @warn "plot $(r.cn): nearest curated visit $(d) is $(Dates.value(d - want)) days from the CN's measurement $(want)"
      push!(out, (cn=String(r.cn), key=r.key, status="ok", measdate=d, l3=String(gg.l3[1]), l4=String(coalesce(gg.l4[1], "")),
                  siteclcd=get(scmap, String(r.cn), 0), lat=Float64(coalesce(r.lat, NaN)),
                  cohorts=DataFrame(sym=String.(gg.sym), age=Int.(gg.age), agb=Float64.(gg.agb))))
    end
    found = Set(String.(pl.cn))
    for cn in unique(cns)
      cn in found || push!(out, (cn=cn, key="", status="no_data", measdate=Dates.Date(2000), l3="", l4="", siteclcd=0, lat=NaN,
                                 cohorts=DataFrame(sym=String[], age=Int[], agb=Float64[])))
    end
    return out
  finally
    DuckDB.DBInterface.close!(con); DuckDB.close(db)
  end
end

"TreeMap plot CN → donor CN for plots without curated cohorts (data/treemap_donors.csv, docs/treemap_donors.md)."
_donor_map(scen::ScenarioConfig) = get_cached!(scen.cache, :donors) do
  d = Dict{String,String}()
  if isfile(scen.donors)
    df = CSV.read(scen.donors, DataFrame; types=Dict(:cn => String, :donor_cn => String))
    for r in eachrow(df)
      ismissing(r.donor_cn) || isempty(r.donor_cn) || (d[r.cn] = r.donor_cn)
    end
  else
    @warn "TreeMap donor table not found" scen.donors
  end
  d
end

"""
    load_scenario_units(plots, warnings) -> rows like `load_scenario_plots` plus `donor` (CN or "") and `lon`

The FIA plot (TreeMap imputation) supplies the vegetation; a plot with no curated cohorts (mostly pre-2008 FIA visits) takes
the cohorts of its TreeMap donor (same forest type, nearest in BA/QMD/height/TPA/canopy). When a plot carries a location
(`l4`, `lon`, `lat` = majority EPA L4 and centroid of its pixels in the AOI), the ecoregion (eco cell, PnET climate), the
site class and the PnET soils come from that location instead of from where the FIA plot was measured.
"""
function load_scenario_units(scen::ScenarioConfig, plots, warnings)
  info = load_scenario_plots(scen, String[String(x.cn) for x in plots])
  dm = _donor_map(scen)
  need = [r.cn for r in info if r.status == "no_data" && haskey(dm, r.cn)]
  dinfo = isempty(need) ? Dict{String,Any}() : Dict{String,Any}(d.cn => d for d in load_scenario_plots(scen, unique([dm[c] for c in need])))
  loc = Dict{String,Any}()
  for x in plots
    l4 = hasproperty(x, :l4) ? String(something(x.l4, "")) : ""
    l3 = isempty(l4) ? "" : l3_of_l4(l4)
    isempty(l3) || (loc[String(x.cn)] = (l4=l4, l3=l3, lon=Float64(x.lon), lat=Float64(x.lat)))
  end
  acres = Dict(String(x.cn) => Float64(get(x, :acres, 1.0)) for x in plots)
  out = NamedTuple[]; ndon = 0; adon = 0.0; nloc = 0; nsc = 0
  for r in info
    donor = ""
    if r.status == "no_data" && haskey(dm, r.cn)
      d = get(dinfo, dm[r.cn], nothing)
      if d !== nothing && d.status == "ok"
        r = merge(d, (cn=r.cn,)); donor = d.cn; ndon += 1; adon += get(acres, r.cn, 0.0)
      end
    end
    lon = NaN
    if r.status == "ok" && haskey(loc, r.cn)
      L = loc[r.cn]; sc = site_class_at(scen, L.l3, L.lon, L.lat)
      sc > 0 && (nsc += 1)
      r = merge(r, (l3=L.l3, l4=L.l4, lat=L.lat, siteclcd=(sc > 0 ? sc : r.siteclcd))); lon = L.lon; nloc += 1
    end
    push!(out, merge(r, (donor=donor, lon=lon)))
  end
  tot = sum(values(acres); init=0.0)
  ndon > 0 && push!(warnings, "$ndon plots ($(round(100 * adon / max(tot, 1e-9); digits=1)) % of the area) have no curated FIA cohorts " *
                              "(mostly pre-2008 visits); their vegetation comes from a TreeMap donor plot of the same forest type")
  nok = count(r -> r.status == "ok", out)
  nok > 0 && nloc < nok && push!(warnings, "$(nok - nloc) of $nok plots have no AOI location; their ecoregion, site class and soils come from the FIA plot's own location")
  out
end

"""Eco cell of a plot under the param set's eco_mapping (fallbacks: 'NA' cell, then the L3's first available cell)."""
function scenario_eco(ps::ParamSet, l3::AbstractString, siteclcd::Integer)
  em = ps.meta.eco_mapping
  cells = get(em, "site_class_cells", Dict("A" => [1, 2, 3], "B" => [4], "C" => [5], "D" => [6, 7]))
  cell_l3s = Set(String.(string.(get(em, "cell_l3s", String[]))))
  cell = if String(l3) in cell_l3s
    siteclcd == 0 ? String(get(em, "missing_cell", "NA")) :
      something(findfirst(v -> siteclcd in v, Dict(String(k) => Int.(v) for (k, v) in cells)), String(get(em, "missing_cell", "NA")))
  else
    String(get(em, "other_cell", "X"))
  end
  e = get(ps.eco_index, "$(l3)|lu=$(cell)", 0)
  e > 0 && return e, ""
  e = get(ps.eco_index, "$(l3)|lu=NA", 0)
  e > 0 && return e, "eco cell $(l3)|lu=$(cell) not in the parameter set; used $(l3)|lu=NA"
  for (k, name) in enumerate(ps.params.ECO_LIST)
    startswith(String(name), String(l3) * "|") && return k, "eco cell $(l3)|lu=$(cell) not in the parameter set; used $(name)"
  end
  return 0, "EPA L3 $(l3) is outside the parameter set's region"
end

"""Map a FIA symbol to the eco-local parameter species index (0 if the species cannot be represented in that eco)."""
function scenario_species(ps::ParamSet, eco::Int, sym::AbstractString)
  t = get(ps.meta.species_map, uppercase(String(sym)), "")
  g = get(ps.sp_index, t, 0)
  g > 0 && (j = get(ps.local_index[eco], g, 0); j > 0 && return j)
  # the target is not present in this eco cell: fall back to a catch-all that is
  for alt in ("_H_HST", "_H_LST", "_H_NA", "_H", "_S_LST", "_S_HST", "_S_NA", "_S")
    g = get(ps.sp_index, alt, 0); g == 0 && continue
    j = get(ps.local_index[eco], g, 0); j > 0 && return j
  end
  0
end

# ---------------------------------------------------------------------------------------------------------------------
# Cohort structure (TPA / QMD / height) — HBAD state, else the empirical translation
# ---------------------------------------------------------------------------------------------------------------------
@inline _gsp(ps::ParamSet, eco::Integer, esp::Integer) = Int(ps.params.ECO_SPECIES_IDS[eco][esp])
function _cohort_struct(ps::ParamSet, site, k::Int)
  if ps.plugin === HBADPlugin.HBAD
    return Float64(site.c_tph[k]), Float64(site.c_dq[k]), Float64(site.c_h[k])
  end
  bio = Float64(site.c_bio[k]); bio > 0 || return 0.0, 0.0, 0.0
  pr = ps.hb.priors.sp[_gsp(ps, site.eco_id, site.c_species[k])]
  HBADPlugin.translate(pr, max(Float64(site.c_age[k]), 1.0), bio)
end

struct SCMetrics
  year::Int; ba::Float64; tpa::Float64; qmd::Float64; tcuft::Float64; bdft::Float64
  rem_tcuft::Float64; rem_bdft::Float64; carbon_total::Float64; carbon_live_ag::Float64; carbon_removed::Float64
  agb::Float64        # g/m² (Pan's native output, for reference)
end

# stand summary of one site + its species×DBH tally (2-inch classes, FE convention; cohort QMD stands for its trees)
function _site_summary(ps::ParamSet, site)
  ba = 0.0; tph = 0.0; vol = 0.0; bd = 0.0; agb = 0.0
  dist = Dict{Tuple{Int16,Int},NTuple{2,Float64}}()
  veq = volume_eqs(ps)
  for k in 1:Int(site.live)
    t, dq, h = _cohort_struct(ps, site, k); b = Float64(site.c_bio[k]); agb += b
    t > 0 || continue
    gsp = _gsp(ps, site.eco_id, site.c_species[k])
    bak = pi / 4 * (dq / 100)^2 * t          # m²/ha
    v, bf = cohort_volume(veq[gsp], t, dq, h) # ft³/ac (total stem), bd ft/ac (Int'l ¼)
    ba += bak; tph += t; vol += v; bd += bf
    sp = String(ps.params.SPECIES_LIST[gsp])
    key = (Int16(ps.spcd[sp]), min(2 * floor(Int, dq / 2.54 / 2), 40))
    c, bb = get(dist, key, (0.0, 0.0))
    dist[key] = (c + t * SC_HA_PER_AC, bb + bak * SC_M2HA_TO_FT2AC)
  end
  tpa = tph * SC_HA_PER_AC; baf = ba * SC_M2HA_TO_FT2AC
  qmd = tpa > 0 ? sqrt(baf / (0.005454154 * tpa)) : 0.0
  (ba=baf, tpa=tpa, qmd=qmd, tcuft=vol, bdft=bd, agb=agb,
   c_ag=agb * SC_CARBON * 0.01, dist=dist)
end

# ---------------------------------------------------------------------------------------------------------------------
# Management
# ---------------------------------------------------------------------------------------------------------------------
_sel_species(spec) = (s = strip(String(something(spec, "all"))); (isempty(s) || lowercase(s) == "all") ? nothing : Set(uppercase.(strip.(split(s, ',')))))

# remove cohort k (all plugins: swap-remove moves every per-cohort field)
function _remove_cohort!(site, k::Int)
  last = Int(site.live)
  k != last && PanCore.swap_cohort!(site, k, last)
  site.live = UIntType(last - 1)
  site.old = min(site.old, site.live)
end

"""Apply one removal to the selected cohorts of a site. Returns (removed AGB g/m², removed total cubic ft³/ac, removed board feet/ac)."""
function _apply_removal!(ps::ParamSet, pc, site, act, sym_ok::Function)
  n = Int(site.live); n == 0 && return 0.0, 0.0, 0.0
  st = [_cohort_struct(ps, site, k) for k in 1:n]; veq = volume_eqs(ps)
  sel = Int[]
  sl = get(act, "selection", Dict{String,Any}())
  alo = Float64(get(sl, "age_lo", 0)); ahi = Float64(get(sl, "age_hi", 9999))
  dlo = Float64(get(sl, "dbh_lo", get(act, "dbh_lo", 0.0))); dhi = Float64(get(sl, "dbh_hi", get(act, "dbh_hi", 999.0)))
  for k in 1:n
    sp = String(ps.params.SPECIES_LIST[_gsp(ps, site.eco_id, site.c_species[k])])
    sym_ok(sp) || continue
    a = Float64(site.c_age[k]); dqi = st[k][2] / 2.54
    (alo <= a <= ahi && dlo <= dqi <= dhi) || continue
    push!(sel, k)
  end
  isempty(sel) && return 0.0, 0.0, 0.0
  rm = get(act, "removal", Dict{String,Any}())
  typ = String(get(rm, "type", "percent_biomass")); val = Float64(get(rm, "value", 0.0)); dir = String(get(rm, "direction", "below"))
  f = zeros(n)
  metric(k) = typ == "residual_ba" ? pi / 4 * (st[k][2] / 100)^2 * st[k][1] * SC_M2HA_TO_FT2AC :
              typ == "residual_tpa" ? st[k][1] * SC_HA_PER_AC :
              st[k][1] * SC_HA_PER_AC * (st[k][2] / 25.4)^1.605   # residual_sdi (summation SDI, imperial)
  if typ == "clearcut"
    f[sel] .= 1.0
  elseif typ == "percent_biomass"
    f[sel] .= clamp(val / 100, 0.0, 1.0)
  else
    total = sum(metric(k) for k in 1:n); excess = total - val
    if excess > 0
      if dir == "proportional"
        s = sum(metric(k) for k in sel); f[sel] .= s > 0 ? clamp(excess / s, 0.0, 1.0) : 0.0
      else
        order = sort(sel; by=k -> st[k][2], rev=(dir == "above"))
        for k in order
          excess <= 1e-9 && break
          c = metric(k); c > 0 || continue
          f[k] = min(1.0, excess / c); excess -= f[k] * c
        end
      end
    end
  end
  rb = 0.0; rv = 0.0; rs = 0.0
  # iterate high→low index so swap-removals do not disturb unvisited cohorts
  for k in n:-1:1
    f[k] > 0 || continue
    b = Float64(site.c_bio[k]); t, dq, h = st[k]
    v, bf = cohort_volume(veq[_gsp(ps, site.eco_id, site.c_species[k])], t, dq, h)
    rb += b * f[k]; rv += v * f[k]; rs += bf * f[k]
    if f[k] >= 0.999 || b * (1 - f[k]) < 1.0
      _remove_cohort!(site, k)
    else
      set_cohort_bio!(pc, site, k, FloatType(b * (1 - f[k])))
    end
  end
  hasproperty(site, :B) && (site.B = FloatType(sum(Float64(site.c_bio[k]) for k in 1:Int(site.live); init=0.0)))
  rb, rv, rs
end

"""Plant `tpa` trees/ac (× survival) of FIA symbol `sym` on a site (new cohort, age 1, QMD 1 cm). Returns false if the species
cannot be represented in the site's eco cell. CSR capacity must have been reserved by the caller."""
function _plant!(ps::ParamSet, pc, site, sym::AbstractString, tpa::Real, survival::Real)
  esp = scenario_species(ps, Int(site.eco_id), sym); esp == 0 && return false
  tph = Float64(tpa) / SC_HA_PER_AC * clamp(Float64(survival), 0, 100) / 100
  tph > 0 || return true
  pr = ps.hb.priors.sp[_gsp(ps, site.eco_id, esp)]
  dq = 1.0; agb = HBADPlugin.agb_of(pr, tph, dq)
  add_cohort!(pc, site, UIntType(esp), FloatType(1), FloatType(agb))
  k = Int(site.live)
  set_cohort_bio!(pc, site, k, FloatType(agb))
  if ps.plugin === HBADPlugin.HBAD          # keep the planted density/size (add_cohort! re-translates from AGB)
    site.c_tph[k] = FloatType(tph); site.c_dq[k] = FloatType(dq)
    site.c_h[k] = FloatType(1.3 + HBADPlugin.chapman_richards(pr.hd_a, pr.hd_b, pr.hd_c, dq)); site.c_bio[k] = FloatType(agb)
  end
  true
end

# normalize the FE plan's actions into events: (cycle, kind, action dict)
function _plan_events(plan, cycles::Int)
  ev = Tuple{Int,String,Dict{String,Any}}[]
  for a in get(plan, "actions", Any[])
    d = Dict{String,Any}(String(k) => v for (k, v) in pairs(a)); kind = String(get(d, "kind", "grow"))
    kind == "grow" && continue
    c0 = Int(get(d, "cycle", 1)); rep = Int(something(get(d, "repeat_every", 0), 0))
    cs = rep > 0 ? collect(c0:rep:cycles) : [c0]
    if kind == "thin"   # FE/FVS thin → removal to a residual BA/TPA/SDI
      m = uppercase(String(get(d, "metric", "BA"))); tgt = Float64(get(d, "target", 0.0))
      d = Dict{String,Any}("kind" => "harvest", "selection" => Dict("species" => get(d, "species", "all"), "dbh_lo" => get(d, "dbh_lo", 0.0), "dbh_hi" => get(d, "dbh_hi", 999.0)),
        "removal" => Dict("type" => tgt <= 0 ? "clearcut" : m == "TPA" ? "residual_tpa" : m == "SDI" ? "residual_sdi" : "residual_ba",
                          "value" => tgt, "direction" => get(d, "direction", "below")),
        "stands" => Dict("rank" => "all", "area_pct" => 100))
      kind = "harvest"
    end
    for c in cs; 0 <= c <= cycles && push!(ev, (c, kind, d)); end
  end
  ev
end

# ---------------------------------------------------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------------------------------------------------
"""
    run_scenario(ps, plots, plan; climate=nothing, seed=1, establishment=true, progress=nothing)

`plots`: vector of (cn, acres). `plan`: Dict with cycles, period, actions (FE plan JSON). Returns
Dict(cn => (status, message, metrics::Vector{SCMetrics}, dist::Dict{Int,Dict{(spcd,dbh_lo)=>(tpa,ba)}}, spsym)) and warnings.
"""
function run_scenario(ps::ParamSet, plots, plan; climate=nothing, seed::Integer=1, establishment::Bool=true, progress=nothing)
  SP = ps.plugin; p = ps.params
  cycles = Int(get(plan, "cycles", 10)); period = Int(get(plan, "period", 10)); horizon = cycles * period
  warnings = String[]
  info = load_scenario_units(ps.scen, plots, warnings)
  acres = Dict(String(x.cn) => Float64(x.acres) for x in plots)
  results = Dict{String,Any}()
  rows = NamedTuple[]; meta = NamedTuple[]
  ndrop = 0.0; ntot = 0.0
  for r in info
    if r.status != "ok"
      results[r.cn] = (status=r.status, message="no FIA cohort data for this plot visit", metrics=SCMetrics[], dist=Dict{Int,Any}(), spsym=Dict{Int16,String}()); continue
    end
    e, msg = scenario_eco(ps, r.l3, r.siteclcd)
    if e == 0
      results[r.cn] = (status="out_of_region", message=msg, metrics=SCMetrics[], dist=Dict{Int,Any}(), spsym=Dict{Int16,String}()); continue
    end
    isempty(msg) || push!(warnings, msg)
    pid = length(meta) + 1
    push!(meta, (cn=r.cn, plot_id=pid, eco=e, measdate=r.measdate, l4=r.l4, l3=r.l3, lat=r.lat, lon=r.lon, key=r.key))
    for c in eachrow(r.cohorts)
      ntot += c.agb
      j = scenario_species(ps, e, c.sym)
      if j == 0; ndrop += c.agb; continue; end
      push!(rows, (plot_id=UIntType(pid), eco_id=e, eco_species_id=UIntType(j), species_id=UIntType(_gsp(ps, e, j)),
                   age_calc=FloatType(c.age), agb_sum=FloatType(c.agb), measdate=r.measdate, start_measdate=r.measdate, sim_year=0))
    end
  end
  ndrop > 0 && push!(warnings, "$(round(100 * ndrop / max(ntot, 1e-9), digits=2)) % of initial AGB is in species the parameter set cannot represent in those eco cells (dropped)")
  isempty(meta) && return results, warnings
  splots = DataFrame(rows)
  # plots with no representable cohort still simulate (empty start, regeneration only): give them a placeholder row? no — report
  have = Set(Int.(splots.plot_id))
  for m in meta
    Int(m.plot_id) in have || push!(rows, (plot_id=UIntType(m.plot_id), eco_id=m.eco, eco_species_id=UIntType(1), species_id=UIntType(_gsp(ps, m.eco, 1)),
                                         age_calc=FloatType(1), agb_sum=FloatType(1), measdate=m.measdate, start_measdate=m.measdate, sim_year=0))
  end
  splots = DataFrame(rows)
  # the plugin context of this run: HBAD = the set's priors; PnET = the per-request climate (+ the cached base constants / species table); BS = no data
  attrs = nothing
  pc = SP === PnETPlugin.PnET ? _pnet_context(ps.scen, climate, meta, horizon, warnings) :
       SP === HBADPlugin.HBAD ? ps.hb : BiomassSuccessionPlugin.BSTables()
  SP === PnETPlugin.PnET && (attrs = _pnet_site_attrs(ps.scen, pc.climate, meta, warnings))
  rng = RNGType(UInt64(seed))
  soa = make_sites(splots, [Int.(ids) for ids in p.ECO_SPECIES_IDS]; plugin=PluginConfig(pc), rng=rng, spinup=false,
                   no_establishment=!establishment, site_attrs=attrs)
  eco_params = generate_eco_params(pc, p); ctx = make_ctx(pc, eco_params)
  site_of = Dict(Int(soa.scalar.mapcode[i]) => i for i in 1:soa.n)
  pm = Dict(m.plot_id => m for m in meta)
  acr = Dict(i => acres[pm[Int(soa.scalar.mapcode[i])].cn] for i in 1:soa.n)
  events = _plan_events(plan, cycles)
  mets = [SCMetrics[] for _ in 1:soa.n]; dists = [Dict{Int,Dict{Tuple{Int16,Int},NTuple{2,Float64}}}() for _ in 1:soa.n]
  y0 = [Dates.year(pm[Int(soa.scalar.mapcode[i])].measdate) for i in 1:soa.n]
  rem = [zeros(3) for _ in 1:soa.n]   # removed (agb g/m², vol m³/ha, sawtimber vol m³/ha) by this cycle's management
  function manage!(c)
    for (ec, kind, act) in events
      ec == c || continue
      if kind == "plant"
        _plant_all!(ps, pc, soa, collect(1:soa.n), act, warnings)
      elseif kind == "harvest"
        sites = _select_stands(ps, soa, act, acr, seed + c)
        spf = _sel_species(get(get(act, "selection", Dict()), "species", "all"))
        ok(sp) = spf === nothing || uppercase(sp) in spf || any(s -> get(ps.meta.species_map, s, "") == sp, spf)
        for i in sites
          rb, rv, rs = _apply_removal!(ps, pc, getsite(soa, i), act, ok)
          rem[i][1] += rb; rem[i][2] += rv; rem[i][3] += rs
        end
        pa = get(act, "plant_after", nothing)
        pa === nothing || _plant_all!(ps, pc, soa, sites, Dict{String,Any}(String(k) => v for (k, v) in pairs(pa)), warnings)
      end
    end
  end
  # Row c = state at elapsed c·period BEFORE cycle c's management (FVS .sum convention); that management's removals are
  # reported on the same row. Then the stand grows `period` years to the next row.
  for c in 0:cycles
    if c > 0
      for t in ((c - 1) * period + 1):(c * period)
        step!(SP, soa, t, ctx)
      end
    end
    sms = [_site_summary(ps, getsite(soa, i)) for i in 1:soa.n]
    manage!(c)
    for i in 1:soa.n
      sm = sms[i]; r_ = rem[i]; t = c * period
      push!(mets[i], SCMetrics(y0[i] + t, sm.ba, sm.tpa, sm.qmd, sm.tcuft, sm.bdft,
        r_[2], r_[3],
        sm.c_ag * (1 + SC_ROOT_FRAC), sm.c_ag, r_[1] * SC_CARBON * 0.01, sm.agb))
      dists[i][y0[i] + t] = sm.dist
      fill!(r_, 0.0)
    end
    progress === nothing || progress(c, cycles)
  end
  spsym = Dict{Int16,String}(Int16(v) => k for (k, v) in ps.spcd)
  for i in 1:soa.n
    m = pm[Int(soa.scalar.mapcode[i])]
    results[m.cn] = (status="ok", message="", metrics=mets[i], dist=dists[i], spsym=spsym)
  end
  results, unique(warnings)
end

# reserve CSR capacity (`extra[i]` new cohorts on site i) — every site's count must be set before readjust_soa!
function _reserve!(soa, extra::Dict{Int,Int})
  for i in 1:soa.n
    soa.scalar._new_cohort_counts[i] = Int(soa.scalar.live[i]) + get(extra, i, 0)
  end
  PanCore.readjust_soa!(soa, (cohort=soa.scalar._new_cohort_counts,))
end

function _plant_all!(ps::ParamSet, pc, soa, sites, act, warnings)
  spec = String(something(get(act, "species", ""), ""))
  syms = [uppercase(strip(x)) for x in split(spec, ',') if !isempty(strip(x))]
  isempty(syms) && return
  _reserve!(soa, Dict(i => length(syms) for i in sites))
  tpa = Float64(something(get(act, "tpa", 0.0), 0.0)); surv = Float64(something(get(act, "survival", 100.0), 100.0))
  for i in sites, sym in syms
    _plant!(ps, pc, getsite(soa, i), sym, tpa, surv) ||
      push!(warnings, "planting: species $sym cannot be represented in eco cell $(ps.params.ECO_LIST[Int(getsite(soa, i).eco_id)]) (skipped)")
  end
end

"""Stands (sites) a harvest applies to: all, or the top-ranked ones until `area_pct` % of the AOI acres is reached."""
function _select_stands(ps::ParamSet, soa, act, acr, seed)
  st = Dict{String,Any}(String(k) => v for (k, v) in pairs(something(get(act, "stands", nothing), Dict())))
  rank = String(get(st, "rank", "all")); pct = Float64(get(st, "area_pct", 100.0))
  ids = collect(1:soa.n)
  rank == "all" && pct >= 100 && return ids
  score(i) = (s = getsite(soa, i); n = Int(s.live);
    rank == "max_biomass" ? sum(Float64(s.c_bio[k]) for k in 1:n; init=0.0) :
    rank == "oldest" ? maximum(Float64(s.c_age[k]) for k in 1:n; init=0.0) :
    Float64(hash((seed, Int(s.mapcode))) % 1_000_000))
  order = sort(ids; by=score, rev=true)
  total = sum(values(acr)); target = pct / 100 * total; out = Int[]; acc = 0.0
  for i in order
    acc >= target && break
    push!(out, i); acc += acr[i]
  end
  out
end

# --- PnET climate + site attributes -----------------------------------------------------------------------------------
const DEFAULT_CLIMATE = Dict{String,Any}("baseline" => Dict{String,Any}("mode" => "forecast", "T" => 30, "K" => 3),
                                         "co2" => Dict{String,Any}("mode" => "constant"), "changes" => Any[])
_co2_observed(scen::ScenarioConfig) = (df = CSV.read(scen.co2_csv, DataFrame); Dict{Int,Float64}(Int(r[1]) => Float64(r[2]) for r in eachrow(df)))

"""
    scenario_climate(l4s, y1, y2, climate) -> (years, Dict(l4 => Dict(var => 12×n matrix)), co2::Vector)

Monthly tmin/tmax/ppt per L4 for calendar years y1..y2: PRISM observed through the last complete year (2024), then the
scenario baseline (forecast | repeat last T | average of the last K windows of T) with the scenario's changes applied
(ClimateScenario.build_climate) — changes act on every year from their start year, historical years included only if the
user dates them there. CO₂: observed annual series, then the scenario pathway.
"""
function scenario_climate(scen::ScenarioConfig, l4s::Vector{String}, y1::Int, y2::Int, climate)
  sc = climate === nothing ? DEFAULT_CLIMATE : climate
  db = DuckDB.DB(scen.climate_db; readonly=true); con = DuckDB.DBInterface.connect(db)
  obs = Dict{String,Dict{String,Matrix{Float64}}}(); fc = Dict{String,Dict{String,Matrix{Float64}}}()
  oy = collect(1895:2024); fy = collect(2025:2150)
  try
    lst = join(("'" * replace(l, "'" => "") * "'" for l in l4s), ",")
    for (tbl, yrs, store) in (("observed_monthly", oy, obs), ("forecast_monthly", fy, fc))
      df = DataFrame(DuckDB.DBInterface.execute(con, """SELECT l4, var, year, month, value FROM $(tbl)
          WHERE l4 IN ($(lst)) AND var IN ('tmin','tmax','ppt') AND year BETWEEN $(yrs[1]) AND $(yrs[end])"""))
      for r in eachrow(df)
        d = get!(store, String(r.l4), Dict{String,Matrix{Float64}}())
        M = get!(d, String(r.var)) do; fill(NaN, 12, length(yrs)); end
        M[Int(r.month), Int(r.year) - yrs[1] + 1] = Float64(r.value)
      end
    end
  finally
    DuckDB.DBInterface.close!(con); DuckDB.close(db)
  end
  out = Dict{String,Dict{String,Matrix{Float64}}}(); years = collect(y1:y2)
  for l in l4s
    haskey(obs, l) || error("no PRISM climate for L4 $l in $(scen.climate_db)")
    o = obs[l]; any(isnan, o["tmin"]) && error("incomplete PRISM record for L4 $l")
    _, d = ClimateScenario.build_climate(o, oy, get(fc, l, nothing), fy, sc; start_year=y1, end_year=y2)
    out[l] = d
  end
  co2 = ClimateScenario.co2_series(years, ClimateScenario._get(sc, :co2, Dict{String,Any}("mode" => "constant")), _co2_observed(scen))
  years, out, co2
end

# The PnET run context without a per-request climate: constants / species table + the 1980-2025 climate cache. Loaded on first use (lazy cache).
_pnet_base_context(scen::ScenarioConfig) = get_cached!(scen.cache, :pnet_base) do
  PnETPlugin.PnETContext(scen.pnet_climate, scen.co2_csv; generic=Dict{String,Any}(), db_path=scen.fia_db)
end

"""The PnET context of a request: the base constants with a per-request climate covering every plot's L4 for its whole run (the cached 1980-2025 cache without a climate DB)."""
function _pnet_context(scen::ScenarioConfig, climate, meta, horizon::Int, warnings)
  base = _pnet_base_context(scen)
  if !isfile(scen.climate_db)
    push!(warnings, "PnET climate: $(scen.climate_db) not found — using the 1980–2025 cache (later years repeat 2025); climate scenarios unavailable")
    return base
  end
  l4s = sort(unique(String(m.l4) for m in meta if !isempty(m.l4)))
  y1 = minimum(Dates.year(m.measdate) for m in meta); y2 = maximum(Dates.year(m.measdate) for m in meta) + horizon + 1
  years, cl, co2 = scenario_climate(scen, l4s, y1, y2, climate)
  ny = length(years); nc = length(l4s)
  tmin = Array{Float64}(undef, 12, ny, nc); tmax = similar(tmin); ppt = similar(tmin)
  for (c, l) in enumerate(l4s)
    tmin[:, :, c] .= cl[l]["tmin"]; tmax[:, :, c] .= cl[l]["tmax"]; ppt[:, :, c] .= cl[l]["ppt"]
  end
  return PnETPlugin.with_climate(base, PnETPlugin.build_climate(l4s, years, tmin, tmax, ppt, co2))
end

function _pnet_site_attrs(scen::ScenarioConfig, cl, meta, warnings)
  idpos = Dict(id => i for (i, id) in enumerate(cl.ids))
  soil = isfile(scen.soil_csv) ? PnETPlugin.load_plot_soil(scen.soil_csv) : Dict()
  located = [m for m in meta if isfinite(m.lon)]
  lsoil = Dict(zip((m.plot_id for m in located), soil_at(scen, Float64[m.lon for m in located], Float64[m.lat for m in located])))
  nls = count(!isnothing, values(lsoil))
  rows = NamedTuple[]; nfb = 0
  for m in meta
    cid = get(idpos, m.l4, 0)
    if cid == 0
      nfb += 1; cid = something(findfirst(id -> startswith(id, m.l3 * "."), cl.ids), 1)
    end
    k = Tuple(parse.(Int, split(m.key, "_")))
    sv = get(lsoil, m.plot_id, nothing)                  # soils of the location (SOLUS), else of the FIA plot
    sv === nothing && (sv = get(soil, k, nothing))
    push!(rows, (plot_id=m.plot_id, clim_id=Int32(cid), cal_year0=Int16(Dates.year(m.measdate)), lat=Float32(isnan(m.lat) ? 33.0 : m.lat), soil="LOAM",
                 soil_por=sv === nothing ? NaN : sv.soil_por, soil_fc=sv === nothing ? NaN : sv.soil_fc,
                 soil_wp=sv === nothing ? NaN : sv.soil_wp, soil_rd=sv === nothing ? NaN : sv.soil_rd))
  end
  nfb > 0 && push!(warnings, "PnET: $nfb plots have an L4 outside the climate cache; an L4 of the same L3 was used")
  length(located) > nls && push!(warnings, "PnET: $(length(located) - nls) of $(length(located)) located plots have no SOLUS soil data at their location; the FIA plot's soil (or a loam default) was used")
  DataFrame(rows)
end

"""
    climate_preview(scen, plots, climate; start_year, years) -> NamedTuple for the UI

AOI-weighted (acres) monthly series of tmin, tmax, tmean, ppt and annual CO₂ for start_year … start_year+years−1 over the
plots' L4s: `baseline` = the scenario's baseline mode with no changes, `modified` = with its changes and CO₂ pathway.
"""
function climate_preview(scen::ScenarioConfig, plots, climate; start_year::Integer, years::Integer)
  info = load_scenario_units(scen, plots, String[])
  acres = Dict(String(x.cn) => Float64(x.acres) for x in plots)
  w = Dict{String,Float64}()
  for r in info
    (r.status == "ok" && !isempty(r.l4)) || continue
    w[r.l4] = get(w, r.l4, 0.0) + acres[r.cn]
  end
  isempty(w) && error("no plot of the AOI has climate data (no FIA cohort data / L4)")
  l4s = sort(collect(keys(w))); tw = sum(values(w))
  sc = climate === nothing ? DEFAULT_CLIMATE : climate
  base_sc = Dict{String,Any}("baseline" => ClimateScenario._get(sc, :baseline, DEFAULT_CLIMATE["baseline"]),
                             "co2" => Dict{String,Any}("mode" => "constant"), "changes" => Any[])
  y1 = Int(start_year); y2 = Int(start_year + years - 1)
  yrs, cb, co2b = scenario_climate(scen, l4s, y1, y2, base_sc)
  _, cm, co2m = scenario_climate(scen, l4s, y1, y2, sc)
  agg(cl, v) = sum(w[l] / tw .* cl[l][v] for l in l4s)
  vars = NamedTuple[]
  for (v, lab, un) in (("tmin", "Minimum temperature", "°C"), ("tmax", "Maximum temperature", "°C"), ("tmean", "Mean temperature", "°C"), ("ppt", "Precipitation", "mm/month"))
    B = v == "tmean" ? 0.5 .* (agg(cb, "tmin") .+ agg(cb, "tmax")) : agg(cb, v)
    M = v == "tmean" ? 0.5 .* (agg(cm, "tmin") .+ agg(cm, "tmax")) : agg(cm, v)
    ann(X) = v == "ppt" ? vec(sum(X; dims=1)) : vec(Statistics.mean(X; dims=1))   # ppt: annual total (mm/yr)
    push!(vars, (name=v, label=lab, units=un, annual_units=(v == "ppt" ? "mm/yr" : un), years=yrs,
                 baseline_annual=round.(ann(B); digits=3), modified_annual=round.(ann(M); digits=3),
                 baseline_monthly=round.(vec(B); digits=3), modified_monthly=round.(vec(M); digits=3)))
  end
  # derived driver PnET actually uses: vapour-pressure deficit from Tday = (Tmax + Tave)/2 and Tmin (PnET climate.jl)
  vpd(cl) = (tn = agg(cl, "tmin"); tx = agg(cl, "tmax"); ta = 0.5 .* (tn .+ tx); td = 0.5 .* (tx .+ ta); PnETPlugin.vpd_kpa.(td, tn))
  VB = vpd(cb); VM = vpd(cm)
  push!(vars, (name="vpd", label="Vapour-pressure deficit (derived)", units="kPa", annual_units="kPa", years=yrs,
               baseline_annual=round.(vec(Statistics.mean(VB; dims=1)); digits=4), modified_annual=round.(vec(Statistics.mean(VM; dims=1)); digits=4),
               baseline_monthly=round.(vec(VB); digits=4), modified_monthly=round.(vec(VM); digits=4)))
  push!(vars, (name="co2", label="Atmospheric CO₂", units="ppm", annual_units="ppm", years=yrs,
               baseline_annual=round.(co2b; digits=2), modified_annual=round.(co2m; digits=2),
               baseline_monthly=round.(repeat(co2b; inner=12); digits=2), modified_monthly=round.(repeat(co2m; inner=12); digits=2)))
  (l4s=l4s, l4_weights=[round(w[l] / tw; digits=4) for l in l4s], start_year=y1, vars=vars)
end
