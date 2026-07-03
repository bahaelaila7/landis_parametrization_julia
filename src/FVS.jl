"""
    FVS

Drive the Forest Vegetation Simulator (FVS) variant **executables** as the
forward simulator for Pan: write `.key` + `.tre` input files, run the binary,
and read results back from the FVS Database Extension (SQLite) output.

This is deliberately independent of the SoA / BiomassSuccession engine. It
operates on plain `DataFrame`s shaped like `curated_trees_fvs` (one row per live
tree x measurement visit) so it can be tested standalone and later wired into a
`simulate_FVS` entry point that reuses Pan's eco / extent / plot filters.

Key facts encoded here (validated against `bin/FVSsn`, see memory `fvs-integration`):
  * `TREEDATA` with no argument reads `<keybase>.tre`; one `STDIDENT…PROCESS`
    block per stand in the `.key`, one tree block per stand in the `.tre`
    terminated by a `-999` line.
  * FVS reads the FIA `SPCD` directly in the species field (no alpha map).
  * Tree-record field order (`base/intree.f`): point, tree, PROB(=TPA), history,
    species, DBH, DG, HT, THT, HTG, ICR, … — the `TREEFMT` below covers all 25
    read variables so Fortran format-reversion can't eat the next line.
  * All of a plot's trees go on FVS *point 1* with `DESIGN BAF=0 invArea=1
    breakDBH=999 nplots=1`, so per-tree PROB=`TPA_UNADJ` sums to plot TPA.
"""
module FVS

using DataFrames
using Printf
using Dates
import DuckDB
import ArchGDAL as AG

const DEFAULT_BIN_DIR = "../ForestVegetationSimulator/bin"

# Covers all 25 variables read by base/intree.f so the format never reverts.
const TREEFMT = "(I4,I4,F9.3,I1,A3,5F7.2,I3,6I3,I3,I3,5I3,F7.1)"

# ---------------------------------------------------------------------------
# Variant selection (foundation for per-ecoregion dispatch)
# ---------------------------------------------------------------------------

"""
    variant_for_eco(eco) -> String

Return the FVS variant code for an ecoregion. Currently returns `"sn"`
(Southern) for everything; replace the body with a real ecoregion→variant
lookup later. The rest of the pipeline only needs the 2-letter code.
"""
variant_for_eco(::AbstractString) = "sn"

variant_binary(variant::AbstractString; bindir=DEFAULT_BIN_DIR) =
  joinpath(bindir, "FVS" * lowercase(variant))

# ---------------------------------------------------------------------------
# Keyword / tree-record formatting
# ---------------------------------------------------------------------------

# FVS keyword line: name in cols 1-10, then right-justified 10-col numeric fields.
# A field given as `nothing` is left blank (FVS uses its default for that field).
function _kw(name::AbstractString, fields::Vararg{Union{Nothing,Real}})
  io = IOBuffer()
  @printf(io, "%-10s", name)
  for f in fields
    if f === nothing
      print(io, " "^10)
    else
      @printf(io, "%10.3f", float(f))
    end
  end
  return String(take!(io))
end

# One TREEDATA record. `point` is the FVS point (we use 1 for all), `tree` a
# stand-unique id. Missing ht/cr are passed as 0 → FVS estimates them.
# `damage` = up to 3 (agent,severity) pairs → the 6 IDAMCD fields; `birth_age` =
# ABIRTH (so FVS TreeAge tracks initial-tree age). TREEFMT reserves both slots
# (…6I3,I3,I3,5I3,F7.1): 6 damage, mort, cut, 5 pest, birth_age — written in full
# so the fixed-column layout never reverts.
function tree_record(point::Integer, tree::Integer, spcd::Integer,
  dbh::Real, ht::Real, cr::Real, tpa::Real;
  damage::NTuple{6,Int}=(0, 0, 0, 0, 0, 0), birth_age::Real=0.0)
  sp = spcd > 0 && spcd <= 999 ? string(spcd) : "OT"
  icr = round(Int, clamp(cr, 0, 99))
  # point tree PROB hist species  DBH  DG   HT   THT  HTG  ICR
  head = @sprintf("%4d%4d%9.3f%1d%-3s%7.2f%7.2f%7.2f%7.2f%7.2f%3d",
    point, tree, float(tpa), 1, sp, float(dbh), 0.0, float(ht), 0.0, 0.0, icr)
  # 6 damage, mort(0), cut(0), 5 pest(0), birth_age
  tail = @sprintf("%3d%3d%3d%3d%3d%3d%3d%3d%3d%3d%3d%3d%3d%7.1f",
    damage[1], damage[2], damage[3], damage[4], damage[5], damage[6],
    0, 0, 0, 0, 0, 0, 0, float(birth_age))
  return head * tail
end

# DESIGN so each record's PROB is read as trees-per-acre directly. Field 4 = NUMBER OF PLOTS: set to the
# stand's point count so multi-plot (strata) stands average density across their FIA plots correctly.
design_keyword(nplots::Integer=1) = _kw("DESIGN", 0.0, 1.0, 999.0, Float64(nplots), 0.0, 1.0)

# STDINFO: location & habitat left blank (FVS default), then age, aspect, slope,
# elevation in 100s of feet. Pass `nothing` for unknown values.
function stdinfo_keyword(; age=0.0, aspect=nothing, slope=nothing, elev_ft=nothing)
  elev = elev_ft === nothing ? nothing : elev_ft / 100
  return _kw("STDINFO", nothing, nothing, age, aspect, slope, elev)
end

"""
    cycle_plan(inv_year, target_years; maxlen=10) -> Vector{Int}

Cycle lengths (years) whose cumulative sums land exactly on each target year,
splitting any gap longer than `maxlen` into ≤`maxlen` pieces so FVS stays in its
accurate cycle-length range. The target years become FVS cycle boundaries (and
thus rows in FVS_Summary / FVS_TreeList).
"""
function cycle_plan(inv_year::Integer, target_years; maxlen::Integer=10)
  lengths = Int[]
  prev = inv_year
  for ty in sort(unique(collect(target_years)))
    gap = ty - prev
    gap <= 0 && continue
    nsub = cld(gap, maxlen)
    base, rem = divrem(gap, nsub)
    for i in 1:nsub
      push!(lengths, base + (i <= rem ? 1 : 0))
    end
    prev = ty
  end
  return lengths
end

# ---------------------------------------------------------------------------
# Writing a run (many stands -> one .key + one .tre + one out.db)
# ---------------------------------------------------------------------------

"""
    StandSpec

One FVS stand to simulate.
  * `id`           : stand identifier (≤26 chars), e.g. "13_1_49_17"
  * `inv_year`     : calendar year of the initial state (sim_year 0)
  * `target_years` : later calendar years to project to (cycle boundaries)
  * `slope/aspect/elev_ft` : site descriptors (or `nothing`)
  * `trees`        : iterable of `(spcd, dbh, ht, cr, tpa, damage, birth_age)` tuples
                     (`damage`=6-tuple of IDAMCD ints, `birth_age`=initial age in yr)
"""
const TreeRec = NamedTuple{(:spcd, :dbh, :ht, :cr, :tpa, :damage, :birth_age),
  Tuple{Int,Float64,Float64,Float64,Float64,NTuple{6,Int},Float64}}
Base.@kwdef struct StandSpec
  id::String
  inv_year::Int
  target_years::Vector{Int}
  slope::Union{Nothing,Float64} = nothing
  aspect::Union{Nothing,Float64} = nothing
  elev_ft::Union{Nothing,Float64} = nothing
  trees::Vector{TreeRec}
  n_plots::Int = 1            # FVS points/plots in this stand (>1 for multi-plot strata stands; ≤ MAXPLT=500)
  points::Vector{Int} = Int[] # per-tree FVS point (1..n_plots); empty ⇒ all trees on point 1 (single-plot stand)
end

function _write_stand_block!(key::IO, tre::IO, s::StandSpec; fiavbc::Bool=false, ffe::Bool=true, compute_db::Bool=false)
  lengths = cycle_plan(s.inv_year, s.target_years)
  isempty(lengths) && (lengths = [10])   # no targets → project one default cycle

  println(key, "STDIDENT")
  println(key, s.id)
  println(key, _kw("INVYEAR", s.inv_year))
  for (i, len) in enumerate(lengths)
    println(key, _kw("TIMEINT", i, len))
  end
  println(key, _kw("NUMCYCLE", length(lengths)))
  println(key, stdinfo_keyword(; aspect=s.aspect, slope=s.slope, elev_ft=s.elev_ft))
  println(key, design_keyword(s.n_plots))
  println(key, "TREEFMT")
  println(key, TREEFMT)
  println(key, "TREEDATA")
  if ffe   # FFE on → aboveground biomass/carbon in FVS_Carbon (expensive per cycle)
    println(key, "FMIN")
    println(key, _kw("CARBREPT", 2))
    println(key, _kw("CARBCALC", 0, 0))
    println(key, "END")
  end
  # FIA-consistent National Volume/Biomass/Carbon (gives FVS_FIAVBC_Summary with
  # aboveground biomass AbvGrdBio). Base keyword; DB table enabled by VBCSUMDB.
  fiavbc && println(key, "FIAVBC")
  # Base TREELIST keyword (all cycles) — required for TREELIDB to populate the
  # FVS_TreeList table with per-tree DBH/species/TPA. Text copy goes to .out.
  println(key, _kw("TREELIST", 0))
  # Database extension → SQLite output. DSNOUT can't be set per stand in a
  # multi-stand run (FVS16 "cannot be redefined"), so we use FVS's default
  # output DB name (FVSOut.db) instead of emitting DSNOUT.
  println(key, "DATABASE")
  println(key, _kw("SUMMARY", 2))
  println(key, _kw("TREELIDB", 2))
  ffe && println(key, _kw("CARBREDB", 1))
  fiavbc && println(key, "VBCSUMDB")   # → FVS_FIAVBC_Summary (requires FIAVBC)
  compute_db && println(key, "COMPUTE")   # → FVS_Compute table (event-monitor Compute vars, for diagnostics)
  println(key, "END")
  println(key, "PROCESS")

  # tree records: all on point 1, stand-unique tree ids, then -999 terminator
  for (j, t) in enumerate(s.trees)
    pt = isempty(s.points) ? 1 : s.points[j]
    println(tre, tree_record(pt, j, t.spcd, t.dbh, t.ht, t.cr, t.tpa;
      damage=t.damage, birth_age=t.birth_age))
  end
  println(tre, "-999")
end

# FVS's default Database-extension output file name (used since DSNOUT can't be
# redefined per stand in a multi-stand run).
const FVS_DEFAULT_DB = "FVSOut.db"

"""
    write_run(stands; dir) -> (keypath, trepath, dbpath)

Write a batched run: one `.key` containing a block per stand and one matching
`.tre`. Returns paths; `dbpath` is the FVS default output DB FVS will write.
"""
function write_run(stands::AbstractVector{StandSpec}; dir::AbstractString,
  basename::AbstractString="run", fiavbc::Bool=false, estab::Symbol=:noauto, ffe::Bool=true,
  ranseed::Union{Nothing,Integer}=nothing, regimpute::Union{Nothing,AbstractString}=nothing,
  compute_db::Bool=false)
  estab in (:auto, :noauto) || error("estab must be :auto or :noauto, got $estab")
  mkpath(dir)
  keypath = joinpath(dir, basename * ".key")
  trepath = joinpath(dir, basename * ".tre")
  dbpath = joinpath(dir, FVS_DEFAULT_DB)
  isfile(dbpath) && rm(dbpath)   # DBS appends; start clean
  open(keypath, "w") do key
    open(trepath, "w") do tre
      println(key, "SCREEN")
      # RANNSEED sets FVS's random stream (mortality allocation, regen/ingrowth). Omit → FVS default seed.
      ranseed !== nothing && println(key, _kw("RANNSEED", Float64(ranseed)))
      # AUTOES = FVS automatic establishment (default); NOAUTOES = observed trees only
      println(key, estab === :auto ? "AUTOES" : "NOAUTOES")
      # REGIMPUTE addfile: inject FIA-imputed natural-regen keywords (ESTAB/Natural, one per species,
      # gated on stocking + in-stand seed source). Placed globally (before the first STDIDENT) so it
      # applies to every stand, exactly like AUTOES. SN's establishment model is partial — AUTOES enables
      # the extension but adds no natural ingrowth by itself; this addfile supplies the regeneration.
      # This FVS build rejects the ADDFILE keyword, so we INLINE the kcp verbatim (a kcp is designed to be
      # merged into the keyword stream). Skip pure-comment (!/*) and blank lines; write originals to keep
      # FVS's column-sensitive formatting intact.
      if regimpute !== nothing
        for ln in eachline(regimpute)
          s = strip(ln)
          (isempty(s) || startswith(s, '!') || startswith(s, '*')) && continue
          println(key, ln)
        end
      end
      for s in stands
        _write_stand_block!(key, tre, s; fiavbc=fiavbc, ffe=ffe, compute_db=compute_db)
      end
      println(key, "STOP")
    end
  end
  return (keypath, trepath, dbpath)
end

# ---------------------------------------------------------------------------
# Running the binary
# ---------------------------------------------------------------------------

"""
    run_fvs(keypath; variant="sn", bindir, ld_library_path=nothing) -> (ok, log)

Run the FVS variant executable on `keypath` (cwd = its directory, so the
`<base>.tre` and `out.db` relative names resolve). Returns whether it exited
normally (FVS uses a non-zero `STOP` code on success) and the captured log.

`libgfortran.so.5` must be reachable; pass its directory via `ld_library_path`
or ensure it is already on the inherited `LD_LIBRARY_PATH` (e.g. run Julia from
the conda/mamba env that provides it).
"""
function run_fvs(keypath::AbstractString; variant::AbstractString="sn",
  bindir::AbstractString=DEFAULT_BIN_DIR,
  ld_library_path::Union{Nothing,AbstractString}=nothing)
  # Absolute path: the command runs with cwd=dir, so a relative bindir would
  # otherwise resolve against the run directory instead of the project root.
  bin = abspath(variant_binary(variant; bindir=bindir))
  isfile(bin) || error("FVS binary not found: $bin")
  dir = dirname(abspath(keypath))
  key = basename(keypath)
  cmd = `$bin --keywordfile=$key`
  if ld_library_path !== nothing
    prev = get(ENV, "LD_LIBRARY_PATH", "")
    cmd = setenv(cmd, "LD_LIBRARY_PATH" => string(ld_library_path, ":", prev))
  end
  log = IOBuffer()
  ok = try
    run(pipeline(Cmd(cmd; dir=dir); stdout=log, stderr=log))
    true
  catch e
    # FVS exits with a STOP code (e.g. 10) on normal completion → not a failure.
    e isa ProcessFailedException
  end
  return (ok, String(take!(log)))
end

# ---------------------------------------------------------------------------
# Reading the SQLite output
# ---------------------------------------------------------------------------

"""
    read_fvs_sqlite(dbpath) -> NamedTuple

Read the FVS Database Extension output tables into `DataFrame`s via DuckDB's
SQLite reader. Missing tables come back as `nothing`. Returns
`(summary, treelist, carbon)`.
"""
function read_fvs_sqlite(dbpath::AbstractString)
  isfile(dbpath) || error("FVS output DB not found: $dbpath")
  con = DuckDB.connect(DuckDB.DB())
  try
    # sqlite_scanner is bundled in recent DuckDB; INSTALL only needs the
    # network if it isn't, so both are best-effort and ATTACH is the real test.
    try
      DuckDB.execute(con, "INSTALL sqlite;")
    catch
    end
    try
      DuckDB.execute(con, "LOAD sqlite;")
    catch
    end
    DuckDB.execute(con, "ATTACH '$(dbpath)' AS fvs (TYPE sqlite, READ_ONLY);")
    tables = Set(DataFrame(DuckDB.execute(con,
      "SELECT table_name FROM information_schema.tables WHERE table_catalog='fvs'")).table_name)
    get_tbl(t) = t in tables ?
                 DataFrame(DuckDB.execute(con, "SELECT * FROM fvs.\"$t\"")) : nothing
    return (summary=get_tbl("FVS_Summary2"),
      treelist=get_tbl("FVS_TreeList"),
      carbon=get_tbl("FVS_Carbon"),
      fiavbc=get_tbl("FVS_FIAVBC_Summary"))
  finally
    close(con)
  end
end

# ---------------------------------------------------------------------------
# High level: build StandSpecs from a curated_trees_fvs subset and run them
# ---------------------------------------------------------------------------

const PLOT_KEY = [:statecd, :unitcd, :countycd, :plot]

_year(d) = year(d isa Date ? d : Date(d))
_firstnn(v) = (i = findfirst(!ismissing, v); i === nothing ? nothing : v[i])

"""
    stands_from_df(df) -> Vector{StandSpec}

Group a `curated_trees_fvs`-shaped frame by plot. The earliest `sim_year`
(== 0) supplies the FVS initial tree list; the calendar years of the later
visits become the projection targets. Requires columns: statecd/unitcd/countycd/
plot, measdate, sim_year, spcd, dia, ht, cr, tpa_unadj, site_slope/aspect/elev.
"""
function stands_from_df(df::DataFrame)
  stands = StandSpec[]
  for g in groupby(df, PLOT_KEY)
    sim0 = minimum(g.sim_year)
    init = g[g.sim_year.==sim0, :]
    isempty(init) && continue
    inv_year = _year(first(init.measdate))
    targets = sort(unique(_year.(g.measdate[g.sim_year.>sim0])))
    has_age = "estimated_age" in names(init)
    trees = TreeRec[(spcd=Int(r.spcd),
      dbh=Float64(r.dia),
      ht=ismissing(r.ht) ? 0.0 : Float64(r.ht),
      cr=ismissing(r.cr) ? 0.0 : Float64(r.cr),
      tpa=Float64(r.tpa_unadj),
      damage=(0, 0, 0, 0, 0, 0),
      birth_age=(has_age && !ismissing(r.estimated_age)) ? Float64(r.estimated_age) : 0.0)
      for r in eachrow(init)]
    k = first(init)
    id = join(string.((k.statecd, k.unitcd, k.countycd, k.plot)), "_")
    push!(stands, StandSpec(
      id=id, inv_year=inv_year, target_years=targets,
      slope=_toF(_firstnn(init.site_slope)),
      aspect=_toF(_firstnn(init.site_aspect)),
      elev_ft=_toF(_firstnn(init.site_elev)),
      trees=trees))
  end
  return stands
end

_toF(x) = x === nothing || ismissing(x) ? nothing : Float64(x)

"""
    simulate(df; dir, variant="sn", ld_library_path=nothing) -> NamedTuple

End-to-end on a `curated_trees_fvs` subset: build stands, write inputs, run FVS,
parse output. Returns `(stands, keypath, dbpath, ok, log, summary, treelist,
carbon)`. Filter `df` (by eco/extent/plot) before calling.
"""
function simulate(df::DataFrame; dir::AbstractString,
  variant::AbstractString="sn", fiavbc::Bool=false,
  ld_library_path::Union{Nothing,AbstractString}=nothing)
  stands = stands_from_df(df)
  isempty(stands) && error("no stands built from df (check filters/columns)")
  keypath, _, dbpath = write_run(stands; dir=dir, fiavbc=fiavbc)
  ok, log = run_fvs(keypath; variant=variant, bindir=DEFAULT_BIN_DIR,
    ld_library_path=ld_library_path)
  out = isfile(dbpath) ? read_fvs_sqlite(dbpath) :
        (summary=nothing, treelist=nothing, carbon=nothing, fiavbc=nothing)
  return (; stands, keypath, dbpath, ok, log,
    out.summary, out.treelist, out.carbon, out.fiavbc)
end

# ---------------------------------------------------------------------------
# Compare: FVS projection vs observed later measurements
# ---------------------------------------------------------------------------

# 1 short ton = 2000 lb; FIA drybio_ag/carbon_ag are lb/tree, FVS carbon tons/acre.
const LB_PER_TON = 2000.0
# lb/acre → g/m²  (453.592 g/lb ÷ 4046.86 m²/acre) — same factor Pan/curation use,
# so FVS biomass lands in Pan's native unit (g/m²).
const LB_ACRE_TO_G_M2 = 453.592 / 4046.86
# tons/acre (short ton) → g/m²  (FIAVBC AbvGrdBio is tons/acre).
const TONS_ACRE_TO_G_M2 = LB_PER_TON * LB_ACRE_TO_G_M2
# FVS-FFE converts aboveground live biomass to carbon with a flat 0.5
# (fire/base/fmcrbout.f), so biomass = carbon / 0.5. Fallback only — FFE carbon
# also omits each stand's final cycle row, so FIAVBC AbvGrdBio is preferred.
const FFE_CARBON_FRAC = 0.5

# Weighted, normalized CDF of `values` over fixed `edges` (length nbins).
function _wcdf(values, weights, edges::Vector{Float64})
  nb = length(edges) - 1
  counts = zeros(Float64, nb)
  tw = 0.0
  for (v, w) in zip(values, weights)
    (ismissing(v) || ismissing(w)) && continue
    b = clamp(searchsortedlast(edges, Float64(v)), 1, nb)
    counts[b] += w
    tw += w
  end
  tw <= 0 && return nothing
  return cumsum(counts) ./ tw
end

# Wasserstein-1 distance between two weighted DBH distributions: ∫|F-G| dx,
# approximated on `edges` of width h as Σ|Fcum-Gcum|·h. `missing` if either side
# has no mass.
function _dbh_w1(ov, ow, fv, fw, edges::Vector{Float64})
  oc = _wcdf(ov, ow, edges)
  fc = _wcdf(fv, fw, edges)
  (oc === nothing || fc === nothing) && return missing
  h = edges[2] - edges[1]
  return sum(abs.(oc .- fc)) * h
end

"""
    compare(df, res; dbh_bin=2.0, max_dbh=60.0) -> DataFrame

Compare an FVS projection (`res` from [`simulate`](@ref)) against the observed
later measurements in the same `curated_trees_fvs` subset `df`. One row per
(stand, observed later-visit year) that FVS also produced output for:

  * `dbh_w1`     : Wasserstein-1 distance between TPA-weighted DBH distributions
                   (observed vs FVS), in DBH units (inches).
  * `obs_agb` / `fvs_agb` : aboveground live biomass (g/m², Pan's native unit) —
                   observed from FIA `agb`, FVS from `Aboveground_Total_Live/0.5`.
                   Apples-to-apples with Pan's tracked aboveground live biomass.
  * `agb_rel_err`: (fvs−obs)/obs.

Stands with no matching observed later visit (e.g. empty `target_years`) are
skipped.
"""
function compare(df::DataFrame, res; dbh_bin::Float64=2.0, max_dbh::Float64=60.0)
  res.treelist === nothing && error("FVS_TreeList missing — was TREELIST enabled?")
  edges = collect(0.0:dbh_bin:max_dbh)
  tl = res.treelist
  tl_year = Int.(tl.Year)
  tl_dbh = Float64.(tl.DBH)
  tl_tpa = Float64.(tl.TPA)
  cb = res.carbon
  fb = hasproperty(res, :fiavbc) ? res.fiavbc : nothing

  work = transform(df,
    PLOT_KEY => ByRow((a, b, c, d) -> join((a, b, c, d), "_")) => :standid,
    :measdate => ByRow(_year) => :year)
  later = work[work.sim_year.>0, :]

  rows = NamedTuple[]
  for g in groupby(later, [:standid, :year])
    sid = first(g.standid)
    yr = first(g.year)
    fmask = (tl.StandID .== sid) .& (tl_year .== yr)
    any(fmask) || continue   # FVS produced no treelist at this year for this stand

    w1 = _dbh_w1(g.dia, g.tpa_unadj, tl_dbh[fmask], tl_tpa[fmask], edges)

    # aboveground live biomass, g/m² (Pan's native unit). Observed uses the same
    # `agb` column Pan's ground truth is built from. FVS prefers FIAVBC AbvGrdBio
    # (FIA-consistent biomass, all cycles); falls back to FFE carbon/0.5.
    obs_agb = sum(skipmissing(g.agb))
    fvs_agb = missing
    if fb !== nothing
      m = (fb.StandID .== sid) .& (Int.(fb.Year) .== yr)
      any(m) && (fvs_agb = Float64(first(fb.AbvGrdBio[m])) * TONS_ACRE_TO_G_M2)
    end
    if fvs_agb === missing && cb !== nothing
      m = (cb.StandID .== sid) .& (Int.(cb.Year) .== yr)
      any(m) && (fvs_agb = Float64(first(cb.Aboveground_Total_Live[m])) /
                           FFE_CARBON_FRAC * TONS_ACRE_TO_G_M2)
    end
    arel = (fvs_agb === missing || obs_agb == 0) ? missing :
           (fvs_agb - obs_agb) / obs_agb

    push!(rows, (standid=sid, year=yr,
      n_obs=nrow(g), n_fvs=sum(fmask),
      obs_tpa=sum(g.tpa_unadj), fvs_tpa=sum(tl_tpa[fmask]),
      dbh_w1=w1, obs_agb=obs_agb, fvs_agb=fvs_agb, agb_rel_err=arel))
  end
  return DataFrame(rows)
end

# ---------------------------------------------------------------------------
# Spatial / raster projection: TreeMap pixels → plot-measurement CNs → FVS
# forward projection → AGB rasters (total + per-species) every `output_every`
# years to `timehorizon`.
# ---------------------------------------------------------------------------

"""
    read_treemap(path; treemap_version=2022)
        -> (cn::Matrix{Union{Missing,Int64}}, gt::Vector{Float64}, proj::String)

Read a TreeMap raster into a per-pixel plot-measurement CN matrix plus the
geotransform and projection (needed to write output rasters on the same grid).
Each pixel value indexes the Raster Attribute Table; the `PLT_CN` (2022) / `CN`
(2016) column gives the FIA plot CN.
"""
function read_treemap(path::AbstractString; treemap_version::Int=2022)
  isfile(path) || error("TreeMap raster not found: $path")
  cn_field = treemap_version == 2016 ? "CN" : "PLT_CN"
  AG.readraster(path) do ds
    band = AG.getband(ds, 1)
    gt = AG.getgeotransform(ds)
    proj = AG.getproj(ds)
    rat = AG.getdefaultRAT(band)
    ncols = AG.ncolumn(rat)
    colnames = [AG.columnname(rat, c) for c in 0:ncols-1]
    valcol = findfirst(==("Value"), colnames)
    isnothing(valcol) && error("No 'Value' column in RAT of $path")
    cncol = findfirst(==(cn_field), colnames)
    isnothing(cncol) && error("No '$cn_field' column in RAT of $path")
    rat_int(r, c) = AG.columntype(rat, c) == AG.GFT_Real ?
                    Int64(round(AG.asdouble(rat, r, c))) : Int64(AG.asint(rat, r, c))
    vat = Dict{Int64,Int64}()
    for r in 0:AG.nrow(rat)-1
      vat[rat_int(r, valcol - 1)] = rat_int(r, cncol - 1)
    end
    A = AG.read(band)
    nodata = AG.getnodatavalue(band)
    cn = Array{Union{Missing,Int64}}(missing, size(A))
    @inbounds for i in eachindex(A, cn)
      cell = A[i]
      (ismissing(cell) || cell == nodata) && continue
      v = get(vat, Int64(cell), missing)
      ismissing(v) || (cn[i] = v)
    end
    return (cn, collect(gt), proj)
  end
end

# Pull tree records for a set of plot-measurement CNs from curated_trees_fvs.
function _query_cns(db_path::AbstractString, cns::Vector{Int64};
  tablename::AbstractString="curated_trees_fvs")
  con = DuckDB.connect(DuckDB.DB(db_path))
  try
    DuckDB.register_data_frame(con, DataFrame(plt_cn=cns), "want_cns")
    return DataFrame(DuckDB.execute(con, """
      SELECT t.plt_cn, t.measdate, t.spcd, t.dia, t.ht, t.cr, t.tpa_unadj,
             t.site_slope, t.site_aspect, t.site_elev
      FROM $tablename t JOIN want_cns w ON t.plt_cn = w.plt_cn
      WHERE t.statuscd = 1 AND t.dia IS NOT NULL AND t.tpa_unadj > 0
    """))
  finally
    close(con)
  end
end

# Each plot-measurement CN is its own stand (sim_year 0 = that measurement),
# projected to inv_year+output_every … inv_year+timehorizon.
function stands_from_cns(df::DataFrame; timehorizon::Int, output_every::Int)
  invyr = Dict{Int64,Int}()
  stands = StandSpec[]
  for g in groupby(df, :plt_cn)
    cn = Int64(first(g.plt_cn))
    iv = _year(first(g.measdate))
    invyr[cn] = iv
    targets = collect((iv+output_every):output_every:(iv+timehorizon))
    has_age = "estimated_age" in names(g)
    trees = TreeRec[(spcd=Int(r.spcd),
              dbh=Float64(r.dia),
              ht=ismissing(r.ht) ? 0.0 : Float64(r.ht),
              cr=ismissing(r.cr) ? 0.0 : Float64(r.cr),
              tpa=Float64(r.tpa_unadj),
              damage=(0, 0, 0, 0, 0, 0),
              birth_age=(has_age && !ismissing(r.estimated_age)) ? Float64(r.estimated_age) : 0.0)
              for r in eachrow(g)]
    push!(stands, StandSpec(id=string(cn), inv_year=iv, target_years=targets,
      slope=_toF(_firstnn(g.site_slope)), aspect=_toF(_firstnn(g.site_aspect)),
      elev_ft=_toF(_firstnn(g.site_elev)), trees=trees))
  end
  return stands, invyr
end

# Run many stands in chunks (each chunk = one FVS process in its own dir),
# concatenating the FIAVBC summary and treelist tables.
function _run_chunked(stands::Vector{StandSpec}; dir::AbstractString,
  variant::AbstractString, chunk::Int, ld_library_path)
  fb = DataFrame[]
  tl = DataFrame[]
  parts = collect(Iterators.partition(stands, chunk))
  for (ci, c) in enumerate(parts)
    cdir = joinpath(dir, "chunk$(ci)")
    keypath, _, dbpath = write_run(collect(c); dir=cdir, fiavbc=true)
    ok, log = run_fvs(keypath; variant=variant, ld_library_path=ld_library_path)
    @info "FVS chunk $ci/$(length(parts)): $(length(c)) stands, ok=$ok"
    isfile(dbpath) || (@warn "no FVS DB for chunk $ci"; continue)
    out = read_fvs_sqlite(dbpath)
    out.fiavbc !== nothing && push!(fb, out.fiavbc)
    out.treelist !== nothing && push!(tl, out.treelist)
  end
  return (isempty(fb) ? nothing : vcat(fb...)), (isempty(tl) ? nothing : vcat(tl...))
end

# Build per-(CN, offset) total AGB (g/m²) and per-species AGB (g/m²) maps.
# Per-species AGB apportions the FIAVBC stand AbvGrdBio by each species' share of
# stand volume (Σ TCuFt·TPA) from the treelist.
function _agb_maps(fiavbc::DataFrame, treelist::DataFrame, invyr::Dict{Int64,Int},
  offsets::Vector{Int})
  # total: (cn, offset) → g/m²
  total = Dict{Tuple{Int64,Int},Float64}()
  fb_year = Int.(fiavbc.Year)
  for i in 1:nrow(fiavbc)
    cn = tryparse(Int64, fiavbc.StandID[i]); cn === nothing && continue
    haskey(invyr, cn) || continue
    off = fb_year[i] - invyr[cn]
    off in offsets || continue
    total[(cn, off)] = Float64(fiavbc.AbvGrdBio[i]) * TONS_ACRE_TO_G_M2
  end
  # species volume per (cn, offset, spcd) and totals per (cn, offset)
  tl_year = Int.(treelist.Year)
  spvol = Dict{Tuple{Int64,Int,Int},Float64}()
  volsum = Dict{Tuple{Int64,Int},Float64}()
  species = Set{Int}()
  for i in 1:nrow(treelist)
    cn = tryparse(Int64, treelist.StandID[i]); cn === nothing && continue
    haskey(invyr, cn) || continue
    off = tl_year[i] - invyr[cn]
    off in offsets || continue
    sp = tryparse(Int, treelist.SpeciesFIA[i]); sp === nothing && continue
    v = Float64(treelist.TCuFt[i]) * Float64(treelist.TPA[i])
    spvol[(cn, off, sp)] = get(spvol, (cn, off, sp), 0.0) + v
    volsum[(cn, off)] = get(volsum, (cn, off), 0.0) + v
    push!(species, sp)
  end
  # species AGB = total × volume share
  species_agb = Dict{Tuple{Int64,Int,Int},Float64}()
  for ((cn, off, sp), v) in spvol
    tot = get(total, (cn, off), missing)
    vs = get(volsum, (cn, off), 0.0)
    (tot === missing || vs <= 0) && continue
    species_agb[(cn, off, sp)] = tot * (v / vs)
  end
  return total, species_agb, sort!(collect(species))
end

# Write one Float32 raster on the TreeMap grid: value(pixel) = lookup[cn(pixel)].
function _write_value_raster(path::AbstractString, cn::Matrix{Union{Missing,Int64}},
  lookup::Dict{Int64,Float64}, gt::Vector{Float64}, proj::String; nodata::Float32=-9999.0f0)
  data = fill(nodata, size(cn))
  @inbounds for i in eachindex(cn, data)
    c = cn[i]
    ismissing(c) && continue
    v = get(lookup, c, missing)
    ismissing(v) || (data[i] = Float32(v))
  end
  AG.create(path; driver=AG.getdriver("GTiff"),
    width=size(cn, 1), height=size(cn, 2), nbands=1, dtype=Float32) do ds
    AG.write!(ds, data, 1)
    AG.setgeotransform!(ds, gt)
    AG.setproj!(ds, proj)
    AG.setnodatavalue!(AG.getband(ds, 1), Float64(nodata))
  end
  return path
end

"""
    simulate_spatial(; treemap_raster, db_path, output_dir, timehorizon=50,
                     output_every=5, variant="sn", treemap_version=2022,
                     chunk=500, ld_library_path=nothing) -> NamedTuple

FVS forward-projection raster pipeline. Reads a TreeMap raster (clipped to the
area of interest), simulates each unique plot-measurement CN from its inventory
(sim_year 0) to `timehorizon` years, and writes GeoTIFFs on the TreeMap grid at
each `output_every`-year step: one total aboveground live biomass raster and one
per species (g/m²; nodata where no CN / no projection).

Returns `(cns, stands, total, species_agb, species, files)`.
"""
function simulate_spatial(; treemap_raster::AbstractString, db_path::AbstractString,
  output_dir::AbstractString, timehorizon::Int=50, output_every::Int=5,
  variant::AbstractString="sn", treemap_version::Int=2022, chunk::Int=500,
  tablename::AbstractString="curated_trees_fvs",
  ld_library_path::Union{Nothing,AbstractString}=nothing)
  mkpath(output_dir)
  @info "Reading TreeMap raster $treemap_raster"
  cn, gt, proj = read_treemap(treemap_raster; treemap_version=treemap_version)
  cns = Int64.(unique(skipmissing(vec(cn))))
  @info "TreeMap: $(length(cns)) unique plot CNs over $(size(cn)) grid"

  df = _query_cns(db_path, cns; tablename=tablename)
  @info "Loaded $(nrow(df)) tree rows for $(length(unique(df.plt_cn))) CNs"
  stands, invyr = stands_from_cns(df; timehorizon=timehorizon, output_every=output_every)
  isempty(stands) && error("no stands built — do the TreeMap CNs exist in $tablename?")

  fiavbc, treelist = _run_chunked(stands; dir=joinpath(output_dir, "fvs_runs"),
    variant=variant, chunk=chunk, ld_library_path=ld_library_path)
  fiavbc === nothing && error("FVS produced no FIAVBC output")

  offsets = collect(0:output_every:timehorizon)
  total, species_agb, species = _agb_maps(fiavbc, treelist, invyr, offsets)

  files = String[]
  for off in offsets
    tot_lk = Dict{Int64,Float64}(cn0 => v for ((cn0, o), v) in total if o == off)
    push!(files, _write_value_raster(
      joinpath(output_dir, "agb_total_y$(off).tif"), cn, tot_lk, gt, proj))
    for sp in species
      sp_lk = Dict{Int64,Float64}(cn0 => v for ((cn0, o, s), v) in species_agb if o == off && s == sp)
      isempty(sp_lk) && continue
      push!(files, _write_value_raster(
        joinpath(output_dir, "agb_sp$(sp)_y$(off).tif"), cn, sp_lk, gt, proj))
    end
  end
  @info "Wrote $(length(files)) rasters to $output_dir"
  return (; cns, stands, total, species_agb, species, files)
end

end # module FVS
