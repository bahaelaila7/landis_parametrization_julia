# Remeasurement-interval data product of the HBAD2 calibration (port of the prototype tools/hbad2/build_intervals.jl, same content).
# For every plot and every pair of consecutive visits (v1 -> v2):
#   stand at v1 = every live tree (species, age, DIA, HT, TPA_UNADJ weight) -> competition environment and initial cohorts;
#   outcome     = the same trees at v2 (alive: DIA2, HT2 | dead | harvested | not found), linked by (plot, SUBP, TREE).
# Ingrowth (trees first tallied at a later visit) is recorded separately (a recruitment process, not a change of a v1 cohort).
# Differences to the prototype (content is identical, row order and the interval id are not):
#   * the interval id `iid` is a canonical enumeration (plots sorted by (STATECD, UNITCD, COUNTYCD, PLOT), intervals by date); the prototype's ids
#     came from Dict iteration order. Compare the two products through the natural key (st, u, c, p, d1) -- `compare_interval_products`;
#   * the database is opened READ_ONLY, every filter is a bound parameter / registered table, and the tree query has a total ORDER BY
#     (STATECD, UNITCD, COUNTYCD, PLOT, MEASDATE, SUBP, TREE, CN); of two rows with the same (plot, SUBP, TREE, date) the later one wins.
import Arrow

"""
    HBAD2Intervals

The three tables of the interval product. `intervals` (one row per consecutive visit pair: `iid st u c p d1 d2 dt k trt dst harv planted eco set l3`),
`trees` (tracked live trees of v1: `iid sb tr sym grp spcd age dia_cm ht_m w_ha cclcd cr out dia2_cm ht2_m w2_ha`, `out` in alive|dead|harvest|missing),
`ingrowth` (first tallied trees: `iid sb sym grp age dia_cm ht_m w_ha`; `iid` = the interval whose v1 is the previous visit).
"""
struct HBAD2Intervals
  intervals::DataFrame
  trees::DataFrame
  ingrowth::DataFrame
end

"""
    read_split_plots(path) -> DataFrame

Read a `split_plots.csv` written by a Pan run (`statecd, unitcd, countycd, plot, set` and optionally `eco`; further columns are ignored). `set` is train | val | test.
"""
function read_split_plots(path::AbstractString)::DataFrame
  isfile(path) || error("read_split_plots: split file not found: $path")
  sp = CSV.read(path, DataFrame)
  for c in (:statecd, :unitcd, :countycd, :plot, :set)
    hasproperty(sp, c) || error("read_split_plots: $path lacks the column $c")
  end
  out = DataFrame(statecd=Int.(sp.statecd), unitcd=Int.(sp.unitcd), countycd=Int.(sp.countycd), plot=Int.(sp.plot), set=String.(sp.set),
                  eco=hasproperty(sp, :eco) ? String.(sp.eco) : fill("", nrow(sp)))
  bad = setdiff(unique(out.set), ["train", "val", "test"])
  isempty(bad) || error("read_split_plots: unknown set labels $(bad) in $path (train | val | test)")
  return out
end

"""
    draw_plot_split(plots; val_frac, test_frac, seed) -> DataFrame

Plot-level random split (used when a run has no `split_csv`): the plots (a DataFrame with `statecd unitcd countycd plot`) are put in key order, shuffled with
`Random.Xoshiro(seed)` and cut into test / val / train. Deterministic.
"""
function draw_plot_split(plots::DataFrame; val_frac::Real, test_frac::Real, seed::Integer)::DataFrame
  (0 <= val_frac && 0 <= test_frac && val_frac + test_frac < 1) || error("draw_plot_split: need 0 <= val_frac, test_frac and val_frac + test_frac < 1")
  p = sort!(unique(select(plots, :statecd, :unitcd, :countycd, :plot)), [:statecd, :unitcd, :countycd, :plot])
  n = nrow(p); perm = Random.randperm(Random.Xoshiro(UInt64(seed)), n)
  nt = round(Int, test_frac * n); nv = round(Int, val_frac * n)
  set = fill("train", n)
  for (r, i) in enumerate(perm)
    r <= nt ? (set[i] = "test") : r <= nt + nv ? (set[i] = "val") : nothing
  end
  return DataFrame(statecd=Int.(p.statecd), unitcd=Int.(p.unitcd), countycd=Int.(p.countycd), plot=Int.(p.plot), set=set, eco=fill("", n))
end

"""
    build_hbad2_intervals(db_path, split; states=[12, 13], table="curated_trees", ref_table="REF_SPECIES") -> HBAD2Intervals

Build the interval product from the tree table `table` of the cohort database `db_path` (opened READ_ONLY) for the plots of `split` (a DataFrame as returned by
[`read_split_plots`](@ref)) in the states `states`. The tree query binds nothing but the registered tables `split_in` / `states_in` (no value is interpolated).
"""
function build_hbad2_intervals(db_path::AbstractString, split::DataFrame; states::AbstractVector{<:Integer}=[12, 13],
                               table::AbstractString="curated_trees", ref_table::AbstractString="REF_SPECIES")::HBAD2Intervals
  isfile(db_path) || error("build_hbad2_intervals: database not found: $db_path")
  con = DuckDB.connect(DuckDB.DB())
  t = try
    DuckDB.execute(con, "ATTACH '$(_sql_literal_path(db_path))' AS src (READ_ONLY);")
    tab = _require_table(con, table, "build_hbad2_intervals table"); rtab = _require_table(con, ref_table, "build_hbad2_intervals ref_table")
    DuckDB.register_data_frame(con, select(split, :statecd, :unitcd, :countycd, :plot, :eco, :set), "split_in")
    DuckDB.register_data_frame(con, DataFrame(st=Int.(collect(states))), "states_in")
    DataFrame(DuckDB.execute(con, """
      SELECT t.STATECD AS st, t.UNITCD AS u, t.COUNTYCD AS c, t.PLOT AS p, t.SUBP AS sb, t.TREE AS tr, t.MEASDATE AS md, t.STATUSCD AS s, t.tree_fate AS fate,
             t.spcd_resolved AS spcd, r.SPECIES_SYMBOL AS sym, r.SFTWD_HRDWD AS grp, t.DIA AS dia, t.HT AS ht, t.CCLCD AS cclcd, t.CR AS cr, t.TPA_UNADJ AS tpa,
             t.estimated_age AS age, t.epa_l3 AS l3, t.cond_stdorgcd AS stdorg, t.cond_siteclcd AS sitecl, t.cond_sicond AS si,
             coalesce(t.TRTCD1, 0) AS trt1, coalesce(t.DSTRBCD1, 0) AS dst1, sp.eco AS eco, sp."set" AS "set"
      FROM src.$(tab) t
      JOIN split_in sp ON sp.statecd = t.STATECD AND sp.unitcd = t.UNITCD AND sp.countycd = t.COUNTYCD AND sp.plot = t.PLOT
      JOIN states_in sn ON sn.st = t.STATECD
      LEFT JOIN src.$(rtab) r ON r.SPCD = t.spcd_resolved
      ORDER BY t.STATECD, t.UNITCD, t.COUNTYCD, t.PLOT, t.MEASDATE, t.SUBP, t.TREE, t.CN"""))
  finally
    nothing   # the in-memory connection is released with the object (as in tree_cohorts.jl)
  end
  return _intervals_from_rows(t)
end

function _intervals_from_rows(t::DataFrame)::HBAD2Intervals
  n = nrow(t)
  md = Dates.Date.(t.md); sym = coalesce.(t.sym, "UNK"); grp = coalesce.(t.grp, "H")
  st = Int.(t.st); un = Int.(t.u); co = Int.(t.c); pl = Int.(t.p); sb = Int.(t.sb); trn = Int.(t.tr); stat = t.s
  fate = coalesce.(t.fate, "")
  # rows are ordered by plot key then date: plot ranges and, inside them, visit ranges are contiguous
  pk(i) = (st[i], un[i], co[i], pl[i])
  plot_rng = Tuple{NTuple{4,Int},UnitRange{Int}}[]
  i = 1
  while i <= n
    j = i
    while j < n && pk(j + 1) == pk(i); j += 1; end
    push!(plot_rng, (pk(i), i:j)); i = j + 1
  end
  byid = Dict{Tuple{NTuple{4,Int},Int,Int,Dates.Date},Int}()
  sizehint!(byid, n)
  for i in 1:n; byid[(pk(i), sb[i], trn[i], md[i])] = i; end        # a duplicate key: the later row wins
  # intervals
  iv_iid = Int[]; iv_st = Int[]; iv_u = Int[]; iv_c = Int[]; iv_p = Int[]; iv_d1 = Dates.Date[]; iv_d2 = Dates.Date[]; iv_dt = Float64[]; iv_k = Int[]
  iv_trt = Bool[]; iv_dst = Bool[]; iv_harv = Bool[]; iv_pl = Bool[]; iv_eco = Union{Missing,String}[]; iv_set = Union{Missing,String}[]; iv_l3 = Union{Missing,String}[]
  vis = Dict{NTuple{4,Int},Vector{Dates.Date}}(); ivkey = Dict{Tuple{NTuple{4,Int},Dates.Date},Int}()
  trt1 = t.trt1; dst1 = t.dst1; stdorg = t.stdorg
  iid = 0
  for (key, rng) in plot_rng
    vs = sort!(unique(md[rng])); vis[key] = vs
    # per-visit flags: any treatment / disturbance code recorded at that visit, any harvest fate of a tree recorded at that visit, planted stand origin
    flags = Dict{Dates.Date,NTuple{4,Bool}}()
    for d in vs
      r = [i for i in rng if md[i] == d]
      flags[d] = (any(>(0), @view trt1[r]), any(>(0), @view dst1[r]), any(f -> startswith(f, "harvest"), @view fate[r]),
                  any(x -> !ismissing(x) && x == 1, @view stdorg[r]))
    end
    length(vs) < 2 && continue
    r1 = first(rng)
    for k in 1:length(vs)-1
      d1, d2 = vs[k], vs[k+1]; iid += 1
      f2 = flags[d2]
      push!(iv_iid, iid); push!(iv_st, key[1]); push!(iv_u, key[2]); push!(iv_c, key[3]); push!(iv_p, key[4]); push!(iv_d1, d1); push!(iv_d2, d2)
      push!(iv_dt, Dates.value(d2 - d1) / 365.25); push!(iv_k, k); push!(iv_trt, f2[1]); push!(iv_dst, f2[2]); push!(iv_harv, f2[3]); push!(iv_pl, flags[d1][4])
      push!(iv_eco, t.eco[r1]); push!(iv_set, t.set[r1]); push!(iv_l3, t.l3[r1])
      ivkey[(key, d1)] = iid
    end
  end
  iv = DataFrame(iid=iv_iid, st=iv_st, u=iv_u, c=iv_c, p=iv_p, d1=iv_d1, d2=iv_d2, dt=iv_dt, k=iv_k, trt=iv_trt, dst=iv_dst, harv=iv_harv, planted=iv_pl,
                 eco=iv_eco, set=iv_set, l3=iv_l3)
  # tracked trees and first-tallied ingrowth
  T_iid = Int[]; T_sb = Int[]; T_tr = Int[]; T_sym = String[]; T_grp = String[]; T_spcd = Int[]; T_age = Float64[]; T_dia = Float64[]; T_ht = Float64[]; T_w = Float64[]
  T_cc = Float64[]; T_cr = Float64[]; T_out = String[]; T_dia2 = Float64[]; T_ht2 = Float64[]; T_w2 = Float64[]
  G_iid = Int[]; G_sb = Int[]; G_sym = String[]; G_grp = String[]; G_age = Float64[]; G_dia = Float64[]; G_ht = Float64[]; G_w = Float64[]
  for (key, rng) in plot_rng
    vs = vis[key]
    for i in rng
      stat[i] == 1 || continue
      k = searchsortedfirst(vs, md[i])
      if k < length(vs)                                   # live at v1 -> tracked outcome at v2
        j = get(byid, (key, sb[i], trn[i], vs[k+1]), 0)
        out = j == 0 ? "missing" : stat[j] == 1 ? "alive" : startswith(fate[j], "harvest") ? "harvest" : "dead"
        push!(T_iid, ivkey[(key, md[i])]); push!(T_sb, sb[i]); push!(T_tr, trn[i]); push!(T_sym, sym[i]); push!(T_grp, grp[i]); push!(T_spcd, Int(t.spcd[i]))
        push!(T_age, Float64(t.age[i])); push!(T_dia, t.dia[i] * 2.54); push!(T_ht, t.ht[i] * 0.3048); push!(T_w, t.tpa[i] * 2.47105)
        push!(T_cc, Float64(coalesce(t.cclcd[i], 0.0))); push!(T_cr, Float64(coalesce(t.cr[i], 0.0))); push!(T_out, out)
        push!(T_dia2, out == "alive" ? t.dia[j] * 2.54 : NaN); push!(T_ht2, out == "alive" ? t.ht[j] * 0.3048 : NaN); push!(T_w2, out == "alive" ? t.tpa[j] * 2.47105 : NaN)
      end
      if k > 1 && !haskey(byid, (key, sb[i], trn[i], vs[k-1]))   # first tallied at a later visit: ingrowth
        push!(G_iid, ivkey[(key, vs[k-1])]); push!(G_sb, sb[i]); push!(G_sym, sym[i]); push!(G_grp, grp[i]); push!(G_age, Float64(t.age[i]))
        push!(G_dia, t.dia[i] * 2.54); push!(G_ht, t.ht[i] * 0.3048); push!(G_w, t.tpa[i] * 2.47105)
      end
    end
  end
  tr = DataFrame(iid=T_iid, sb=T_sb, tr=T_tr, sym=T_sym, grp=T_grp, spcd=T_spcd, age=T_age, dia_cm=T_dia, ht_m=T_ht, w_ha=T_w, cclcd=T_cc, cr=T_cr, out=T_out,
                 dia2_cm=T_dia2, ht2_m=T_ht2, w2_ha=T_w2)
  ig = DataFrame(iid=G_iid, sb=G_sb, sym=G_sym, grp=G_grp, age=G_age, dia_cm=G_dia, ht_m=G_ht, w_ha=G_w)
  return HBAD2Intervals(iv, tr, ig)
end

const INTERVAL_FILES = ("intervals.arrow", "interval_trees.arrow", "interval_ingrowth.arrow")

"Write the three tables as Arrow files under `dir` (`intervals.arrow`, `interval_trees.arrow`, `interval_ingrowth.arrow`; the big two zstd-compressed)."
function write_hbad2_intervals(dir::AbstractString, x::HBAD2Intervals)
  mkpath(dir)
  Arrow.write(joinpath(dir, INTERVAL_FILES[1]), x.intervals)
  Arrow.write(joinpath(dir, INTERVAL_FILES[2]), x.trees; compress=:zstd)
  Arrow.write(joinpath(dir, INTERVAL_FILES[3]), x.ingrowth; compress=:zstd)
  return dir
end

"Read the three Arrow files of `dir` into memory (copies; the files are not kept open)."
function read_hbad2_intervals(dir::AbstractString)::HBAD2Intervals
  for f in INTERVAL_FILES
    isfile(joinpath(dir, f)) || error("read_hbad2_intervals: $(joinpath(dir, f)) not found")
  end
  rd(f) = DataFrame(Arrow.Table(joinpath(dir, f)); copycols=true)
  return HBAD2Intervals(rd(INTERVAL_FILES[1]), rd(INTERVAL_FILES[2]), rd(INTERVAL_FILES[3]))
end

"""
    load_hbad2_intervals(cache_dir, db_path, split; states, table, rebuild=false) -> HBAD2Intervals

The interval product of `split` from the Arrow cache in `cache_dir`, built (and written there) when the cache is missing or `rebuild`. The cache records the database path and
the split in `cache_key.txt`; a cache built for another database / split / state list is refused (delete it or pass `rebuild=true`).
"""
function load_hbad2_intervals(cache_dir::AbstractString, db_path::AbstractString, split::DataFrame; states::AbstractVector{<:Integer}=[12, 13],
                              table::AbstractString="curated_trees", rebuild::Bool=false)::HBAD2Intervals
  sp = sort(select(split, :statecd, :unitcd, :countycd, :plot, :set), [:statecd, :unitcd, :countycd, :plot])
  key = string("db=", abspath(db_path), "\ntable=", table, "\nstates=", join(sort(collect(states)), ","), "\nsplit_rows=", nrow(sp), "\nsplit_hash=",
               string(hash((sp.statecd, sp.unitcd, sp.countycd, sp.plot, sp.set)), base=16), "\n")
  kf = joinpath(cache_dir, "cache_key.txt")
  have = all(f -> isfile(joinpath(cache_dir, f)), INTERVAL_FILES) && isfile(kf)
  if have && !rebuild
    read(kf, String) == key || error("interval cache $cache_dir was built for another database / split / state list (cache_key.txt differs); delete it or set rebuild")
    return read_hbad2_intervals(cache_dir)
  end
  @info "building the HBAD2 interval product" db=db_path cache=cache_dir plots=nrow(sp)
  x = build_hbad2_intervals(db_path, split; states=states, table=table)
  write_hbad2_intervals(cache_dir, x)
  write(kf, key)
  return x
end

"""
    compare_interval_products(a::HBAD2Intervals, b::HBAD2Intervals) -> NamedTuple

Order-independent comparison of two interval products (e.g. the prototype's Arrow files and a rebuild): rows are matched through the natural key (plot, v1 date) of
their interval -- never through `iid` -- and, inside an interval, through (SUBP, TREE) for tracked trees and the multiset of rows for ingrowth. Returns the counts of
rows only in `a` / only in `b` and the number of rows with a differing value (floats compared with `isequal`, i.e. exactly), per table.
"""
function compare_interval_products(a::HBAD2Intervals, b::HBAD2Intervals)
  nat(iv) = Dict(r.iid => (r.st, r.u, r.c, r.p, r.d1) for r in eachrow(iv))
  na, nb = nat(a.intervals), nat(b.intervals)
  ivrow(iv) = Dict((r.st, r.u, r.c, r.p, r.d1) => (r.d2, r.dt, r.k, r.trt, r.dst, r.harv, r.planted, r.eco, r.set, r.l3) for r in eachrow(iv))
  ia, ib = ivrow(a.intervals), ivrow(b.intervals)
  iv_only_a = count(k -> !haskey(ib, k), keys(ia)); iv_only_b = count(k -> !haskey(ia, k), keys(ib))
  iv_diff = count(k -> haskey(ib, k) && !isequal(ia[k], ib[k]), keys(ia))
  tcols = (:sb, :tr, :sym, :grp, :spcd, :age, :dia_cm, :ht_m, :w_ha, :cclcd, :cr, :out, :dia2_cm, :ht2_m, :w2_ha)
  gcols = (:sb, :sym, :grp, :age, :dia_cm, :ht_m, :w_ha)
  # NaN-safe row keys: replace NaN by a sentinel so that isequal-based hashing is exact
  fixnan(x) = x isa Float64 && isnan(x) ? :NaN : x
  function rowsof2(df, nm, cols)
    d = Dict{Any,Int}()
    for i in 1:nrow(df)
      key = (nm[df.iid[i]], (fixnan(df[i, c]) for c in cols)...)
      d[key] = get(d, key, 0) + 1
    end
    d
  end
  ta, tb = rowsof2(a.trees, na, tcols), rowsof2(b.trees, nb, tcols)
  ga, gb = rowsof2(a.ingrowth, na, gcols), rowsof2(b.ingrowth, nb, gcols)
  only(x, y) = sum(max(v - get(y, k, 0), 0) for (k, v) in x; init=0)
  return (intervals=(rows_a=nrow(a.intervals), rows_b=nrow(b.intervals), only_a=iv_only_a, only_b=iv_only_b, differing=iv_diff),
          trees=(rows_a=nrow(a.trees), rows_b=nrow(b.trees), only_a=only(ta, tb), only_b=only(tb, ta)),
          ingrowth=(rows_a=nrow(a.ingrowth), rows_b=nrow(b.ingrowth), only_a=only(ga, gb), only_b=only(gb, ga)))
end
