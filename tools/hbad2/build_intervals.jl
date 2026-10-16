# HBAD v2 data product (GA+FL, read-only DB): remeasurement intervals with the TRACKED fate of every tree.
# For each plot and each pair of consecutive visits (v1 -> v2):
#   stand at v1 = every live tree (species, age, DIA, HT, TPA_UNADJ weight) -> the competition environment and the initial cohorts;
#   outcome     = the same trees at v2 (alive: DIA2, HT2 | dead | harvested | not found), linked by (plot, SUBP, TREE).
# Ingrowth (trees first tallied at v2) is recorded separately (it is a recruitment process, not a change of a v1 cohort).
#   ./julia_gdal.sh --project=. --threads=2 tools/hbad2/build_intervals.jl
using DuckDB, DataFrames, CSV, Arrow, Statistics, Dates, Printf
const DB = "/workspace/landis_region4/data_r4/slim_r4.duckdb"
const SPLIT = "/workspace/landis_region4/runs_local/gaflst_split/split_plots.csv"
const OUT = joinpath(@__DIR__, "..", "..", "results", "hbad2"); mkpath(OUT)
con = DBInterface.connect(DuckDB.DB, ":memory:")
DBInterface.execute(con, "ATTACH '$DB' AS s (READ_ONLY)")
sp = CSV.read(SPLIT, DataFrame); sp.eco = String.(sp.eco); sp.set = String.(sp.set)
DuckDB.register_data_frame(con, select(sp, :statecd, :unitcd, :countycd, :plot, :eco, :set), "sp")
q(sql) = DataFrame(DBInterface.execute(con, sql))
t = q("""
  SELECT t.STATECD st, t.UNITCD u, t.COUNTYCD c, t.PLOT p, t.SUBP sb, t.TREE tr, t.MEASDATE md, t.STATUSCD s, t.tree_fate fate,
         t.spcd_resolved spcd, r.SPECIES_SYMBOL sym, r.SFTWD_HRDWD grp, t.DIA dia, t.HT ht, t.CCLCD cclcd, t.CR cr, t.TPA_UNADJ tpa,
         t.estimated_age age, t.epa_l3 l3, t.cond_stdorgcd stdorg, t.cond_siteclcd sitecl, t.cond_sicond si,
         coalesce(t.TRTCD1,0) trt1, coalesce(t.DSTRBCD1,0) dst1, sp.eco eco, sp."set" "set"
  FROM s.curated_trees t JOIN sp ON sp.statecd=t.STATECD AND sp.unitcd=t.UNITCD AND sp.countycd=t.COUNTYCD AND sp.plot=t.PLOT
  LEFT JOIN s.REF_SPECIES r ON r.SPCD = t.spcd_resolved
  WHERE t.STATECD IN (12,13)""")
DBInterface.close!(con)
t.md = Date.(t.md); t.sym = coalesce.(t.sym, "UNK"); t.grp = coalesce.(t.grp, "H")
@info "tree-visit rows" nrow(t)
t.pk = [(a, b, c, d) for (a, b, c, d) in zip(t.st, t.u, t.c, t.p)]
# visits per plot
pv = combine(groupby(t, :pk), :md => (x -> [sort(unique(x))]) => :visits)
vis = Dict(r.pk => r.visits for r in eachrow(pv))
byid = Dict{Tuple{NTuple{4,Int},Int,Int,Date},Int}()     # (plot, subp, tree, date) -> row
for (i, r) in enumerate(eachrow(t)); byid[(r.pk, r.sb, r.tr, r.md)] = i; end
# plot-visit flags: any treatment / disturbance code recorded at that visit, any harvest fate of a tree recorded at that visit
pflag = combine(groupby(t, [:pk, :md]), :trt1 => (x -> any(>(0), x)) => :trt, :dst1 => (x -> any(>(0), x)) => :dst,
                :fate => (x -> any(f -> startswith(coalesce(f, ""), "harvest"), x)) => :harv, :stdorg => (x -> any(==(1), skipmissing(x))) => :planted)
pfl = Dict((r.pk, r.md) => r for r in eachrow(pflag))
rows = NamedTuple[]; ing = NamedTuple[]; ivs = NamedTuple[]
iid = 0
for (pk, vs) in vis
  length(vs) < 2 && continue
  for k in 1:length(vs)-1
    d1, d2 = vs[k], vs[k+1]; global iid += 1
    dt = Dates.value(d2 - d1) / 365.25
    f2 = pfl[(pk, d2)]
    push!(ivs, (iid=iid, st=pk[1], u=pk[2], c=pk[3], p=pk[4], d1=d1, d2=d2, dt=dt, k=k, trt=f2.trt, dst=f2.dst, harv=f2.harv, planted=pfl[(pk, d1)].planted))
  end
end
ivdf = DataFrame(ivs); ivkey = Dict((r.st, r.u, r.c, r.p, r.d1) => r.iid for r in eachrow(ivdf))
for (i, r) in enumerate(eachrow(t))
  vs = vis[r.pk]; k = searchsortedfirst(vs, r.md)
  if r.s == 1 && k < length(vs)                                     # live at v1 -> tracked outcome at v2
    j = get(byid, (r.pk, r.sb, r.tr, vs[k+1]), 0)
    out = j == 0 ? "missing" : t.s[j] == 1 ? "alive" : startswith(coalesce(t.fate[j], ""), "harvest") ? "harvest" : "dead"
    push!(rows, (iid=ivkey[(r.pk..., r.md)], sb=r.sb, tr=r.tr, sym=r.sym, grp=r.grp, spcd=r.spcd, age=Float64(r.age), dia_cm=r.dia * 2.54, ht_m=r.ht * 0.3048,
                 w_ha=r.tpa * 2.47105, cclcd=coalesce(r.cclcd, 0.0), cr=coalesce(r.cr, 0.0), out=out,
                 dia2_cm=(out == "alive" ? t.dia[j] * 2.54 : NaN), ht2_m=(out == "alive" ? t.ht[j] * 0.3048 : NaN), w2_ha=(out == "alive" ? t.tpa[j] * 2.47105 : NaN)))
  end
  if r.s == 1 && k > 1 && !haskey(byid, (r.pk, r.sb, r.tr, vs[k-1]))   # first tallied at a later visit: ingrowth
    push!(ing, (iid=ivkey[(r.pk..., vs[k-1])], sym=r.sym, grp=r.grp, age=Float64(r.age), dia_cm=r.dia * 2.54, ht_m=r.ht * 0.3048, w_ha=r.tpa * 2.47105))
  end
end
tr = DataFrame(rows); ig = DataFrame(ing)
meta = unique(select(t, :st, :u, :c, :p, :eco, :set, :l3)); meta = combine(groupby(meta, [:st, :u, :c, :p]), first)
ivdf = leftjoin(ivdf, meta, on=[:st, :u, :c, :p])
Arrow.write(joinpath(OUT, "intervals.arrow"), ivdf); Arrow.write(joinpath(OUT, "interval_trees.arrow"), tr; compress=:zstd)
Arrow.write(joinpath(OUT, "interval_ingrowth.arrow"), ig; compress=:zstd)
@printf("intervals %d (plots %d); tracked trees %d; ingrowth %d\n", nrow(ivdf), length(unique(ivdf.p .+ 100000 .* ivdf.c)), nrow(tr), nrow(ig))
println(combine(groupby(tr, :out), nrow => :n))
println(combine(groupby(ivdf, [:set]), nrow => :n, :trt => mean => :trt, :dst => mean => :dst, :harv => mean => :harv, :planted => mean => :planted, :dt => mean => :dt))
