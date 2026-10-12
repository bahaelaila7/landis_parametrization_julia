# Region-wide B_MAX + ANPP data floors (Julia port of FIA_DATA_PREP/build_floors_4cell_shade.py and
# build_anpp_floor_4cell_shade.py), computed ONLY from the run's TRAIN plots (split_plots.csv, so a species-
# balanced train cap also de-biases the plantation-driven p99) with the run's EXACT species categories
# (species_map.csv) — no re-implemented tiering. Run the config once with `split_only: true` to produce both.
#   B_MAX floor(cat,group) = max(12000, p99 category-own biomass per plot-visit, 20 × p99 ΔAGB/Δt [3≤gap≤15 yr])
#   ANPP floor(cat,group)  = p99 of plot-wide SPECIES increments (sum-then-diff, ÷ max(1, gap yr)), mapped to cat
#   invariant enforced: bmax_floor ≥ 20 × anpp_floor (replaces the 853 "anppfix" +5 % hand bump)
#   groups: --cell-split species get one group per site cell (merged by --cell-merge) inside each L3; all other
#   categories pool the L3's cells. A group with < --min-n values falls back to the L2 pool, then the region pool.
# Writes <out>_bmax.csv (category,l3,lu,n,p99_biom,p99_incr,bmax_floor,level) and <out>_anpp.csv
# (category,l3,lu,n,anpp_floor,anpp_floor_max,level), one row per (category, L3, cell present in train).
#   ./julia_gdal.sh --project=. tools/build_floors_region.jl --run <split_only output dir> --db data_r4/slim_r4.duckdb \
#       --out data_r4/floor_r4 [--cell-split PITA,PIEL] [--cell-merge PIEL:A=AB,B=AB] [--min-n 30]
import DuckDB, DataFrames, CSV, Statistics, Dates
using DataFrames: DataFrame, groupby, combine, nrow, select, innerjoin
args = Dict{String,String}(); let i = 1; while i <= length(ARGS); args[ARGS[i][3:end]] = ARGS[i+1]; i += 2; end; end
run = args["run"]; db = args["db"]; out = args["out"]
csplit = Set(split(get(args, "cell-split", "PITA,PIEL"), ","))
cmerge = Dict{String,Dict{String,String}}()
for spec in split(get(args, "cell-merge", "PIEL:A=AB,B=AB"), ";")
  isempty(spec) && continue; sp, m = split(spec, ":")
  cmerge[sp] = Dict(String(a) => String(b) for (a, b) in (split(kv, "=") for kv in split(m, ",")))
end
min_n = parse(Int, get(args, "min-n", "30")); LB = 453.592 / 4046.86; RATIO = 20.0

sp_plots = CSV.read(joinpath(run, "split_plots.csv"), DataFrame; types=Dict(:eco => String))
tr = sp_plots[sp_plots.set .== "train", :]
tr.l3 = [String(first(split(e, "|lu="))) for e in tr.eco]
tr.cell = [(p = split(e, "|lu="); length(p) == 2 ? String(p[2]) : "") for e in tr.eco]
smap = CSV.read(joinpath(run, "species_map.csv"), DataFrame; types=Dict(:species_symbol => String, :effective_species => String))
cat_of = Dict(uppercase(strip(r.species_symbol)) => r.effective_species for r in eachrow(smap))
println("train plots: $(nrow(tr)) in $(length(unique(tr.l3))) L3s; categories: $(length(unique(values(cat_of))))")

con = DuckDB.DBInterface.connect(DuckDB.DB, ":memory:")
q(s) = DataFrame(DuckDB.DBInterface.execute(con, s))
DuckDB.DBInterface.execute(con, "ATTACH '$db' AS src (READ_ONLY)")
DuckDB.register_data_frame(con, select(tr, [:statecd, :unitcd, :countycd, :plot]), "trp")
# plot-wide species biomass (g/m²) per visit, live trees, train plots only
b = q("""SELECT UPPER(TRIM(r.SPECIES_SYMBOL)) sym, t.STATECD s, t.UNITCD u, t.COUNTYCD c, t.PLOT p, t.MEASDATE md,
           SUM(t.DRYBIO_AG*t.TPA_UNADJ*$LB) v
         FROM src.curated_trees t JOIN trp ON t.STATECD=trp.statecd AND t.UNITCD=trp.unitcd AND t.COUNTYCD=trp.countycd AND t.PLOT=trp.plot
         JOIN src.REF_SPECIES r ON r.SPCD = COALESCE(t.spcd_resolved, t.SPCD)
         WHERE t.STATUSCD=1 AND t.TPA_UNADJ>0 AND t.DRYBIO_AG>0 GROUP BY ALL""")
println("species×plot×visit biomass rows: $(nrow(b))")
pinfo = Dict((r.statecd, r.unitcd, r.countycd, r.plot) => (r.l3, r.cell) for r in eachrow(tr))
b.cat = [get(cat_of, s, "_H_NA") for s in b.sym]
b.l3 = [pinfo[(r.s, r.u, r.c, r.p)][1] for r in eachrow(b)]
b.cell = [pinfo[(r.s, r.u, r.c, r.p)][2] for r in eachrow(b)]
yrs(a, z) = Dates.value(z - a) / 365.25
# --- collect values keyed (cat, l3, cell) ---
B = Dict{NTuple{3,String},Vector{Float64}}(); IB = Dict{NTuple{3,String},Vector{Float64}}(); IA = Dict{NTuple{3,String},Vector{Float64}}()
push2!(d, k, v) = push!(get!(d, k, Float64[]), v)
cb = combine(groupby(b, [:cat, :l3, :cell, :s, :u, :c, :p, :md]), :v => sum => :v)   # category-own biomass per visit
for r in eachrow(cb); push2!(B, (r.cat, r.l3, r.cell), r.v); end
for g in groupby(cb, [:cat, :s, :u, :c, :p])                                          # category ΔAGB/Δt, 3–15 yr gaps
  o = sortperm(g.md)
  for k in 2:length(o)
    gap = yrs(g.md[o[k-1]], g.md[o[k]]); 3 <= gap <= 15 && push2!(IB, (g.cat[1], g.l3[1], g.cell[1]), (g.v[o[k]] - g.v[o[k-1]]) / gap)
  end
end
for g in groupby(b, [:sym, :s, :u, :c, :p])                                           # species sum-then-diff (ANPP)
  o = sortperm(g.md)
  for k in 2:length(o)
    gap = max(1.0, yrs(g.md[o[k-1]], g.md[o[k]])); push2!(IA, (g.cat[1], g.l3[1], g.cell[1]), (g.v[o[k]] - g.v[o[k-1]]) / gap)
  end
end
p99(v) = isempty(v) ? nothing : Statistics.quantile(v, 0.99)
l2(l3) = (x = split(l3, "."); join(x[1:min(2, end)], "."))
cells_in = Dict(l => sort(unique(tr.cell[tr.l3 .== l])) for l in unique(tr.l3))
cats = sort(unique(b.cat))
collectv(d, cat, pred) = reduce(vcat, (v for ((c, l, ce), v) in d if c == cat && pred(l, ce)); init=Float64[])
rb = NamedTuple[]; ra = NamedTuple[]
for cat in cats, l3 in sort(collect(keys(cells_in)))
  cl = cells_in[l3]
  groups = (cat in csplit && !any(c -> c in ("", "X"), cl)) ?
    (let m = get(cmerge, cat, Dict{String,String}()), gs = Dict{String,Vector{String}}()
       for c in cl; push!(get!(gs, get(m, c, c), String[]), c); end; collect(values(gs)) end) : [cl]
  for grp in groups
    ladder = [("l3", (l, ce) -> l == l3 && ce in grp), ("l2", (l, ce) -> l2(l) == l2(l3)), ("region", (l, ce) -> true)]
    bv = Float64[]; iv = Float64[]; av = Float64[]; lvB = ""; lvA = ""
    for (name, pr) in ladder
      bv = collectv(B, cat, pr); iv = collectv(IB, cat, pr); lvB = name
      length(bv) >= min_n && break
    end
    for (name, pr) in ladder
      av = collectv(IA, cat, pr); lvA = name
      length(av) >= min_n && break
    end
    isempty(bv) && isempty(av) && continue
    pb = something(p99(bv), 0.0); pi = something(p99(iv), 0.0); pa = something(p99(av), 0.0)
    af = max(0.0, pa)
    bf = max(12000.0, pb, RATIO * pi, RATIO * af)
    for cell in sort(grp)
      push!(rb, (category=cat, l3=l3, lu=cell, n=length(bv), p99_biom=round(pb), p99_incr=round(pi, digits=1), bmax_floor=round(bf), level=lvB))
      push!(ra, (category=cat, l3=l3, lu=cell, n=length(av), anpp_floor=round(af, digits=1), anpp_floor_max=round(isempty(av) ? 0.0 : maximum(av), digits=1), level=lvA))
    end
  end
end
CSV.write(out * "_bmax.csv", DataFrame(rb)); CSV.write(out * "_anpp.csv", DataFrame(ra))
bdf = DataFrame(rb)
println("wrote $(out)_bmax.csv ($(nrow(bdf)) rows) and $(out)_anpp.csv; bmax_floor>12000 in $(count(>(12000), bdf.bmax_floor)) rows; levels: ",
        Dict(l => count(==(l), bdf.level) for l in unique(bdf.level)))
