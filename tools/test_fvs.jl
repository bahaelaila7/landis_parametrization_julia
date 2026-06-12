# Standalone test of the FVS runner module (Phase 2).
#   julia --project=. tools/test_fvs.jl [DB_PATH] [ECO] [MAXPLOTS]
# Run from an env where libgfortran.so.5 is on LD_LIBRARY_PATH (e.g. the mamba env).
#
# Loads curated_trees_fvs for one ecoregion (undisturbed, longitudinal plots),
# runs FVS on up to MAXPLOTS plots, and prints the parsed Summary/TreeList/Carbon.

import DuckDB
using DataFrames

include(joinpath(@__DIR__, "..", "src", "FVS.jl"))
using .FVS

db       = length(ARGS) >= 1 ? ARGS[1] : "/workspace/FIASQLITE2PGSQL/FIADB.duckdb"
eco      = length(ARGS) >= 2 ? ARGS[2] : "8.5.3.75g"
maxplots = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 5

println("DB=$db  eco=$eco  maxplots=$maxplots")

con = DuckDB.connect(DuckDB.DB(db))
df = DataFrame(DuckDB.execute(con, """
    SELECT * FROM curated_trees_fvs
    WHERE epa_l4 = '$eco'
      AND subp_has_dstrb = false
      AND plot_meas_num > 1
"""))
close(con)
println("loaded $(nrow(df)) tree-rows")

# keep only the first MAXPLOTS plots for a quick debug run
pk = [:statecd, :unitcd, :countycd, :plot]
plots = unique(select(df, pk))
keep = first(plots, min(maxplots, nrow(plots)))
df = innerjoin(df, keep, on=pk)
println("running FVS on $(nrow(keep)) plots, $(nrow(df)) tree-rows")

dir = joinpath(@__DIR__, "..", "tmp", "fvs_run")
res = FVS.simulate(df; dir=dir, variant=FVS.variant_for_eco(eco), fiavbc=true)

println("\n=== ok=$(res.ok)  db=$(res.dbpath) ===")
println("--- FVS log tail ---")
println(join(last(split(res.log, '\n'), 25), '\n'))

for (name, tbl) in (("Summary", res.summary), ("TreeList", res.treelist), ("Carbon", res.carbon), ("FIAVBC_Summary", res.fiavbc))
    println("\n=== FVS_$name ===")
    if tbl === nothing
        println("  (table missing)")
    else
        println("  cols: ", names(tbl))
        show(first(tbl, 12); allcols=true); println()
    end
end

println("\n=== COMPARE (FVS vs observed later measurements) ===")
cmp = FVS.compare(df, res)
if nrow(cmp) == 0
    println("  no stand×year had both observed later visits and FVS output")
else
    show(cmp; allcols=true, allrows=true); println()
    using Statistics
    println("\n  median DBH W1 = ", round(median(skipmissing(cmp.dbh_w1)), digits=3), " in")
    are = collect(skipmissing(cmp.agb_rel_err))
    isempty(are) || println("  median AGB rel.err = ", round(median(are), digits=3),
                            "  (FVS vs observed, g/m²)")
end
