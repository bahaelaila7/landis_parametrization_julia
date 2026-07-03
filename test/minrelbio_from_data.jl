# MIN_REL_BIOMASS shade-class boundaries from FIA, per stratum (eco×lu), the ALSTKCD way.
# Shade class = canopy density; ALSTKCD is the observed stocking class (1=overstocked/densest … 5=nonstocked/open),
# so shade_class = 6-ALSTKCD (1=open … 5=deep shade). For each stratum we bin plot-visits by shade class and
# look at their ≥5" live stand biomass (g/m²); the boundaries between classes are the ABSOLUTE thresholds A[c]
# = B_MAX_ECO · MIN_REL[c]. (We estimate A[c] directly — no B_MAX needed; divide by B_MAX_ECO later for the
# relative form.) Boundary A[c] = geometric mean of adjacent class medians. Also a separation check: how well
# biomass alone recovers the ALSTKCD class (tight → biomass-shade abstraction holds; overlap → it's leaky).
#   Biomass = Σ (DRYBIO_AG · TPA_UNADJ)·0.11208  [lbs/acre → g/m²], live ≥5" DBH.  Run:
#   ./julia_gdal.sh --project=. test/minrelbio_from_data.jl
using DataFrames, DuckDB, Printf, CSV, Statistics
con = DBInterface.connect(DuckDB.DB("/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb"))

df = DataFrame(DBInterface.execute(con, """
  WITH so AS (SELECT statecd,unitcd,countycd,plot, CASE WHEN bool_or(stdorgcd=1) THEN 'artificial' ELSE 'natural' END lu FROM COND GROUP BY 1,2,3,4),   -- STDORGCD 2-way land_use (replaces old 4-way)
       strat AS (SELECT tt.statecd,tt.unitcd,tt.countycd,tt.plot, any_value(tt.epa_l3) l3, any_value(so.lu) lu
                 FROM tree_trajectories tt JOIN so ON so.statecd=tt.statecd AND so.unitcd=tt.unitcd AND so.countycd=tt.countycd AND so.plot=tt.plot
                 WHERE tt.epa_l3 IN ('8.3.5','8.5.3') GROUP BY 1,2,3,4),
       agb AS (SELECT STATECD st,UNITCD un,COUNTYCD co,PLOT pl,INVYR iv, sum(DRYBIO_AG*TPA_UNADJ)*0.11208 agb
               FROM curated_trees WHERE STATUSCD=1 AND DIA>=5 AND DRYBIO_AG IS NOT NULL AND TPA_UNADJ IS NOT NULL GROUP BY 1,2,3,4,5),
       alstk AS (SELECT STATECD st,UNITCD un,COUNTYCD co,PLOT pl,INVYR iv, arg_max(ALSTKCD,CONDPROP_UNADJ) alstkcd
                 FROM COND WHERE COND_STATUS_CD=1 AND ALSTKCD IS NOT NULL AND CONDPROP_UNADJ IS NOT NULL GROUP BY 1,2,3,4,5)
  SELECT s.l3, s.lu, a.agb, k.alstkcd
  FROM agb a JOIN alstk k ON a.st=k.st AND a.un=k.un AND a.co=k.co AND a.pl=k.pl AND a.iv=k.iv
  JOIN strat s ON s.statecd=a.st AND s.unitcd=a.un AND s.countycd=a.co AND s.plot=a.pl WHERE a.agb>0"""))
df.alstkcd = Int.(df.alstkcd); df.shade = 6 .- df.alstkcd; df.stratum = String.(df.l3) .* "|lu=" .* String.(df.lu)
println("plot-visits with AGB+ALSTKCD+stratum: ", nrow(df))

out = DataFrame(stratum=String[], shade_class=Int[], n=Int[], agb_q25=Float64[], agb_med=Float64[], agb_q75=Float64[])
thr = DataFrame(stratum=String[], enter_class=Int[], A_boundary_gm2=Float64[])
for strt in sort(unique(df.stratum))
  sub = df[df.stratum .== strt, :]
  q75s = Dict{Int,Float64}()
  for sc in 1:5
    s2 = sub[sub.shade .== sc, :]; nrow(s2)==0 && continue
    q = Statistics.quantile(Float64.(s2.agb), [0.25,0.5,0.75]); q75s[sc]=q[3]
    push!(out, (strt, sc, nrow(s2), round(q[1];digits=1), round(q[2];digits=1), round(q[3];digits=1)))
  end
  # boundary to ENTER class c = p75 of class c-1 → monotone by construction; the transition into class 5
  # (overstocked) uses class 4's p75, sidestepping overstocked stands' leaky low biomass. class 1 = 0.
  push!(thr, (strt, 1, 0.0))
  for c in 2:5
    haskey(q75s, c-1) || continue
    push!(thr, (strt, c, round(q75s[c-1]; digits=1)))
  end
end

for strt in sort(unique(out.stratum))
  println("\n=== $strt — ≥5\" stand AGB (g/m²) by shade class (ALSTKCD-derived) ===")
  @printf("%-11s %6s %8s %8s %8s\n","shade_cls","n","q25","median","q75")
  for r in eachrow(out[out.stratum.==strt,:]); @printf("%-11d %6d %8.0f %8.0f %8.0f\n", r.shade_class, r.n, r.agb_q25, r.agb_med, r.agb_q75); end
  tt = thr[thr.stratum.==strt,:]
  println("  boundaries A[c] (g/m², enter class c): ", join(["c$(r.enter_class)=$(r.A_boundary_gm2)" for r in eachrow(tt)], "  "))
  # separation check: classify each plot by A[c] and compare to its ALSTKCD shade class
  sub = df[df.stratum.==strt,:]; A=sort(tt,:enter_class).A_boundary_gm2
  pred = [searchsortedlast(A, Float64(a)) for a in sub.agb]      # highest class whose boundary <= agb
  match = mean(pred .== sub.shade); adj = mean(abs.(pred .- sub.shade) .<= 1)
  @printf("  biomass→class recovery: exact=%.1f%%, within±1=%.1f%%\n", 100match, 100adj)
end
mkpath("runs"); CSV.write("runs/minrelbio_boundaries.csv", thr); CSV.write("runs/minrelbio_agb_by_class.csv", out)
println("\nwrote runs/minrelbio_boundaries.csv (A[c] absolute thresholds) + runs/minrelbio_agb_by_class.csv")
