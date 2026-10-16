# Observed FIA stocking cloud (GA+FL, all plot-visits of the interval data, live trees ≥ 1 in): trees/ha, QMD, SDI, per plot-visit
# (plot scale) and per subplot-visit (×4). → results/hbad2/obs_stocking.csv
include(joinpath(@__DIR__, "common.jl"))
raw = DataFrame(Arrow.Table(joinpath(R2DIR, "interval_trees.arrow"))); iv = DataFrame(Arrow.Table(joinpath(R2DIR, "intervals.arrow")))
pl = Dict(r.iid => r.planted for r in eachrow(iv))
f(g, m) = (N = sum(g.w_ha) * m; BA = sum((pi / 4) .* (g.dia_cm ./ 100) .^ 2 .* g.w_ha) * m; (N=N, QMD=sqrt(BA / N * 4 / pi) * 100, BA=BA, SDI=sum(g.w_ha .* (g.dia_cm ./ 25.4) .^ 1.605) * m))
a = combine(groupby(raw, :iid), AsTable([:w_ha, :dia_cm]) => (t -> f(t, 1.0)) => AsTable); a.scale .= "plot"
b = combine(groupby(raw, [:iid, :sb]), AsTable([:w_ha, :dia_cm]) => (t -> f(t, 4.0)) => AsTable); b.scale .= "subplot"
o = vcat(select(a, :iid, :N, :QMD, :BA, :SDI, :scale), select(b, :iid, :N, :QMD, :BA, :SDI, :scale)); o.planted = [pl[i] for i in o.iid]
CSV.write(joinpath(R2DIR, "obs_stocking.csv"), o)
for sc in ("plot", "subplot"); x = o[o.scale .== sc, :]; @printf("%s: n %d  SDI p50 %.0f p90 %.0f p95 %.0f p99 %.0f max %.0f\n", sc, nrow(x), quantile(x.SDI, [0.5, 0.9, 0.95, 0.99])..., maximum(x.SDI)); end
