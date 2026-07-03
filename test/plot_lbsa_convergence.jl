# LBSA convergence plot from convergence.csv: best-so-far loss (left axis, log) + max temperature
# (right axis, log) vs trial, with DASHED VERTICAL LINES at every restart. Incumbent loss faint.
#   Run: ./julia_gdal.sh --project=. test/plot_lbsa_convergence.jl <output_dir>
import CSV, DataFrames, CairoMakie
const MK = CairoMakie; const DF = DataFrames
outdir = ARGS[1]
df = CSV.read(joinpath(outdir, "convergence.csv"), DF.DataFrame)
restarts = df.trial[df.restart .== 1]
println("trials=$(DF.nrow(df)), restarts=$(length(restarts)), best=$(round(minimum(df.best_loss);sigdigits=5))")

fig = MK.Figure(size=(1150, 540))
ax = MK.Axis(fig[1, 1]; xlabel="trial", ylabel="loss (log)", yscale=log10,
  title="LBSA convergence — best loss + max temperature ($(length(restarts)) restarts)")
ax2 = MK.Axis(fig[1, 1]; ylabel="max temperature (log)", yaxisposition=:right, yscale=log10, ygridvisible=false)
MK.hidespines!(ax2); MK.hidexdecorations!(ax2); MK.linkxaxes!(ax, ax2)

# restart markers first (behind the curves)
for r in restarts
  MK.vlines!(ax, r; color=(:red, 0.35), linestyle=:dash, linewidth=0.7)
end
# incumbent (faint) + best-so-far on the loss axis
MK.lines!(ax, df.trial, max.(df.current_loss, eps()); color=(:gray, 0.35), linewidth=0.5)
MK.lines!(ax, df.trial, df.best_loss; color=:navy, linewidth=1.6)
# max temperature on the right axis (where defined & >0)
tm = .!isnan.(df.t_max) .& (df.t_max .> 0)
MK.lines!(ax2, df.trial[tm], df.t_max[tm]; color=(:darkorange, 0.8), linewidth=1.0)

MK.Legend(fig[1, 2],
  [MK.LineElement(color=:navy, linewidth=2), MK.LineElement(color=:gray), MK.LineElement(color=:darkorange, linewidth=2), MK.LineElement(color=:red, linestyle=:dash)],
  ["best loss", "incumbent", "max temp", "restart"]; framevisible=true)
out = joinpath(outdir, "convergence.png")
MK.save(out, fig); println("wrote $out")
