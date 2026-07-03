# Convergence plots for an MO-CMA-ES run, from metrics.csv (iteration,best_train_loss,best_val_loss,
# pop_size,archive_size). Two panels: (1) representative train & val loss vs generation (log-y),
# (2) archive size + population size (IPOP restarts show as λ jumps). Writes convergence.png.
#   Run: ./julia_gdal.sh --project=. test/plot_mocmaes_convergence.jl <config.yml>
import CairoMakie, DataFrames, CSV, YAML
const MK = CairoMakie; const DF = DataFrames
cfg = YAML.load_file(ARGS[1]); outdir = cfg["output_dir"]
m = CSV.read(joinpath(outdir, "metrics.csv"), DF.DataFrame)
it = Float64.(m.iteration)
fig = MK.Figure(size=(1000, 760))
MK.Label(fig[0, 1:2], "MO-CMA-ES convergence — $(basename(outdir))"; fontsize=14, font=:bold)
# panel 1: loss
ax1 = MK.Axis(fig[1, 1:2]; xlabel="generation", ylabel="representative loss (ΣW+ΣAGB)", yscale=log10,
  title="representative train vs val loss")
MK.lines!(ax1, it, Float64.(m.best_train_loss); color=:steelblue, linewidth=2, label="train")
vv = Float64.(m.best_val_loss); ok = isfinite.(vv)
MK.lines!(ax1, it[ok], vv[ok]; color=:firebrick, linewidth=2, label="val")
MK.axislegend(ax1; position=:rt)
btr = minimum(Float64.(m.best_train_loss))
MK.text!(ax1, 0.98, 0.02; text="best train=$(round(btr,digits=4))", space=:relative, align=(:right, :bottom), fontsize=10)
# panel 2: archive + pop size
ax2 = MK.Axis(fig[2, 1:2]; xlabel="generation", ylabel="count", title="archive size & population (λ) — IPOP restarts = λ jumps")
MK.lines!(ax2, it, Float64.(m.archive_size); color=:seagreen, linewidth=2, label="archive size")
MK.lines!(ax2, it, Float64.(m.pop_size); color=:darkorange, linewidth=2, linestyle=:dash, label="pop size λ")
MK.axislegend(ax2; position=:lt)
out = joinpath(outdir, "convergence.png"); MK.save(out, fig)
println("wrote $out  (", DF.nrow(m), " generations; best train ", round(btr, digits=4),
        ", best val ", round(minimum(vv[ok]); digits=4), ")")
