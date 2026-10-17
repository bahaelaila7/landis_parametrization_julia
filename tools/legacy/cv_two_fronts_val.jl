# Aggregate the per-fold reselection CSVs (fold_k/cv_reselect_metrics.csv from the CV_RESELECT hook) into the
# 8 positions (p100 & last × {extreme_w, extreme_agb, knee, "median"}), weighted-averaged across folds by the
# fold's validation-plot count (n_val), and plot the two Pareto fronts (p100-selected vs last) in BOTH objective
# spaces: cv_two_fronts_val.png (held-out) and cv_two_fronts_train.png (train) — same archives/positions.
#   ./julia_gdal.sh --project=. tools/cv_two_fronts_val.jl <cv_output_dir> [n_folds]
using CSV, DataFrames, CairoMakie, Printf
const MK = CairoMakie
CVDIR = ARGS[1]
NF    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5

dfs = DataFrame[]
for k in 1:NF
    p = joinpath(CVDIR, "fold_$k", "cv_reselect_metrics.csv")
    isfile(p) ? push!(dfs, CSV.read(p, DataFrame)) : @warn "missing $p"
end
df = vcat(dfs...)
wavg(v, n) = sum(v .* n) / sum(n)
agg = combine(groupby(df, [:archive, :position]),
    [:A_W, :n_val]        => wavg => :A_W,        [:A_AGB, :n_val]       => wavg => :A_AGB,
    [:A_W_train, :n_val]  => wavg => :A_W_train,  [:A_AGB_train, :n_val] => wavg => :A_AGB_train)
CSV.write(joinpath(CVDIR, "cv_two_fronts.csv"), agg)
println("=== weighted CV fronts over $(length(dfs)) folds (val + train, 8 positions) ===")
show(sort(agg, [:archive, :A_W]); allrows=true, allcols=true); println()

# short, human-readable run label from the output-dir name (strips the common boilerplate)
runlabel(dir) = uppercase(replace(replace(basename(rstrip(dir,'/')), r"_outputs$"=>"", "fl5_l4cover_mocmaes_simA_cv5_bmaxfloor_"=>"", "_domall_stdorg"=>""), "_"=>" "))
const RUNLABEL = runlabel(CVDIR)

function plot_fronts(wcol, acol, space, out)
    fig = MK.Figure(size=(820, 640))
    ax = MK.Axis(fig[1,1],
                 xlabel="age-distribution loss  (A_W, Wasserstein)   —  cross-fold, weighted by validation size",
                 ylabel="biomass loss  (A_AGB)",
                 title="$RUNLABEL — $space CV collapsed fronts: p100 vs last iteration vs p101 (union non-dom)",
                 titlesize=14)
    for (atype, col, mk) in (("p100", :seagreen, :utriangle), ("last", :crimson, :diamond), ("p101", :purple, :pentagon))
        sub = sort(agg[agg.archive .== atype, :], wcol)
        MK.lines!(ax, sub[!, wcol], sub[!, acol], color=(col,0.65), linewidth=2.2)
        MK.scatter!(ax, sub[!, wcol], sub[!, acol], color=col, marker=mk, markersize=14, strokecolor=:black, strokewidth=1)
        for r in eachrow(sub); MK.text!(ax, r[wcol], r[acol], text="  "*r.position, fontsize=9, color=col, align=(:left,:center)); end
    end
    MK.axislegend(ax, [MK.MarkerElement(color=c, marker=m, markersize=13, strokecolor=:black, strokewidth=1) for (c,m) in ((:seagreen,:utriangle),(:crimson,:diamond),(:purple,:pentagon))],
                  ["p100 (best-on-val archive)","last iteration","p101 (union non-dom)"], position=:rt)
    MK.save(out, fig); println("wrote $out")
end
plot_fronts(:A_W,       :A_AGB,       "validation", joinpath(CVDIR, "cv_collapsed_fronts_val.png"))
plot_fronts(:A_W_train, :A_AGB_train, "train",      joinpath(CVDIR, "cv_collapsed_fronts_train.png"))
