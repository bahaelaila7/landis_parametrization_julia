# Per-fold convergence plots (10 subplots: 5 folds × [convergence | pop+archive], row-major = alternating).
# Convergence subplot: TRAIN lowest-aggregate (metrics.csv best_train_loss = min-train-agg representative) vs
# VAL lowest-aggregate (per checkpoint, min over the archive of A_W_val+A_AGB_val, from cv_val_fronts.csv — the
# CORRECT archive-min-val, not the logged rep-val which is only the train-argmin's val, shown faint dotted for
# contrast). Aggregate = A_W + A_AGB. Pop+archive subplot: pop size (λ) and archive size vs iteration (twin y).
#   ./julia_gdal.sh --project=. tools/cv_convergence.jl <cv_output_dir> [n_folds]
using CSV, DataFrames, CairoMakie, Printf
const MK = CairoMakie
CVDIR = ARGS[1]
NF    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5
runlabel(dir) = uppercase(replace(replace(basename(rstrip(dir,'/')), r"_outputs$"=>"", "fl5_l4cover_mocmaes_simA_cv5_bmaxfloor_"=>"", "_domall_stdorg"=>""), "_"=>" "))
const RUNLABEL = runlabel(CVDIR)

fig = MK.Figure(size=(1300, 320*NF + 40))
MK.Label(fig[0, :], "$RUNLABEL — per-fold convergence (training vs held-out) & search population", fontsize=17, font=:bold)
for k in 1:NF
    fd = joinpath(CVDIR, "fold_$k")
    m  = CSV.read(joinpath(fd, "metrics.csv"), DataFrame)
    vf = CSV.read(joinpath(fd, "cv_val_fronts.csv"), DataFrame)
    rs = CSV.read(joinpath(fd, "cv_reselect_metrics.csv"), DataFrame)
    p100gen = rs[rs.archive .== "p100", :sel_gen][1]   # iteration whose archive was selected as p100 (val-area best)
    # ALL curves from ONE consistent source (cv_val_fronts.csv → same post-hoc val seeds), so rep_val ≥ val_min
    # by construction (rep is a member). aggregate = A_W + A_AGB.
    vf.agg_train = vf.A_W_train .+ vf.A_AGB_train
    vf.agg_val   = vf.A_W       .+ vf.A_AGB
    g = combine(groupby(vf, :gen), :agg_train => minimum => :train, :agg_val => minimum => :val)
    sort!(g, :gen)

    # --- convergence subplot (col 1) ---
    axc = MK.Axis(fig[k,1], xlabel="iteration (generation)", ylabel="lowest archive aggregate loss  (A_W + A_AGB)",
                  title="Fold $k — best-in-archive loss: training vs held-out")
    MK.lines!(axc, g.gen, g.train, color=:steelblue, linewidth=2, label="training")
    MK.lines!(axc, g.gen, g.val, color=:crimson, linewidth=2, label="held-out (validation)")
    MK.scatter!(axc, g.gen, g.val, color=:crimson, markersize=5)
    MK.vlines!(axc, [p100gen], color=(:black,0.75), linestyle=:dash, linewidth=1.5, label="best-on-val archive (iter $p100gen)")
    MK.axislegend(axc, position=:rt, framevisible=true, labelsize=9)

    # --- population + archive subplot (col 2), twin y ---
    axp = MK.Axis(fig[k,2], xlabel="iteration (generation)", ylabel="population size (λ)", title="Fold $k — search population (λ) & archive size", ylabelcolor=:purple)
    MK.lines!(axp, m.iteration, m.pop_size, color=:purple, linewidth=2)
    MK.vlines!(axp, [p100gen], color=(:black,0.5), linestyle=:dash, linewidth=1.3)
    axa = MK.Axis(fig[k,2], yaxisposition=:right, ylabel="archive size", ylabelcolor=:seagreen)
    MK.hidespines!(axa); MK.hidexdecorations!(axa)
    MK.linkxaxes!(axp, axa)
    MK.lines!(axa, m.iteration, m.archive_size, color=:seagreen, linewidth=2)
    println("fold_$k: ckpts=$(nrow(g)) λ:$(minimum(m.pop_size))→$(maximum(m.pop_size)) arch:$(minimum(m.archive_size))-$(maximum(m.archive_size)) | train_min=$(round(minimum(g.train),digits=3)) val_min=$(round(minimum(g.val),digits=3))@gen$(g.gen[argmin(g.val)])")
end
out = joinpath(CVDIR, "cv_convergence.png"); MK.save(out, fig)
println("wrote $out")
