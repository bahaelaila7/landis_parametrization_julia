# Two CV "fronts" from a config's folds, purely on candidates' OWN stored objectives (ΣW, ΣAGB) — NO
# re-simulation, NO renormalization. For each fold: rank checkpoints by the shared-rectangle area rule
# to get the p100 (best) archive; also take the last-iteration archive. From EACH archive designate 4
# positions {extreme_w, extreme_agb, knee, "median"} (own-bbox geometry). Each position's (A_W,A_AGB)
# = that candidate's stored objectives. Then weighted-average each position across folds by the fold's
# validation-plot count (from fold_k/cv_front_plots.csv). 8 positions × (A_W,A_AGB) = 16 scalars → two
# 4-point Pareto fronts (p100 vs last), plotted together.
#   ./julia_gdal.sh --project=. tools/cv_two_fronts.jl <cv_output_dir> [n_folds]
using JLD2, CairoMakie, CSV, DataFrames, Statistics, Printf
const MK = CairoMakie
CVDIR = ARGS[1]
NF    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5

# --- geometry on a point-list (each pt = (ΣW, ΣAGB)) ---
function area_in(pts, Wm, Am)
    p = sort(pts, by=q->(q[1],-q[2])); p1=p[1]; a=(p1[1]-Wm)*(p1[2]-Am)
    for i in 1:length(p)-1; a += (p[i+1][1]-p[i][1])*((p[i][2]-Am)+(p[i+1][2]-Am))/2; end
    a
end
function positions(pts)                         # -> Dict(position => (ΣW,ΣAGB))
    ws=[q[1] for q in pts]; as=[q[2] for q in pts]
    eW = pts[argmin(ws)]; eA = pts[argmin(as)]
    nw(w)=(hi=maximum(ws);lo=minimum(ws); hi>lo ? (w-lo)/(hi-lo) : 0.0); na(a)=(hi=maximum(as);lo=minimum(as); hi>lo ? (a-lo)/(hi-lo) : 0.0)
    if length(pts) < 3
        return Dict("extreme_w"=>eW, "extreme_agb"=>eA, "knee"=>eW, "median"=>eA)
    end
    p1=(nw(eW[1]),na(eW[2])); pk=(nw(eA[1]),na(eA[2])); d12=hypot(pk[1]-p1[1],pk[2]-p1[2])
    perpq(q)= d12<=0 ? 0.0 : abs((pk[1]-p1[1])*(p1[2]-na(q[2])) - (p1[1]-nw(q[1]))*(pk[2]-p1[2]))/d12
    knee = pts[argmax(perpq(q) for q in pts)]
    M=((p1[1]+pk[1])/2,(p1[2]+pk[2])/2)         # midpoint of extremes chord; corner = (0,0)
    segd(q)= begin qn=(nw(q[1]),na(q[2])); ABx=-M[1];ABy=-M[2];d2=ABx^2+ABy^2; t=d2<=0 ? 0.0 : clamp(((qn[1]-M[1])*ABx+(qn[2]-M[2])*ABy)/d2,0,1); hypot(qn[1]-(M[1]+t*ABx),qn[2]-(M[2]+t*ABy)) end
    med = pts[argmin(segd(q) for q in pts)]
    Dict("extreme_w"=>eW, "extreme_agb"=>eA, "knee"=>knee, "median"=>med)
end

POSES = ["extreme_w","extreme_agb","knee","median"]
rows = NamedTuple[]     # (fold, archive, position, A_W, A_AGB, n_val)
for k in 1:NF
    fd = joinpath(CVDIR, "fold_$k")
    files = filter(f -> occursin(r"search_state@\d+\.jld2$", f), readdir(fd; join=true))
    isempty(files) && (@warn "no checkpoints in $fd"; continue)
    gen(f)=parse(Int, match(r"@(\d+)\.jld2", f).captures[1])
    archs = Dict{Int,Vector{Tuple{Float64,Float64}}}()
    for f in files
        st = JLD2.load_object(f)
        pts = [(sum(Float64.(@view c.fx.objectives[1:2:end])), sum(Float64.(@view c.fx.objectives[2:2:end]))) for c in collect(st.archive)]
        isempty(pts) || (archs[gen(f)] = pts)
    end
    gens = sort(collect(keys(archs)))
    allp = vcat((archs[g] for g in gens)...)
    Wm=minimum(p[1] for p in allp); Am=minimum(p[2] for p in allp)
    ranked = sort(gens, by = g -> (area_in(archs[g], Wm, Am), -length(archs[g])))
    p100_g = ranked[1]; last_g = gens[end]
    n_val = length(unique(CSV.read(joinpath(fd, "cv_front_plots.csv"), DataFrame).plot_id))
    for (atype, g) in (("p100", p100_g), ("last", last_g))
        pos = positions(archs[g])
        for nm in POSES
            (aw, aa) = pos[nm]
            push!(rows, (fold=k, archive=atype, position=nm, A_W=aw, A_AGB=aa, n_val=n_val, sel_gen=g))
        end
    end
    println("fold_$k: p100=@$p100_g last=@$last_g n_val=$n_val")
end
df = DataFrame(rows)
CSV.write(joinpath(CVDIR, "cv_two_fronts_perfold.csv"), df)

# weighted average (by n_val) per (archive, position) -> 16 scalars
agg = combine(groupby(df, [:archive, :position]),
    [:A_W, :n_val] => ((w, n) -> sum(w .* n) / sum(n)) => :A_W,
    [:A_AGB, :n_val] => ((a, n) -> sum(a .* n) / sum(n)) => :A_AGB)
CSV.write(joinpath(CVDIR, "cv_two_fronts.csv"), agg)
println("\n=== weighted-avg CV fronts (16 scalars) ===")
show(sort(agg, [:archive, :A_W]); allrows=true, allcols=true); println()

# --- plot the two 4-point fronts ---
fig = MK.Figure(size=(820, 660))
ax = MK.Axis(fig[1,1], xlabel="A_W (held-out, val-plot weighted)", ylabel="A_AGB (held-out, val-plot weighted)",
             title="$(basename(CVDIR)) — CV fronts: p100-selected vs last-iteration")
for (atype, col, mk) in (("p100", :seagreen, :utriangle), ("last", :crimson, :diamond))
    sub = sort(agg[agg.archive .== atype, :], :A_W)
    xs = sub.A_W; ys = sub.A_AGB
    MK.lines!(ax, xs, ys, color=(col,0.6), linewidth=2)
    MK.scatter!(ax, xs, ys, color=col, marker=mk, markersize=13, strokecolor=:black, strokewidth=1)
    for r in eachrow(sub); MK.text!(ax, r.A_W, r.A_AGB, text="  "*r.position, fontsize=10, color=col, align=(:left,:center)); end
end
MK.axislegend(ax, [MK.MarkerElement(color=c, marker=m, markersize=13, strokecolor=:black, strokewidth=1) for (c,m) in ((:seagreen,:utriangle),(:crimson,:diamond))],
              ["p100-selected","last-iteration"], position=:rt)
out = joinpath(CVDIR, "cv_two_fronts.png"); MK.save(out, fig)
println("wrote $out  (+ cv_two_fronts.csv, cv_two_fronts_perfold.csv)")
