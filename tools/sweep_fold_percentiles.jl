# Sweep ALL archive checkpoints of one fold, rank each front by the shared-rectangle area rule
# (min area -> max knee -> max count), and plot the fronts at percentile positions of THAT ranking
# (p100=best, then p75/p50/p25). INTRA-fold: the train split (and thus cell-normalization) is frozen
# across all checkpoints, so the stored front objectives are directly comparable — no re-sim needed.
#
# RANKING uses the FULL rectangle (all candidates of all checkpoints). The PLOT uses a TIGHT rectangle
# bounding only the four shown fronts (p25/p50/p75/p100), so candidates from unplotted checkpoints don't
# blow out the axes; closures/fills/knee/"median" are drawn against that tight box. Legend shows the
# full-rectangle RANK plus the tight-box area/knee of each shown front.
#   ./julia_gdal.sh --project=. tools/sweep_fold_percentiles.jl <fold_dir> [out.png]
using JLD2, CairoMakie, Statistics, Printf
const MK = CairoMakie
FOLD = ARGS[1]
OUT  = length(ARGS) >= 2 ? ARGS[2] : joinpath(FOLD, "sweep_percentiles.png")

files = sort(filter(f -> occursin(r"search_state@\d+\.jld2$", f), readdir(FOLD; join=true)),
             by = f -> parse(Int, match(r"@(\d+)\.jld2", f).captures[1]))
gen(f) = parse(Int, match(r"@(\d+)\.jld2", f).captures[1])

fronts = Dict{Int,Vector{Tuple{Float64,Float64}}}()
for f in files
    st = JLD2.load_object(f)
    pts = [(sum(Float64.(@view c.fx.objectives[1:2:end])), sum(Float64.(@view c.fx.objectives[2:2:end])))
           for c in collect(st.archive)]
    isempty(pts) && continue
    sort!(pts, by = p -> (p[1], -p[2]))
    fronts[gen(f)] = pts
end
gens = sort(collect(keys(fronts)))

# --- rectangle-parameterized geometry (rect = (Wmin,Wmax,Amin,Amax)) ---
function area_in(pts, Wm, Am)                    # left block (upper-extreme horizontal) + trapezoid to Am
    p1 = pts[1]; a = (p1[1]-Wm)*(p1[2]-Am)
    for i in 1:length(pts)-1; (w1,a1)=pts[i]; (w2,a2)=pts[i+1]; a += (w2-w1)*((a1-Am)+(a2-Am))/2; end
    a
end
nx(w,r)= r[2]>r[1] ? (w-r[1])/(r[2]-r[1]) : 0.0
ny(a,r)= r[4]>r[3] ? (a-r[3])/(r[4]-r[3]) : 0.0
perp(p,p1,pk,d12)= d12<=0 ? 0.0 : abs((pk[1]-p1[1])*(p1[2]-p[2]) - (p1[1]-p[1])*(pk[2]-p1[2]))/d12
function knee_in(pts, r)
    length(pts) < 3 && return 0.0
    p1=(nx(pts[1][1],r),ny(pts[1][2],r)); pk=(nx(pts[end][1],r),ny(pts[end][2],r)); d12=hypot(pk[1]-p1[1],pk[2]-p1[2])
    maximum(perp((nx(p[1],r),ny(p[2],r)),p1,pk,d12) for p in pts)
end
function knee_pt_in(pts, r)
    length(pts) < 3 && return pts[1]
    p1=(nx(pts[1][1],r),ny(pts[1][2],r)); pk=(nx(pts[end][1],r),ny(pts[end][2],r)); d12=hypot(pk[1]-p1[1],pk[2]-p1[2])
    pts[argmax(perp((nx(p[1],r),ny(p[2],r)),p1,pk,d12) for p in pts)]
end

# --- RANK all checkpoints on the FULL rectangle ---
allpts = vcat((fronts[g] for g in gens)...)
FR = (minimum(p[1] for p in allpts), maximum(p[1] for p in allpts), minimum(p[2] for p in allpts), maximum(p[2] for p in allpts))
# p101 = GLOBAL non-dominated front over EVERY candidate in EVERY archive (union envelope; ≥ every p_x by construction)
p101 = (let allc = unique(allpts); sort(filter(p -> !any(q -> q[1] <= p[1] && q[2] <= p[2] && q != p, allc), allc), by = p -> (p[1], -p[2])) end)
scored = [(g=g, area=area_in(fronts[g], FR[1], FR[3]), knee=knee_in(fronts[g], FR), n=length(fronts[g])) for g in gens]
order = sort(scored, by = s -> (s.area, -s.knee, -s.n))
N = length(order)
PCTS = [100, 75, 50, 25]
pick = Dict(100=>1, 75=>clamp(round(Int,0.25*(N-1))+1,1,N), 50=>clamp(round(Int,0.50*(N-1))+1,1,N), 25=>clamp(round(Int,0.75*(N-1))+1,1,N))
sel_g = Dict(p => order[pick[p]].g for p in PCTS)
last_gen = gens[end]                                     # final iteration's archive
last_pct = findfirst(p -> sel_g[p] == last_gen, PCTS)    # percentile it coincides with (or nothing)
show_last = isnothing(last_pct)                          # draw the last front separately only if it's not a p_x

# --- TIGHT rectangle over the shown fronts (+ the last front if it's separate) for readable drawing ---
selpts = vcat((fronts[sel_g[p]] for p in PCTS)..., (show_last ? fronts[last_gen] : Tuple{Float64,Float64}[])..., p101)
Wmin=minimum(p[1] for p in selpts); Wmax=maximum(p[1] for p in selpts)
Amin=minimum(p[2] for p in selpts); Amax=maximum(p[2] for p in selpts)
TR = (Wmin,Wmax,Amin,Amax)

COLORS = Dict(100=>:seagreen, 75=>:steelblue, 50=>:darkorange, 25=>:crimson)
MARKERS= Dict(100=>:utriangle, 75=>:circle, 50=>:diamond, 25=>:rect)
MSIZES = Dict(100=>7, 75=>16, 50=>11, 25=>8)

println("fold=$FOLD  checkpoints=$N")
println("  FULL rect (ranking):  W∈[$(round(FR[1],sigdigits=4)),$(round(FR[2],sigdigits=4))] AGB∈[$(round(FR[3],sigdigits=4)),$(round(FR[4],sigdigits=4))]")
println("  TIGHT rect (plotted): W∈[$(round(Wmin,sigdigits=4)),$(round(Wmax,sigdigits=4))] AGB∈[$(round(Amin,sigdigits=4)),$(round(Amax,sigdigits=4))]")
fig = MK.Figure(size=(900, 720))
runlabel(dir) = uppercase(replace(replace(basename(rstrip(dir,'/')), r"_outputs$"=>"", "fl5_l4cover_mocmaes_simA_cv5_bmaxfloor_"=>"", "_domall_stdorg"=>""), "_"=>" "))
foldlabel = titlecase(replace(basename(FOLD), "_"=>" "))
ax = MK.Axis(fig[1,1], xlabel="age-distribution loss  (ΣW, Wasserstein)", ylabel="biomass loss  (ΣAGB)",
             title="$(runlabel(dirname(FOLD))) $foldlabel — training archive fronts at area-rank percentiles (p100 = best of $N)")
bx, by = FR[1], FR[3]     # housing-rectangle bottom-left corner — area + closures use this (every front's point is interior ⇒ area > 0)
MK.lines!(ax, [bx,FR[2],FR[2],bx,bx], [by,by,FR[4],FR[4],by], color=(:black,0.35), linewidth=1)   # housing rectangle (top-right may be off the zoomed view)
# p101 union envelope drawn FIRST (underneath) so the p100 front + its knee/median/chord designation sit ON TOP
MK.lines!(ax, [p[1] for p in p101], [p[2] for p in p101], color=(:purple,0.9), linewidth=2.6)
MK.scatter!(ax, [p[1] for p in p101], [p[2] for p in p101], color=(:purple,0.9), marker=:pentagon, markersize=11, strokecolor=:white, strokewidth=0.6)
# draw a closed front: polyline + upper-extreme horizontal closure to left edge + lower-extreme vertical closure to bottom edge
function draw_front!(pts, c, mk, ms, afill; dotted=false)
    xs=[q[1] for q in pts]; ys=[q[2] for q in pts]
    MK.poly!(ax, MK.Point2f.(vcat(bx,bx,xs,xs[end]), vcat(by,ys[1],ys,by)), color=(c,afill), strokewidth=0)
    MK.lines!(ax, xs, ys, color=(c,0.6), linewidth=2, linestyle=(dotted ? :dot : :solid))
    MK.scatter!(ax, xs, ys, color=(c,0.45), marker=mk, markersize=ms, strokecolor=c, strokewidth=1.2)
    MK.lines!(ax, [bx,xs[1]], [ys[1],ys[1]], color=c, linestyle=:dash, linewidth=1.1)
    MK.lines!(ax, [xs[end],xs[end]], [ys[end],by], color=c, linestyle=:dash, linewidth=1.1)
end
# designate positions on the p100 front for ANY size (n=1: all one candidate; n=2: knee=median; n>=3: full)
function designate!(pts)
    n=length(pts); eW=pts[1]; eA=pts[end]                        # extremes: min-W (first) & min-AGB (last, after sort)
    Md = ((eW[1]+eA[1])/2, (eW[2]+eA[2])/2); Cd = (bx, by)
    if n == 1
        mp = pts[1]; kp = pts[1]
    else
        A=(nx(Md[1],TR),ny(Md[2],TR)); B=(nx(Cd[1],TR),ny(Cd[2],TR))
        segd(q)=(qn=(nx(q[1],TR),ny(q[2],TR)); ABx=B[1]-A[1];ABy=B[2]-A[2];d2=ABx^2+ABy^2; t=d2<=0 ? 0.0 : clamp(((qn[1]-A[1])*ABx+(qn[2]-A[2])*ABy)/d2,0,1); hypot(qn[1]-(A[1]+t*ABx),qn[2]-(A[2]+t*ABy)))
        mp = argmin(segd, pts)
        kp = n >= 3 ? knee_pt_in(pts, TR) : mp                   # knee = median when n<3
        MK.lines!(ax, [eW[1],eA[1]], [eW[2],eA[2]], color=(:black,0.5), linestyle=:dot, linewidth=1.3)          # extremes chord
        MK.lines!(ax, [Md[1],Cd[1]], [Md[2],Cd[2]], color=(:purple,0.65), linestyle=:dashdot, linewidth=1.4)    # midpoint→corner
        MK.scatter!(ax, [Md[1]], [Md[2]], color=:purple, marker=:xcross, markersize=12)
    end
    MK.scatter!(ax, [eW[1],eA[1]], [eW[2],eA[2]], color=:black, marker=:cross, markersize=15)                   # extremes
    MK.scatter!(ax, [kp[1]], [kp[2]], color=:gold, marker=:star5, markersize=26, strokecolor=:black, strokewidth=1.4)
    MK.text!(ax, kp[1], kp[2], text=(n>=3 ? "  knee" : "  knee=median"), align=(:left,:center), fontsize=12, color=:black)
    MK.scatter!(ax, [mp[1]], [mp[2]], color=:magenta, marker=:hexagon, markersize=22, strokecolor=:black, strokewidth=1.3)
    n>=3 && MK.text!(ax, mp[1], mp[2], text="\"median\"  ", align=(:right,:center), fontsize=12, color=:purple)
    bp = pts[argmin(p[1]+p[2] for p in pts)]                      # 5th position: best_aggregate = min (A_W + A_AGB)
    MK.scatter!(ax, [bp[1]], [bp[2]], color=:dodgerblue, marker=:diamond, markersize=17, strokecolor=:black, strokewidth=1.2)
    n>=3 && MK.text!(ax, bp[1], bp[2], text="best-agg", align=(:center,:bottom), fontsize=11, color=:dodgerblue)
    n==1 && MK.text!(ax, eW[1], eW[2], text="  single candidate (all 5 positions)", align=(:left,:center), fontsize=11, color=:black)
end
labels = String[]
for p in PCTS
    g = sel_g[p]; pts = fronts[g]
    draw_front!(pts, COLORS[p], MARKERS[p], MSIZES[p], 0.07)
    p == 100 && designate!(pts)
    lab = @sprintf("p%d  (@%d, rank %d/%d): area=%.4g knee=%.3f n=%d", p, g, pick[p], N, area_in(pts,bx,by), knee_in(pts,TR), length(pts))
    g == last_gen && (lab *= "   [= last iter]")
    push!(labels, lab); println("  ", lab)
end
leg_elems = Any[[MK.LineElement(color=COLORS[p],linewidth=2),
                 MK.MarkerElement(color=(COLORS[p],0.45),marker=MARKERS[p],markersize=MSIZES[p],strokecolor=COLORS[p],strokewidth=1.2)] for p in PCTS]
if show_last                                             # final-iteration front (black, dotted) for reference
    draw_front!(fronts[last_gen], :black, :star4, 10, 0.04; dotted=true)
    push!(leg_elems, [MK.LineElement(color=:black,linewidth=2,linestyle=:dot), MK.MarkerElement(color=(:black,0.5),marker=:star4,markersize=10,strokecolor=:black,strokewidth=1)])
    push!(labels, @sprintf("last iter (@%d): area=%.4g n=%d", last_gen, area_in(fronts[last_gen],bx,by), length(fronts[last_gen])))
end
# p101 legend entry (the envelope itself is drawn earlier, UNDER the p_x fronts so p100's designation shows on top)
push!(leg_elems, [MK.LineElement(color=(:purple,0.9),linewidth=2.6), MK.MarkerElement(color=(:purple,0.9),marker=:pentagon,markersize=11,strokecolor=:white,strokewidth=0.6)])
push!(labels, @sprintf("p101 union non-dom (all archives): area=%.4g knee=%.3f n=%d", area_in(p101,bx,by), knee_in(p101,TR), length(p101)))
Wmax=max(Wmax, maximum(p[1] for p in p101)); Amax=max(Amax, maximum(p[2] for p in p101))
mx = 0.03*(Wmax-bx); my = 0.03*(Amax-by)
MK.xlims!(ax, bx-mx, Wmax+mx); MK.ylims!(ax, by-my, Amax+my)
MK.axislegend(ax, leg_elems, labels, position=:rt, framevisible=true)
MK.save(OUT, fig); println("wrote $OUT")
