# Demo: shared-rectangle "area under the Pareto front" for a few archive checkpoints of one fold.
# Uses the STORED per-candidate objectives (ΣW = sum of odd entries, ΣAGB = sum of even entries) —
# the real reselection pipeline substitutes held-out (val) objectives. Closure per the spec:
#   upper extreme (min-W / max-AGB) -> dashed HORIZONTAL line to the rectangle's LEFT edge;
#   lower extreme (max-W / min-AGB) -> dashed VERTICAL line to the rectangle's BOTTOM edge.
# Area = left block [Wmin,W(p1)]x[AGBmin,AGB(p1)] + trapezoid under the polyline down to AGBmin.
using JLD2, CairoMakie
const MK = CairoMakie
F3 = "runs/fl5_l4cover_mocmaes_simA_cv5_bmaxfloor_l1_domall_stdorg_outputs/fold_3"
CPS = [60, 63, 77]
COLORS = [:steelblue, :darkorange, :seagreen]
MARKERS = [:circle, :diamond, :utriangle]     # distinct shape per front
MSIZES  = [16, 11, 7]                          # descending so overlapping markers stay visible (concentric)

fronts = Dict{Int,Vector{Tuple{Float64,Float64}}}()
for n in CPS
    st = JLD2.load_object(joinpath(F3, "search_state@$n.jld2"))
    pts = [(sum(Float64.(@view c.fx.objectives[1:2:end])), sum(Float64.(@view c.fx.objectives[2:2:end])))
           for c in collect(st.archive)]
    sort!(pts, by = p -> (p[1], -p[2]))          # W ascending (AGB descending along a min-min front)
    fronts[n] = pts
end

allpts = vcat(values(fronts)...)
Wmin = minimum(p[1] for p in allpts); Wmax = maximum(p[1] for p in allpts)
Amin = minimum(p[2] for p in allpts); Amax = maximum(p[2] for p in allpts)

# area under a front within the shared rectangle (left horizontal extension + trapezoid to AGBmin)
function front_area(pts)
    p1 = pts[1]
    area = (p1[1] - Wmin) * (p1[2] - Amin)       # left block from upper extreme's horizontal line
    for i in 1:length(pts)-1
        (w1,a1) = pts[i]; (w2,a2) = pts[i+1]
        area += (w2 - w1) * ((a1 - Amin) + (a2 - Amin)) / 2
    end
    area
end
# knee = max perpendicular distance from the extremes line, in rectangle-normalized coords
function knee_dist(pts)
    length(pts) < 3 && return 0.0
    nx(w) = Wmax>Wmin ? (w-Wmin)/(Wmax-Wmin) : 0.0; ny(a) = Amax>Amin ? (a-Amin)/(Amax-Amin) : 0.0
    p1 = (nx(pts[1][1]), ny(pts[1][2])); pk = (nx(pts[end][1]), ny(pts[end][2]))
    d12 = hypot(pk[1]-p1[1], pk[2]-p1[2]); d12 <= 0 && return 0.0
    maximum(abs((pk[1]-p1[1])*(p1[2]-ny(p[2])) - (p1[1]-nx(p[1]))*(pk[2]-p1[2]))/d12 for p in pts)
end

fig = MK.Figure(size=(880, 720))
ax = MK.Axis(fig[1,1], xlabel="ΣW (age-shape loss)", ylabel="ΣAGB loss",
             title="L1 fold_3 — area under the Pareto front in a shared rectangle")
# shared rectangle
MK.lines!(ax, [Wmin,Wmax,Wmax,Wmin,Wmin], [Amin,Amin,Amax,Amax,Amin], color=:black, linewidth=1.5)
labels = String[]
for (i,n) in enumerate(CPS)
    pts = fronts[n]; c = COLORS[i]
    xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
    # filled closure polygon (faint)
    px = vcat(Wmin, Wmin, xs, xs[end]); py = vcat(Amin, ys[1], ys, Amin)
    MK.poly!(ax, MK.Point2f.(px, py), color=(c, 0.08), strokewidth=0)
    MK.lines!(ax, xs, ys, color=(c, 0.6), linewidth=2)          # the front polyline
    MK.scatter!(ax, xs, ys, color=(c, 0.45), marker=MARKERS[i], markersize=MSIZES[i],
                strokecolor=c, strokewidth=1.2)                 # alpha fill + distinct shape/size per front
    MK.lines!(ax, [Wmin, xs[1]], [ys[1], ys[1]], color=c, linestyle=:dash, linewidth=1.3)   # upper->horizontal (left)
    MK.lines!(ax, [xs[end], xs[end]], [ys[end], Amin], color=c, linestyle=:dash, linewidth=1.3) # lower->vertical (down)
    a = front_area(pts)
    push!(labels, "@$n: area=$(round(a, sigdigits=4)), knee=$(round(knee_dist(pts), digits=3)), n=$(length(pts))")
    println(labels[end])
end
MK.axislegend(ax, [[MK.LineElement(color=COLORS[i], linewidth=2),
                    MK.MarkerElement(color=(COLORS[i], 0.45), marker=MARKERS[i], markersize=MSIZES[i], strokecolor=COLORS[i], strokewidth=1.2)] for i in 1:length(CPS)],
              labels, position=:rt, framevisible=true)
out = joinpath(F3, "demo_front_area.png"); MK.save(out, fig)
println("\nrectangle: W∈[$(round(Wmin,sigdigits=4)),$(round(Wmax,sigdigits=4))]  AGB∈[$(round(Amin,sigdigits=4)),$(round(Amax,sigdigits=4))]")
println("wrote $out")
