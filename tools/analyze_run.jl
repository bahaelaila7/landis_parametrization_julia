# Post-hoc analysis of a completed MO run (IgelMOCMAES / MO-CMA-ES / any archive-based optimizer).
# Analyzes the TRAIN and VALIDATION objective spaces INDEPENDENTLY ("each in a vacuum"): the only thing
# tying them together is that archive admittance during the search was on the TRAIN objectives (val is
# scored post-hoc into cv_val_cache.csv). Nothing here re-mixes the two.
#
#   Phase 1 (this file, NO simulation): loads search_state@<gen>.jld2 checkpoints + cv_val_cache.csv and emits
#     (2) convergence   — best-aggregate & area-under-front vs generation                 → convergence_{train,val}.png/.csv
#     (3) front sweep    — p25/p50/p75/p100 (by front quality) + last-gen + p101 union     → sweep_fronts_{train,val}.png + sweep_summary_{train,val}.csv
#     (4) p101 positions — extreme-W, extreme-AGB, knee, median, best-aggregate            → positions_{train,val}.csv
#     (7) param tables   — candidate×(species,tier-cell)×param + wide (species,cell)×<param>_mean/_cv over p101 → params_{train,val}.csv + param_summary_{train,val}.csv
#   Phase 2 (later, simulation): items 5,6 (scatter/sMAPE/TOST for positions; p101 TOST front) + optional --test.
#
#   Run:  ./julia_gdal.sh --project=. tools/analyze_run.jl <run_dir> [config.yml] [--warmup N] [--sim] [--all] [--outsub DIR]
#     <run_dir> : directory holding search_state@<gen>.jld2 + cv_val_cache.csv (the run's output_dir)
#     config.yml: the run's config (2nd positional) — REQUIRED for Phase 2 (--sim); reproduces the train/val split
#     --warmup N: exclude generations <= N from the sweep/p101 (immature; default 10)
#     --sim     : run Phase 2 (items 5,6) — re-simulate candidates for scatter/sMAPE/TOST (needs config.yml + DB)
#     --all     : Phase 2 evaluates ALL p101 candidates (default: only the 5 designated positions)
#     --outsub  : write analysis outputs to <run_dir>/<DIR> (default: <run_dir>/analysis)
using Pan
import JLD2, CSV, DataFrames, Statistics, CairoMakie, YAML
const MK = CairoMakie; const DF = DataFrames; const ST = Statistics; const P = Pan   # P._expandenv for temp-config path expansion

# ─────────────────────────────── args ───────────────────────────────
length(ARGS) >= 1 || error("usage: analyze_run.jl <run_dir> [config.yml] [--warmup N] [--sim] [--all] [--test <param.jld2>] [--outsub DIR]")
run_dir = ARGS[1]
isdir(run_dir) || error("run_dir not found: $run_dir")
config_path = (length(ARGS) >= 2 && !startswith(ARGS[2], "--")) ? ARGS[2] : nothing   # optional 2nd positional
_argval(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i+1])
_hasflag(flag) = any(==(flag), ARGS)
warmup  = parse(Int, _argval("--warmup", "10"))    # exclude gen <= warmup from the sweep/p101
do_sim  = _hasflag("--sim")                        # Phase 2 (re-sim scatter/sMAPE/TOST)
eval_all = _hasflag("--all")                       # Phase 2 evaluates all p101 (else the 5 positions)
outsub  = _argval("--outsub", "analysis")
outdir  = joinpath(run_dir, outsub); mkpath(outdir)
println("analyze_run: run_dir=$run_dir  warmup=$warmup  → $outdir")

# ─────────── --test <param.jld2>: evaluate ONE params set on the HELD-OUT TEST split (standalone) ───────────
# Drives the styled scripts test-only (PAN_ONLY_TEST=1) → linear scatter + sMAPE + TOST on the test set for the
# given params. Needs only the params file + config (not the run's checkpoints). Output → <outdir>/test_<name>/.
let test_param = _argval("--test", "")
  if test_param != ""
    config_path === nothing && error("--test needs the run's config.yml as the 2nd positional arg")
    isfile(test_param) || error("--test params file not found: $test_param")
    println("── --test: evaluating $(basename(test_param)) on the HELD-OUT TEST split ──")
    cfg = YAML.load_file(config_path)                                    # expanded temp config, output_dir → run_dir
    for (k, v) in cfg; v isa AbstractString && (cfg[k] = P._expandenv(v)); end
    cfg["output_dir"] = abspath(run_dir)
    tmpcfg = joinpath(outdir, "_tmp_test_config.yml"); open(io -> YAML.write(io, cfg), tmpcfg, "w")
    sub = joinpath(outsub, "test_" * replace(basename(test_param), r"\.jld2$" => ""))
    mkpath(joinpath(run_dir, sub))
    nthr = parse(Int, get(ENV, "PAN_ANALYSIS_CPUS", string(Sys.CPU_THREADS)))
    for (scr, extra) in (("scatter_sim_obs_tiered.jl", String[]), ("smape_agebin.jl", String[]), ("tost_sim_obs.jl", ["0.05,0.10,0.15,0.20"]))
      cmd = addenv(`./julia_gdal.sh --project=. --threads=$nthr test/$scr $tmpcfg $extra`,
                   "PAN_PARAMS" => abspath(test_param), "PAN_OUTSUB" => sub, "PAN_ONLY_TEST" => "1", "PAN_EVAL_TEST" => "1", "PAN_OUT" => "")
      logf = joinpath(run_dir, sub, replace(scr, ".jl" => "") * ".log")
      try; run(pipeline(cmd; stdout=logf, stderr=logf)); println("  ✓ $scr"); catch e; println("  ✗ $scr FAILED (see $logf): $e"); end
    end
    rm(tmpcfg; force=true)
    println("=== --test DONE → $(joinpath(run_dir, sub))  (scatter_sim_obs_test_*, smape_agebin_test, tost_*pct_test) ===")
    exit()
  end
end

# ─────────────────────── load checkpoints + val cache ───────────────────────
# checkpoint files: search_state@<gen>.jld2 → IgelState with .archive (Vector of candidates x,fx) + .representative
ckpt_files = filter(f -> occursin(r"^search_state@\d+\.jld2$", f), readdir(run_dir))
isempty(ckpt_files) && error("no search_state@<gen>.jld2 checkpoints in $run_dir")
gen_of(f) = parse(Int, match(r"@(\d+)\.jld2$", f).captures[1])
ckpt_files = sort(ckpt_files, by=gen_of)
gens = [(gen_of(f), JLD2.load_object(joinpath(run_dir, f)).archive) for f in ckpt_files]
println("loaded $(length(gens)) checkpoints: gens $(gen_of(ckpt_files[1]))..$(gen_of(ckpt_files[end]))")

# TRAIN objectives come straight off the candidate: (sum W-terms, sum AGB-terms) over the per-eco objective vector
# (odd = Wasserstein, even = AGB; length 2 for aggregated runs → obj[1], obj[2]).
train_pt(c) = (Float64(sum(@view c.fx.objectives[1:2:end])), Float64(sum(@view c.fx.objectives[2:2:end])))
# stable candidate identity = rounded full objective vector (same ckey the driver's p101 dump uses)
ckey(c) = Tuple(round.(Float64.(collect(c.fx.objectives)), digits=10))
_rk(w, a) = (round(w, digits=6), round(a, digits=6))

# VALIDATION objectives: cv_val_cache.csv rows (A_W_train,A_AGB_train,A_W,A_AGB) → look up by TRAIN key.
valcache = Dict{Tuple{Float64,Float64},Tuple{Float64,Float64}}()
let f = joinpath(run_dir, "cv_val_cache.csv")
  isfile(f) || error("cv_val_cache.csv not found in $run_dir (needed for the validation analysis)")
  for r in DF.eachrow(CSV.read(f, DF.DataFrame))
    valcache[_rk(Float64(r.A_W_train), Float64(r.A_AGB_train))] = (Float64(r.A_W), Float64(r.A_AGB))
  end
end
val_pt(c) = get(valcache, _rk(train_pt(c)...), nothing)   # nothing if this candidate was never val-scored
println("val cache: $(length(valcache)) candidates scored on the held-out validation set")

# ─────────────────────── generalized front math (train OR val) ───────────────────────
# All operate on `pairs` :: Vector{(candidate, (W,A))}. `pf` maps a candidate → its point on the chosen split.
nondom(pairs) = filter(p -> !any(q -> q[2][1] <= p[2][1] && q[2][2] <= p[2][2] && q[2] != p[2], pairs), pairs)
function area_front(pts, Wm, Am)             # shared-rectangle + trapezoidal area under the front (floor = global min)
  isempty(pts) && return NaN
  p = sort(pts, by=q -> (q[1], -q[2])); p1 = p[1]; a = (p1[1] - Wm) * (p1[2] - Am)
  for i in 1:length(p)-1; a += (p[i+1][1] - p[i][1]) * ((p[i][2] - Am) + (p[i+1][2] - Am)) / 2; end
  a
end
function knee_dist(pts)                       # max normalized perpendicular distance from the extremes chord
  length(pts) < 3 && return 0.0
  ws = [q[1] for q in pts]; as = [q[2] for q in pts]
  nw(w) = (hi = maximum(ws); lo = minimum(ws); hi > lo ? (w - lo) / (hi - lo) : 0.0)
  na(a) = (hi = maximum(as); lo = minimum(as); hi > lo ? (a - lo) / (hi - lo) : 0.0)
  eW = pts[argmin(ws)]; eA = pts[argmin(as)]; p1 = (nw(eW[1]), na(eW[2])); pk = (nw(eA[1]), na(eA[2]))
  d12 = hypot(pk[1] - p1[1], pk[2] - p1[2])
  d12 <= 0 ? 0.0 : maximum(abs((pk[1]-p1[1])*(p1[2]-na(q[2])) - (p1[1]-nw(q[1]))*(pk[2]-p1[2])) / d12 for q in pts)
end
# 5 positions on a set of candidate/point pairs' NON-DOMINATED front: extreme-W, extreme-AGB, knee, median, best-agg.
function positions(pairs)
  nd = sort(nondom(pairs), by = p -> (p[2][1], -p[2][2]))
  cs = [p[1] for p in nd]; vp = [p[2] for p in nd]; n = length(nd)
  n == 0 && return NamedTuple[]
  ws = [q[1] for q in vp]; as = [q[2] for q in vp]
  iW = argmin(ws); iA = argmin(as); iB = argmin(ws .+ as)
  n == 1 && return [(name=nm, cand=cs[1], pt=vp[1]) for nm in ("extreme_w","extreme_agb","knee","median","best_aggregate")]
  nw(w) = (hi = maximum(ws); lo = minimum(ws); hi > lo ? (w - lo) / (hi - lo) : 0.0)
  na(a) = (hi = maximum(as); lo = minimum(as); hi > lo ? (a - lo) / (hi - lo) : 0.0)
  p1 = (nw(vp[iW][1]), na(vp[iW][2])); pk = (nw(vp[iA][1]), na(vp[iA][2])); d12 = hypot(pk[1]-p1[1], pk[2]-p1[2])
  M = ((p1[1]+pk[1])/2, (p1[2]+pk[2])/2)
  segi(i) = (qn=(nw(vp[i][1]),na(vp[i][2])); ABx=-M[1]; ABy=-M[2]; d2=ABx^2+ABy^2;
             t = d2<=0 ? 0.0 : clamp(((qn[1]-M[1])*ABx+(qn[2]-M[2])*ABy)/d2, 0, 1);
             hypot(qn[1]-(M[1]+t*ABx), qn[2]-(M[2]+t*ABy)))
  im = argmin(segi(i) for i in 1:n)
  ik = if n == 2; im else
    perpi(i) = d12<=0 ? 0.0 : abs((pk[1]-p1[1])*(p1[2]-na(vp[i][2])) - (p1[1]-nw(vp[i][1]))*(pk[2]-p1[2]))/d12
    argmax(perpi(i) for i in 1:n)
  end
  [(name="extreme_w", cand=cs[iW], pt=vp[iW]), (name="extreme_agb", cand=cs[iA], pt=vp[iA]),
   (name="knee", cand=cs[ik], pt=vp[ik]), (name="median", cand=cs[im], pt=vp[im]),
   (name="best_aggregate", cand=cs[iB], pt=vp[iB])]
end

# ─────────────────────── per-split driver ───────────────────────
# `pf` extracts a candidate's point for this split (nothing → candidate absent from this split, e.g. val-uncached).
function analyze_split(split::String, pf)
  # keep only candidates that HAVE a point on this split
  gpairs = [(g, [(c, pf(c)) for c in arch if pf(c) !== nothing]) for (g, arch) in gens]
  all_pts = [p[2] for (_, prs) in gpairs for p in prs]
  isempty(all_pts) && (println("  [$split] no points — skipping"); return nothing)
  Wm = minimum(p[1] for p in all_pts); Am = minimum(p[2] for p in all_pts)   # rectangle floor = global min on this split

  # ---- (2) convergence: per-gen best-aggregate + area under the gen's front ----
  conv = DF.DataFrame(gen=Int[], best_aggregate=Float64[], area=Float64[], front_size=Int[])
  for (g, prs) in gpairs
    isempty(prs) && continue
    front = nondom(prs); pts = [p[2] for p in front]
    push!(conv, (g, minimum(p[2][1] + p[2][2] for p in prs), area_front(pts, Wm, Am), length(front)))
  end
  CSV.write(joinpath(outdir, "convergence_$split.csv"), conv)   # merged train+val convergence figure drawn after both splits

  # ---- (3) front sweep: rank eligible gens by quality → p25/p50/p75/p100; + last gen; + p101 union ----
  elig = [(g, prs) for (g, prs) in gpairs if g > warmup && !isempty(prs)]
  isempty(elig) && (elig = [(g, prs) for (g, prs) in gpairs if !isempty(prs)])   # tiny run: fall back to all
  # quality key: (area asc, knee desc, count desc) — best front first
  qkey(prs) = (front = nondom(prs); pts = [p[2] for p in front]; (area_front(pts, Wm, Am), -knee_dist(pts), -length(front)))
  ranked = sort(elig, by = gp -> qkey(gp[2]))
  L = length(ranked)
  pick(q) = ranked[clamp(ceil(Int, (1 - q) * L), 1, L)]     # p100→rank1(best); p25→rank~0.75L
  sel = Dict("p100"=>pick(1.0), "p75"=>pick(0.75), "p50"=>pick(0.50), "p25"=>pick(0.25))
  last_gp = gpairs[end]
  # p101 = union non-dominated over EVERY eligible candidate across all eligible gens
  uniq = Dict{Any,Any}(); for (_, prs) in elig, (c, _) in prs; uniq[ckey(c)] = c; end
  uniq_pairs = [(c, pf(c)) for c in values(uniq) if pf(c) !== nothing]
  p101 = sort(nondom(uniq_pairs), by = p -> (p[2][1], -p[2][2]))

  fronts = [("p25", nondom(sel["p25"][2]), sel["p25"][1]), ("p50", nondom(sel["p50"][2]), sel["p50"][1]),
            ("p75", nondom(sel["p75"][2]), sel["p75"][1]), ("p100", nondom(sel["p100"][2]), sel["p100"][1]),
            ("last", nondom(last_gp[2]), last_gp[1]), ("p101", p101, -1)]
  swsum = DF.DataFrame(front=String[], sel_gen=Int[], front_size=Int[], area=Float64[], knee=Float64[])
  for (nm, front, g) in fronts; push!(swsum, (nm, g, length(front), area_front([p[2] for p in front], Wm, Am), knee_dist([p[2] for p in front]))); end
  CSV.write(joinpath(outdir, "sweep_summary_$split.csv"), swsum)

  # ---- (3)+(4) reference-style sweep (cf. tools/sweep_fold_percentiles*.jl): shaded area-under-front for each
  #      percentile + the p101 union envelope, with the 5 positions (✚extremes, ★knee, ⬡"median", ◆best-agg)
  #      designated ON the p101 front — so they line up with positions_$split.csv and the candidate folders.
  frip = Dict(nm => sort([p[2] for p in front], by = q -> q[1]) for (nm, front, _) in fronts)   # name → sorted pts
  shownpts = vcat((frip[nm] for nm in ("p25","p50","p75","p100","last","p101") if !isempty(frip[nm]))...)
  bx, by = Wm, Am                                                   # housing floor (global min = the area reference corner)
  Wmax = maximum(p[1] for p in shownpts); Amax = maximum(p[2] for p in shownpts)
  TR = (minimum(p[1] for p in shownpts), Wmax, minimum(p[2] for p in shownpts), Amax)           # tight box for knee/median normalisation
  nx(w) = TR[2] > TR[1] ? (w - TR[1]) / (TR[2] - TR[1]) : 0.0
  ny(a) = TR[4] > TR[3] ? (a - TR[3]) / (TR[4] - TR[3]) : 0.0
  perpd(p, p1, pk, d12) = d12 <= 0 ? 0.0 : abs((pk[1]-p1[1])*(p1[2]-p[2]) - (p1[1]-p[1])*(pk[2]-p1[2])) / d12
  knee_pt(pts) = length(pts) < 3 ? pts[1] : (p1=(nx(pts[1][1]),ny(pts[1][2])); pk=(nx(pts[end][1]),ny(pts[end][2])); d12=hypot(pk[1]-p1[1],pk[2]-p1[2]); pts[argmax(perpd((nx(p[1]),ny(p[2])),p1,pk,d12) for p in pts)])
  COLORS = Dict("p25"=>:crimson, "p50"=>:darkorange, "p75"=>:steelblue, "p100"=>:seagreen)
  MARKERS = Dict("p25"=>:rect, "p50"=>:diamond, "p75"=>:circle, "p100"=>:utriangle); MSIZES = Dict("p25"=>8, "p50"=>11, "p75"=>16, "p100"=>7)
  fig = MK.Figure(size=(940, 760))
  ax = MK.Axis(fig[1,1]; xlabel="age-distribution loss  (A_W)", ylabel="biomass loss  (A_AGB)",
               title="$split — archive fronts at area-rank percentiles (p100 = best of $L) + p101 union & 5 positions")
  MK.lines!(ax, [bx,Wmax,Wmax,bx,bx], [by,by,Amax,Amax,by]; color=(:black,0.35), linewidth=1)      # housing rectangle
  p101p = frip["p101"]                                              # p101 envelope FIRST (underneath) so designation sits on top
  MK.lines!(ax, [p[1] for p in p101p], [p[2] for p in p101p]; color=(:purple,0.9), linewidth=2.6)
  MK.scatter!(ax, [p[1] for p in p101p], [p[2] for p in p101p]; color=(:purple,0.9), marker=:pentagon, markersize=11, strokecolor=:white, strokewidth=0.6)
  draw_front!(pts, c, mk, ms, afill; dotted=false) = begin
    isempty(pts) && return
    xs=[q[1] for q in pts]; ys=[q[2] for q in pts]
    MK.poly!(ax, MK.Point2f.(vcat(bx,bx,xs,xs[end]), vcat(by,ys[1],ys,by)); color=(c,afill), strokewidth=0)
    MK.lines!(ax, xs, ys; color=(c,0.6), linewidth=2, linestyle=(dotted ? :dot : :solid))
    MK.scatter!(ax, xs, ys; color=(c,0.45), marker=mk, markersize=ms, strokecolor=c, strokewidth=1.2)
    MK.lines!(ax, [bx,xs[1]], [ys[1],ys[1]]; color=c, linestyle=:dash, linewidth=1.1)
    MK.lines!(ax, [xs[end],xs[end]], [ys[end],by]; color=c, linestyle=:dash, linewidth=1.1)
  end
  designate!(pts) = begin                                          # 5 positions (n=1: all coincide; n=2: knee=median; n≥3: full)
    n=length(pts); eW=pts[1]; eA=pts[end]; Md=((eW[1]+eA[1])/2,(eW[2]+eA[2])/2); Cd=(bx,by)
    if n == 1; mp=pts[1]; kp=pts[1]
    else
      A=(nx(Md[1]),ny(Md[2])); B=(nx(Cd[1]),ny(Cd[2]))
      segd(q)=(qn=(nx(q[1]),ny(q[2])); ABx=B[1]-A[1];ABy=B[2]-A[2];d2=ABx^2+ABy^2; t=d2<=0 ? 0.0 : clamp(((qn[1]-A[1])*ABx+(qn[2]-A[2])*ABy)/d2,0,1); hypot(qn[1]-(A[1]+t*ABx),qn[2]-(A[2]+t*ABy)))
      mp=argmin(segd, pts); kp = n>=3 ? knee_pt(pts) : mp
      MK.lines!(ax, [eW[1],eA[1]], [eW[2],eA[2]]; color=(:black,0.5), linestyle=:dot, linewidth=1.3)           # extremes chord
      MK.lines!(ax, [Md[1],Cd[1]], [Md[2],Cd[2]]; color=(:purple,0.65), linestyle=:dashdot, linewidth=1.4)     # midpoint→corner
      MK.scatter!(ax, [Md[1]], [Md[2]]; color=:purple, marker=:xcross, markersize=12)
    end
    MK.scatter!(ax, [eW[1],eA[1]], [eW[2],eA[2]]; color=:black, marker=:cross, markersize=15)                  # extremes ✚
    MK.scatter!(ax, [kp[1]], [kp[2]]; color=:gold, marker=:star5, markersize=26, strokecolor=:black, strokewidth=1.4)  # knee ★
    MK.text!(ax, kp[1], kp[2]; text=(n>=3 ? "  knee" : "  knee=median"), align=(:left,:center), fontsize=12, color=:black)
    MK.scatter!(ax, [mp[1]], [mp[2]]; color=:magenta, marker=:hexagon, markersize=22, strokecolor=:black, strokewidth=1.3)  # median ⬡
    n>=3 && MK.text!(ax, mp[1], mp[2]; text="\"median\"  ", align=(:right,:center), fontsize=12, color=:purple)
    bp = pts[argmin(p[1]+p[2] for p in pts)]
    MK.scatter!(ax, [bp[1]], [bp[2]]; color=:dodgerblue, marker=:diamond, markersize=17, strokecolor=:black, strokewidth=1.2)  # best-agg ◆
    n>=3 && MK.text!(ax, bp[1], bp[2]; text="best-agg", align=(:center,:bottom), fontsize=11, color=:dodgerblue)
    n==1 && MK.text!(ax, eW[1], eW[2]; text="  single candidate (all 5 positions)", align=(:left,:center), fontsize=11, color=:black)
  end
  labels = String[]; leg = Any[]
  for nm in ("p25","p50","p75","p100")
    pts = frip[nm]; isempty(pts) && continue
    draw_front!(pts, COLORS[nm], MARKERS[nm], MSIZES[nm], 0.07)
    push!(leg, [MK.LineElement(color=COLORS[nm],linewidth=2), MK.MarkerElement(color=(COLORS[nm],0.45),marker=MARKERS[nm],markersize=MSIZES[nm],strokecolor=COLORS[nm],strokewidth=1.2)])
    sr = swsum[swsum.front .== nm, :][1, :]
    push!(labels, "$nm (@$(sr.sel_gen), n=$(sr.front_size)): area=$(round(sr.area,sigdigits=3)) knee=$(round(sr.knee,digits=3))")
  end
  if !isempty(frip["last"]) && frip["last"] != frip["p100"]        # last iter (black dotted) only if not already a p_x
    draw_front!(frip["last"], :black, :star4, 10, 0.04; dotted=true)
    push!(leg, [MK.LineElement(color=:black,linewidth=2,linestyle=:dot), MK.MarkerElement(color=(:black,0.5),marker=:star4,markersize=10,strokecolor=:black,strokewidth=1)])
    lr = swsum[swsum.front .== "last", :][1, :]; push!(labels, "last iter (@$(lr.sel_gen), n=$(lr.front_size))")
  end
  !isempty(p101p) && designate!(p101p)                              # 5 positions on the p101 union front
  push!(leg, [MK.LineElement(color=(:purple,0.9),linewidth=2.6), MK.MarkerElement(color=(:purple,0.9),marker=:pentagon,markersize=11,strokecolor=:white,strokewidth=0.6)])
  push!(labels, "p101 union (n=$(length(p101p))) — ✚extremes ★knee ⬡median ◆best-agg")
  mxp = 0.03*(Wmax-bx) + eps(); myp = 0.03*(Amax-by) + eps(); MK.xlims!(ax, bx-mxp, Wmax+mxp); MK.ylims!(ax, by-myp, Amax+myp)
  MK.axislegend(ax, leg, labels; position=:rt, framevisible=true)
  MK.save(joinpath(outdir, "sweep_fronts_$split.png"), fig)

  # ---- (4) 5 positions on the p101 union front ----
  pos = positions(uniq_pairs)
  cand_id = Dict(ckey(p[1]) => i for (i, p) in enumerate(p101))    # id = rank on p101 (by val/train W asc)
  posdf = DF.DataFrame(position=String[], p101_id=Int[], W=Float64[], AGB=Float64[], aggregate=Float64[])
  for r in pos
    push!(posdf, (r.name, get(cand_id, ckey(r.cand), 0), r.pt[1], r.pt[2], r.pt[1] + r.pt[2]))
  end
  CSV.write(joinpath(outdir, "positions_$split.csv"), posdf)

  # ---- (7) param tables over the p101 union candidates ----
  write_param_tables(split, [p[1] for p in p101], cand_id, pf)

  println("  [$split] gens=$(length(gpairs)) uniq=$(length(uniq)) p101=$(length(p101)) sweepL=$L  → convergence/sweep/positions/params")
  return (conv=conv, sweep=swsum, positions=posdf, p101=p101)
end

# item 7 — flatten BiomassSuccessionParams to candidate × (species, tier-cell) × param, mirroring the scatter panels:
# a POOLED species (identical species-params across all its ecos) collapses to ONE "pooled" row; a TIERED species
# splits into its cell-groups (PITA A/B/C/D, PIEL AB/C/D). The grouping is inferred config-free from the per-eco
# species params (ANPP_MAX/B_MAX/PROB_MORT/PROB_ESTAB), so it matches param_split_species/param_tier_merge exactly.
const PARAMS = ["D","LONGEVITY","MATURITY","SHADE_TOL","S","ANPP_MAX","B_MAX","PROB_MORT","PROB_ESTAB","MIN_REL_BIOMASS"]
_cell(eco) = (ps = split(String(eco), "|lu="); length(ps) == 2 ? String(ps[2]) : String(eco))   # stratum → site-cell (or full eco if no |lu=)
function param_rows!(rows, ci, tr, vl, p)
  occ = Dict{Int,Vector{Tuple{Int,Int}}}()                                  # gsp → its (eco_id, sp_local) occurrences
  for eco_id in eachindex(p.ECO_LIST), (sp_local, gsp) in enumerate(p.ECO_SPECIES_IDS[eco_id])
    push!(get!(occ, Int(gsp), Tuple{Int,Int}[]), (eco_id, sp_local))
  end
  _pe(eco, sl) = length(p.PROB_ESTAB_SPP) >= eco ? Float64(p.PROB_ESTAB_SPP[eco][sl]) : NaN
  sig(e) = (eco = e[1]; sl = e[2]; (round(Float64(p.ANPP_MAX_SPP[eco][sl]), digits=4), round(Float64(p.B_MAX_SPP[eco][sl]), digits=4),
            round(Float64(p.PROB_MORT_SPP[eco][sl]), digits=6), round(isnan(_pe(eco, sl)) ? 0.0 : _pe(eco, sl), digits=6)))
  for gsp in sort(collect(keys(occ)))
    groups = Dict{Any,Vector{Tuple{Int,Int}}}()                             # tier groups = ecos sharing the species-param signature
    for e in occ[gsp]; push!(get!(groups, sig(e), Tuple{Int,Int}[]), e); end
    pooled = length(groups) == 1
    for (_, members) in sort(collect(groups), by = g -> minimum(m[1] for m in g[2]))
      eco, sl = members[1]                                                  # representative eco (species-params identical within group)
      cell = pooled ? "pooled" : join(sort(unique(_cell(p.ECO_LIST[e]) for (e, _) in members)), "")
      mat = length(p.MATURITY) >= gsp ? Float64(p.MATURITY[gsp]) : NaN
      push!(rows, (candidate=ci, train_agg=tr, val_agg=vl, species=p.SPECIES_LIST[gsp], cell=cell,
        D=Float64(p.D[gsp]), LONGEVITY=Float64(p.LONGEVITY[gsp]), MATURITY=mat, SHADE_TOL=Float64(p.SHADE_TOL[gsp]),
        S=Float64(p.S[eco][sl]), ANPP_MAX=Float64(p.ANPP_MAX_SPP[eco][sl]), B_MAX=Float64(p.B_MAX_SPP[eco][sl]),
        PROB_MORT=Float64(p.PROB_MORT_SPP[eco][sl]), PROB_ESTAB=_pe(eco, sl), MIN_REL_BIOMASS=Float64(p.MIN_REL_BIOMASS[eco][1])))
    end
  end
end
function write_param_tables(split, cands, cand_id, pf)
  rows = NamedTuple[]
  for c in cands
    tp = train_pt(c); vp = val_pt(c)
    param_rows!(rows, cand_id[ckey(c)], tp[1] + tp[2], vp === nothing ? NaN : vp[1] + vp[2], c.x)
  end
  df = DF.sort(DF.DataFrame(rows), [:candidate, :species, :cell])
  CSV.write(joinpath(outdir, "params_$split.csv"), df)
  # summary — same (species, cell) row layout as params_$split, wide <param>_mean/<param>_cv over the p101 candidates
  # (cv = coefficient of variation = std/|mean|·100%; 0 when mean is 0 or <2 candidates)
  srows = NamedTuple[]
  for gdf in DF.groupby(df, [:species, :cell])
    nt = (species=gdf.species[1], cell=gdf.cell[1], n_candidates=DF.nrow(gdf))
    for pr in PARAMS
      v = filter(!isnan, Float64.(gdf[!, pr]))
      m = isempty(v) ? NaN : ST.mean(v); s = length(v) > 1 ? ST.std(v) : 0.0
      cv = (isnan(m) || m == 0) ? 0.0 : 100.0 * s / abs(m)
      nt = merge(nt, NamedTuple{(Symbol(pr * "_mean"), Symbol(pr * "_cv"))}((m, cv)))
    end
    push!(srows, nt)
  end
  CSV.write(joinpath(outdir, "param_summary_$split.csv"), DF.sort(DF.DataFrame(srows), [:species, :cell]))
end

# ─────────────────────── run both splits ───────────────────────
println("── TRAIN analysis ──"); R_train = analyze_split("train", train_pt)
println("── VALIDATION analysis ──"); R_val = analyze_split("val", val_pt)

# ---- (2) convergence: ONE figure — train+val curves on shared subplots + archive size vs generation ----
let arch_sz = DF.DataFrame(gen=[g for (g, _) in gens], archive_size=[length(a) for (_, a) in gens])
  fig = MK.Figure(size=(960, 1000))
  ax1 = MK.Axis(fig[1,1]; xlabel="generation", ylabel="best aggregate (W+AGB)", title="best-aggregate over generations")
  ax2 = MK.Axis(fig[2,1]; xlabel="generation", ylabel="area under archive front", title="front area over generations (lower = tighter front)")
  for (R, lab, col) in ((R_train, "train", :steelblue), (R_val, "val", :firebrick))
    R === nothing && continue
    MK.lines!(ax1, R.conv.gen, R.conv.best_aggregate; color=col, linewidth=2, label=lab); MK.scatter!(ax1, R.conv.gen, R.conv.best_aggregate; color=col, markersize=5)
    MK.lines!(ax2, R.conv.gen, R.conv.area; color=col, linewidth=2, label=lab); MK.scatter!(ax2, R.conv.gen, R.conv.area; color=col, markersize=5)
  end
  MK.axislegend(ax1; position=:rt); MK.axislegend(ax2; position=:rt)
  ax3 = MK.Axis(fig[3,1]; xlabel="generation", ylabel="archive size", title="archive size over generations")
  MK.lines!(ax3, arch_sz.gen, arch_sz.archive_size; color=:seagreen, linewidth=2); MK.scatter!(ax3, arch_sz.gen, arch_sz.archive_size; color=:seagreen, markersize=5)
  MK.save(joinpath(outdir, "convergence.png"), fig)
  CSV.write(joinpath(outdir, "archive_size.csv"), arch_sz)
end
println("=== analyze_run Phase 1 DONE → $outdir ===")

# ═══════════════════════════════════ PHASE 2 (--sim): items 5 & 6 ═══════════════════════════════════
# For the designated candidates (the 5 positions per split, or ALL p101 with --all), reproduce the EXISTING
# per-candidate styled outputs by driving the project's own scripts (PAN_PARAMS + PAN_OUTSUB) — so the plots are
# byte-identical in style to the reference runs:
#   test/scatter_sim_obs_tiered.jl → per-species-tier sim-vs-obs panels (scatter_sim_obs_{split}_{mode}.png)
#   test/smape_agebin.jl           → sMAPE by species × age-bin heatmap (smape_agebin_{split}.png)
#   test/tost_sim_obs.jl           → per-species TOST forest plot (tost_{pct}pct_{split}.png) + tost_sim_obs_{pct}pct.csv
# Each candidate is simulated once (its params.jld2 under analysis/candidates/cand<gid>/); jobs run in parallel.
# Item 6 (p101 TOST-front, 4 margin subplots coloured by #species equivalent) is then assembled from those CSVs.
if do_sim
  config_path === nothing && error("--sim needs the run's config.yml as the 2nd positional arg")
  println("\n═══ PHASE 2: driving styled scripts (config=$config_path, eval_all=$eval_all) ═══")
  MARGINS = [0.05, 0.10, 0.15, 0.20]; MPCT = round.(Int, 100 .* MARGINS)
  # 1) expanded temp config: expand every ${VAR}/~ path, point output_dir at THIS run_dir (so PAN_OUTSUB nests under it)
  cfg = YAML.load_file(config_path)
  for (k, v) in cfg; v isa AbstractString && (cfg[k] = P._expandenv(v)); end
  cfg["output_dir"] = abspath(run_dir)
  tmpcfg = joinpath(outdir, "_tmp_config.yml"); open(io -> YAML.write(io, cfg), tmpcfg, "w")

  # 2) collect the candidates to evaluate = union over both splits' designated positions (or all p101 with --all).
  #    gid = a stable global id per unique candidate (by objective key); dump each one's params for the scripts.
  candir(gid) = joinpath(outdir, "candidates", "cand$gid")
  key2gid = Dict{Any,Int}(); gid_params = Dict{Int,Any}(); roles = NamedTuple[]
  for (split, R) in (("train", R_train), ("val", R_val))
    R === nothing && continue
    ids = eval_all ? collect(1:length(R.p101)) : sort(unique(R.positions.p101_id))
    posname = Dict{Int,Vector{String}}(); for r in DF.eachrow(R.positions); push!(get!(posname, r.p101_id, String[]), r.position); end
    for i in ids
      c = R.p101[i][1]; k = ckey(c)
      gid = get!(key2gid, k, length(key2gid) + 1); gid_params[gid] = c.x
      push!(roles, (gid=gid, split=split, p101_id=i, positions=join(get(posname, i, ["(all)"]), "+"),
                    W=R.p101[i][2][1], AGB=R.p101[i][2][2]))
    end
  end
  for (gid, px) in gid_params; mkpath(candir(gid)); JLD2.save_object(joinpath(candir(gid), "params.jld2"), px); end
  CSV.write(joinpath(outdir, "candidates_manifest.csv"), DF.DataFrame(roles))
  ncand = length(gid_params); println("  $ncand unique candidate(s) → analysis/candidates/cand<gid>/ (manifest: candidates_manifest.csv)")

  # 3) drive the 3 scripts per candidate, in parallel. Each nested run does BOTH splits internally.
  # concurrency: PAN_ANALYSIS_CPUS caps the core budget (Sys.CPU_THREADS reports the whole node under SLURM → set it
  # to $SLURM_CPUS_PER_TASK to avoid over-subscription). nconc parallel candidates × nthr threads each.
  ncpu = parse(Int, get(ENV, "PAN_ANALYSIS_CPUS", string(Sys.CPU_THREADS)))
  nconc = clamp(fld(ncpu, 4), 1, 6); nthr = max(1, fld(ncpu, nconc))
  jobs = [(gid, scr, extra) for gid in keys(gid_params)
          for (scr, extra) in (("scatter_sim_obs_tiered.jl", String[]), ("smape_agebin.jl", String[]),
                               ("tost_sim_obs.jl", [join(string.(MARGINS), ",")]))]
  println("  driving $(length(jobs)) script-jobs ($ncand cand × 3), $nconc concurrent × $nthr threads …")
  asyncmap(jobs; ntasks=nconc) do (gid, scr, extra)
    cmd = addenv(`./julia_gdal.sh --project=. --threads=$nthr test/$scr $tmpcfg $extra`,
                 "PAN_PARAMS" => abspath(joinpath(candir(gid), "params.jld2")),
                 "PAN_OUTSUB" => joinpath(outsub, "candidates", "cand$gid"), "PAN_OUT" => "")
    logf = joinpath(candir(gid), replace(scr, ".jl" => "") * ".log")
    try
      run(pipeline(cmd; stdout=logf, stderr=logf)); print("  ✓ cand$gid/$scr\n")
    catch e
      print("  ✗ cand$gid/$scr FAILED (see $logf): $e\n")
    end
    nothing
  end

  # 4) item 6 — p101 TOST-front: per split, 4 margin subplots; each evaluated candidate coloured by #species
  #    passing TOST at that margin (read back from the scripts' tost_sim_obs_<pct>pct.csv).
  function tost_front(split, R)
    R === nothing && return
    gid_of = Dict(ckey(R.p101[i][1]) => get(key2gid, ckey(R.p101[i][1]), 0) for i in 1:length(R.p101))
    evi = [i for i in 1:length(R.p101) if get(gid_of, ckey(R.p101[i][1]), 0) != 0 && haskey(gid_params, gid_of[ckey(R.p101[i][1])])]
    isempty(evi) && (println("  [$split] no evaluated candidates for tost_front"); return)
    # per-point label = candidate number (folder cand<gid>) + its designated position(s)
    _abbr = Dict("extreme_w"=>"eW", "extreme_agb"=>"eAGB", "knee"=>"knee", "median"=>"med", "best_aggregate"=>"best")
    pos_of = Dict{Int,String}()
    for r in DF.eachrow(R.positions)
      a = get(_abbr, r.position, r.position); pos_of[r.p101_id] = haskey(pos_of, r.p101_id) ? pos_of[r.p101_id] * "+" * a : a
    end
    lbl_of = Dict(i => "c$(gid_of[ckey(R.p101[i][1])])" * (haskey(pos_of, i) ? " " * pos_of[i] : "") for i in evi)
    ntot = 0; countpass = Dict{Tuple{Int,Int},Int}()   # (p101_idx, pct) → #species equivalent
    for i in evi
      gid = gid_of[ckey(R.p101[i][1])]
      for pct in MPCT
        f = joinpath(candir(gid), "tost_sim_obs_$(pct)pct.csv"); isfile(f) || continue
        t = CSV.read(f, DF.DataFrame); tt = t[t.split .== split, :]
        countpass[(i, pct)] = count(tt.equivalent); ntot = max(ntot, DF.nrow(tt))
      end
    end
    ntot = max(ntot, 1)
    fig = MK.Figure(size=(1180, 1020))
    MK.Label(fig[0, 1:2], "$split — p101 front, coloured by #species passing TOST equivalence  (of up to $ntot species)"; fontsize=14, font=:bold)
    xs = [p[2][1] for p in R.p101]; ys = [p[2][2] for p in R.p101]; o = sortperm(xs); sc = nothing
    for (mi, pct) in enumerate(MPCT)
      row, col = fldmod1(mi, 2)
      ax = MK.Axis(fig[row, col]; xlabel="W objective", ylabel="AGB objective", title="±$(pct)% band")
      MK.lines!(ax, xs[o], ys[o]; color=(:gray, 0.35), linewidth=1)            # full p101 front (context)
      ex = [xs[i] for i in evi]; ey = [ys[i] for i in evi]; cs = [get(countpass, (i, pct), 0) for i in evi]
      sc = MK.scatter!(ax, ex, ey; color=cs, colormap=:viridis, colorrange=(0, ntot), markersize=15, strokecolor=:black, strokewidth=0.5)
      MK.text!(ax, ex, ey; text=[lbl_of[i] for i in evi], fontsize=9, align=(:left, :bottom), offset=(6, 4), color=:black)
      xr = maximum(xs) - minimum(xs); xr = xr > 0 ? xr : 1.0                 # right padding so the rightmost label isn't clipped
      MK.xlims!(ax, minimum(xs) - 0.05xr, maximum(xs) + 0.22xr)
    end
    sc !== nothing && MK.Colorbar(fig[1:2, 3], sc; label="# species passing ±band TOST ($split)")
    MK.save(joinpath(outdir, "tost_front_$split.png"), fig)
    println("  [$split] tost_front → $(length(evi)) candidates coloured across ±5/10/15/20%")
  end
  tost_front("train", R_train); tost_front("val", R_val)
  rm(tmpcfg; force=true)
  println("=== analyze_run Phase 2 DONE → $outdir  (per-candidate styled plots under analysis/candidates/) ===")
end
