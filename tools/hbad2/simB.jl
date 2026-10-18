# HBAD v2 Sim B: free stand runs on real plots (natural, never treated/disturbed/harvested), from the first visit with recruitment,
# compared at every later visit with the WHOLE observed stand (all live trees ≥ 1 in, ingrowth included, nothing tracked).
# Stand metrics: BA, N (all, ≥ 5 in), QMD, Lorey height (BA-weighted H), species BA composition. Baseline = persistence (no change).
#   ./julia_gdal.sh --project=. --threads=4 tools/hbad2/simB.jl <tag> [set=val]
include(joinpath(@__DIR__, "common.jl"))
using Distributions: Normal, ccdf
const OBS_SIGMA = parse(Float64, get(ENV, "HBAD2_OBS_SIGMA", "0"))   # within-cohort sd of ln D for the observation operator (0 = mean-field)
tag = ARGS[1]; SET = length(ARGS) >= 2 ? ARGS[2] : "val"
iv = DataFrame(Arrow.Table(joinpath(R2DIR, "intervals.arrow"))); raw = DataFrame(Arrow.Table(joinpath(R2DIR, "interval_trees.arrow")))
ig = DataFrame(Arrow.Table(joinpath(R2DIR, "interval_ingrowth.arrow")))
tr = load_trees(); keys_ = sort(unique(tr.key)); nsp = length(keys_); kix = Dict(k => i for (i, k) in enumerate(keys_))
big = Set(k for k in keys_ if k[1] != '_'); keyof(s, g) = s in big ? s : "_" * g
lon = CSV.read("/workspace/species_longevity_ref.csv", DataFrame); lond = Dict(String(r.species_symbol) => Float64(r.longevity_yr) for r in eachrow(lon))
longev = [median([get(lond, s, 300.0) for s in unique(tr.sym[tr.key .== k])]) for k in keys_]
# tag "fit" (fit_<tag>/theta.csv) or a joint point "j1/l0.50" (joint_j1/theta_l0.50.csv + its free recruitment recr_l0.50.csv)
JOINT = occursin("/", tag)
FD = JOINT ? joinpath(R2DIR, "joint_" * split(tag, "/")[1]) : joinpath(R2DIR, "fit_" * tag)
check_meta(FD); θ = read_theta(joinpath(FD, JOINT ? "theta_$(split(tag, "/")[2]).csv" : "theta.csv"), nsp); P, G = decode(θ, nsp, longev)
rc = CSV.read(joinpath(R2DIR, SITE_SUB ? "recruitment_coefs_subplot.csv" : "recruitment_coefs.csv"), DataFrame); rcd = Dict(String(r.key) => r for r in eachrow(rc))
RMULT = parse(Float64, get(ENV, "HBAD2_RECR_MULT", "1"))   # diagnostic only
R = [RecP(rcd[k].rho0 + log(RMULT), rcd[k].rhoS, rcd[k].rhoL, 2.54, 3.0, 5.0) for k in keys_]
rfile = JOINT ? joinpath(FD, "recr_$(split(tag, "/")[2]).csv") : joinpath(FD, "recr.csv")   # a fit's own recruitment (free or κ-shifted) when it has one
if isfile(rfile)
  z = vec(CSV.read(rfile, DataFrame).theta)
  R = [RecP(clamp(z[3k-2], -12, 2), clamp(z[3k-1], -1, 4), clamp(z[3k], -10, 2), 2.54, 3.0, 5.0) for k in 1:nsp]
end
# plots: in SET, ≥ 2 intervals, every interval clean and natural
iv.pk = [(a, b, c, d) for (a, b, c, d) in zip(iv.st, iv.u, iv.c, iv.p)]
pl = combine(groupby(iv, :pk), :set => first => :set, [:trt, :dst, :harv, :planted] => ((a, b, c, d) -> !any(a .| b .| c .| d)) => :clean, nrow => :nint)
pl = pl[(coalesce.(pl.set, "") .== SET) .& pl.clean .& (pl.nint .>= 2), :]
@info "Sim B plots" n=nrow(pl)
raw.key = [keyof(s, g) for (s, g) in zip(raw.sym, raw.grp)]; ig.key = [keyof(s, g) for (s, g) in zip(ig.sym, ig.grp)]
rbyi = groupby(raw, :iid); igbyi = groupby(ig, :iid); ivbypk = groupby(iv, :pk)
function stand(key, D, H, w)                     # stand metrics of tree lists
  ba = (pi / 4) .* (D ./ 100) .^ 2 .* w; BA = sum(ba)
  (BA=BA, N=sum(w), N5=sum(w[D .>= 12.7]), Nc=[sum(w[(D .>= a) .& (D .< b)]) for (a, b) in ((0, 5), (5, 12.7), (12.7, 25), (25, 1e9))], QMD=sqrt(BA / sum(w) * 4 / pi) * 100, HL=sum(ba .* H) / BA,
   comp=[sum(ba[key .== k]) for k in keys_] ./ BA)
end
obs = Vector{Vector{Tuple{Int,Any}}}(); init = []
for r in eachrow(pl)
  ivs = sort(DataFrame(ivbypk[(r.pk,)]), :k); y0 = ivs.d1[1]; v = Tuple{Int,Any}[]
  for (j, ivr) in enumerate(eachrow(ivs))
    haskey(rbyi, (ivr.iid,)) || (v = nothing; break)
    t = DataFrame(rbyi[(ivr.iid,)])
    j == 1 && push!(init, t)
    j > 1 && push!(v, (round(Int, Dates.value(ivr.d1 - y0) / 365.25), stand(t.key, t.dia_cm, t.ht_m, t.w_ha)))
    if j == nrow(ivs)                          # last visit: tracked survivors (their v2 weights) + ingrowth
      a = t[t.out .== "alive", :]; g = haskey(igbyi, (ivr.iid,)) ? DataFrame(igbyi[(ivr.iid,)]) : DataFrame(key=String[], dia_cm=Float64[], ht_m=Float64[], w_ha=Float64[])
      push!(v, (round(Int, Dates.value(ivr.d2 - y0) / 365.25), stand(vcat(a.key, g.key), vcat(a.dia2_cm, g.dia_cm), vcat(a.ht2_m, g.ht_m), vcat(a.w2_ha, g.w_ha))))
    end
  end
  v === nothing ? (length(init) > length(obs) && pop!(init)) : push!(obs, v)
end
# initial cohorts (species key × 10-yr birth class) at visit 1, capacity for recruits
CAP_EXTRA = VIGOUR ? 330 : 120; QZ = VIGOUR ? collect(VIG_Z) : [0.0]; QW = VIGOUR ? collect(VIG_W) : [1.0]
off = [1]; spv = Int[]; N0 = Float64[]; D0 = Float64[]; H0 = Float64[]; A0 = Float64[]; live = Int[]; site_plot = Int[]; zq = Float64[]
for (pi_, t) in enumerate(init)
  t.bc = floor.(Int, -t.age ./ BIRTH_W) .* 100 .+ dclass.(t.dia_cm)
  for tt in (SITE_SUB ? [t[t.sb .== b, :] for b in 1:4] : [t])        # subplot sites: each subplot (empty ones too) is a site, N per ha ×4
    if nrow(tt) > 0
      g = combine(groupby(tt, [:key, :bc]), :w_ha => (w -> sum(w) * SITE_MULT) => :N, [:w_ha, :dia_cm] => ((w, x) -> sqrt(sum(w .* x .^ 2) / sum(w))) => :D,
                  [:w_ha, :ht_m] => ((w, x) -> sum(w .* x) / sum(w)) => :H, [:w_ha, :age] => ((w, x) -> sum(w .* x) / sum(w)) => :A)
      for r in eachrow(g), (z, w) in zip(QZ, QW); push!(spv, kix[r.key]); push!(N0, r.N * w); push!(D0, r.D); push!(H0, r.H); push!(A0, max(r.A, 1.0)); push!(zq, z); end
      push!(live, nrow(g) * length(QZ))
    else
      push!(live, 0)
    end
    for _ in 1:CAP_EXTRA; push!(spv, 1); push!(N0, 0.0); push!(D0, 2.54); push!(H0, 3.0); push!(A0, 5.0); push!(zq, 0.0); end
    push!(off, length(spv) + 1); push!(site_plot, pi_)
  end
end
ns = length(live); npl = length(obs)
S = Sites(off, spv, copy(N0), copy(D0), copy(H0), copy(A0), zeros(Int, ns), zeros(ns), fill(false, ns), copy(zq)); F = FreeSites(S, copy(live), nsp; vigour=VIGOUR)
years = [maximum(first.(obs[site_plot[s]])) for s in 1:ns]
snaps = [Dict{Int,Any}() for _ in 1:ns]
want = [Set(first.(obs[site_plot[s]])) for s in 1:ns]
function snap(s, y)                                 # raw cohort arrays; plot-level metrics are computed after the run
  y in want[s] || return
  lo = S.off[s]; hi = lo + F.live[s] - 1; keep = [i for i in lo:hi if S.N[i] > 0]
  snaps[s][y] = (Int.(S.sp[keep]), S.D[keep], S.H[keep], S.N[keep] ./ SITE_MULT)
end
plotstand(k, D, H, N) = OBS_SIGMA > 0 ? stand_ln([keys_[x] for x in k], D, H, N, OBS_SIGMA) : stand([keys_[x] for x in k], D, H, N)
# observation operator: each cohort = lognormal D with quadratic mean D_c and sd σ of ln D; FIA tallies trees ≥ 2.54 cm
Nab(N, Dq, σ, T) = N * ccdf(Normal(), (log(T) - (log(Dq) - σ^2)) / σ)
BAab(N, Dq, σ, T) = N * (pi / 4) * (Dq / 100)^2 * ccdf(Normal(), (log(T) - (log(Dq) - σ^2)) / σ - 2σ)
function stand_ln(key, D, H, w, σ)
  ba = [BAab(w[i], D[i], σ, 2.54) for i in eachindex(D)]; BA = sum(ba); N = sum(Nab(w[i], D[i], σ, 2.54) for i in eachindex(D))
  cl(a, b) = sum(Nab(w[i], D[i], σ, max(a, 2.54)) - (b < 1e8 ? Nab(w[i], D[i], σ, b) : 0.0) for i in eachindex(D))
  (BA=BA, N=N, N5=sum(Nab(w[i], D[i], σ, 12.7) for i in eachindex(D)), Nc=[cl(0, 5), cl(5, 12.7), cl(12.7, 25), cl(25, 1e9)],
   QMD=sqrt(BA / N * 4 / pi) * 100, HL=sum(ba .* H) / BA, comp=[sum(ba[key .== k]) for k in keys_] ./ BA)
end
t0 = time(); run_free!(F, P, G, R, years, snap); @info "free runs" secs=round(time() - t0, digits=2) dropped=sum(F.dropped)
sites_of = [findall(==(p), site_plot) for p in 1:npl]
cat_(xs, j) = reduce(vcat, (x[j] for x in xs); init=(j == 1 ? Int[] : Float64[]))
init_arr(s) = (r = off[s]:off[s]+live[s]-1; (Int.(spv[r]), D0[r], H0[r], N0[r] ./ SITE_MULT))
st0 = [(xs = [init_arr(s) for s in sites_of[p]]; stand([keys_[x] for x in cat_(xs, 1)], cat_(xs, 2), cat_(xs, 3), cat_(xs, 4))) for p in 1:npl]
rows = NamedTuple[]
for p in 1:npl, (y, o) in obs[p]
  xs = [snaps[s][y] for s in sites_of[p]]; m = plotstand(cat_(xs, 1), cat_(xs, 2), cat_(xs, 3), cat_(xs, 4)); p0 = st0[p]
  push!(rows, (s=p, yr=y, gap=y <= 7 ? "≤7" : y <= 14 ? "8-14" : y <= 21 ? "15-21" : ">21",
    BA_o=o.BA, BA_m=m.BA, BA_0=p0.BA, N5_o=o.N5, N5_m=m.N5, N5_0=p0.N5, N_o=o.N, N_m=m.N, N_0=p0.N, QMD_o=o.QMD, QMD_m=m.QMD, QMD_0=p0.QMD,
    HL_o=o.HL, HL_m=m.HL, HL_0=p0.HL, Nc_o=o.Nc, Nc_m=m.Nc, Nc_0=p0.Nc, comp_m=0.5sum(abs.(m.comp .- o.comp)), comp_0=0.5sum(abs.(p0.comp .- o.comp))))
end
df = DataFrame(rows); CSV.write(joinpath(FD, "simB_$(SET)" * (JOINT ? "_" * split(tag, "/")[2] : "") * ".csv"), df)
# per-species BA (m²/ha) of every plot-visit: observed / simulated / initial (long format) for composition analysis
sprows = NamedTuple[]
for p in 1:npl, (y, o) in obs[p]
  xs = [snaps[s][y] for s in sites_of[p]]; m = plotstand(cat_(xs, 1), cat_(xs, 2), cat_(xs, 3), cat_(xs, 4)); p0 = st0[p]
  for (k, key) in enumerate(keys_)
    push!(sprows, (plot=p, yr=y, key=key, ba_o=o.comp[k] * o.BA, ba_m=m.comp[k] * m.BA, ba_0=p0.comp[k] * p0.BA))
  end
end
CSV.write(joinpath(FD, "simB_species_$(SET)" * (JOINT ? "_" * split(tag, "/")[2] : "") * ".csv"), DataFrame(sprows))
lr(a, b) = log.(max.(a, 1e-6) ./ max.(b, 1e-6))
@printf("Sim B %s: %d plots (%d sites), %d plot-visits\n", SET, npl, ns, nrow(df))
@printf("%-6s %6s | %-27s | %-27s | %-27s | %-27s | %-27s | %-13s\n", "gap", "n", "BA  model/persist (bias)", "N≥5in", "N all", "QMD", "Lorey H", "composition")
for g in ("≤7", "8-14", "15-21", ">21", "all")
  d = g == "all" ? df : df[df.gap .== g, :]; nrow(d) == 0 && continue
  cell(x) = (m = lr(d[!, Symbol(x, "_m")], d[!, Symbol(x, "_o")]); p = lr(d[!, Symbol(x, "_0")], d[!, Symbol(x, "_o")]); f = isfinite.(m) .& isfinite.(p); m = m[f]; p = p[f];
             @sprintf("%5.1f/%5.1f%% (%+5.1f/%+5.1f)", 100median(abs.(m)), 100median(abs.(p)), 100mean(m), 100mean(p)))
  @printf("%-6s %6d | %s | %s | %s | %s | %s | %.3f/%.3f\n", g, nrow(d), cell("BA"), cell("N5"), cell("N"), cell("QMD"), cell("HL"), mean(filter(isfinite, d.comp_m)), mean(filter(isfinite, d.comp_0)))
end
d = df
for x in ("BA", "N5", "QMD", "HL")
  dob = d[!, Symbol(x, "_o")] .- d[!, Symbol(x, "_0")]; dm = d[!, Symbol(x, "_m")] .- d[!, Symbol(x, "_0")]; f = isfinite.(dob) .& isfinite.(dm); dob = dob[f]; dm = dm[f]
  @printf("change skill %-4s: corr(sim Δ, obs Δ) = %.3f   mean obs Δ %.3f  mean sim Δ %.3f\n", x, cor(dm, dob), mean(dob), mean(dm))
end

println("mean N by size class (trees/ha) [<5 cm, 5-12.7, 12.7-25, ≥25]: obs / sim / initial")
for g in ("≤7", "8-14", "15-21", ">21")
  d = df[df.gap .== g, :]; nrow(d) == 0 && continue
  f(c) = join((@sprintf("%.0f", x) for x in sum(d[!, c]) ./ nrow(d)), " ")
  @printf("  %-6s obs [%s]  sim [%s]  init [%s]\n", g, f(:Nc_o), f(:Nc_m), f(:Nc_0))
end
