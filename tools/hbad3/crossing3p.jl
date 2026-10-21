# HBAD3P in-growth diagnostic on tracked trees (one remeasurement interval, VAL cohorts starting below a threshold):
#   observed: survivors (tree count) and survivors whose v2 diameter is >= the threshold (tree counts of the cohort's tracked trees)
#   model   : expected survivors (cohort N ratio) and expected crossers (sub-cohorts' survivors x lognormal fraction above the threshold, σ 0.16)
# by initial diameter class of the cohort. Separates the free-run in-growth deficit into survival vs growth-to-threshold.
#   (env as fit3p) ./julia_gdal.sh --project=. --threads=4 tools/hbad3/crossing3p.jl <fit3p tag> [threshold cm = 12.7]
include(joinpath(@__DIR__, "common3.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "plugins", "HBAD3", "HBAD3P.jl")); import .HBAD3P
include(joinpath(@__DIR__, "lib3p.jl"))
tag = ARGS[1]; THR = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 12.7; FD = joinpath(R3DIR, "fit3p_" * tag); check_meta(FD)
tr = load_trees(); keys_ = sort(unique(tr.key)); nsp = length(keys_)
M = load_fit3p(FD, keys_); P, G = M.P, M.G; wp = water_pairs(); FAREA = meta_farea(FD)
d = tr[tr.set .== "val", :]
C = sima_sites(tr, "val", keys_, wp, FAREA ? interval_forest(DataFrame(Arrow.Table(joinpath(DATA_DIR, "intervals.arrow"))), forest_table()) : nothing)
SP = C.S; S = SP; ns = length(S.off) - 1
# start-of-interval overtopping per cohort (zero-length run), then the real run
SP0 = HBAD3P.SitesP(S.off, S.sp, S.N0, S.D0, S.H0, S.A0, zeros(Int, ns), fill(1e-9, ns), S.planted, S.zq, SP.W, SP.Wn, SP.CR0)
HBAD3P.simulate!(SP0, P, G); O0 = copy(SP0.O)
HBAD3P.simulate!(SP, P, G)
# observed tree-level crossing per cohort (same keys as build_cohorts: site, key, bc)
d.bc = floor.(Int, (d.y1 .- d.age) ./ BIRTH_W) .* 100 .+ dclass.(d.dia_cm)
d.gD = [r.out == "alive" && r.dia2_cm > 0 ? (r.dia2_cm - r.dia_cm) / r.dt : NaN for r in eachrow(d)]
obs = combine(groupby(d, [:site, :key, :bc]), :out => (o -> count(!=("missing"), o)) => :n1, :out => (o -> count(==("alive"), o)) => :s,
              :gD => (g -> (x = filter(isfinite, g); isempty(x) ? NaN : sum(x))) => :gsum,
              [:out, :dia2_cm] => ((o, x) -> count(i -> o[i] == "alive" && x[i] >= THR, eachindex(o))) => :x)
ci = Dict((r.site, r.key, r.bc) => i for (i, r) in enumerate(eachrow(C.coh)))
aboveT(D) = 1 - cdf(Normal(), (log(THR) - (log(D) - 0.16^2)) / 0.16)   # D = sub-cohort QMD = lognormal median × e^{σ²}
pS = zeros(nrow(C.coh)); pX = zeros(nrow(C.coh)); pG = zeros(nrow(C.coh))
for i in eachindex(SP.N)
  p = C.parent[i]; N0c = C.agg.N0[p]; pS[p] += SP.N[i] / N0c; pX[p] += SP.N[i] * aboveT(SP.D[i]) / N0c
  pG[p] += SP.N[i] * (SP.D[i] - SP.D0[i])                       # survivor-weighted increment (normalised below)
end
pG ./= max.(pS .* C.agg.N0, 1e-12)
obs.coh = [ci[(r.site, r.key, r.bc)] for r in eachrow(obs)]
fo = [searchsortedfirst(C.parent, q) for q in 1:nrow(C.coh)]
obs.O = O0[fo[obs.coh]]
obs.D1 = C.coh.D1[obs.coh]; obs.ps = pS[obs.coh]; obs.px = pX[obs.coh]; obs.pg = pG[obs.coh] ./ C.obs.dt[obs.coh]
o = obs[(obs.D1 .< THR) .& (obs.n1 .> 0), :]
o.cls = cut = [x < 5 ? "2.5-5" : x < 8 ? "5-8" : x < 10.5 ? "8-10.5" : "10.5-12.7" for x in o.D1]
t = combine(groupby(o, :cls), nrow => :cohorts, :n1 => sum => :trees, :s => sum => :surv_obs, [:n1, :ps] => ((n, p) -> sum(n .* p)) => :surv_sim,
            :x => sum => :cross_obs, [:n1, :px] => ((n, p) -> sum(n .* p)) => :cross_sim,
            [:gsum, :s] => ((g, s) -> sum(filter(isfinite, g)) / sum(s[isfinite.(g)])) => :dD_obs, [:pg, :s] => ((g, s) -> sum(g .* s) / sum(s)) => :dD_sim)
t.cross_ratio = t.cross_sim ./ t.cross_obs; t.surv_ratio = t.surv_sim ./ t.surv_obs
sort!(t, :cls, by=x -> parse(Float64, split(x, "-")[1]))
println("tracked VAL cohorts starting below $(THR) cm (region $(get(ENV, "HBAD2_REGION", "")), fit $(tag)): one interval")
show(stdout, t; allrows=true, summary=false, eltypes=false); println()
o.Oc = [x < 0.1 ? "O<.1 open" : x < 0.3 ? "O .1-.3" : x < 0.6 ? "O .3-.6" : "O >.6 shaded" for x in o.O]
u = combine(groupby(o, [:cls, :Oc]), :s => sum => :surv, [:gsum, :s] => ((g, s) -> sum(filter(isfinite, g)) / sum(s[isfinite.(g)])) => :dD_obs,
            [:pg, :s] => ((g, s) -> sum(g .* s) / sum(s)) => :dD_sim)
u.ratio = u.dD_sim ./ u.dD_obs; sort!(u, [:cls, :Oc])
println("mean annual dD of survivors by initial class x start-of-interval overtopping:"); show(stdout, u; allrows=true, summary=false, eltypes=false); println()
o.planted = C.coh.planted[o.coh]
v = combine(groupby(o[o.Oc .== "O<.1 open", :], [:cls, :planted]), :s => sum => :surv, [:gsum, :s] => ((g, s) -> sum(filter(isfinite, g)) / sum(s[isfinite.(g)])) => :dD_obs,
            [:pg, :s] => ((g, s) -> sum(g .* s) / sum(s)) => :dD_sim)
v.ratio = v.dD_sim ./ v.dD_obs; sort!(v, [:cls, :planted])
println("OPEN-GROWN (O < .1) small cohorts by stand origin:"); show(stdout, v; allrows=true, summary=false, eltypes=false); println()
tot = (sum(t.cross_sim) / sum(t.cross_obs), sum(t.surv_sim) / sum(t.surv_obs)); println(@sprintf("ALL: crossing sim/obs %.3f | survival sim/obs %.3f", tot...))
