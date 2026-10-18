# HBAD2 recruitment data fit (port of the prototype tools/hbad2/recruitment_check.jl, branch hbad-v2): Poisson log-linear model of the sapling ingrowth
# into the 1-in tally (first-tallied trees with D < 12.7 cm; one microplot sapling = 185.2 trees/ha) of the undisturbed NATURAL intervals, per species key, on TRAIN:
#   E[count of a site and key] = dt · exp(ρ0' + ρS ln(1 + BA_key(D ≥ 12.7 cm) [m²/ha]) + ρL SDI/600)      (sites = subplots, densities ×4; ρ0 = ρ0' + ln 4)
# The coefficients are the `recruitment_csv` of an hbad2_calib run (recruitment: fixed | kappa) and the start of `recruitment: free`.
# Species keys and filters are those of the calibration (HBAD2Calib.load_tracked_trees with the given switches). VAL deviance of the nested models is printed.
#   PAN_DATA=/workspace/landis_region4/data_r4 ./julia_gdal.sh --project=. tools/hbad2_recruitment_fit.jl --split <split_plots.csv> --states 12,13 --cache <dir> --out <csv>
#       [--db <duckdb>] [--min_sp_trees 3000] [--compare <reference csv>]   (--compare: max abs difference of the coefficients, exit 1 if > 1e-8)
using Pan, DataFrames, CSV, Printf, Statistics, LinearAlgebra
const HC = Pan.HBAD2Calib
argval(name, default) = (i = findfirst(==(name), ARGS); i === nothing ? default : ARGS[i+1])
spl = Pan.Data.read_split_plots(argval("--split", ""))
states = parse.(Int, split(argval("--states", "12,13"), ","))
db = argval("--db", joinpath(get(ENV, "PAN_DATA", "/workspace/landis_region4/data_r4"), "slim_r4.duckdb"))
sw = HC.CalibSwitches(; min_sp_trees=parse(Int, argval("--min_sp_trees", "3000")))
sw.site_sub || error("the recruitment fit is defined for subplot sites")
x = Pan.Data.load_hbad2_intervals(argval("--cache", ""), db, spl; states=states)
tr = HC.load_tracked_trees(x, sw); keys_ = sort(unique(tr.key))
big = Set(k for k in keys_ if k[1] != '_')
ig = copy(x.ingrowth); ig.key = [s in big ? s : "_" * g for (s, g) in zip(ig.sym, ig.grp)]
ok = Set(unique(tr.iid)); ig = filter(r -> r.iid in ok && r.dia_cm < 12.7, ig)
ig.site = ig.iid .* 10 .+ ig.sb
cnt = Dict((r.site, r.key) => r.n for r in eachrow(combine(groupby(ig, [:site, :key]), nrow => :n)))
tr.ba = (pi / 4) .* (tr.dia_cm ./ 100) .^ 2 .* tr.w_ha; tr.sdi = tr.w_ha .* (tr.dia_cm ./ 25.4) .^ 1.605
st = combine(groupby(tr, :site), :sdi => sum => :SDI, :ba => sum => :BA, :dt => first => :dt, :set => first => :set, :planted => first => :planted, :iid => first => :iid)
ivm = Dict(r.iid => r for r in eachrow(unique(select(tr, :iid, :dt, :set, :planted))))
for sid in sort(collect(setdiff(Set(ig.site), Set(st.site))))       # subplots without trees at v1 (gaps) that receive ingrowth: sites with SDI 0
  i = sid ÷ 10; m = ivm[i]; push!(st, (site=sid, SDI=0.0, BA=0.0, dt=m.dt, set=m.set, planted=m.planted, iid=i))
end
st = st[.!st.planted, :]; sort!(st, :site)
println("recruitment fitted on NATURAL (non-planted) sites: ", nrow(st), " | keys ", length(keys_), ": ", join(keys_, ","))
sp = combine(groupby(tr, [:site, :key]), [:ba, :dia_cm] => ((b, d) -> sum(b[d .>= 12.7])) => :BAsp)
spd = Dict((r.site, r.key) => r.BAsp for r in eachrow(sp))
function pois(X, y, off; it=40)
  b = zeros(size(X, 2)); b[1] = log(max(mean(y), 1e-3)) - mean(off)
  for _ in 1:it
    η = X * b .+ off; μ = exp.(η); z = (η .- off) .+ (y .- μ) ./ μ
    b = (X' * (μ .* X) + 1e-6I) \ (X' * (μ .* z))
  end
  b
end
dev(X, b, y, off) = (μ = exp.(X * b .+ off); 2sum(@. ifelse(y > 0, y * log(y / μ), 0.0) - (y - μ)))
coefs = DataFrame(key=String[], rho0=Float64[], rhoS=Float64[], rhoL=Float64[])
@printf("%-8s %7s %7s %8s %9s | VAL deviance: intercept / +seed / +light / +seed+light\n", "key", "n_val", "obs_val", "pred_val", "train_y")
for k in keys_
  rows(set) = [(BAsp=get(spd, (r.site, k), 0.0), RD=r.SDI / 600, dt=r.dt, y=get(cnt, (r.site, k), 0)) for r in eachrow(st) if r.set == set]
  T = rows("train"); V = rows("val")
  yT = Float64[r.y for r in T]; yV = Float64[r.y for r in V]; oT = log.([r.dt for r in T]); oV = log.([r.dt for r in V])
  Xf(R, f) = (m = length(f(R[1])); X = Matrix{Float64}(undef, length(R), m); for (i, r) in enumerate(R); X[i, :] = f(r); end; X)   # preallocated (reduce(vcat) is quadratic)
  specs = (r -> [1.0], r -> [1.0, log1p(r.BAsp)], r -> [1.0, r.RD], r -> [1.0, log1p(r.BAsp), r.RD])
  devs = Float64[]; local b, predv
  for (j, f) in enumerate(specs)
    bb = pois(Xf(T, f), yT, oT); push!(devs, dev(Xf(V, f), bb, yV, oV))
    j == 4 && (b = bb; predv = sum(exp.(Xf(V, f) * bb .+ oV)))
  end
  push!(coefs, (k, b[1] + log(sw.site_mult), b[2], b[3]))
  @printf("%-8s %7d %7d %8.0f %9d | %s\n", k, length(V), sum(yV), predv, sum(yT), join((@sprintf("%.0f", d) for d in devs), " / "))
end
out = argval("--out", ""); isempty(out) || (mkpath(dirname(abspath(out))); CSV.write(out, coefs); println("wrote ", out))
show(stdout, coefs; allrows=true); println()
ref = argval("--compare", "")
if !isempty(ref)
  r = CSV.read(ref, DataFrame); rd = Dict(String(x.key) => x for x in eachrow(r))
  Set(keys(rd)) == Set(coefs.key) || error("compare: key sets differ")
  mx = maximum(maximum(abs.([c.rho0 - rd[c.key].rho0, c.rhoS - rd[c.key].rhoS, c.rhoL - rd[c.key].rhoL])) for c in eachrow(coefs))
  @printf("compare with %s: max |difference| %.3e\n", ref, mx)
  mx <= 1e-8 || (println("RECRUITMENT FIT MISMATCH"); exit(1))
  println("RECRUITMENT FIT MATCH")
end
