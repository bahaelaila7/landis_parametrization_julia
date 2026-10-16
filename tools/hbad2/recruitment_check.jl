# HBAD v2 recruitment process (Biomass-Succession-inspired: probability × seed source × light), fitted on sapling ingrowth into the
# 1-in tally (microplot; one tree = 185.2 trees/ha) of undisturbed intervals. Poisson log-linear per species key (fit TRAIN, VAL deviance):
#   E[count] = dt/185.2 · exp(ρ0 + ρS ln(1 + BA_sp≥5in) + ρL RD + ρC ln(1 + n_saplings_sp))
# RD = stand SDI / 600 at v1.  Compares nested models (no seed term, no light term, + conspecific saplings).
#   ./julia_gdal.sh --project=. tools/hbad2/recruitment_check.jl
include(joinpath(@__DIR__, "common.jl"))
iv = DataFrame(Arrow.Table(joinpath(R2DIR, "intervals.arrow"))); ig = DataFrame(Arrow.Table(joinpath(R2DIR, "interval_ingrowth.arrow")))
tr = load_trees(); keys_ = sort(unique(tr.key)); kset = Set(keys_)
big = Set(k for k in keys_ if k[1] != '_')
ig.key = [s in big ? s : "_" * g for (s, g) in zip(ig.sym, ig.grp)]
ok = Set(unique(tr.iid)); ig = filter(r -> r.iid in ok && r.dia_cm < 12.7, ig)
cnt = Dict((r.iid, r.key) => r.n for r in eachrow(combine(groupby(ig, [:iid, :key]), nrow => :n)))
tr.ba = (pi / 4) .* (tr.dia_cm ./ 100) .^ 2 .* tr.w_ha; tr.sdi = tr.w_ha .* (tr.dia_cm ./ 25.4) .^ 1.605
st = combine(groupby(tr, :iid), :sdi => sum => :SDI, :ba => sum => :BA, :dt => first => :dt, :set => first => :set, :planted => first => :planted)
NATURAL = get(ENV, "HBAD2_RECR_ALL", "0") != "1"
NATURAL && (st = st[.!st.planted, :]; println("recruitment fitted on NATURAL (non-planted) intervals only: ", nrow(st)))
sp = combine(groupby(tr, [:iid, :key]), [:ba, :dia_cm] => ((b, d) -> sum(b[d .>= 12.7])) => :BAsp, :dia_cm => (d -> count(<(12.7), d)) => :nsap)
spd = Dict((r.iid, r.key) => r for r in eachrow(sp))
rows = NamedTuple[]
for r in eachrow(st), k in keys_
  s = get(spd, (r.iid, k), nothing)
  push!(rows, (iid=r.iid, key=k, set=r.set, dt=r.dt, RD=r.SDI / 600, BAsp=s === nothing ? 0.0 : s.BAsp, nsap=s === nothing ? 0 : s.nsap, y=get(cnt, (r.iid, k), 0)))
end
d = DataFrame(rows)
function pois(X, y, off; it=40)
  b = zeros(size(X, 2)); b[1] = log(max(mean(y), 1e-3)) - mean(off)
  for _ in 1:it
    η = X * b .+ off; μ = exp.(η); z = (η .- off) .+ (y .- μ) ./ μ
    b = (X' * (μ .* X) + 1e-6I) \ (X' * (μ .* z))
  end
  b
end
dev(X, b, y, off) = (μ = exp.(X * b .+ off); 2sum(@. ifelse(y > 0, y * log(y / μ), 0.0) - (y - μ)))
specs = [("intercept", r -> [1.0]), ("+seed", r -> [1.0, log1p(r.BAsp)]), ("+light", r -> [1.0, r.RD]), ("+seed+light", r -> [1.0, log1p(r.BAsp), r.RD]),
         ("+seed+light+sapl", r -> [1.0, log1p(r.BAsp), r.RD, log1p(r.nsap)])]
@printf("%-8s %6s %8s %8s | VAL deviance per model: %s\n", "key", "n_val", "obs_val", "pred_val", join(first.(specs), " / "))
coefs = NamedTuple[]
for k in keys_
  T = d[(d.key .== k) .& (d.set .== "train"), :]; V = d[(d.key .== k) .& (d.set .== "val"), :]
  offT = log.(T.dt ./ 185.2 .* 185.2); offV = log.(V.dt)            # counts per interval: offset log dt (rate in trees per interval-year)
  devs = Float64[]; predv = 0.0
  for (nm, f) in specs
    XT = reduce(vcat, (f(r)' for r in eachrow(T))); XV = reduce(vcat, (f(r)' for r in eachrow(V)))
    b = pois(XT, Float64.(T.y), offT); push!(devs, dev(XV, b, Float64.(V.y), offV))
    if nm == "+seed+light"; predv = sum(exp.(XV * b .+ offV)); push!(coefs, (key=k, rho0=b[1], rhoS=b[2], rhoL=b[3])); end
  end
  @printf("%-8s %6d %8d %8.0f | %s\n", k, nrow(V), sum(V.y), predv, join((@sprintf("%.0f", x) for x in devs), " / "))
end
c = DataFrame(coefs); CSV.write(joinpath(R2DIR, "recruitment_coefs.csv"), c); println(c)
