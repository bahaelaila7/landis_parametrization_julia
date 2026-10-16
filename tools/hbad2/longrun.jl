# HBAD v2 emergent behaviour: free runs with recruitment from bare ground (one eco, all species, seed source = own BA + seed rain
# baseline as fitted), mean of replicate-free deterministic expected-value runs. Recruitment (tools/hbad2/recruitment_check.jl):
# R_sp [trees/ha/yr] = 185.2 · exp(ρ0 + ρS ln(1 + BA_sp, D ≥ 12.7 cm) + ρL RD); recruits enter at D = 2.54 cm, H = 3 m, age 5,
# merged into the species' cohort of the current 5-yr recruitment class (TPA of a cohort still only decreases after its class closes).
#   ./julia_gdal.sh --project=. tools/hbad2/longrun.jl <tag> [years=150] [species csv subset]
include(joinpath(@__DIR__, "common.jl"))
tag = ARGS[1]; YEARS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 150
tr = load_trees(); keys_ = sort(unique(tr.key)); nsp = length(keys_)
lon = CSV.read("/workspace/species_longevity_ref.csv", DataFrame); lond = Dict(String(r.species_symbol) => Float64(r.longevity_yr) for r in eachrow(lon))
longev = [median([get(lond, s, 300.0) for s in unique(tr.sym[tr.key .== k])]) for k in keys_]
θ = read_theta(joinpath(R2DIR, "fit_" * tag, "theta.csv"), nsp); P, G = decode(θ, nsp, longev)
rc = CSV.read(joinpath(R2DIR, "recruitment_coefs.csv"), DataFrame); rcd = Dict(String(r.key) => r for r in eachrow(rc))
subset = length(ARGS) >= 3 ? split(ARGS[3], ",") : keys_
act = [k in subset for k in keys_]
K = 400
S = Sites([1, K + 1], fill(1, K), zeros(K), fill(2.54, K), fill(3.0, K), fill(5.0, K), [0], [0.0], [false])
live = 0; cls_of = Dict{Tuple{Int,Int},Int}()
rows = NamedTuple[]
for y in 1:YEARS
  global live
  if live > 0
    S.off[2] = live + 1                                        # kernel acts on the live slots only
    HBAD2.step_site!(S, P, G, 1, live, 1.0, false)
  end
  # recruitment from the post-step state
  sdi_ = sum(S.N[i] * (S.D[i] / 25.4)^1.605 for i in 1:live; init=0.0)
  smax = live == 0 ? 600.0 : sum(S.N[i] * (S.D[i] / 25.4)^1.605 * P[S.sp[i]].sdimax for i in 1:live) / max(sdi_, 1e-9)
  RD = sdi_ / smax
  for k in 1:nsp
    act[k] || continue
    r = rcd[keys_[k]]
    ba = sum(pi / 4 * (S.D[i] / 100)^2 * S.N[i] for i in 1:live if S.sp[i] == k && S.D[i] >= 12.7; init=0.0)
    R = 185.2 * exp(r.rho0 + r.rhoS * log1p(ba) + r.rhoL * RD)
    c = (k, y ÷ 5); j = get(cls_of, c, 0)
    if j == 0
      live < K || continue
      live += 1; j = live; cls_of[c] = j
      S.sp[j] = k; S.N[j] = R; S.D[j] = 2.54; S.H[j] = 3.0; S.A[j] = 5.0
    else                                                       # merge recruits into the open class cohort (quadratic-mean D, mean H)
      n0 = S.N[j]; S.D[j] = sqrt((n0 * S.D[j]^2 + R * 2.54^2) / (n0 + R)); S.H[j] = (n0 * S.H[j] + R * 3.0) / (n0 + R); S.N[j] += R
    end
  end
  if y % 10 == 0 || y in (1, 5)
    N = sum(S.N[1:live]); BA = sum(pi / 4 * (S.D[i] / 100)^2 * S.N[i] for i in 1:live)
    N5 = sum(S.N[i] for i in 1:live if S.D[i] >= 12.7; init=0.0); BA5 = sum(pi / 4 * (S.D[i] / 100)^2 * S.N[i] for i in 1:live if S.D[i] >= 12.7; init=0.0)
    QMD5 = N5 > 0 ? sqrt(BA5 / N5 * 4 / pi) * 100 : NaN
    top = sortperm(S.H[1:live]; rev=true); Htop = S.H[top[1]]
    shares = [sum(pi / 4 * (S.D[i] / 100)^2 * S.N[i] for i in 1:live if S.sp[i] == k; init=0.0) / BA for k in 1:nsp]
    push!(rows, (yr=y, N_all=N, N_5in=N5, BA=BA, QMD_5in=QMD5, Htop=Htop, RD=RD, cohorts=count(>(1.0), S.N[1:live]),
                 top3=join(["$(keys_[k]):$(round(Int, 100shares[k]))" for k in sortperm(shares; rev=true)[1:3]], " ")))
  end
end
df = DataFrame(rows); show(stdout, df; allrows=true, summary=false, eltypes=false); println()
CSV.write(joinpath(R2DIR, "fit_" * tag, "longrun_$(YEARS).csv"), df)
