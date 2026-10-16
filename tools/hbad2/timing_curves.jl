# Species free-growth trajectories implied by a fitted HBAD v2 (the "curve over cohort age" of each species): integrate the kernel
# for one open-grown cohort (N = 1 tree/ha → O ≈ 0, RD ≈ 0) from the recruit state (D 2.54 cm, H 3 m, age 5) for 150 yr, and the
# increments at the observed free-growing (CCLCD 1-3) class means of E3 (O = 0.15, RD = 0.4, typical of dominants) vs observed.
#   ./julia_gdal.sh --project=. tools/hbad2/timing_curves.jl <tag>   → results/hbad2/fit_<tag>/timing_{curves,vs_obs}.csv
include(joinpath(@__DIR__, "common.jl"))
tag = ARGS[1]; OUTD = joinpath(R2DIR, "fit_" * tag)
tr = load_trees(); keys_ = sort(unique(tr.key)); nsp = length(keys_)
lon = CSV.read("/workspace/species_longevity_ref.csv", DataFrame); lond = Dict(String(r.species_symbol) => Float64(r.longevity_yr) for r in eachrow(lon))
longev = [median([get(lond, s, 300.0) for s in unique(tr.sym[tr.key .== k])]) for k in keys_]
θ = read_theta(joinpath(OUTD, "theta.csv"), nsp); P, G = decode(θ, nsp, longev)
rows = NamedTuple[]
for k in 1:nsp
  S = Sites([1, 2], [k], [1.0], [2.54], [3.0], [5.0], [0], [0.0], [false])
  for a in 5:150
    D0, H0 = S.D[1], S.H[1]; HBAD2.step_site!(S, P, G, 1, 1, 1.0, false)
    push!(rows, (key=keys_[k], age=a + 1, D=S.D[1], H=S.H[1], dD=S.D[1] - D0, dH=S.H[1] - H0))
  end
end
CSV.write(joinpath(OUTD, "timing_curves.csv"), DataFrame(rows))
# increments at observed free-growing states
e3 = CSV.read(joinpath(R2DIR, "timing_free_growing.csv"), DataFrame); kix = Dict(k => i for (i, k) in enumerate(keys_))
smx(p) = p.sdimax
out = NamedTuple[]
for r in eachrow(e3)
  haskey(kix, r.key) || continue
  p = P[kix[r.key]]; o = 0.15; RD = 0.4
  dD = HBAD2.potD(G, r.D, p, 1.0) * exp(-p.gOD * o) * exp(p.gRD * RD); dH = HBAD2.potH(G, r.H, p, 1.0) * exp(-p.gOH * o + p.gRH * RD)
  push!(out, (key=r.key, acl=r.acl, n=r.n, D=r.D, H=r.H, dD_obs=r.dD, dD_mod=dD, dH_obs=r.dH, dH_mod=dH))
end
o = DataFrame(out); CSV.write(joinpath(OUTD, "timing_vs_obs.csv"), o)
@printf("timing check (%d species×age classes): corr dD %.3f, corr dH %.3f, corr dH:dD ratio %.3f\n", nrow(o), cor(o.dD_obs, o.dD_mod), cor(o.dH_obs, o.dH_mod),
        cor(o.dH_obs ./ o.dD_obs, o.dH_mod ./ o.dD_mod))
for g in groupby(o, :key)
  @printf("  %-6s dH:dD obs %s | model %s\n", g.key[1], join((@sprintf("%.1f", x) for x in g.dH_obs ./ g.dD_obs), " "), join((@sprintf("%.1f", x) for x in g.dH_mod ./ g.dD_mod), " "))
end
