# Threshold crossing of TRACKED trees, model (Sim A cohorts incl. vigour sub-cohorts, lognormal within-sub-cohort operator σ_obs)
# vs data: among survivors that start below a threshold T (5 cm, 12.7 cm, 25 cm), the fraction above T at the next visit.
#   ./julia_gdal.sh --project=. --threads=6 tools/hbad2/crossing_check.jl <fit tag> [set]
include(joinpath(@__DIR__, "common.jl"))
using Distributions: Normal, cdf
tag = ARGS[1]; SET = length(ARGS) >= 2 ? ARGS[2] : "val"; FD = joinpath(R2DIR, "fit_" * tag); check_meta(FD)
σo = parse(Float64, get(ENV, "HBAD2_OBS_SIGMA", "0.11"))
tr = load_trees(); keys_ = sort(unique(tr.key)); nsp = length(keys_)
lon = CSV.read("/workspace/species_longevity_ref.csv", DataFrame); lond = Dict(String(r.species_symbol) => Float64(r.longevity_yr) for r in eachrow(lon))
longev = [median([get(lond, s, 300.0) for s in unique(tr.sym[tr.key .== k])]) for k in keys_]
P, G = decode(read_theta(joinpath(FD, "theta.csv"), nsp), nsp, longev)
C = build_cohorts(tr, SET, keys_); simulate!(C.S, P, G)
Fge(D, T) = D <= 0 ? 0.0 : 1 - cdf(Normal(), (log(T) - (log(D) - σo^2)) / σo)
d = tr[tr.set .== SET, :]; al = d.out .== "alive"
for (lo, T) in ((2.54, 5.0), (5.0, 12.7), (12.7, 25.0))
  # data: survivors starting in [lo, T) → fraction with D2 ≥ T (weights = expansion, per tree)
  m = al .& (d.dia_cm .>= lo) .& (d.dia_cm .< T)
  fd = sum(d.w_ha[m] .* (d.dia2_cm[m] .>= T)) / sum(d.w_ha[m])
  # model: survivors (N at the end) of sub-cohorts that START in [lo, T) (by their initial QMD), fraction above T at the end
  num = 0.0; den = 0.0
  for i in eachindex(C.S.N)
    (C.S.D0[i] >= lo && C.S.D0[i] < T) || continue
    num += C.S.N[i] * Fge(C.S.D[i], T); den += C.S.N[i]
  end
  @printf("%s %s survivors starting %.2f–%.1f cm: fraction ≥ %.1f cm at the next visit — data %.3f  model %.3f  (ratio %.2f)\n", tag, SET, lo, T, T, fd, num / den, (num / den) / fd)
end
