# Per-stratum TOST equivalence test of held-out simulated AGB vs reference AGB (Sim A cross-validation).
#
# Reads cv_heldout_plots.csv + cv_eco_map.csv (written by tools/cv_aggregate.jl). Each held-out plot-measurement
# is predicted once (by the fold that held it out); pooling across folds covers every plot once. For each
# stratum (eco = epa_l3|land_use):
#   * effect size (a priori, ABSOLUTE g/m²): 0.05 and 0.10 × mean reference plot AGB over plots with ref AGB
#     ≤ the stratum's p95 (trims the heavy upper tail so the margin isn't inflated by outliers).
#   * paired TOST on d = sim_agb − ref_agb: two one-sided t-tests vs ±margin; equivalent iff the 90% CI of
#     mean(d) ⊂ [−margin, +margin]  (⇔ TOST p = max(p_lower, p_upper) < α, α=0.05).
# Runs for the `best` and `knee` (median-front) representatives.
#
#   ./julia_gdal.sh --project=. tools/cv_tost.jl <cv_output_dir>
import CSV, DataFrames, Statistics, Distributions
const DF = DataFrames; const S = Statistics; const Dist = Distributions

CVDIR = ARGS[1]
ALPHA = 0.05
REPS = ["best", "knee"]
MARGINS = [(name="5pct", frac=0.05), (name="10pct", frac=0.10)]

plots = CSV.read(joinpath(CVDIR, "cv_heldout_plots.csv"), DF.DataFrame)
ecomap = Dict(Int(r.eco_id) => String(r.eco_label) for r in CSV.File(joinpath(CVDIR, "cv_eco_map.csv")))

# per-stratum reference mean over plots with ref ≤ p95 (dedupe to one row per held-out plot-measurement)
refu = unique(DF.select(plots, [:fold, :eco_id, :plot_id, :sim_year, :ref_agb]))
effect = Dict{Int,Float64}()
for gd in DF.groupby(refu, :eco_id)
  r = Float64.(gd.ref_agb); p95 = S.quantile(r, 0.95)
  effect[Int(gd.eco_id[1])] = S.mean(r[r .<= p95])
end

out = DF.DataFrame(stratum=String[], representative=String[], margin=String[], effect_gm2=Float64[],
  n=Int[], mean_diff=Float64[], ci90_lo=Float64[], ci90_hi=Float64[], p_tost=Float64[], equivalent=Bool[])

for rep in REPS
  sub = DF.subset(plots, :rep_kind => DF.ByRow(==(rep)))
  for gd in DF.groupby(sub, :eco_id)
    e = Int(gd.eco_id[1]); label = get(ecomap, e, "eco$e")
    d = Float64.(gd.sim_agb) .- Float64.(gd.ref_agb)
    n = length(d); n < 2 && continue
    md = S.mean(d); se = S.std(d) / sqrt(n); dof = n - 1
    td = Dist.TDist(dof); tcrit = Dist.quantile(td, 1 - ALPHA)     # 90% CI half-width factor
    ci_lo = md - tcrit * se; ci_hi = md + tcrit * se
    refmean = get(effect, e, NaN)
    for m in MARGINS
      Δ = m.frac * refmean
      p1 = 1 - Dist.cdf(td, (md + Δ) / se)   # H0: μ ≤ −Δ   (lower bound)
      p2 = Dist.cdf(td, (md - Δ) / se)       # H0: μ ≥ +Δ   (upper bound)
      p_tost = max(p1, p2)
      equiv = (ci_lo > -Δ) && (ci_hi < Δ)
      push!(out, (label, rep, m.name, round(Δ, digits=1), n, round(md, digits=1),
        round(ci_lo, digits=1), round(ci_hi, digits=1), round(p_tost, digits=4), equiv))
    end
  end
end

sort!(out, [:representative, :stratum, :margin])
CSV.write(joinpath(CVDIR, "tost_results.csv"), out)
println("=== per-stratum reference mean plot AGB (≤p95), g/m² ===")
for e in sort(collect(keys(effect))); println("  $(get(ecomap,e,"eco$e")): ", round(effect[e], digits=0)); end
println("\n=== TOST equivalence (sim vs ref AGB), 90% CI, α=$ALPHA ===")
show(out, allrows=true, allcols=true); println()
println("\nwrote tost_results.csv to $CVDIR")
