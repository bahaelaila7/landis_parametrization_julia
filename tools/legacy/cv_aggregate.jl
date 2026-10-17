# Aggregate per-fold CV front held-out metrics across folds.
#
# Each fold's run wrote (via the driver's end-of-run _dump_cv_front) <cv_dir>/fold_<k>/cv_front_metrics.csv
# (the 4 front representatives — best / extreme-W / extreme-AGB / knee — scored on that fold's HELD-OUT plots
# under the fold's LIVE train-frozen loss, i.e. the SAME objective the candidates were selected under) and
# cv_front_plots.csv (per-plot sim vs ref AGB, for TOST). This tool concatenates them with a `fold` column,
# averages the representatives' held-out losses across folds, and pools per-plot predictions for tools/cv_tost.jl.
#
#   ./julia_gdal.sh --project=. tools/cv_aggregate.jl <cv_output_dir> [n_folds]
import CSV, DataFrames, Statistics
const DF = DataFrames; const S = Statistics

CVDIR = ARGS[1]
NFOLDS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5

mparts = DF.DataFrame[]; pparts = DF.DataFrame[]; ecomap = nothing
for f in 1:NFOLDS
  fd = joinpath(CVDIR, "fold_$f")
  mf = joinpath(fd, "cv_front_metrics.csv"); pf = joinpath(fd, "cv_front_plots.csv")
  (isfile(mf) && isfile(pf)) || (@warn "fold $f: missing cv_front_*.csv — skipping (did the fold finish?)"; continue)
  m = CSV.read(mf, DF.DataFrame); m.fold .= f; push!(mparts, m)
  p = CSV.read(pf, DF.DataFrame); p.fold .= f; push!(pparts, p)
  (isnothing(ecomap) && isfile(joinpath(fd, "cv_eco_map.csv"))) && (global ecomap = CSV.read(joinpath(fd, "cv_eco_map.csv"), DF.DataFrame))
end
isempty(mparts) && error("no fold outputs under $CVDIR — run scripts/run_cv.sh first")

metrics = reduce(vcat, mparts); plots = reduce(vcat, pparts)
CSV.write(joinpath(CVDIR, "cv_metrics.csv"), metrics)
CSV.write(joinpath(CVDIR, "cv_heldout_plots.csv"), plots)
isnothing(ecomap) || CSV.write(joinpath(CVDIR, "cv_eco_map.csv"), ecomap)

summ = DF.combine(DF.groupby(metrics, :rep_kind),
  :total => S.mean => :total_mean, :total => (x -> length(x) > 1 ? S.std(x) : 0.0) => :total_sd,
  :W => S.mean => :W_mean, :AGB => S.mean => :AGB_mean, DF.nrow => :n_folds)
sort!(summ, :total_mean)
CSV.write(joinpath(CVDIR, "cv_summary.csv"), summ)

println("=== CV held-out summary — mean over $(length(unique(metrics.fold))) folds (each fold's train-frozen metric) ===")
show(summ, allrows=true, allcols=true); println()
println("\nwrote cv_metrics.csv, cv_summary.csv, cv_heldout_plots.csv, cv_eco_map.csv to $CVDIR")
println("next: ./julia_gdal.sh --project=. tools/cv_tost.jl $CVDIR")
