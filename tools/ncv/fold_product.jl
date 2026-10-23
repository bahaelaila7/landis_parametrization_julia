# Per-role copy of the HBAD2 / HBAD3P interval product (docs/nested_cv_gafl.md §1): intervals.arrow with its `set` column replaced from a split file
# (plots the file does not list get `missing`, so no stage uses them), interval_trees.arrow and interval_ingrowth.arrow symlinked. The prototype stages
# (tools/hbad2, tools/hbad3 of branch hbad3-ncv) read it through HBAD2_DATA_DIR / HBAD2_R2DIR.
#   ./julia_gdal.sh --project=. tools/ncv/fold_product.jl <base product dir> <split.csv> <out dir>
using Arrow, DataFrames, CSV
base, splitf, out = ARGS
iv = DataFrame(Arrow.Table(joinpath(base, "intervals.arrow")); copycols=true)
sp = CSV.read(splitf, DataFrame)
setof = Dict((Int(r.statecd), Int(r.unitcd), Int(r.countycd), Int(r.plot)) => String(r.set) for r in eachrow(sp))
length(setof) == nrow(sp) || error("$(splitf): a plot is listed more than once")
iv.set = Union{Missing,String}[get(setof, (r.st, r.u, r.c, r.p), missing) for r in eachrow(iv)]
mkpath(out)
Arrow.write(joinpath(out, "intervals.arrow"), iv)
for f in ("interval_trees.arrow", "interval_ingrowth.arrow")
  dst = joinpath(out, f); islink(dst) && rm(dst)
  symlink(abspath(joinpath(base, f)), dst)
end
cnt = Dict(s => count(isequal(s), iv.set) for s in ("train", "val", "test"))
println("$(out): intervals $(nrow(iv)); train $(cnt["train"]) val $(cnt["val"]) test $(cnt["test"]) unassigned $(count(ismissing, iv.set)) (plots in split $(nrow(sp)))")
