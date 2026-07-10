# Reshape cv_val_cache.csv (flat map: train-objectives -> cached val-objectives) into cv_val_fronts.csv
# (gen, A_W, A_AGB) that tools/sweep_fold_percentiles_val.jl expects: for every search_state@N.jld2, take
# each archive member's TRAIN objectives, look up its cached VAL score, and emit one (gen=N, val_W, val_A) row.
#   ./julia_gdal.sh --project=. scratchpad/build_val_fronts.jl <run_output_dir>
using Pan
import JLD2, CSV, DataFrames
const DF = DataFrames
RUN = ARGS[1]

cache = CSV.read(joinpath(RUN, "cv_val_cache.csv"), DF.DataFrame)
key(w, a) = (round(Float64(w), digits=6), round(Float64(a), digits=6))
lut = Dict{Tuple{Float64,Float64},Tuple{Float64,Float64}}()
for r in eachrow(cache); lut[key(r.A_W_train, r.A_AGB_train)] = (Float64(r.A_W), Float64(r.A_AGB)); end
println("cache: $(DF.nrow(cache)) val-scored members")

files = sort(filter(f -> occursin(r"search_state@\d+\.jld2$", f), readdir(RUN; join=true)),
             by = f -> parse(Int, match(r"@(\d+)\.jld2", f).captures[1]))
gen(f) = parse(Int, match(r"@(\d+)\.jld2", f).captures[1])

out = DF.DataFrame(gen=Int[], A_W=Float64[], A_AGB=Float64[])
missref = Ref(0)                         # Ref (mutated, not reassigned) sidesteps top-level for-loop global scoping
for f in files
    st = JLD2.load_object(f)
    for c in collect(st.archive)
        o = Float64.(c.fx.objectives)
        wt = sum(o[1:2:end]); at = sum(o[2:2:end])
        v = get(lut, key(wt, at), nothing)
        v === nothing ? (missref[] += 1) : push!(out, (gen(f), v[1], v[2]))
    end
end
CSV.write(joinpath(RUN, "cv_val_fronts.csv"), out)
println("cv_val_fronts.csv: $(DF.nrow(out)) rows / $(length(unique(out.gen))) checkpoints; $(missref[]) missing")
