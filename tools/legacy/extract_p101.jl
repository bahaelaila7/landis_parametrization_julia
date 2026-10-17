# Extract the p101 union-nondominated candidates of an MO run: scan every search_state@N.jld2,
# collect (W_total, A_total, params) per archive member, build the global non-dominated envelope,
# save each unique member's params to <outdir>/p101_candidates/cand_XX.jld2 + a manifest CSV.
#   ./julia_gdal.sh --project=. scratchpad/extract_p101.jl <run_output_dir>
using Pan
import JLD2, Printf

RUN = ARGS[1]
OUT = joinpath(RUN, "p101_candidates"); mkpath(OUT)

files = sort(filter(f -> occursin(r"search_state@\d+\.jld2$", f), readdir(RUN; join=true)),
             by = f -> parse(Int, match(r"@(\d+)\.jld2", f).captures[1]))
gen(f) = parse(Int, match(r"@(\d+)\.jld2", f).captures[1])

# gather every archive member across all checkpoints: (W, A, gen, params)
recs = NamedTuple[]
for f in files
    st = JLD2.load_object(f)
    for c in collect(st.archive)
        o = Float64.(c.fx.objectives)
        W = sum(o[1:2:end]); A = sum(o[2:2:end])
        push!(recs, (W=W, A=A, g=gen(f), x=c.x))
    end
end
println("scanned $(length(files)) checkpoints, $(length(recs)) total archive members")

# union non-dominated over (W, A); dedup identical objective points (keep earliest gen)
key(r) = (round(r.W, digits=8), round(r.A, digits=8))
seen = Dict{Tuple{Float64,Float64},Any}()
for r in sort(recs, by = r -> r.g)                 # earliest gen wins on ties
    k = key(r); haskey(seen, k) || (seen[k] = r)
end
uniq = collect(values(seen))
nondom = filter(p -> !any(q -> q.W <= p.W && q.A <= p.A && key(q) != key(p), uniq), uniq)
sort!(nondom, by = r -> (r.W, -r.A))
println("p101 union non-dominated: n=$(length(nondom))")

open(joinpath(OUT, "manifest.csv"), "w") do io
    println(io, "idx,W_total,A_total,source_gen,file")
    for (i, r) in enumerate(nondom)
        fn = Printf.@sprintf("cand_%02d.jld2", i)
        JLD2.save_object(joinpath(OUT, fn), r.x)
        println(io, join([i, round(r.W, digits=6), round(r.A, digits=6), r.g, fn], ","))
        println(Printf.@sprintf("  cand_%02d  W=%.5f  A=%.4f  (gen %d)", i, r.W, r.A, r.g))
    end
end
println("wrote $(length(nondom)) candidates + manifest.csv to $OUT")
