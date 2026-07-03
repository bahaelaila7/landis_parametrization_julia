# 4-objective Pareto dominance over a DUAL Sobol sample (A_W, A_AGB, B_W, B_AGB), read from objs_blob.
# Fast non-dominated sort → front rank per candidate; scatter-matrix of the 4 objectives colored by
# front, Pareto front starred. Minimize all objectives.
#   Run: ./julia_gdal.sh --project=. test/sobol_dominance4.jl <losses.duckdb>
using DuckDB, DataFrames, Serialization, CairoMakie, Statistics
const MK = CairoMakie
dbpath = ARGS[1]
con = DBInterface.connect(DuckDB.DB(dbpath))
df = DataFrame(DBInterface.execute(con, "SELECT sobol_idx, mean_loss, objs_blob FROM sobol_results WHERE objs_blob IS NOT NULL ORDER BY sobol_idx"))
O = [Serialization.deserialize(IOBuffer(Vector{UInt8}(r.objs_blob))) for r in eachrow(df)]
M = reduce(vcat, [reshape(Float64.(o), 1, :) for o in O])
N, k = size(M)
objnames = k == 4 ? ["A_W", "A_AGB", "B_W", "B_AGB"] : ["A_W", "A_AGB"]
println("N=$N  objectives=$k  ", objnames)

dominates(a, b) = all(a .<= b) && any(a .< b)
function nd_fronts(M)
  N = size(M, 1)
  S = [Int[] for _ in 1:N]; n = zeros(Int, N); rank = zeros(Int, N); F = [Int[]]
  for p in 1:N
    for q in 1:N
      p == q && continue
      if dominates(@view(M[p, :]), @view(M[q, :])); push!(S[p], q)
      elseif dominates(@view(M[q, :]), @view(M[p, :])); n[p] += 1; end
    end
    n[p] == 0 && (rank[p] = 1; push!(F[1], p))
  end
  i = 1
  while !isempty(F[i])
    Q = Int[]
    for p in F[i], q in S[p]; n[q] -= 1; n[q] == 0 && (rank[q] = i + 1; push!(Q, q)); end
    push!(F, Q); i += 1
  end
  rank, F[1:end-1]
end
rank, F = nd_fronts(M)
println("fronts: ", length(F), "   Pareto (front-1) size: ", length(F[1]), " / $N")
println("Pareto sobol_idx: ", sort(df.sobol_idx[F[1]]))
for (oi, nm) in enumerate(objnames)
  println("  $nm range: ", round(minimum(M[:, oi]), sigdigits=4), " – ", round(maximum(M[:, oi]), sigdigits=4))
end

fig = MK.Figure(size = (240 * k + 60, 240 * k + 60))
cmap = MK.cgrad(:viridis, rev = true)
for i in 1:k, j in 1:k
  ax = MK.Axis(fig[i, j]; xlabel = (i == k ? objnames[j] : ""), ylabel = (j == 1 ? objnames[i] : ""),
    xlabelsize = 11, ylabelsize = 11)
  if i == j
    MK.hist!(ax, M[:, i]; bins = 20, color = (:gray, 0.7))
  else
    MK.scatter!(ax, M[:, j], M[:, i]; color = rank, colormap = cmap, markersize = 5)
    MK.scatter!(ax, M[F[1], j], M[F[1], i]; color = :red, markersize = 11, marker = :star5)
  end
end
MK.Label(fig[0, 1:k], "Dual Sobol (N=$N) — $k-objective dominance  (red ★ = Pareto front, n=$(length(F[1])); $(length(F)) fronts)";
  fontsize = 15)
out = joinpath(dirname(dbpath), "sobol_dominance4.png")
MK.save(out, fig; px_per_unit = 2)
println("wrote ", out)
