# UMAP of the Sobol sample's parameter sets, coloured by aggregate / ΣW / ΣAGB (yellow = lower = better).
# Reads sumW/sumAGB directly from sobol_results (stored by parametrize_sobol) — no re-evaluation.
# Same param-feature embedding for all three → identical layout, only the colour changes.
#   Run:  ./julia_gdal.sh --project=. test/sobol_umap.jl <losses.duckdb> [out_prefix]
using Pan
import DuckDB, DataFrames, Serialization, UMAP, Random, Statistics, CairoMakie
const MK = CairoMakie; const DF = DataFrames

dbpath = ARGS[1]
prefix = length(ARGS) >= 2 ? ARGS[2] : joinpath(dirname(dbpath), "sobol_umap")

function feat(p)
  v = Float64[]
  for gsp in eachindex(p.SPECIES_LIST)
    push!(v, Float64(p.D[gsp]), Float64(p.LONGEVITY[gsp]), Float64(length(p.MATURITY) >= gsp ? p.MATURITY[gsp] : 0), Float64(p.SHADE_TOL[gsp]))
  end
  for eco_id in eachindex(p.ECO_LIST), sp_local in eachindex(p.ECO_SPECIES_IDS[eco_id])
    gsp = Int(p.ECO_SPECIES_IDS[eco_id][sp_local])
    push!(v, Float64(p.S[gsp]), Float64(p.ANPP_MAX_SPP[eco_id][sp_local]), Float64(p.B_MAX_SPP[eco_id][sp_local]),
      Float64(p.PROB_MORT_SPP[eco_id][sp_local]), Float64(length(p.PROB_ESTAB_SPP) >= eco_id ? p.PROB_ESTAB_SPP[eco_id][sp_local] : 0))
  end
  for eco_id in eachindex(p.ECO_LIST); push!(v, Float64(p.MIN_REL_BIOMASS[eco_id][1])); end
  return v
end

con = DuckDB.connect(DuckDB.DB(dbpath))
cols = Set(string(r.name) for r in (DuckDB.execute(con, "PRAGMA table_info('sobol_results')") |> DF.DataFrame |> eachrow))
hasWA = ("sumW" in cols) && ("sumAGB" in cols)
sel = hasWA ? "mean_loss, sumW, sumAGB, params_blob" : "mean_loss, params_blob"
res = DuckDB.execute(con, "SELECT $sel FROM sobol_results ORDER BY mean_loss ASC") |> DF.DataFrame
println("Sobol candidates: $(DF.nrow(res))  (W/AGB columns: $hasWA)")
feats = Vector{Float64}[]; agg = Float64[]; W = Float64[]; A = Float64[]
for r in eachrow(res)
  push!(feats, feat(Serialization.deserialize(IOBuffer(r.params_blob)))); push!(agg, Float64(r.mean_loss))
  hasWA && (push!(W, Float64(r.sumW)); push!(A, Float64(r.sumAGB)))
end

X = reduce(hcat, feats); mu = Statistics.mean(X; dims=2); sd = Statistics.std(X; dims=2); sd[sd.==0] .= 1
Z = (X .- mu) ./ sd
Random.seed!(7)
emb = UMAP.fit(Z, 2; n_neighbors=max(2, min(15, size(Z, 2) - 1)), min_dist=0.4).embedding
function plot_umap(cval, label, fname)
  fig = MK.Figure(size=(860, 680))
  MK.Label(fig[0, 1:2], "Sobol sample ($(length(cval))) — UMAP — colour = $label (yellow = lower = better)"; fontsize=13, font=:bold)
  ax = MK.Axis(fig[1, 1]; xlabel="UMAP-1", ylabel="UMAP-2")
  ord = sortperm(cval; rev=true)   # draw worst (high) first → best (low; incl. Pareto front-1) LAST = on top
  MK.scatter!(ax, emb[1, ord], emb[2, ord]; color=cval[ord], colormap=MK.cgrad(:viridis; rev=true), markersize=11, strokecolor=:black, strokewidth=0.4)
  MK.Colorbar(fig[1, 2]; colormap=MK.cgrad(:viridis; rev=true), colorrange=(minimum(cval), maximum(cval)), label="$label (yellow = lower = better)")
  MK.save(fname, fig); println("wrote $fname")
end
plot_umap(agg, "aggregate error", "$(prefix).png")
if hasWA
  plot_umap(W, "ΣW (age-distribution loss)", "$(prefix)_W.png")
  plot_umap(A, "ΣAGB (biomass loss)", "$(prefix)_AGB.png")
  # balanced aggregate: min/max-rescale W and AGB to [0,1] each, THEN sum (equal weight, scale-free)
  rescale(x) = (lo = minimum(x); hi = maximum(x); hi > lo ? (x .- lo) ./ (hi - lo) : zero(x))
  bal = rescale(W) .+ rescale(A)
  plot_umap(bal, "balanced: rescaled ΣW + rescaled ΣAGB (0–2)", "$(prefix)_balanced.png")
  # dominance: non-dominated sort over (ΣW, ΣAGB) → Pareto front rank (1 = non-dominated = best)
  let n = length(W)
    dom(i, j) = (W[i] <= W[j] && A[i] <= A[j]) && (W[i] < W[j] || A[i] < A[j])
    front = fill(0, n); remaining = Set(1:n); fr = 0
    while !isempty(remaining)
      fr += 1
      nd = [i for i in remaining if !any(j -> dom(j, i), remaining)]
      for i in nd; front[i] = fr; delete!(remaining, i); end
    end
    plot_umap(Float64.(front), "Pareto front rank over (ΣW,ΣAGB)  (1 = non-dominated)", "$(prefix)_dominance.png")
    println("dominance: $(maximum(front)) fronts, front-1 (Pareto) = $(count(==(1),front)) candidates")
  end
  println("ranges: agg $(round(minimum(agg),sigdigits=3))–$(round(maximum(agg),sigdigits=3)) | W $(round(minimum(W),sigdigits=3))–$(round(maximum(W),sigdigits=3)) | AGB $(round(minimum(A),sigdigits=3))–$(round(maximum(A),sigdigits=3))")
end
