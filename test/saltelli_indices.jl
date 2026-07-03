# Saltelli sensitivity indices from a saltelli_results table (written by parametrize_sobol w/ saltelli=true).
# First-order  Sᵢ  (Saltelli et al. 2010):  Sᵢ  = (1/N) Σ f_B·(f_ABᵢ − f_A) / V
# Total-effect Sₜᵢ (Jansen 1999):           Sₜᵢ = (1/2N) Σ (f_A − f_ABᵢ)²   / V
# per param-TYPE group, for each objective (aggregate mean_loss, A_W=sumW, A_AGB=sumAGB), w/ bootstrap CIs.
#   Run: ./julia_gdal.sh --project=. test/saltelli_indices.jl runs/fl5_l4cover_saltelli_K10_512_outputs/losses.duckdb
using DataFrames, DuckDB, Statistics, Random, Printf, CairoMakie
const MK = CairoMakie
Random.seed!(1)
dbpath = length(ARGS) >= 1 ? ARGS[1] : "runs/fl5_l4cover_saltelli_K10_512_outputs/losses.duckdb"
con = DBInterface.connect(DuckDB.DB(dbpath))
df = DataFrame(DBInterface.execute(con, "SELECT idx, tag, mean_loss, sumW, sumAGB FROM saltelli_results ORDER BY idx"))
println("rows: ", nrow(df), "  tags: ", sort(unique(df.tag)))

# align each tag's rows by idx → the n-th row of every block is base-sample n
col(tag, c) = (sub = sort(df[df.tag .== tag, :], :idx); Float64.(sub[!, c]))
groups = sort([t for t in unique(df.tag) if startswith(t, "AB:")])
gname(t) = replace(t, "AB:" => "")

function indices(fA, fB, fAB; nboot=1000)
  N = length(fA); V = var(vcat(fA, fB))
  Si  = mean(fB .* (fAB .- fA)) / V
  STi = 0.5 * mean((fA .- fAB) .^ 2) / V
  Sib = Float64[]; STib = Float64[]
  for _ in 1:nboot
    j = rand(1:N, N); a = fA[j]; b = fB[j]; ab = fAB[j]; Vb = var(vcat(a, b))
    Vb > 0 || continue
    push!(Sib, mean(b .* (ab .- a)) / Vb); push!(STib, 0.5 * mean((a .- ab) .^ 2) / Vb)
  end
  (Si=Si, STi=STi, Si_lo=quantile(Sib, 0.025), Si_hi=quantile(Sib, 0.975),
   STi_lo=quantile(STib, 0.025), STi_hi=quantile(STib, 0.975))
end

objectives = [("aggregate", :mean_loss), ("A_W", :sumW), ("A_AGB", :sumAGB)]
fig = MK.Figure(size=(1150, 360 * length(objectives)))
for (oi, (oname, ocol)) in enumerate(objectives)
  fA = col("A", ocol); fB = col("B", ocol)
  finite = isfinite.(fA) .& isfinite.(fB)
  println("\n=== objective: $oname  (V=$(round(var(vcat(fA,fB)),sigdigits=3)), N=$(sum(finite))) ===")
  @printf("  %-12s %10s %10s %22s %22s\n", "param-type", "S_i", "S_Ti", "S_i 95% CI", "S_Ti 95% CI")
  names = String[]; Sis = Float64[]; STis = Float64[]; Sil = Float64[]; Sih = Float64[]; STil = Float64[]; STih = Float64[]
  for g in groups
    fAB = col(g, ocol); ok = finite .& isfinite.(fAB)
    r = indices(fA[ok], fB[ok], fAB[ok])
    @printf("  %-12s %10.4f %10.4f   [%6.3f,%6.3f]   [%6.3f,%6.3f]\n", gname(g), r.Si, r.STi, r.Si_lo, r.Si_hi, r.STi_lo, r.STi_hi)
    push!(names, gname(g)); push!(Sis, r.Si); push!(STis, r.STi)
    push!(Sil, r.Si_lo); push!(Sih, r.Si_hi); push!(STil, r.STi_lo); push!(STih, r.STi_hi)
  end
  ord = sortperm(STis, rev=true)
  ax = MK.Axis(fig[oi, 1]; xticks=(1:length(names), names[ord]), xticklabelrotation=π/4,
    ylabel="sensitivity index", title="Saltelli — $oname (Sᵢ first-order vs Sₜᵢ total-effect)")
  xs = 1:length(names)
  MK.barplot!(ax, xs .- 0.18, STis[ord]; width=0.34, color=(:firebrick, 0.85), label="Sₜᵢ (total)")
  MK.barplot!(ax, xs .+ 0.18, Sis[ord];  width=0.34, color=(:steelblue, 0.85), label="Sᵢ (first-order)")
  MK.errorbars!(ax, xs .- 0.18, STis[ord], STis[ord] .- STil[ord], STih[ord] .- STis[ord]; whiskerwidth=6, color=:black)
  MK.errorbars!(ax, xs .+ 0.18, Sis[ord], Sis[ord] .- Sil[ord], Sih[ord] .- Sis[ord]; whiskerwidth=6, color=:black)
  MK.hlines!(ax, [0.0]; color=:gray, linestyle=:dash); oi == 1 && MK.axislegend(ax; position=:rt)
end
out = joinpath(dirname(dbpath), "saltelli_indices.png")
MK.save(out, fig; px_per_unit=2)
println("\nwrote ", out)
println("read: Sₜᵢ≈Sᵢ → purely additive effect; Sₜᵢ≫Sᵢ → that knob acts mainly through INTERACTIONS; Sₜᵢ≈0 → inert.")
