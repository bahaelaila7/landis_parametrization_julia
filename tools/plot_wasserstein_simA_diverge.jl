# Illustration #2: Sim A injects the observed cohorts at t0 (sim = obs, W1 = 0), then the sim DRIFTS —
# older cohorts (longest under model dynamics) diverge most while the youngest stay pinned, so the two
# age-CDFs coincide on the young end and split toward the old end. Dots = cohorts (obs from a real plot;
# sim = drifted), dashed step lines = the cumulative (cumsum/CDF), shaded area between = W1.
#   Run: ./julia_gdal.sh --project=. tools/plot_wasserstein_simA_diverge.jl
import CairoMakie, DuckDB, DataFrames
const MK = CairoMakie; const DF = DataFrames

con = DuckDB.connect(DuckDB.DB("/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb"))
co = DF.DataFrame(DuckDB.execute(con, "SELECT age_calc AS age, sum(agb) AS agb FROM curated_cohorts_landis " *
  "WHERE statecd=13 AND unitcd=2 AND countycd=7 AND plot=27 GROUP BY age_calc ORDER BY age_calc"))
ages = Float64.(co.age); w = Float64.(co.agb); w ./= sum(w); cum = cumsum(w)
amin = minimum(ages)
println("cohorts: ages=", Int.(ages), "  weights=", round.(w; digits=3))

grid = collect(0.0:0.5:130.0); dg = grid[2] - grid[1]
stepF(xa, xc, a) = (i = searchsortedlast(xa, a); i == 0 ? 0.0 : xc[i])
Fobs = [stepF(ages, cum, a) for a in grid]
# drift pinned at the youngest cohort (Sim A start), growing with age: sim_age = age + df·(age − amin)
simset(df) = (let sa = ages .+ df .* (ages .- amin), o = sortperm(sa); (sa[o], cumsum(w[o])); end)

panels = [(0.0, "t₀ — Sim A injected: aligned"), (0.45, "t₁ — drifted (older cohorts diverge)")]
fig = MK.Figure(size=(1200, 510))
MK.Label(fig[0, 1:2], "Sim A starts aligned with obs at t₀, then drifts → W1 grows from the OLD (right) end (plot 13-2-7-27)"; fontsize=13, font=:bold)
for (i, (df, lbl)) in enumerate(panels)
  sa, scum = simset(df)
  Fsim = [stepF(sa, scum, a) for a in grid]
  w1 = sum(abs.(Fsim .- Fobs)) * dg
  ax = MK.Axis(fig[1, i]; title="$(lbl):  W1 = $(round(w1; digits=1)) yr", xlabel="cohort age (yr)",
    ylabel="cumulative AGB fraction (CDF)", limits=(0, 130, -0.03, 1.05))
  MK.band!(ax, grid, min.(Fobs, Fsim), max.(Fobs, Fsim); color=(:darkorange, 0.30))
  MK.stairs!(ax, [0.0; ages; 130.0], [0.0; cum; 1.0]; color=:navy, linestyle=:dash, linewidth=2.0, step=:post)
  MK.scatter!(ax, ages, cum; color=:navy, markersize=11)
  MK.stairs!(ax, [0.0; sa; 130.0], [0.0; scum; 1.0]; color=:firebrick, linestyle=:dash, linewidth=2.0, step=:post)
  MK.scatter!(ax, sa, scum; color=:firebrick, markersize=11, marker=:diamond)
  MK.Legend(fig[2, i], [MK.MarkerElement(color=:navy, marker=:circle), MK.MarkerElement(color=:firebrick, marker=:diamond), MK.PolyElement(color=(:darkorange, 0.45))],
    ["observed cohorts (+CDF)", "simulated cohorts (+CDF)", "W1 (area between CDFs)"]; orientation=:horizontal, framevisible=false)
end
out = "tools/wasserstein_simA_diverge.png"
MK.save(out, fig); println("wrote $out")
