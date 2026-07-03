# Illustration: Wasserstein-1 on a REAL plot's cohort age distribution. Dots = cohorts (obs from data; sim
# = the same cohorts shifted, standing in for a model mismatch). Dashed step lines = the AGB-weighted
# cumulative age distribution (cumsum/CDF) for obs and sim. Shaded area between the two = W1 (earth-mover).
# Two panels: a close sim (small area) vs a more diverging sim (larger area).
#   Run: ./julia_gdal.sh --project=. tools/plot_wasserstein_illustration.jl
import CairoMakie, DuckDB, DataFrames
const MK = CairoMakie; const DF = DataFrames

con = DuckDB.connect(DuckDB.DB("/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb"))
co = DF.DataFrame(DuckDB.execute(con, "SELECT age_calc AS age, sum(agb) AS agb FROM curated_cohorts_landis " *
  "WHERE statecd=48 AND unitcd=1 AND countycd=373 AND plot=37 AND measdate=DATE '2025-06-18' GROUP BY age_calc ORDER BY age_calc"))
ages = Float64.(co.age); w = Float64.(co.agb); w ./= sum(w)
cum = cumsum(w)                                    # cumulative AGB fraction after each cohort age
println("cohorts: ages=", Int.(ages), "  weights=", round.(w; digits=3))

grid = collect(0.0:0.5:170.0); dg = grid[2] - grid[1]
stepF(xa, xc, a) = (i = searchsortedlast(xa, a); i == 0 ? 0.0 : xc[i])   # right-continuous step CDF
Fobs = [stepF(ages, cum, a) for a in grid]

# sim = the cohorts shifted by Δ (a stand-in for a model that mis-places the age structure)
simset(Δ) = (let sa = ages .+ Δ, o = sortperm(sa); (sa[o], cumsum(w[o])); end)

panels = [(12.0, "close sim"), (42.0, "diverging sim")]
fig = MK.Figure(size=(1200, 500))
MK.Label(fig[0, 1:2], "Wasserstein-1 on a real plot's cohort age-CDF (plot 48-1-373-37, $(length(ages)) cohorts) — shaded area = W1"; fontsize=14, font=:bold)
for (i, (Δ, lbl)) in enumerate(panels)
  sa, scum = simset(Δ)
  Fsim = [stepF(sa, scum, a) for a in grid]
  w1 = sum(abs.(Fsim .- Fobs)) * dg
  ax = MK.Axis(fig[1, i]; title="$(lbl):  W1 = $(round(w1; digits=1)) yr", xlabel="cohort age (yr)",
    ylabel="cumulative AGB fraction (CDF)", limits=(0, 170, -0.03, 1.05))
  MK.band!(ax, grid, min.(Fobs, Fsim), max.(Fobs, Fsim); color=(:darkorange, 0.30))               # = W1
  # obs: dashed step cumsum + cohort dots
  MK.stairs!(ax, [0.0; ages; 170.0], [0.0; cum; 1.0]; color=:navy, linestyle=:dash, linewidth=2.0, step=:post)
  MK.scatter!(ax, ages, cum; color=:navy, markersize=11)
  # sim: dashed step cumsum + cohort dots
  MK.stairs!(ax, [0.0; sa; 170.0], [0.0; scum; 1.0]; color=:firebrick, linestyle=:dash, linewidth=2.0, step=:post)
  MK.scatter!(ax, sa, scum; color=:firebrick, markersize=11, marker=:diamond)
  MK.Legend(fig[2, i], [MK.MarkerElement(color=:navy, marker=:circle), MK.MarkerElement(color=:firebrick, marker=:diamond), MK.PolyElement(color=(:darkorange, 0.45))],
    ["observed cohorts (+CDF)", "simulated cohorts (+CDF)", "W1 (area between CDFs)"]; orientation=:horizontal, framevisible=false)
end
out = "tools/wasserstein_illustration.png"
MK.save(out, fig); println("wrote $out")
