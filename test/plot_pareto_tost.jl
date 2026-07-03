# Pareto front of the single-cov Sim-A MO-CMA-ES archive (2 objectives: A_W vs A_AGB), one panel per
# stratum (eco×lu), each candidate coloured by the number of species that PASS the ±$(PCT)% TOST equivalence
# test in that stratum (train split). Shows where on the W↔AGB trade-off the most species are equivalent.
#   Run: ./julia_gdal.sh --project=. test/plot_pareto_tost.jl
import CairoMakie, DataFrames, CSV
const MK = CairoMakie; const DF = DataFrames
SPLIT = length(ARGS) >= 1 ? ARGS[1] : "train"
O = length(ARGS) >= 3 ? ARGS[3] : "runs/fl5_l4cover_mocmaes_Aonly_Sglobal_ipop_v3_strata_singlecov_outputs"
PCT = length(ARGS) >= 2 ? ARGS[2] : "20"

arch = CSV.read("$O/archive_candidates.csv", DF.DataFrame)
cand = Int.(arch.candidate); AW = Float64.(arch.A_W); AAGB = Float64.(arch.A_AGB); agg = Float64.(arch.aggregate)
strata = ["8.3.5|lu=natural", "8.3.5|lu=artificial", "8.5.3|lu=natural", "8.5.3|lu=artificial"]

# per (candidate, stratum): #TOST-equivalent species and #species tested (train)
neq = Dict{Tuple{Int,String},Int}(); ntot = Dict{Tuple{Int,String},Int}()
for i in cand
  f = "$O/candidates/candidate_$i/tost_sim_obs_$(PCT)pct.csv"; isfile(f) || continue
  t = CSV.read(f, DF.DataFrame); tt = t[t.split .== SPLIT, :]
  for st in strata
    sub = tt[tt.eco .== st, :]
    neq[(i, st)] = count(sub.equivalent); ntot[(i, st)] = DF.nrow(sub)
  end
end
gmax = max(1, maximum(get(neq, (i, st), 0) for i in cand for st in strata))   # floor at 1 so colorrange (0,gmax) is valid even when all-zero (strict ±5% val)
rep = cand[argmin(agg)]   # representative = min aggregate
println("archive: $(length(cand)) candidates; representative = candidate $rep; max #TOST-equiv (any stratum) = $gmax")

fig = MK.Figure(size=(1150, 950))
MK.Label(fig[0, 1:2], "Single-cov Sim-A Pareto front (A_W vs A_AGB) — coloured by #species passing ±$(PCT)% TOST ($(SPLIT)), per stratum"; fontsize=14, font=:bold)
sc_ref = Ref{Any}(nothing)
for (k, st) in enumerate(strata)
  r, c = fldmod1(k, 2)
  ncell = maximum(get(ntot, (i, st), 0) for i in cand)
  ax = MK.Axis(fig[r, c]; xlabel="A_W  (Wasserstein objective)", ylabel="A_AGB  (AGB objective)",
    title="$(replace(st, "|lu=" => "  ")) — up to $ncell species")
  o = sortperm(AW)
  MK.lines!(ax, AW[o], AAGB[o]; color=(:gray, 0.35), linewidth=1)              # front curve
  cs = [get(neq, (i, st), 0) for i in cand]
  sc_ref[] = MK.scatter!(ax, AW, AAGB; color=cs, colormap=:viridis, colorrange=(0, gmax), markersize=15, strokewidth=0.5, strokecolor=:black)
  ri = findfirst(==(rep), cand)
  MK.scatter!(ax, [AW[ri]], [AAGB[ri]]; color=:transparent, marker=:diamond, markersize=22, strokecolor=:red, strokewidth=2)  # representative
  for (j, i) in enumerate(cand)                                                # candidate id labels
    MK.text!(ax, AW[j], AAGB[j]; text=string(i), fontsize=7, align=(:center, :center), color=:white)
  end
end
MK.Colorbar(fig[1:2, 3], sc_ref[]; label="# species passing ±$(PCT)% TOST ($(SPLIT))")
MK.Label(fig[3, 1:2], "◇ red = representative (min-aggregate).  Same 18 non-dominated candidates in every panel; colour = that stratum's TOST-equivalent count."; fontsize=10)
out = "$O/pareto_tost_$(PCT)pct_by_stratum_$(SPLIT).png"
MK.save(out, fig); println("wrote $out")
