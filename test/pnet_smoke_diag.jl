# Sim-vs-obs site AGB at the LAST measurement for the best candidate of a PnET (or any) run.
#   PAN_DATA=... PAN_OUT=... ./julia_gdal.sh --project=. --threads=auto test/pnet_smoke_diag.jl <config.yml> [overrides: key=value ...]
# Runs the search through Pan.run_from_yaml; right after the search a hook re-simulates the TRAIN plots with the best_params of the
# newest checkpoint and reports, over plots, the distribution (median / p10 / p90) of simulated vs observed total AGB at each plot's
# last measurement, for the pure model prediction (pre-injection) and for what the loss sees (post set-sync injection).
using Pan, JLD2, DataFrames, Statistics, Printf, CSV
const P = Pan

q(v, p) = (s = sort(v); s[clamp(round(Int, p * (length(s) - 1)) + 1, 1, length(s))])

function diag(d)
  outd = d.output_dir
  ck = sort(filter(f -> occursin(r"^best_params@\d+\.jld2$", f), readdir(outd)); by=f -> parse(Int, match(r"@(\d+)", f)[1]))
  best = JLD2.load_object(joinpath(outd, last(ck)))
  println("DIAG: best params = $(last(ck))")
  idict = isnothing(d.injection_cohorts) ? nothing : P._build_injection_dict(d.injection_cohorts, d.ref_soa)
  iyears = isnothing(d.injection_cohorts) ? Set{Int}() : Set(Int.(d.injection_cohorts.sim_year))
  msy = maximum(maximum.(filter(!isempty, d.site_sim_years.sim_years)))
  obs = combine(groupby(d.splots, [:plot_id, :sim_year]), :agb_sum => sum => :obs)
  rows = DataFrame(plot_id=Int[], mode=String[], sim_year=Int[], sim=Float64[], obs=Float64[])
  for (mode, pre) in (("pre-injection (pure model)", true), ("post-injection (loss view)", false))
    res = P.fit_params(d.ref_soa, best, msy, d.n_species, d.eco_species_ids, d.spdf_plts, d.site_sim_years, false, d.spinup_cohorts, d.loss_params;
      debug=false, search_tier=3, injection_dict=idict, injection_years=iyears, seeds=[UInt64(1)], cache_preinject=pre)
    cached = res[1][2]
    sim = Dict{Tuple{Int,Int},Float64}()
    for (pid, sy, _esp, _age, bio) in cached
      sim[(Int(pid), Int(sy))] = get(sim, (Int(pid), Int(sy)), 0.0) + Float64(bio)
    end
    last_sy = Dict{Int,Int}()
    for (pid, sy) in keys(sim); last_sy[pid] = max(get(last_sy, pid, 0), sy); end
    ob = Dict((Int(r.plot_id), Int(r.sim_year)) => Float64(r.obs) for r in eachrow(obs))
    S = Float64[]; O = Float64[]
    for (pid, sy) in last_sy
      sy > 0 && haskey(ob, (pid, sy)) || continue
      push!(S, sim[(pid, sy)]); push!(O, ob[(pid, sy)]); push!(rows, (pid, mode, sy, S[end], O[end]))
    end
    R = S ./ O
    @printf("DIAG [%s] n_plots=%d  (g/m2)\n  simulated AGB: median %.0f  p10 %.0f  p90 %.0f\n  observed  AGB: median %.0f  p10 %.0f  p90 %.0f\n  sim/obs ratio: median %.3f  p10 %.3f  p90 %.3f   median(sim)/median(obs) = %.3f   corr(log sim, log obs) = %.3f\n",
      mode, length(S), median(S), q(S, 0.1), q(S, 0.9), median(O), q(O, 0.1), q(O, 0.9), median(R), q(R, 0.1), q(R, 0.9), median(S) / median(O),
      cor(log.(max.(S, 1.0)), log.(max.(O, 1.0))))
  end
  CSV.write(joinpath(outd, "pnet_diag_last_measurement.csv"), rows)
  println("DIAG: wrote ", joinpath(outd, "pnet_diag_last_measurement.csv"))
end

P.POST_SEARCH_HOOK[] = diag
ov = Dict{String,Any}()
for a in ARGS[2:end]
  k, v = split(a, '='; limit=2)
  ov[k] = something(tryparse(Int, v), tryparse(Float64, v), v)
end
P.run_from_yaml(ARGS[1]; overrides=ov)
