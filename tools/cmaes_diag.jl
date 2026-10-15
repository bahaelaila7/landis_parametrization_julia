# Landscape / noise diagnostic for the CMA-ES training objective, around a parameter set (any succession plugin).
#   PAN_DATA=... PAN_OUT=... ./julia_gdal.sh --project=. --threads=8 tools/cmaes_diag.jl <config.yml> <params.jld2> <out_dir>
# Prepares the training context exactly as a search would (run_from_yaml, trials=1, then POST_SEARCH_HOOK) and measures:
#  A. repeat-evaluation noise of the MO objectives [ΣW, ΣAGB] (same seed twice; 3 different seeds);
#  B. 1-D slices u0 + t·z (z ~ N(0, I) in the search's u-space, as a CMA offspring with σ = t) for coarse and fine t:
#     roughness = residual SD around a quadratic fit in t, vs the slope signal;
#  C. offspring clouds at σ ∈ {0.08, 0.02, 0.005}: Δ objectives, fraction dominating the parent, and the correlation of
#     Δ(ΣAGB) with Δ(per-species bias loss) — the quantity the TOST equivalence test measures;
#  D. dead dimensions: slots whose ±0.02 u-step does not change the decoded native value (clamps / data floors).
using Pan, JLD2, Statistics, Random, Printf, DataFrames, CSV
const P = Pan; const PU = P.PU
cfgpath, ppath, outd = ARGS[1], ARGS[2], ARGS[3]; mkpath(outd)

function diag(d)
  SP = P.succession_plugin_of(d.ref_soa)
  p0 = JLD2.load_object(ppath)
  pd = P.make_param_dists(SP, d.n_species, length(d.eco_list), d.eco_species_ids; no_establishment=true, fit_establishment=false)
  slots = PU.build_slots(pd, p0)
  u0 = PU.params_to_u(p0, pd, slots); nd = length(u0)
  idict = isnothing(d.injection_cohorts) ? nothing : P._build_injection_dict(d.injection_cohorts, d.ref_soa)
  iyears = isnothing(d.injection_cohorts) ? Set{Int}() : Set(Int.(d.injection_cohorts.sim_year))
  msy = maximum(maximum.(filter(!isempty, d.site_sim_years.sim_years)))
  inj = d.injection_cohorts
  injc = combine(groupby(DataFrame(plot_id=Int.(inj.plot_id), sim_year=Int.(inj.sim_year), esp=Int.(inj.eco_species_id),
      age=Int.(inj.age_calc), obs=Float64.(inj.agb_sum))[Int.(inj.sim_year) .> 0, :], [:plot_id, :sim_year, :esp, :age]), :obs => sum => :obs)
  plot2eco = Dict(Int(r.plot_id) => Int(r.eco_id) for r in eachrow(unique(select(d.splots, [:plot_id, :eco_id]))))
  # one evaluation: MO objectives (as the driver scores them) + per-global-species mean log bias of matched cohorts
  sc = P.compute_loss_scales(d.loss_params, d.tier, d.spdf_plts, nothing, d.eco_species_ids, d.n_species; cbal=d.scale_seed.cbal, ncoh=d.scale_seed.ncoh)   # the reference's loss scales
  function ev(p; seed=UInt64(1))
    r = P.fit_params(d.ref_soa, p, msy, d.n_species, d.eco_species_ids, d.spdf_plts, d.site_sim_years, false, d.spinup_cohorts, d.loss_params;
      debug=false, search_tier=d.tier, injection_dict=idict, injection_years=iyears, seeds=[seed], cache_preinject=true, scales=sc)
    rr, el, k = P._agg_reps(r, d.loss_params)
    objs = Float64.(P._mo_objectives(el, d.eco_species_ids, d.loss_params, sc))
    sim = Dict{NTuple{4,Int},Float64}()
    for (pid, sy, esp, age, bio) in r[k][2]
      sy > 0 || continue; key = (Int(pid), Int(sy), Int(esp), Int(age)); sim[key] = get(sim, key, 0.0) + Float64(bio)
    end
    acc = Dict{Int,Vector{Float64}}()
    for row in eachrow(injc)
      s = get(sim, (row.plot_id, row.sim_year, row.esp, row.age), nothing); s === nothing && continue
      gsp = Int(d.eco_species_ids[plot2eco[row.plot_id]][row.esp])
      push!(get!(acc, gsp, Float64[]), log1p(s) - log1p(row.obs))
    end
    bias = Dict(g => mean(v) for (g, v) in acc if length(v) >= 30)
    bl = mean(b^2 for b in values(bias))            # per-species bias loss (TOST-aligned)
    (objs=objs, bl=bl, bias=bias)
  end
  dec(u) = PU.u_to_params(clamp.(u, 0.0, 1.0), pd, slots, p0)
  log_rows = DataFrame(test=String[], sigma=Float64[], dir=Int[], t=Float64[], W=Float64[], AGB=Float64[], biasloss=Float64[])
  t0 = time(); e0 = ev(p0); te = time() - t0
  @printf("DIAG plugin=%s slots=%d  base: ΣW=%.6f ΣAGB=%.6f biasloss=%.5f  (eval %.1fs)\n", SP, nd, e0.objs[1], e0.objs[2], e0.bl, te)
  # A. repeat noise
  ea = ev(p0); @printf("A. same seed twice: |ΔW|=%.3g |ΔAGB|=%.3g\n", abs(ea.objs[1] - e0.objs[1]), abs(ea.objs[2] - e0.objs[2]))
  for s in (UInt64(2), UInt64(3), UInt64(4))
    es = ev(p0; seed=s); @printf("A. seed %d: ΔW=%.3g ΔAGB=%.3g Δbiasloss=%.3g\n", s, es.objs[1] - e0.objs[1], es.objs[2] - e0.objs[2], es.bl - e0.bl)
    push!(log_rows, ("seed", 0.0, Int(s), 0.0, es.objs[1], es.objs[2], es.bl))
  end
  # D. dead dimensions
  rng = Random.Xoshiro(7)
  nv0 = PU.params_to_nativevec(p0, pd, slots); dead = 0
  for k in 1:nd
    up = copy(u0); up[k] = clamp(u0[k] + 0.02, 0, 1); dn = copy(u0); dn[k] = clamp(u0[k] - 0.02, 0, 1)
    a = PU.params_to_nativevec(dec(up), pd, slots); b = PU.params_to_nativevec(dec(dn), pd, slots)
    (a[k] == nv0[k] && b[k] == nv0[k]) && (dead += 1)
  end
  @printf("D. dead slots (±0.02 in u changes nothing after decode): %d / %d  (u at bounds: %d)\n", dead, nd, count(x -> x <= 1e-6 || x >= 1 - 1e-6, u0))
  # B. slices
  for (lab, ts) in (("coarse", [-0.08, -0.04, -0.02, 0.02, 0.04, 0.08]), ("fine", [-0.004, -0.002, -0.001, 0.001, 0.002, 0.004]))
    for dir in 1:3
      z = randn(rng, nd); T = [0.0]; F = [e0.objs[2]]; Wv = [e0.objs[1]]
      for t in ts
        e = ev(dec(u0 .+ t .* z)); push!(T, t); push!(F, e.objs[2]); push!(Wv, e.objs[1])
        push!(log_rows, ("slice_" * lab, 0.0, dir, t, e.objs[1], e.objs[2], e.bl))
      end
      X = hcat(ones(length(T)), T, T .^ 2)
      for (nm, Y) in (("AGB", F), ("W", Wv))
        β = X \ Y; res = Y .- X * β
        @printf("B. %-6s dir %d %-3s: slope*t_max=%.3g  curvature*t_max^2=%.3g  residual SD=%.3g  (Δrange %.3g)\n", lab, dir, nm,
          β[2] * maximum(abs.(ts)), β[3] * maximum(abs.(ts))^2, std(res), maximum(Y) - minimum(Y))
      end
    end
  end
  # C. offspring clouds
  for σ in (0.08, 0.02, 0.005)
    dW = Float64[]; dA = Float64[]; dB = Float64[]; dom = 0
    for j in 1:12
      e = ev(dec(u0 .+ σ .* randn(rng, nd)))
      push!(dW, e.objs[1] - e0.objs[1]); push!(dA, e.objs[2] - e0.objs[2]); push!(dB, e.bl - e0.bl)
      (e.objs[1] <= e0.objs[1] && e.objs[2] <= e0.objs[2] && (e.objs[1] < e0.objs[1] || e.objs[2] < e0.objs[2])) && (dom += 1)
      push!(log_rows, ("offspring", σ, j, 0.0, e.objs[1], e.objs[2], e.bl))
    end
    @printf("C. σ=%.3f: ΔW median %.3g [%.3g, %.3g]  ΔAGB median %.3g [%.3g, %.3g]  Δbiasloss median %.3g  dominating parent %d/12  corr(ΔAGB, Δbiasloss)=%.2f  corr(ΔW, ΔAGB)=%.2f\n",
      σ, median(dW), minimum(dW), maximum(dW), median(dA), minimum(dA), maximum(dA), median(dB), dom, cor(dA, dB), cor(dW, dA))
  end
  CSV.write(joinpath(outd, "cmaes_diag.csv"), log_rows)
  println("DIAG_DONE → ", joinpath(outd, "cmaes_diag.csv"))
end

P.POST_SEARCH_HOOK[] = diag
P.run_from_yaml(cfgpath; overrides=Dict{String,Any}("output_dir" => outd, "trials" => 1, "igel_mu" => 2, "igel_sobol_init" => false,
  "search_mode" => "igelmo", "n_reps" => 1, "start_from" => ppath))
