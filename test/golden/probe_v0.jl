# Probe adapter for the pre-refactor API (Pan.POST_SEARCH_HOOK + fit_params positional signature; since step 3a the hook context carries `scale_seed`
# and the probe passes the training reference's LossScales explicitly; since step 3c it also carries the run's ParamSpace as `space`; since step 3d the run's PluginConfig as `plugin`).
# Evaluates ONE deterministic parameter set (fixed-seed template + fixed u) at tier 3 on the training context and writes:
#   probe_eco_loss.csv  eco,species,sp_w_loss,sp_agb_loss     (per eco × global species; zero rows of absent species skipped)
#   probe_summary.csv   key,value                              (totals, counts, u round-trip error)
#   probe_roundtrip.csv dim,param,u,u_back
#   probe_cohorts.csv   ref_cn,year,species,age,bio            (pre-injection sim state at measurement years, sorted)
using Random
const PU = Pan.PU
function golden_probe(d, cfg, outdir)
  pc = d.plugin.ctx   # the run's plugin context (since step 3d the hook context carries it as `plugin`)
  noest = Bool(get(cfg, "no_establishment", false))
  space = Pan.with_establishment(pc, d.space; no_establishment=noest, fit_establishment=false)   # the run's parameter space (options, pooling, floors) over the probe's subset
  pd = space.dists
  tmpl = Pan.generate_params(pc, d.species_list, d.eco_list, d.eco_species_ids; rng=Random.Xoshiro(7), no_establishment=noest)
  slots = space.slots
  u = rand(Random.Xoshiro(11), length(slots))
  p = PU.u_to_params(u, space, slots, tmpl)
  u2 = PU.params_to_u(p, space, slots)
  injection = Pan.build_injection_plan(d.injection_cohorts, d.ref_soa, d.eco_species_ids, d.loss_params, d.scoring)   # this reference's injection dict / shock / exclusions
  msy = maximum(maximum.(filter(!isempty, d.site_sim_years.sim_years)))
  sc = Pan.compute_loss_scales(d.loss_params, 3, d.spdf_plts, nothing, d.eco_species_ids, d.n_species; cbal=d.scale_seed.cbal, ncoh=d.scale_seed.ncoh, dom=d.scale_seed.dom, per_plot_mean=d.simb.t4_per_plot_mean)   # the training reference's loss scales
  r = Pan.fit_params(d.ref_soa, p, msy, d.n_species, d.eco_species_ids, d.spdf_plts, d.site_sim_years, false, d.spinup_cohorts, d.loss_params;
    debug=false, search_tier=3, injection=injection, seeds=[UInt64(1)], cache_preinject=true, scales=sc, scoring=d.scoring, simb=d.simb, plugin=d.plugin)
  run, state, eco = r[1]
  open(joinpath(outdir, "probe_eco_loss.csv"), "w") do io
    println(io, "eco,species,sp_w_loss,sp_agb_loss")
    for (e, l) in enumerate(eco), j in eachindex(l.sp_w_loss)   # sp_*_loss are indexed by GLOBAL species id
      (l.sp_w_loss[j] == 0 && l.sp_agb_loss[j] == 0 && !(j in d.eco_species_ids[e])) && continue
      println(io, d.eco_list[e], ",", d.species_list[j], ",", Float64(l.sp_w_loss[j]), ",", Float64(l.sp_agb_loss[j]))
    end
  end
  open(joinpath(outdir, "probe_roundtrip.csv"), "w") do io   # u → params → u per slot (flat regions / floors show up here)
    println(io, "dim,param,u,u_back")
    for (k, (pi, idx)) in enumerate(slots); println(io, k, ",", pd.params[pi].name, ",", u[k], ",", u2[k]); end
  end
  open(joinpath(outdir, "probe_summary.csv"), "w") do io
    println(io, "key,value")
    println(io, "total_loss,", Float64(PU.get_total_loss(run, d.loss_params)))
    println(io, "n_slots,", length(slots))
    println(io, "n_sites,", d.ref_soa.n)
    println(io, "n_species,", d.n_species)
    println(io, "n_ecos,", length(d.eco_list))
    println(io, "u_roundtrip_maxerr,", maximum(abs.(u2 .- u); init=0.0))
    println(io, "n_state_cohorts,", length(state))
  end
  open(joinpath(outdir, "probe_cohorts.csv"), "w") do io
    println(io, "ref_cn,year,species,age,bio")
    for c in sort(state; by=c -> (c[1], c[2], c[3], c[4], c[5]))
      println(io, Int(c[1]), ",", c[2], ",", d.species_list[Int(c[3])], ",", Int(c[4]), ",", Float64(c[5]))
    end
  end
  nothing
end
install_probe!(cfg, outdir) = (Pan.POST_SEARCH_HOOK[] = d -> try
    golden_probe(d, cfg, outdir)
  catch e
    open(io -> println(io, "PROBE_ERROR ", sprint(showerror, e)[1:min(end, 400)]), joinpath(outdir, "probe_error.txt"), "w")
  end; nothing)   # global hook: nothing to pass to run_from_yaml
