# Probe adapter for the pre-refactor API (Pan.POST_SEARCH_HOOK + fit_params positional signature).
# Evaluates ONE deterministic parameter set (fixed-seed template + fixed u) at tier 3 on the training context and writes:
#   probe_eco_loss.csv  eco,species,sp_w_loss,sp_agb_loss     (per eco × global species; zero rows of absent species skipped)
#   probe_summary.csv   key,value                              (totals, counts, u round-trip error)
#   probe_roundtrip.csv dim,param,u,u_back
#   probe_cohorts.csv   ref_cn,year,species,age,bio            (pre-injection sim state at measurement years, sorted)
using Random
const PU = Pan.PU
function golden_probe(d, cfg, outdir)
  SP = Pan.succession_plugin_of(d.ref_soa)
  noest = Bool(get(cfg, "no_establishment", false))
  pd = Pan.make_param_dists(SP, d.n_species, length(d.eco_list), d.eco_species_ids; no_establishment=noest, fit_establishment=false)
  tmpl = Pan.generate_params(SP, d.species_list, d.eco_list, d.eco_species_ids; rng=Random.Xoshiro(7), no_establishment=noest)
  slots = PU.build_slots(pd, tmpl)
  u = rand(Random.Xoshiro(11), length(slots))
  p = PU.u_to_params(u, pd, slots, tmpl)
  u2 = PU.params_to_u(p, pd, slots)
  inj = d.injection_cohorts
  idict = inj === nothing ? nothing : Pan._build_injection_dict(inj, d.ref_soa)
  iy = inj === nothing ? Set{Int}() : Set(Int.(inj.sim_year))
  msy = maximum(maximum.(filter(!isempty, d.site_sim_years.sim_years)))
  r = Pan.fit_params(d.ref_soa, p, msy, d.n_species, d.eco_species_ids, d.spdf_plts, d.site_sim_years, false, d.spinup_cohorts, d.loss_params;
    debug=false, search_tier=3, injection_dict=idict, injection_years=iy, seeds=[UInt64(1)], cache_preinject=true)
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
    println(io, "total_loss,", Float64(PU.get_total_loss(run)))
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
  end)
