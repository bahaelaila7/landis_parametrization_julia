# Integration check for B2 / step 3e (needs the GA+FL data; PAN_DATA=/workspace/landis_region4/data_r4):
#   PAN_PARALLEL=candidate ./julia_gdal.sh --project=. --threads=2 test/integration/parallel_sites_restore.jl <out_dir>
# Site-parallelism is no longer process state: it is `soa.parallel`, set per evaluation by `fit_params(...; parallel_sites)`. A 1-trial
# igelmo run in candidate mode (the driver evaluates its candidates sites-serial) must therefore leave nothing behind that makes a
# LATER evaluation serial: neither the reference SoA nor a reused work SoA keeps the flag, and there is no global toggle left.
import YAML; using Pan, Test
include(joinpath(@__DIR__, "..", "golden", "scenarios.jl"))
base, ov, _ = SCENARIOS["bs_igelmo"]
over = merge(common(), ov, Dict{String,Any}("output_dir" => ARGS[1], "trials" => 1, "igel_mu" => 2))
ctx = Ref{Any}(nothing)
Pan.run_from_yaml(base; overrides=over, post_search_hook=d -> (ctx[] = d))
d = ctx[]
@test !isdefined(Pan, :PARALLEL_SITES) && !isdefined(Pan.PanCore, :PARALLEL_SITES)
@test d.ref_soa.parallel === true                          # the driver never touched the caller's reference
@test deepcopy(d.ref_soa).parallel === true
# per-evaluation flag: a work SoA reused by a serial evaluation is parallel again in the next default evaluation
msy = maximum(maximum.(filter(!isempty, d.site_sim_years.sim_years)))
tmpl = Pan.generate_params(d.plugin.ctx, d.species_list, d.eco_list, d.eco_species_ids; rng=Pan.Random.Xoshiro(7), no_establishment=true)
inj = Pan.build_injection_plan(d.injection_cohorts, d.ref_soa, d.eco_species_ids, d.loss_params, d.scoring)
sc = Pan.compute_loss_scales(d.loss_params, 3, d.spdf_plts, nothing, d.eco_species_ids, d.n_species; cbal=d.scale_seed.cbal, ncoh=d.scale_seed.ncoh, dom=d.scale_seed.dom, per_plot_mean=d.simb.t4_per_plot_mean)
ws = deepcopy(d.ref_soa)
ev(; kw...) = Pan.fit_params(d.ref_soa, tmpl, msy, d.n_species, d.eco_species_ids, d.spdf_plts, d.site_sim_years, false, d.spinup_cohorts, d.loss_params;
  debug=false, search_tier=3, injection=inj, seeds=[UInt64(1)], scales=sc, scoring=d.scoring, simb=d.simb, plugin=d.plugin, work_soa=ws, kw...)
r_ser = ev(parallel_sites=false); @test ws.parallel === false
r_par = ev();                     @test ws.parallel === true
tot(r) = convert(Float64, Pan.PU.get_total_loss(r[1][1], d.loss_params))
@test isfinite(tot(r_ser)) && isfinite(tot(r_par))          # serial and threaded evaluations of the same params/seed agree up to summation order
@test isapprox(tot(r_ser), tot(r_par); rtol=1e-4)
println("parallel_sites_restore: OK")
