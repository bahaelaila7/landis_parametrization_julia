# Check: gdigel's in-loop group bias m_g at a params set (printed per group) — compare with tools/tost_select.jl output.
#   ./julia_gdal.sh --project=. --threads=8 tools/gd_check.jl <config.yml> <params.jld2> <out_dir>
using Pan, JLD2, Statistics, Printf
const P = Pan; const PU = P.PU
cfgpath, ppath, outd = ARGS[1], ARGS[2], ARGS[3]
function chk(d)
  SP = P.succession_plugin_of(d.ref_soa); p0 = JLD2.load_object(ppath)
  pd = P.make_param_dists(SP, d.n_species, length(d.eco_list), d.eco_species_ids; no_establishment=true, fit_establishment=false)
  slots = PU.build_slots(pd, p0)
  cg, labels = P._gd_groups(pd, slots, d.eco_species_ids, d.species_list, d.eco_list)
  ix = P._gd_index(d.injection_cohorts, d.splots, cg, labels, d.scoring; real_only=true)
  injection = P.build_injection_plan(d.injection_cohorts, d.ref_soa, d.eco_species_ids, d.loss_params, d.scoring)
  msy = maximum(maximum.(filter(!isempty, d.site_sim_years.sim_years)))
  r = P.fit_params(d.ref_soa, p0, msy, d.n_species, d.eco_species_ids, d.spdf_plts, d.site_sim_years, false, d.spinup_cohorts, d.loss_params;
    debug=false, search_tier=3, injection=injection, seeds=[UInt64(1)], cache_preinject=true, scoring=d.scoring, simb=d.simb)
  L = P._gd_losses(ix, r[1][2])
  for g in 1:ix.ngroups; @printf("GDCHK %-24s n=%5d bias=%7.2f%%\n", labels[g], L.n[g], 100 * expm1(L.m[g])); end
  @printf("GDCHK pass05=%d pass10=%d pass20=%d\n", P._gd_pass(L.m, L.se, 0.05), P._gd_pass(L.m, L.se, 0.1), P._gd_pass(L.m, L.se, 0.2))
end
P.POST_SEARCH_HOOK[] = chk
P.run_from_yaml(cfgpath; overrides=Dict{String,Any}("output_dir" => outd, "trials" => 1, "igel_mu" => 2, "igel_sobol_init" => false, "search_mode" => "igelmo", "n_reps" => 1, "start_from" => ppath))
