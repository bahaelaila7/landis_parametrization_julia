# Counts how many disturbance-shock entries map to sim cohorts (GA+FL train split) at a params set; and partial-drop cohort-years.
#   PAN_DATA=... ./julia_gdal.sh --project=. --threads=4 tools/shock_stats.jl <config.yml> <params.jld2> <out_dir>
using Pan, JLD2
const P = Pan
cfgpath, ppath, outd = ARGS[1], ARGS[2], ARGS[3]
function chk(d)
  p0 = JLD2.load_object(ppath)
  ic = d.injection_cohorts
  println("STATS injection cohort rows (train): ", P.DataFrames.nrow(ic), "; with drop>0: ", count(>(0), ic.disturbance_drop_pct), "; sites with drop: ", length(unique(ic.plot_id[ic.disturbance_drop_pct .> 0])))
  sh = P.DataFrames.metadata(ic, "disturbance_shock")
  println("STATS plot-level full-removal entries (train): ", P.DataFrames.nrow(sh))
  idict = P._build_injection_dict(ic, d.ref_soa); iy = Set(Int.(ic.sim_year))
  msy = maximum(maximum.(filter(!isempty, d.site_sim_years.sim_years)))
  foreach(a -> a[] = 0, P.SHOCK_STATS)
  P.fit_params(d.ref_soa, p0, msy, d.n_species, d.eco_species_ids, d.spdf_plts, d.site_sim_years, false, d.spinup_cohorts, d.loss_params;
    debug=false, search_tier=3, injection_dict=idict, injection_years=iy, seeds=[UInt64(1)])
  s = [a[] for a in P.SHOCK_STATS]
  println("STATS shock: full-removal entries applied=$(s[1]) matched sim cohorts (removed)=$(s[2]); partial cohorts with drop>0 =$(s[3]) matched sim cohorts (scaled, :scale mode only)=$(s[4])")
  pl = P._get_excl_plan(idict, d.ref_soa, d.eco_species_ids, d.loss_params)
  println("STATS cohort-level exclusions: ", sum(length(v) for y in values(pl.coh) for v in values(y)), " cohorts in ", sum(length(y) for y in values(pl.coh)), " site-years")
end
P.POST_SEARCH_HOOK[] = chk
P.run_from_yaml(cfgpath; overrides=Dict{String,Any}("output_dir" => outd, "trials" => 1, "igel_mu" => 2, "igel_sobol_init" => false, "search_mode" => "igelmo", "n_reps" => 1, "start_from" => ppath))
