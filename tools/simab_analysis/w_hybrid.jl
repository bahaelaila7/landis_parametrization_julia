# Is the A–W front a genuine trade-off or a search artifact? Evaluate gdigel's own objectives (A = Σ a_g, W = Σ w_g, train)
# on (1) points interpolated in u between the archive's min_A and min_W ends, (2) min_A with one parameter family taken from min_W.
#   ./julia_gdal.sh --project=. --threads=4 w_hybrid.jl <config.yml> <archive_dir> <gd_state.jld2> [w_scope]
using Pan, JLD2, CSV, DataFrames, Printf, Statistics
const P = Pan; const PU = P.PU
cfgpath, adir, stpath = ARGS[1:3]; wscope = length(ARGS) >= 4 ? ARGS[4] : "survivors"
arch = CSV.read(joinpath(adir, "mo_archive.csv"), DataFrame); U = JLD2.load_object(joinpath(adir, "mo_archive_u.jld2"))
ck = JLD2.load_object(stpath)
function run(d)
  P.PARALLEL_SITES[] = true
  SP = P.succession_plugin_of(d.ref_soa)
  pd = P.make_param_dists(SP, d.n_species, length(d.eco_list), d.eco_species_ids; no_establishment=true, fit_establishment=false)
  slots = ck.slots; tmpl = ck.template
  cg, labels = P._gd_groups(pd, slots, d.eco_species_ids, d.species_list, d.eco_list)
  ix = P._gd_index(d.injection_cohorts, d.splots, cg, labels; real_only=true); gc = P._gd_groupcells(ix.cell_group, ix.ngroups)
  idict = P._build_injection_dict(d.injection_cohorts, d.ref_soa); iy = Set(Int.(d.injection_cohorts.sim_year))
  msy = maximum(maximum.(filter(!isempty, d.site_sim_years.sim_years))); nvis = P._gd_cell_visits(d.spdf_plts, d.eco_species_ids, 1)
  function ev(u)
    p = PU.u_to_params(clamp.(u, 0.0, 1.0), pd, slots, tmpl)
    r = P.fit_params(d.ref_soa, p, msy, d.n_species, d.eco_species_ids, d.spdf_plts, d.site_sim_years, false, d.spinup_cohorts, d.loss_params;
      debug=false, search_tier=3, injection_dict=idict, injection_years=iy, seeds=[UInt64(1)], cache_preinject=true)
    L = P._gd_losses(ix, r[1][2]); num, den = P._gd_cell_w(r[1][3], d.eco_species_ids, nvis); w = P._gd_group_w(num, den, gc)
    (A=sum(filter(!isnan, L.a)), W=sum(filter(!isnan, w)))
  end
  ia = argmin(U.A); iw = argmin(U.W)
  uA = U.u[ia]; uW = U.u[iw]
  for (nm, u) in (("min_A", uA), ("min_W", uW)); r = ev(u); @printf("WHYB %-28s A=%.3f W=%.4f\n", nm, r.A, r.W); end
  for t in (0.25, 0.5, 0.75); r = ev((1 - t) .* uA .+ t .* uW); @printf("WHYB interp t=%.2f%-17s A=%.3f W=%.4f\n", t, "", r.A, r.W); end
  fam = Dict{Symbol,Vector{Int}}()
  for (k, (pi, _)) in enumerate(slots); push!(get!(fam, pd.params[pi].name, Int[]), k); end
  for (f, ks) in fam
    u = copy(uA); u[ks] .= uW[ks]; r = ev(u); @printf("WHYB min_A + min_W's %-12s A=%.3f W=%.4f  (%d dims)\n", String(f), r.A, r.W, length(ks))
  end
  println("WHYB front (A, W) for reference: ", join([@sprintf("(%.2f, %.3f)", a, w) for (a, w) in zip(arch.sumA, arch.sumW)][sortperm(arch.sumA)][1:4:end], " "))
end
P.POST_SEARCH_HOOK[] = run
P.run_from_yaml(cfgpath; overrides=Dict{String,Any}("output_dir" => joinpath(adir, "..", "_ctx_hyb"), "trials" => 1, "igel_mu" => 2,
  "igel_sobol_init" => false, "search_mode" => "igelmo", "n_reps" => 1, "w_scope" => wscope))
