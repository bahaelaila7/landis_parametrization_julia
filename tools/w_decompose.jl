# DIAGNOSTIC: decompose / sensitivity of A and W (new vs legacy order). See docs/mo_gdigel.md.
#   ./julia_gdal.sh --project=. --threads=8 tools/w_decompose.jl <config.yml> <start.jld2> <out_dir> [extra params jld2 ...]
using Pan, JLD2, Printf, Random, Statistics
const P = Pan; const PU = P.PU
cfgpath, ppath, outd = ARGS[1], ARGS[2], ARGS[3]; extra = ARGS[4:end]
function run(d)
  p0 = JLD2.load_object(ppath)
  sp = P.with_establishment(d.plugin.ctx, d.space; no_establishment=true, fit_establishment=false); slots = sp.slots   # the run's parameter space (pooling, floors) over the Sim A parameter subset
  cg, labels = P._gd_groups(sp.dists, slots, d.eco_species_ids, d.species_list, d.eco_list)
  ix = P._gd_index(d.injection_cohorts, d.splots, cg, labels, d.scoring; real_only=true)
  gc = P._gd_groupcells(ix.cell_group, ix.ngroups)
  injection = P.build_injection_plan(d.injection_cohorts, d.ref_soa, d.eco_species_ids, d.loss_params, d.scoring)
  msy = maximum(maximum.(filter(!isempty, d.site_sim_years.sim_years)))
  nvis = P._gd_cell_visits(d.spdf_plts, d.eco_species_ids, 1)
  println("DIAG groups=$(ix.ngroups) pairs=$(length(ix.pair_group))")
  sc = P.compute_loss_scales(d.loss_params, 3, d.spdf_plts, nothing, d.eco_species_ids, d.n_species; cbal=d.scale_seed.cbal, ncoh=d.scale_seed.ncoh, dom=d.scale_seed.dom, per_plot_mean=d.simb.t4_per_plot_mean)   # the reference's loss scales
  wd = P.WDiag(ctx=(d.spdf_plts, d.n_species, d.eco_species_ids, d.loss_params, sc))   # collector of the per-year cohort-set statistics
  function ev(p; lbi=true, mode=:none)
    fill!(wd.cat, 0)
    sco = P.with_options(d.scoring; loss_before_injection=lbi, w_diag=mode); fill!(wd.stats, 0)
    r = P.fit_params(d.ref_soa, p, msy, d.n_species, d.eco_species_ids, d.spdf_plts, d.site_sim_years, false, d.spinup_cohorts, d.loss_params;
      debug=false, search_tier=3, injection=injection, seeds=[UInt64(1)], cache_preinject=true, scales=sc, scoring=sco, simb=d.simb, plugin=d.plugin, w_diag=wd)
    L = P._gd_losses(ix, r[1][2])
    num, den = P._gd_cell_w(r[1][3], d.eco_species_ids, nvis, d.loss_params, sc)
    w = P._gd_group_w(num, den, gc, d.loss_params)
    raw = sum(sum(num[e]) for e in eachindex(num)); dn = sum(sum(den[e]) for e in eachindex(den))
    (A=sum(filter(!isnan, L.a)), W=sum(filter(!isnan, w)), rawnum=raw, den=dn, stats=copy(wd.stats), cat=copy(wd.cat))
  end
  function show(tag, r); @printf("DIAG %-34s SumA=%8.3f SumW=%9.3f  rawW=%11.2f  Wscale=%9.2f\n", tag, r.A, r.W, r.rawnum, r.den); end
  # ---- 1. decomposition at the start point (p0) and at the first archive point of the v2 smoke
  sets = Pair{String,Any}["start" => p0]
  for f in extra; push!(sets, basename(f) => JLD2.load_object(f)); end
  for (nm, p) in sets[1:min(2, end)]
    println("DIAG ===== decomposition: $nm")
    leg = ev(p; lbi=false);                 show("legacy order", leg)
    new = ev(p; mode=:stats);               show("new order", new)
    nd = ev(p; mode=:nodeath);              show("new, sim-only cohorts removed (oracle)", nd)
    orc = ev(p; mode=:oracle);              show("new, sim set := obs set (oracle shape)", orc)
    S = new.stats; C = new.cat
    @printf("DIAG   ref-only species records: no previous visit n=%d W=%.1f | species observed at PREVIOUS visit n=%d W=%.1f | (rest = new species at this site)\n", C[5, 2], C[5, 1], C[6, 2], C[6, 1])
    for (i, nm2) in enumerate(["both, same age set", "both, age sets differ (some overlap)", "ref only (species missing in sim)", "both, no age overlap"])
      @printf("DIAG   per-species-record W[%s]: n=%d  sumW(unnormalised)=%.1f  mean=%.2f\n", nm2, C[i, 2], C[i, 1], C[i, 2] > 0 ? C[i, 1] / C[i, 2] : 0.0)
    end
    @printf("DIAG   W(b) natural-death sim-only cohorts  = %9.2f (%.1f%% of new rawW)\n", new.rawnum - nd.rawnum, 100 * (new.rawnum - nd.rawnum) / new.rawnum)
    @printf("DIAG   W(c+d) obs-only cohorts+missing sp   = %9.2f (%.1f%%)\n", nd.rawnum - orc.rawnum, 100 * (nd.rawnum - orc.rawnum) / new.rawnum)
    @printf("DIAG   W(a)  matched-set shape (oracle)     = %9.2f (%.1f%%)\n", orc.rawnum, 100 * orc.rawnum / new.rawnum)
    @printf("DIAG   site-years=%d | matched n=%d simbio=%.4g obsbio=%.4g | sim-only n=%d bio=%.4g | obs-only n=%d bio=%.4g\n", S[15], S[1], S[2], S[3], S[4], S[5], S[6], S[7])
    @printf("DIAG   missing-species (obs sp w/o sim) cohorts=%d obsbio=%.4g | extra-species sim cohorts=%d bio=%.4g | sim-only with |age diff|=1 obs mate: %d | excluded sim n=%d obs bio=%.4g\n", S[8], S[9], S[10], S[11], S[14], S[12], S[13])
    @printf("DIAG   sim-only share of sim cohorts=%.1f%% of sim biomass=%.1f%% ; obs-only share of obs cohorts=%.1f%% of obs biomass=%.1f%%\n",
      100 * S[4] / (S[1] + S[4]), 100 * S[5] / (S[2] + S[5]), 100 * S[6] / (S[1] + S[6]), 100 * S[7] / (S[3] + S[7]))
  end
  # ---- 2. sensitivity
  println("DIAG ===== sensitivity (new order): components over parameter sets")
  nd_ = length(slots); rng = Random.Xoshiro(7)
  for k in 1:4; push!(sets, "uniform_u$k" => PU.u_to_params(rand(rng, nd_), sp, slots, p0)); end
  rows = NamedTuple[]; dead = Tuple[]
  for (nm, p) in sets
    n = ev(p; mode=:stats); o = ev(p; mode=:oracle); x = ev(p; mode=:nodeath)
    St = n.stats; push!(dead, (nm, St[4], St[5] / (St[2] + St[5]), St[5]))
    push!(rows, (nm=nm, A=n.A, W=n.W, rawW=n.rawnum, rawW_nodeath=x.rawnum, rawW_oracle=o.rawnum))
    @printf("DIAG %-26s SumA=%8.3f SumW=%9.3f | rawW new=%10.2f  nodeath=%10.2f  oracle(shape)=%10.2f | b=new-nodeath=%9.2f  c+d=nodeath-oracle=%9.2f\n", nm, n.A, n.W, n.rawnum, x.rawnum, o.rawnum, n.rawnum - x.rawnum, x.rawnum - o.rawnum)
  end
  for f in (:A, :W, :rawW, :rawW_nodeath, :rawW_oracle)
    v = [getfield(r, f) for r in rows]
    @printf("DIAG spread %-13s min=%10.2f max=%10.2f  (max-min)/mean=%.3f\n", String(f), minimum(v), maximum(v), (maximum(v) - minimum(v)) / mean(v))
  end
  for (nm, n, sh, b) in dead; @printf("DIAG dead %-26s sim-only cohorts=%d  biomass share of sim biomass=%.2f%%\n", nm, n, 100 * sh); end
  bs = [r.rawW - r.rawW_nodeath for r in rows]; cds = [r.rawW_nodeath - r.rawW_oracle for r in rows]
  @printf("DIAG spread b(natural death) min=%.2f max=%.2f ; c+d min=%.2f max=%.2f\n", minimum(bs), maximum(bs), minimum(cds), maximum(cds))
end
P.run_from_yaml(cfgpath; post_search_hook=run, overrides=Dict{String,Any}("output_dir" => outd, "trials" => 1, "igel_mu" => 2, "igel_sobol_init" => false, "search_mode" => "igelmo", "n_reps" => 1, "start_from" => ppath))
