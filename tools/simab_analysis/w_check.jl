# Does the Sim A age-structure loss W behave as intended on the real pipeline?
# For every scored (site, visit T, species) record, evaluate Pan's own W kernel (forgive band → normalised CDF → weighted
# W1 → softplus, exactly as calculate_species_loss_acc!) against the reference Pan scores against, for three "sim" states:
#   model       the simulated cohorts at T (pre-injection snapshot), survivors scope applied as in the run
#   oracle      the OBSERVED cohorts at T (must give W ≈ 0 — otherwise the kernel or reference is broken)
#   persistence the observed cohorts of the previous visit, aged by Δ (uniform growth: what "no age-differential growth" scores)
#   ./julia_gdal.sh --project=. --threads=8 runs_local/simab_analysis/w_check.jl <config.yml> <params.jld2> <out_dir> [w_scope]
using Pan, JLD2, Printf, Statistics, DataFrames, CSV
const P = Pan; const PU = P.PU; const FT = P.FloatType; const UT = P.UIntType
cfgpath, ppath, outd = ARGS[1], ARGS[2], ARGS[3]; wscope = length(ARGS) >= 4 ? ARGS[4] : "survivors"
function check(d)
  P.PARALLEL_SITES[] = true
  p = JLD2.load_object(ppath)
  idict = P._build_injection_dict(d.injection_cohorts, d.ref_soa); iy = Set(Int.(d.injection_cohorts.sim_year))
  msy = maximum(maximum.(filter(!isempty, d.site_sim_years.sim_years)))
  r = P.fit_params(d.ref_soa, p, msy, d.n_species, d.eco_species_ids, d.spdf_plts, d.site_sim_years, false, d.spinup_cohorts, d.loss_params;
    debug=false, search_tier=3, injection_dict=idict, injection_years=iy, seeds=[UInt64(1)], cache_preinject=true)
  state = r[1][2]                                   # (ref_cn, year, local sp, age, bio) at every measurement year, pre-injection
  plan = P._get_excl_plan(idict, d.ref_soa, d.eco_species_ids, d.loss_params)
  lp = d.loss_params; bw = lp.age_bins.bin_widths; nb = length(bw)
  # index sim cohorts by (ref_cn, year)
  simby = Dict{Tuple{Int,Int},Vector{Tuple{UT,Int,FT}}}()
  for (cn, yr, sp, age, bio) in state; push!(get!(simby, (Int(cn), Int(yr)), Tuple{UT,Int,FT}[]), (UT(sp), Int(age), FT(bio))); end
  site_years = Dict{Int,Vector{Int}}()
  for (yr, sites) in idict, (si, _) in sites; push!(get!(site_years, si, Int[]), yr); end
  for v in values(site_years); sort!(v); end
  obsby = Dict{Tuple{Int,Int},Vector{Tuple{UT,Int,FT}}}()
  for (yr, sites) in idict, (si, cohorts) in sites; obsby[(si, yr)] = [(UT(sp), Int(a), FT(b)) for (sp, a, b, _) in cohorts]; end
  ages = zeros(FT, 2000)
  function wrec(cs, rec, gsp, eco)          # Pan's W for one record from a cohort list (sp-filtered)
    fill!(ages, zero(FT)); for (_, a, b) in cs; ages[a] += b; end
    cdf = PU.smoothen_bin_cdf_forgive(ages, rec.sp_age_agb; age_bins=lp.age_bins)
    raw = PU._w1_sum(rec.sp_age_cdf, cdf, bw, gsp, eco)
    PU._w_finish(raw, gsp, eco), raw
  end
  rows = NamedTuple[]
  for (si, ys) in site_years
    site = P.getsite(d.ref_soa, si); cn = Int(site.ref_cn); eco = Int(site.eco_id)
    gtyrs = get(d.spdf_plts, (UT(cn), UT(eco)), nothing); gtyrs === nothing && (gtyrs = get(d.spdf_plts, (cn, eco), nothing))
    gtyrs === nothing && continue
    for (k, yr) in enumerate(ys)
      k == 1 && continue                                   # year 0 = initial state, not scored
      py = ys[k-1]; dlt = yr - py
      gt = get(gtyrs, yr, nothing); gt === nothing && continue
      ex = (yd = get(plan.coh, yr, nothing); yd === nothing ? nothing : get(yd, si, nothing))
      altref = ex === nothing ? nothing : get(get(plan.ref, yr, Dict()), si, nothing)
      G = altref === nothing ? gt : altref
      obsT = filter(c -> ex === nothing || !any(t -> t[1] == c[1] && t[2] == c[2], ex), obsby[(si, yr)])
      simT = filter(c -> ex === nothing || !any(t -> t[1] == c[1] && t[2] == c[2], ex), get(simby, (cn, yr), Tuple{UT,Int,FT}[]))
      if wscope == "survivors"
        oset = Set((c[1], c[2]) for c in obsT); simT = filter(c -> (c[1], c[2]) in oset, simT)
      end
      prev = [(c[1], c[2] + dlt, c[3]) for c in obsby[(si, py)]]
      oset_all = Set((c[1], c[2]) for c in obsT)
      prev = filter(c -> (c[1], c[2]) in oset_all, prev)            # persistence on the survivors only (same cohorts as obs)
      pbio = Dict((c[1], c[2]) => c[3] for c in prev)
      for (sp, rec) in G.records
        G.keys[Int(sp)] || continue
        gsp = Int(d.eco_species_ids[eco][Int(sp)])
        fsp(v) = filter(c -> c[1] == sp, v)
        so, oo, po = fsp(simT), fsp(obsT), fsp(prev)
        isempty(oo) && continue
        wm, rm = isempty(so) ? (NaN, NaN) : wrec(so, rec, gsp, eco)
        wo, ro = wrec(oo, rec, gsp, eco)
        wp, rp = isempty(po) ? (NaN, NaN) : wrec(po, rec, gsp, eco)
        mage(v) = (b = sum(c[3] for c in v; init=0f0); b > 0 ? sum(c[2] * c[3] for c in v) / b : NaN)
        decl = any(c -> haskey(pbio, (c[1], c[2])) && c[3] < pbio[(c[1], c[2])], oo)
        push!(rows, (si=si, yr=yr, sp=Int(sp), gsp=gsp, eco=eco, n_obs=length(oo), n_sim=length(so), decline=decl, visit=k - 1, t=yr, mage_sim=mage(so), mage_obs=mage(oo), mage_prev=mage(po),
                     W_model=wm, W_oracle=wo, W_persist=wp, raw_model=rm, raw_oracle=ro, raw_persist=rp))
      end
    end
  end
  df = DataFrame(rows); mkpath(outd); CSV.write(joinpath(outd, "w_check_records.csv"), df)
  ok = filter(r -> !isnan(r.W_model) && !isnan(r.W_persist), df)
  @printf("WCHK records=%d (with sim and persistence: %d)\n", nrow(df), nrow(ok))
  @printf("WCHK ΣW oracle=%.3f  (max per record %.4g; records with W_oracle>1e-3: %d)\n", sum(df.W_oracle), maximum(df.W_oracle), count(>(1e-3), df.W_oracle))
  @printf("WCHK ΣW model=%.2f  ΣW persistence=%.2f  (same records)\n", sum(ok.W_model), sum(ok.W_persist))
  @printf("WCHK records model<persist: %.1f%%  model>persist: %.1f%%  equal: %.1f%%\n", 100mean(ok.W_model .< ok.W_persist .- 1e-6),
          100mean(ok.W_model .> ok.W_persist .+ 1e-6), 100mean(abs.(ok.W_model .- ok.W_persist) .<= 1e-6))
  m = filter(r -> r.n_obs >= 2, ok)
  @printf("WCHK multi-cohort records=%d  ΣW model=%.2f persist=%.2f  single-cohort records ΣW model=%.4f\n", nrow(m), sum(m.W_model), sum(m.W_persist),
          sum(filter(r -> r.n_obs == 1, ok).W_model))
  @printf("WCHK corr(W_model, W_persist) over records = %.3f\n", cor(ok.W_model, ok.W_persist))
  dd = filter(r -> r.decline, m); nd = filter(r -> !r.decline, m)
  @printf("WCHK multi-cohort records WITH a declining cohort: n=%d  ΣW model=%.2f (%.1f%% of model W)  persist=%.2f\n", nrow(dd), sum(dd.W_model), 100sum(dd.W_model)/sum(m.W_model), sum(dd.W_persist))
  @printf("WCHK multi-cohort records without declines:        n=%d  ΣW model=%.2f  persist=%.2f\n", nrow(nd), sum(nd.W_model), sum(nd.W_persist))
end
P.POST_SEARCH_HOOK[] = check
P.run_from_yaml(cfgpath; overrides=Dict{String,Any}("output_dir" => joinpath(outd, "_ctx"), "trials" => 1, "igel_mu" => 2, "igel_sobol_init" => false,
  "search_mode" => "igelmo", "n_reps" => 1, "start_from" => ppath, "w_scope" => wscope,
  "override_injection_replace" => (length(ARGS) >= 5 && ARGS[5] == "replace")))
