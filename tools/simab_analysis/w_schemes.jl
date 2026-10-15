# Which visits/cohorts should the Sim A age-structure term W score? Three schemes, same kernel as the Sim A loss
# (forgive band → normalised CDF → weighted W1 → softplus), survivors scope, cohort-level exclusion plan:
#   all       every scored visit of every plot (current Sim A loss)
#   plotlast  only each plot's last visit
#   plotmean  every visit, but each plot's W averaged over its scored visits (equal weight per plot, all anchors kept)
#   cohlast   each cohort (plot, species, birth year) only at its LAST eligible observation: at visit T a record contains only the
#             cohorts that are not observed again at the next visit (or T is the plot's last visit); reference and sim both
#             restricted to those cohorts.
# The reference CDFs are rebuilt from the observed cohort subsets (checked against Pan's own references for scheme `all`).
# Outputs: per-record table for the evaluation point (imbalance by record length) and, over a Sim A MO archive, train-vs-val
# Spearman of each scheme (does its departure from the floor rank solutions reproducibly?).
#   ./julia_gdal.sh --project=. --threads=4 w_schemes.jl <config.yml> <eval_params.jld2> <archive_dir> <out_dir> [w_scope]
using Pan, JLD2, CSV, DataFrames, Statistics, Printf
import YAML
const P = Pan; const PU = P.PU; const D = P.Data; const FT = P.FloatType; const UT = P.UIntType
cfgpath, evalp_path, adir, outd = ARGS[1:4]; wscope = length(ARGS) >= 5 ? ARGS[5] : "survivors"
cfg = YAML.load_file(cfgpath); mkpath(outd)
arch = CSV.read(joinpath(adir, "mo_archive.csv"), DataFrame)
const SCHEMES = (:all, :plotlast, :cohlast, :plotmean)

function run(d)
  P.PARALLEL_SITES[] = true
  lp = d.loss_params; esids = d.eco_species_ids; bw = lp.age_bins.bin_widths
  _, _, species_list_v, esids_v, splots_val, _ = P.prepare_from_cfg(cfg)
  (esids_v == esids && species_list_v == d.species_list) || error("val split layout differs from training")
  SPc = P.setup_plugin_from_cfg(cfg, d.eco_list; species_list=d.species_list, eco_species_ids=esids)
  function mkctx(sp)
    spdf = PU.smoothen_ref_years(sp, lp, Int(maximum(sp.age_calc)); debug=false)
    spdf_plts = D.make_spdf_dict(spdf, esids); ssy = D.get_site_sim_years(spdf); spc = D.get_spinup_cohorts(sp)
    inj = D.get_injection_cohorts(sp; all_cohorts=true)
    rs = P.make_sites_like(SPc, sp, esids; rng=P.RNGType(1), spinup=false, no_establishment=true)
    (; rs, spdf_plts, ssy, spc, inj, idict=P._build_injection_dict(inj, rs), iy=Set(Int.(inj.sim_year)), msy=maximum(maximum.(filter(!isempty, ssy.sim_years))))
  end
  ctx_train = (; rs=d.ref_soa, spdf_plts=d.spdf_plts, ssy=d.site_sim_years, spc=d.spinup_cohorts, inj=d.injection_cohorts,
               idict=P._build_injection_dict(d.injection_cohorts, d.ref_soa), iy=Set(Int.(d.injection_cohorts.sim_year)),
               msy=maximum(maximum.(filter(!isempty, d.site_sim_years.sim_years))))
  ctx_val = mkctx(splots_val)
  agesO = zeros(FT, 2000); agesS = zeros(FT, 2000)
  function wsub(so, oo, gsp, eco)        # W of sim cohorts `so` against observed cohorts `oo` (reference rebuilt from oo)
    fill!(agesO, zero(FT)); fill!(agesS, zero(FT))
    for (_, a, b) in oo; agesO[a] += b; end
    for (_, a, b) in so; agesS[a] += b; end
    ref_cdf = PU.smoothen_bin_cdf(agesO; w=lp.smoothing_weights, age_bins=lp.age_bins)
    obs_bin = PU.bin_ages(agesO; age_bins=lp.age_bins.bins_idx, last_bin_open=lp.age_bins.last_bin_open)
    sim_cdf = PU.smoothen_bin_cdf_forgive(agesS, obs_bin; age_bins=lp.age_bins)
    PU._w_finish(PU._w1_sum(ref_cdf, sim_cdf, bw, gsp, eco), gsp, eco)
  end
  function wpan(so, rec, gsp, eco)      # Pan's own reference record (sanity check for scheme :all)
    fill!(agesS, zero(FT)); for (_, a, b) in so; agesS[a] += b; end
    PU._w_finish(PU._w1_sum(rec.sp_age_cdf, PU.smoothen_bin_cdf_forgive(agesS, rec.sp_age_agb; age_bins=lp.age_bins), bw, gsp, eco), gsp, eco)
  end
  function score(p, c; records::Bool=false)
    r = P.fit_params(c.rs, p, c.msy, d.n_species, esids, c.spdf_plts, c.ssy, false, c.spc, lp;
      debug=false, search_tier=3, injection_dict=c.idict, injection_years=c.iy, seeds=[UInt64(1)], cache_preinject=true)
    plan = P._get_excl_plan(c.idict, c.rs, esids, lp)
    simby = Dict{Tuple{Int,Int},Vector{Tuple{UT,Int,FT}}}()
    for (cn, yr, sp, age, bio) in r[1][2]; push!(get!(simby, (Int(cn), Int(yr)), Tuple{UT,Int,FT}[]), (UT(sp), Int(age), FT(bio))); end
    site_years = Dict{Int,Vector{Int}}(); obsraw = Dict{Tuple{Int,Int},Vector{Tuple{UT,Int,FT}}}()
    for (yr, sites) in c.idict, (si, cohorts) in sites
      push!(get!(site_years, si, Int[]), yr); obsraw[(si, yr)] = [(UT(sp), Int(a), FT(b)) for (sp, a, b, _) in cohorts]
    end
    for v in values(site_years); sort!(v); end
    tot = Dict(s => 0.0 for s in SCHEMES); tot_pan = 0.0; ncoh = Dict(s => 0 for s in SCHEMES)
    rows = NamedTuple[]
    for (si, ys) in site_years
      site_sum = 0.0; site_nv = 0
      site = P.getsite(c.rs, si); cn = Int(site.ref_cn); eco = Int(site.eco_id)
      gtyrs = get(c.spdf_plts, (UT(cn), UT(eco)), nothing); gtyrs === nothing && continue
      for (k, yr) in enumerate(ys)
        k == 1 && continue
        gt = get(gtyrs, yr, nothing); gt === nothing && continue
        ex = (yd = get(plan.coh, yr, nothing); yd === nothing ? nothing : get(yd, si, nothing))
        alt = ex === nothing ? nothing : get(get(plan.ref, yr, Dict()), si, nothing); G = alt === nothing ? gt : alt
        keep(x) = ex === nothing || !any(t -> t[1] == x[1] && t[2] == x[2], ex)
        obsT = filter(keep, obsraw[(si, yr)]); oset = Set((x[1], x[2]) for x in obsT)
        simT = filter(x -> keep(x) && (wscope != "survivors" || (x[1], x[2]) in oset), get(simby, (cn, yr), Tuple{UT,Int,FT}[]))
        lastvisit = k == length(ys); site_nv += 1
        nxt = lastvisit ? nothing : Set((x[1], x[2]) for x in obsraw[(si, ys[k+1])]); dn = lastvisit ? 0 : ys[k+1] - yr
        for (sp, rec) in G.records
          G.keys[Int(sp)] || continue
          gsp = Int(esids[eco][Int(sp)])
          oo = filter(x -> x[1] == sp, obsT); so = filter(x -> x[1] == sp, simT)
          (isempty(oo) || isempty(so)) && continue
          w_all = wsub(so, oo, gsp, eco); tot[:all] += w_all; site_sum += w_all; ncoh[:all] += length(oo); tot_pan += wpan(so, rec, gsp, eco)
          w_pl = lastvisit ? w_all : 0.0; tot[:plotlast] += w_pl; lastvisit && (ncoh[:plotlast] += length(oo))
          ol = lastvisit ? oo : filter(x -> !((x[1], x[2] + dn) in nxt), oo)        # cohorts not seen again
          w_cl = 0.0
          if !isempty(ol)
            lset = Set((x[1], x[2]) for x in ol); sl = filter(x -> (x[1], x[2]) in lset, so)
            isempty(sl) || (w_cl = wsub(sl, ol, gsp, eco); ncoh[:cohlast] += length(ol))
          end
          tot[:cohlast] += w_cl
          records && push!(rows, (si=si, visit=k - 1, nvis=length(ys) - 1, sp=Int(sp), n=length(oo), n_last=length(ol), w_all=w_all, w_plotlast=w_pl, w_cohlast=w_cl))
        end
      end
      site_nv > 0 && (tot[:plotmean] += site_sum / site_nv)
    end
    (tot=tot, tot_pan=tot_pan, ncoh=ncoh, rows=rows)
  end
  # ---- 1. evaluation point: sanity + per-record table
  pe = JLD2.load_object(evalp_path)
  s = score(pe, ctx_train; records=true)
  @printf("WSCH sanity (train, eval point): ΣW all own-ref=%.3f  Pan-ref=%.3f\n", s.tot[:all], s.tot_pan)
  for sc in SCHEMES; @printf("WSCH eval point train: %-8s ΣW=%10.2f  scored cohort-obs=%d\n", sc, s.tot[sc], s.ncoh[sc]); end
  CSV.write(joinpath(outd, "records_evalpoint_train.csv"), DataFrame(s.rows))
  # ---- 2. archive: train vs val per scheme
  out = NamedTuple[]
  for r in eachrow(arch)
    p = JLD2.load_object(joinpath(adir, "mo_archive", r.file))
    tr = score(p, ctx_train); va = score(p, ctx_val)
    push!(out, (idx=r.idx, A_rel=r.A_rel, W_rel=r.W_rel, all_train=tr.tot[:all], all_val=va.tot[:all], plotlast_train=tr.tot[:plotlast],
                plotlast_val=va.tot[:plotlast], cohlast_train=tr.tot[:cohlast], cohlast_val=va.tot[:cohlast],
                plotmean_train=tr.tot[:plotmean], plotmean_val=va.tot[:plotmean]))
    @printf("WSCH arch %3d done\n", r.idx)
  end
  df = DataFrame(out); CSV.write(joinpath(outd, "archive_schemes.csv"), df)
  spear(x, y) = cor(invperm(sortperm(x)), invperm(sortperm(y)))
  for sc in SCHEMES
    a = df[!, Symbol("$(sc)_train")]; b = df[!, Symbol("$(sc)_val")]
    @printf("WSCH %-8s Spearman train-vs-val=%.3f  train range/min=%.2f%%  val range/min=%.2f%%  Spearman vs all(train)=%.3f\n", sc, spear(a, b),
            100(maximum(a) - minimum(a)) / minimum(a), 100(maximum(b) - minimum(b)) / minimum(b), spear(a, df.all_train))
  end
end
P.POST_SEARCH_HOOK[] = run
P.run_from_yaml(cfgpath; overrides=Dict{String,Any}("output_dir" => joinpath(outd, "_ctx"), "trials" => 1, "igel_mu" => 2,
  "igel_sobol_init" => false, "search_mode" => "igelmo", "n_reps" => 1, "w_scope" => wscope))
