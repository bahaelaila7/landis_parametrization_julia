# Is the departure of W from its floor a real signal? For every archive point of a Sim A MO run, evaluate W (Σ per-record
# sp_w_loss, the training loss settings: unbinned W1, hinge, softplus, count balance, w_scope as given) and A (mean
# |log1p Σsim − log1p Σobs| over (plot, visit, eco-species) pairs) on TRAIN and on VAL, then compare the rankings.
#   ./julia_gdal.sh --project=. --threads=8 runs_local/simab_analysis/w_signal.jl <config.yml> <archive_dir> <out.csv> [w_scope]
using Pan, JLD2, CSV, DataFrames, Statistics, Printf
import YAML
const P = Pan; const PU = P.PU; const D = P.Data; const FT = P.FloatType
cfgpath, adir, outcsv = ARGS[1], ARGS[2], ARGS[3]; wscope = length(ARGS) >= 4 ? ARGS[4] : "all"
cfg = YAML.load_file(cfgpath)
arch = CSV.read(joinpath(adir, "mo_archive.csv"), DataFrame)
function run(d)
  P.PARALLEL_SITES[] = true
  lp = d.loss_params; esids = d.eco_species_ids
  # VAL reference with the training loss settings (same split: prepare_from_cfg reproduces the run's split)
  _, eco_list_v, species_list_v, esids_v, splots_val, _ = P.prepare_from_cfg(cfg)
  esids_v == esids && species_list_v == d.species_list || error("val split species/eco layout differs from training")
  SPc = P.setup_plugin_from_cfg(cfg, d.eco_list; species_list=d.species_list, eco_species_ids=esids)
  function mkctx(sp, ref_soa)
    spdf = PU.smoothen_ref_years(sp, lp, Int(maximum(sp.age_calc)); debug=false)
    spdf_plts = D.make_spdf_dict(spdf, esids); ssy = D.get_site_sim_years(spdf); spc = D.get_spinup_cohorts(sp)
    inj = D.get_injection_cohorts(sp; all_cohorts=true)
    rs = ref_soa === nothing ? P.make_sites_like(SPc, sp, esids; rng=P.RNGType(1), spinup=false, no_establishment=true) : ref_soa
    (; rs, spdf_plts, ssy, spc, inj, idict=P._build_injection_dict(inj, rs), iy=Set(Int.(inj.sim_year)),
       msy=maximum(maximum.(filter(!isempty, ssy.sim_years))))
  end
  ctxs = ("train" => (; rs=d.ref_soa, spdf_plts=d.spdf_plts, ssy=d.site_sim_years, spc=d.spinup_cohorts, inj=d.injection_cohorts,
                         idict=P._build_injection_dict(d.injection_cohorts, d.ref_soa), iy=Set(Int.(d.injection_cohorts.sim_year)),
                         msy=maximum(maximum.(filter(!isempty, d.site_sim_years.sim_years)))),
          "val" => mkctx(splots_val, nothing))
  function evalp(p, c)
    r = P.fit_params(c.rs, p, c.msy, d.n_species, esids, c.spdf_plts, c.ssy, false, c.spc, lp;
      debug=false, search_tier=3, injection_dict=c.idict, injection_years=c.iy, seeds=[UInt64(1)], cache_preinject=true)
    W = sum(sum(el.sp_w_loss) for el in r[1][3])
    sim = Dict{Tuple{Int,Int,Int},Float64}()
    for (cn, yr, sp, age, bio) in r[1][2]; yr > 0 || continue; k = (Int(cn), Int(yr), Int(sp)); sim[k] = get(sim, k, 0.0) + bio; end
    obs = Dict{Tuple{Int,Int,Int},Float64}()
    for row in eachrow(c.inj); row.sim_year > 0 || continue; k = (Int(row.plot_id), Int(row.sim_year), Int(row.eco_species_id)); obs[k] = get(obs, k, 0.0) + row.agb_sum; end
    errs = [abs(log1p(sim[k]) - log1p(v)) for (k, v) in obs if haskey(sim, k)]
    (W=W, A=mean(errs))
  end
  rows = NamedTuple[]
  for r in eachrow(arch)
    p = JLD2.load_object(joinpath(adir, "mo_archive", r.file))
    tr = evalp(p, ctxs[1][2]); va = evalp(p, ctxs[2][2])
    push!(rows, (idx=r.idx, role=coalesce(r.role, ""), A_rel=r.A_rel, W_rel=r.W_rel, W_train=tr.W, A_train=tr.A, W_val=va.W, A_val=va.A))
    @printf("WSIG %3d W_train=%.2f W_val=%.2f A_train=%.4f A_val=%.4f\n", r.idx, tr.W, va.W, tr.A, va.A)
  end
  df = DataFrame(rows); CSV.write(outcsv, df)
  sp(x, y) = cor(invperm(sortperm(x)), invperm(sortperm(y)))
  @printf("WSIG Spearman train-vs-val: W %.3f   A %.3f   (n=%d)\n", sp(df.W_train, df.W_val), sp(df.A_train, df.A_val), nrow(df))
  for (nm, x) in (("W_train", df.W_train), ("W_val", df.W_val), ("A_train", df.A_train), ("A_val", df.A_val))
    @printf("WSIG %-8s min=%.4f max=%.4f  range/min=%.3f%%\n", nm, minimum(x), maximum(x), 100 * (maximum(x) - minimum(x)) / minimum(x))
  end
end
P.POST_SEARCH_HOOK[] = run
P.run_from_yaml(cfgpath; overrides=Dict{String,Any}("output_dir" => joinpath(dirname(outcsv), "_ctx_wsig"), "trials" => 1, "igel_mu" => 2,
  "igel_sobol_init" => false, "search_mode" => "igelmo", "n_reps" => 1, "w_scope" => wscope))
