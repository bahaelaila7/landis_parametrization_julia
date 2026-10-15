# Context-based checks for tools/simab_eval.jl (bounds, mortality, establishment, identifiability). The data context is prepared by run_from_yaml
# exactly as in training (all flags of the config active), then the checks run in the POST_SEARCH_HOOK.
#   ./julia_gdal.sh --project=. tools/simab_eval_ctx.jl <A|B> <config.yml> <params.jld2> <out.jld2> [archive_dir]
using Pan, JLD2, CSV, DataFrames, Statistics
const P = Pan; const PU = P.PU
mode, cfgpath, ppath, outp = ARGS[1], ARGS[2], ARGS[3], ARGS[4]; arch_dir = length(ARGS) >= 5 ? ARGS[5] : ""
cfg = Pan.load_config(cfgpath)
fam_name(s) = s === :ANPP_MAX_SPP ? "ANPP_ratio" : s === :B_MAX_SPP ? "B_MAX" : String(s)
function check(d)
  SP = P.succession_plugin_of(d.ref_soa); p = JLD2.load_object(ppath)
  pd = P.make_param_dists(SP, d.n_species, length(d.eco_list), d.eco_species_ids; no_establishment=cfg["no_establishment"], fit_establishment=false)
  slots = PU.build_slots(pd, p); u = PU.params_to_u(p, pd, slots)
  res = Dict{String,Any}("mode" => mode)
  # ---- 2. bound hugging per family: fraction of fitted dimensions with u <= 0.01 or >= 0.99
  fam = Dict{String,Vector{Float64}}()
  for (i, (pi, t)) in enumerate(slots); push!(get!(fam, fam_name(pd.params[pi].name), Float64[]), u[i]); end
  res["bounds"] = Dict(f => (n=length(v), hug=count(x -> x <= 0.01 || x >= 0.99, v), lo=count(<=(0.01), v), hi=count(>=(0.99), v)) for (f, v) in fam)
  # ---- final PROB_MORT table with labels (implied annual death rate = 1-(1-p)^1 = p)
  rows = NamedTuple[]
  for (e, ids) in enumerate(d.eco_species_ids), (j, gsp) in enumerate(ids)
    push!(rows, (eco=String(d.eco_list[e]), species=String(d.species_list[gsp]), pmort=hasproperty(p, :PROB_MORT_SPP) ? Float64(p.PROB_MORT_SPP[e][j]) : NaN,
                 pestab=hasproperty(p, :PROB_ESTAB_SPP) ? Float64(p.PROB_ESTAB_SPP[e][j]) : NaN))
  end
  res["cells"] = DataFrame(rows)
  if mode == "A"
    # ---- 3. EMPIRICAL natural-death rate per (eco, species) on train: same cohort/interval definitions as the death term (Sim A run at the params)
    idict = P._build_injection_dict(d.injection_cohorts, d.ref_soa); iy = Set(Int.(d.injection_cohorts.sim_year))
    msy = maximum(maximum.(filter(!isempty, d.site_sim_years.sim_years)))
    acc = P.DeathAcc(d.eco_species_ids)
    P.fit_params(d.ref_soa, p, msy, d.n_species, d.eco_species_ids, d.spdf_plts, d.site_sim_years, false, d.spinup_cohorts, d.loss_params;
      debug=false, search_tier=3, injection_dict=idict, injection_years=iy, seeds=[UInt64(1)], death_acc=acc)
    grid = exp.(range(log(1e-6), log(0.5); length=400)); erow = NamedTuple[]
    for (e, ids) in enumerate(d.eco_species_ids), (j, gsp) in enumerate(ids)
      nd = sum(acc.dhist[e][j]); ns = acc.n_surv[e][j]; expo = acc.surv_dt[e][j] + sum(k * acc.dhist[e][j][k] for k in 1:P.DEATH_MAXDT)
      n = ns + nd; n == 0 && continue
      vals = [P.death_cell(acc, e, j, q)[1] for q in grid]
      push!(erow, (eco=String(d.eco_list[e]), species=String(d.species_list[gsp]), n_cohort_intervals=n, n_deaths=nd, exposure_yr=expo, p_emp=nd == 0 ? 0.0 : grid[argmin(vals)]))
    end
    res["empirical"] = DataFrame(erow)
    # ---- 6. identifiability from an archive dir (behavioural points: F = max of the ref-normalised objectives <= 1.5 x knee's F)
    if !isempty(arch_dir) && isfile(joinpath(arch_dir, "mo_archive.csv"))
      A = CSV.read(joinpath(arch_dir, "mo_archive.csv"), DataFrame)
      Fv = [maximum(skipmissing([r.A_rel, r.W_rel, hasproperty(A, :D_rel) ? r.D_rel : missing])) for r in eachrow(A)]
      kn = findfirst(r -> occursin("knee", String(coalesce(r.role, ""))), eachrow(A)); kn === nothing && (kn = argmin(Fv))
      ok = findall(<=(1.5 * Fv[kn]), Fv); pts = [JLD2.load_object(joinpath(arch_dir, "mo_archive", A.file[i])) for i in ok]
      spread = Dict{String,Any}()
      for (f, fld) in (("S", :S), ("B_MAX", :B_MAX_SPP), ("ANPP", :ANPP_MAX_SPP), ("PROB_MORT", :PROB_MORT_SPP))
        hasproperty(p, fld) || continue
        vals = [[Float64(x) for v in getproperty(q, fld) for x in v] for q in pts]
        m = reduce(hcat, vals); fl = f == "PROB_MORT" ? 1e-3 : 1e-9
        r = [maximum(m[i, :]) / max(minimum(m[i, :]), fl) for i in 1:size(m, 1) if maximum(m[i, :]) > fl]
        isempty(r) || (spread[f] = (npoints=length(ok), ndims=length(r), median_ratio=median(r), frac_gt5=mean(r .> 5)))
      end
      lv = [Float64.(q.D) for q in pts if hasproperty(q, :D)]
      if !isempty(lv); m = reduce(hcat, lv); r = [maximum(m[i, :]) / minimum(m[i, :]) for i in 1:size(m, 1)]; spread["D"] = (npoints=length(ok), ndims=length(r), median_ratio=median(r), frac_gt5=mean(r .> 5)); end
      res["identifiability"] = spread; res["archive_F_knee"] = Fv[kn]
    end
  else
    # ---- 4. establishment: data window [mult x data, 1] and share of cells pinned at the lower bound
    fl = PU.PROB_ESTAB_FLOOR[]
    if fl !== nothing && hasproperty(p, :PROB_ESTAB_SPP)
      inwin = 0; pinned = 0; n = 0
      for ((e, j), f) in fl
        n += 1; v = Float64(p.PROB_ESTAB_SPP[e][j]); v >= Float64(f) - 1e-4 && (inwin += 1)
        (v - Float64(f)) / max(1.0 - Float64(f), 1e-6) <= 0.01 && (pinned += 1)
      end
      res["estab"] = (cells_with_data=n, in_window=inwin, pinned_low=pinned, total_cells=sum(length(x) for x in d.eco_species_ids))
    end
  end
  JLD2.save_object(outp, res)
  println("ctx($mode) done → $outp")
end
P.POST_SEARCH_HOOK[] = check
P.run_from_yaml(cfgpath; overrides=Dict{String,Any}("output_dir" => joinpath(dirname(outp), "_ctx_$(mode)"), "trials" => 1, "igel_mu" => 2, "igel_sobol_init" => false,
  "search_mode" => "igelmo", "n_reps" => 1, "start_from" => ppath))
