# DIAGNOSTIC ONLY (default :none = no effect). ScoringOptions.w_diag at a measurement year, new order, just before scoring:
#   :stats   — accumulate cohort-set statistics sim-vs-obs into the WDiag collector (no state change)
#   :nodeath — additionally REMOVE sim cohorts that are absent from the observed set (oracle natural deaths)
#   :oracle  — run the sync injection BEFORE scoring (sim set := obs set; matched keep sim biomass): the shape-only W
# The statistics are collected into a `WDiag` that the caller passes to fit_params(...; w_diag=WDiag(...)) and reads afterwards (tools/w_decompose.jl);
# without one, the :nodeath / :oracle state changes still apply and the statistics go to a throw-away collector.
#   stats: 16 counters (see the S[...] updates below); cat: rows 1 both-exact 2 both-mismatched 3 ref-only(missing species) 4 both-but-nothing-matched
#   5/6 ref-only without / with a previous visit; cols ΣW, count. ctx (optional, for the per-species W categories of :stats):
#   (spdf_plts, n_species, eco_species_ids, loss_params, scales).
struct WDiag{C}
  stats::Vector{Float64}
  cat::Matrix{Float64}
  ctx::C
end
WDiag(; ctx=nothing) = WDiag(zeros(Float64, 16), zeros(Float64, 6, 2), ctx)
function _w_diag_apply!(pc, soa, year, injection_dict, plan, opts::ScoringOptions, wd::WDiag)
  mode = opts.w_diag
  S = wd.stats
  for (si, cohorts) in injection_dict[year]
    site = getsite(soa, si); site.active || continue
    exl = plan === nothing ? nothing : (yd = get(plan.coh, year, nothing); yd === nothing ? nothing : get(yd, si, nothing))
    isex(sp, age) = exl !== nothing && any(e -> e[1] == sp && e[2] == age, exl)
    csp = site.c_species; cage = site.c_age; cbio = site.c_bio
    S[15] += 1
    obs_sp = Set{UIntType}(); sim_sp = Set{UIntType}()
    for (sp, age, bio, _) in cohorts
      isex(sp, age) && (S[13] += bio; continue)
      push!(obs_sp, sp)
      found = false
      for j in 1:Int(site.live); csp[j] == sp && cage[j] == age && (found = true; break); end
      if found; S[1] += 1; S[3] += bio; else
        S[6] += 1; S[7] += bio
      end
    end
    for j in 1:Int(site.live)
      isex(csp[j], cage[j]) && (S[12] += 1; continue)
      push!(sim_sp, csp[j])
      inobs = false
      for (sp, age, _, _) in cohorts; (csp[j] == sp && cage[j] == age) && (inobs = true; break); end
      if inobs; S[2] += cbio[j]; else
        S[4] += 1; S[5] += cbio[j]
        for (sp, age, _, _) in cohorts   # age near-miss: same species, |age diff| == 1 present in obs
          (csp[j] == sp && abs(cage[j] - age) == 1) && (S[14] += 1; break)
        end
      end
    end
    for (sp, age, bio, _) in cohorts; isex(sp, age) && continue; sp in sim_sp || (S[8] += 1; S[9] += bio); end
    for j in 1:Int(site.live); isex(csp[j], cage[j]) && continue; csp[j] in obs_sp || (S[10] += 1; S[11] += cbio[j]); end
    if wd.ctx !== nothing && mode === :stats
      (spdf_plts, n_species, esids, lpar, lsc) = wd.ctx
      gt = spdf_plts[(site.ref_cn, site.eco_id)][year]; ecl = nothing
      if exl !== nothing; gt = plan.ref[year][si]; ecl = exl; end
      sl = PU.calculate_site_loss2(year, site, n_species, esids, gt, lpar, lsc; excl_c=ecl)
      smap = esids[Int(site.eco_id)]
      for spj in 1:length(gt.keys)
        gt.keys[spj] || continue
        w = Float64(sl.sp_w_loss[smap[spj]])
        simset = Set{Float64}(Float64(cage[j]) for j in 1:Int(site.live) if csp[j] == spj && !isex(csp[j], cage[j]))
        obsset = Set{Float64}(Float64(a) for (sp, a, _, _) in cohorts if sp == spj && !isex(sp, a))
        c = isempty(simset) ? 3 : (simset == obsset ? 1 : (isempty(intersect(simset, obsset)) ? 4 : 2))
        wd.cat[c, 1] += w; wd.cat[c, 2] += 1
        if c == 3   # why is the species absent in the sim? look at the site's previous observed visit
          py = -1
          for (y2, lst) in injection_dict; y2 < year && y2 > py && any(t -> t[1] == si, lst) && (py = y2); end
          if py < 0; wd.cat[5, 2] += 1; wd.cat[5, 1] += w   # no previous visit
          else
            prev = first(t[2] for t in injection_dict[py] if t[1] == si)
            if any(t -> t[1] == spj, prev); wd.cat[6, 2] += 1; wd.cat[6, 1] += w   # species WAS observed at the previous visit
            else; wd.cat[2, 1] += 0; end
          end
        end
      end
    end
    if mode === :nodeath
      j = 1
      while j <= Int(site.live)
        inobs = false
        for (sp, age, _, _) in cohorts; (csp[j] == sp && cage[j] == age) && (inobs = true; break); end
        inobs ? (j += 1) : _shock_remove_cohort!(site, j)
      end
    end
  end
  mode === :oracle && _inject_observed_cohorts!(pc, soa, injection_dict[year], opts; override=true, replace=false, sync=true, skip_scale=true)
  nothing
end
