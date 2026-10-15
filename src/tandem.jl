# In-tandem Sim A ⊕ Sim B (gdigel objective :tandem, 2026-10-16). Every parameter candidate is evaluated twice, each simulation
# set up as it would be on its own:
#   Sim A — the run's config (sync injection, scoring rule, per-year W, its cohort table); local per-plot errors.
#   Sim B — the Sim B cohort table (`tandem_b.tablename`, timber-only), its own coarse tier-4 age bins (`tandem_b.bins_idx`), free
#           establishment + observed disturbance (simB_* keys of the config), its own loss scales; aggregate (species × stratum ×
#           FIA cycle) errors.
# Sim B uses EXACTLY Sim A's plot split and is relabelled into Sim A's species/eco layout (species by symbol through Sim A's
# species map, eco cell by plot), so one parameter vector and one set of eco×species indices serve both simulations.

"Sim B cohort rows (table `tablename`) for the plots of `a_splots`, aggregated like Data.make_splots and labelled in Sim A's layout."
function _tandem_b_splots(db_path::AbstractString, tablename::AbstractString, a_splots::DataFrame, species_list::Vector{String},
                          eco_species_ids::Vector{Vector{Int}}, spmap::DataFrame)
  keys4 = [:statecd, :unitcd, :countycd, :plot]
  peco = Dict{NTuple{4,Int},Int}((Int(r.statecd), Int(r.unitcd), Int(r.countycd), Int(r.plot)) => Int(r.eco_id)
                                 for r in eachrow(unique(select(a_splots, vcat(keys4, [:eco_id])))))
  db = DuckDB.DB(); con = DuckDB.connect(db)
  DuckDB.execute(con, "ATTACH '$(db_path)' AS s (READ_ONLY)")
  pk = DataFrame(statecd=[k[1] for k in keys(peco)], unitcd=[k[2] for k in keys(peco)], countycd=[k[3] for k in keys(peco)], plot=[k[4] for k in keys(peco)])
  DuckDB.register_data_frame(con, pk, "tandem_plots")
  raw = DuckDB.execute(con, """
    SELECT t.plt_cn, t.statecd, t.unitcd, t.countycd, t.plot, t.measdate, t.species_symbol, t.age_calc, t.agb,
           coalesce(t.disturbance_drop_pct, 0.0) AS disturbance_drop_pct, t.cycle, t.intro_type
    FROM s.$(tablename) t JOIN tandem_plots USING (statecd, unitcd, countycd, plot)
    WHERE t.agb > 0 AND t.age_calc >= 1""") |> DataFrame
  close(db)
  sym2eff = Dict{String,String}(String(r.species_symbol) => String(r.effective_species) for r in eachrow(spmap))
  sid = Dict{String,Int}(s => i for (i, s) in enumerate(species_list))
  espmap = Dict{Tuple{Int,Int},Int}((e, Int(ids[j])) => j for (e, ids) in enumerate(eco_species_ids) for j in eachindex(ids))
  raw.eco_id = [peco[(Int(a), Int(b), Int(c), Int(d))] for (a, b, c, d) in zip(raw.statecd, raw.unitcd, raw.countycd, raw.plot)]
  raw.effective_species = [get(sym2eff, String(s), "") for s in raw.species_symbol]
  raw.species_id = [get(sid, e, 0) for e in raw.effective_species]
  n0 = nrow(raw)
  filter!(r -> r.species_id > 0 && haskey(espmap, (r.eco_id, r.species_id)), raw)
  nrow(raw) < n0 && @info "tandem Sim B: dropped $(n0 - nrow(raw)) of $n0 cohort rows whose species is not in Sim A's layout for that eco cell"
  raw.age_calc = Int.(raw.age_calc); raw.disturbance_drop_pct = Float32.(raw.disturbance_drop_pct)
  raw._phantom = Float32[coalesce(t, "") == "phantom_backfill" ? 1f0 : 0f0 for t in raw.intro_type]
  fields = [:plt_cn, :statecd, :unitcd, :countycd, :plot, :eco_id, :measdate, :species_id, :effective_species, :age_calc, :cycle]
  sp = combine(groupby(raw, fields, sort=false), nrow => :count, :agb => sum => :agb_sum,
    [:agb, :disturbance_drop_pct] => Data._weighted_cohort_drop => :disturbance_drop_pct,
    [:agb, :_phantom] => ((a, p) -> (t = sum(a); t > 0 ? Float32(sum(a .* p) / t) : 0f0)) => :phantom_frac)
  sm = combine(groupby(sp, keys4, sort=false)) do rows; (; start_measdate=[minimum(rows.measdate)]); end
  sp = sort!(innerjoin(sp, sm, on=keys4), [:measdate, :statecd, :unitcd, :countycd, :plot, :age_calc, :species_id])
  sp.sim_year = round.(Int, Dates.value.(Dates.Day.(sp.measdate .- sp.start_measdate)) ./ 365.25)
  sp.plot_id = groupindices(groupby(sp, keys4)) .|> UIntType
  sp.eco_species_id = UIntType[espmap[(e, s)] for (e, s) in zip(sp.eco_id, sp.species_id)]
  sp.eco_id = UIntType.(sp.eco_id); sp.species_id = UIntType.(sp.species_id)
  Data.mark_estab_year!(sp)
  if !isempty(Data.DISTURBANCE_FULL_PATH[])   # plot-level full removals (same export as Sim A), keyed in Sim A's layout
    shock_rows = Data._load_disturbance_rows(Data.DISTURBANCE_FULL_PATH[])
    sym2sid = Dict{String,Int}(s => sid[e] for (s, e) in sym2eff if haskey(sid, e))
    metadata!(sp, "disturbance_shock", Data._plot_shocks!(sp, shock_rows, sym2sid, espmap); style=:note)
  end
  @info "tandem Sim B: $(nrow(sp)) plot-level cohort rows on $(length(unique(sp.plot_id))) plots from $(tablename) (Sim A has $(length(peco)) plots)"
  sp
end

"Sim B context for one split: references with Sim B's own loss bins, the free-process dual_b, and its sim-year table."
function _tandem_b_context(b_splots::DataFrame, eco_list, eco_species_ids, lpB::PU.LossParams, cycle_years::Real, rng)
  spdf = PU.smoothen_ref_years(b_splots, lpB, Int(maximum(b_splots.age_calc)); debug=false)
  spdf_plts = Data.make_spdf_dict(spdf, eco_species_ids); ssy = Data.get_site_sim_years(spdf); spc = Data.get_spinup_cohorts(b_splots)
  inj = Data.get_injection_cohorts(b_splots; all_cohorts=true)
  cycle_map, n_cycles = Data.build_cycle_map(b_splots; cycle_years=Float64(cycle_years))
  nb = length(lpB.age_bins.bins_idx) + Int(lpB.age_bins.last_bin_open)
  t4_ref = [[zeros(FloatType, length(eco_species_ids[e]), nb) for _ in 1:n_cycles] for e in eachindex(eco_list)]
  for ((pid, eid), yd) in spdf_plts, (sy, gt) in yd
    cyc = get(cycle_map, (Int(pid), Int(sy)), 0); cyc == 0 && continue
    for (spe, rec) in gt.records; t4_ref[eid][cyc][spe, :] .+= diff([0f0; rec.sp_age_cdf]) .* rec.sp_agb_sum; end
  end
  dual_b = _build_dual_b(b_splots, eco_species_ids, eco_list, spdf_plts, lpB, t4_ref, cycle_map, n_cycles, inj, spc, rng, false; b_only=true)
  (; dual_b, spdf_plts, ssy, spc, lp=lpB, msy=maximum(maximum.(filter(!isempty, ssy.sim_years))))
end

"Set Sim B's tier-4 loss scales (B slots) for this context — serially, before a locked candidate batch."
_tandem_set_b_scales!(B, eco_species_ids, n_species) =
  _set_loss_scales!(B.dual_b.tier, B.dual_b.spdf_plts, B.dual_b.t4_ref, B.lp, eco_species_ids, n_species; cell_plots=B.dual_b.cell_plots)

"Run Sim B for `params` (free process, `nreps` stochastic reps from the run's fixed seed) → its per-eco tier-4 SiteLosses."
function _tandem_b_eval(B, params, n_species, eco_species_ids, seed::UInt64, nreps::Int)
  r = fit_params(B.dual_b.ref_soa, params, B.msy, n_species, eco_species_ids, B.spdf_plts, B.ssy, false, B.spc, B.lp;
    debug=false, search_tier=3, dual_b=B.dual_b, seeds=fill(seed, nreps))
  r[1][3]
end

"Per-group Sim B errors: ratio-of-sums over the group's cells of the tier-4 W and AGB terms with the Sim B (B-slot) scales."
function _tandem_group_b(eco_losses, eco_species_ids, groupcells)
  Ws = PU.W_SCALE_B[]; As = PU.AGB_SCALE_B[]
  G = length(groupcells); bw = fill(NaN, G); ba = fill(NaN, G)
  for g in 1:G
    nw = 0.0; dw = 0.0; na = 0.0; da = 0.0
    for (e, j) in groupcells[g]
      gsp = Int(eco_species_ids[e][j]); el = eco_losses[e]
      (size(Ws, 1) >= gsp && size(Ws, 2) >= e) || continue
      dw += Ws[gsp, e]; da += As[gsp, e]; nw += el.sp_w_loss[gsp]; na += el.sp_agb_loss[gsp]
    end
    dw > 0 && (bw[g] = nw / dw); da > 0 && (ba[g] = na / da)
  end
  bw, ba
end
