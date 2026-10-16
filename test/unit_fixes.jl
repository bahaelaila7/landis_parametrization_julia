# Regression tests for the Step-1 bug fixes (/workspace/pan_refactor/BUGS.md). One @testset per fix, named by its bug id.
module UnitFixes
using Test, Pan

@testset "B21 dual_mode: one parser, Bool accepted" begin
  pm = Pan._parse_dual_mode
  @test pm(false) === :off && pm(nothing) === :off && pm("off") === :off && pm("false") === :off && pm("OFF") === :off
  @test pm(true) === :joint && pm("joint") === :joint && pm("true") === :joint
  @test pm("b") === :b && pm("B") === :b
  @test_throws ErrorException pm("bogus")
  # the disturbance-shock config reader used to call String(::Bool) on `dual_mode: true`
  @test (Pan._disturbance_shock_path(Dict{String,Any}("dual_mode" => true, "loss_before_injection" => false)); true)
end

@testset "B1 injection flags do not leak between configs" begin
  # the options are immutable values built from the config alone: a config without the keys gets the defaults, whatever ran before it
  on = Pan.scoring_options(Dict{String,Any}("override_injection" => true, "override_injection_sync" => true))
  @test on.override && on.sync
  off = Pan.scoring_options(Dict{String,Any}("loss_before_injection" => true))                 # no injection keys ⇒ the schema defaults
  @test !off.override && !off.sync && !off.replace && off.disturbance === :off
  @test Pan._disturbance_shock_path(Dict{String,Any}("loss_before_injection" => true)) == ""   # no sync in this config ⇒ no shock
  @test Pan._disturbance_shock_path(Dict{String,Any}("override_injection" => true, "override_injection_sync" => true)) != ""
  @test Pan.data_options(Dict{String,Any}("loss_before_injection" => true)).disturbance_path == "" && Pan.scoring_options(Dict{String,Any}("fia_cycle" => true, "fia_cycle_merge" => 2)).cycle == Pan.Data.CycleOptions(use_fia_cycle=true, merge=2)
  @test !isdefined(Pan, :OVERRIDE_INJECTION) && !isdefined(Pan, :OVERRIDE_INJECTION_SYNC)      # no module-level flag left to leak
end


@testset "B7 cbal_pct RANKW: a stratum without training cohorts gets weight 0, not NaN" begin
  PU = Pan.PU; FT = Pan.PanCore.FloatType
  lp = PU.LossParams(age_bins=PU.AgeBins(bins_idx=[10, 20], last_bin_open=true), smoothing_weights=FT[1], rankw_mode=:cbal_pct)
  nc = zeros(Int, 4, 3); nc[1, 1] = 40; nc[2, 1] = 5; nc[3, 3] = 12   # eco 2 has species but no cohorts
  esids = [[1, 2], [1, 4], [3]]
  R = Pan._rankw(lp, zeros(FT, 4, 3), esids; split_size=[10, 0, 4], ncoh=nc)
  @test !any(isnan, R)
  @test all(iszero, R[:, 2])
  @test R[1, 1] > 0 && R[2, 1] > 0 && R[3, 3] > 0
  @test sum(R) ≈ 1
end


@testset "B5 unknown search_mode is an error, not a silent LBSA run" begin
  @test Pan._check_search_mode("igelmo") === nothing && Pan._check_search_mode("sobol") === nothing
  @test all(m -> Pan._check_search_mode(m) === nothing, Pan.SEARCH_MODES)
  @test_throws ErrorException Pan._check_search_mode("igel_mo")
  @test_throws ErrorException Pan._check_search_mode("sa")
end


@testset "B14 loss_before_injection is defined by the injection mode" begin
  r(d) = (dd = Dict{String,Any}(d); Pan._resolve_loss_before_injection((k, dflt) -> get(dd, k, dflt)))   # the resolver on raw values (load_config calls it; see unit_config_reader.jl)
  sync = ("override_injection" => true, "override_injection_sync" => true)
  @test r([sync...]) === true                                            # the 2026-10-15 rule
  @test r([sync..., "loss_before_injection" => false]) === false         # legacy order on request
  @test r(["override_injection" => true]) === false                      # append mode: legacy order by default
  @test r([sync..., "override_injection_replace" => true]) === false     # replace mode: legacy order by default
  @test_throws ErrorException r(["override_injection" => true, "loss_before_injection" => true])
  @test_throws ErrorException r([sync..., "override_injection_replace" => true, "loss_before_injection" => true])
  @test r(Pair{String,Any}[]) === true                                   # no injection: flag has no effect, keep the default
end


@testset "B11 a scale matrix that does not cover a cell is an error, not 0" begin
  FT = Pan.PanCore.FloatType
  M = FT[1 2; 3 4]
  @test Pan._cell(M, 2, 1) == 3
  @test_throws ErrorException Pan._cell(M, 3, 1)
  @test_throws ErrorException Pan._cell(zeros(FT, 0, 0), 1, 1)
end


@testset "B12 injection state is owned by the reference's InjectionPlan, not by global caches" begin
  @test !isdefined(Pan, :_EXCL_CACHE) && !isdefined(Pan, :_SHOCK_REGISTRY) && !isdefined(Pan, :_EXCL_LOCK) && !isdefined(Pan, :_clear_run_caches!)
  FT = Pan.FloatType; UI = Pan.UIntType
  esids = [[1, 2]]
  lp = Pan.PU.LossParams(age_bins=Pan.PU.AgeBins(bins_idx=collect(1:100), last_bin_open=true), smoothing_weights=FT[one(FT)])
  mk(cohs) = Pan.make_sites_from_communities(Pan.DataFrame(mapcode=fill(1, length(cohs)), eco_id=fill(1, length(cohs)), eco_species_id=[c[1] for c in cohs],
                                                            age_calc=[c[2] for c in cohs], agb_sum=[c[3] for c in cohs]), esids, Pan.Random.Xoshiro(1))
  ref = mk([(1, 40, 100.0), (1, 20, 80.0)])
  inj(drop) = Pan.DataFrame(plot_id=[1, 1, 1, 1], sim_year=[2, 2, 5, 5], eco_species_id=[1, 1, 1, 1], age_calc=[37, 17, 40, 20],
                            agb_sum=[90.0, 60.0, 100.0, 80.0], disturbance_drop_pct=[0.0, 0.0, 0.0, drop])
  sync = Pan.ScoringOptions(; override=true, sync=true, disturbance=:exclude_overwrite)
  @test Pan.build_injection_plan(nothing, ref, esids, lp, sync) === nothing
  p1 = Pan.build_injection_plan(inj(0.3), ref, esids, lp, sync)
  p2 = Pan.build_injection_plan(inj(0.0), ref, esids, lp, sync)
  @test p1.years == Set([2, 5]) && haskey(p1.dict, 5) && p1.shock === nothing && p1.legacy_excl === nothing
  @test p1.excl isa Pan._ExclPlan && p1.excl.coh[5][1] == [(UI(1), FT(20))]            # the disturbed cohort is excluded in the shock order
  @test !haskey(p2.excl.coh, 5)                                                       # a second reference has its own plan: nothing is shared or keyed by identity
  @test p1.dict !== p2.dict && p1.excl !== p2.excl
  # outside the shock order the cohort-level plan does not exist; the legacy order keeps the species-level exclusion
  legacy = Pan.with_options(sync; loss_before_injection=false)
  pl = Pan.build_injection_plan(inj(0.3), ref, esids, lp, legacy)
  @test pl.excl === nothing && pl.legacy_excl[5][1] == Set([UI(1)])
  # a plan built under one scoring rule is refused under another (its exclusions depend on it)
  @test Pan._check_plan_options(p1, sync) === nothing
  @test_throws ErrorException Pan._check_plan_options(p1, legacy)
  @test_throws ErrorException Pan._check_plan_options(p1, Pan.with_options(sync; disturbance=:scale))
  @test_throws ErrorException Pan._check_plan_options(p1, Pan.with_options(sync; exclude_recruits_at_entry=false))
end

@testset "B31 death term: the score-only PROB_MORT is zeroed on the parameters, not on the call's environment" begin
  bp = (PROB_MORT_SPP=[[0.1, 0.2], [0.3]], B_MAX=[1.0, 2.0])
  z = Pan._without_mortality(bp)
  @test z.PROB_MORT_SPP == [[0.0, 0.0], [0.0]] && z.B_MAX == bp.B_MAX && bp.PROB_MORT_SPP[1][1] == 0.1   # inputs untouched
end

end
