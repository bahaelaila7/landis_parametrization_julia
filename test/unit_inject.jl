# Injection modes of the scoring rule (Step 4b): one function per mode, chosen once by a singleton type; build_reference's tables.
module UnitInject
using Test, Random, DataFrames, Dates, Pan
const P = Pan; const PU = P.PU
const UI = P.UIntType; const FT = P.FloatType
const PC0 = P.Plugins.BiomassSuccessionPlugin.BSTables()

# one site, species 1 and 2 of eco 1: (sp1,40) 100, (sp1,20) 80, (sp2,30) 60, (sp2,10) 50
function fresh()
  sp = DataFrame(mapcode=[1, 1, 1, 1], eco_id=[1, 1, 1, 1], eco_species_id=[1, 1, 2, 2], age_calc=[40, 20, 30, 10], agb_sum=[100.0, 80.0, 60.0, 50.0])
  soa = P.make_sites_from_communities(sp, [[1, 2]], Random.Xoshiro(1))
  s = P.getsite(soa, 1); s.B = sum(Float64(s.c_bio[j]) for j in 1:Int(s.live))
  soa
end
cohorts(soa) = (s = P.getsite(soa, 1); Dict((Int(s.c_species[j]), Int(s.c_age[j])) => Float64(s.c_bio[j]) for j in 1:Int(s.live)))
# observed: (sp1,40) 90 (matched), (sp2,10) 30 with drop 0.4 (matched), (sp1,5) 7 (recruit)
inj = P._SiteInjectionYear([(1, Tuple{UI,FT,FT,FT}[(UI(1), FT(40), FT(90), FT(0)), (UI(2), FT(10), FT(30), FT(0.4)), (UI(1), FT(5), FT(7), FT(0))])])

@testset "injection mode selection and typing" begin
  @test P._inject_mode(false, true, true) isa P.InjAppend            # no override => append, whatever the other flags say
  @test P._inject_mode(true, true, true) isa P.InjReplace            # replace beats sync
  @test P._inject_mode(true, false, true) isa P.InjSync
  @test P._inject_mode(true, false, false) isa P.InjOverride
  for M in (P.InjAppend, P.InjReplace, P.InjSync, P.InjOverride)
    @test Base.return_types(P._inject!, (M, typeof(PC0), typeof(fresh()), typeof(inj), P.ScoringOptions, Bool))[1] === Nothing
  end
  @test Base.return_types(P._inject_mode, (Bool, Bool, Bool))[1] <: Union{P.InjAppend,P.InjReplace,P.InjSync,P.InjOverride}
end

@testset "append: every observed cohort is added" begin
  soa = fresh(); P._inject_observed_cohorts!(PC0, soa, inj, P.ScoringOptions())
  s = P.getsite(soa, 1); @test Int(s.live) == 7
end
@testset "replace: the site equals the observed cohorts" begin
  soa = fresh(); P._inject_observed_cohorts!(PC0, soa, inj, P.ScoringOptions(; override=true, replace=true))
  c = cohorts(soa); @test c == Dict((1, 40) => 90.0, (2, 10) => 30.0, (1, 5) => 7.0)
  @test isapprox(Float64(P.getsite(soa, 1).B), 127.0; atol=1e-3)
end
@testset "sync: sim-only cohorts die, matched keep sim biomass, recruits enter with observed biomass" begin
  soa = fresh(); sc = P.ScoringOptions(; override=true, sync=true, disturbance=:off)
  P._inject_observed_cohorts!(PC0, soa, inj, sc)
  c = cohorts(soa)
  @test Set(keys(c)) == Set([(1, 40), (2, 10), (1, 5)])
  @test isapprox(c[(1, 40)], 100.0; atol=0.5) && isapprox(c[(2, 10)], 50.0; atol=0.5) && isapprox(c[(1, 5)], 7.0; atol=1e-3)
  s = P.getsite(soa, 1); @test isapprox(Float64(s.B), sum(values(c)); rtol=1e-5) && Int(s.old) == 3
  soa = fresh(); P._inject_observed_cohorts!(PC0, soa, inj, P.ScoringOptions(; override=true, sync=true, disturbance=:scale))
  @test isapprox(cohorts(soa)[(2, 10)], 30.0; atol=0.5)                               # (1-0.4)*50
  soa = fresh(); P._inject_observed_cohorts!(PC0, soa, inj, P.ScoringOptions(; override=true, sync=true, disturbance=:scale); skip_scale=true)
  @test isapprox(cohorts(soa)[(2, 10)], 50.0; atol=0.5)                               # already shocked before scoring
  soa = fresh(); P._inject_observed_cohorts!(PC0, soa, inj, P.ScoringOptions(; override=true, sync=true, disturbance=:exclude_overwrite))
  @test isapprox(cohorts(soa)[(2, 10)], 30.0; atol=0.5)                               # observed survivor biomass
end
@testset "override in place: matched biomass overwritten, nothing removed, unmatched appended" begin
  soa = fresh(); P._inject_observed_cohorts!(PC0, soa, inj, P.ScoringOptions(; override=true))
  c = cohorts(soa)
  @test length(c) == 5 && c[(1, 40)] == 90.0 && c[(2, 10)] == 30.0 && c[(1, 5)] == 7.0 && c[(1, 20)] > 0 && c[(2, 30)] > 0
  @test isapprox(Float64(P.getsite(soa, 1).B), sum(values(c)); rtol=1e-5)
end

@testset "reference_tables: smoothed table and the (plot, eco) -> year index" begin
  d0 = Date(2000, 1, 1)
  df = DataFrame(plot_id=UI[1], eco_id=UI[1], measdate=[d0 + Day(round(Int, 5 * 365.25))], start_measdate=[d0], eco_species_id=UI[1], age_calc=[40], agb_sum=FT[100])
  lp = PU.LossParams(; age_bins=PU.AgeBins(bins_idx=[10, 20, 40, 80], last_bin_open=true), smoothing_weights=FT[1], lambda=FT(1))
  spdf, plts = P.reference_tables(df, [[1, 2]], lp)
  @test spdf isa DataFrame && haskey(plts, (UI(1), UI(1))) && haskey(plts[(UI(1), UI(1))], 5)
  @test plts[(UI(1), UI(1))][5].keys[1]
end
end # module
