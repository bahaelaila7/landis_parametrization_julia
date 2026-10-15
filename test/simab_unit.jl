# Unit tests: Sim A death term / survivors-only W / m-objective Chebyshev (docs/simAB_options_experiment.md).
#   source wt_env.sh && ./julia_gdal.sh --project=. --threads=2 test/simab_unit.jl
using Pan, Random, DataFrames, Dates
const P = Pan; const PU = P.PU; const GD = P.GDIgel
nfail = Ref(0); npass = Ref(0)
macro check(cond, msg)
  quote
    if $(esc(cond)); npass[] += 1; else nfail[] += 1; println("  FAIL: ", $(esc(msg))); end
  end
end
UI = P.UIntType; FT = P.FloatType
lp = PU.LossParams(age_bins=PU.AgeBins(bins_idx=collect(1:400), last_bin_open=true), smoothing_weights=FT[one(FT)])
esids = [[1, 2]]
function site_with(cohs)
  df = DataFrame(mapcode=fill(1, length(cohs)), eco_id=fill(1, length(cohs)), eco_species_id=[c[1] for c in cohs], age_calc=[c[2] for c in cohs], agb_sum=[c[3] for c in cohs])
  soa = P.make_sites_from_communities(df, esids, Random.Xoshiro(1)); s = P.getsite(soa, 1); s.B = sum(Float64(s.c_bio[j]) for j in 1:Int(s.live)); soa
end
scoring = P.ScoringOptions(; disturbance=:off, exclude_recruits_at_entry=false)
# visits at year 2 and 5 (Δt = 3). Observed at T=5: A=(sp1,40) survivor, E=(sp2,10) survivor. Sim has A, E and C=(sp1,20) (absent from obs ⇒ natural death).
inj2 = P._SiteInjectionYear([(1, Tuple{UI,FT,FT,FT}[(UI(1), FT(37), FT(90), FT(0)), (UI(1), FT(17), FT(60), FT(0)), (UI(2), FT(7), FT(30), FT(0))])])
inj5 = P._SiteInjectionYear([(1, Tuple{UI,FT,FT,FT}[(UI(1), FT(40), FT(100), FT(0)), (UI(2), FT(10), FT(35), FT(0))])])
idict = P._SiteInjectionDict(2 => inj2, 5 => inj5)
soa = site_with([(1, 40, 100.0), (1, 20, 70.0), (2, 10, 35.0)])
plan = P._build_excl_plan(idict, soa, esids, lp, scoring)
@check plan.prev[5][1] == 2 "previous visit of year 5 is year 2 ($(get(plan.prev, 5, nothing)))"
acc = P.DeathAcc(esids)
P._death_collect!(acc, soa, 5, idict, plan)
@check acc.n_surv[1][1] == 1 && acc.surv_dt[1][1] == 3.0 "species 1: one survivor with Δt=3 ($(acc.n_surv[1][1]), $(acc.surv_dt[1][1]))"
@check acc.dhist[1][1][3] == 1 && sum(acc.dhist[1][1]) == 1 "species 1: one natural death with Δt=3"
@check acc.n_surv[1][2] == 1 && sum(acc.dhist[1][2]) == 0 "species 2: one survivor, no death"
p = 0.05
nll_surv = -3 * log(1 - p); nll_death = -log(1 - (1 - p)^3)
s1, n1 = P.death_cell(acc, 1, 1, p)
@check n1 == 2 && isapprox(s1, nll_surv + nll_death; rtol=1e-12) "cell (1,1): Σ NLL = survivor $(nll_surv) + death $(nll_death) (got $s1, n=$n1)"
@check isapprox(P.death_nll_survivor(0.0, 4.0), -4 * log1p(-1e-6)) "p clamped below at 1e-6"
@check isapprox(P.death_nll_survivor(0.9, 1.0), -log(0.5)) "p clamped above at 0.5"
# group value: d_g = mean over the group's cohorts (cells pooled)
d = P._gd_group_d(acc, [[p, p]], esids, [[(1, 1), (1, 2)]])
s2, n2 = P.death_cell(acc, 1, 2, p)
@check isapprox(d[1], (s1 + s2) / (n1 + n2); rtol=1e-12) "group d_g = Σ NLL / # cohorts over the group's cells ($(d[1]))"
@check isnan(P._gd_group_d(P.DeathAcc(esids), [[p, p]], esids, [[(1, 1)]])[1]) "group without cohorts: d_g = NaN"
# the death NLL is minimised at the empirical annual rate: 1 death in 3 yr among 1 survivor(3 yr) → compare grid
ps = range(0.01, 0.5; length=2000); vals = [P.death_cell(acc, 1, 1, q)[1] for q in ps]
@check 0.05 < ps[argmin(vals)] < 0.5 "NLL minimum is interior ($(ps[argmin(vals)]))"
# survivors-only: sim-only cohort C removed, survivors untouched, B consistent
P._survivors_only!(soa, 5, idict)
site = P.getsite(soa, 1)
cs = Set((Int(site.c_species[j]), Int(site.c_age[j])) for j in 1:Int(site.live))
@check cs == Set([(1, 40), (2, 10)]) "survivors-only keeps the matched cohorts only: $cs"
@check isapprox(Float64(site.B), 135.0; atol=0.5) "site.B updated (135 expected, got $(site.B))"
# survivors-only W excludes the sim-only cohort: W of species 1 equals the matched-only W
gt_obs = let d0 = Date(2000, 1, 1)
  df = DataFrame(plot_id=UI[1], eco_id=UI[1], measdate=[d0 + Day(round(Int, 5 * 365.25))], start_measdate=[d0], eco_species_id=UI[1], age_calc=[40], agb_sum=FT[100])
  P.Data.make_spdf_dict(PU.smoothen_ref_years(df, lp, 60), esids)[(UI(1), UI(1))][5]
end
soa_all = site_with([(1, 40, 100.0), (1, 20, 70.0)]); soa_surv = site_with([(1, 40, 100.0)])
w_all = PU.calculate_site_loss2(5, P.getsite(soa_all, 1), 2, esids, gt_obs, lp, PU.LossScales()).sp_w_loss[1]
w_surv = PU.calculate_site_loss2(5, P.getsite(soa_surv, 1), 2, esids, gt_obs, lp, PU.LossScales()).sp_w_loss[1]
@check w_all > 1e-3 && abs(w_surv) < 1e-6 "W with the natural-death cohort = $w_all > 0; survivors only = $w_surv ≈ 0"
# simplex lattice
g4 = GD.simplex_weights(4)
@check length(g4) == 15 && all(w -> isapprox(sum(w), 1.0) && all(>=(0), w), g4) "simplex lattice H=4 has 15 points summing to 1"
@check length(unique(g4)) == 15 && [1.0, 0.0, 0.0] in g4 && [0.0, 0.0, 1.0] in g4 "lattice contains the vertices"
@check length(GD.simplex_weights(2)) == 6 "H=2 → 6 points"
# m=2 identity: the m=2 scalarisation is unchanged, and m=3 with weights (λ, 1−λ, 0) reduces to it
mkst(m) = GD.GDState([GD.BlockDef([1], [1, 2])], 1; obj=:mo, G=2, aref=3.0, wref=5.0, dref=2.0, lams=[0.3], m=m)
f2 = [1.0, 2.0, 4.0, NaN]
st2 = mkst(2); A, W = GD.mo_sums(f2, 2)
@check GD.lineage_total(st2, 0.3, f2) == max(0.3 * A / 3.0, 0.7 * W / 5.0) "m=2 lineage_total is the original formula"
f3 = vcat([1.0, 2.0], [4.0, NaN], [0.5, 1.5]); st3 = mkst(3)
A3, W3, D3 = GD.mo_sums3(f3, 2)
@check (A3, W3, D3) == (3.0, 4.0, 2.0) "mo_sums3: $((A3,W3,D3))"
@check GD.lineage_total(st3, [0.3, 0.7, 0.0], f3) == max(0.3 * 3.0 / 3.0, 0.7 * 4.0 / 5.0) "m=3 with a zero death weight equals the m=2 value"
@check GD.lineage_total(st3, [0.0, 0.0, 1.0], f3) == 2.0 / 2.0 "m=3 pure-death weight: D/D_ref"
@check GD.block_score(st3, [1/3, 1/3, 1/3], f3, [2]) == max((1/3) * 2.0 / 3.0, 0.0, (1/3) * 1.5 / 2.0) "m=3 block score uses only the block's groups"
println("simab_unit: $(npass[]) passed, $(nfail[]) failed")
exit(nfail[] == 0 ? 0 : 1)
