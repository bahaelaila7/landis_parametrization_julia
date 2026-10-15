# Unit test: disturbance shock before scoring (docs/mo_gdigel.md §2). Run:
#   source wt_env.sh && ./julia_gdal.sh --project=. --threads=2 test/disturbance_shock_unit.jl
using Pan, Random, DataFrames, Dates
const P = Pan
nfail = Ref(0); npass = Ref(0)
macro check(cond, msg)
  quote
    if $(esc(cond)); npass[] += 1; else nfail[] += 1; println("  FAIL: ", $(esc(msg))); end
  end
end
UI = P.UIntType; FT = P.FloatType

# one site, species 1 and 2 of eco 1.  Simulated cohorts at measurement year T:
#   A = (sp1, 40) 100 undisturbed;  B = (sp1, 20) 80 FULLY removed by disturbance;
#   C = (sp2, 30) 60 died naturally in the data (absent from obs, NOT disturbed);  D = (sp2, 10) 50 partial disturbance drop 0.4
function fresh()
  sp = DataFrame(mapcode=[1, 1, 1, 1], eco_id=[1, 1, 1, 1], eco_species_id=[1, 1, 2, 2], age_calc=[40, 20, 30, 10], agb_sum=[100.0, 80.0, 60.0, 50.0])
  soa = P.make_sites_from_communities(sp, [[1, 2]], Random.Xoshiro(1))
  s = P.getsite(soa, 1); s.B = sum(Float64(s.c_bio[j]) for j in 1:Int(s.live))   # make_sites_from_communities leaves B to the caller
  soa
end
function cohorts(soa)
  s = P.getsite(soa, 1)
  Dict((Int(s.c_species[j]), Int(s.c_age[j])) => Float64(s.c_bio[j]) for j in 1:Int(s.live))
end
obs_bio = Dict((1, 40) => 100.0, (2, 10) => 50.0 / 0.6)          # observed (survivors only): A, D (D's observed state is what is left after the drop)
inj = P._SiteInjectionYear([(1, Tuple{UI,FT,FT,FT}[(UI(1), FT(40), FT(100), FT(0)), (UI(2), FT(10), FT(30), FT(0.4))])])
shk = P._SiteShockYear([(1, Tuple{UI,FT}[(UI(1), FT(20))])])

for (mode, d_expect_after_shock) in ((:scale, 50.0 * 0.6), (:exclude_overwrite, 50.0), (:exclude_noscale, 50.0), (:off, 50.0))
  P.OVERRIDE_INJECTION_DISTURBANCE[] = mode
  soa = fresh(); s = P.getsite(soa, 1)
  @check Int(s.live) == 4 "setup: 4 cohorts ($mode)"
  P._apply_disturbance_shock!(soa, shk, inj)
  c = cohorts(soa)
  @check Int(s.live) == 3 && !haskey(c, (1, 20)) "$mode: B (fully removed) is gone BEFORE scoring"
  @check haskey(c, (1, 40)) && isapprox(c[(1, 40)], 100.0; atol=0.5) "$mode: A (undisturbed) keeps its simulated biomass ($(get(c,(1,40),NaN)))"
  @check haskey(c, (2, 30)) && isapprox(c[(2, 30)], 60.0; atol=0.5) "$mode: C (natural death in the data) is still simulated and scored ($(get(c,(2,30),NaN)))"
  @check haskey(c, (2, 10)) && isapprox(c[(2, 10)], d_expect_after_shock; atol=0.5) "$mode: D after shock = $d_expect_after_shock (got $(get(c,(2,10),NaN)))"
  @check isapprox(Float64(s.B), sum(values(c)); rtol=1e-5) "$mode: site.B consistent with cohorts ($(s.B) vs $(sum(values(c))))"
  # the injection for the NEXT interval: D must be scaled exactly once (skip_scale), C removed by the sync's Pass A
  P._inject_observed_cohorts!(soa, inj; override=true, sync=true, skip_scale=true)
  c2 = cohorts(soa)
  @check !haskey(c2, (2, 30)) && !haskey(c2, (1, 20)) "$mode: sync removes C (and B stays gone)"
  if mode === :scale
    @check isapprox(c2[(2, 10)], 30.0; atol=0.5) "scale: D scaled exactly once by the shock ($(c2[(2,10)]) ~ 30 = 50*(1-0.4))"
  end
end
# legacy path (skip_scale = false, no shock): the sync scales D itself, once
P.OVERRIDE_INJECTION_DISTURBANCE[] = :scale
soa = fresh(); P._inject_observed_cohorts!(soa, inj; override=true, sync=true)
c = cohorts(soa)
@check isapprox(c[(2, 10)], 30.0; atol=0.5) "legacy sync scaling unchanged (D = 30 once)"
# double application would give 18: confirm the guard matters
soa = fresh(); P._apply_disturbance_shock!(soa, nothing, inj); P._inject_observed_cohorts!(soa, inj; override=true, sync=true)
@check isapprox(cohorts(soa)[(2, 10)], 18.0; atol=0.5) "without skip_scale a shocked cohort would be scaled twice (18) -> skip_scale is required"

# ---------------- cohort-level exclusion (two cohorts of the SAME species, one disturbed) and recruits at entry ----------------
const PU = P.PU
lp = PU.LossParams(age_bins=PU.AgeBins(bins_idx=collect(1:400), last_bin_open=true), smoothing_weights=FT[one(FT)])
esids = [[1, 2]]
function site_with(cohs)   # cohs = [(sp, age, bio)]
  df = DataFrame(mapcode=fill(1, length(cohs)), eco_id=fill(1, length(cohs)), eco_species_id=[c[1] for c in cohs], age_calc=[c[2] for c in cohs], agb_sum=[c[3] for c in cohs])
  soa = P.make_sites_from_communities(df, esids, Random.Xoshiro(1)); soa
end
function full_ref(rows)    # rows = [(sp, age, agb)] observed at sim_year 5 -> SPDFGroundTruth with the standard code path
  d0 = Dates.Date(2000, 1, 1)
  df = DataFrame(plot_id=UI[1 for _ in rows], eco_id=UI[1 for _ in rows], measdate=[d0 + Dates.Day(round(Int, 5 * 365.25)) for _ in rows], start_measdate=[d0 for _ in rows],
    eco_species_id=UI[r[1] for r in rows], age_calc=[r[2] for r in rows], agb_sum=FT[r[3] for r in rows])
  P.Data.make_spdf_dict(PU.smoothen_ref_years(df, lp, 60), esids)[(UI(1), UI(1))][5]
end
wloss(soa, gt, excl_c) = begin
  site = P.getsite(soa, 1)
  PU.calculate_site_loss2(5, site, 2, esids, gt, lp; excl_c=excl_c).sp_w_loss[1]
end
# observed at T=5: (sp1,40) 100 undisturbed, (sp1,20) 80 partially disturbed (drop .3). Sim: (40) matches obs, (20) is far off.
P.OVERRIDE_INJECTION_DISTURBANCE[] = :exclude_overwrite
P.EXCLUDE_RECRUITS_AT_ENTRY[] = false
inj5 = P._SiteInjectionYear([(1, Tuple{UI,FT,FT,FT}[(UI(1), FT(40), FT(100), FT(0)), (UI(1), FT(20), FT(80), FT(0.3))])])
idict = P._SiteInjectionDict(5 => inj5)
soa = site_with([(1, 40, 100.0), (1, 20, 500.0)])
plan = P._build_excl_plan(idict, soa, esids, lp)
@check haskey(plan.coh, 5) && plan.coh[5][1] == [(UI(1), FT(20))] "cohort-level plan excludes only (sp1, age 20): $(get(plan.coh,5,nothing))"
gt_full = full_ref([(1, 40, 100.0), (1, 20, 80.0)])
gt_alt = plan.ref[5][1]
w_full = wloss(soa, gt_full, nothing)
w_cohort = wloss(soa, gt_alt, plan.coh[5][1])
@check w_full > 1e-3 "without exclusion the badly simulated disturbed cohort costs W=$w_full"
@check abs(w_cohort) < 1e-6 "with cohort-level exclusion only the undisturbed (matching) cohort is compared: W=$w_cohort"
@check Set(keys(gt_alt.records)) == Set([UI(1)]) && isapprox(gt_alt.records[UI(1)].sp_agb_sum, 100.0; atol=1e-3) "alt reference = undisturbed cohort only (agb $(gt_alt.records[UI(1)].sp_agb_sum))"
# the species is NOT dropped as a whole: a sim error in the UNDISTURBED cohort is still scored
soa_bad = site_with([(1, 55, 100.0), (1, 20, 500.0)])
@check wloss(soa_bad, gt_alt, plan.coh[5][1]) > 1e-3 "an error in the undisturbed cohort of the same species is still penalised ($(wloss(soa_bad, gt_alt, plan.coh[5][1])))"
# species with no cohort left on either side contributes nothing (no 'missing species' penalty)
inj5b = P._SiteInjectionYear([(1, Tuple{UI,FT,FT,FT}[(UI(2), FT(30), FT(50), FT(0.5))])])
plan_b = P._build_excl_plan(P._SiteInjectionDict(5 => inj5b), soa, esids, lp)
soa_b = site_with([(2, 30, 400.0)])
@check abs(wloss(soa_b, plan_b.ref[5][1], plan_b.coh[5][1])) < 1e-9 && isempty(plan_b.ref[5][1].records) "sole cohort of a species disturbed: species contributes nothing (alt ref empty, no missing penalty)"
# the acc! (tier-3) variant agrees
let site = P.getsite(soa, 1), nsp = 2
  wacc = zeros(FT, nsp); aacc = zeros(FT, nsp)
  PU.calculate_site_loss2_acc!(wacc, aacc, 5, site, nsp, esids, gt_alt, lp, Int[], FT[], Bool[], zeros(FT, 401), zeros(FT, 401); excl_c=plan.coh[5][1])
  @check abs(wacc[1]) < 1e-6 "tier-3 (acc!) cohort-level exclusion agrees: $(wacc[1])"
end
# recruits at entry: visit at year 2 has (sp1, age 37) only; at year 5: (sp1,40) [= 37 + 3 yr] and (sp1,20) [new] -> only (20) is a recruit
P.OVERRIDE_INJECTION_DISTURBANCE[] = :off
P.EXCLUDE_RECRUITS_AT_ENTRY[] = true
inj2 = P._SiteInjectionYear([(1, Tuple{UI,FT,FT,FT}[(UI(1), FT(37), FT(90), FT(0))])])
inj5c = P._SiteInjectionYear([(1, Tuple{UI,FT,FT,FT}[(UI(1), FT(40), FT(100), FT(0)), (UI(1), FT(20), FT(80), FT(0))])])
plan_r = P._build_excl_plan(P._SiteInjectionDict(2 => inj2, 5 => inj5c), soa, esids, lp)
@check !haskey(plan_r.coh, 2) "first visit: nothing is a recruit"
@check plan_r.coh[5][1] == [(UI(1), FT(20))] "recruit (no record at the previous visit) excluded from the reference at entry: $(get(plan_r.coh,5,nothing))"
@check isapprox(plan_r.ref[5][1].records[UI(1)].sp_agb_sum, 100.0; atol=1e-3) "recruit absent from alt reference"
P.EXCLUDE_RECRUITS_AT_ENTRY[] = false
plan_r0 = P._build_excl_plan(P._SiteInjectionDict(2 => inj2, 5 => inj5c), soa, esids, lp)
@check isempty(plan_r0.coh) "exclude_recruits_at_entry=false: no recruit exclusion"
P.EXCLUDE_RECRUITS_AT_ENTRY[] = true
# previous visit = year 0 (the initial state) is a valid previous visit: (sp1,20) is a recruit relative to year 0, (sp1,40)=35+5 is not
inj0 = P._SiteInjectionYear([(1, Tuple{UI,FT,FT,FT}[(UI(1), FT(35), FT(90), FT(0))])])
plan_0 = P._build_excl_plan(P._SiteInjectionDict(0 => inj0, 5 => inj5c), soa, esids, lp)
@check haskey(plan_0.coh, 5) && plan_0.coh[5][1] == [(UI(1), FT(20))] "recruit test against a year-0 previous visit: $(get(plan_0.coh,5,nothing))"
println("disturbance_shock_unit: $(npass[]) passed, $(nfail[]) failed")
exit(nfail[] == 0 ? 0 : 1)
