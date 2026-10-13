# HBAD plugin unit tests.  Run:  source wt_env.sh && ./julia_gdal.sh --project=. --threads=2 test/hbad_unit.jl
using Pan, Random, Printf, Statistics
using Pan.Plugins: HBADPlugin
using Pan.PanCore
const H = HBADPlugin
const HB = Pan._HB
include(joinpath(@__DIR__, "..", "tools", "hbad_translate_lib.jl"))   # reference: hb_init(), hbad_translate
hb_init()

nfail = Ref(0); npass = Ref(0)
macro check(cond, msg)
  quote
    if $(esc(cond)); npass[] += 1; else nfail[] += 1; println("  FAIL: ", $(esc(msg))); end
  end
end
rel(a, b) = abs(a - b) / max(abs(b), 1e-300)

species_list = ["PITA", "PIEL", "QUNI", "ACRU", "LITU", "SAPA", "CAAL27", "_S_HST", "_H_LST", "_H_NA", "_GRP_41_LST", "_GRP_28_LST", "_GRP_8_HST", "QUAL"]
eco_species_ids = [[1, 2, 3, 4, 8, 9], [1, 3, 5, 6, 7, 10, 11], [2, 12, 13, 14, 1, 4]]
eco_list = ["e1", "e2", "e3"]
pri = H.configure!(; species_list=species_list, eco_species_ids=eco_species_ids, verbose=false)
println("species -> translation key / prior key / growth key / mort key")
for (i, s) in enumerate(species_list)
  @printf("  %-12s %-8s %-8s %-8s %-8s sdimax p90/p95/p99 = %.0f/%.0f/%.0f\n", s, pri.trans_key[i], pri.prior_key[i], pri.growth_key[i], pri.mort_key[i], pri.sp[i].sdi_p90, pri.sp[i].sdi_p95, pri.sp[i].sdi_p99)
end
grp_of(name) = H.species_group(name, Dict(String(r.species) => String(r.grp) for r in eachrow(HB_GRP_DF)))
HB_GRP_DF = H.DataFrame(species=collect(keys(HB_GRP[])), grp=collect(values(HB_GRP[])))

# ----------------------------------------------------------------------------------------------------------
println("\n(1) translation parity vs tools/hbad_translate_lib.jl")
rng = MersenneTwister(1)
ncmp = 0; nclamp = 0; maxerr = Dict(:tph => 0.0, :dq => 0.0, :h => 0.0, :agb => 0.0)
nall = 600
for t in 1:nall
  g = rand(rng, 1:length(species_list))
  age = rand(rng) < 0.5 ? rand(rng, 1:300) : rand(rng) * 300 + 1
  agb = exp(log(0.05) + (log(25000.0) - log(0.05)) * rand(rng))
  p = pri.sp[g]
  tph, dq, h = H.translate(p, Float64(age), agb)
  ref = hbad_translate(species_list[g], age, agb; grp=grp_of(species_list[g]))
  unclamped = ref.qmd_cm > 2.54 + 1e-9 && ref.qmd_cm < 170 - 1e-9
  if unclamped
    global ncmp += 1
    maxerr[:tph] = max(maxerr[:tph], rel(tph, ref.tpa_ha)); maxerr[:dq] = max(maxerr[:dq], rel(dq, ref.qmd_cm)); maxerr[:h] = max(maxerr[:h], rel(h, ref.h_mean_m))
  else
    global nclamp += 1                      # lib clamps qmd: plugin stays AGB-consistent instead
    maxerr[:agb] = max(maxerr[:agb], rel(H.agb_of(p, tph, dq), agb))
  end
end
@printf("  compared %d (unclamped) cases: max rel err tph=%.2e dq=%.2e h=%.2e ; %d lib-clamped cases: max AGB-consistency err %.2e\n", ncmp, maxerr[:tph], maxerr[:dq], maxerr[:h], nclamp, maxerr[:agb])
@check maxerr[:tph] <= 1e-3 && maxerr[:dq] <= 1e-3 && maxerr[:h] <= 1e-3 "translation parity"
@check ncmp >= 200 "need >= 200 compared cases"
@check maxerr[:agb] <= 1e-9 "clamped cases are AGB-consistent"
# tiny AGB (recruits)
for g in (1, 2, 8, 9, 11), agb in (0.5, 1.0, 3.0, 10.0)
  tph, dq, h = H.translate(pri.sp[g], 5.0, agb)
  ref = hbad_translate(species_list[g], 5, agb; grp=grp_of(species_list[g]))
  @printf("  tiny %-12s agb=%5.1f: plugin tph=%.2f dq=%.2f h=%.2f | lib tph=%.2f qmd=%.2f h=%.2f\n", species_list[g], agb, tph, dq, h, ref.tpa_ha, ref.qmd_cm, ref.h_mean_m)
  @check tph >= H.HB_MIN_TPA_HA - 1e-9 || dq <= H.HB_DQ_MIN + 1e-9 "tph floor"
end
# lib's own AGB check of the A/mq structure: tph/(acre) * m(qmd) vs input
# ----------------------------------------------------------------------------------------------------------
println("\n(2) add_cohort! round trip (c_bio == agb exactly; allometric AGB within 0.5%)")

function build_soa(site_specs::Vector; eco_ids=nothing, rng=MersenneTwister(7))
  n = length(site_specs)
  eco_ids === nothing && (eco_ids = [1 + (i - 1) % length(eco_species_ids) for i in 1:n])
  counts = Int32[length(s) for s in site_specs]
  spc = Int32[length(eco_species_ids[e]) for e in eco_ids]
  soa = Pan.soa_type(HB)((cohort=counts, species=spc))
  for i in 1:n
    site = getsite(soa, i)
    site.active = true; site.rng = Random.Xoshiro(i); site.mapcode = i; site.ecocode = eco_ids[i]; site.eco_id = eco_ids[i]; site.ref_cn = i
    site.old = 0; site.live = 0
    Pan._init_site_plugin!(HB, site, true)
    for (esp, age, agb) in site_specs[i]
      H.add_cohort!(site, esp, age, agb)
    end
    site.B = sum(site.c_bio)
  end
  return soa
end
function make_ctx_default(; mode=:default, mutate=nothing)
  prm = H.generate_hbad_params(species_list, eco_list, eco_species_ids; rng=MersenneTwister(3), mode=mode)
  mutate === nothing || (prm = mutate(prm))
  eco = H.generate_hbad_eco_params(prm)
  return Pan.make_ctx(HB, eco), prm
end
rng = MersenneTwister(11)
specs = [[(rand(rng, 1:length(eco_species_ids[1 + (i - 1) % 3])), rand(rng, 1:200), exp(rand(rng) * 9 + 0.5)) for _ in 1:rand(rng, 3:8)] for i in 1:60]
soa = build_soa(specs)
maxrt = 0.0; nbad = 0
for i in 1:soa.n
  site = getsite(soa, i)
  e = Int(site.eco_id)
  for k in 1:Int(site.live)
    esp, age, agb = specs[i][k]
    @check site.c_bio[k] == Float32(agb) "c_bio exact (site $i cohort $k)"
    p = pri.sp[eco_species_ids[e][esp]]
    global maxrt = max(maxrt, rel(H.agb_of(p, Float64(site.c_tph[k]), Float64(site.c_dq[k])), agb))
  end
end
@printf("  max |allometric AGB(state) - agb| / agb = %.2e over %d cohorts\n", maxrt, sum(length, specs))
@check maxrt <= 0.005 "round trip <= 0.5%"
# set_cohort_bio!: scales tph, keeps dq/h
site = getsite(soa, 1); k = 1
dq0 = site.c_dq[k]; h0 = site.c_h[k]; b0 = site.c_bio[k]; p = pri.sp[eco_species_ids[1][Int(site.c_species[k])]]
Pan.set_cohort_bio!(HB, site, k, 0.5 * b0)
@check site.c_bio[k] == 0.5f0 * b0 && site.c_dq[k] == dq0 && site.c_h[k] == h0 "set_cohort_bio! keeps size"
@check rel(H.agb_of(p, Float64(site.c_tph[k]), Float64(site.c_dq[k])), 0.5 * b0) < 1e-5 "set_cohort_bio! allometric consistency"

# ----------------------------------------------------------------------------------------------------------
println("\n(3) zero-allocation annual step (serial site loop, after warm-up)")
ctx, prm = make_ctx_default()
rng = MersenneTwister(5)
# young, thin, moderately dense stands with short horizon: no cohort removal in the measured steps
specs3 = [[(rand(rng, 1:length(eco_species_ids[1 + (i - 1) % 3])), rand(rng, 10:60), 300 + 4000rand(rng)) for _ in 1:6] for i in 1:40]
soa3 = build_soa(specs3)
PanCore.PARALLEL_SITES[] = false
for y in 1:3; Pan.step!(HB, soa3, y, ctx); end
nlive0 = copy(soa3.scalar.live)
al = @allocated Pan.step!(HB, soa3, 4, ctx)
@check soa3.scalar.live == nlive0 "no cohort removed in measured step (so no CSR resize)"
alloc_site(soa, i, ctx) = (ep = ctx.HBAD.eco[Int(soa.scalar.eco_id[i])]; cst = ctx.HBAD.cst; H.site_year!(soa, i, ep, cst); @allocated H.site_year!(soa, i, ep, cst))
alloc_site(soa3, 3, ctx)
al_site = alloc_site(soa3, 3, ctx)
@printf("  allocated bytes: step! (40 sites, serial) = %d ; site_year! = %d\n", al, al_site)
@check al == 0 && al_site == 0 "zero allocation"
PanCore.PARALLEL_SITES[] = true
Pan.step!(HB, soa3, 5, ctx)
al_thr = @allocated Pan.step!(HB, soa3, 6, ctx)
@printf("  threaded step! (nthreads=%d) allocates %d bytes (task spawn overhead only, independent of #sites/cohorts)\n", Threads.nthreads(), al_thr)

# ----------------------------------------------------------------------------------------------------------
println("\n(4) single PITA cohort, age 5, 30 years, G_MULT = M_MULT = 1")
# AGB from data-like value: PITA age 5 cohort with ~ 300 g/m2 (young plantation-type stand)
agb5 = 300.0
soa4 = build_soa([[(1, 5, agb5)]]; eco_ids=[1])
ctx4, _ = make_ctx_default()
p = pri.sp[1]
@printf("  prior PITA: SDImax p95=%.0f ; CR a=%.1f b=%.4f c=%.2f\n", p.sdi_p95, p.hd_a, p.hd_b, p.hd_c)
@printf("  %4s %8s %8s %7s %6s %7s %9s %7s %7s\n", "age", "tph", "dq_cm", "h_m", "BA", "SDI", "AGB_g_m2", "RD", "mort%")
site = getsite(soa4, 1)
maxsdi = 0.0
prev_tph = Float64(site.c_tph[1])
function row4(y, mort)
  s4 = getsite(soa4, 1)
  ba = pi / 4 * (s4.c_dq[1] / 100)^2 * s4.c_tph[1]
  @printf("  %4d %8.1f %8.2f %7.2f %6.2f %7.1f %9.1f %7.2f %7.2f\n", Int(s4.c_age[1]), s4.c_tph[1], s4.c_dq[1], s4.c_h[1], max(s4.BA, ba), max(s4.SDI, s4.c_tph[1] * (s4.c_dq[1] / 25.4)^1.605), s4.c_bio[1], s4.RD, mort)
end
row4(0, NaN)
PanCore.PARALLEL_SITES[] = false
morts = Float64[]
for y in 1:30
  s4 = getsite(soa4, 1)
  t0 = Float64(s4.c_tph[1])
  Pan.step!(HB, soa4, y, ctx4)
  s4 = getsite(soa4, 1)
  m = 100 * (1 - Float64(s4.c_tph[1]) / t0)
  push!(morts, m)
  global maxsdi = max(maxsdi, Float64(s4.SDI))
  y % 5 == 0 && row4(y, m)
end
site = getsite(soa4, 1)
hd_cr = 1.3 + H.chapman_richards(p.hd_a, p.hd_b, p.hd_c, Float64(site.c_dq[1]))
@printf("  final h=%.2f m vs H-D curve at dq=%.1f: %.2f m ; max SDI over run = %.1f (p95 %.0f); mean mortality %.2f %%/yr (yr 1-30), %.2f %% (yr 1-5)\n",
  site.c_h[1], site.c_dq[1], hd_cr, maxsdi, p.sdi_p95, mean(morts), mean(morts[1:5]))
@check maxsdi <= p.sdi_p99 * 1.3 + 1 "SDI within SDImax range"
@check 0.3 < mean(morts) < 8 "mortality plausible"
@check rel(Float64(site.c_h[1]), hd_cr) < 0.35 "height near H-D curve"

# ----------------------------------------------------------------------------------------------------------
println("\n(5) mixed site: BAL ordering and self-thinning order")
# eco 1: species idx 1=PITA 2=PIEL 3=QUNI 4=ACRU 5=_S_HST(global 8) 6=_H_LST(global 9)
mixed = [[(1, 30, 6000.0), (3, 40, 3000.0), (4, 25, 800.0), (2, 12, 250.0)]]
soa5 = build_soa(mixed; eco_ids=[1])
ctx5, prm5 = make_ctx_default()
site = getsite(soa5, 1)
dqs = Float64.(site.c_dq); tphs = Float64.(site.c_tph)
ba = [pi / 4 * (d / 100)^2 * t for (d, t) in zip(dqs, tphs)]
bal_ref = [sum(ba[m] for m in eachindex(ba) if dqs[m] > dqs[k]; init=0.0) for k in eachindex(ba)]
PanCore.PARALLEL_SITES[] = false
Pan.step!(HB, soa5, 1, ctx5)   # snapshot BAL is taken before longevity/growth (no removal here, so order unchanged)
site = getsite(soa5, 1)
@printf("  dq=%s\n  BAL(ref)=%s\n  BAL(plugin)=%s\n", round.(dqs; digits=2), round.(bal_ref; digits=3), round.(Float64.(site.c_bal); digits=3))
@check all(abs.(Float64.(site.c_bal) .- bal_ref) .< 1e-3 .* max.(bal_ref, 1)) "BAL snapshot"
@check issorted(Float64.(site.c_bal)[sortperm(-dqs)]) "BAL increases as dq decreases"
# self-thinning: no growth, no mortality, overstocked site
zero_mut = prm -> (G = [zeros(FloatType, length(v)) for v in prm.G_MULT]; M = [zeros(FloatType, length(v)) for v in prm.M_MULT];
  H.HBADParams(ECO_LIST=prm.ECO_LIST, SPECIES_LIST=prm.SPECIES_LIST, ECO_SPECIES_IDS=prm.ECO_SPECIES_IDS, LONGEVITY=prm.LONGEVITY, SDIMAX=prm.SDIMAX, G_MULT=G, M_MULT=M))
ctx5z, prm5z = make_ctx_default(mutate=zero_mut)
dense = [[(1, 30, 9000.0), (3, 40, 6000.0), (4, 25, 4000.0), (2, 12, 2500.0)]]
soa5b = build_soa(dense; eco_ids=[1])
site = getsite(soa5b, 1)
tph_before = Float64.(site.c_tph); dq_before = Float64.(site.c_dq)
eco1 = ctx5z.HBAD.eco[1]
_, sdi_b, sdm_b = H.stand_stats(soa5b.csr.c_species, soa5b.csr.c_tph, soa5b.csr.c_dq, 0, 4, eco1.sdimax)
Pan.step!(HB, soa5b, 1, ctx5z)
site = getsite(soa5b, 1)
@printf("  overstocked: SDI before=%.0f SDImax(site)=%.0f ; after SDI=%.1f RD=%.3f ; thinned %.1f tph\n", sdi_b, sdm_b, site.SDI, site.RD, site.thin_tph)
# identify by species/age
order = sortperm(dq_before)   # smallest first
live_species = Int.(site.c_species); live_age = Float64.(site.c_age) .- 1
nl = Int(site.live)
tph_after = [begin
    j = findfirst(m -> live_species[m] == dense[1][k][1] && live_age[m] == dense[1][k][2], 1:nl)
    j === nothing ? 0.0 : Float64(site.c_tph[j])
  end for k in 1:4]
@printf("  tph before=%s\n  tph after =%s\n", round.(tph_before; digits=1), round.(tph_after; digits=1))
@check sdi_b > sdm_b "stand is overstocked"
@check abs(site.SDI / sdm_b - 1) < 2e-3 || site.RD <= 1.0 + 2e-3 "SDI brought to SDImax (RD<=1)"
# smallest removed first: if cohort at rank r lost trees, all smaller-rank cohorts must be gone entirely
lost = tph_before .- tph_after
let ok = true, seen_partial = false
  for r in order
    if seen_partial && lost[r] > 1e-3
      ok = false
    end
    if lost[r] > 1e-3 && tph_after[r] > 1e-3
      seen_partial = true      # a partially thinned cohort: larger ones must be untouched
    end
  end
  @check ok "smallest-dq-first thinning"
end
@check lost[order[1]] > 0 "the smallest cohort was thinned first"

# ----------------------------------------------------------------------------------------------------------
println("\n(6) swap-removal consistency vs reference loop")
rng = MersenneTwister(21)
nco = 14
cohorts = [(1 + (k - 1) % 6, 10 + 7k, 200.0 + 900rand(rng)) for k in 1:nco]       # distinct (esp, age)
soa6 = build_soa([cohorts]; eco_ids=[1])
ctx6z, prm6z = make_ctx_default(mutate=prm -> (z = zero_mut(prm);
    # make some species die by longevity: species 2 (PIEL) longevity 30, species 4 (ACRU) 60
    L = copy(z.LONGEVITY); L[2] = 30; L[4] = 60;
    H.HBADParams(ECO_LIST=z.ECO_LIST, SPECIES_LIST=z.SPECIES_LIST, ECO_SPECIES_IDS=z.ECO_SPECIES_IDS, LONGEVITY=L, SDIMAX=fill(FloatType(1500), length(z.SDIMAX)), G_MULT=z.G_MULT, M_MULT=z.M_MULT)))
site = getsite(soa6, 1)
orig = [(Int(site.c_species[k]), Float64(site.c_age[k]), Float64(site.c_tph[k]), Float64(site.c_dq[k]), Float64(site.c_h[k])) for k in 1:nco]
# reference removal by loop: longevity rule on pre-aging age (eco1 species -> global)
eco1 = ctx6z.HBAD.eco[1]
keep = [orig[k][2] < eco1.longevity[orig[k][1]] for k in 1:nco]
ba0 = [pi / 4 * (o[4] / 100)^2 * o[3] for o in orig]
bal0 = [sum(ba0[m] for m in 1:nco if orig[m][4] > orig[k][4]; init=0.0) for k in 1:nco]
Pan.step!(HB, soa6, 1, ctx6z)
site = getsite(soa6, 1)
nl = Int(site.live)
@check nl == count(keep) "survivor count $(nl) == $(count(keep))"
function check6(site, nl, orig, keep, bal0)
  okf = true
  for j in 1:nl
    key = (Int(site.c_species[j]), Float64(site.c_age[j]) - 1)
    q = findfirst(m -> (orig[m][1], orig[m][2]) == key, eachindex(orig))
    if q === nothing || !keep[q]; okf = false; continue; end
    okf &= Float64(site.c_tph[j]) == orig[q][3] && Float64(site.c_dq[j]) == orig[q][4] && Float64(site.c_h[j]) == orig[q][5]
    okf &= abs(Float64(site.c_bal[j]) - bal0[q]) < 1e-4 * max(bal0[q], 1)
    pp = pri.sp[eco_species_ids[1][key[1]]]
    okf &= rel(Float64(site.c_bio[j]), H.agb_of(pp, orig[q][3], orig[q][4])) < 1e-5
  end
  return okf
end
okf = check6(site, nl, orig, keep, bal0)
@check okf "every per-cohort field (species, age, tph, dq, h, bal, bio) followed its cohort through swap-removal"
@printf("  %d of %d cohorts kept by longevity rule; CSR capacity after step = %d\n", nl, nco, Int(soa6.refs.cohort[2] - soa6.refs.cohort[1]))
# tph<1 removal
soa6b = build_soa([[(1, 20, 300.0), (3, 20, 2.0), (4, 20, 400.0)]]; eco_ids=[1])
s6b = getsite(soa6b, 1); Pan.set_cohort_bio!(HB, s6b, 2, 1e-5)
println("  before: tph=", Float64.(getsite(soa6b, 1).c_tph), " live=", getsite(soa6b, 1).live)
Pan.step!(HB, soa6b, 1, ctx6z)
println("  after: live=", getsite(soa6b, 1).live, " tph=", Float64.(getsite(soa6b, 1).c_tph))
@check Int(getsite(soa6b, 1).live) == 2 "cohort below 1 tree/ha removed"

# ----------------------------------------------------------------------------------------------------------
println("\n(7) determinism: repeated runs, serial vs threaded (nthreads=$(Threads.nthreads()))")
rng = MersenneTwister(33)
specs7 = [[(rand(rng, 1:length(eco_species_ids[1 + (i - 1) % 3])), rand(rng, 1:150), exp(rand(rng) * 8 + 1)) for _ in 1:rand(rng, 2:12)] for i in 1:300]
function run7(parallel::Bool; years=25)
  soa = build_soa(specs7)
  ctx, _ = make_ctx_default(mode=:jitter)
  PanCore.PARALLEL_SITES[] = parallel
  for y in 1:years; Pan.step!(HB, soa, y, ctx); end
  return soa
end
sig(soa) = hash((soa.refs.cohort, soa.csr.c_species, soa.csr.c_age, soa.csr.c_bio, soa.csr.c_tph, soa.csr.c_dq, soa.csr.c_h, soa.scalar.B, soa.scalar.BA, soa.scalar.SDI, soa.scalar.live))
a = run7(false); b = run7(false); c = run7(true); d = run7(true)
@printf("  hashes: serial %s %s | threaded %s %s\n", string(sig(a), base=16), string(sig(b), base=16), string(sig(c), base=16), string(sig(d), base=16))
@check sig(a) == sig(b) && sig(c) == sig(d) && sig(a) == sig(c) "byte-identical across runs and serial/threaded"
@printf("DETERMINISM_HASH %s (live=%d cohorts)\n", string(sig(a), base=16), sum(a.scalar.live))
PanCore.PARALLEL_SITES[] = true

# plugin-level spot checks: self-thinning keeps the stand below SDImax after 25 years of the random sites
let soa = a, ctx = make_ctx_default(mode=:jitter)[1]
  mx = maximum(soa.scalar.RD[i] for i in 1:soa.n if soa.scalar.live[i] > 0)
  @printf("  after 25 yr: max RD over sites = %.4f, sites alive = %d / %d\n", mx, count(>(0), soa.scalar.live), soa.n)
  @check mx <= 1.0 + 1e-3 "RD <= 1 after thinning"
end

# throughput (informational): 3000 sites x ~8 cohorts x 25 years
let rng = MersenneTwister(44)
  big = [[(rand(rng, 1:length(eco_species_ids[1 + (i - 1) % 3])), rand(rng, 1:150), exp(rand(rng) * 8 + 1)) for _ in 1:rand(rng, 3:12)] for i in 1:3000]
  soa = build_soa(big); ctx, _ = make_ctx_default(mode=:jitter)
  for par in (false, true)
    PanCore.PARALLEL_SITES[] = par
    so = deepcopy(soa); Pan.step!(HB, so, 0, ctx)   # warm-up
    so = deepcopy(soa)
    t = @elapsed for y in 1:25; Pan.step!(HB, so, y, ctx); end
    @printf("  throughput: %d sites x 25 yr, %s (%d threads): %.3f s  = %.2f us per site-year\n", so.n, par ? "threaded" : "serial", par ? Threads.nthreads() : 1, t, 1e6 * t / (so.n * 25))
  end
  PanCore.PARALLEL_SITES[] = true
end

println("\nHBAD unit tests: $(npass[]) passed, $(nfail[]) failed")
nfail[] == 0 || exit(1)
