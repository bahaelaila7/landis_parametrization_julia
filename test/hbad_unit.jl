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
hctx0 = H.HBADContext(species_list, eco_species_ids; verbose=false)   # the run context: priors + eco -> species table + options (immutable)
pri = hctx0.priors
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
  counts = Int32[max(length(s), 1) for s in site_specs]
  spc = Int32[length(eco_species_ids[e]) for e in eco_ids]
  soa = Pan.soa_type(HB)((cohort=counts, species=spc))
  for i in 1:n
    site = getsite(soa, i)
    site.active = true; site.rng = Random.Xoshiro(i); site.mapcode = i; site.ecocode = eco_ids[i]; site.eco_id = eco_ids[i]; site.ref_cn = i
    site.old = 0; site.live = 0
    Pan._init_site_plugin!(HB, site, true)
    for (esp, age, agb) in site_specs[i]
      H.add_cohort!(hctx0, site, esp, age, agb)
    end
    site.B = sum(site.c_bio)
  end
  return soa
end
function make_ctx_default(; mode=:default, mutate=nothing)
  prm = H.generate_hbad_params(hctx0, species_list, eco_list, eco_species_ids; rng=MersenneTwister(3), mode=mode)
  mutate === nothing || (prm = mutate(prm))
  eco = H.generate_hbad_eco_params(hctx0, prm)
  return Pan.make_ctx(hctx0, eco), prm
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
Pan.set_cohort_bio!(hctx0, site, k, 0.5 * b0)
@check site.c_bio[k] == 0.5f0 * b0 && site.c_dq[k] == dq0 && site.c_h[k] == h0 "set_cohort_bio! keeps size"
@check rel(H.agb_of(p, Float64(site.c_tph[k]), Float64(site.c_dq[k])), 0.5 * b0) < 1e-5 "set_cohort_bio! allometric consistency"

# ----------------------------------------------------------------------------------------------------------
println("\n(3) zero-allocation annual step (serial site loop, after warm-up)")
ctx, prm = make_ctx_default()
rng = MersenneTwister(5)
# young, thin, moderately dense stands with short horizon: no cohort removal in the measured steps
specs3 = [[(rand(rng, 1:length(eco_species_ids[1 + (i - 1) % 3])), rand(rng, 10:60), 300 + 4000rand(rng)) for _ in 1:6] for i in 1:40]
soa3 = build_soa(specs3)
soa3.parallel = false
for y in 1:3; Pan.step!(HB, soa3, y, ctx); end
nlive0 = copy(soa3.scalar.live)
al = @allocated Pan.step!(HB, soa3, 4, ctx)
@check soa3.scalar.live == nlive0 "no cohort removed in measured step (so no CSR resize)"
alloc_site(soa, i, ctx) = (ep = ctx.HBAD.eco[Int(soa.scalar.eco_id[i])]; cst = ctx.HBAD.cst; H.site_year!(soa, i, ep, cst); @allocated H.site_year!(soa, i, ep, cst))
alloc_site(soa3, 3, ctx)
al_site = alloc_site(soa3, 3, ctx)
@printf("  allocated bytes: step! (40 sites, serial) = %d ; site_year! = %d\n", al, al_site)
@check al == 0 && al_site == 0 "zero allocation"
soa3.parallel = true
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
soa4.parallel = false
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
soa5.parallel = false
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
s6b = getsite(soa6b, 1); Pan.set_cohort_bio!(hctx0, s6b, 2, 1e-5)
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
  soa.parallel = parallel
  for y in 1:years; Pan.step!(HB, soa, y, ctx); end
  return soa
end
sig(soa) = hash((soa.refs.cohort, soa.csr.c_species, soa.csr.c_age, soa.csr.c_bio, soa.csr.c_tph, soa.csr.c_dq, soa.csr.c_h, soa.scalar.B, soa.scalar.BA, soa.scalar.SDI, soa.scalar.live))
a = run7(false); b = run7(false); c = run7(true); d = run7(true)
@printf("  hashes: serial %s %s | threaded %s %s\n", string(sig(a), base=16), string(sig(b), base=16), string(sig(c), base=16), string(sig(d), base=16))
@check sig(a) == sig(b) && sig(c) == sig(d) && sig(a) == sig(c) "byte-identical across runs and serial/threaded"
@printf("DETERMINISM_HASH %s (live=%d cohorts)\n", string(sig(a), base=16), sum(a.scalar.live))

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
    so = deepcopy(soa); so.parallel = par; Pan.step!(HB, so, 0, ctx)   # warm-up
    so = deepcopy(soa); so.parallel = par
    t = @elapsed for y in 1:25; Pan.step!(HB, so, y, ctx); end
    @printf("  throughput: %d sites x 25 yr, %s (%d threads): %.3f s  = %.2f us per site-year\n", so.n, par ? "threaded" : "serial", par ? Threads.nthreads() : 1, t, 1e6 * t / (so.n * 25))
  end
end

# ----------------------------------------------------------------------------------------------------------
println("\n(8) establishment + planting")
hb_with(prm; kw...) = H.HBADParams(; merge(NamedTuple{fieldnames(H.HBADParams)}(map(f -> getfield(prm, f), fieldnames(H.HBADParams))), NamedTuple(kw))...)
# eco 1 species (local 1..6) = PITA PIEL QUNI ACRU _S_HST _H_LST = global 1 2 3 4 8 9; tolerance by global species id
tol_by_global = UIntType[1, 2, 3, 4, 3, 3, 3, 5, 5, 3, 3, 3, 3, 3]
estab_ctx(; p=1.0, rain=true, mutate=nothing, mode=:default) = begin
  prm = H.generate_hbad_params(hctx0, species_list, eco_list, eco_species_ids; rng=MersenneTwister(3), mode=mode)
  prm = hb_with(prm; PROB_ESTAB_SPP=[fill(FloatType(p), length(v)) for v in prm.PROB_ESTAB_SPP], SHADE_TOL=tol_by_global, ESTAB_SEED_RAIN=rain)
  mutate === nothing || (prm = mutate(prm))
  Pan.make_ctx(hctx0, H.generate_hbad_eco_params(hctx0, prm)), prm
end
estab_on!(soa) = (soa.scalar.no_establish .= false; soa)
const BSL = Pan.BiomassSuccessionPlugin
# defaults of the new parameters
let prm = H.generate_hbad_params(hctx0, species_list, eco_list, eco_species_ids; rng=MersenneTwister(3))
  bsl = BSL.generate_biomass_params(BSL.BSTables(), species_list, eco_list, eco_species_ids; rng=MersenneTwister(1))
  @check prm.SUFFICIENT_LIGHT == bsl.SUFFICIENT_LIGHT "SUFFICIENT_LIGHT default == BiomassSuccession matrix"
  @check prm.SUFFICIENT_LIGHT[6][5] == 0.5f0 "corner (shade class 5, tolerance 5) == 0.50"
  @check all(v -> all(0 .< v .< 1), prm.PROB_ESTAB_SPP) "default annual PROB_ESTAB in (0,1)"
  @check isapprox(prm.PROB_ESTAB_SPP[1][1], 1 - 0.4^(1 / 8); atol=1e-6) "per-cycle 0.60 -> annual 1-(1-0.6)^(1/8)"
  @check prm.MIN_REL_RD[1] == FloatType[0.15, 0.30, 0.45, 0.60, 0.75] && prm.ESTAB_TPH[1] == 750 && !prm.ESTAB_SEED_RAIN "other defaults"
  @check isapprox(H.annual_prob(0.6, 8), 0.10818; atol=1e-4) "annual_prob(0.6, 8)"
  @check H.shade_class_of(0.0, (0.15, 0.3, 0.45, 0.6, 0.75)) == 0 && H.shade_class_of(0.5, (0.15, 0.3, 0.45, 0.6, 0.75)) == 3 && H.shade_class_of(1.0, (0.15, 0.3, 0.45, 0.6, 0.75)) == 5 "shade class thresholds"
end
build_soa_ser(a...; kw...) = (s = build_soa(a...; kw...); s.parallel = false; s)   # sections (a)-(f): serial sites

println("(a) open sites with seed rain recruit tolerant and intolerant species")
ctxa, prma = estab_ctx(p=1.0)
nA = 400
soaa = estab_on!(build_soa_ser([Tuple{Int,Int,Float64}[] for _ in 1:nA]; eco_ids=fill(1, nA)))
Pan.step!(HB, soaa, 1, ctxa)
cnt_a = zeros(Int, 6)
okstate = true
for i in 1:nA
  site = getsite(soaa, i)
  for k in 1:Int(site.live)
    cnt_a[Int(site.c_species[k])] += 1
    j = Int(site.c_species[k]); pp = pri.sp[eco_species_ids[1][j]]
    global okstate; okstate &= site.c_age[k] == 1 && site.c_tph[k] == 750 && site.c_dq[k] == 1 && abs(site.c_h[k] - (1.3 + H.chapman_richards(pp.hd_a, pp.hd_b, pp.hd_c, 1.0))) < 1e-5 &&
               rel(Float64(site.c_bio[k]), H.agb_of(pp, 750.0, 1.0)) < 1e-5
  end
end
@printf("  recruits per species after 1 yr of seed rain on %d open sites (PITA tol1 .. _H_LST): %s ; expected PITA 0.75*N\n", nA, string(cnt_a))
@check okstate "recruit state: age 1, tph 750, dq 1.0, h = 1.3+CR(1), c_bio = allometric AGB"
@check all(cnt_a .> 0) "every species (tolerant and intolerant) recruited on open sites"
@check abs(cnt_a[1] / nA - 0.75) < 0.07 && all(abs.(cnt_a[2:6] ./ nA .- 1.0) .< 1e-9) "open-site rates = PROB_ESTAB x SUFFICIENT_LIGHT[class 0]"
@check all(soaa.scalar.shade_class[1:nA] .== 0) "open site -> shade class 0"
# mature seed source required without seed rain
ctxa2, _ = estab_ctx(p=1.0, rain=false)
soaa2 = estab_on!(build_soa_ser([Tuple{Int,Int,Float64}[] for _ in 1:20]; eco_ids=fill(1, 20)))
Pan.step!(HB, soaa2, 1, ctxa2)
@check sum(soaa2.scalar.live[1:20]) == 0 "no seed rain and no mature cohort -> no recruits"
soaa3 = estab_on!(build_soa_ser([[(1, 40, 3000.0), (3, 5, 20.0)] for _ in 1:40]; eco_ids=fill(1, 40)))   # PITA mature (40 >= 20), QUNI (age 5) not
Pan.step!(HB, soaa3, 1, ctxa2)
rec_sp = Int[]; for i in 1:40, k in 1:Int(getsite(soaa3, i).live); getsite(soaa3, i).c_age[k] == 1 && push!(rec_sp, Int(getsite(soaa3, i).c_species[k])); end
@check !isempty(rec_sp) && all(==(1), rec_sp) "only the species with a mature cohort (PITA) recruits (got $(unique(rec_sp)))"

println("(b) dense sites: shade-intolerant species never establish; tolerance 5 at most 0.5")
ctxb, _ = estab_ctx(p=1.0)
nB = 600
soab = estab_on!(build_soa_ser([[(1, 30, 9000.0), (3, 40, 6000.0), (4, 25, 4000.0), (2, 12, 2500.0)] for _ in 1:nB]; eco_ids=fill(1, nB)))
Pan.step!(HB, soab, 1, ctxb)
cnt_b = zeros(Int, 6)
for i in 1:nB
  site = getsite(soab, i)
  for k in 1:Int(site.live); site.c_age[k] == 1 && (cnt_b[Int(site.c_species[k])] += 1); end
end
@printf("  max RD=%.3f min RD=%.3f ; shade classes: %s ; recruits per species (tol 1,2,3,4,5,5): %s of %d\n", maximum(soab.scalar.RD[1:nB]), minimum(soab.scalar.RD[1:nB]), string(unique(soab.scalar.shade_class[1:nB])), string(cnt_b), nB)
@check all(soab.scalar.shade_class[1:nB] .== 5) "dense site (RD > 0.75) -> shade class 5"
@check cnt_b[1] == 0 && cnt_b[2] == 0 && cnt_b[3] == 0 "tolerance 1, 2, 3 never establish at shade class 5"
@check abs(cnt_b[4] / nB - 0.25) < 0.06 "tolerance 4 at class 5: 0.25"
@check abs(cnt_b[5] / nB - 0.5) < 0.07 && abs(cnt_b[6] / nB - 0.5) < 0.07 "tolerance 5 at class 5: prob x 0.5 (got $(cnt_b[5] / nB), $(cnt_b[6] / nB))"

println("(c) no duplicate age-1 cohorts")
# direct: a cohort that becomes age 1 in this step meets the establishment of the same species -> merged
ctxc, _ = estab_ctx(p=1.0, mode=:default, mutate=prm -> hb_with(prm, SHADE_TOL=fill(UIntType(3), length(prm.SHADE_TOL))))
soac = estab_on!(build_soa_ser([[(2, 0, 30.0)] for _ in 1:10]; eco_ids=fill(1, 10)))     # PIEL cohort of age 0 -> age 1 after the step
tph_pre = Float64(getsite(soac, 1).c_tph[1]); dq_pre = Float64(getsite(soac, 1).c_dq[1])
Pan.step!(HB, soac, 1, ctxc)
nm = 0; allone = true
for i in 1:10
  site = getsite(soac, i)
  ones_ = [k for k in 1:Int(site.live) if site.c_species[k] == 2 && site.c_age[k] == 1]
  global allone; allone &= length(ones_) == 1
  global nm; nm += length(ones_)
end
site = getsite(soac, 1); k2 = findfirst(k -> site.c_species[k] == 2 && site.c_age[k] == 1, 1:Int(site.live))
@printf("  merged PIEL age-1 cohort: tph %.1f -> %.1f (incl. +750 recruits), dq %.2f -> %.2f\n", tph_pre, site.c_tph[k2], dq_pre, site.c_dq[k2])
@check allone "exactly one age-1 PIEL cohort per site after merge"
@check site.c_tph[k2] > 750 && site.c_tph[k2] < tph_pre + 750 + 1 && site.c_dq[k2] > 1.0 "merged density adds, dq is the quadratic mean"
# long run: no (species, age) duplicate anywhere
soac2 = estab_on!(build_soa_ser([Tuple{Int,Int,Float64}[] for _ in 1:30]; eco_ids=fill(1, 30)))
for y in 1:30; Pan.step!(HB, soac2, y, ctxa); end
dup = 0
for i in 1:30
  site = getsite(soac2, i); seen = Set{Tuple{UInt32,Float32}}()
  for k in 1:Int(site.live); key = (site.c_species[k], site.c_age[k]); key in seen && (global dup += 1); push!(seen, key); end
end
@check dup == 0 "no duplicate (species, age) cohorts over 30 yr of establishment (found $dup)"

println("(d) allocation-free steady state with establishment ON (draws consumed, no recruit) and determinism")
ctxd, _ = estab_ctx(p=0.0)
soad = estab_on!(build_soa_ser(specs3; eco_ids=[1 + (i - 1) % 3 for i in 1:40]))
for y in 1:3; Pan.step!(HB, soad, y, ctxd); end
nl0 = copy(soad.scalar.live)
ald = @allocated Pan.step!(HB, soad, 4, ctxd)
@check soad.scalar.live == nl0 "no change in cohort counts in the measured step"
@printf("  step! with establishment on, PROB_ESTAB=0, seed rain (all species draw): %d bytes\n", ald)
@check ald == 0 "zero allocations per annual step in steady state"
alloc_site(soad, 3, ctxd); al_sd = alloc_site(soad, 3, ctxd)
@check al_sd == 0 "site_year! with establishment: 0 allocations (got $al_sd)"
function run_estab(parallel::Bool; years=30)
  rng_ = MersenneTwister(77)
  specs = [rand(rng_) < 0.4 ? Tuple{Int,Int,Float64}[] : [(rand(rng_, 1:6), rand(rng_, 1:60), exp(rand(rng_) * 7 + 1))] for _ in 1:300]
  soa = estab_on!(build_soa_ser(specs; eco_ids=fill(1, 300)))
  ctx, _ = estab_ctx(p=0.3, mode=:jitter)
  soa.parallel = parallel
  for y in 1:years; Pan.step!(HB, soa, y, ctx); end
  return soa
end
sigE(soa) = hash((soa.refs.cohort, soa.csr.c_species, soa.csr.c_age, soa.csr.c_bio, soa.csr.c_tph, soa.csr.c_dq, soa.csr.c_h, soa.scalar.B, soa.scalar.BA, soa.scalar.shade_class, soa.scalar.live))
ea = run_estab(false); eb = run_estab(true); ec = run_estab(true)
@printf("ESTAB_HASH serial %s | threaded %s %s (nthreads=%d, live=%d cohorts)\n", string(sigE(ea), base=16), string(sigE(eb), base=16), string(sigE(ec), base=16), Threads.nthreads(), sum(ea.scalar.live))
@check sigE(ea) == sigE(eb) && sigE(eb) == sigE(ec) "establishment run byte-identical serial vs threaded vs repeat"
@check sum(ea.scalar.live) > 300 "establishment actually produced cohorts"

println("(e) plant! round trip (+ generic hook, BiomassSuccession, PnET)")
soae = estab_on!(build_soa_ser([[(1, 30, 3000.0)], Tuple{Int,Int,Float64}[], [(3, 10, 200.0), (4, 12, 300.0)]]; eco_ids=[1, 1, 1]))
Pan.step!(HB, soae, 0, estab_ctx(p=0.0, rain=false)[1])      # primes BA / SDI / RD (no establishment: p = 0)
Pan.plant_sites!(hctx0, soae, [(1, 5, 600.0), (2, 1, 1000.0), (2, 2, 400.0), (3, 6, 250.0)])   # eco1 local species: 5 = _S_HST, 6 = _H_LST
for (si, esp, tph_) in [(1, 5, 600.0), (2, 1, 1000.0), (2, 2, 400.0), (3, 6, 250.0)]
  site = getsite(soae, si); k = findfirst(k -> site.c_species[k] == esp && site.c_age[k] == 1, 1:Int(site.live))
  pp = pri.sp[eco_species_ids[1][esp]]
  @check k !== nothing && site.c_tph[k] == tph_ && site.c_dq[k] == 1.0 && rel(Float64(site.c_bio[k]), H.agb_of(pp, tph_, 1.0)) < 1e-5 "planted cohort state (site $si sp $esp)"
end
for si in 1:3
  site = getsite(soae, si)
  @check abs(Float64(site.B) - sum(Float64.(site.c_bio[1:Int(site.live)]))) < 1e-2 * max(1, Float64(site.B)) * 1e-3 + 1e-3 "site B == sum c_bio after planting (site $si)"
  ba_, sdi_, _ = H.stand_stats(soae.csr.c_species, soae.csr.c_tph, soae.csr.c_dq, Int(soae.refs.cohort[si]) - 1, Int(site.live), ctxa.HBAD.eco[1].sdimax)
  @check rel(Float64(site.BA), ba_) < 1e-4 && rel(Float64(site.SDI), sdi_) < 1e-4 "site BA / SDI updated incrementally (site $si)"
end
@check Int(getsite(soae, 2).live) == 2 && Int(getsite(soae, 1).live) == 2 "cohort counts after planting"
# second planting of the same species and age merges (no duplicate), with a different dq
Pan.plant_sites!(hctx0, soae, [(1, 5, 400.0)]; dq=3.0)
site = getsite(soae, 1)
k = findfirst(k -> site.c_species[k] == 5 && site.c_age[k] == 1, 1:Int(site.live))
@check Int(site.live) == 2 && count(j -> site.c_species[j] == 5 && site.c_age[j] == 1, 1:Int(site.live)) == 1 "same species+age merged"
@check site.c_tph[k] == 1000 && abs(site.c_dq[k] - sqrt((600 * 1.0 + 400 * 9.0) / 1000)) < 1e-5 "merge: densities add, dq quadratic mean ($(site.c_dq[k]))"
# planting with an explicit age and dq, then the model grows it
Pan.plant_sites!(hctx0, soae, [(2, 3, 300.0)]; age=8, dq=6.0)
site = getsite(soae, 2); k = findfirst(k -> site.c_species[k] == 3, 1:Int(site.live))
@check site.c_age[k] == 8 && site.c_dq[k] == 6.0 "plant! age and dq arguments"
ctxe, _ = estab_ctx(p=0.0, rain=false)
for y in 1:5; Pan.step!(HB, soae, y, ctxe); end
@check all(Float64(getsite(soae, i).B) > 0 && isfinite(Float64(getsite(soae, i).B)) for i in 1:3) "stand keeps running after planting"
# capacity guard
soag = estab_on!(build_soa_ser([[(1, 30, 3000.0)]]; eco_ids=[1]))
errg = try; HBADPlugin.plant!(hctx0, getsite(soag, 1), 2, 100.0); nothing; catch e; sprint(showerror, e); end
@check errg !== nothing && occursin("no free cohort slot", errg) "plant! without capacity errors clearly"
# BiomassSuccession: new cohort with LANDIS initial biomass; PnET: clear error
let bsprm = BSL.generate_biomass_params(BSL.BSTables(), species_list, eco_list, eco_species_ids; rng=MersenneTwister(2), no_establishment=false)
  bseco = Pan.generate_eco_params(BSL.BSTables(), bsprm)
  soabs = Pan.soa_type(Pan._BS)((cohort=Int32[1, 1], species=Int32[6, 6]))
  for i in 1:2
    st = getsite(soabs, i); st.active = true; st.rng = Random.Xoshiro(i); st.eco_id = 1; st.ecocode = 1; st.mapcode = i; st.ref_cn = i; st.live = 0; st.old = 0
    Pan._init_site_plugin!(Pan._BS, st, false)
  end
  getsite(soabs, 1).B = 0f0; getsite(soabs, 2).B = 12000f0
  Pan.plant_sites!(BSL.BSTables(), soabs, [(1, 2, 500.0), (2, 2, 500.0)]; age=1, eco_params=bseco)
  b1 = getsite(soabs, 1).c_bio[1]; b2 = getsite(soabs, 2).c_bio[1]
  e = bseco[1]
  @check b1 == trunc(BSL.calculate_initial_biomass(e.ANPP_MAX_SPP[2], 0f0, e.B_MAX_ECO)) && b2 == trunc(BSL.calculate_initial_biomass(e.ANPP_MAX_SPP[2], 12000f0, e.B_MAX_ECO)) "BS plant!: LANDIS initial biomass (trunc as add_cohort!)"
  @check b2 < b1 && getsite(soabs, 1).live == 1 && getsite(soabs, 1).c_species[1] == 2 "BS plant!: more shade -> smaller initial biomass; cohort recorded"
  @printf("  BS plant!: initial biomass open site %.0f g/m2, site B=12000 -> %.0f g/m2\n", b1, b2)
end
pnc_ = Pan.Plugins.PnETPlugin.PnETContext(; climate=Pan.Plugins.PnETPlugin.build_climate(["x"], [2000], zeros(Float32, 12, 1, 1), zeros(Float32, 12, 1, 1), zeros(Float32, 12, 1, 1), [400.0]))
errp = try; Pan.plant!(pnc_, nothing, 1, 100.0); nothing; catch e; sprint(showerror, e); end
@check errp !== nothing && occursin("not supported", errp) "PnET plant! errors with a clear message"

println("(f) 100-yr clear-cut trajectory with seed rain (eco 1: PITA PIEL QUNI ACRU _S_HST _H_LST; annual PROB_ESTAB default 0.108; mean of 200 replicate sites)")
ctxf, prmf = estab_ctx(p=H.annual_prob(0.6), rain=true, mode=:default)
nF = 200
soaf = estab_on!(build_soa_ser([Tuple{Int,Int,Float64}[] for _ in 1:nF]; eco_ids=fill(1, nF)))
for i in 1:nF; getsite(soaf, i).rng = Random.Xoshiro(1000 + i); end
soaf.parallel = true
names6 = ["PITA", "PIEL", "QUNI", "ACRU", "_S_HST", "_H_LST"]
@printf("  %4s %7s %7s %7s | %s | %6s %6s %6s\n", "yr", "TPA", "BA", "QMD", "BA share % " * join(lpad.(names6, 7), ""), "maxRD", "ncoh", "cls")
traj = Tuple[]
for y in 1:100
  Pan.step!(HB, soaf, y, ctxf)
  if y % 10 == 0 || y == 1 || y == 5
    tpa = 0.0; ba = 0.0; ssq = 0.0; bas = zeros(6); nco = 0
    for i in 1:nF
      site = getsite(soaf, i)
      for k in 1:Int(site.live)
        t = Float64(site.c_tph[k]); d = Float64(site.c_dq[k]); b = H._PI4E4 * d * d * t
        tpa += t; ba += b; ssq += t * d * d; bas[Int(site.c_species[k])] += b
      end
      nco += Int(site.live)
    end
    qmd = tpa > 0 ? sqrt(ssq / tpa) : 0.0
    @printf("  %4d %7.0f %7.2f %7.2f | %s | %6.3f %6.1f %6.2f\n", y, tpa / nF, ba / nF, qmd, join([@sprintf("%7.1f", 100 * x / max(ba, 1e-9)) for x in bas], ""),
      maximum(soaf.scalar.RD[1:nF]), nco / nF, mean(Float64.(soaf.scalar.shade_class[1:nF])))
    push!(traj, (y, tpa / nF, ba / nF, qmd, maximum(soaf.scalar.RD[1:nF])))
  end
end
@check all(t -> t[5] <= 1.03, traj) "RD stays <= 1.03 over the 100-yr run (thinning to 1, then the new recruits add <= ~0.7 % RD each)"
@check traj[end][3] > 5 && traj[end][3] < 55 "mean BA after 100 yr plausible (got $(traj[end][3]))"
s1 = getsite(soaf, 1)
@printf("  replicate 1 at yr 100: %d cohorts, BA=%.1f m2/ha, TPA=%.0f, shade class %d\n", Int(s1.live), s1.BA, sum(Float64.(s1.c_tph[1:Int(s1.live)])), Int(s1.shade_class))

println("(g) params round trip: JLD2, and upgrade of a pre-establishment JLD2 file")
using Pan.JLD2
let prm = H.generate_hbad_params(hctx0, species_list, eco_list, eco_species_ids; rng=MersenneTwister(3), mode=:jitter, seed_rain=true)
  f = tempname() * ".jld2"; JLD2.save_object(f, prm); q = JLD2.load_object(f); rm(f)
  @check q isa H.HBADParams && all(getfield(q, n) == getfield(prm, n) for n in fieldnames(H.HBADParams)) && q.ESTAB_SEED_RAIN "JLD2 round trip keeps every field incl. the establishment block"
  old = joinpath(@__DIR__, "..", "results", "hbad_gafl", "params_gafl_hbad.jld2")
  if isfile(old)
    raw = JLD2.load_object(old)
    up = H.upgrade_params(raw)
    @check up isa H.HBADParams && up.G_MULT == raw.G_MULT && up.SDIMAX == raw.SDIMAX && all(==(0.5f0), [up.SUFFICIENT_LIGHT[6][5]]) && !up.ESTAB_SEED_RAIN "pre-establishment params file upgraded with defaults"
    eco_ = H.generate_hbad_eco_params(H.HBADContext(String.(raw.SPECIES_LIST), raw.ECO_SPECIES_IDS; verbose=false), raw)       # the fallback method upgrades on the fly
    @check length(eco_) == length(raw.ECO_LIST) && all(isapprox.(eco_[1].min_rel_rd, (0.15, 0.30, 0.45, 0.60, 0.75); atol=1e-6)) "generate_hbad_eco_params(old file) works"
    @printf("  upgraded %s: %d ecos, %d species\n", basename(old), length(up.ECO_LIST), length(up.SPECIES_LIST))
  end
end

println("\nHBAD unit tests: $(npass[]) passed, $(nfail[]) failed")
nfail[] == 0 || exit(1)
