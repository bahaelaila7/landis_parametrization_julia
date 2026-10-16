# Loss functions in src/parametrize/utils.jl: the one loss implementation (calculate_site_loss2_acc! / calculate_species_loss_acc!, scratch-buffer
# based) and its allocating SiteLoss wrapper calculate_site_loss2, plus the allocating bin_ages / smoothen_bin_cdf_forgive against their in-place twins.
module UnitLoss
using Test, Random, Pan
const PU = Pan.PU
const PC = Pan.PanCore
const FT = PC.FloatType
const UT = PC.UIntType
const BaseSite = Pan.Plugins.BaseSitePlugin.BaseSite
const BS = Pan.Plugins.BiomassSuccessionPlugin.BiomassSuccession
const SoaBS = PC.SiteSoA{Tuple{BaseSite,BS}}

# The loss options are the fields of PU.LossParams (immutable, built per test); the reference normalisation is a PU.LossScales.

# ---------------------------------------------------------------- synthetic site + reference
function make_site(rng; ncoh=10, nsp=3, ages_max=90, drop_species=Int[])
  soa = SoaBS((cohort=Int32[ncoh], species=Int32[nsp]))
  s = PC.getsite(soa, 1)
  s.active = true; s.rng = Random.Xoshiro(1); s.mapcode = 1; s.ecocode = 1; s.eco_id = 1; s.ref_cn = 1; s.old = 0; s.live = ncoh
  s.sp_mature .= false
  cand = setdiff(1:nsp, drop_species)
  for k in 1:ncoh
    s.c_species[k] = UT(rand(rng, cand)); s.c_age[k] = FT(rand(rng, 1:ages_max)); s.c_bio[k] = FT(rand(rng) * 3000 + 10)
  end
  soa
end
function make_ref(rng, lp::PU.LossParams; nsp=3, keys=[true, true, false], maxage=100)
  recs = Dict{UT,PU.SPDFRecord}()
  nb = length(lp.age_bins.bins_idx) + (lp.age_bins.last_bin_open ? 1 : 0)
  for sp in 1:nsp
    keys[sp] || continue
    ages = zeros(FT, maxage + (length(lp.smoothing_weights) >> 1))
    for _ in 1:rand(rng, 1:6); ages[rand(rng, 1:maxage)] += FT(rand(rng) * 2500 + 5); end
    cdf = PU.smoothen_bin_cdf(ages; w=lp.smoothing_weights, age_bins=lp.age_bins)
    agb = PU.bin_ages(ages; age_bins=lp.age_bins.bins_idx, last_bin_open=lp.age_bins.last_bin_open)
    recs[UT(sp)] = PU.SPDFRecord(sp_agb_sum=sum(ages), sp_age_cdf=cdf, sp_age_agb=agb)
  end
  PU.SPDFGroundTruth(keys=BitVector(keys), records=recs)
end
lossparams(; bins=[10, 20, 40, 80], open=true, sw=1, kw...) = PU.LossParams(; age_bins=PU.AgeBins(bins_idx=bins, last_bin_open=open),
  smoothing_weights=sw == 1 ? FT[1] : PU.get_smoothing_window(smoothing_window=sw), lambda=FT(1.0), kw...)

function run_pair(soa, gt, lp, sc=PU.LossScales(); excluded=nothing, excl_c=nothing, nsp=3, esids=[[1, 2, 3]])
  site = PC.getsite(soa, 1)
  L = PU.calculate_site_loss2(5, site, nsp, esids, gt, lp, sc; excluded=excluded, excl_c=excl_c)
  w = zeros(FT, nsp); a = zeros(FT, nsp)
  sag = PU.calculate_site_loss2_acc!(w, a, 5, site, nsp, esids, gt, lp, sc, Int[], FT[], Bool[], FT[], FT[]; excluded=excluded, excl_c=excl_c)
  (L, w, a, sag)
end

# ---------------------------------------------------------------- bin_ages vs bin_ages!
@testset "bin_ages == bin_ages!" begin
  rng = Random.Xoshiro(11)
  for bins in ([5], [10, 20, 40, 80], [3, 6, 9, 12, 15, 18, 21, 24, 27, 30], collect(1:30)), open in (true, false), len in (1, 7, 30, 120)
    (!open && length(bins) < 2) && continue
    ages = FT.(rand(rng, len) .* 100)
    r = PU.bin_ages(ages; age_bins=bins, last_bin_open=open)
    out = FT[]; r2 = PU.bin_ages!(out, ages; age_bins=bins, last_bin_open=open)
    @test r2 === out && r == out && length(out) == length(bins) + (open ? 1 : 0)
    # `out` is reused with a stale content/size
    out2 = fill(FT(99), 3); PU.bin_ages!(out2, ages; age_bins=bins, last_bin_open=open); @test out2 == r
    # mass conservation: with an open last bin nothing is dropped
    open && @test sum(r) ≈ sum(ages) rtol = 1e-5
  end
  # semantics: element i (1-based position == integer age) goes to the first bin whose upper edge is > i; closed last bin drops the tail
  ages = FT[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  @test PU.bin_ages(ages; age_bins=[3, 6], last_bin_open=true) == FT[3, 12, 40]      # positions 1-2 | 3-5 | 6-10  (each upper edge belongs to the NEXT bin)
  @test PU.bin_ages(ages; age_bins=[3, 6], last_bin_open=false) == FT[3, 12]
  @test PU.bin_ages!(FT[], ages; age_bins=[3, 6], last_bin_open=false) == FT[3, 12]
  # bin_widths
  @test PU.get_bin_widths(age_bins=[10, 20, 40, 80], last_bin_open=true) == [10, 10, 20, 40]
  @test PU.get_bin_widths(age_bins=[10, 20, 40, 80], last_bin_open=false) == [10, 10, 20]
  @test PU.find_age_bin(5, PU.AgeBins(bins_idx=[10, 20], last_bin_open=true)) == 1
  @test PU.find_age_bin(10, PU.AgeBins(bins_idx=[10, 20], last_bin_open=true)) == 2
  @test PU.find_age_bin(500, PU.AgeBins(bins_idx=[10, 20], last_bin_open=true)) == 3
  @test PU.find_age_bin(500, PU.AgeBins(bins_idx=[10, 20], last_bin_open=false)) == 0
end

@testset "smoothen_bin_cdf_forgive == smoothen_bin_cdf_forgive!" begin
  rng = Random.Xoshiro(2)
  ab = lossparams().age_bins
  for hp in (0.0, 0.05), tr in 1:20
    lp = lossparams(; w_hinge_pct=FT(hp))
    p = FT.(rand(rng, 100) .* 50); obs = FT.(rand(rng, 5) .* 800)
    c1 = PU.smoothen_bin_cdf_forgive(p, obs, lp; age_bins=ab); c2 = PU.smoothen_bin_cdf_forgive!(FT[], p, obs, lp; age_bins=ab)
    @test c1 == c2
    @test issorted(c1) && (c1[end] == 1 || c1[end] == 0)
  end
end

# ---------------------------------------------------------------- site-level equivalence over option combos
@testset "calculate_site_loss2 == calculate_site_loss2_acc! (option sweep)" begin
  rng = Random.Xoshiro(2024)
  nchecked = 0
  setups = [(bins=[10, 20, 40, 80], open=true, sw=1), (bins=[10, 20, 40, 80], open=true, sw=3), (bins=[10, 20, 40, 80], open=false, sw=1),
            (bins=collect(2:60), open=true, sw=1)]    # last = "unbinned"-style per-year grid
  for st in setups, whinge in (false, true), ahinge in (false, true), softplus in (false, true), cbal in (false, true), cellnorm in (false, true)
    lp = lossparams(; st..., w_hinge=whinge, agb_hinge=ahinge, w_softplus=softplus, cbal_on=cbal, cell_norm=cellnorm,
                    agb_hinge_p=ahinge ? FT(2) : FT(1), w_normalize=!cellnorm, agb_normalize=!cellnorm, w_p=FT(2))
    nb = length(lp.age_bins.bins_idx) + (lp.age_bins.last_bin_open ? 1 : 0)
    # cbal: coarse bins = lp's own bins (identity map), non-trivial weights per (gsp, eco)
    cb = cbal ? PU.CbalTables(reshape([FT.(0.5 .+ rand(rng, nb)) for _ in 1:3], 3, 1), collect(1:length(lp.age_bins.bin_widths))) : PU.NO_CBAL
    sc = PU.LossScales(w_global=FT(3), agb_global=FT(1000), softplus_beta=FT(0.01), cbal=cb)
    for trial in 1:3
      drop = trial == 2 ? [2] : Int[]                                  # trial 2: species 2 in REF but absent from SIM
      soa = make_site(rng; ncoh=rand(rng, 4:14), drop_species=drop)
      gt = make_ref(rng, lp; keys=trial == 3 ? [true, true, true] : [true, true, false])
      L, w, a, sag = run_pair(soa, gt, lp, sc)
      @test L.sp_w_loss == w                                           # bit-identical (same arithmetic order)
      @test L.sp_agb_loss == a
      @test L.site_agb_loss == sag
      @test L.num_sites == 1 && L.num_obs == 1
      nchecked += 1
    end
  end
  @test nchecked == length(setups) * 2^5 * 3
end

@testset "excluded / excl_c / empty-site paths agree" begin
  rng = Random.Xoshiro(5)
  for whinge in (false, true), ahinge in (false, true)
    lp = lossparams(; w_hinge=whinge, agb_hinge=ahinge)
    begin
      for trial in 1:6
        soa = make_site(rng; ncoh=10); gt = make_ref(rng, lp; keys=[true, true, true])
        s = PC.getsite(soa, 1)
        # excluded eco-species
        L, w, a, sag = run_pair(soa, gt, lp; excluded=Set{UT}([UT(2)]))
        @test L.sp_w_loss == w && L.sp_agb_loss == a && L.site_agb_loss == sag
        @test w[2] == 0 && a[2] == 0                                                     # the excluded species contributes nothing (even when only in REF)
        # cohort-level exclusions: drop two actual (species, age) cohorts
        ex = Tuple{UT,FT}[(s.c_species[1], s.c_age[1]), (s.c_species[3], s.c_age[3])]
        L, w, a, sag = run_pair(soa, gt, lp; excl_c=ex)
        @test L.sp_w_loss == w && L.sp_agb_loss == a && L.site_agb_loss == sag
        # excluding EVERY cohort leaves all REF species as "absent from sim"
        exall = Tuple{UT,FT}[(s.c_species[k], s.c_age[k]) for k in 1:10]
        L, w, a, sag = run_pair(soa, gt, lp; excl_c=exall)
        @test L.sp_w_loss == w && L.sp_agb_loss == a && L.site_agb_loss == sag
        @test all(sp -> w[sp] ≈ sum(lp.age_bins.bin_widths .* gt.records[UT(sp)].sp_age_cdf[1:4]), 1:3)   # W = Σ bw·ref_cdf when sim is absent
        @test all(sp -> a[sp] ≈ gt.records[UT(sp)].sp_agb_sum, 1:3) || ahinge
        # empty site (live == 0)
        s.live = 0
        L, w, a, sag = run_pair(soa, gt, lp)
        @test L.sp_w_loss == w && L.sp_agb_loss == a && L.site_agb_loss == sag
      end
    end
  end
end

@testset "loss semantics (spot checks on hand-built input)" begin
  lp = lossparams()
  # one species, one cohort exactly matching the reference => W == 0 and (hinge-free sqrt form) AGB loss == 0
  soa = SoaBS((cohort=Int32[1], species=Int32[1])); s = PC.getsite(soa, 1)
  s.active = true; s.rng = Random.Xoshiro(1); s.eco_id = 1; s.live = 1; s.sp_mature .= false
  s.c_species[1] = UT(1); s.c_age[1] = FT(25); s.c_bio[1] = FT(500)
  ages = zeros(FT, 100); ages[25] = 500
  rec = PU.SPDFRecord(sp_agb_sum=FT(500), sp_age_cdf=PU.smoothen_bin_cdf(ages; w=lp.smoothing_weights, age_bins=lp.age_bins), sp_age_agb=PU.bin_ages(ages; age_bins=lp.age_bins.bins_idx, last_bin_open=true))
  gt = PU.SPDFGroundTruth(keys=BitVector([true]), records=Dict(UT(1) => rec))
  sc0 = PU.LossScales()
  begin
    L = PU.calculate_site_loss2(1, s, 1, [[1]], gt, lp, sc0)
    @test L.sp_w_loss == FT[0] && L.sp_agb_loss == FT[0] && L.site_agb_loss == 0
    # sim cohort in the wrong age bin: W = |Δcdf|·width summed
    s.c_age[1] = FT(55)
    L = PU.calculate_site_loss2(1, s, 1, [[1]], gt, lp, sc0)
    # cdf bins [10,20,40,80,open]; ref mass in bin 3 (age 25 -> bin [20,40)), sim in bin 4 ([40,80)): |ΔF| = 1 only in bin 3 => W = width3 = 20
    @test L.sp_w_loss[1] == FT(20)
    # AGB (non-hinge) term: per-bin (sqrt(sim) - sqrt(obs))^2 summed = 500 + 500
    @test L.sp_agb_loss[1] ≈ 1000 rtol = 1e-5
    # absent-from-sim species: W = Σ bw·ref_cdf, AGB loss = lambda · ref total
    s.live = 0
    L = PU.calculate_site_loss2(1, s, 1, [[1]], gt, lp, sc0)
    @test L.sp_w_loss[1] == sum(lp.age_bins.bin_widths .* rec.sp_age_cdf[1:4]) && L.sp_agb_loss[1] == 500 && L.site_agb_loss == 500
    # lambda scales the AGB term only
    lp2 = PU.LossParams(age_bins=lp.age_bins, smoothing_weights=lp.smoothing_weights, lambda=FT(0.25))
    L = PU.calculate_site_loss2(1, s, 1, [[1]], gt, lp2, sc0); @test L.sp_agb_loss[1] == 125
  end
end

@testset "calculate_species_loss_acc! accumulates (+=) into the per-species slots" begin
  rng = Random.Xoshiro(77)
  for sw in (1, 3), whinge in (false, true), ahinge in (false, true), trial in 1:4
    lp = lossparams(; sw=sw, w_hinge=whinge, agb_hinge=ahinge, agb_hinge_p=FT(2)); sc = PU.LossScales()
    soa = make_site(rng; ncoh=12); site = PC.getsite(soa, 1); gt = make_ref(rng, lp; keys=[true, false, true])
    p = sortperm(site.c_species); sp = site.c_species[p[1]]
    hi = findlast(==(sp), site.c_species[p])
    maxage = Int(ceil(maximum(site.c_age))) + (length(lp.smoothing_weights) >> 1)
    w = zeros(FT, 3); a = zeros(FT, 3)
    call!() = PU.calculate_species_loss_acc!(; sp=sp, gsp=Int(sp), site=site, ages=zeros(FT, maxage), p=p, sp_start_idx=1, sp_end_idx=hi, spdf_plt=gt, loss_params=lp, scales=sc,
      w_acc=w, agb_acc=a, site_agb_loss=FT(0), cdf=FT[], agb_bins=FT[])
    sag = call!(); w1 = copy(w); a1 = copy(a)
    call!()
    @test a == 2 .* a1 && w == 2 .* w1              # a second call adds the same amount again
    @test w1[Int(sp)] >= 0 && a1[Int(sp)] >= 0 && all(==(0), w1[setdiff(1:3, Int(sp))])
    @test sag == call!() && sag isa FT
  end
end

# ---------------------------------------------------------------- hinge, total loss, SiteLoss algebra
@testset "_hinge_relu" begin
  lpb(b) = lossparams(; agb_hinge_beta=FT(b))
  @test PU._hinge_relu(lpb(0), FT(3)) == 3 && PU._hinge_relu(lpb(0), FT(-2)) == 0 && PU._hinge_relu(lpb(0), FT(0)) == 0   # beta <= 0 => hard ReLU
  @test PU._hinge_relu(lpb(-5), FT(3)) == 3 && PU._hinge_relu(lpb(-5), FT(-2)) == 0
  h(x) = PU._hinge_relu(lpb(1), FT(x))                 # softplus_beta
  @test h(0) ≈ log(2) rtol = 1e-6                      # NOT 0 at the origin: softplus(0) = ln2/β
  @test isapprox(h(20), 20; atol=1e-6)
  @test isapprox(h(-20), 0; atol=1e-6)
  @test isapprox(h(1), log1p(exp(1.0)); rtol=1e-6)
  @test isapprox(h(-1), log1p(exp(-1.0)); rtol=1e-6)
  @test h(1) - h(-1) ≈ 1 atol = 1e-6                   # softplus identity: sp(x) - sp(-x) = x
  h4(x) = PU._hinge_relu(lpb(4), FT(x))
  @test isapprox(h4(0), log(2) / 4; rtol=1e-6)
  @test isapprox(h4(5), 5; atol=1e-3)
  @test issorted(h4.(-5:0.5:5))                        # monotone
  # thresholds: flat vs pct-of-obs clamped to [min,max]
  lp_flat = lossparams(; agb_hinge_threshold=FT(10), agb_hinge_pct=FT(0))
  @test PU._agb_hinge_thresh(lp_flat, FT(1000)) == 10
  lp_pct = lossparams(; agb_hinge_pct=FT(0.04), agb_hinge_pct_min=FT(10), agb_hinge_pct_max=FT(200))
  @test PU._agb_hinge_thresh(lp_pct, FT(100)) == 10 && PU._agb_hinge_thresh(lp_pct, FT(2000)) == 80 && PU._agb_hinge_thresh(lp_pct, FT(1e6)) == 200
  lpw = lossparams()
  @test PU._w_hinge_thresh(lpw, FT(100)) == 2 && PU._w_hinge_thresh(lpw, FT(300)) == 3 && PU._w_hinge_thresh(lpw, FT(1e5)) == 5   # default 1% clamped to [2,5]
  # power helpers
  @test PU._agb_pow(lossparams(; agb_hinge_p=FT(2)), FT(3)) == 9
  @test PU._agb_pow(lossparams(; agb_hinge_p=FT(1)), FT(3)) == 3
  @test PU._agb_pow(lossparams(; agb_hinge_p=FT(1.5)), FT(4)) ≈ 8
end

@testset "get_total_loss / SiteLoss algebra" begin
  mk(w, a, ns=1, no=1) = PU.SiteLoss(sp_w_loss=FT.(w), sp_agb_loss=FT.(a), site_agb_loss=FT(0), num_sites=ns, num_obs=no)
  l = mk([1, 2, 3], [10, 20], 4, 4)
  # default: (alpha*ΣW + beta*ΣAGB)/num_obs with alpha=loss_alpha(1), beta=1
  @test PU.get_total_loss(l, lossparams()) == (6 + 30) / 4
  @test PU.get_total_loss(l, lossparams(), FT(2)) == (6 + 2 * 30) / 4
  @test PU.get_total_loss(l, lossparams(; loss_alpha=FT(0.5)), FT(2)) == (0.5 * 6 + 2 * 30) / 4
  @test PU.get_total_loss(l, lossparams(; loss_alpha=FT(0))) == 30 / 4      # alpha = 0 keeps only the AGB term
  # normalized terms are already intensive => no /num_obs
  @test PU.get_total_loss(l, lossparams(; w_normalize=true)) == 36
  @test PU.get_total_loss(l, lossparams(; agb_normalize=true)) == 36
  # piecewise: sqrt below pivot, square above, on the AGGREGATES
  lpp = lossparams(; loss_piecewise=true, w_pivot=FT(4), agb_pivot=FT(10))
  @test PU.get_total_loss(l, lpp) ≈ (6 / 4)^2 + (30 / 10)^2     # ΣW=6 > 4 -> (6/4)^2 ; ΣAGB=30 > 10 -> 3^2
  @test PU._piecewise(FT(1), FT(4)) == 0.5 && PU._piecewise(FT(8), FT(4)) == 4 && PU._piecewise(FT(3), FT(0)) == 3
  # + and sum add every field (num_sites / num_obs included)
  a = mk([1, 2], [3, 4], 1, 2); b = mk([10, 20], [30, 40], 2, 3)
  c = a + b
  @test c.sp_w_loss == FT[11, 22] && c.sp_agb_loss == FT[33, 44] && c.num_sites == 3 && c.num_obs == 5
  s = sum([a, b, a])
  @test s.sp_w_loss == FT[12, 24] && s.num_sites == 4 && s.num_obs == 7
  @test a.sp_w_loss == FT[1, 2]                      # sum() copies, does not alias/mutate the first element
  @test_throws ErrorException sum(PU.SiteLoss[])
end

# ---------------------------------------------------------------- other loss helpers
@testset "wasserstein1d / smoothing / count-balance weights" begin
  @test PU.wasserstein1d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0
  @test PU.wasserstein1d([0.0], [2.0]) == 2
  @test PU.wasserstein1d([0.0, 1.0], [1.0, 2.0]) ≈ 1 && PU.wasserstein1d(Int[], [1.0]) == 0
  @test PU.wasserstein1d([3.0, 1.0], [1.0, 3.0]) == 0                  # order independent
  w = PU.get_smoothing_window(smoothing_window=3, smoothing_variance=FT(1))
  @test length(w) == 7 && sum(w) ≈ 1 && w == reverse(w) && argmax(w) == 4
  @test PU.smooth_ages(ages=FT[0, 0, 1, 0, 0], smoothing_window=FT[1]) == FT[0, 0, 1, 0, 0]       # window length 1 => identity
  sm = PU.smooth_ages(ages=FT[0, 0, 0, 5, 0, 0, 0], smoothing_window=w)
  @test sum(sm) ≈ 1 && argmax(sm) == 4                                                              # smoothing normalizes to unit mass (!) when mass > 0
  @test PU.smooth_ages(ages=zeros(FT, 7), smoothing_window=w) == zeros(FT, 7)
  # count-balance weights: occupied bins normalized to mean 1, empty bins neutral (1)
  import DataFrames
  cb = PU.AgeBins(bins_idx=[10, 20, 40], last_bin_open=true)
  df = DataFrames.DataFrame(species_id=[1, 1, 1, 1, 1, 1], eco_id=fill(1, 6), age_calc=[5, 5, 5, 5, 5, 30])
  begin
    tabs = PU.build_cbal_tables(df, cb, cb, [[1]], 1; beta=0.9)
    W = tabs.w[1, 1]
    @test length(W) == 4 && W[2] == 1 && W[4] == 1                       # bins 2 and 4 empty -> neutral
    @test (W[1] + W[3]) / 2 ≈ 1                                          # occupied bins have mean 1
    @test W[3] > W[1]                                                    # rarer bin (1 cohort vs 5) gets MORE weight
    @test tabs.coarse == [PU.find_age_bin(b - 1, cb) for b in cb.bins_idx]
  end
end
end # module
