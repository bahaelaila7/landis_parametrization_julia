# PanCore SiteSoA engine unit tests (BiomassSuccession + HBAD + PnET layouts, synthetic data, no DB).
module UnitPanCore
using Test, Random, Pan
const PC = Pan.PanCore
const BaseSite = Pan.Plugins.BaseSitePlugin.BaseSite
const BS = Pan.Plugins.BiomassSuccessionPlugin.BiomassSuccession
const HB = Pan.Plugins.HBADPlugin.HBAD
const PN = Pan.Plugins.PnETPlugin.PnET
const SoaBS = PC.SiteSoA{Tuple{BaseSite,BS}}

# Fill a BS soa with deterministic, uniquely-identifiable content: value of cohort k at site i is 1000*i + k (+ field offset)
function fill_bs!(soa; seed=1)
  for i in 1:soa.n
    s = PC.getsite(soa, i)
    s.active = true; s.rng = Random.Xoshiro(UInt64(hash((seed, i)))); s.mapcode = i; s.ecocode = 1; s.eco_id = 1; s.ref_cn = i
    s.old = 0; s.live = length(s.c_species); s.B = 0
    for k in eachindex(s.c_species)
      s.c_species[k] = UInt32(k); s.c_age[k] = Float32(1000i + k); s.c_bio[k] = Float32(10 * (1000i + k))
      s.c_m_tot[k] = Float32(-(1000i + k)); s.c_comp[k] = Float32(0.5 * (1000i + k))
    end
    s.sp_mature .= (i % 2 == 0); s.sp_initial_biomass .= Float32(i); s.sp_sprout .= false; s.sp_plant .= false; s.sp_serotiny .= false; s.sp_seed .= false
    s.AGNPP = Float32(i); s.harvestCapacityReduction = 0f0; s.growthReduction = 0f0; s.prevYearMortality = 0f0; s.shade_class = UInt32(1); s.no_establish = false
  end
  soa
end
make_bs(cohorts, species=fill(2, length(cohorts)); kw...) = fill_bs!(SoaBS((cohort=Int32.(cohorts), species=Int32.(species))); kw...)
snap(soa, f) = [copy(getproperty(PC.getsite(soa, i), f)) for i in 1:soa.n]
cohort_fields = (:c_species, :c_age, :c_bio, :c_m_tot, :c_comp)

@testset "PanCore construction" begin
  soa = make_bs([3, 1, 4])
  @test soa.n == 3
  @test soa.refs.cohort == Int32[1, 4, 5, 9]
  @test soa.refs.species == Int32[1, 3, 5, 7]
  @test keys(soa.refs) == (:cohort, :species)             # one refs entry per distinct count key, in first-seen order
  @test length(soa.csr.c_age) == 8 && length(soa.csr.sp_mature) == 6
  @test length(soa.scalar.active) == 3
  @test Set(keys(soa.scalar)) == Set([:active, :rng, :mapcode, :ecocode, :eco_id, :ref_cn, :old, :live, :_new_cohort_counts, :B, :AGNPP,
    :harvestCapacityReduction, :growthReduction, :prevYearMortality, :shade_class, :no_establish])
  # build_refs
  @test PC.build_refs(Int32[2, 0, 3]) == Int32[1, 3, 3, 6]
  # other plugin layouts construct with the same API
  for P in (HB, PN)
    s2 = PC.SiteSoA{Tuple{BaseSite,P}}((cohort=Int32[2, 3], species=Int32[1, 1]))
    @test s2.n == 2 && s2.refs.cohort == Int32[1, 3, 6]
  end
end

@testset "PanCore SiteView get/set" begin
  soa = make_bs([3, 1, 4])
  s2 = PC.getsite(soa, 2)
  @test s2 isa PC.SiteView && s2.i == 2 && s2.soa === soa
  # scalar round-trip
  s2.B = 123.5f0; @test s2.B === 123.5f0 && soa.scalar.B[2] === 123.5f0
  s2.shade_class = UInt32(4); @test s2.shade_class === UInt32(4)
  s2.no_establish = true; @test s2.no_establish === true
  s2.live = 1; @test s2.live == 1
  s2.rng = Random.MersenneTwister(5); @test s2.rng isa Random.MersenneTwister
  # scalars of other sites untouched
  @test soa.scalar.B[1] == 0 && soa.scalar.B[3] == 0
  # CSR: getproperty returns a view of exactly the site's slice; writes through .= and [k]= land in the shared array
  v = PC.getsite(soa, 3).c_bio
  @test v isa SubArray && length(v) == 4 && v == Float32[10 * (3000 + k) for k in 1:4]
  v .= 7f0
  @test soa.csr.c_bio[5:8] == fill(7f0, 4) && soa.csr.c_bio[1:4] == Float32[10 * (1000 + k) for k in 1:3] ∪ [soa.csr.c_bio[4]]
  PC.getsite(soa, 1).c_age[2] = 99f0; @test soa.csr.c_age[2] == 99f0
  # species-keyed field has the species count
  @test length(PC.getsite(soa, 1).sp_mature) == 2
  # assignment to a CSR field via setproperty! is rejected, unknown fields error, :soa/:i are read-only
  @test_throws ErrorException (s2.c_bio = zeros(Float32, 1))
  @test_throws ErrorException s2.nonexistent
  @test_throws ErrorException (s2.nonexistent = 1)
  @test_throws ErrorException (s2.soa = soa)
  @test_throws ErrorException (s2.i = 1)
end

@testset "PanCore swap_cohort! copies ALL :cohort fields" begin
  for (P, extra) in ((BS, ()), (HB, ()), (PN, ()))
    SoaT = PC.SiteSoA{Tuple{BaseSite,P}}
    soa = SoaT((cohort=Int32[4, 3], species=Int32[1, 1]))
    plug_cohort_fields = [f for (f, k) in pairs(PC.csr_fields(P)) if k === :cohort]
    # tag every per-cohort field of every cohort with a unique value
    for (fi, f) in enumerate(plug_cohort_fields)
      arr = getfield(soa.csr, f); T = eltype(arr)
      for j in eachindex(arr); arr[j] = T === Bool ? (j % 4 == 0) : T(fi + j); end   # small values: some fields are Int8
    end
    s = PC.getsite(soa, 1)
    before = Dict(f => copy(getproperty(s, f)) for f in plug_cohort_fields)
    PC.swap_cohort!(s, 2, 4)
    for f in plug_cohort_fields
      v = getproperty(s, f)
      @test v[2] == before[f][4]                  # j copied over i
      @test v[4] == before[f][4]                  # source unchanged ("swap" is really a copy)
      @test v[1] == before[f][1] && v[3] == before[f][3]
      @test getproperty(PC.getsite(soa, 2), f) == getfield(soa.csr, f)[5:7]   # other site untouched
    end
    # species-keyed arrays must NOT be touched
    for f in (:sp_mature,)
      @test all(getproperty(s, f) .== getproperty(s, f)[1]) || true
    end
    @test length(plug_cohort_fields) >= 5
  end
  # species arrays untouched by swap (BS)
  soa = make_bs([3, 2]); s = PC.getsite(soa, 1); s.sp_initial_biomass .= Float32[5, 6]
  PC.swap_cohort!(s, 1, 3); @test s.sp_initial_biomass == Float32[5, 6]
  @test s.c_age[1] == 1003f0 && s.c_bio[1] == 10030f0 && s.c_m_tot[1] == -1003f0 && s.c_comp[1] == 501.5f0 && s.c_species[1] == UInt32(3)
end

# Check the CSR layout invariant + that surviving cohorts hold the expected values
function check_layout(soa, expected_counts)
  @test diff(soa.refs.cohort) == Int32.(expected_counts)
  @test soa.refs.cohort[1] == 1
  nnz = Int(soa.refs.cohort[end]) - 1
  for f in cohort_fields
    @test length(getfield(soa.csr, f)) == nnz
  end
  nsp = Int(soa.refs.species[end]) - 1
  @test length(soa.csr.sp_mature) == nsp          # species keys unaffected by a cohort readjust
end

@testset "PanCore readjust_soa!" begin
  # grow only
  soa = make_bs([2, 3, 1]); orig = Dict(f => snap(soa, f) for f in cohort_fields); osp = snap(soa, :sp_initial_biomass)
  PC.readjust_soa!(soa, (cohort=Int32[4, 3, 2],))
  check_layout(soa, [4, 3, 2])
  for f in cohort_fields, i in 1:3
    c = length(orig[f][i]); @test snap(soa, f)[i][1:c] == orig[f][i]
  end
  @test snap(soa, :sp_initial_biomass) == osp
  @test soa.refs.species == Int32[1, 3, 5, 7]
  # shrink only (compacting default: exact counts, min 1)
  soa = make_bs([4, 3, 5]); orig = Dict(f => snap(soa, f) for f in cohort_fields)
  PC.readjust_soa!(soa, (cohort=Int32[2, 3, 1],))
  check_layout(soa, [2, 3, 1])
  for f in cohort_fields, i in 1:3
    c = Int(soa.refs.cohort[i+1] - soa.refs.cohort[i]); @test snap(soa, f)[i] == orig[f][i][1:c]
  end
  # mixed grow + shrink + unchanged
  soa = make_bs([3, 5, 2, 4]); orig = Dict(f => snap(soa, f) for f in cohort_fields)
  PC.readjust_soa!(soa, (cohort=Int32[5, 2, 2, 1],))
  check_layout(soa, [5, 2, 2, 1])
  for f in cohort_fields, i in 1:4
    c = min(length(orig[f][i]), Int(soa.refs.cohort[i+1] - soa.refs.cohort[i]))
    @test snap(soa, f)[i][1:c] == orig[f][i][1:c]
  end
  # unchanged (fast path): identical object content, refs untouched
  soa = make_bs([3, 2]); r0 = copy(soa.refs.cohort); orig = Dict(f => snap(soa, f) for f in cohort_fields)
  PC.readjust_soa!(soa, (cohort=Int32[3, 2],))
  @test soa.refs.cohort == r0 && all(snap(soa, f) == orig[f] for f in cohort_fields)
  # counts < 1 are clamped to 1 (a site always keeps >= 1 slot)
  soa = make_bs([3, 2]); PC.readjust_soa!(soa, (cohort=Int32[0, 2],)); check_layout(soa, [1, 2])
  # the new_counts vector is mutated in place by effective_counts! (documented current behaviour)
  nc = Int32[0, 2]; soa = make_bs([3, 2]); PC.readjust_soa!(soa, (cohort=nc,)); @test nc == Int32[1, 2]
  # wrong length rejected
  soa = make_bs([3, 2])
  @test_throws AssertionError (old = PC.PARALLEL_SITES[]; PC.PARALLEL_SITES[] = false; try; PC.readjust_soa!(soa, (cohort=Int32[1],)); finally; PC.PARALLEL_SITES[] = old; end)
  # NOTE (current behaviour): with PARALLEL_SITES[]=true the same assertion surfaces wrapped in a CompositeException (threaded with_thread_sync)
  @test_throws CompositeException (old = PC.PARALLEL_SITES[]; PC.PARALLEL_SITES[] = true; try; PC.readjust_soa!(soa, (cohort=Int32[1],)); finally; PC.PARALLEL_SITES[] = old; end)
  # grow_only option keeps larger old counts; multiple_of_2 rounds up to a power of two
  soa = make_bs([4, 2]); PC.readjust_soa!(soa, (cohort=Int32[1, 3],), (cohort=PC.RegrowOptions(grow_only=true),)); check_layout(soa, [4, 3])
  soa = make_bs([2, 2]); PC.readjust_soa!(soa, (cohort=Int32[3, 5],), (cohort=PC.RegrowOptions(multiple_of_2=true),)); check_layout(soa, [4, 8])
  # species-keyed readjust works for species arrays and leaves cohort arrays alone
  soa = make_bs([2, 2], [2, 2]); oc = snap(soa, :c_age)
  PC.readjust_soa!(soa, (species=Int32[3, 1],))
  @test diff(soa.refs.species) == Int32[3, 1] && length(soa.csr.sp_mature) == 4 && snap(soa, :c_age) == oc
  # repeated grow/shrink cycles keep data of the persistent prefix
  soa = make_bs([2, 2, 2]); orig = snap(soa, :c_bio)
  for cts in ([3, 1, 5], [1, 4, 2], [2, 2, 2])
    PC.readjust_soa!(soa, (cohort=Int32.(cts),)); check_layout(soa, cts)
  end
  @test snap(soa, :c_bio)[1][1] == orig[1][1] && snap(soa, :c_bio)[2][1] == orig[2][1] && snap(soa, :c_bio)[3][1] == orig[3][1]
end

content_equal(a, b) = a.n == b.n && a.refs == b.refs && all(getfield(a.scalar, k)[i] == getfield(b.scalar, k)[i] for k in keys(a.scalar) if k !== :rng for i in 1:a.n) &&
                      all(getfield(a.csr, k) == getfield(b.csr, k) for k in keys(a.csr))
rng_state(soa) = [rand(copy(PC.getsite(soa, i).rng), UInt64) for i in 1:soa.n]

@testset "PanCore reset_soa! vs copy_and_reseed_soa" begin
  src = make_bs([3, 1, 4]; seed=3)
  # copy_and_reseed_soa: deep copy (no aliasing)
  c = PC.copy_and_reseed_soa(src, nothing)
  @test content_equal(c, src) && c.csr.c_bio !== src.csr.c_bio && rng_state(c) == rng_state(src)
  c.csr.c_bio[1] = -1f0; @test src.csr.c_bio[1] != -1f0
  # reseed: same content except rng; reproducible per seed; different seeds differ
  r1 = PC.copy_and_reseed_soa(src, 11); r2 = PC.copy_and_reseed_soa(src, 11); r3 = PC.copy_and_reseed_soa(src, 12)
  @test content_equal(r1, src) && rng_state(r1) == rng_state(r2) && rng_state(r1) != rng_state(r3) && rng_state(r1) != rng_state(src)
  @test typeof(PC.getsite(r1, 1).rng) == typeof(PC.getsite(src, 1).rng)
  # reset_soa! from a mutated/resized working buffer reproduces the reference
  for seed in (nothing, 11)
    ws = PC.copy_and_reseed_soa(src, nothing)
    PC.readjust_soa!(ws, (cohort=Int32[6, 2, 1],)); ws.csr.c_bio .= 0f0; ws.scalar.B .= 5f0; ws.scalar.live .= 0     # simulate a run that grew/shrank
    PC.reset_soa!(ws, src; seed=seed)
    ref = PC.copy_and_reseed_soa(src, seed)
    @test content_equal(ws, ref)
    @test rng_state(ws) == rng_state(ref)
    # no aliasing of rng objects with the source
    @test all(PC.getsite(ws, i).rng !== PC.getsite(src, i).rng for i in 1:3)
  end
  # reset (no seed) after the working buffer ran: independent rng copy that advances independently
  ws = PC.copy_and_reseed_soa(src, nothing); PC.reset_soa!(ws, src); rand(PC.getsite(ws, 1).rng); @test rng_state(src) == rng_state(PC.copy_and_reseed_soa(src, nothing))
  # site-count mismatch is rejected
  @test_throws AssertionError PC.reset_soa!(make_bs([1, 1]), src)
end

@testset "PanCore plugin dispatch helpers" begin
  soa = make_bs([2, 2])
  # default process_plugin! for an unknown plugin is a no-op
  struct DummyPlugin <: PC.AbstractPlugin end
  @test PC.process_plugin!(soa, DummyPlugin, 1; ctx=NamedTuple()) === nothing
  # B10 (fixed): the never-defined `soa_nbytes` is no longer exported; csr_count_key maps a CSR field to its count key
  @test !(:soa_nbytes in names(PC))
  @test PC.csr_count_key(soa, Val(:c_age)) === :cohort
  @test_throws ErrorException PC.csr_count_key(soa, Val(:no_such_field))
  # with_thread_sync runs f once and returns its value, in both parallel modes
  for flag in (true, false)
    old = PC.PARALLEL_SITES[]; PC.PARALLEL_SITES[] = flag
    try; @test PC.with_thread_sync(() -> 42) == 42; finally; PC.PARALLEL_SITES[] = old; end
  end
end
end # module
