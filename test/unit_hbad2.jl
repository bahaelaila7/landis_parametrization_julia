# HBAD2 plugin unit tests: allocation-free steady state, determinism (state hash independent of repetition and of the thread setting of the site loop),
# monotone N, cohort add / remove / swap consistency, plant! (D 2.54 cm, H 3 m, age 1, vigour expansion), parameter loading / species-key mapping, the tree loader constants.
module UnitHBAD2
using Test, Random, Pan, SHA
const HB2 = Pan.Plugins.HBAD2Plugin
const PC = Pan.PanCore
const KEYS = ["ACRU", "LIST2", "LITU", "MAVI2", "NYBI", "PIEL", "PITA", "QULA3", "QUNI", "TAAS", "_H", "_S"]
const NSP = length(KEYS)

function setup(; ns=24, no_est=false, seed=3)
  pcfg = Pan.PluginConfig(HB2.HBAD2Context(KEYS, [collect(1:NSP)]; verbose=false))
  rng = Random.Xoshiro(seed)
  off = [1]; sp = Int[]; N = Float64[]; D = Float64[]; H = Float64[]; A = Float64[]; zq = Float64[]
  VZ = (-sqrt(3.0), 0.0, sqrt(3.0)); VW = (1 / 6, 2 / 3, 1 / 6)
  for s in 1:ns
    for _ in 1:(s % 5 == 0 ? 0 : rand(rng, 1:6))
      k = rand(rng, 1:NSP); n = 20 + 800 * rand(rng); d = 3 + 40 * rand(rng)^2; h = 3 + 0.9 * d^0.8; a = 1 + 100 * rand(rng)
      for q in 1:3; push!(sp, k); push!(N, n * VW[q]); push!(D, d); push!(H, h); push!(A, a); push!(zq, VZ[q]); end
    end
    push!(off, length(sp) + 1)
  end
  soa = Pan.make_sites_hbad2_arrays(pcfg, off, sp, A, N, D, H, zq; rng=Random.Xoshiro(1), no_establishment=no_est)
  params = Pan.generate_params(pcfg.ctx, KEYS, ["eco"], [collect(1:NSP)])
  ctx = Pan.make_ctx(pcfg.ctx, Pan.generate_eco_params(pcfg.ctx, params))
  return pcfg, soa, ctx
end
function statehash(soa)
  io = IOBuffer()
  for k in (:c_species, :c_age, :c_tph, :c_dq, :c_h, :c_zq, :c_bio), x in (getfield(soa.csr, k),); write(io, x); end
  write(io, soa.scalar.live); write(io, soa.scalar.B)
  bytes2hex(sha256(take!(io)))
end

@testset "hbad2 plugin" begin
  @testset "parameters and species keys" begin
    t = HB2.load_v2_tables()
    @test length(t.keys) == 12 && t.glob.form == 2 && t.glob.σv == 0.7 && t.glob.mA == 10.0
    @test t.keys == KEYS
    pcfg = Pan.PluginConfig(HB2.HBAD2Context(["PITA", "QUAL", "_H", "_S", "_GRP_30_X", "FOO"], [collect(1:6)]; verbose=false))
    @test pcfg.ctx.species_key == ["PITA", "_H", "_H", "_S", "_H", "_H"]
    @test Pan.plugin_type(pcfg) === HB2.HBAD2
    @test_throws ErrorException Pan.make_param_dists(pcfg.ctx, 6, 1, [collect(1:6)])
    @test Pan.Data.TREE_VIG_Z == HB2.VIG_Z && Pan.Data.TREE_VIG_W == HB2.VIG_W
  end

  @testset "allocation-free steady state, determinism, monotone N" begin
    for no_est in (true, false)
      pcfg, soa, ctx = setup(; no_est=no_est)
      SP = Pan.succession_plugin_of(soa)
      soa.parallel = false
      for y in 1:12; Pan.step!(SP, soa, y, ctx); end      # warm up; recruit classes open at years 1 and 6, 11
      # a year in which no recruit class opens (classes open every RCLASS = 5 years: year 13 is neither 12+1..): find one
      Pan.step!(SP, soa, 13, ctx)
      a = @allocated Pan.step!(SP, soa, 14, ctx)
      @test a == 0
      # determinism: repeated run and serial vs threaded site loop give the same hash
      pa, sa, ca = setup(; no_est=no_est); pb, sb, cb = setup(; no_est=no_est); sb.parallel = false
      for y in 1:20; Pan.step!(SP, sa, y, ca); Pan.step!(SP, sb, y, cb); end
      @test statehash(sa) == statehash(sb)
      pc2, sc2, cc2 = setup(; no_est=no_est)
      for y in 1:20; Pan.step!(SP, sc2, y, cc2); end
      @test statehash(sc2) == statehash(sa)
      # N only decreases for cohorts that exist (recruit classes accumulate while open: their N may rise only within RCLASS years of opening)
      p0, s0, c0 = setup(; no_est=true)
      prev = copy(s0.csr.c_tph); live0 = copy(s0.scalar.live)
      for y in 1:10
        Pan.step!(SP, s0, y, c0)
        @test all(s0.csr.c_tph .<= prev .* (1 + 1e-15))
        prev = copy(s0.csr.c_tph)
      end
    end
    pcfg, soa, ctx = setup(; no_est=false)
    SP = Pan.succession_plugin_of(soa)
    n0 = copy(soa.scalar.live)
    for y in 1:30; Pan.step!(SP, soa, y, ctx); end
    @test all(soa.scalar.live .>= n0)
    # N rises only in open recruit classes (c_ropen > 0)
    prev = Dict{Tuple{Int,Int},Float64}()
    p1, s1, c1 = setup(; no_est=false)
    for y in 1:12
      Pan.step!(SP, s1, y, c1)
      for s in 1:s1.n
        o = Int(s1.refs.cohort[s]) - 1
        for k in 1:Int(s1.scalar.live[s])
          key = (s, k); v = s1.csr.c_tph[o+k]
          haskey(prev, key) && s1.csr.c_ropen[o+k] == 0 && @test v <= prev[key] * (1 + 1e-15)
          prev[key] = v
        end
      end
    end
  end

  @testset "cohort add / remove / swap keep every field consistent" begin
    pcfg, soa, ctx = setup(; no_est=true)
    site = Pan.getsite(soa, 1)
    n = Int(site.live); @test n >= 3
    snap(k) = (site.c_species[k], site.c_age[k], site.c_tph[k], site.c_dq[k], site.c_h[k], site.c_zq[k], site.c_bio[k], site.c_O[k], site.c_SDI[k], site.c_V[k], site.c_ropen[k])
    last_ = snap(n)
    PC.swap_cohort!(site, 1, n)
    @test snap(1) == last_
    site.live = n - 1                                   # swap-remove
    @test Int(site.live) == n - 1
    # add_cohort! (biomass-driven, one mean-field cohort) needs a free slot: grow by one
    cnt = soa.scalar._new_cohort_counts; cnt .= soa.scalar.live; cnt[1] += 1
    PC.readjust_soa!(soa, (cohort=cnt,))
    site = Pan.getsite(soa, 1)
    Pan.add_cohort!(pcfg.ctx, site, 2, 30.0, 4000.0)
    k = Int(site.live)
    @test site.c_species[k] == 2 && site.c_age[k] == 30.0 && site.c_zq[k] == 0.0 && Float64(site.c_bio[k]) ≈ 4000.0
    @test site.c_tph[k] > 0 && site.c_dq[k] > 0 && site.c_h[k] > 1.3
    Pan.set_cohort_bio!(pcfg.ctx, site, k, 2000.0)
    @test Float64(site.c_bio[k]) ≈ 2000.0
    # the kernel keeps running on the modified site
    SP = Pan.succession_plugin_of(soa)
    params = Pan.generate_params(pcfg.ctx, KEYS, ["eco"], [collect(1:NSP)])
    cx = Pan.make_ctx(pcfg.ctx, Pan.generate_eco_params(pcfg.ctx, params))
    Pan.step!(SP, soa, 1, cx)
    @test all(isfinite, soa.csr.c_tph) && all(isfinite, soa.csr.c_dq)
  end

  @testset "plant!" begin
    pcfg, soa, ctx = setup(; no_est=true)
    si = 5                                              # empty site (s % 5 == 0)
    @test soa.scalar.live[si] == 0
    @test Pan.plant_slots_reserved(pcfg.ctx) == 3
    Pan.plant_sites!(pcfg.ctx, soa, [(si, 7, 1000.0)])
    site = Pan.getsite(soa, si)
    @test Int(site.live) == 3
    @test sort(collect(site.c_zq)) == [-sqrt(3.0), 0.0, sqrt(3.0)]
    @test sum(site.c_tph) ≈ 1000.0
    @test all(==(2.54), site.c_dq) && all(==(3.0), site.c_h) && all(==(1.0), site.c_age) && all(==(UInt32(7)), site.c_species)
    @test sort(collect(site.c_tph)) ≈ sort([1000 / 6, 1000 * 2 / 3, 1000 / 6])
    @test Pan.plant_slot_needed(pcfg.ctx, site, 7, 1) == 0 && Pan.plant_slot_needed(pcfg.ctx, site, 8, 1) == 3
    Pan.plant_sites!(pcfg.ctx, soa, [(si, 7, 500.0)])    # merges into the existing sub-cohorts
    site = Pan.getsite(soa, si)
    nl = Int(site.live)   # capacity may exceed live (the harness reserves 3 slots per planting)
    @test nl == 3 && sum(site.c_tph[1:nl]) ≈ 1500.0 && all(≈(2.54), site.c_dq[1:nl])
    # and the planted trees then grow
    SP = Pan.succession_plugin_of(soa)
    params = Pan.generate_params(pcfg.ctx, KEYS, ["eco"], [collect(1:NSP)])
    cx = Pan.make_ctx(pcfg.ctx, Pan.generate_eco_params(pcfg.ctx, params))
    Pan.step!(SP, soa, 1, cx)
    site = Pan.getsite(soa, si)
    @test all(site.c_dq .> 2.54) && all(site.c_age .== 2.0)
  end
end
end
