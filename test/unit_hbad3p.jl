# HBAD3P plugin unit tests: parameters / key mapping / roster, vigour quadrature, allocation-free steady state, determinism (serial vs threaded site loop), monotone N, cohort add / swap / set_cohort_bio,
# plant!, management helpers with the forest-area convention (reported values x area), the cleanup option, the free-run machinery (recruitment through process_plugin!).
module UnitHBAD3P
using Test, Random, Pan, SHA
const H3 = Pan.Plugins.HBAD3PPlugin
const PC = Pan.PanCore
const KEYS = ["ACRU", "LIST2", "LITU", "MAVI2", "NYBI", "PIEL", "PITA", "QULA3", "QUNI", "TAAS", "_H", "_S"]
const NSP = length(KEYS)
const YAML_PATH = joinpath(dirname(@__DIR__), "results", "params_yaml", "r4_hbad3p_gafl.yaml")

function setup(; ns=24, no_est=false, seed=3, area=nothing, rmult=nothing)
  pcfg = Pan.PluginConfig(H3.HBAD3PContext(KEYS, [collect(1:NSP)]; verbose=false))
  rng = Random.Xoshiro(seed)
  off = [1]; sp = Int[]; N = Float64[]; D = Float64[]; H = Float64[]; A = Float64[]; zq = Float64[]; CR = Float64[]
  for s in 1:ns
    for _ in 1:(s % 5 == 0 ? 0 : rand(rng, 1:6))
      k = rand(rng, 1:NSP); n = 20 + 800 * rand(rng); d = 3 + 40 * rand(rng)^2; h = 3 + 0.9 * d^0.8; a = 1 + 100 * rand(rng); cr = 0.2 + 0.6rand(rng)
      for q in 1:5; push!(sp, k); push!(N, n * H3.VIG_W[q]); push!(D, d); push!(H, h); push!(A, a); push!(zq, H3.VIG_Z[q]); push!(CR, cr); end
    end
    push!(off, length(sp) + 1)
  end
  soa = Pan.make_sites_hbad3p_arrays(pcfg, off, sp, A, N, D, H, zq, CR; rng=Random.Xoshiro(1), no_establishment=no_est, area=area, rmult=rmult,
                                     W=0.6 .+ 0.4 .* rand(rng, ns), Wn=0.8, seedplot=BitMatrix(rand(rng, Bool, NSP, ns)))
  params = Pan.generate_params(pcfg.ctx, KEYS, ["eco"], [collect(1:NSP)])
  ctx = Pan.make_ctx(pcfg.ctx, Pan.generate_eco_params(pcfg.ctx, params))
  return pcfg, soa, ctx
end
function statehash(soa)
  io = IOBuffer()
  for k in (:c_species, :c_age, :c_tph, :c_dq, :c_h, :c_zq, :c_cr, :c_bio), x in (getfield(soa.csr, k),); write(io, x); end
  write(io, soa.scalar.live); write(io, soa.scalar.B)
  bytes2hex(sha256(take!(io)))
end

@testset "hbad3p plugin" begin
  @testset "parameters, species keys, roster" begin
    t = H3.load_3p_tables()
    @test t.keys == KEYS && t.farea == 1 && length(t.sp) == 12
    @test t.glob.sdimax == 1500.0 && t.glob.mA == 10.0 && t.glob.σv == 0.5101870943473263 && t.glob.μ == 2.0 && fieldcount(H3.GlobPP) == 32
    @test t.rec.λ0 == 3.500574838797933 && t.rec.ε0 == 0.3069224690625089
    @test t.sp[1].lnε == log(1.808055984465558) && t.sp[1].Hmax == 29.260800000000003 && t.sp[1].ST == 4.0 && t.sp[1].longevity == 150.0 && t.sp[1].lnF == -2.43560757695965
    pcfg = Pan.PluginConfig(H3.HBAD3PContext(["PITA", "QUAL", "_H", "_S", "_GRP_30_X", "FOO"], [collect(1:6)]; verbose=false))
    @test pcfg.ctx.species_key == ["PITA", "_H", "_H", "_S", "_H", "_H"]
    @test Pan.plugin_type(pcfg) === H3.HBAD3P
    @test_throws ErrorException Pan.make_param_dists(pcfg.ctx, 6, 1, [collect(1:6)])
    @test_throws ErrorException Pan.spinup!(H3.HBAD3P, nothing, nothing, nothing)
    @test Pan.Data.TREE3P_VIG_Z == H3.VIG_Z && Pan.Data.TREE3P_VIG_W == H3.VIG_W
    @test sum(H3.VIG_W) ≈ 1 atol = 1e-12
    @test sum(H3.VIG_W .* H3.VIG_Z) ≈ 0 atol = 1e-12
    @test sum(H3.VIG_W .* H3.VIG_Z .^ 2) ≈ 1 atol = 1e-9
    # yaml roster == csv directory; exact round trip; unknown / missing keys are errors
    text = read(YAML_PATH, String)
    p, meta = Pan.params_from_yaml(text)
    @test p isa H3.HBAD3PParams && meta.plugin == "hbad3p" && p.KEYS == p.SPECIES_LIST == KEYS && p.FAREA == 1
    ctx = H3.HBAD3PContext(String.(p.SPECIES_LIST), [Int.(i) for i in p.ECO_SPECIES_IDS]; verbose=false)
    q = Pan.generate_params(ctx, String.(p.SPECIES_LIST), String.(p.ECO_LIST), [Int.(i) for i in p.ECO_SPECIES_IDS])
    @test isempty(Pan.params_diff(p, q))
    p2, _ = Pan.params_from_yaml(Pan.params_to_yaml(p; name="x", species_map=meta.species_map, eco_mapping=meta.eco_mapping, data=meta.data))
    @test isempty(Pan.params_diff(p, p2))
    @test_throws ErrorException Pan.params_from_yaml(replace(text, "kc:" => "kcx:"; count=1))
    @test_throws ErrorException Pan.params_from_yaml(replace(text, "lnF:" => "lnZ:"; count=1))
    @test_throws ErrorException Pan.params_from_yaml(replace(text, "vig_n: 5" => "vig_n: 3"; count=1))
    @test meta.species_map["PITA"] == "PITA" && meta.species_map["QUAL"] == "_H" && meta.species_map["ABAM"] == "_S"
    @test haskey(meta.data, "plot_normal_water") && isfile(joinpath(dirname(@__DIR__), meta.data["plot_normal_water"]))
  end

  @testset "allocation-free steady state, determinism, monotone N" begin
    for no_est in (true, false)
      pcfg, soa, ctx = setup(; no_est=no_est)
      SP = Pan.succession_plugin_of(soa)
      soa.parallel = false
      for y in 1:12; Pan.step!(SP, soa, y, ctx); end      # warm up; recruit classes open at years 1, 6, 11
      Pan.step!(SP, soa, 13, ctx)                         # a year in which no recruit class opens
      a = @allocated Pan.step!(SP, soa, 14, ctx)
      @test a == 0
      pa, sa, ca = setup(; no_est=no_est); pb, sb, cb = setup(; no_est=no_est); sb.parallel = false
      for y in 1:20; Pan.step!(SP, sa, y, ca); Pan.step!(SP, sb, y, cb); end
      @test statehash(sa) == statehash(sb)                # serial vs threaded site loop
      pc2, sc2, cc2 = setup(; no_est=no_est)
      for y in 1:20; Pan.step!(SP, sc2, y, cc2); end
      @test statehash(sc2) == statehash(sa)               # repeated run
      p0, s0, c0 = setup(; no_est=true)
      prev = copy(s0.csr.c_tph)
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
    @test all(isfinite, soa.scalar.BA) && all(soa.scalar.RD .>= 0)
  end

  @testset "free run: recruits enter at D 2.54, A 5, CR crhi; rmult = 0 sites do not recruit; seed rain" begin
    pcfg, soa, ctx = setup(; no_est=false, rmult=[i % 4 == 0 ? 0.0 : 1.0 for i in 1:24])
    SP = Pan.succession_plugin_of(soa)
    n0 = copy(soa.scalar.live); Pan.step!(SP, soa, 1, ctx)
    for i in 1:soa.n
      if i % 4 == 0
        @test soa.scalar.live[i] == n0[i]
      else
        @test soa.scalar.live[i] == n0[i] + 12 * 5         # every species opens a class of 5 vigour nodes in year 1
        o = Int(soa.refs.cohort[i]) - 1
        r = o+Int(n0[i])+1:o+Int(soa.scalar.live[i])
        @test all(==(2.54), soa.csr.c_dq[r]) && all(==(5.0), soa.csr.c_age[r]) && all(==(pcfg.ctx.glob.crhi), soa.csr.c_cr[r]) && all(==(Int32(1)), soa.csr.c_ropen[r])
      end
    end
  end

  @testset "cohort add / remove / swap keep every field consistent" begin
    pcfg, soa, ctx = setup(; no_est=true)
    site = Pan.getsite(soa, 1)
    n = Int(site.live); @test n >= 5
    snap(k) = (site.c_species[k], site.c_age[k], site.c_tph[k], site.c_dq[k], site.c_h[k], site.c_zq[k], site.c_cr[k], site.c_bio[k], site.c_O[k], site.c_SDI[k], site.c_V[k], site.c_ropen[k])
    last_ = snap(n)
    PC.swap_cohort!(site, 1, n)
    @test snap(1) == last_
    site.live = n - 1
    cnt = soa.scalar._new_cohort_counts; cnt .= soa.scalar.live; cnt[1] += 1
    PC.readjust_soa!(soa, (cohort=cnt,))
    site = Pan.getsite(soa, 1)
    Pan.add_cohort!(pcfg.ctx, site, 2, 30.0, 4000.0)
    k = Int(site.live)
    @test site.c_species[k] == 2 && site.c_age[k] == 30.0 && site.c_zq[k] == 0.0 && site.c_cr[k] == 0.4 && Float64(site.c_bio[k]) ≈ 4000.0
    @test site.c_tph[k] > 0 && site.c_dq[k] > 0 && site.c_h[k] > 1.3
    Pan.set_cohort_bio!(pcfg.ctx, site, k, 2000.0)
    @test Float64(site.c_bio[k]) ≈ 2000.0
    SP = Pan.succession_plugin_of(soa)
    params = Pan.generate_params(pcfg.ctx, KEYS, ["eco"], [collect(1:NSP)])
    cx = Pan.make_ctx(pcfg.ctx, Pan.generate_eco_params(pcfg.ctx, params))
    Pan.step!(SP, soa, 1, cx)
    @test all(isfinite, soa.csr.c_tph) && all(isfinite, soa.csr.c_dq) && all(isfinite, soa.csr.c_cr)
  end

  @testset "plant!" begin
    pcfg, soa, ctx = setup(; no_est=true)
    si = 5                                              # empty site (s % 5 == 0)
    @test soa.scalar.live[si] == 0
    @test Pan.plant_slots_reserved(pcfg.ctx) == 5
    Pan.plant_sites!(pcfg.ctx, soa, [(si, 7, 1000.0)])
    site = Pan.getsite(soa, si)
    @test Int(site.live) == 5
    @test sort(collect(site.c_zq)) == collect(H3.VIG_Z)
    @test sum(site.c_tph) ≈ 1000.0
    spp = H3.identity_eco_params(pcfg.ctx).sp[7]; G = pcfg.ctx.glob
    @test all(==(2.54), site.c_dq) && all(==(H3.hstar(spp, G, 2.54)), site.c_h) && all(==(1.0), site.c_age) && all(==(UInt32(7)), site.c_species) && all(==(G.crhi), site.c_cr)
    @test sort(collect(site.c_tph)) ≈ sort(1000 .* collect(H3.VIG_W))
    @test Pan.plant_slot_needed(pcfg.ctx, site, 7, 1) == 0 && Pan.plant_slot_needed(pcfg.ctx, site, 8, 1) == 5
    Pan.plant_sites!(pcfg.ctx, soa, [(si, 7, 500.0)])    # merges into the existing sub-cohorts
    site = Pan.getsite(soa, si)
    nl = Int(site.live)
    @test nl == 5 && sum(site.c_tph[1:nl]) ≈ 1500.0 && all(≈(2.54), site.c_dq[1:nl])
    SP = Pan.succession_plugin_of(soa)
    params = Pan.generate_params(pcfg.ctx, KEYS, ["eco"], [collect(1:NSP)])
    cx = Pan.make_ctx(pcfg.ctx, Pan.generate_eco_params(pcfg.ctx, params))
    Pan.step!(SP, soa, 1, cx)
    site = Pan.getsite(soa, si)
    @test all(site.c_dq .> 2.54) && all(site.c_age .== 2.0)
  end

  @testset "forest-area convention: reported B / BA / SDI and c_bio are per ha of the plot (x area); RD and the dynamics are not" begin
    _, s1, c1 = setup(; no_est=true, area=1.0)
    _, s2, c2 = setup(; no_est=true, area=0.5)
    SP = Pan.succession_plugin_of(s1)
    for y in 1:5; Pan.step!(SP, s1, y, c1); Pan.step!(SP, s2, y, c2); end
    @test s1.csr.c_tph[1:sum(s1.scalar.live)] == s2.csr.c_tph[1:sum(s2.scalar.live)]        # same state evolution (areas differ only in the report)
    nz = findall(>(0), s1.scalar.live)
    @test s2.scalar.BA[nz] ≈ 0.5 .* s1.scalar.BA[nz] rtol = 1e-6
    @test s2.scalar.SDI[nz] ≈ 0.5 .* s1.scalar.SDI[nz] rtol = 1e-6
    @test s2.scalar.B[nz] ≈ 0.5 .* s1.scalar.B[nz] rtol = 1e-5
    @test s2.scalar.RD[nz] == s1.scalar.RD[nz]
    # management helpers
    site = Pan.getsite(s2, nz[1]); k = 1
    t0 = site.c_tph[k]; b0 = Float64(site.c_bio[k])
    H3.scale_cohort!(Pan.PluginConfig(H3.HBAD3PContext(KEYS, [collect(1:NSP)]; verbose=false)).ctx, site, k, 0.25)
    @test site.c_tph[k] == 0.25 * t0
    @test Float64(site.c_bio[k]) ≈ 0.25 * b0 rtol = 1e-5
  end

  @testset "cleanup option" begin
    @test H3.HBAD3PConst().cleanup_tph == 0.0
    function run(thr)
      pcfg = Pan.PluginConfig(H3.HBAD3PContext(KEYS, [collect(1:NSP)]; verbose=false, fixed=H3.HBAD3PConst(cleanup_tph=thr)))
      sp = [3, 3, 3, 3, 3, 7, 7, 7, 7, 7]; N = [60.0, 240.0, 60.0, 60.0, 60.0, 5e-4, 5e-4, 5e-4, 5e-4, 5e-4]
      D = fill(8.0, 10); D[6:10] .= 4.0; Hh = fill(9.0, 10); A = fill(30.0, 10); zq = repeat(collect(H3.VIG_Z), 2); CR = fill(0.4, 10)
      soa = Pan.make_sites_hbad3p_arrays(pcfg, [1, 11], sp, A, N, D, Hh, zq, CR; rng=Random.Xoshiro(1), no_establishment=true)
      params = Pan.generate_params(pcfg.ctx, KEYS, ["eco"], [collect(1:NSP)])
      ctx = Pan.make_ctx(pcfg.ctx, Pan.generate_eco_params(pcfg.ctx, params))
      for y in 1:5; Pan.step!(Pan.succession_plugin_of(soa), soa, y, ctx); end
      soa
    end
    s0 = run(0.0); s1 = run(1e-3)
    @test Int(s0.scalar.live[1]) == 10 && Int(s1.scalar.live[1]) == 5
    @test all(s1.csr.c_tph[1:5] .>= 1e-3)
    @test maximum(abs.(s1.csr.c_tph[1:5] .- s0.csr.c_tph[1:5]) ./ s0.csr.c_tph[1:5]) < 1e-5
  end
end
end
