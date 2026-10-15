# src/parametrize/utils.jl: build_slots / u_to_params / params_to_u / build_groups / floors / pooling (BiomassSuccession, tiny synthetic setup).
module UnitParams
using Test, Random, Pan
const PU = Pan.PU
const BSP = Pan.BSP
const SP = Pan.Plugins.BiomassSuccessionPlugin.BiomassSuccession
import Distributions as Dists

const SPECIES = ["A", "B", "C"]
const ECOS = ["e1", "e2"]
const ESI = [[1, 2, 3], [1, 3]]                    # eco 1 has species 1,2,3 ; eco 2 has species 1,3 (global ids)
const NESS = sum(length, ESI)                      # 5 eco-species cells

# Save/restore every mutable global this file touches so the file is order-independent.
function with_globals(f)
  saved = (PU.BMAX_FLOOR[], PU.ANPP_FLOOR[], PU.PROB_ESTAB_FLOOR[], PU.PARAM_SPLIT_SETS[], PU.PARAM_TIER_MERGE[], PU.POOL_LABEL[], PU.FLOOR_POOLED_MAX[],
    PU.CMAES_BLOCKS[], PU.ECO_L3[], PU.ECO_L2[])
  PU.BMAX_FLOOR[] = nothing; PU.ANPP_FLOOR[] = nothing; PU.PROB_ESTAB_FLOOR[] = nothing
  PU.PARAM_SPLIT_SETS[] = typeof(PU.PARAM_SPLIT_SETS[])(); PU.PARAM_TIER_MERGE[] = typeof(PU.PARAM_TIER_MERGE[])(); PU.POOL_LABEL[] = typeof(PU.POOL_LABEL[])()
  PU.FLOOR_POOLED_MAX[] = false; PU.CMAES_BLOCKS[] = :eco; PU.ECO_L3[] = Int[]; PU.ECO_L2[] = Int[]
  try
    return f()
  finally
    PU.BMAX_FLOOR[], PU.ANPP_FLOOR[], PU.PROB_ESTAB_FLOOR[], PU.PARAM_SPLIT_SETS[], PU.PARAM_TIER_MERGE[], PU.POOL_LABEL[], PU.FLOOR_POOLED_MAX[],
    PU.CMAES_BLOCKS[], PU.ECO_L3[], PU.ECO_L2[] = saved
  end
end

setup(; kw...) = begin
  pd = Pan.make_param_dists(SP, length(SPECIES), length(ECOS), ESI; kw...)
  tpl = Pan.generate_params(SP, SPECIES, ECOS, ESI; rng=Random.Xoshiro(1), kw...)
  (pd, tpl, PU.build_slots(pd, tpl))
end
pidx(pd, name) = findfirst(p -> p.name === name, pd.params)
dims_of(slots, pi) = [d for (d, (p, _)) in enumerate(slots) if p == pi]

@testset "param dist layout (BS plugin)" begin
  with_globals() do
    pd, tpl, slots = setup()
    @test [p.name for p in pd.params] == [:D, :LONGEVITY, :MATURITY, :SHADE_TOL, :S, :B_MAX_SPP, :ANPP_MAX_SPP, :PROB_MORT_SPP, :PROB_ESTAB_SPP, :MIN_REL_BIOMASS]
    @test pd.weights_cumsum[end] ≈ 1.0 && issorted(pd.weights_cumsum)
    # B_MAX is decoded BEFORE ANPP (the ratio reparam reads the freshly decoded B_MAX)
    @test pidx(pd, :B_MAX_SPP) < pidx(pd, :ANPP_MAX_SPP)
    # slot count: 3 species-scoped x 4 params + 5 eco-species x 5 params + 2 eco x 1
    @test length(slots) == 4 * 3 + 5 * NESS + 2
    for (name, n) in ((:D, 3), (:LONGEVITY, 3), (:MATURITY, 3), (:SHADE_TOL, 3), (:S, 5), (:B_MAX_SPP, 5), (:ANPP_MAX_SPP, 5), (:PROB_MORT_SPP, 5), (:PROB_ESTAB_SPP, 5), (:MIN_REL_BIOMASS, 2))
      @test length(dims_of(slots, pidx(pd, name))) == n
    end
    @test slots[1] == (1, 1) && slots[13] == (5, (1, 1)) && slots[end] == (10, 2)
    # no_establishment: drops MATURITY/SHADE_TOL/PROB_MORT/PROB_ESTAB/MIN_REL
    pd2, _, s2 = setup(no_establishment=true)
    @test [p.name for p in pd2.params] == [:D, :LONGEVITY, :S, :B_MAX_SPP, :ANPP_MAX_SPP]
    @test length(s2) == 2 * 3 + 3 * NESS
    # template shape facts used below
    @test tpl.ECO_SPECIES_IDS == ESI && length(tpl.B_MAX_SPP[1]) == 3 && length(tpl.B_MAX_SPP[2]) == 2
  end
end

# B17 (fixed): levels whose cdf(k) sat on a quantile boundary used to decode to k+1 (SHADE_TOL 3; MATURITY 7, 14, 28).
# params_to_u now encodes integer levels at the middle of their mass interval, so every level round-trips exactly.
@testset "integer-level round trip is exact (B17)" begin
  with_globals() do
    pd, tpl, slots = setup()
    for (f, lvls) in ((:SHADE_TOL, 1:5), (:MATURITY, 1:50))
      for k in lvls
        t = deepcopy(tpl); getproperty(t, f)[1] = k
        k2 = Int(getproperty(PU.u_to_params(PU.params_to_u(t, pd, slots), pd, slots, tpl), f)[1])
        @test k2 == k
      end
    end
    # LONGEVITY (quantum 50) and B_MAX (quantum 100) are exact on their grids
    for k in 100:50:600
      t = deepcopy(tpl); t.LONGEVITY[1] = k
      @test PU.u_to_params(PU.params_to_u(t, pd, slots), pd, slots, tpl).LONGEVITY[1] == k
    end
  end
end

@testset "u_to_params / params_to_u round trip (no floors)" begin
  with_globals() do
    pd, tpl, slots = setup()
    @test PU.BMAX_FLOOR[] === nothing && PU.ANPP_FLOOR[] === nothing && PU.PROB_ESTAB_FLOOR[] === nothing
    rng = Random.Xoshiro(7)
    lev(p) = Float64(Dists.maximum(p.dist) - Dists.minimum(p.dist)) + 1        # discrete level count
    # the quantum / discreteness-induced u error: snapping moves the value by <= quantum/2 (or 0.5 for plain integer priors);
    # one decode->encode step then moves u by at most (shift + 1 level)/levels because cdf(integer k) >= u.
    tol(p) = p.dist isa Dists.DiscreteUniform ? ((p.quantum > 0 ? p.quantum / 2 : 0.0) + 1) / lev(p) + 1e-9 : 1e-5
    for trial in 1:20
      u = 0.02 .+ 0.96 .* rand(rng, length(slots))
      p = PU.u_to_params(u, pd, slots, tpl)
      @test p isa typeof(tpl)
      u2 = PU.params_to_u(p, pd, slots)
      for (d, (pi, _)) in enumerate(slots)
        @test abs(u[d] - u2[d]) <= tol(pd.params[pi])
      end
      # idempotence: re-decoding the recovered u reproduces the params exactly (quantization is the only loss)
      p2 = PU.u_to_params(u2, pd, slots, tpl)
      for f in (:D, :LONGEVITY, :S, :B_MAX_SPP, :PROB_MORT_SPP, :PROB_ESTAB_SPP, :MIN_REL_BIOMASS)
        @test getproperty(p2, f) == getproperty(p, f)
      end
      # integer levels round-trip exactly (B17)
      for f in (:SHADE_TOL, :MATURITY)
        @test getproperty(p2, f) == getproperty(p, f)
      end
      @test all(isapprox.(reduce(vcat, p2.ANPP_MAX_SPP), reduce(vcat, p.ANPP_MAX_SPP); rtol=1e-5))
      # decoded values respect bounds, quantum and types
      @test all(x -> 12000 <= x <= 35000 && mod(x, 100) == 0, reduce(vcat, p.B_MAX_SPP))
      @test all(x -> 100 <= x <= 600 && mod(x, 50) == 0, p.LONGEVITY)
      @test all(x -> 1 <= x <= 5, p.SHADE_TOL) && eltype(p.SHADE_TOL) == UInt32
      @test all(x -> 1 <= x <= 50, p.MATURITY)
      @test all(x -> 0.01f0 <= x <= 1f0, reduce(vcat, p.S))
      @test all(x -> 0.2f0 <= x <= 1f0, reduce(vcat, p.PROB_ESTAB_SPP))
      # template fields that are not tunable pass through untouched
      @test p.ECO_SPECIES_IDS == tpl.ECO_SPECIES_IDS && p.PROB_RESPROUT == tpl.PROB_RESPROUT && p.SUFFICIENT_LIGHT == tpl.SUFFICIENT_LIGHT
    end
    # template is not mutated by decoding (copy-on-write appliers)
    b0 = deepcopy(tpl.B_MAX_SPP); PU.u_to_params(fill(0.9, length(slots)), pd, slots, tpl); @test tpl.B_MAX_SPP == b0
    # u outside [0,1] is clamped (decode is total on R^d)
    pe = PU.u_to_params(fill(-3.0, length(slots)), pd, slots, tpl); pz = PU.u_to_params(fill(1e-10, length(slots)), pd, slots, tpl)
    @test pe.B_MAX_SPP == pz.B_MAX_SPP && pe.S == pz.S
    # u=0.5 gives the prior median for continuous Uniform params
    ph = PU.u_to_params(fill(0.5, length(slots)), pd, slots, tpl)
    @test all(≈(0.505f0; atol=1e-5), reduce(vcat, ph.S))
    @test all(==(23500f0), reduce(vcat, ph.B_MAX_SPP))
  end
end

@testset "ANPP ratio reparam" begin
  with_globals() do
    pd, tpl, slots = setup()
    ia = pidx(pd, :ANPP_MAX_SPP); ib = pidx(pd, :B_MAX_SPP)
    da = dims_of(slots, ia)
    @test pd.params[ia].bounds == (20f0, 35f0)                              # the ratio window, not an ANPP window
    for ur in (0.01, 0.3, 0.5, 0.99)
      u = fill(0.5, length(slots)); u[dims_of(slots, ib)] .= 0.37; u[da] .= ur
      p = PU.u_to_params(u, pd, slots, tpl)
      ratio = 20.0 + ur * 15.0
      for (e, ids) in enumerate(ESI), s in eachindex(ids)
        @test p.ANPP_MAX_SPP[e][s] ≈ p.B_MAX_SPP[e][s] / ratio rtol = 1e-5     # ANPP = B_MAX / ratio
        @test 20 - 1e-3 <= p.B_MAX_SPP[e][s] / p.ANPP_MAX_SPP[e][s] <= 35 + 1e-3
      end
      # inverse recovers the ratio u exactly (up to Float32 storage)
      @test all(isapprox.(PU.params_to_u(p, pd, slots)[da], ur; atol=1e-5))
    end
    # legacy seed with a real ANPP whose implied ratio is outside [20,35] is clamped into the window (u -> 1-1e-6 / 1e-6)
    leg = deepcopy(tpl); leg.B_MAX_SPP[1][1] = 20000f0; leg.ANPP_MAX_SPP[1][1] = 100f0      # ratio 200 -> above window
    @test PU.params_to_u(leg, pd, slots)[da[1]] ≈ 1 - 1e-6
    leg.ANPP_MAX_SPP[1][1] = 10000f0                                                       # ratio 2 -> below window
    @test PU.params_to_u(leg, pd, slots)[da[1]] ≈ 1e-6
    leg.ANPP_MAX_SPP[1][1] = 0f0                                                           # ANPP <= 0 -> falls back to the prior mean ratio
    @test PU.params_to_u(leg, pd, slots)[da[1]] ≈ 0.5
    # ANPP_FLOOR: ANPP = max(B/ratio, floor); inverse stays round-trip stable
    PU.ANPP_FLOOR[] = Dict{Tuple{Int,Int},Float32}((1, 1) => 5000f0)
    u = fill(0.5, length(slots)); p = PU.u_to_params(u, pd, slots, tpl)
    @test p.ANPP_MAX_SPP[1][1] == 5000f0 && p.ANPP_MAX_SPP[1][2] < 5000f0              # only cell (1,1) floored; cells absent from the dict get floor 0
    p2 = PU.u_to_params(PU.params_to_u(p, pd, slots), pd, slots, tpl)
    @test p2.ANPP_MAX_SPP[1][1] == 5000f0
  end
end

@testset "BMAX / PROB_ESTAB floors" begin
  with_globals() do
    pd, tpl, slots = setup()
    ib = pidx(pd, :B_MAX_SPP); db = dims_of(slots, ib)
    flo = Dict{Tuple{Int,Int},Float32}((e, s) => 20000f0 for (e, ids) in enumerate(ESI) for s in eachindex(ids))
    PU.BMAX_FLOOR[] = flo
    # floor BELOW the upper bound: uniform u rescaled onto [floor, upper] (then snapped to the 100 quantum)
    for (uv, expect) in ((0.0, 20000f0), (0.5, 27500f0), (1.0, 35000f0))
      u = fill(0.5, length(slots)); u[db] .= uv
      p = PU.u_to_params(u, pd, slots, tpl)
      @test all(==(expect), reduce(vcat, p.B_MAX_SPP))
    end
    u = fill(0.5, length(slots)); u[db] .= 0.25
    p = PU.u_to_params(u, pd, slots, tpl)
    @test all(x -> x in (23700f0, 23800f0), reduce(vcat, p.B_MAX_SPP))   # 23750 snapped to the 100 grid (tie)
    # inverse of the rescale
    pu = PU.params_to_u(p, pd, slots)
    @test all(isapprox.(pu[db], 0.25; atol = 50 / 15000 + 1e-9))
    # floor ABOVE the upper bound (40000 > 35000): the floor wins, the slot is pinned at the floor for EVERY u (B18 — u ≤ 0 used
    # to decode to the upper bound 35000, below the floor)
    PU.BMAX_FLOOR[] = Dict{Tuple{Int,Int},Float32}((e, s) => 40000f0 for (e, ids) in enumerate(ESI) for s in eachindex(ids))
    for uv in (-0.5, 0.0, 1e-9, 0.3, 0.99, 1.0, 1.7)
      u = fill(0.5, length(slots)); u[db] .= uv
      p = PU.u_to_params(u, pd, slots, tpl)
      @test all(==(40000f0), reduce(vcat, p.B_MAX_SPP))
    end
    # ...and the inverse is degenerate (negative window): clamped to 1e-6
    p = PU.u_to_params(fill(0.5, length(slots)), pd, slots, tpl)
    @test all(==(1e-6), PU.params_to_u(p, pd, slots)[db])
    # cells missing from the floor dict fall back to 12000
    PU.BMAX_FLOOR[] = Dict{Tuple{Int,Int},Float32}((1, 1) => 30000f0)
    u = fill(0.5, length(slots)); u[db] .= 0.0
    p = PU.u_to_params(u, pd, slots, tpl)
    @test p.B_MAX_SPP[1][1] == 30000f0 && p.B_MAX_SPP[1][2] == 12000f0 && p.B_MAX_SPP[2][2] == 12000f0
    PU.BMAX_FLOOR[] = nothing
    # PROB_ESTAB floor: rescale onto [floor, 1.0]
    ip = pidx(pd, :PROB_ESTAB_SPP); dp = dims_of(slots, ip)
    PU.PROB_ESTAB_FLOOR[] = Dict{Tuple{Int,Int},Float32}((e, s) => 0.6f0 for (e, ids) in enumerate(ESI) for s in eachindex(ids))
    for (uv, expect) in ((0.0, 0.6f0), (0.5, 0.8f0), (1.0, 1.0f0))
      u = fill(0.5, length(slots)); u[dp] .= uv
      p = PU.u_to_params(u, pd, slots, tpl)
      @test all(≈(expect; atol=1e-6), reduce(vcat, p.PROB_ESTAB_SPP))
    end
    u = fill(0.5, length(slots)); u[dp] .= 0.7
    @test all(isapprox.(PU.params_to_u(PU.u_to_params(u, pd, slots, tpl), pd, slots)[dp], 0.7; atol=1e-5))
    # cells absent from the PROB_ESTAB dict fall back to the param's own lower bound (0.2)
    PU.PROB_ESTAB_FLOOR[] = Dict{Tuple{Int,Int},Float32}()
    u = fill(0.5, length(slots)); u[dp] .= 0.0
    @test all(≈(0.2f0; atol=1e-6), reduce(vcat, PU.u_to_params(u, pd, slots, tpl).PROB_ESTAB_SPP))
  end
end

@testset "slots: split sets / pooling / shared" begin
  with_globals() do
    pd, tpl, _ = setup()
    ib = pidx(pd, :B_MAX_SPP)
    # PARAM_SPLIT_SETS: species NOT in the set are SHARED (one u-dim, Vector target of all its (eco,sp_local) positions)
    PU.PARAM_SPLIT_SETS[] = Dict(:B_MAX_SPP => Set([1]))
    slots = PU.build_slots(pd, tpl); db = dims_of(slots, ib)
    tg = [slots[d][2] for d in db]
    @test tg == Any[(1, 1), (2, 1), [(1, 2)], [(1, 3), (2, 2)]]              # gsp1 split per eco; gsp2 (1 position) and gsp3 (2 positions) shared
    p = PU.u_to_params(let u = fill(0.5, length(slots)); u[db] .= [0.1, 0.9, 0.5, 0.7]; u end, pd, slots, tpl)
    @test p.B_MAX_SPP[1][3] == p.B_MAX_SPP[2][2]                             # shared slot broadcast across ecos
    @test p.B_MAX_SPP[1][1] != p.B_MAX_SPP[2][1]
    # params_to_u of a shared slot reads the representative (first) target
    @test PU.params_to_u(p, pd, slots)[db[4]] > 0.5
    PU.PARAM_SPLIT_SETS[] = typeof(PU.PARAM_SPLIT_SETS[])()
    # POOL_LABEL (param_pooling): one u-dim per label (sorted); singleton label -> plain tuple target, else Vector
    PU.POOL_LABEL[] = Dict(:B_MAX_SPP => Dict((1, 1) => "z|A", (2, 1) => "z|A", (1, 2) => "y|B", (1, 3) => "x|C", (2, 2) => "x|C"))
    slots = PU.build_slots(pd, tpl); db = dims_of(slots, ib)
    @test [slots[d][2] for d in db] == Any[[(1, 3), (2, 2)], (1, 2), [(1, 1), (2, 1)]]    # sorted by label: x|C, y|B, z|A
    p = PU.u_to_params(let u = fill(0.5, length(slots)); u[db] .= [0.2, 0.5, 0.8]; u end, pd, slots, tpl)
    @test p.B_MAX_SPP[1][3] == p.B_MAX_SPP[2][2] && p.B_MAX_SPP[1][1] == p.B_MAX_SPP[2][1] && length(unique(reduce(vcat, p.B_MAX_SPP))) == 3
    # a pooled slot's BMAX floor = the FIRST member's (legacy) unless FLOOR_POOLED_MAX
    PU.BMAX_FLOOR[] = Dict{Tuple{Int,Int},Float32}((1, 3) => 15000f0, (2, 2) => 30000f0, (1, 1) => 12000f0, (2, 1) => 12000f0, (1, 2) => 12000f0)
    u = fill(0.5, length(slots)); u[db] .= 0.0
    @test PU.u_to_params(u, pd, slots, tpl).B_MAX_SPP[2][2] == 15000f0
    PU.FLOOR_POOLED_MAX[] = true
    @test PU.u_to_params(u, pd, slots, tpl).B_MAX_SPP[2][2] == 30000f0
    # ANPP of a pooled slot is re-derived per eco from THAT eco's B_MAX, with one shared ratio
    PU.BMAX_FLOOR[] = nothing; PU.FLOOR_POOLED_MAX[] = false
    ia = pidx(pd, :ANPP_MAX_SPP)
    PU.POOL_LABEL[] = Dict(:B_MAX_SPP => Dict((1, 1) => "a", (2, 1) => "a", (1, 2) => "b", (1, 3) => "c", (2, 2) => "c"),
                           :ANPP_MAX_SPP => Dict((1, 1) => "a", (2, 1) => "a", (1, 2) => "b", (1, 3) => "c", (2, 2) => "c"))
    slots = PU.build_slots(pd, tpl); da = dims_of(slots, ia)
    @test length(da) == 3
    u = fill(0.5, length(slots)); u[da] .= 0.5
    p = PU.u_to_params(u, pd, slots, tpl)
    @test p.ANPP_MAX_SPP[1][1] ≈ p.B_MAX_SPP[1][1] / 27.5f0 && p.ANPP_MAX_SPP[2][1] ≈ p.B_MAX_SPP[2][1] / 27.5f0
  end
end

@testset "build_groups" begin
  with_globals() do
    pd, tpl, slots = setup()
    g = PU.build_groups(pd, slots, BSP.BIOMASS_PER_ECO_GROUPS)
    @test sort(reduce(vcat, g)) == collect(1:length(slots))                  # a partition of 1:d
    @test length(g) == 5 + 2                                                  # longevity_decay, maturity, shade (global) + growth_biomass x 2 ecos + min_rel x 2 ecos
    @test g[1] == collect(1:6)                                                # D + LONGEVITY share the :longevity_decay block
    @test g[2] == [7, 8, 9] && g[3] == [10, 11, 12]                           # first-seen order
    # per-eco growth_biomass blocks contain exactly one eco's S/B_MAX/ANPP/MORT/ESTAB dims
    for blk in g[4:5]
      ecos = unique(slots[d][2][1] for d in blk)
      @test length(ecos) == 1
    end
    @test g[4] == [13, 14, 15, 18, 19, 20, 23, 24, 25, 28, 29, 30, 33, 34, 35] && g[5] == [16, 17, 21, 22, 26, 27, 31, 32, 36, 37]
    @test g[6] == [38] && g[7] == [39]                                         # :min_rel split per eco (EcoSampler target is an Int)
    # not per-eco ⇒ the group stays whole
    g1 = PU.build_groups(pd, slots, Set{Symbol}())
    @test length(g1) == 5 && sort(reduce(vcat, g1)) == collect(1:length(slots))
    # :l3 mode: per-eco blocks keyed by ECO_L3; two ecos in the same L3 merge into one block
    PU.CMAES_BLOCKS[] = :l3; PU.ECO_L3[] = [7, 7]
    gl = PU.build_groups(pd, slots, BSP.BIOMASS_PER_ECO_GROUPS)
    @test length(gl) == 5 && sort(reduce(vcat, gl)) == collect(1:length(slots))
    PU.ECO_L3[] = [7, 8]
    @test PU.build_groups(pd, slots, BSP.BIOMASS_PER_ECO_GROUPS) == g
    # a pooled slot spanning L3s of one L2 gets the L2 block, spanning several L2s the tail block (-1)
    PU.POOL_LABEL[] = Dict(:B_MAX_SPP => Dict((1, 1) => "s", (2, 1) => "s", (1, 2) => "t", (1, 3) => "u", (2, 2) => "v"))
    sl = PU.build_slots(pd, tpl)
    PU.ECO_L2[] = [1, 1]
    @test PU._block_eco([(1, 1), (2, 1)]) == 10_001
    PU.ECO_L2[] = [1, 2]
    @test PU._block_eco([(1, 1), (2, 1)]) == -1
    @test PU._block_eco((2, 5)) == 8 && PU._block_eco(2) == 8 && PU._block_eco(nothing) == 0
  end
end

@testset "misc slot helpers: nativevec, quanta, integer_u_min_std, sobol" begin
  with_globals() do
    pd, tpl, slots = setup()
    nv = PU.params_to_nativevec(tpl, pd, slots)
    @test length(nv) == length(slots) && nv[1] == tpl.D[1] && nv[13] == tpl.S[1][1] && nv[18] == tpl.B_MAX_SPP[1][1]
    q = PU.slot_quanta(pd, slots)
    @test q[4] == 50.0 && q[1] == Float64(pd.params[1].sigma) && q[18] == 100.0       # quantum if set, else sigma
    s = PU.integer_u_min_std(pd, slots)
    @test s[1] == 0.0 && s[13] == 0.0                                                   # continuous priors: no floor
    @test s[4] ≈ 0.3 / 11                                                               # LONGEVITY: (600-100)/50+1 levels
    @test s[7] ≈ 0.3 / 50 && s[10] ≈ 0.3 / 5                                            # MATURITY (no quantum, step 1), SHADE_TOL
    @test s[18] ≈ 0.3 / 231                                                             # B_MAX: 23000/100+1 levels
    # sobol_samples: N decoded params, all within bounds
    ss = PU.sobol_samples(pd, tpl, 8)
    @test length(ss) == 8 && all(p -> all(x -> 12000 <= x <= 35000, reduce(vcat, p.B_MAX_SPP)), ss)
    @test ss[1].B_MAX_SPP == [fill(23500f0, 3), fill(23500f0, 2)]    # Sobol.jl's first emitted point is the centre u=0.5 (the origin is skipped)
    # saltelli design: N*(K+2) samples with matching tags; K = number of distinct param names
    smp, tags = PU.saltelli_design(pd, tpl, 4)
    K = length(pd.params)
    @test length(smp) == 4 * (K + 2) && length(tags) == length(smp)
    @test tags[1:4] == fill("A", 4) && tags[5:8] == fill("B", 4) && tags[9] == "AB:D" && tags[end] == "AB:MIN_REL_BIOMASS"
  end
end

@testset "mutate_params: weights, pooling, ANPP ratio, pinned slots (B9)" begin
  with_globals() do
    pd, tpl, slots = setup()
    p0 = PU.u_to_params(fill(0.5, length(slots)), pd, slots, tpl)
    # 1. parameter choice follows the sampling weights — including the LAST parameter (never chosen before)
    w = diff(vcat(0.0, pd.weights_cumsum)); n = 20_000; cnt = zeros(Int, length(w)); rng = Random.Xoshiro(3)
    for _ in 1:n; _, k = PU.mutate_params(p0, pd; rng=rng, slots=slots); cnt[k] += 1; end
    @test all(abs.(cnt ./ n .- w) .< 0.02)
    @test cnt[end] > 0
    # 2. ANPP stays ANPP: B_MAX / ratio with the ratio inside its box (the old code wrote ratio values 20–35 into ANPP)
    ai = pidx(pd, :ANPP_MAX_SPP); bi = pidx(pd, :B_MAX_SPP)
    one_param(k) = (c = zeros(length(pd.weights_cumsum)); c[k:end] .= 1.0; c)   # dynamic_cumsum that always picks param k
    rng = Random.Xoshiro(5); p = p0
    for _ in 1:200
      p, k = PU.mutate_params(p, pd; rng=rng, slots=slots, dynamic_cumsum=one_param(ai), mutation_mode=PU.BothMutations)
      @test k == ai
    end
    for (e, ids) in enumerate(ESI), j in eachindex(ids)
      r = p.B_MAX_SPP[e][j] / p.ANPP_MAX_SPP[e][j]
      @test 20 - 1e-3 <= r <= 35 + 1e-3
    end
    # 3. mutating B_MAX keeps each target's ratio (ANPP re-derived), and changes nothing else
    q = p
    for _ in 1:50; q, _ = PU.mutate_params(q, pd; rng=rng, slots=slots, dynamic_cumsum=one_param(bi)); end
    @test q.B_MAX_SPP != p.B_MAX_SPP
    for (e, ids) in enumerate(ESI), j in eachindex(ids)
      @test isapprox(q.B_MAX_SPP[e][j] / q.ANPP_MAX_SPP[e][j], p.B_MAX_SPP[e][j] / p.ANPP_MAX_SPP[e][j]; rtol=1e-4)
    end
    @test q.S == p.S && q.D == p.D
    # 4. pooled parameter: one move changes every cell of the pooled slot identically
    PU.POOL_LABEL[] = Dict(:B_MAX_SPP => Dict((1, 1) => "a", (2, 1) => "a", (1, 2) => "b", (1, 3) => "c", (2, 2) => "c"))
    sp = PU.build_slots(pd, tpl)
    r = p0
    for _ in 1:30
      r, _ = PU.mutate_params(r, pd; rng=rng, slots=sp, dynamic_cumsum=one_param(bi))
      @test r.B_MAX_SPP[1][1] == r.B_MAX_SPP[2][1] && r.B_MAX_SPP[1][3] == r.B_MAX_SPP[2][2]
    end
    PU.POOL_LABEL[] = typeof(PU.POOL_LABEL[])()
    # 5. a slot pinned by a floor above the box cannot change: returns unchanged instead of looping forever
    PU.BMAX_FLOOR[] = Dict{Tuple{Int,Int},Float32}((e, j) => 40000f0 for (e, ids) in enumerate(ESI) for j in eachindex(ids))
    pin = PU.u_to_params(fill(0.5, length(slots)), pd, slots, tpl)
    t = @elapsed (pp, _) = PU.mutate_params(pin, pd; rng=rng, slots=slots, dynamic_cumsum=one_param(bi))
    @test pp.B_MAX_SPP == pin.B_MAX_SPP && t < 5
    PU.BMAX_FLOOR[] = nothing
    # 6. loss-weighted slot choice (SamplingContext): all moves land on the only species with loss
    ctx = PU.SamplingContext(Float32[0, 1, 0])          # only species 2 ("B", eco 1 local 2) carries loss
    m = p0
    for _ in 1:30
      m2, _ = PU.mutate_params(m, pd; rng=rng, slots=slots, dynamic_cumsum=one_param(bi), ctx=ctx)
      changed = [(e, j) for (e, ids) in enumerate(ESI) for j in eachindex(ids) if m2.B_MAX_SPP[e][j] != m.B_MAX_SPP[e][j]]
      @test isempty(changed) || changed == [(1, 2)]
      m = m2
    end
  end
end

end
