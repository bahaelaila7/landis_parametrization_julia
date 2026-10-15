# src/parametrize/utils.jl: build_slots / u_to_params / params_to_u / build_groups / floors / pooling (BiomassSuccession, tiny synthetic setup).
module UnitParams
using Test, Random, Pan
const PU = Pan.PU
const BSP = Pan.BSP
const BSPl = Pan.Plugins.BiomassSuccessionPlugin
const PC0 = BSPl.BSTables()                          # the plugin context of the plain legacy space (no pins, no floors)
import Distributions as Dists

const SPECIES = ["A", "B", "C"]
const ECOS = ["e1", "e2"]
const ESI = [[1, 2, 3], [1, 3]]                    # eco 1 has species 1,2,3 ; eco 2 has species 1,3 (global ids)
const NESS = sum(length, ESI)                      # 5 eco-species cells

# The parameter space is an explicit immutable ParamSpace (Step 3c): variants are built with `mkspace` / `retie` / `PU.with_floors`.
mkspace(; pc=PC0, no_establishment=false, opts=PU.ParamSpaceOptions(), tying=PU.ParamTying(), floors=PU.ParamFloors(), blocks=PU.CmaBlocks()) =
  Pan.make_param_space(pc, SPECIES, ECOS, ESI; no_establishment=no_establishment, opts=opts, tying=tying, floors=floors, blocks=blocks)
retie(sp; tying=sp.tying, blocks=sp.blocks) = PU.ParamSpace(sp.dists, sp.n_species, sp.eco_list, sp.eco_species_ids; opts=sp.opts, tying=tying, floors=sp.floors, blocks=blocks)
setup(; no_establishment=false) = begin
  sp = mkspace(; no_establishment=no_establishment)
  tpl = Pan.generate_params(PC0, SPECIES, ECOS, ESI; rng=Random.Xoshiro(1), no_establishment=no_establishment)
  (sp, tpl, sp.slots)
end
pidx(pd, name) = findfirst(p -> p.name === name, pd.params)
dims_of(slots, pi) = [d for (d, (p, _)) in enumerate(slots) if p == pi]

@testset "param dist layout (BS plugin)" begin
  let
    sp, tpl, slots = setup(); pd = sp.dists
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
    sp2, _, s2 = setup(no_establishment=true); pd2 = sp2.dists
    @test [p.name for p in pd2.params] == [:D, :LONGEVITY, :S, :B_MAX_SPP, :ANPP_MAX_SPP]
    @test length(s2) == 2 * 3 + 3 * NESS
    # template shape facts used below
    @test tpl.ECO_SPECIES_IDS == ESI && length(tpl.B_MAX_SPP[1]) == 3 && length(tpl.B_MAX_SPP[2]) == 2
  end
end

# B17 (fixed): levels whose cdf(k) sat on a quantile boundary used to decode to k+1 (SHADE_TOL 3; MATURITY 7, 14, 28).
# params_to_u now encodes integer levels at the middle of their mass interval, so every level round-trips exactly.
@testset "integer-level round trip is exact (B17)" begin
  let
    sp, tpl, slots = setup(); pd = sp.dists
    for (f, lvls) in ((:SHADE_TOL, 1:5), (:MATURITY, 1:50))
      for k in lvls
        t = deepcopy(tpl); getproperty(t, f)[1] = k
        k2 = Int(getproperty(PU.u_to_params(PU.params_to_u(t, sp, slots), sp, slots, tpl), f)[1])
        @test k2 == k
      end
    end
    # LONGEVITY (quantum 50) and B_MAX (quantum 100) are exact on their grids
    for k in 100:50:600
      t = deepcopy(tpl); t.LONGEVITY[1] = k
      @test PU.u_to_params(PU.params_to_u(t, sp, slots), sp, slots, tpl).LONGEVITY[1] == k
    end
  end
end

@testset "u_to_params / params_to_u round trip (no floors)" begin
  let
    sp, tpl, slots = setup(); pd = sp.dists
    @test sp.floors.bmax === nothing && sp.floors.anpp === nothing && sp.floors.prob_estab === nothing
    rng = Random.Xoshiro(7)
    lev(p) = Float64(Dists.maximum(p.dist) - Dists.minimum(p.dist)) + 1        # discrete level count
    # the quantum / discreteness-induced u error: snapping moves the value by <= quantum/2 (or 0.5 for plain integer priors);
    # one decode->encode step then moves u by at most (shift + 1 level)/levels because cdf(integer k) >= u.
    tol(p) = p.dist isa Dists.DiscreteUniform ? ((p.quantum > 0 ? p.quantum / 2 : 0.0) + 1) / lev(p) + 1e-9 : 1e-5
    for trial in 1:20
      u = 0.02 .+ 0.96 .* rand(rng, length(slots))
      p = PU.u_to_params(u, sp, slots, tpl)
      @test p isa typeof(tpl)
      u2 = PU.params_to_u(p, sp, slots)
      for (d, (pi, _)) in enumerate(slots)
        @test abs(u[d] - u2[d]) <= tol(pd.params[pi])
      end
      # idempotence: re-decoding the recovered u reproduces the params exactly (quantization is the only loss)
      p2 = PU.u_to_params(u2, sp, slots, tpl)
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
    b0 = deepcopy(tpl.B_MAX_SPP); PU.u_to_params(fill(0.9, length(slots)), sp, slots, tpl); @test tpl.B_MAX_SPP == b0
    # u outside [0,1] is clamped (decode is total on R^d)
    pe = PU.u_to_params(fill(-3.0, length(slots)), sp, slots, tpl); pz = PU.u_to_params(fill(1e-10, length(slots)), sp, slots, tpl)
    @test pe.B_MAX_SPP == pz.B_MAX_SPP && pe.S == pz.S
    # u=0.5 gives the prior median for continuous Uniform params
    ph = PU.u_to_params(fill(0.5, length(slots)), sp, slots, tpl)
    @test all(≈(0.505f0; atol=1e-5), reduce(vcat, ph.S))
    @test all(==(23500f0), reduce(vcat, ph.B_MAX_SPP))
  end
end

@testset "ANPP ratio reparam" begin
  let
    sp, tpl, slots = setup(); pd = sp.dists
    ia = pidx(pd, :ANPP_MAX_SPP); ib = pidx(pd, :B_MAX_SPP)
    da = dims_of(slots, ia)
    @test pd.params[ia].bounds == (20f0, 35f0)                              # the ratio window, not an ANPP window
    for ur in (0.01, 0.3, 0.5, 0.99)
      u = fill(0.5, length(slots)); u[dims_of(slots, ib)] .= 0.37; u[da] .= ur
      p = PU.u_to_params(u, sp, slots, tpl)
      ratio = 20.0 + ur * 15.0
      for (e, ids) in enumerate(ESI), s in eachindex(ids)
        @test p.ANPP_MAX_SPP[e][s] ≈ p.B_MAX_SPP[e][s] / ratio rtol = 1e-5     # ANPP = B_MAX / ratio
        @test 20 - 1e-3 <= p.B_MAX_SPP[e][s] / p.ANPP_MAX_SPP[e][s] <= 35 + 1e-3
      end
      # inverse recovers the ratio u exactly (up to Float32 storage)
      @test all(isapprox.(PU.params_to_u(p, sp, slots)[da], ur; atol=1e-5))
    end
    # legacy seed with a real ANPP whose implied ratio is outside [20,35] is clamped into the window (u -> 1-1e-6 / 1e-6)
    leg = deepcopy(tpl); leg.B_MAX_SPP[1][1] = 20000f0; leg.ANPP_MAX_SPP[1][1] = 100f0      # ratio 200 -> above window
    @test PU.params_to_u(leg, sp, slots)[da[1]] ≈ 1 - 1e-6
    leg.ANPP_MAX_SPP[1][1] = 10000f0                                                       # ratio 2 -> below window
    @test PU.params_to_u(leg, sp, slots)[da[1]] ≈ 1e-6
    leg.ANPP_MAX_SPP[1][1] = 0f0                                                           # ANPP <= 0 -> falls back to the prior mean ratio
    @test PU.params_to_u(leg, sp, slots)[da[1]] ≈ 0.5
    # ParamFloors.anpp: ANPP = max(B/ratio, floor); inverse stays round-trip stable
    sp = PU.with_floors(sp, PU.ParamFloors(anpp=Dict{Tuple{Int,Int},Float32}((1, 1) => 5000f0)))
    u = fill(0.5, length(slots)); p = PU.u_to_params(u, sp, slots, tpl)
    @test p.ANPP_MAX_SPP[1][1] == 5000f0 && p.ANPP_MAX_SPP[1][2] < 5000f0              # only cell (1,1) floored; cells absent from the dict get floor 0
    p2 = PU.u_to_params(PU.params_to_u(p, sp, slots), sp, slots, tpl)
    @test p2.ANPP_MAX_SPP[1][1] == 5000f0
  end
end

@testset "BMAX / PROB_ESTAB floors" begin
  let
    sp, tpl, slots = setup(); pd = sp.dists
    ib = pidx(pd, :B_MAX_SPP); db = dims_of(slots, ib)
    flo = Dict{Tuple{Int,Int},Float32}((e, s) => 20000f0 for (e, ids) in enumerate(ESI) for s in eachindex(ids))
    sp = PU.with_floors(sp, PU.ParamFloors(bmax=flo))
    # floor BELOW the upper bound: uniform u rescaled onto [floor, upper] (then snapped to the 100 quantum)
    for (uv, expect) in ((0.0, 20000f0), (0.5, 27500f0), (1.0, 35000f0))
      u = fill(0.5, length(slots)); u[db] .= uv
      p = PU.u_to_params(u, sp, slots, tpl)
      @test all(==(expect), reduce(vcat, p.B_MAX_SPP))
    end
    u = fill(0.5, length(slots)); u[db] .= 0.25
    p = PU.u_to_params(u, sp, slots, tpl)
    @test all(x -> x in (23700f0, 23800f0), reduce(vcat, p.B_MAX_SPP))   # 23750 snapped to the 100 grid (tie)
    # inverse of the rescale
    pu = PU.params_to_u(p, sp, slots)
    @test all(isapprox.(pu[db], 0.25; atol = 50 / 15000 + 1e-9))
    # floor ABOVE the upper bound (40000 > 35000): the floor wins, the slot is pinned at the floor for EVERY u (B18 — u ≤ 0 used
    # to decode to the upper bound 35000, below the floor)
    sp = PU.with_floors(sp, PU.ParamFloors(bmax=Dict{Tuple{Int,Int},Float32}((e, s) => 40000f0 for (e, ids) in enumerate(ESI) for s in eachindex(ids))))
    for uv in (-0.5, 0.0, 1e-9, 0.3, 0.99, 1.0, 1.7)
      u = fill(0.5, length(slots)); u[db] .= uv
      p = PU.u_to_params(u, sp, slots, tpl)
      @test all(==(40000f0), reduce(vcat, p.B_MAX_SPP))
    end
    # ...and the inverse is degenerate (negative window): clamped to 1e-6
    p = PU.u_to_params(fill(0.5, length(slots)), sp, slots, tpl)
    @test all(==(1e-6), PU.params_to_u(p, sp, slots)[db])
    # cells missing from the floor dict fall back to 12000
    sp = PU.with_floors(sp, PU.ParamFloors(bmax=Dict{Tuple{Int,Int},Float32}((1, 1) => 30000f0)))
    u = fill(0.5, length(slots)); u[db] .= 0.0
    p = PU.u_to_params(u, sp, slots, tpl)
    @test p.B_MAX_SPP[1][1] == 30000f0 && p.B_MAX_SPP[1][2] == 12000f0 && p.B_MAX_SPP[2][2] == 12000f0
    # PROB_ESTAB floor: rescale onto [floor, 1.0]
    ip = pidx(pd, :PROB_ESTAB_SPP); dp = dims_of(slots, ip)
    sp = PU.with_floors(sp, PU.ParamFloors(prob_estab=Dict{Tuple{Int,Int},Float32}((e, s) => 0.6f0 for (e, ids) in enumerate(ESI) for s in eachindex(ids))))
    for (uv, expect) in ((0.0, 0.6f0), (0.5, 0.8f0), (1.0, 1.0f0))
      u = fill(0.5, length(slots)); u[dp] .= uv
      p = PU.u_to_params(u, sp, slots, tpl)
      @test all(≈(expect; atol=1e-6), reduce(vcat, p.PROB_ESTAB_SPP))
    end
    u = fill(0.5, length(slots)); u[dp] .= 0.7
    @test all(isapprox.(PU.params_to_u(PU.u_to_params(u, sp, slots, tpl), sp, slots)[dp], 0.7; atol=1e-5))
    # cells absent from the PROB_ESTAB dict fall back to the param's own lower bound (0.2)
    sp = PU.with_floors(sp, PU.ParamFloors(prob_estab=Dict{Tuple{Int,Int},Float32}()))
    u = fill(0.5, length(slots)); u[dp] .= 0.0
    @test all(≈(0.2f0; atol=1e-6), reduce(vcat, PU.u_to_params(u, sp, slots, tpl).PROB_ESTAB_SPP))
  end
end

@testset "build_param_floors: data floors from the plugin tables, B18 warning once per build" begin
  ecos = ["8.3.5|lu=natural", "8.3.7|lu=artificial"]
  tb = BSPl.BSTables(bmax_floor=Dict(("A", "8.3.5", "natural") => 40000.0, ("B", "8.3.5", "natural") => 20000.0),
                     anpp_floor=Dict(("A", "8.3.5", "natural") => 900.0),
                     prob_estab=Dict(("A", "8.3.5", "natural") => 0.5, ("C", "8.3.7", "artificial") => 0.9))
  o = PU.ParamSpaceOptions(prob_estab_lower_bound=true, prob_estab_lower_mult=0.8)
  fl = nothing
  @test_logs (:info, r"bmax_floor_from_data: B_MAX lower bound") (:warn, r"pinned at the floor") (:info, r"prob_estab_lower_bound") (:info, r"anpp_floor_from_data") match_mode=:any (fl = Pan.build_param_floors(tb, o, ecos, SPECIES, ESI))
  @test fl.bmax[(1, 1)] == 40000f0 && fl.bmax[(1, 2)] == 20000f0 && fl.bmax[(2, 1)] == 12000f0          # max(12000, table); (eco, sp_local)
  @test fl.anpp[(1, 1)] == 900f0 && fl.anpp[(1, 2)] == 0f0
  @test fl.prob_estab[(1, 1)] ≈ 0.4f0 && fl.prob_estab[(2, 2)] ≈ 0.72f0 && !haskey(fl.prob_estab, (1, 2))   # mult x data; absent cells keep the param's own bound
  # floors off: no table or no lower-bound switch
  f0 = Pan.build_param_floors(BSPl.BSTables(), o, ecos, SPECIES, ESI)
  @test f0.bmax === nothing && f0.anpp === nothing && f0.prob_estab === nothing
  # the tables are an argument, not module state: the first build is unchanged by the second
  @test fl.bmax[(1, 1)] == 40000f0
  # the built floors decode: B_MAX of cell (1,1) is pinned above the box at the floor (B18)
  sp = PU.with_floors(mkspace(), fl); tpl = Pan.generate_params(PC0, SPECIES, ECOS, ESI; rng=Random.Xoshiro(1))
  @test PU.u_to_params(fill(0.5, length(sp.slots)), sp, sp.slots, tpl).B_MAX_SPP[1][1] == 40000f0
end

@testset "BSTables: pins and seeds are arguments of the generators / the space" begin
  # LONGEVITY pinned by a table (absent species -> default) or a fixed value: out of the search, seeded into generated params
  tb = BSPl.BSTables(longevity=Dict("A" => 120.0), longevity_default=300.0, shade_tol=Dict("B" => 5), shade_tol_default=2)
  p = Pan.generate_params(tb, SPECIES, ECOS, ESI; rng=Random.Xoshiro(1))
  @test p.LONGEVITY == Float32[120, 300, 300] && p.SHADE_TOL == UInt32[2, 5, 2]
  names = [q.name for q in mkspace(; pc=tb).dists.params]
  @test !(:LONGEVITY in names) && !(:SHADE_TOL in names) && :MATURITY in names
  pf = Pan.generate_params(BSPl.BSTables(fixed_longevity=250.0), SPECIES, ECOS, ESI; rng=Random.Xoshiro(1))
  @test all(==(250f0), pf.LONGEVITY) && !(:LONGEVITY in [q.name for q in mkspace(; pc=BSPl.BSTables(fixed_longevity=250.0)).dists.params])
  @test all(==(400f0), Pan.generate_params(PC0, SPECIES, ECOS, ESI; rng=Random.Xoshiro(1)).LONGEVITY) && :LONGEVITY in [q.name for q in mkspace().dists.params]
  # the defaults are not shared between two table objects
  @test BSPl.BSTables().longevity === nothing && BSPl.BSTables(longevity=Dict("A" => 1.0)).longevity == Dict("A" => 1.0) && BSPl.BSTables().longevity === nothing
end

@testset "slots: split sets / pooling / shared" begin
  let
    sp, tpl, _ = setup(); pd = sp.dists
    ib = pidx(pd, :B_MAX_SPP)
    # ParamTying.split_sets: species NOT in the set are SHARED (one u-dim, Vector target of all its (eco,sp_local) positions)
    sp = retie(sp; tying=PU.ParamTying(split_sets=Dict(:B_MAX_SPP => Set([1]))))
    slots = sp.slots; @test slots == PU.build_slots(sp, tpl); db = dims_of(slots, ib)
    tg = [slots[d][2] for d in db]
    @test tg == Any[(1, 1), (2, 1), [(1, 2)], [(1, 3), (2, 2)]]              # gsp1 split per eco; gsp2 (1 position) and gsp3 (2 positions) shared
    p = PU.u_to_params(let u = fill(0.5, length(slots)); u[db] .= [0.1, 0.9, 0.5, 0.7]; u end, sp, slots, tpl)
    @test p.B_MAX_SPP[1][3] == p.B_MAX_SPP[2][2]                             # shared slot broadcast across ecos
    @test p.B_MAX_SPP[1][1] != p.B_MAX_SPP[2][1]
    # params_to_u of a shared slot reads the representative (first) target
    @test PU.params_to_u(p, sp, slots)[db[4]] > 0.5
    # ParamTying.labels (param_pooling): one u-dim per label (sorted); singleton label -> plain tuple target, else Vector
    sp = retie(sp; tying=PU.ParamTying(labels=Dict(:B_MAX_SPP => Dict((1, 1) => "z|A", (2, 1) => "z|A", (1, 2) => "y|B", (1, 3) => "x|C", (2, 2) => "x|C"))))
    slots = sp.slots; db = dims_of(slots, ib)
    @test [slots[d][2] for d in db] == Any[[(1, 3), (2, 2)], (1, 2), [(1, 1), (2, 1)]]    # sorted by label: x|C, y|B, z|A
    p = PU.u_to_params(let u = fill(0.5, length(slots)); u[db] .= [0.2, 0.5, 0.8]; u end, sp, slots, tpl)
    @test p.B_MAX_SPP[1][3] == p.B_MAX_SPP[2][2] && p.B_MAX_SPP[1][1] == p.B_MAX_SPP[2][1] && length(unique(reduce(vcat, p.B_MAX_SPP))) == 3
    # a pooled slot's BMAX floor = the FIRST member's (legacy) unless ParamTying.floor_pooled_max
    sp = PU.with_floors(sp, PU.ParamFloors(bmax=Dict{Tuple{Int,Int},Float32}((1, 3) => 15000f0, (2, 2) => 30000f0, (1, 1) => 12000f0, (2, 1) => 12000f0, (1, 2) => 12000f0)))
    u = fill(0.5, length(slots)); u[db] .= 0.0
    @test PU.u_to_params(u, sp, slots, tpl).B_MAX_SPP[2][2] == 15000f0
    spmax = retie(sp; tying=PU.ParamTying(labels=sp.tying.labels, floor_pooled_max=true))
    @test PU.u_to_params(u, spmax, slots, tpl).B_MAX_SPP[2][2] == 30000f0
    # ANPP of a pooled slot is re-derived per eco from THAT eco's B_MAX, with one shared ratio
    sp = PU.with_floors(sp, PU.ParamFloors())
    ia = pidx(pd, :ANPP_MAX_SPP)
    sp = retie(sp; tying=PU.ParamTying(labels=Dict(:B_MAX_SPP => Dict((1, 1) => "a", (2, 1) => "a", (1, 2) => "b", (1, 3) => "c", (2, 2) => "c"),
                                                   :ANPP_MAX_SPP => Dict((1, 1) => "a", (2, 1) => "a", (1, 2) => "b", (1, 3) => "c", (2, 2) => "c"))))
    slots = sp.slots; da = dims_of(slots, ia)
    @test length(da) == 3
    u = fill(0.5, length(slots)); u[da] .= 0.5
    p = PU.u_to_params(u, sp, slots, tpl)
    @test p.ANPP_MAX_SPP[1][1] ≈ p.B_MAX_SPP[1][1] / 27.5f0 && p.ANPP_MAX_SPP[2][1] ≈ p.B_MAX_SPP[2][1] / 27.5f0
  end
end

@testset "build_groups" begin
  let
    sp, tpl, slots = setup(); pd = sp.dists
    g = PU.build_groups(sp, slots)
    @test sp.blocks.per_eco_groups == BSP.BIOMASS_PER_ECO_GROUPS              # the plugin's per-eco groups are filled in by make_param_space
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
    g1 = PU.build_groups(PU.ParamSpace(pd, 3, ECOS, ESI), slots)
    @test length(g1) == 5 && sort(reduce(vcat, g1)) == collect(1:length(slots))
    # :l3 mode: per-eco blocks keyed by eco_l3; two ecos in the same L3 merge into one block
    perg = BSP.BIOMASS_PER_ECO_GROUPS
    bl3(l3, l2=Int[]) = PU.CmaBlocks(mode=:l3, eco_l3=l3, eco_l2=l2, per_eco_groups=perg)
    gl = PU.build_groups(PU.ParamSpace(pd, 3, ECOS, ESI; blocks=bl3([7, 7])), slots)
    @test length(gl) == 5 && sort(reduce(vcat, gl)) == collect(1:length(slots))
    @test PU.build_groups(PU.ParamSpace(pd, 3, ECOS, ESI; blocks=bl3([7, 8])), slots) == g
    # a pooled slot spanning L3s of one L2 gets the L2 block, spanning several L2s the tail block (-1)
    @test PU._block_eco(bl3([7, 8], [1, 1]), [(1, 1), (2, 1)]) == 10_001
    @test PU._block_eco(bl3([7, 8], [1, 2]), [(1, 1), (2, 1)]) == -1
    b8 = bl3([7, 8], [1, 2])
    @test PU._block_eco(b8, (2, 5)) == 8 && PU._block_eco(b8, 2) == 8 && PU._block_eco(b8, nothing) == 0
    # the Pan-side builder: L3 / L2 indices of the eco cells ("8.3.5|lu=X" -> L3 "8.3.5" -> L2 "8.3")
    cb = Pan.cma_blocks("l3", ["8.3.5|lu=A", "8.3.5|lu=B", "8.3.7", "9.4.7"])
    @test cb.mode === :l3 && cb.eco_l3 == [1, 1, 2, 3] && cb.eco_l2 == [1, 1, 1, 2]
    @test Pan.cma_blocks("eco", ["8.3.5"]).mode === :eco && isempty(Pan.cma_blocks("eco", ["8.3.5"]).eco_l3)
  end
end

@testset "ParamSpace: options decide which parameters exist (the former BSP switches)" begin
  names(o; kw...) = [p.name for p in mkspace(; opts=o, kw...).dists.params]
  O(; kw...) = PU.ParamSpaceOptions(; kw...)
  @test names(O()) == [:D, :LONGEVITY, :MATURITY, :SHADE_TOL, :S, :B_MAX_SPP, :ANPP_MAX_SPP, :PROB_MORT_SPP, :PROB_ESTAB_SPP, :MIN_REL_BIOMASS]
  @test names(O(fix_growth=true)) == [:LONGEVITY, :MATURITY, :SHADE_TOL, :PROB_MORT_SPP, :PROB_ESTAB_SPP, :MIN_REL_BIOMASS]   # {D,S,B_MAX,ANPP_MAX} pinned
  @test :MATURITY ∉ names(O(fix_maturity=true)) && :MIN_REL_BIOMASS ∉ names(O(fix_min_rel=true)) && :PROB_MORT_SPP ∉ names(O(fix_pmort=true))
  # no_establishment: PROB_MORT is searched only with the Sim A death term, in its own box
  @test :PROB_MORT_SPP ∉ names(O(); no_establishment=true) && :PROB_MORT_SPP ∈ names(O(death_pmort_sima=true); no_establishment=true)
  pm(o; kw...) = (sp = mkspace(; opts=o, kw...); sp.dists.params[findfirst(p -> p.name === :PROB_MORT_SPP, sp.dists.params)])
  @test pm(O(pmort_upper_b=0.05f0)).bounds == (0f0, 0.05f0) && pm(O(death_pmort_sima=true, pmort_upper_a=0.1f0); no_establishment=true).bounds == (0f0, 0.1f0)
  # boxes
  sp = mkspace(; opts=O(bmax_bounds=(12000, 40000), anpp_ratio_bounds=(18f0, 30f0)))
  @test sp.dists.params[pidx(sp.dists, :B_MAX_SPP)].bounds == (12000, 40000) && sp.dists.params[pidx(sp.dists, :ANPP_MAX_SPP)].bounds == (18f0, 30f0)
  # two spaces in one process do not leak into each other (no module state)
  @test length(mkspace(; opts=O(fix_growth=true)).slots) < length(mkspace().slots) && length(mkspace().slots) == 4 * 3 + 5 * NESS + 2
  # with_dists re-derives the layout; with_floors keeps it
  s0 = mkspace(); s1 = PU.with_dists(s0, mkspace(; no_establishment=true).dists)
  @test length(s1.slots) == 2 * 3 + 3 * NESS && PU.with_floors(s0, PU.ParamFloors()).slots === s0.slots
  # the space is immutable and its slot layout equals build_slots of a template of the same data
  @test !ismutabletype(PU.ParamSpace) && s0.slots == PU.build_slots(s0, Pan.generate_params(PC0, SPECIES, ECOS, ESI; rng=Random.Xoshiro(1)))
end

@testset "dominance mask (explicit, allocation-free)" begin
  sl = ["PIEL", "QURU", "_H", "_GRP1"]
  @test Pan.dominance_mask(sl, "all") == Bool[]
  @test Pan.dominance_mask(sl, "exact") == Bool[1, 1, 0, 0]
  @test Pan.dominance_mask(sl, "exact+grp") == Bool[1, 1, 0, 1]
  @test Pan.dominance_mask(["_H", "_S"], "exact") == Bool[]            # an empty selection means all species (as the former empty Set did)
  dom = Pan.dominance_mask(sl, "exact")
  @test Pan._dom_included(dom, 1) && !Pan._dom_included(dom, 3) && Pan._dom_included(Bool[], 3)
  f(d) = (n = 0; for g in 1:4; Pan._dom_included(d, g) && (n += 1); end; n)
  f(dom); @test (@allocated f(dom)) == 0
  # the mask rides in the reference scales and is shared by every derived scale set
  seed = PU.LossScales(dom=dom)
  @test seed.dom === dom && PU.LossScales().dom == Bool[]
end

@testset "misc slot helpers: nativevec, quanta, integer_u_min_std, sobol" begin
  let
    sp, tpl, slots = setup(); pd = sp.dists
    nv = PU.params_to_nativevec(tpl, sp, slots)
    @test length(nv) == length(slots) && nv[1] == tpl.D[1] && nv[13] == tpl.S[1][1] && nv[18] == tpl.B_MAX_SPP[1][1]
    q = PU.slot_quanta(sp, slots)
    @test q[4] == 50.0 && q[1] == Float64(pd.params[1].sigma) && q[18] == 100.0       # quantum if set, else sigma
    s = PU.integer_u_min_std(sp, slots)
    @test s[1] == 0.0 && s[13] == 0.0                                                   # continuous priors: no floor
    @test s[4] ≈ 0.3 / 11                                                               # LONGEVITY: (600-100)/50+1 levels
    @test s[7] ≈ 0.3 / 50 && s[10] ≈ 0.3 / 5                                            # MATURITY (no quantum, step 1), SHADE_TOL
    @test s[18] ≈ 0.3 / 231                                                             # B_MAX: 23000/100+1 levels
    # sobol_samples: N decoded params, all within bounds
    ss = PU.sobol_samples(sp, tpl, 8)
    @test length(ss) == 8 && all(p -> all(x -> 12000 <= x <= 35000, reduce(vcat, p.B_MAX_SPP)), ss)
    @test ss[1].B_MAX_SPP == [fill(23500f0, 3), fill(23500f0, 2)]    # Sobol.jl's first emitted point is the centre u=0.5 (the origin is skipped)
    # saltelli design: N*(K+2) samples with matching tags; K = number of distinct param names
    smp, tags = PU.saltelli_design(sp, tpl, 4)
    K = length(pd.params)
    @test length(smp) == 4 * (K + 2) && length(tags) == length(smp)
    @test tags[1:4] == fill("A", 4) && tags[5:8] == fill("B", 4) && tags[9] == "AB:D" && tags[end] == "AB:MIN_REL_BIOMASS"
  end
end

@testset "mutate_params: weights, pooling, ANPP ratio, pinned slots (B9)" begin
  let
    sp, tpl, slots = setup(); pd = sp.dists
    p0 = PU.u_to_params(fill(0.5, length(slots)), sp, slots, tpl)
    # 1. parameter choice follows the sampling weights — including the LAST parameter (never chosen before)
    w = diff(vcat(0.0, pd.weights_cumsum)); n = 20_000; cnt = zeros(Int, length(w)); rng = Random.Xoshiro(3)
    for _ in 1:n; _, k = PU.mutate_params(p0, sp; rng=rng, slots=slots); cnt[k] += 1; end
    @test all(abs.(cnt ./ n .- w) .< 0.02)
    @test cnt[end] > 0
    # 2. ANPP stays ANPP: B_MAX / ratio with the ratio inside its box (the old code wrote ratio values 20–35 into ANPP)
    ai = pidx(pd, :ANPP_MAX_SPP); bi = pidx(pd, :B_MAX_SPP)
    one_param(k) = (c = zeros(length(pd.weights_cumsum)); c[k:end] .= 1.0; c)   # dynamic_cumsum that always picks param k
    rng = Random.Xoshiro(5); p = p0
    for _ in 1:200
      p, k = PU.mutate_params(p, sp; rng=rng, slots=slots, dynamic_cumsum=one_param(ai), mutation_mode=PU.BothMutations)
      @test k == ai
    end
    for (e, ids) in enumerate(ESI), j in eachindex(ids)
      r = p.B_MAX_SPP[e][j] / p.ANPP_MAX_SPP[e][j]
      @test 20 - 1e-3 <= r <= 35 + 1e-3
    end
    # 3. mutating B_MAX keeps each target's ratio (ANPP re-derived), and changes nothing else
    q = p
    for _ in 1:50; q, _ = PU.mutate_params(q, sp; rng=rng, slots=slots, dynamic_cumsum=one_param(bi)); end
    @test q.B_MAX_SPP != p.B_MAX_SPP
    for (e, ids) in enumerate(ESI), j in eachindex(ids)
      @test isapprox(q.B_MAX_SPP[e][j] / q.ANPP_MAX_SPP[e][j], p.B_MAX_SPP[e][j] / p.ANPP_MAX_SPP[e][j]; rtol=1e-4)
    end
    @test q.S == p.S && q.D == p.D
    # 4. pooled parameter: one move changes every cell of the pooled slot identically
    spool = retie(sp; tying=PU.ParamTying(labels=Dict(:B_MAX_SPP => Dict((1, 1) => "a", (2, 1) => "a", (1, 2) => "b", (1, 3) => "c", (2, 2) => "c"))))
    r = p0
    for _ in 1:30
      r, _ = PU.mutate_params(r, spool; rng=rng, slots=spool.slots, dynamic_cumsum=one_param(bi))
      @test r.B_MAX_SPP[1][1] == r.B_MAX_SPP[2][1] && r.B_MAX_SPP[1][3] == r.B_MAX_SPP[2][2]
    end
    # 5. a slot pinned by a floor above the box cannot change: returns unchanged instead of looping forever
    spin = PU.with_floors(sp, PU.ParamFloors(bmax=Dict{Tuple{Int,Int},Float32}((e, j) => 40000f0 for (e, ids) in enumerate(ESI) for j in eachindex(ids))))
    pin = PU.u_to_params(fill(0.5, length(slots)), spin, slots, tpl)
    t = @elapsed (pp, _) = PU.mutate_params(pin, spin; rng=rng, slots=slots, dynamic_cumsum=one_param(bi))
    @test pp.B_MAX_SPP == pin.B_MAX_SPP && t < 5
    # 6. loss-weighted slot choice (SamplingContext): all moves land on the only species with loss
    ctx = PU.SamplingContext(Float32[0, 1, 0])          # only species 2 ("B", eco 1 local 2) carries loss
    m = p0
    for _ in 1:30
      m2, _ = PU.mutate_params(m, sp; rng=rng, slots=slots, dynamic_cumsum=one_param(bi), ctx=ctx)
      changed = [(e, j) for (e, ids) in enumerate(ESI) for j in eachindex(ids) if m2.B_MAX_SPP[e][j] != m.B_MAX_SPP[e][j]]
      @test isempty(changed) || changed == [(1, 2)]
      m = m2
    end
  end
end

end
