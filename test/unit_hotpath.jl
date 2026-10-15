# Hot-path guarantees of the loss kernels (the per-site accumulators run inside the threaded site loop of fit_params):
#   * type-stable: no `Any` / abstract result types in the inferred code of the accumulator methods (the check behind `@code_warntype` having no red);
#   * allocation-free: one accumulator call allocates nothing once warmed up (scratch buffers are caller-owned).
# Exercised for the BiomassSuccession plugin on a small synthetic site, for the main option combinations (W hinge, AGB hinge, softplus, count-balance,
# global and per-cell normalisation).
module UnitHotpath
using Test, Random, Pan
const PU = Pan.PU
const PC = Pan.PanCore
const FT = PC.FloatType
const UT = PC.UIntType
const BaseSite = Pan.Plugins.BaseSitePlugin.BaseSite
const BS = Pan.Plugins.BiomassSuccessionPlugin.BiomassSuccession
const SoaBS = PC.SiteSoA{Tuple{BaseSite,BS}}

function make_site(rng; ncoh=12, nsp=3, ages_max=90)
  soa = SoaBS((cohort=Int32[ncoh], species=Int32[nsp]))
  s = PC.getsite(soa, 1)
  s.active = true; s.rng = Random.Xoshiro(1); s.mapcode = 1; s.ecocode = 1; s.eco_id = 1; s.ref_cn = 1; s.old = 0; s.live = ncoh
  s.sp_mature .= false
  for k in 1:ncoh
    s.c_species[k] = UT(rand(rng, 1:nsp)); s.c_age[k] = FT(rand(rng, 1:ages_max)); s.c_bio[k] = FT(rand(rng) * 3000 + 10)
  end
  soa
end
function make_ref(rng, lp; nsp=3, keys=[true, true, false], maxage=100)
  recs = Dict{UT,PU.SPDFRecord}()
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

# Red flags in the optimised, inferred code of a call: statements typed `Any` (or any non-concrete, non-small-Union type) whose value is USED, followed
# through every non-inlined callee defined in the Pan modules (`:invoke` targets) — what `@code_warntype` would show in red. Statements whose value is
# discarded (`resize!(...)`, `gc_preserve_begin`, ...) carry an `Any` type in the IR but are not a stability problem. `Union{Nothing,T}` of concrete
# members (the optional `excluded` / `excl_c` arguments) is the only union allowed.
function _ssa_uses(x, used::Set{Int})
  x isa Core.SSAValue && push!(used, x.id)
  if x isa Expr
    foreach(a -> _ssa_uses(a, used), x.args)
  elseif x isa Core.ReturnNode
    isdefined(x, :val) && _ssa_uses(x.val, used)
  elseif x isa Core.GotoIfNot
    _ssa_uses(x.cond, used)
  elseif x isa Core.PhiNode
    for k in eachindex(x.values); isassigned(x.values, k) && _ssa_uses(x.values[k], used); end
  elseif x isa Core.PiNode
    _ssa_uses(x.val, used)
  end
  used
end
function red_flags(tt::Type; seen=Set{Any}(), depth=0)
  bad = Any[]
  depth > 6 && return bad
  tt in seen && return bad
  push!(seen, tt)
  for (ci, _) in Base.code_typed_by_type(tt; optimize=true)
    ci isa Core.CodeInfo || continue
    used = Set{Int}(); foreach(st -> _ssa_uses(st, used), ci.code)
    for (i, st) in enumerate(ci.code)
      (st isa Core.ReturnNode || st isa Core.GotoNode || st isa Core.GotoIfNot) && continue
      T = ci.ssavaluetypes[i]
      T isa Type || continue
      i in used || continue                                            # a discarded value cannot destabilise anything downstream
      (st isa Expr && st.head in (:gc_preserve_begin, :foreigncall)) && continue   # compiler-internal handles, not Julia values
      if T === Any || (T isa Union ? any(!isconcretetype, Base.uniontypes(T)) : (!isconcretetype(T) && !(T <: Type) && !(T <: Tuple)))
        push!(bad, (T, st))
      end
      if st isa Expr && st.head === :invoke && st.args[1] isa Core.MethodInstance
        mi = st.args[1]
        parentmodule(mi.def.module) === parentmodule(@__MODULE__) || mi.def.module in (Pan, PU, PC, Pan.Plugins.BiomassSuccessionPlugin, Pan.Plugins.BaseSitePlugin) || continue
        append!(bad, red_flags(mi.specTypes; seen=seen, depth=depth + 1))
      end
    end
  end
  bad
end
function body_sig(f, argtypes::Tuple, kwtypes::Tuple)    # signature of the keyword-body method of `f(args...; kw...)`
  m = which(f, Tuple{argtypes...})
  Tuple{kwtypes..., typeof(f), argtypes...}, Base.bodyfunction(m)
end

@testset "hot path: calculate_site_loss2_acc! / calculate_species_loss_acc!" begin
  rng = Random.Xoshiro(3)
  esids = [[1, 2, 3]]; nsp = 3
  for (whinge, ahinge, softplus, cbal, cellnorm) in ((true, true, true, true, true), (true, true, false, false, false), (true, false, false, false, true), (true, true, true, false, false))
    lp = PU.LossParams(age_bins=PU.AgeBins(bins_idx=[10, 20, 40, 80], last_bin_open=true), smoothing_weights=FT[1], lambda=FT(1),
      w_hinge=whinge, agb_hinge=ahinge, w_softplus=softplus, cbal_on=cbal, cell_norm=cellnorm, w_normalize=!cellnorm, agb_normalize=!cellnorm, w_p=FT(2))
    nb = 5
    cb = cbal ? PU.CbalTables(reshape([FT.(0.5 .+ rand(rng, nb)) for _ in 1:3], 3, 1), collect(1:4)) : PU.NO_CBAL
    sc = PU.LossScales(w_global=FT(3), agb_global=FT(1000), softplus_beta=FT(0.01), softplus_beta_cell=softplus ? fill(FT(0.02), 3, 1) : zeros(FT, 0, 0), cbal=cb)
    soa = make_site(rng); site = PC.getsite(soa, 1); gt = make_ref(rng, lp)
    w = zeros(FT, nsp); a = zeros(FT, nsp)
    perm = Int[]; ages = FT[]; insite = Bool[]; cdf = FT[]; agb_bins = FT[]
    call() = PU.calculate_site_loss2_acc!(w, a, 5, site, nsp, esids, gt, lp, sc, perm, ages, insite, cdf, agb_bins)
    call(); call()                                   # warm-up (compiles; sizes the scratch buffers)
    @test (@allocated call()) == 0                   # one accumulator call allocates nothing
    @test (@allocated call()) == 0
    # type stability of the site accumulator and of the per-species kernel
    argt = (Vector{FT}, Vector{FT}, Int, typeof(site), Int, Vector{Vector{Int}}, typeof(gt), typeof(lp), typeof(sc), Vector{Int}, Vector{FT}, Vector{Bool}, Vector{FT}, Vector{FT})
    sig, bodyf = body_sig(PU.calculate_site_loss2_acc!, argt, (FT, Nothing, Nothing))   # kw defaults: lp, excluded, excl_c
    @test isempty(red_flags(Tuple{typeof(bodyf), sig.parameters...}))
    @test Base.return_types(PU.calculate_site_loss2_acc!, argt)[1] == FT
    # the leaf kernels
    p = sortperm(site.c_species)
    sp_call() = PU.calculate_species_loss_acc!(; sp=site.c_species[p[1]], gsp=Int(site.c_species[p[1]]), site=site, ages=ages, p=p, sp_start_idx=1, sp_end_idx=1,
      spdf_plt=gt, loss_params=lp, scales=sc, w_acc=w, agb_acc=a, site_agb_loss=FT(0), cdf=cdf, agb_bins=agb_bins)
    sp_call(); sp_call()
    @test (@allocated sp_call()) == 0
    @test (@allocated PU._w_finish(lp, sc, FT(1.5), 1, 1)) == 0
    @test (@allocated PU._agb_finish(lp, sc, FT(1.5))) == 0
    @test (@allocated PU._hinge_relu(lp, FT(1.5))) == 0
    @test (@allocated PU._w1_sum(lp, sc, gt.records[UT(1)].sp_age_cdf, gt.records[UT(1)].sp_age_cdf, lp.age_bins.bin_widths, 1, 1)) == 0
  end
end

# The injection / shock functions run per measurement year inside the site loop of fit_params: with the options passed as an immutable struct they must
# not allocate (removal of a fully disturbed cohort + scaling of a partially disturbed one, and the scoring-order predicate).
@testset "hot path: _apply_disturbance_shock! / shock_order" begin
  rng = Random.Xoshiro(5)
  soas = [make_site(Random.Xoshiro(5)) for _ in 1:4]            # identical fresh sites: the removal path runs once per site, so measure on a fresh one
  s0 = PC.getsite(soas[1], 1); live0 = Int(s0.live)
  sp1, age1 = s0.c_species[1], s0.c_age[1]; sp2, age2 = s0.c_species[2], s0.c_age[2]
  scoring = Pan.ScoringOptions(; override=true, sync=true, disturbance=:scale)
  shk = Pan._SiteShockYear([(1, Tuple{UT,FT}[(sp1, age1)])])
  inj = Pan._SiteInjectionYear([(1, Tuple{UT,FT,FT,FT}[(sp2, age2, FT(100), FT(0.4))])])
  Pan._apply_disturbance_shock!(soas[1], shk, inj, scoring)    # warm-up (compiles)
  Pan._apply_disturbance_shock!(soas[2], shk, inj, scoring)
  @test (@allocated Pan._apply_disturbance_shock!(soas[3], shk, inj, scoring)) == 0
  @test Int(PC.getsite(soas[3], 1).live) == live0 - 1             # it did remove the shocked cohort
  off = Pan.with_options(scoring; disturbance=:off)
  Pan._apply_disturbance_shock!(soas[4], nothing, inj, off)
  @test (@allocated Pan._apply_disturbance_shock!(soas[4], nothing, inj, off)) == 0
  @test (@allocated Pan.shock_order(scoring)) == 0
  @test Base.return_types(Pan._apply_disturbance_shock!, (typeof(soas[1]), typeof(shk), typeof(inj), Pan.ScoringOptions))[1] === Nothing
  @test isempty(red_flags(Tuple{typeof(Pan._apply_disturbance_shock!), typeof(soas[1]), typeof(shk), typeof(inj), Pan.ScoringOptions}))
end
end # module
