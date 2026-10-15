# src/succession_api.jl: generic succession-plugin interface dispatches to BS / PnET / HBAD (tiny synthetic SoAs, no DB).
module UnitAPI
using Test, Random, Pan
const PC = Pan.PanCore
const BaseSite = Pan.Plugins.BaseSitePlugin.BaseSite
const BS = Pan.Plugins.BiomassSuccessionPlugin.BiomassSuccession
const PN = Pan.Plugins.PnETPlugin.PnET
const HB = Pan.Plugins.HBADPlugin.HBAD
const BSM = Pan.Plugins.BiomassSuccessionPlugin
const PNM = Pan.Plugins.PnETPlugin
const HBM = Pan.Plugins.HBADPlugin

const PCFG = Pan.PluginConfig

# a tiny synthetic PnET climate (one L4, two calendar years), as test/pnet_unit.jl
function tiny_climate()
  tn = Array{Float32,3}(undef, 12, 2, 1); tx = similar(tn); pp = similar(tn)
  for y in 1:2, m in 1:12
    tn[m, y, 1] = 5 + m; tx[m, y, 1] = 15 + m; pp[m, y, 1] = 90
  end
  PNM.build_climate(["8.5.3.test"], [2000, 2001], tn, tx, pp, [400.0, 400.0])
end

const SPECIES = ["PITA", "PIEL", "QUNI"]
const ECOS = ["e1", "e2"]
const ESI = [[1, 2, 3], [1, 3]]

@testset "succession_plugin_of / soa_type" begin
  for P in (BS, PN, HB)
    T = Pan.soa_type(P)
    @test T === PC.SiteSoA{Tuple{BaseSite,P}}
    @test Pan.succession_plugin_of(T) === P                       # type-level
    soa = T((cohort=Int32[2, 2], species=Int32[1, 1]))
    @test Pan.succession_plugin_of(soa) === P                     # instance-level
  end
  # a tuple with two non-BaseSite plugins (or none) is rejected at compile time
  @test_throws Exception Pan.succession_plugin_of(PC.SiteSoA{Tuple{BaseSite,BS,HB},NamedTuple{(),Tuple{}},NamedTuple{(),Tuple{}},NamedTuple{(),Tuple{}}})
  @test_throws Exception Pan.succession_plugin_of(PC.SiteSoA{Tuple{BaseSite},NamedTuple{(),Tuple{}},NamedTuple{(),Tuple{}},NamedTuple{(),Tuple{}}})
end

@testset "generics: method tables per plugin" begin
  ctxs = Dict(BS => BSM.BSTables(), PN => PNM.PnETContext(; climate=tiny_climate()), HB => HBM.HBADContext(SPECIES, ESI; verbose=false))
  for P in (BS, PN, HB)
    pc = ctxs[P]
    @test Pan.plugin_type(pc) === P && Pan.plugin_type(PCFG(pc)) === P          # the plugin type is a type parameter of the config
    @test Pan.bs_tables(pc) isa BSM.BSTables && Pan.bs_tables(PCFG(pc)) isa BSM.BSTables
    @test hasmethod(Pan.step!, Tuple{Type{P},Any,Integer,Any})
    @test hasmethod(Pan.spinup!, Tuple{Type{P},Any,Any,Any})
    @test hasmethod(Pan.add_cohort!, Tuple{typeof(pc),Any})
    @test hasmethod(Pan.make_param_dists, Tuple{typeof(pc),Int,Int,Vector{Vector{Int}}})
    @test hasmethod(Pan.generate_params, Tuple{typeof(pc),Vector{String},Vector{String},Vector{Vector{Int}}})
    @test hasmethod(Pan.generate_eco_params, Tuple{typeof(pc),Any})
    @test hasmethod(Pan.make_ctx, Tuple{typeof(pc),Any})
    @test hasmethod(Pan.per_eco_groups, Tuple{Type{P}})
    @test hasmethod(Pan.plant!, Tuple{typeof(pc),Any,Integer,Real})
    @test hasmethod(Pan.plant_slot_needed, Tuple{typeof(pc),Any,Integer})
    @test hasmethod(Pan.init_sites!, Tuple{typeof(pc),Any,Any})
    @test hasmethod(Pan.set_cohort_bio!, Tuple{typeof(pc),Any,Integer,Any})
  end
  @test Pan.params_type(BS) === BSM.BiomassSuccessionParams
  @test Pan.params_type(PN) === PNM.PnETParams
  @test Pan.params_type(HB) === HBM.HBADParams
  @test Pan.per_eco_groups(BS) === Pan.BSP.BIOMASS_PER_ECO_GROUPS
  @test Pan.per_eco_groups(PN) === Pan.PNP.PNET_PER_ECO_GROUPS
  @test Pan.per_eco_groups(HB) === Pan.HBP.HBAD_PER_ECO_GROUPS
  @test :growth_biomass in Pan.per_eco_groups(BS)
end

@testset "BiomassSuccession end-to-end through the generic API" begin
  P = Pan.succession_plugin_of(Pan.soa_type(BS))
  @test P === BS
  pc = BSM.BSTables()
  pd = Pan.make_param_dists(pc, 3, 2, ESI)
  @test pd isa Pan.PU.ParamDists{BSM.BiomassSuccessionParams}
  params = Pan.generate_params(pc, SPECIES, ECOS, ESI; rng=Random.Xoshiro(1))
  @test params isa Pan.params_type(P)
  @test Pan.succession_plugin_for(params) === P
  eco = Pan.generate_eco_params(pc, params)
  @test length(eco) == 2 && length(eco[1].B_MAX_SPP) == 3 && length(eco[2].B_MAX_SPP) == 2
  @test eco[1].B_MAX_ECO == maximum(params.B_MAX_SPP[1])
  ctx = Pan.make_ctx(pc, eco)
  @test keys(ctx) == (:BiomassSuccession,) && ctx.BiomassSuccession.eco_params === eco
  # tiny SoA: 2 sites, cohort capacity 4
  soa = Pan.soa_type(P)((cohort=Int32[4, 4], species=Int32[3, 2]))
  for i in 1:2
    s = PC.getsite(soa, i)
    s.active = true; s.rng = Random.Xoshiro(UInt64(i)); s.mapcode = i; s.ecocode = i; s.eco_id = i; s.ref_cn = i; s.old = 0; s.live = 0
    Pan._init_site_plugin!(BS, s, true)                                # no_establishment=true
  end
  @test Pan.init_sites!(pc, soa, nothing) === soa                       # BS: no site attributes needed
  s1 = PC.getsite(soa, 1)
  Pan.add_cohort!(pc, s1, UInt32(1), 10f0, 500f0); Pan.add_cohort!(pc, s1, UInt32(2), 20f0, 300f0)
  @test s1.live == 2 && s1.c_species[1:2] == UInt32[1, 2] && s1.c_age[1:2] == Float32[10, 20] && s1.c_bio[1:2] == Float32[500, 300]
  Pan.set_cohort_bio!(pc, s1, 2, 123f0); @test s1.c_bio[2] == 123f0        # BS: plain assignment
  @test Pan.plant_slot_needed(pc, s1, 1) == 1
  # step! delegates to process_plugin!(soa, BS, year; ctx=ctx.BiomassSuccession)
  age0 = copy(s1.c_age[1:2])
  PC.PARALLEL_SITES[] = false
  try
    Pan.step!(P, soa, 1, ctx)
  finally
    PC.PARALLEL_SITES[] = true
  end
  s1 = PC.getsite(soa, 1)
  @test s1.live >= 1 && all(s1.c_age[1:Int(s1.live)] .>= 0)
  # step! equals calling the plugin entry point directly on an identical copy
  a = deepcopy(soa); b = deepcopy(soa)
  PC.PARALLEL_SITES[] = false
  try
    Pan.step!(P, a, 2, ctx); PC.process_plugin!(b, BS, 2; ctx=ctx.BiomassSuccession)
  finally
    PC.PARALLEL_SITES[] = true
  end
  @test a.csr.c_bio == b.csr.c_bio && a.csr.c_age == b.csr.c_age && a.refs == b.refs
  # plant_sites!: reserves slots (readjust_soa!) then plants; BS needs eco_params indexed by eco_id
  n0 = PC.getsite(soa, 2).live
  Pan.plant_sites!(pc, soa, [(2, 1, 0.0)]; eco_params=eco)
  s2 = PC.getsite(soa, 2)
  @test s2.live == n0 + 1 && s2.c_species[Int(s2.live)] == 1 && s2.c_age[Int(s2.live)] == 1f0 && s2.c_bio[Int(s2.live)] >= 2f0
  # inactive site is skipped
  soa.scalar.active[1] = false; l1 = PC.getsite(soa, 1).live
  Pan.plant_sites!(pc, soa, [(1, 1, 0.0)]; eco_params=eco); @test PC.getsite(soa, 1).live == l1
  # spinup! delegates to the BS spinup function
  @test which(Pan.spinup!, Tuple{Type{BS},Any,Any,Any}).sig.parameters[2] == Type{BS}
end

@testset "PnET and HBAD: dispatch, params and documented errors" begin
  # --- PnET
  pnc = PNM.PnETContext(; climate=tiny_climate())
  ppars = Pan.generate_params(pnc, SPECIES, ["8.5.3", "8.5.4"], ESI; rng=Random.Xoshiro(3))
  @test ppars isa PNM.PnETParams && Pan.succession_plugin_for(ppars) === PN
  @test Pan.make_param_dists(pnc, 3, 2, ESI) isa Pan.PU.ParamDists{PNM.PnETParams}
  eco = Pan.generate_eco_params(pnc, ppars); @test length(eco) == 2
  # the PnET ctx carries the context's climate / constants / scratch pool; contexts are immutable (with_climate copies, sharing the scratch pool)
  pctx = Pan.make_ctx(pnc, eco)
  @test keys(pctx) == (:PnET,) && pctx.PnET.clim === pnc.climate && pctx.PnET.cst === pnc.fixed && pctx.PnET.scratch === pnc.scratch
  @test length(pnc.scratch) >= Threads.nthreads() && !ismutabletype(PNM.PnETContext)
  c2 = PNM.with_climate(pnc, tiny_climate()); @test c2.scratch === pnc.scratch && c2.climate !== pnc.climate
  # PFT assignment reads the context's species table (no module state): SFTWD "S" makes an unknown symbol a softwood
  tb = BSM.BSTables(longevity=Dict("PITA" => 99.0))
  @test PNM.pft_of("XXXX", Dict("XXXX" => "S")) == PNM.PFT_NLE && PNM.pft_of("XXXX", Dict("XXXX" => "H")) == PNM.PFT_BLD
  pq = Pan.generate_params(PNM.PnETContext(; climate=tiny_climate(), tables=tb), SPECIES, ["8.5.3", "8.5.4"], ESI; rng=Random.Xoshiro(3))
  @test pq.LONGEVITY[1] == 99f0
  @test_throws ErrorException Pan.spinup!(PN, nothing, nothing, nothing)
  @test_throws ErrorException Pan.plant!(pnc, nothing, 1, 1.0)
  @test_throws ErrorException Pan.plant_slot_needed(pnc, nothing, 1)
  @test_throws ErrorException Pan.init_sites!(pnc, nothing, nothing)               # PnET REQUIRES site attributes
  # --- HBAD
  hbc = HBM.HBADContext(SPECIES, ESI; verbose=false)
  hpars = Pan.generate_params(hbc, SPECIES, ECOS, ESI; rng=Random.Xoshiro(3))
  @test hpars isa HBM.HBADParams && Pan.succession_plugin_for(hpars) === HB
  @test Pan.make_param_dists(hbc, 3, 2, ESI) isa Pan.PU.ParamDists{HBM.HBADParams}
  heco = Pan.generate_eco_params(hbc, hpars); @test length(heco) == 2
  hctx = Pan.make_ctx(hbc, heco)
  @test keys(hctx) == (:HBAD,) && hctx.HBAD.eco === heco && hctx.HBAD.cst === hbc.fixed
  @test_throws ErrorException Pan.spinup!(HB, nothing, nothing, nothing)
  @test Pan.init_sites!(hbc, :soa, nothing) === :soa
  # the former silent re-configuration (ensure_configured!) is an explicit error: a context only serves the species list / eco table it was built for
  @test_throws ErrorException Pan.generate_eco_params(HBM.HBADContext(["PITA", "PIEL"], [[1, 2], [1]]; verbose=false), hpars)
  @test_throws ErrorException Pan.generate_eco_params(HBM.HBADContext(SPECIES, [[1, 2, 3], [2, 3]]; verbose=false), hpars)
  # two contexts for two species lists coexist (no shared state): same params, other priors table -> other eco params
  hb2 = HBM.HBADContext(["QUNI", "PITA", "PIEL"], [[1, 2, 3], [1, 3]]; verbose=false)
  p2 = Pan.generate_params(hb2, ["QUNI", "PITA", "PIEL"], ECOS, ESI; rng=Random.Xoshiro(3))
  @test Pan.generate_eco_params(hbc, hpars)[1].pri[1] === hbc.priors.sp[1] && Pan.generate_eco_params(hb2, p2)[1].pri[1] === hb2.priors.sp[1] && hbc.priors.sp[1] != hb2.priors.sp[1]
  # --- the three plugins expose different ctx keys, keyed by plugin NAME (consumed by PanCore.get_ctx)
  @test PC.get_ctx((HBAD=1,), HB) == 1 && PC.get_ctx((HBAD=1,), BS) == NamedTuple()
  # --- every plugin resolves its OWN params type via params_type and generate_params round trips through succession_plugin_for
  for (P, prm) in ((BS, Pan.generate_params(BSM.BSTables(), SPECIES, ECOS, ESI; rng=Random.Xoshiro(1))), (PN, ppars), (HB, hpars))
    @test typeof(prm) === Pan.params_type(P) && Pan.succession_plugin_for(prm) === P
  end
end
end # module
