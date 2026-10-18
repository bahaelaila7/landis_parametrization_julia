# HBAD2 in the scenario engine (stage 2): yaml roster, tree-based plots (4 subplot sites), plot-level metrics, management (residual targets met exactly,
# percent_biomass, clearcut, species selection, planting + planted flag), negligible-cohort cleanup (option, off by default), thread independence.
# Needs the national FIA database (PAN_FIA_DB, opened READ_ONLY); skipped when it is not available.
module UnitScenarioHBAD2
using Test, Pan, DuckDB, DataFrames, SHA, Dates
const P = Pan
const HB2 = Pan.Plugins.HBAD2Plugin
const YAML_PATH = joinpath(dirname(@__DIR__), "results", "params_yaml", "r4_hbad2_v2g.yaml")

@testset "hbad2 roster v2h (yaml) == results/hbad2/v2h csv" begin
  pv, metav = P.params_from_yaml(read(joinpath(dirname(@__DIR__), "results", "params_yaml", "r4_hbad2_v2h.yaml"), String))
  @test pv isa HB2.HBAD2Params && metav.plugin == "hbad2" && length(pv.SPECIES_LIST) == 12
  ctxv = HB2.HBAD2Context(String.(pv.SPECIES_LIST), [Int.(i) for i in pv.ECO_SPECIES_IDS]; params_dir=joinpath(dirname(@__DIR__), "results", "hbad2", "v2h"), verbose=false)
  qv = P.generate_params(ctxv, String.(pv.SPECIES_LIST), String.(pv.ECO_LIST), [Int.(i) for i in pv.ECO_SPECIES_IDS])
  @test isempty(P.params_diff(pv, qv))
  pv2, _ = P.params_from_yaml(P.params_to_yaml(pv; name="x", species_map=metav.species_map, eco_mapping=metav.eco_mapping))
  @test isempty(P.params_diff(pv, pv2))
  pg, _ = P.params_from_yaml(read(YAML_PATH, String))
  @test !isempty(P.params_diff(pg, pv))          # v2h is a different parameter set than v2g
end

@testset "hbad2 roster (yaml)" begin
  text = read(YAML_PATH, String)
  p, meta = P.params_from_yaml(text)
  @test p isa HB2.HBAD2Params && meta.plugin == "hbad2"
  @test p.SPECIES_LIST == p.KEYS && length(p.SPECIES_LIST) == 12
  # identical to the parameters built from the csv files of results/hbad2
  ctx = HB2.HBAD2Context(String.(p.SPECIES_LIST), [Int.(i) for i in p.ECO_SPECIES_IDS]; verbose=false)
  q = P.generate_params(ctx, String.(p.SPECIES_LIST), String.(p.ECO_LIST), [Int.(i) for i in p.ECO_SPECIES_IDS])
  @test isempty(P.params_diff(p, q))
  # exact round trip
  p2, _ = P.params_from_yaml(P.params_to_yaml(p; name="x", species_map=meta.species_map, eco_mapping=meta.eco_mapping))
  @test isempty(P.params_diff(p, p2))
  # unknown keys / missing parameters / bad species_map targets stay errors
  @test_throws ErrorException P.params_from_yaml(replace(text, "mSD:" => "mSDx:"; count=1))
  @test_throws ErrorException P.params_from_yaml(replace(text, "rho0:" => "rhoZ:"; count=1))
  @test_throws ErrorException P.params_from_yaml(replace(text, "\"PITA\"" => "\"NOPE\""; count=1))
  @test meta.species_map["PITA"] == "PITA" && meta.species_map["QUAL"] == "_H" && meta.species_map["ABAM"] == "_S"
end

@testset "cleanup option (kernel)" begin
  @test HB2.HBAD2Const().cleanup_tph == 0.0                       # OFF by default: stage-1 parity untouched
  KEYS = ["ACRU", "LIST2", "LITU", "MAVI2", "NYBI", "PIEL", "PITA", "QULA3", "QUNI", "TAAS", "_H", "_S"]
  n = length(KEYS)
  function run(thr)
    pcfg = P.PluginConfig(HB2.HBAD2Context(KEYS, [collect(1:n)]; verbose=false, fixed=HB2.HBAD2Const(cleanup_tph=thr)))
    # one site: a dense hardwood stand + two nearly extinct cohorts
    sp = [3, 3, 3, 7, 7, 7, 11, 11, 11]; N = [60.0, 240.0, 60.0, 5e-4, 2e-3, 5e-4, 30.0, 120.0, 30.0]
    D = fill(8.0, 9); D[4:6] .= 4.0; H = fill(9.0, 9); A = fill(30.0, 9); zq = repeat([-sqrt(3.0), 0.0, sqrt(3.0)], 3)
    soa = P.make_sites_hbad2_arrays(pcfg, [1, 10], sp, A, N, D, H, zq; rng=P.RNGType(1), no_establishment=true)
    params = P.generate_params(pcfg.ctx, KEYS, ["eco"], [collect(1:n)])
    ctx = P.make_ctx(pcfg.ctx, P.generate_eco_params(pcfg.ctx, params))
    for y in 1:5; P.step!(P.succession_plugin_of(soa), soa, y, ctx); end
    soa
  end
  s0 = run(0.0); s1 = run(1e-3)
  @test Int(s0.scalar.live[1]) == 9                                # off: cohorts are never removed
  @test Int(s1.scalar.live[1]) == 7                                # on: the cohorts below 1e-3 trees/ha (5e-4 x2) are gone
  @test all(s1.csr.c_tph[1:7] .>= 1e-3)
  # the surviving cohorts keep their order and evolve (almost) identically
  keep = [1, 2, 3, 5, 7, 8, 9]
  @test maximum(abs.(s1.csr.c_tph[1:7] .- s0.csr.c_tph[keep]) ./ s0.csr.c_tph[keep]) < 1e-5
end

const SCEN = P.ScenarioConfig()
if !isfile(SCEN.fia_db)
  @info "unit_scenario_hbad2: FIA database not found ($(SCEN.fia_db)): scenario tests skipped"
else
  db = DuckDB.DB(SCEN.fia_db; readonly=true); con = DuckDB.DBInterface.connect(db)
  cns = String.(DataFrame(DuckDB.DBInterface.execute(con, """SELECT p.CN cn FROM PLOT p JOIN (SELECT DISTINCT plt_cn k FROM $(SCEN.cohort_table) WHERE epa_l3 IN ('8.3.5','8.5.3')) c
    ON c.k = p.STATECD || '_' || p.UNITCD || '_' || p.COUNTYCD || '_' || p.PLOT WHERE p.INVYR >= 2015 ORDER BY p.CN LIMIT 3""")).cn)
  DuckDB.DBInterface.close!(con); DuckDB.close(db)
  ps = P.load_paramset(read(YAML_PATH, String), SCEN)
  plots = [(cn=c, acres=10.0 + i) for (i, c) in enumerate(cns)]

  # the engine's own start-up for the plots (ecoregion from the plot's location), then the state helpers
  function start(; cleanup=1e-3, seed=1)
    w = String[]; info = P.load_scenario_units(SCEN, plots, w); meta = NamedTuple[]
    for r in info
      e, _ = P.scenario_eco(ps, r.l3, r.siteclcd)
      push!(meta, (cn=r.cn, plot_id=length(meta) + 1, eco=e, measdate=r.measdate, l4=r.l4, l3=r.l3, lat=r.lat, lon=r.lon, key=r.key))
    end
    pc, soa = P._hbad2_sites(ps, meta, seed, false, cleanup, w)
    units = [findall(i -> Int(soa.scalar.mapcode[i]) == m.plot_id, 1:soa.n) for m in meta]
    return pc, soa, units
  end
  plot_mean(soa, sites, f) = sum(f(P.getsite(soa, i)) for i in sites) / length(sites)
  tph(s) = sum(Float64(s.c_tph[k]) for k in 1:Int(s.live); init=0.0)
  ba_ft2ac(s) = sum(pi / 4 * (Float64(s.c_dq[k]) / 100)^2 * Float64(s.c_tph[k]) for k in 1:Int(s.live); init=0.0) * P.SC_M2HA_TO_FT2AC
  sdi_ac(s) = sum(Float64(s.c_tph[k]) * P.SC_HA_PER_AC * (Float64(s.c_dq[k]) / 25.4)^1.605 for k in 1:Int(s.live); init=0.0)
  harvest(type, value, dir="below") = Dict{String,Any}("kind" => "harvest", "selection" => Dict("species" => "all"), "removal" => Dict("type" => type, "value" => value, "direction" => dir))
  all_ok = sp -> true

  @testset "plots: 4 subplot sites per plot, per-ha state, native metrics" begin
    pc, soa, units = start()
    @test soa.n == 4 * length(cns) && all(length(u) == 4 for u in units)
    @test all(Int(soa.scalar.eco_id[i]) > 0 for i in 1:soa.n)
    @test !any(soa.scalar.planted)
    for u in units
      sm = P._plot_summary(ps, [P.getsite(soa, i) for i in u])
      @test sm.tpa > 0 && sm.ba > 0 && sm.qmd > 0 && sm.tcuft > 0 && isfinite(sm.agb) && sm.agb > 0
      # plot mean of the per-ha site densities
      @test sm.tpa ≈ plot_mean(soa, u, tph) * P.SC_HA_PER_AC rtol = 1e-12
      @test sm.ba ≈ plot_mean(soa, u, ba_ft2ac) rtol = 1e-12
    end
  end

  @testset "removal: residual targets are met exactly by the plot mean" begin
    for (typ, f, cur) in (("residual_tpa", tph, s -> tph(s) * P.SC_HA_PER_AC), ("residual_ba", ba_ft2ac, ba_ft2ac), ("residual_sdi", sdi_ac, sdi_ac)),
        dir in ("below", "above", "proportional")
      pc, soa, units = start()
      u = units[1]; sites = [P.getsite(soa, i) for i in u]
      now = plot_mean(soa, u, cur); target = 0.55 * now
      rb, rv, rs = P._apply_removal!(ps, pc, sites, harvest(typ, target, dir), all_ok)
      @test plot_mean(soa, u, cur) ≈ target rtol = 1e-9
      @test rb > 0 && rv > 0
      # a target above the current value removes nothing
      @test P._apply_removal!(ps, pc, sites, harvest(typ, 10 * now, dir), all_ok) == (0.0, 0.0, 0.0)
      # other plots untouched
      @test plot_mean(soa, units[2], tph) > 0
    end
  end

  @testset "removal: percent_biomass, clearcut, species selection" begin
    pc, soa, units = start()
    u = units[1]; sites = [P.getsite(soa, i) for i in u]
    before = [copy(Float64.(s.c_tph[1:Int(s.live)])) for s in sites]
    P._apply_removal!(ps, pc, sites, harvest("percent_biomass", 30.0), all_ok)
    for (s, b) in zip(sites, before)
      @test Float64.(s.c_tph[1:Int(s.live)]) ≈ 0.7 .* b rtol = 1e-12
      @test sum(Float64.(s.c_bio[1:Int(s.live)])) ≈ Float64(s.B) rtol = 1e-5
    end
    # species selection: only PITA cohorts are cleared
    pc, soa, units = start()
    sites = [P.getsite(soa, i) for i in units[1]]
    pita = findfirst(==("PITA"), String.(ps.params.SPECIES_LIST))
    other = sum(sum(Float64(s.c_tph[k]) for k in 1:Int(s.live) if Int(s.c_species[k]) != pita; init=0.0) for s in sites)
    P._apply_removal!(ps, pc, sites, harvest("clearcut", 0.0), sp -> sp == "PITA")
    @test all(s -> all(Int(s.c_species[k]) != pita for k in 1:Int(s.live)), sites)
    @test sum(sum(Float64(s.c_tph[k]) for k in 1:Int(s.live); init=0.0) for s in sites) ≈ other rtol = 1e-12
    # clearcut empties the stand
    pc, soa, units = start()
    sites = [P.getsite(soa, i) for i in units[1]]
    P._apply_removal!(ps, pc, sites, harvest("clearcut", 0.0), all_ok)
    @test all(s -> Int(s.live) == 0 && Float64(s.B) == 0, sites)
  end

  @testset "planting: density, size, planted flag" begin
    pc, soa, units = start()
    u = units[1]; t0 = plot_mean(soa, u, tph)
    w = String[]
    P._plant_all!(ps, pc, soa, u, Dict{String,Any}("species" => "PITA", "tpa" => 200.0, "survival" => 50.0), w)
    @test isempty(w)
    @test plot_mean(soa, u, tph) - t0 ≈ 100.0 / P.SC_HA_PER_AC rtol = 1e-9          # trees/ha added = tpa x survival / 0.4047
    @test all(i -> soa.scalar.planted[i], u)                                      # the plot's sites are a planted stand now ...
    @test !any(soa.scalar.planted[i] for i in 1:soa.n if !(i in u))               # ... other plots are not
    pita = findfirst(==("PITA"), String.(ps.params.SPECIES_LIST))
    for i in u
      s = P.getsite(soa, i)
      ks = [k for k in 1:Int(s.live) if Int(s.c_species[k]) == pita && s.c_age[k] == 1.0]
      @test length(ks) == 3 && all(k -> s.c_dq[k] == 2.54 && s.c_h[k] == 3.0, ks)
      @test sort(Float64[s.c_zq[k] for k in ks]) == sort([-sqrt(3.0), 0.0, sqrt(3.0)])
    end
  end

  @testset "scenario run: harvest (residual_ba) + plant, finite metrics, determinism" begin
    plan = Dict("cycles" => 4, "period" => 5, "actions" => Any[
      Dict("kind" => "harvest", "cycle" => 1, "selection" => Dict("species" => "all"), "removal" => Dict("type" => "residual_ba", "value" => 60.0, "direction" => "below"), "stands" => Dict("rank" => "all", "area_pct" => 100)),
      Dict("kind" => "plant", "cycle" => 2, "species" => "PITA,QUAL", "tpa" => 300.0, "survival" => 80.0)])
    res, warns = P.run_scenario(ps, plots, plan; seed=3)
    @test length(res) == length(cns)
    for cn in cns
      r = res[cn]; @test r.status == "ok" && length(r.metrics) == 5
      for m in r.metrics
        @test all(isfinite, (m.ba, m.tpa, m.qmd, m.tcuft, m.bdft, m.rem_tcuft, m.rem_bdft, m.carbon_total, m.carbon_live_ag, m.carbon_removed, m.agb))
        @test m.tpa >= 0 && m.ba >= 0
      end
      @test r.metrics[2].rem_tcuft >= 0                                          # row of cycle 1 carries the harvest
      @test all(!isempty(d) for d in values(r.dist))
    end
    res2, _ = P.run_scenario(ps, plots, plan; seed=3)
    h(res) = bytes2hex(sha256(join((string(res[c].metrics) for c in cns), "\n")))
    @test h(res) == h(res2)
    # a plot without data is reported, not simulated
    r0, _ = P.run_scenario(ps, [(cn="0", acres=1.0)], plan)
    @test r0["0"].status == "no_data"
  end

  @testset "thread independence of the annual step on scenario plots" begin
    function run_steps(serial)
      pc, soa, units = start()
      soa.parallel = !serial
      params = ps.params; eco = P.generate_eco_params(pc, params); ctx = P.make_ctx(pc, eco)
      SP = P.succession_plugin_of(soa)
      for y in 1:30; P.step!(SP, soa, y, ctx); end
      io = IOBuffer()          # live cohorts only (the CSR slack beyond `live` is uninitialised memory)
      for i in 1:soa.n
        o = Int(soa.refs.cohort[i]) - 1
        for k in (:c_species, :c_age, :c_tph, :c_dq, :c_h, :c_zq, :c_bio); write(io, getfield(soa.csr, k)[o+1:o+Int(soa.scalar.live[i])]); end
      end
      write(io, soa.scalar.live); bytes2hex(sha256(take!(io)))
    end
    @test run_steps(true) == run_steps(false)
  end

  @testset "cleanup in scenario runs keeps metrics (small effect) and shrinks the cohort count" begin
    pc0, soa0, _ = start(; cleanup=0.0); pc1, soa1, _ = start(; cleanup=1e-3)
    @test pc0.fixed.cleanup_tph == 0.0 && pc1.fixed.cleanup_tph == 1e-3
    eco0 = P.make_ctx(pc0, P.generate_eco_params(pc0, ps.params)); eco1 = P.make_ctx(pc1, P.generate_eco_params(pc1, ps.params))
    SP = P.succession_plugin_of(soa0)
    for y in 1:60; P.step!(SP, soa0, y, eco0); P.step!(SP, soa1, y, eco1); end
    @test sum(soa1.scalar.live) < sum(soa0.scalar.live)
    @test sum(soa1.scalar.BA) ≈ sum(soa0.scalar.BA) rtol = 1e-3
  end
end
end
