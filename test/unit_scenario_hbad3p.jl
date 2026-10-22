# HBAD3P in the scenario engine: yaml roster, tree-based plots (4 subplot sites, diameter classes closed below, forest-area convention a = max(f, 0.25), crown ratio, plot-normal water W = Wn),
# plot-level metrics (x area), management at constant tree size (residual targets met exactly, percent_biomass, clearcut, species selection, planting + planted flag), scenario runs (no-op and thinning plan),
# determinism and thread independence. Needs the national FIA database (PAN_FIA_DB, opened READ_ONLY); skipped when it is not available.
module UnitScenarioHBAD3P
using Test, Pan, DuckDB, DataFrames, SHA, Dates
const P = Pan
const H3 = Pan.Plugins.HBAD3PPlugin
const YAML_PATH = joinpath(dirname(@__DIR__), "results", "params_yaml", "r4_hbad3p_gafl.yaml")

const SCEN = P.ScenarioConfig()
if !isfile(SCEN.fia_db)
  @info "unit_scenario_hbad3p: FIA database not found ($(SCEN.fia_db)): scenario tests skipped"
else
  db = DuckDB.DB(SCEN.fia_db; readonly=true); con = DuckDB.DBInterface.connect(db)
  # a few Georgia / Florida plots (curated cohorts present, recent visit): the region of the fitted parameters; two of them have partially forested subplots (forest-area convention a < 1)
  base_q(extra) = """SELECT p.CN cn FROM PLOT p JOIN (SELECT DISTINCT plt_cn k FROM $(SCEN.cohort_table) WHERE epa_l3 IN ('8.3.5','8.5.3')) c
    ON c.k = p.STATECD || '_' || p.UNITCD || '_' || p.COUNTYCD || '_' || p.PLOT WHERE p.INVYR >= 2015 AND p.STATECD IN (12, 13) $(extra) ORDER BY p.CN LIMIT 2"""
  part = String.(DataFrame(DuckDB.DBInterface.execute(con, base_q("""AND EXISTS (SELECT 1 FROM SUBP_COND s JOIN COND k ON k.PLT_CN = s.PLT_CN AND k.CONDID = s.CONDID WHERE s.PLT_CN = p.CN
        GROUP BY s.SUBP HAVING sum(CASE WHEN k.COND_STATUS_CD = 1 THEN s.SUBPCOND_PROP ELSE 0 END) BETWEEN 0.3 AND 0.8)"""))).cn)
  full = String.(DataFrame(DuckDB.DBInterface.execute(con, base_q(""))).cn)
  cns = unique(vcat(part, full))
  DuckDB.DBInterface.close!(con); DuckDB.close(db)
  ps = P.load_paramset(read(YAML_PATH, String), SCEN)
  plots = [(cn=c, acres=10.0 + i) for (i, c) in enumerate(cns)]
  @test ps.plugin === H3.HBAD3P && ps.h3 !== nothing && length(ps.h3.wn) > 20_000 && isempty(ps.meta.warnings)

  function start(; cleanup=1e-3, seed=1)
    w = String[]; info = P.load_scenario_units(SCEN, plots, w); meta = NamedTuple[]
    for r in info
      e, _ = P.scenario_eco(ps, r.l3, r.siteclcd)
      push!(meta, (cn=r.cn, plot_id=length(meta) + 1, eco=e, measdate=r.measdate, l4=r.l4, l3=r.l3, lat=r.lat, lon=r.lon, key=r.key))
    end
    pc, soa = P._hbad3p_sites(ps, meta, seed, false, cleanup, w)
    units = [findall(i -> Int(soa.scalar.mapcode[i]) == m.plot_id, 1:soa.n) for m in meta]
    return pc, soa, units, meta, w
  end
  plot_mean(soa, sites, f) = sum(f(P.getsite(soa, i)) for i in sites) / length(sites)
  tph(s) = sum(Float64(s.c_tph[k]) for k in 1:Int(s.live); init=0.0) * Float64(s.area)
  ba_ft2ac(s) = sum(pi / 4 * (Float64(s.c_dq[k]) / 100)^2 * Float64(s.c_tph[k]) for k in 1:Int(s.live); init=0.0) * Float64(s.area) * P.SC_M2HA_TO_FT2AC
  sdi_ac(s) = sum(Float64(s.c_tph[k]) * P.SC_HA_PER_AC * (Float64(s.c_dq[k]) / 25.4)^1.605 for k in 1:Int(s.live); init=0.0) * Float64(s.area)
  harvest(type, value, dir="below") = Dict{String,Any}("kind" => "harvest", "selection" => Dict("species" => "all"), "removal" => Dict("type" => type, "value" => value, "direction" => dir))
  all_ok = sp -> true

  @testset "plots: 4 subplot sites per plot, forest area, water normal, crown ratio, classes closed below" begin
    pc, soa, units, meta, w = start()
    @test soa.n == 4 * length(cns) && all(length(u) == 4 for u in units)
    @test all(Int(soa.scalar.eco_id[i]) > 0 for i in 1:soa.n) && !any(soa.scalar.planted)
    @test all(0.25 .<= soa.scalar.area .<= 1.0) && all(soa.scalar.rmult .>= 0)
    @test any(<(1.0), soa.scalar.area)                         # the partially forested subplots are scaled (a = max(f, 0.25))
    # W = Wn = the plot normal of the table (or the regional median, with a warning)
    for (m, u) in zip(meta, units)
      wn = get(ps.h3.wn, m.key, NaN)
      for i in u
        @test soa.scalar.W[i] == soa.scalar.Wn[i]
        @test soa.scalar.W[i] == (isnan(wn) ? ps.h3.wn_median : wn)
      end
    end
    for u in units
      sm = P._plot_summary(ps, [P.getsite(soa, i) for i in u])
      @test sm.tpa > 0 && sm.ba > 0 && sm.qmd > 0 && sm.tcuft > 0 && isfinite(sm.agb) && sm.agb > 0
      @test sm.tpa ≈ plot_mean(soa, u, tph) * P.SC_HA_PER_AC rtol = 1e-12
      @test sm.ba ≈ plot_mean(soa, u, ba_ft2ac) rtol = 1e-12
    end
    # crown ratio in [0.05, 0.95]; vigour nodes in groups of five; ages > 0
    for i in 1:soa.n
      s = P.getsite(soa, i); n = Int(s.live)
      n == 0 && continue
      @test n % 5 == 0 && all(0.05 .<= s.c_cr[1:n] .<= 0.95) && all(s.c_age[1:n] .> 0) && all(s.c_h[1:n] .>= 1.37)
      @test s.c_zq[1:5] == collect(H3.VIG_Z)
    end
    # independent recomputation of the stem density and basal area of the first plot's visit from the tree table (per subplot: 4 sum(w) / a)
    db2 = DuckDB.DB(SCEN.fia_db; readonly=true); c2 = DuckDB.DBInterface.connect(db2)
    try
      m = meta[1]; k = parse.(Int, split(m.key, "_"))
      tr = DataFrame(DuckDB.DBInterface.execute(c2, """SELECT SUBP sb, DIA, TPA_UNADJ tpa FROM curated_trees WHERE STATECD=$(k[1]) AND UNITCD=$(k[2]) AND COUNTYCD=$(k[3]) AND PLOT=$(k[4])
            AND MEASDATE = DATE '$(Dates.Date(m.measdate))' AND STATUSCD=1 AND DIA>0 AND HT>0 AND estimated_age>0 AND TPA_UNADJ>0"""))
      for sb in 1:4
        t = tr[tr.sb .== sb, :]; s = P.getsite(soa, units[1][sb]); n = Int(s.live)
        @test sum(s.c_tph[1:n]) * Float64(s.area) ≈ 4 * sum(Float64.(t.tpa) .* 2.47105; init=0.0) rtol = 1e-10
        ba_ref = sum(pi / 4 * (Float64.(t.DIA) .* 2.54 ./ 100) .^ 2 .* Float64.(t.tpa) .* 2.47105; init=0.0) * 4
        ba_sim = sum(pi / 4 * (Float64.(s.c_dq[1:n]) ./ 100) .^ 2 .* Float64.(s.c_tph[1:n]); init=0.0) * Float64(s.area)
        @test ba_sim ≈ ba_ref rtol = 1e-10                     # quadratic means preserve the basal area
      end
    finally
      DuckDB.DBInterface.close!(c2); DuckDB.close(db2)
    end
  end

  @testset "removal: constant tree size; residual targets met exactly by the plot mean" begin
    for (typ, cur) in (("residual_tpa", s -> tph(s) * P.SC_HA_PER_AC), ("residual_ba", ba_ft2ac), ("residual_sdi", sdi_ac)), dir in ("below", "above", "proportional")
      pc, soa, units, = start()
      u = units[1]; sites = [P.getsite(soa, i) for i in u]
      now = plot_mean(soa, u, cur); target = 0.55 * now
      before = [(copy(s.c_dq[1:Int(s.live)]), copy(s.c_h[1:Int(s.live)]), copy(s.c_cr[1:Int(s.live)]), copy(s.c_age[1:Int(s.live)]), copy(s.c_species[1:Int(s.live)]), Int(s.live)) for s in sites]
      rb, rv, rs = P._apply_removal!(ps, pc, sites, harvest(typ, target, dir), all_ok)
      @test plot_mean(soa, u, cur) ≈ target rtol = 1e-9
      @test rb > 0 && rv > 0
      # cohorts keep their size (dq, h, cr, age): only counts change; whole cohorts leave by swap-removal, so match by value
      for (s, b) in zip(sites, before)
        for k in 1:Int(s.live)
          j = findfirst(i -> b[1][i] == s.c_dq[k] && b[2][i] == s.c_h[k] && b[4][i] == s.c_age[k] && b[5][i] == s.c_species[k], 1:b[6])
          @test j !== nothing && s.c_cr[k] == b[3][j]
        end
      end
      @test P._apply_removal!(ps, pc, sites, harvest(typ, 10 * now, dir), all_ok) == (0.0, 0.0, 0.0)
      @test plot_mean(soa, units[2], tph) > 0
    end
  end

  @testset "removal: percent_biomass (QMD of the removed trees = QMD of the cohort), clearcut, species selection" begin
    pc, soa, units, = start()
    u = units[1]; sites = [P.getsite(soa, i) for i in u]
    before = [(copy(Float64.(s.c_tph[1:Int(s.live)])), copy(Float64.(s.c_dq[1:Int(s.live)]))) for s in sites]
    P._apply_removal!(ps, pc, sites, harvest("percent_biomass", 30.0), all_ok)
    for (s, b) in zip(sites, before)
      n = Int(s.live)
      @test Float64.(s.c_tph[1:n]) ≈ 0.7 .* b[1] rtol = 1e-12
      @test Float64.(s.c_dq[1:n]) == b[2]
      @test sum(Float64.(s.c_bio[1:n])) ≈ Float64(s.B) rtol = 1e-5
      @test Float64(s.B) > 0 || n == 0
    end
    pc, soa, units, = start()
    sites = [P.getsite(soa, i) for i in units[1]]
    sp1 = String(ps.params.SPECIES_LIST[Int(sites[findfirst(s -> Int(s.live) > 0, sites)].c_species[1])])
    other = sum(sum(Float64(s.c_tph[k]) for k in 1:Int(s.live) if String(ps.params.SPECIES_LIST[Int(s.c_species[k])]) != sp1; init=0.0) for s in sites)
    P._apply_removal!(ps, pc, sites, harvest("clearcut", 0.0), sp -> sp == sp1)
    @test all(s -> all(String(ps.params.SPECIES_LIST[Int(s.c_species[k])]) != sp1 for k in 1:Int(s.live)), sites)
    @test sum(sum(Float64(s.c_tph[k]) for k in 1:Int(s.live); init=0.0) for s in sites) ≈ other rtol = 1e-12
    pc, soa, units, = start()
    sites = [P.getsite(soa, i) for i in units[1]]
    P._apply_removal!(ps, pc, sites, harvest("clearcut", 0.0), all_ok)
    @test all(s -> Int(s.live) == 0 && Float64(s.B) == 0, sites)
  end

  @testset "planting: density per forested ha, recruit size, planted flag" begin
    pc, soa, units, = start()
    u = units[1]; t0 = plot_mean(soa, u, tph)
    w = String[]
    P._plant_all!(ps, pc, soa, u, Dict{String,Any}("species" => "PITA", "tpa" => 200.0, "survival" => 50.0), w)
    @test isempty(w)
    added = 100.0 / P.SC_HA_PER_AC                                                  # trees per ha of the forested stand = tpa x survival / 0.4047
    @test plot_mean(soa, u, tph) - t0 ≈ added * sum(soa.scalar.area[i] for i in u) / 4 rtol = 1e-9
    @test all(i -> soa.scalar.planted[i], u) && !any(soa.scalar.planted[i] for i in 1:soa.n if !(i in u))
    pita = findfirst(==("PITA"), String.(ps.params.SPECIES_LIST))
    for i in u
      s = P.getsite(soa, i)
      ks = [k for k in 1:Int(s.live) if Int(s.c_species[k]) == pita && s.c_age[k] == 1.0]
      @test length(ks) == 5 && all(k -> s.c_dq[k] == 2.54 && s.c_cr[k] == pc.glob.crhi, ks)
      @test sort(Float64[s.c_zq[k] for k in ks]) == collect(H3.VIG_Z)
    end
  end

  @testset "scenario runs: no-op and thinning plans (finite, plausible, deterministic)" begin
    noop = Dict("cycles" => 5, "period" => 5, "actions" => Any[])
    thin = Dict("cycles" => 5, "period" => 5, "actions" => Any[
      Dict("kind" => "harvest", "cycle" => 1, "selection" => Dict("species" => "all"), "removal" => Dict("type" => "residual_tpa", "value" => 100.0, "direction" => "below"), "stands" => Dict("rank" => "all", "area_pct" => 100)),
      Dict("kind" => "plant", "cycle" => 3, "species" => "PITA", "tpa" => 300.0, "survival" => 80.0)])
    r0, w0 = P.run_scenario(ps, plots, noop; seed=3)
    r1, w1 = P.run_scenario(ps, plots, thin; seed=3)
    for cn in cns
      a = r0[cn]; b = r1[cn]
      @test a.status == "ok" && length(a.metrics) == 6 && b.status == "ok" && length(b.metrics) == 6
      for m in vcat(a.metrics, b.metrics)
        @test all(isfinite, (m.ba, m.tpa, m.qmd, m.tcuft, m.bdft, m.rem_tcuft, m.rem_bdft, m.carbon_total, m.carbon_live_ag, m.carbon_removed, m.agb))
        @test m.tpa >= 0 && 0 <= m.ba <= 400 && m.qmd >= 0
      end
      # no-op: basal area stays within a plausible band of its start over 25 years (no collapse, no blow-up)
      @test all(0.4 * a.metrics[1].ba .<= [m.ba for m in a.metrics] .<= 2.5 * a.metrics[1].ba)
      # thinning at cycle 1 (applied after the row of cycle 1): rows 2.. have fewer trees per acre than the no-op after the removal, the removed volume is booked on row 2
      @test b.metrics[1].tpa == a.metrics[1].tpa && b.metrics[1].ba == a.metrics[1].ba          # same start, row 1 is before any management
      @test b.metrics[2].rem_tcuft > 0 || a.metrics[2].tpa <= 100 / 1                              # (stands already below the residual target remove nothing)
      @test b.metrics[2].tpa <= a.metrics[2].tpa + 1e-9
      @test b.metrics[3].ba <= a.metrics[3].ba * 1.0000001 || b.metrics[3].tpa <= a.metrics[3].tpa  # thinned stand carries less basal area / fewer trees right after the thinning
    end
    r0b, _ = P.run_scenario(ps, plots, noop; seed=3)
    h(res) = bytes2hex(sha256(join((string(res[c].metrics) for c in cns), "\n")))
    @test h(r0) == h(r0b)
    r, _ = P.run_scenario(ps, [(cn="0", acres=1.0)], noop)
    @test r["0"].status == "no_data"
  end

  @testset "climate: water index from the scenario climate (normal, baseline = no anomaly, drought scenario, year mapping)" begin
    if !isfile(SCEN.climate_db)
      @info "climate DB not found: HBAD3P climate tests skipped"
    else
      pc, soa, units, meta, w = start()
      wat = P._hbad3p_water!(ps, soa, meta, 60, nothing, w)
      @test wat !== nothing
      for m in meta
        k4 = join(split(m.key, "_"), "_"); tab = get(ps.h3.wn, k4, NaN)
        wn = soa.scalar.Wn[units[m.plot_id][1]]
        @test all(i -> soa.scalar.Wn[i] == wn, units[m.plot_id])
        isnan(tab) || @test abs(wn - tab) < 0.02          # the fit's table: L4-mean latitude, FIA-plot soil; here the plot's latitude / location soil
        ww = wat.W[m.plot_id]; yrs = wat.first_year:(wat.first_year + length(ww) - 1)
        fut = [ww[i] for (i, y) in enumerate(yrs) if y >= 2024 + P.HBAD3P_WATER_WINDOW]
        @test !isempty(fut) && all(x -> x ≈ wn, fut)        # the unchanged baseline is the normal: no spurious anomaly
        obs = [ww[i] for (i, y) in enumerate(yrs) if 1995 <= y <= 2020]
        @test maximum(obs) - minimum(obs) > 0.005             # observed years keep their own (smoothed) variability
      end
      dry = Dict{String,Any}("baseline" => Dict{String,Any}("mode" => "forecast"), "co2" => Dict{String,Any}("mode" => "constant"),
              "changes" => Any[Dict{String,Any}("var" => "tmean", "kind" => "add", "value" => 3.0, "shape" => "step", "start_year" => 2025),
                               Dict{String,Any}("var" => "ppt", "kind" => "mul", "value" => -30.0, "shape" => "step", "start_year" => 2025)])
      pc2, soa2, units2, meta2, w2 = start()
      wd = P._hbad3p_water!(ps, soa2, meta2, 60, dry, w2)
      for m in meta2
        ww = wd.W[m.plot_id]; yrs = wd.first_year:(wd.first_year + length(ww) - 1); wn = soa2.scalar.Wn[units2[m.plot_id][1]]
        @test wn ≈ soa.scalar.Wn[units[m.plot_id][1]]       # the normal ignores the scenario's changes
        @test all(ww[i] < wn - 0.02 for (i, y) in enumerate(yrs) if y >= 2030)
        # year mapping: simulated year t = visit year + t − 1
        P._hbad3p_set_water!(soa2, wd, 7); y = Dates.year(m.measdate) + 6
        @test soa2.scalar.W[units2[m.plot_id][1]] == ww[y - wd.first_year + 1]
      end
      noop = Dict("cycles" => 5, "period" => 10, "actions" => Any[])
      rb, _ = P.run_scenario(ps, plots, noop; seed=3)
      rd, _ = P.run_scenario(ps, plots, noop; seed=3, climate=dry)
      rd2, _ = P.run_scenario(ps, plots, noop; seed=3, climate=dry)
      for c in keys(rb)
        rb[c].status == "ok" || continue
        @test rd[c].metrics[end].ba < rb[c].metrics[end].ba           # hotter and drier: less growth, more drought mortality
        @test rd[c].metrics[1].ba == rb[c].metrics[1].ba              # same start
        @test [m.ba for m in rd[c].metrics] == [m.ba for m in rd2[c].metrics]   # deterministic
      end
    end
  end

  @testset "thread independence of the annual step on scenario plots" begin
    function run_steps(serial)
      pc, soa, units, = start()
      soa.parallel = !serial
      eco = P.generate_eco_params(pc, ps.params); ctx = P.make_ctx(pc, eco)
      SP = P.succession_plugin_of(soa)
      for y in 1:30; P.step!(SP, soa, y, ctx); end
      io = IOBuffer()
      for i in 1:soa.n
        o = Int(soa.refs.cohort[i]) - 1
        for k in (:c_species, :c_age, :c_tph, :c_dq, :c_h, :c_zq, :c_cr, :c_bio); write(io, getfield(soa.csr, k)[o+1:o+Int(soa.scalar.live[i])]); end
      end
      write(io, soa.scalar.live); bytes2hex(sha256(take!(io)))
    end
    @test run_steps(true) == run_steps(false)
  end
end
end
