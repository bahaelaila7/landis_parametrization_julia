# HBAD2 calibration: interval product, kernel entry points vs the vendored prototype kernel, parameter layout, config schema.
# (The objectives against the prototype harness on the real data are tools/hbad2_calib_parity.jl; the end-to-end search is test/hbad2_calib_e2e.jl.)
module UnitHBAD2Calib
using Test, Random, Pan, DuckDB, DataFrames, Dates, CSV
module Proto
include("ref_hbad2_proto.jl")
end
include("hbad2_parity_lib.jl")
const PR = Proto.HBAD2
const D = Pan.Data
const HC = Pan.HBAD2Calib

# ---------------------------------------------------------------------------------------------------------------- interval product on a synthetic tree table
function synth_trees_db(path)
  db = DuckDB.DB(path); con = DuckDB.DBInterface.connect(db)
  DuckDB.execute(con, """CREATE TABLE curated_trees(STATECD INT, UNITCD INT, COUNTYCD INT, PLOT INT, SUBP INT, TREE INT, MEASDATE DATE, STATUSCD INT, tree_fate VARCHAR,
    spcd_resolved INT, DIA DOUBLE, HT DOUBLE, CCLCD DOUBLE, CR DOUBLE, TPA_UNADJ DOUBLE, estimated_age DOUBLE, epa_l3 VARCHAR, cond_stdorgcd INT, cond_siteclcd INT, cond_sicond DOUBLE,
    TRTCD1 INT, DSTRBCD1 INT, CN VARCHAR)""")
  DuckDB.execute(con, "CREATE TABLE REF_SPECIES(SPCD INT, SPECIES_SYMBOL VARCHAR, SFTWD_HRDWD VARCHAR)")
  DuckDB.execute(con, "INSERT INTO REF_SPECIES VALUES (131, 'PITA', 'S'), (316, 'ACRU', 'H')")
  cn = Ref(0)
  function row(plot, subp, tree, date, status, fate, spcd, dia, ht, age; trt=0, dst=0, stdorg=2, tpa=6.018046)
    cn[] += 1
    DuckDB.execute(con, "INSERT INTO curated_trees VALUES (12, 1, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, 2.0, 30.0, ?, ?, '8.5.3', ?, 3, 70.0, ?, ?, ?)",
                   [plot, subp, tree, Date(date), status, fate, spcd, dia, ht, tpa, age, stdorg, trt, dst, "cn$(cn[])"])
  end
  # plot 1: three visits. tree 1 alive throughout; tree 2 dies; tree 3 harvested; tree 4 missing at v2; tree 5 ingrowth at v2; tree 6 (subplot 2) alive throughout
  for (d, dia) in (("2000-06-01", 10.0), ("2006-06-01", 11.0), ("2012-06-01", 12.0))
    row(1, 1, 1, d, 1, "live", 131, dia, 60.0, 30.0)
    row(1, 2, 6, d, 1, "live", 316, dia + 1, 50.0, 25.0)
  end
  row(1, 1, 2, "2000-06-01", 1, "live", 316, 8.0, 40.0, 20.0); row(1, 1, 2, "2006-06-01", 2, "mortality", 316, 8.0, 40.0, 20.0)
  row(1, 1, 3, "2000-06-01", 1, "live", 316, 9.0, 41.0, 21.0); row(1, 1, 3, "2006-06-01", 2, "harvest_cut", 316, 9.0, 41.0, 21.0)
  row(1, 1, 4, "2000-06-01", 1, "live", 131, 7.0, 35.0, 18.0)
  row(1, 1, 5, "2006-06-01", 1, "live", 316, 5.0, 30.0, 8.0); row(1, 1, 5, "2012-06-01", 1, "live", 316, 6.0, 34.0, 14.0)
  # plot 2: a single visit (no interval)
  row(2, 1, 1, "2001-06-01", 1, "live", 131, 20.0, 80.0, 40.0)
  # plot 3: two visits, planted at v1, a treatment code recorded at v2
  row(3, 1, 1, "2002-06-01", 1, "live", 131, 12.0, 50.0, 12.0; stdorg=1); row(3, 1, 1, "2010-06-01", 1, "live", 131, 14.0, 56.0, 20.0; trt=10)
  DuckDB.DBInterface.close!(con); DuckDB.close(db)
  return path
end

mktempdir() do dir
  db = synth_trees_db(joinpath(dir, "trees.duckdb"))
  split = DataFrame(statecd=[12, 12, 12], unitcd=[1, 1, 1], countycd=[1, 1, 1], plot=[1, 2, 3], set=["train", "val", "test"], eco=["a", "b", "c"])
  x = D.build_hbad2_intervals(db, split)
  iv = x.intervals
  @testset "interval product" begin
    @test nrow(iv) == 3 && iv.iid == [1, 2, 3]                       # plot 1: two intervals, plot 3: one, plot 2: none
    @test iv.p == [1, 1, 3] && iv.k == [1, 2, 1] && iv.set == ["train", "train", "test"]
    @test iv.d1 == [Date(2000, 6, 1), Date(2006, 6, 1), Date(2002, 6, 1)] && iv.d2 == [Date(2006, 6, 1), Date(2012, 6, 1), Date(2010, 6, 1)]
    @test iv.dt[1] ≈ Dates.value(Date(2006, 6, 1) - Date(2000, 6, 1)) / 365.25
    @test iv.planted == [false, false, true] && iv.trt == [false, false, true] && iv.harv == [true, false, false] && iv.dst == [false, false, false]   # harvest fate seen at v2 of interval 1
    tr = x.trees
    t1 = sort(tr[tr.iid .== 1, :], [:sb, :tr])
    @test t1.tr == [1, 2, 3, 4, 6] && t1.out == ["alive", "dead", "harvest", "missing", "alive"]
    @test t1.dia2_cm[1] ≈ 11.0 * 2.54 && isnan(t1.dia2_cm[2]) && t1.dia_cm[1] ≈ 10.0 * 2.54 && t1.ht_m[1] ≈ 60.0 * 0.3048 && t1.w_ha[1] ≈ 6.018046 * 2.47105
    @test t1.sym[1] == "PITA" && t1.grp[1] == "S" && t1.sym[2] == "ACRU" && t1.grp[2] == "H"
    @test sort(tr[tr.iid .== 2, :tr]) == [1, 5, 6]                    # tree 5 is tracked in the second interval
    ig = x.ingrowth
    @test nrow(ig) == 1 && ig.iid == [1] && ig.sb == [1] && ig.dia_cm[1] ≈ 5.0 * 2.54 && ig.age[1] == 8.0     # first tallied at v2 of interval 1
    @test D.compare_interval_products(x, x).trees.only_a == 0
    # Arrow cache: round trip, key check, rebuild
    cache = joinpath(dir, "cache")
    y = D.load_hbad2_intervals(cache, db, split)
    z = D.load_hbad2_intervals(cache, db, split)
    c = D.compare_interval_products(z, x)
    @test c.intervals.only_a == 0 && c.intervals.only_b == 0 && c.intervals.differing == 0 && c.trees.only_a == 0 && c.trees.only_b == 0 && c.ingrowth.only_a == 0 && c.ingrowth.only_b == 0
    split2 = copy(split); split2.set[1] = "val"
    @test_throws ErrorException D.load_hbad2_intervals(cache, db, split2)          # cache of another split
    @test nrow(D.load_hbad2_intervals(cache, db, split2; rebuild=true).intervals) == 3
    @test_throws ErrorException D.build_hbad2_intervals(joinpath(dir, "nope.duckdb"), split)
    @test D.list_hbad2_plots(db) == DataFrame(statecd=[12, 12, 12], unitcd=[1, 1, 1], countycd=[1, 1, 1], plot=[1, 2, 3])
    # the split writer / reader and the drawn split
    sp = D.draw_plot_split(DataFrame(statecd=fill(12, 40), unitcd=1, countycd=1, plot=1:40); val_frac=0.25, test_frac=0.1, seed=7)
    @test count(==("val"), sp.set) == 10 && count(==("test"), sp.set) == 4 && count(==("train"), sp.set) == 26
    @test sp == D.draw_plot_split(DataFrame(statecd=fill(12, 40), unitcd=1, countycd=1, plot=40:-1:1); val_frac=0.25, test_frac=0.1, seed=7)     # input order does not matter
    CSV.write(joinpath(dir, "split.csv"), sp); @test D.read_split_plots(joinpath(dir, "split.csv")) == sp
  end
end

# ---------------------------------------------------------------------------------------------------------------- kernel entry points vs the vendored prototype
@testset "kernel entry points" begin
  P, G, R, keys_ = proto_params(PR, HB2.DEFAULT_PARAMS_DIR)
  nsp = length(keys_)
  pcfg = Pan.PluginConfig(HB2.HBAD2Context(keys_, [collect(1:nsp)]; verbose=false))
  ep = Pan.generate_eco_params(pcfg.ctx, Pan.generate_params(pcfg.ctx, keys_, ["eco"], [collect(1:nsp)]))[1]
  # Sim A: whole years + fractional step
  init = synth_init(Random.Xoshiro(5), 80, nsp; cap_extra=0)
  ns = length(init.live0)
  nyr = [2 + s % 9 for s in 1:ns]; frac = [s % 3 == 0 ? 0.0 : 0.13 + 0.07 * (s % 11) for s in 1:ns]
  S = PR.Sites(copy(init.off), Int32.(init.sp), copy(init.N0), copy(init.D0), copy(init.H0), copy(init.A0), nyr, frac, copy(init.planted), copy(init.zq0))
  PR.simulate!(S, P, G)
  soa = pan_soa(pcfg, init; no_establishment=true)
  mx = 0.0; nco = 0
  for s in 1:ns
    HB2.run_interval!(soa, s, ep, nyr[s], frac[s])
    o = Int(soa.refs.cohort[s]) - 1; n = Int(soa.scalar.live[s]); lo = init.off[s]
    for c in 1:n
      for (a, b) in ((soa.csr.c_tph[o+c], S.N[lo+c-1]), (soa.csr.c_dq[o+c], S.D[lo+c-1]), (soa.csr.c_h[o+c], S.H[lo+c-1]), (soa.csr.c_age[o+c], S.A[lo+c-1]))
        mx = max(mx, abs(a - b) / max(abs(b), 1e-300))
      end
    end
    nco += n
  end
  @test nco > 200 && mx == 0.0
  # free runs with a capacity that forces dropped recruit classes
  for cap in (400, 14)
    init2 = synth_init(Random.Xoshiro(9), 40, nsp; cap_extra=cap, nmax=4)
    ns2 = length(init2.live0)
    recP, droppedP = proto_free(PR, init2, P, G, R, 30)
    off2 = ones(Int, ns2 + 1); for s in 1:ns2; off2[s+1] = off2[s] + init2.live0[s]; end
    idx = reduce(vcat, [collect(init2.off[s]:init2.off[s]+init2.live0[s]-1) for s in 1:ns2]; init=Int[])
    soa2 = Pan.make_sites_hbad2_arrays(pcfg, off2, init2.sp[idx], init2.A0[idx], init2.N0[idx], init2.D0[idx], init2.H0[idx], init2.zq0[idx];
                                      rng=Random.Xoshiro(1), no_establishment=false, planted=init2.planted, extra_capacity=cap)
    dropped = 0; bad = 0; mx2 = 0.0
    for y in 1:30, s in 1:ns2
      dropped += HB2.site_year_free!(soa2, s, ep, 3)
      o = Int(soa2.refs.cohort[s]) - 1; n = Int(soa2.scalar.live[s]); r = recP[y][s]
      if n != length(r[1]) || Float64.(soa2.csr.c_species[o+1:o+n]) != r[1]; bad += 1; continue; end
      for c in 1:n; mx2 = max(mx2, abs(soa2.csr.c_tph[o+c] - r[2][c]) / max(abs(r[2][c]), 1e-300), abs(soa2.csr.c_dq[o+c] - r[3][c]) / max(abs(r[3][c]), 1e-300)); end
    end
    @test bad == 0 && mx2 == 0.0 && dropped == droppedP
    cap == 14 && @test droppedP > 0
    cap == 400 && @test droppedP == 0
  end
end

# ---------------------------------------------------------------------------------------------------------------- parameter layout
@testset "theta layout and decode" begin
  sw = HC.CalibSwitches()
  @test sw.form == 2 && sw.size_split == [5.0, 12.7, 25.0, 40.0] && sw.site_sub && sw.site_mult == 4.0 && sw.vigour && sw.sigv_max == 0.7
  @test sw.islog == (true, true, false, true, true, false, false, false, false, false, false, false, true, true)      # form 2: cD, cH linear
  @test HC.CalibSwitches(form=1).islog[3] && HC.CalibSwitches(form=1).lo[1] == 8.0 && sw.lo[1] == 2.0 && sw.hi[4] == 300.0
  @test HC.CalibSwitches(bounds=Dict("gOD" => [0.0, 5.0])).hi[7] == 5.0
  @test_throws ErrorException HC.CalibSwitches(bounds=Dict("nope" => [0.0, 5.0]))
  @test_throws ErrorException HC.CalibSwitches(form=3)
  keys_ = ["ACRU", "PITA", "_H"]
  L = HC.ThetaLayout(keys_; recr=:kappa, freeze=["mbg", "mD", "G:mD2", "G:sigma_v"])
  @test L.ng == 3 * 14 + 7 && length(L) == L.ng + 1 && L.nrec == 1
  @test sort(L.frozen) == sort([(s - 1) * 14 + 11 for s in 1:3] ∪ [(s - 1) * 14 + 12 for s in 1:3] ∪ [3 * 14 + 1, 3 * 14 + 6])
  θ0 = collect(range(-1.0, 1.0; length=length(L)))
  z = HC.restrict(L, θ0); @test length(z) == length(L.free_idx) && HC.expand(L, θ0, z) == θ0
  z2 = z .+ 0.5; θ2 = HC.expand(L, θ0, z2); @test all(θ2[L.frozen] .== θ0[L.frozen]) && all(θ2[L.free_idx] .== θ0[L.free_idx] .+ 0.5)
  @test_throws ErrorException HC.ThetaLayout(keys_; freeze=["bogus"])
  @test_throws ErrorException HC.ThetaLayout(keys_; freeze=["G:bogus"])
  P, G, σw = HC.decode_theta(vcat(zeros(3 * 14), [10.0, 5.0, -5.0, 9.0, 9.0, -3.0, log(0.5)]), 3, sw, [150.0, 300.0, 400.0])
  @test G.mD2 == 3.0 && G.pH == 1.0 && G.pD == -1.0 && G.pM == 2.0 && G.mSD == 2.0 && G.σv == 0.7 && G.mA == 10.0 && G.form == 2 && σw ≈ 0.5   # clamps; sigma_v = min(|.|, bound)
  @test P[2].longevity == 300.0 && P[1].Hmax == 2.0 && P[1].rH == 1.0 && P[1].cD == 0.0 && P[1].Dmax == 5.0      # zeros clamp into the form-2 box (Hmax, Dmax log-space; cD linear)
  g0 = HC.decode_theta(vcat(zeros(3 * 14), [0.0, 0.0, 0.0, 0.0, 0.0, 5.0, -10.0]), 3, HC.CalibSwitches(vigour=false), [1.0, 1.0, 1.0])
  @test g0[2] == HC.HBAD2Plugin.GlobP(0.0, 10.0, 0.0, 0.0, 0.0, 2, 0.0, 0.0) && g0[3] == 0.02                      # no vigour: sigma_v = 0; sigma_w floor
  R = HC.recruitment_of(vcat(zeros(3 * 14 + 7), 0.4), L, [-3.0, 0.5, -2.0, -4.0, 0.1, -1.0, -5.0, 0.2, -3.0])
  @test R[1].rho0 ≈ -3.0 + 0.4 && R[2].rhoS == 0.1 && R[3].rhoL == -3.0 && R[1].D0 == 2.54 && R[1].A0 == 5.0
  # prototype layouts of earlier fits are padded
  mktempdir() do d
    CSV.write(joinpath(d, "t.csv"), DataFrame(theta=collect(1.0:(3 * 14 + 6))))
    @test length(HC.read_theta_csv(joinpath(d, "t.csv"), 3)) == 3 * 14 + 7 && HC.read_theta_csv(joinpath(d, "t.csv"), 3)[end] ≈ log(0.11)
    @test_throws ErrorException HC.read_theta_csv(joinpath(d, "t.csv"), 4)
  end
end

# ---------------------------------------------------------------------------------------------------------------- config schema
@testset "hbad2_calib config" begin
  cfg = Pan.load_config(Dict{String,Any}("succession_plugin" => "hbad2", "hbad2_calib" => Dict{String,Any}("objective" => "joint", "lambda" => [0.75], "freeze" => ["mbg", "G:mD2"], "form" => 2)))
  hc = cfg["hbad2_calib"]
  @test hc["objective"] == "joint" && hc["lambda"] == [0.75] && hc["freeze"] == ["mbg", "G:mD2"] && hc["sigv_max"] == 0.7 && hc["recruitment"] == "fixed" && hc["size_split"] == [5.0, 12.7, 25.0, 40.0]
  @test Pan.is_hbad2_calibration(cfg) && !Pan.is_hbad2_calibration(Pan.load_config(Dict{String,Any}("succession_plugin" => "hbad2")))
  @test_throws Pan.ConfigError Pan.load_config(Dict{String,Any}("hbad2_calib" => Dict{String,Any}("lamda" => [0.5])))             # unknown key, with suggestion
  @test_throws Pan.ConfigError Pan.load_config(Dict{String,Any}("hbad2_calib" => Dict{String,Any}("objective" => "bogus")))
  @test_throws Pan.ConfigError Pan.load_config(Dict{String,Any}("hbad2_calib" => Dict{String,Any}("lambda" => [1.5])))
  @test_throws Pan.ConfigError Pan.load_config(Dict{String,Any}("hbad2_calib" => Dict{String,Any}("ideal" => [1.0])))
  @test_throws Pan.ConfigError Pan.load_config(Dict{String,Any}("hbad2_calib" => Dict{String,Any}("start_recr" => "x.csv")))        # only with recruitment: free
  @test length(Pan.HBAD2_CALIB_SCHEMA) == length(Pan.HBAD2_CALIB_LIST) && all(s -> !isempty(s.doc), Pan.HBAD2_CALIB_LIST)
  for s in Pan.HBAD2_CALIB_LIST                                                       # every default coerces back to itself
    s.default === nothing && continue
    @test Pan._resolve_value("hbad2_calib.", s, deepcopy(s.default)) == s.default
  end
  @test occursin("hbad2_calib.freeze", Pan.config_docs())
end
end
