# PnET plugin unit tests.   Run:  cd <worktree> && source wt_env.sh && ./julia_gdal.sh --project=. test/pnet_unit.jl
using Test, Random
using Pan
const PN = Pan.Plugins.PnETPlugin
const PC = Pan.PanCore
const BaseSite = Pan.Plugins.BaseSitePlugin.BaseSite

# ------------------------------------------------------------------------------------------------------------------
# Fixture: constant (year-independent) monthly climate for one "L4", 2 calendar years
# ------------------------------------------------------------------------------------------------------------------
const TMIN = Float64[2, 3, 6, 10, 15, 19, 21, 21, 18, 12, 6, 3]
const TMAX = Float64[14, 16, 21, 25, 29, 32, 33, 33, 30, 26, 20, 15]
const PPT = Float64[110, 95, 120, 85, 95, 125, 150, 135, 100, 75, 85, 100]
const CO2 = 400.0

function make_climate(ppt::Vector{Float64})
  tn = Array{Float32,3}(undef, 12, 2, 1); tx = similar(tn); pp = similar(tn)
  for y in 1:2, m in 1:12
    tn[m, y, 1] = TMIN[m]; tx[m, y, 1] = TMAX[m]; pp[m, y, 1] = ppt[m]
  end
  PN.build_climate(["8.5.3.test"], [2000, 2001], tn, tx, pp, [CO2, CO2])
end

SoaT = PC.SiteSoA{Tuple{BaseSite,PN.PnET}}

function make_soa(counts::Vector{Int}; seed=1, lat=33.0, soil="LOAM", year0=2000)
  n = length(counts)
  soa = SoaT((cohort=Int32.(counts), species=fill(Int32(3), n)))
  for i in 1:n
    s = PC.getsite(soa, i)
    s.active = true; s.rng = Random.Xoshiro(UInt64(hash((seed, i)))); s.mapcode = i; s.ecocode = 1; s.eco_id = 1; s.ref_cn = i
    s.old = 0; s.live = 0; s.B = 0
    s.c_species .= 0; s.c_age .= 0; s.c_bio .= 0; s.c_tbio .= 0; s.sp_mature .= false
    PN.init_site!(soa, i; clim_id=1, cal_year0=year0, lat=lat, soil=soil)
  end
  return soa
end

species = ["PITA", "QURU", "TAAS"]
Random.seed!(7)
pnctx = PN.PnETContext(; climate=make_climate(PPT))
params = PN.generate_pnet_params(pnctx, species, ["8.5.3"], [[1, 2, 3]]; rng=Random.Xoshiro(3))
@assert params.PFT == UInt8[1, 2, 4] "PFT assignment: $(params.PFT)"
cst0 = PN.PnETConst()
ep = PN.generate_pnet_eco_params(params, cst0)[1]

# ------------------------------------------------------------------------------------------------------------------
# (c) add_cohort! reproduces the observed reported AGB within 2 g/m2
# ------------------------------------------------------------------------------------------------------------------
@testset "(c) add_cohort! circular solve" begin
  clim = make_climate(PPT)
  for sp in 1:3, agb in (50.0, 500.0, 5000.0, 20000.0)
    soa = make_soa([2])
    s = PC.getsite(soa, 1)
    PN.add_cohort!(s, sp, 30, agb; eco=ep, cst=cst0)
    err = abs(s.c_bio[1] - agb)
    # independent check of "reported" from the stored pools
    fbg = ep.fracbelowg[sp]
    tb = Float64(s.c_tbio[1]); folid = ep.maxfracfol[sp] * exp(-ep.fractwd[sp] * tb) * tb      # IdealFol (stored Fol = IdealFol*(1-TOfol))
    @test isapprox(Float64(s.c_fol[1]), folid * (1 - ep.tofol[sp]); rtol=1e-5, atol=1e-6)
    rep = ((1 - fbg) * tb + folid) * s.c_clp[1]
    @test err <= 2.0
    @test abs(rep - agb) <= 2.0
    @test s.c_tbio[1] > 0 && s.c_bmax[1] == s.c_tbio[1]
  end
  println("(c) done")
end

# (g) unreachable target: reported AGB peaks at finite TotalBiomass (foliage/LAI saturates). An observed AGB above the
#     peak must start the cohort AT the peak with finite pools (was: runaway TB ≈ 8e6 → reported 0 → cohort deleted).
@testset "(g) solve beyond the reported-AGB peak" begin
  clim = make_climate(PPT)
  for sp in 1:3
    tbs = 10.0 .^ range(0, 7; length=4000)
    reps = [PN.reported_of_tb(t, ep, sp) for t in tbs]
    peak = maximum(reps); tbpk = tbs[argmax(reps)]
    n0 = PN.UNREACHABLE_INIT[]
    tb = PN.solve_total_biomass(1.5 * peak, ep, sp)
    @test isfinite(tb) && tb > 0
    @test PN.reported_of_tb(tb, ep, sp) >= 0.995 * peak        # starts at (numerically) the max achievable
    @test abs(log(tb / tbpk)) < 0.05                            # near the grid argmax
    @test PN.UNREACHABLE_INIT[] == n0 + 1                       # counted
    tb2 = PN.solve_total_biomass(0.5 * peak, ep, sp)            # reachable targets unchanged: rising-branch root
    @test abs(PN.reported_of_tb(tb2, ep, sp) - 0.5 * peak) <= 2.0 && tb2 < tbpk
    println("(g) sp $sp: peak reported $(round(peak)) at TB $(round(tbpk)); solve(1.5×peak) → TB $(round(tb))")
  end
end

# ------------------------------------------------------------------------------------------------------------------
# (a) one cohort, constant climate, no precipitation, no events: per-month sublayer-summed fluxes vs an independent
#     transcription of the spec equations (docs/pnet_spec.md sections 3, 4.4, 5.x, 6.1) written here with explicit sublayer arrays.
# ------------------------------------------------------------------------------------------------------------------
mutable struct RefState
  tb::Float64; fol::Float64; nsc::Float64; bmax::Float64; age::Float64; growm::Int; lastlai::Float64; clp::Float64; cgs::Float64
  adjff::Float64; fradsum::Float64; fradn::Float64; water::Float64
end

# spec 3 daylength, transcribed literally (atan forms), hours
function ref_hr(doy, latdeg)
  latrad = latdeg * 2 * pi / 360
  z = 0.39785 * sin(4.868961 + 0.017203 * doy + 0.033446 * sin(6.224111 + 0.017202 * doy))
  decl = abs(z) < 0.7 ? atan(z / sqrt(1 - z^2)) : pi / 2 - atan(sqrt(1 - z^2) / z)
  z2 = -tan(decl) * tan(latrad)
  h = if z2 >= 1
    0.0
  elseif z2 <= -1
    pi
  else
    ta = abs(z2)
    ac = ta < 0.7 ? pi / 2 - atan(ta / sqrt(1 - ta^2)) : atan(sqrt(1 - ta^2) / ta)
    z2 < 0 ? pi - ac : ac
  end
  return decl, 2 * (h * 24) / (2 * pi), h
end

function ref_vp(a, b, c, T) a * exp(b * T / (T + c)) end
function ref_year(st::RefState, sp, ep, soilc, cst, latdeg)
  # soilc = (por, fc, wp, A, B). returns vector of (gpp, folresp, netpsn, transp) per month
  out = NTuple{4,Float64}[]
  IM = 5
  days = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
  doys = (15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349)
  rd = 1000.0 * ep.rd_mult
  for m in 1:12
    tmin, tmax = TMIN[m], TMAX[m]
    tave = (tmin + tmax) / 2; tday = (tmax + tave) / 2
    decl, hr, hsun = ref_hr(doys[m], latdeg)
    dayl = 3600 * hr; nightl = 3600 * (24 - hr)
    es = tday < 0 ? ref_vp(0.61078, 21.87456, 265.5, tday) : ref_vp(0.61078, 17.26939, 237.3, tday)
    em = tmin < 0 ? ref_vp(0.61078, 21.87456, 265.5, tmin) : ref_vp(0.61078, 17.26939, 237.3, tmin)
    vpd = es - em
    # PAR (Bristow-Campbell + FAO-56 Ra)
    phi = latdeg * 2 * pi / 360
    dr = 1 + 0.033 * cos(2 * pi * doys[m] / 365)
    ra = (24 * 60 / pi) * 0.0820 * dr * (hsun * sin(phi) * sin(decl) + cos(phi) * cos(decl) * sin(hsun))
    tau = 0.7 * (1 - exp(-0.004 * (tmax - tmin)^2.4))
    par0 = tau * ra * 1e6 * 0.45 * 4.57 / dayl
    # species-month variables
    topt, tlo, thi = ep.psntopt[sp], ep.psntmin[sp], ep.psntmax[sp]
    ftemp = if tday < tlo || tday > thi
      0.0
    elseif tday <= topt
      tmx = topt + (topt - tlo); max(0, (tmx - tday) * (tday - tlo) / ((tmx - tlo) / 2)^2)
    else
      tmn = topt + (topt - thi); max(0, (thi - tday) * (tday - tmn) / ((thi - tmn) / 2)^2)
    end
    dvpd = max(0, 1 - 0.05 * vpd^2)
    jh2o = 0.239 * vpd / (8314.47 * (tmin + 273))
    basefrac = 0.138071 - 0.0024519 * tave
    q10b = 3.22 - 0.046 * (tave + topt) / 2
    q10(T) = q10b^((T - topt) / 10)
    wtd = min(1, (q10(tday) * dayl + q10(tmin) * nightl) / (dayl + nightl))
    maintf = 0.002 * wtd
    q10fol = q10(tave)
    # January: adjFracFol from last season
    if m == 1
      st.adjff = st.fradn > 0 ? ep.fracfol[sp] + (ep.maxfracfol[sp] - ep.fracfol[sp]) * (st.fradsum / st.fradn)^ep.fracfolshape[sp] : ep.fracfol[sp]
      st.fradsum = 0; st.fradn = 0
    end
    lai = zeros(IM); frad = zeros(IM)
    gpp = 0.0; fr = 0.0; net = 0.0; tr = 0.0
    for i in 0:IM-1
      # maintenance
      maint = (1 / IM) * min(st.nsc, maintf * st.tb * 0.45)
      st.nsc = max(0, st.nsc - maint)
      if m == 12 && i == IM - 1
        fa = exp(-ep.fractwd[sp] * st.bmax)
        nscfrac = st.nsc / (fa * (st.tb + st.fol) * 0.45)
        @assert nscfrac > 0.01 "ref cohort died"
        @assert st.tb >= ep.initbiomass[sp] "ref cohort below InitBiomass"
        sen = round(0.33 * st.tb) * 0.02 + round(0.67 * st.tb) * ep.towood[sp]
        st.tb -= sen; st.bmax = max(st.bmax, st.tb)
        fa = exp(-ep.fractwd[sp] * st.bmax)
        alloc = max(st.nsc - 0.05 * fa * st.tb * 0.45, 0)
        st.tb += alloc / 0.45; st.bmax = max(st.bmax, st.tb); st.nsc = max(st.nsc - alloc, 0); st.age += 1
      end
      if i == 0
        if tmin <= ep.leafonmint[sp]
          if st.growm >= 1
            fa = exp(-ep.fractwd[sp] * st.bmax)
            nf = st.nsc / (fa * (st.tb + st.fol) * 0.45)
            lit = nf <= 0.01 ? st.fol : ep.tofol[sp] * st.fol
            st.fol -= lit
          end
          st.growm = -1
        else
          st.growm = st.growm >= 1 ? st.growm + 1 : 1
        end
        if st.growm >= 1 && st.growm != 2
          fa = exp(-ep.fractwd[sp] * st.bmax)
          ideal = st.adjff * fa * st.tb
          cost = 0.0; tent = 0.0
          if st.growm < 2
            if ideal > st.fol
              cost = max(0, min(st.nsc, 0.45 * (ideal - st.fol))); tent = cost / 0.45
            end
            st.lastlai = 0
          end
          tent > 0.01 && (st.fol += tent)
          st.nsc = max(0, st.nsc - cost)
        end
      end
      # LAI (spec 4.4)
      above = sum(lai[1:i])
      lmax = max(0.01, 25 - above)
      l = (1 / IM) * st.fol / (ep.slwmax[sp] - ep.slwdel[sp] * i * (1 / IM) * st.fol)
      lai[i+1] = (st.fol > 0 && l <= 0) ? lmax / (IM - i) : min(lmax, l)
      layerpar = par0 * exp(-ep.k[sp] * (sum(lai[1:i+1]) - lai[i+1]))
      frad[i+1] = 1 - exp(-layerpar * log(2) / ep.halfsat[sp])
      if st.growm >= 1
        # soil water -> pressure head (Saxton-Rawls), Feddes
        por, fc, wp, A, B = soilc
        ph = st.water > por ? 0.0 : round(A * st.water^(-B) * 0.1019977334, digits=2)
        h1, h2, h3, h4 = ep.h1[sp], ep.h2[sp], ep.h3[sp], ep.h4[sp]
        fw = ph <= h1 ? 0.0 : (ph < min(h1, h2) || ph >= h4) ? 0.0 : ph > h3 ? 1 - (ph - h3) / (h4 - h3) : ph < h2 ? ph / (h2 - h1) - h1 / (h2 - h1) : 1.0
        cica = -0.075 * ep.folN[sp] + 0.875
        ciel = CO2 * cica
        dam = max(0, (ciel - 40) / (ciel + 2 * 40) * (350 + 2 * 40) / (350 - 40))
        V = 8314.47 * (tmin + 273) / 101.3
        jco2 = 0.139 * ((CO2 - ciel) / V) * 1e-6
        amax = dam * (ep.amaxa[sp] + ep.amaxb[sp] * ep.folN[sp])
        bfr = basefrac * amax
        gross_amax = amax * 0.75 + bfr
        refgp = days[m] * (gross_amax * dvpd * dayl * 12) / 1e9
        fage = max(0, 1 - (st.age / ep.longevity[sp])^ep.psnagered[sp])
        gpot = (1 / IM) * ftemp * frad[i+1] * fage * refgp * st.fol
        ptr = 0.0015 * (gpot / (jco2 / jh2o)) * st.clp
        wiltw = (ep.h4[sp] * 9.804139432 / A)^(1 / -B)
        avail = (st.water - wiltw) * rd
        if ptr > avail
          t = max(avail, 0.0)
          g = st.clp > 0 ? (t / 0.0015) * (jco2 / jh2o) / st.clp : 0.0
        else
          g = gpot * fw; t = ptr * fw
        end
        st.water = max(0, st.water - t / rd)
        fres = bfr * q10fol * days[m] * 86400 * 12 / 1e9 * st.fol / IM
        nt = g - fres
        st.nsc = max(0, st.nsc + nt)
        gpp += g; fr += fres; net += nt; tr += t
      end
    end
    # layer post (one cohort, one layer)
    laisum = sum(lai)
    if st.growm >= 1 && laisum > st.lastlai
      st.lastlai = laisum
    end
    st.clp = min(st.lastlai / ep.maxlai[sp], st.cgs)
    if st.growm >= 1
      st.fradsum += sum(frad) / IM; st.fradn += 1
    end
    push!(out, (gpp, fr, net, tr))
  end
  return out
end

@testset "(a) single cohort, constant climate: monthly fluxes vs independent spec transcription" begin
  clim = make_climate(zeros(12))
  cst = PN.PnETConst(precip_events=Int32(0))
  for sp in (1, 2)
    soa = make_soa([2]; lat=33.0)
    s = PC.getsite(soa, 1)
    PN.add_cohort!(s, sp, 30, 6000.0; eco=ep, cst=cst)
    ca = PN.CohArrays(soa.csr)
    # reference state = the plugin's freshly initialised winter state
    st = RefState(ca.tbio[1], ca.fol[1], ca.nsc[1], ca.bmax[1], ca.age[1], Int(ca.growm[1]), ca.lastlai[1], ca.clp[1], ca.cgs[1], ca.adjff[1], 0.0, 0.0, soa.scalar.soil_fc[1])
    soilc = PN.saxton_rawls("LOAM")
    refs = ref_year(st, sp, ep, soilc, cst, 33.0)
    got = NTuple{4,Float64}[]
    cb = (m, ca_, o, n, w) -> push!(got, (Float64(ca_.gpp_m[o+1]), Float64(ca_.folresp_m[o+1]), Float64(ca_.netpsn_m[o+1]), Float64(ca_.transp_m[o+1])))
    scr = PN.new_scratch_pool()[1]
    PN.site_year!(soa, 1, 1, ep, cst, clim, scr, soa.scalar.rng[1]; cb=cb)
    @test length(got) == 12
    worst = 0.0
    for m in 1:12, q in 1:4
      a, b = got[m][q], refs[m][q]
      if abs(b) < 1e-9
        @test abs(a) < 1e-6
      else
        worst = max(worst, abs(a - b) / abs(b))
        @test isapprox(a, b; rtol=2e-4, atol=1e-6)
      end
    end
    ann = sum(x -> x[3], got)
    println("(a) species $sp: max rel err over 12 months x 4 fluxes = $(round(worst; sigdigits=3)); annual NetPsn = $(round(ann; digits=1)) gC/m2; annual transp = $(round(sum(x->x[4], got); digits=1)) mm; final TB plugin=$(round(ca.tbio[1]; digits=2)) ref=$(round(st.tb; digits=2))")
    @test isapprox(Float64(ca.tbio[1]), st.tb; rtol=1e-4)
    @test isapprox(Float64(ca.nsc[1]), st.nsc; rtol=2e-3, atol=1e-3)
    @test isapprox(Float64(ca.fol[1]), st.fol; rtol=1e-4, atol=1e-4)
  end
end

# ------------------------------------------------------------------------------------------------------------------
# (b) annual carbon bookkeeping: d(TB + Fol + NSC/C) = (sum NetPsn - sum MaintResp)/C - foliage senescence - turnover
# ------------------------------------------------------------------------------------------------------------------
@testset "(b) carbon closure over several years (3 cohorts, precipitation events, layered canopy)" begin
  clim = make_climate(PPT)
  cst = PN.PnETConst()
  soa = make_soa([4]; seed=11)
  s = PC.getsite(soa, 1)
  PN.add_cohort!(s, 1, 40, 8000.0; eco=ep, cst=cst)
  PN.add_cohort!(s, 2, 40, 6000.0; eco=ep, cst=cst)
  PN.add_cohort!(s, 2, 15, 2500.0; eco=ep, cst=cst)
  ca = PN.CohArrays(soa.csr)
  scr = PN.new_scratch_pool()[1]
  worst = 0.0
  for yr in 1:2   # climate cache holds 2 calendar years; reuse them
    n = Int(s.live)
    S0 = Dict{Tuple{Int,Int},Float64}()
    for k in 1:n
      S0[(Int(ca.species[k]), Int(round(ca.age[k])) + 1)] = Float64(ca.tbio[k]) + Float64(ca.fol[k]) + Float64(ca.nsc[k]) / cst.cfrac
    end
    ibok = [ca.tbio[k] >= ep.initbiomass[Int(ca.species[k])] for k in 1:n]
    PN.site_year!(soa, 1, yr, ep, cst, clim, scr, soa.scalar.rng[1])
    n_after = Int(s.live)
    n_after < n && println("(b) yr $yr: $(n - n_after) cohort(s) removed at year end (closure is checked on the survivors)")
    @test n_after >= 2
    for k in 1:Int(s.live)
      key = (Int(ca.species[k]), Int(round(ca.age[k])))
      haskey(S0, key) || continue
      S1 = Float64(ca.tbio[k]) + Float64(ca.fol[k]) + Float64(ca.nsc[k]) / cst.cfrac
      dS = S1 - S0[key]
      pred = (Float64(ca.acc_net[k]) - Float64(ca.acc_maint[k])) / cst.cfrac - Float64(ca.acc_folsen[k]) - Float64(ca.acc_turn[k])
      rel = abs(dS - pred) / max(abs(pred), 1.0)
      worst = max(worst, rel)
      println("(b) yr $yr sp $(key[1]) age $(key[2]): dS=$(round(dS; digits=3)) predicted=$(round(pred; digits=3)) (NetPsn/C=$(round(ca.acc_net[k]/cst.cfrac; digits=2)) maint/C=$(round(ca.acc_maint[k]/cst.cfrac; digits=2)) folsen=$(round(ca.acc_folsen[k]; digits=2)) turn=$(round(ca.acc_turn[k]; digits=2)))")
      @test rel < 1e-3
    end
  end
  println("(b) worst relative closure error = $(worst)")
end


# ------------------------------------------------------------------------------------------------------------------
# set_cohort_bio!: reported AGB equals the requested value after proportional rescaling (and a pending cohort just stores it)
# ------------------------------------------------------------------------------------------------------------------
@testset "set_cohort_bio! rescales pools so reported AGB == newbio" begin
  clim = make_climate(PPT)
  soa = make_soa([3])
  s = PC.getsite(soa, 1)
  PN.add_cohort!(s, 2, 25, 3000.0; eco=ep)
  PN.add_cohort!(s, 1, 25, 1200.0)                  # pending (pools not yet initialised)
  tb0, fol0, nsc0 = s.c_tbio[1], s.c_fol[1], s.c_nsc[1]
  PN.set_cohort_bio!(s, 1, 1500.0)
  @test s.c_bio[1] == 1500.0f0
  @test isapprox(s.c_tbio[1], tb0 * 0.5f0; rtol=1e-6) && isapprox(s.c_fol[1], fol0 * 0.5f0; rtol=1e-6) && isapprox(s.c_nsc[1], nsc0 * 0.5f0; rtol=1e-6)
  fbg = ep.fracbelowg[2]
  folid = ep.maxfracfol[2] * exp(-ep.fractwd[2] * tb0) * tb0
  @test isapprox(((1 - fbg) * s.c_tbio[1] + 0.5 * folid) * s.c_clp[1], 1500.0; rtol=2e-3)   # reported formula (with the ideal foliage the reported value is based on) scales exactly; CLP unchanged
  PN.set_cohort_bio!(s, 2, 900.0)                   # pending: store only
  @test s.c_bio[2] == 900.0f0 && s.c_tbio[2] == 0
  # a pending cohort is initialised from its stored AGB at the start of the next year and lands within 2 g/m2 of it
  scr = PN.new_scratch_pool()[1]
  PN.materialize_pending!(PN.CohArrays(soa.csr), Int(soa.refs.cohort[1]) - 1, 2, ep, PN.PnETConst())
  @test s.c_tbio[2] > 0 && abs(s.c_bio[2] - 900.0) <= 2.0
end

# ------------------------------------------------------------------------------------------------------------------
# (d) determinism: same seed => identical state; (e) allocation of a full process_plugin! year
# ------------------------------------------------------------------------------------------------------------------
function build_multi(nsite; seed)
  soa = make_soa(fill(4, nsite); seed=seed)
  rng = Random.Xoshiro(99)
  for i in 1:nsite
    s = PC.getsite(soa, i)
    PN.add_cohort!(s, 1, 35 + i % 7, 4000.0 + 100 * i; eco=ep)
    PN.add_cohort!(s, 2, 30 + i % 5, 3000.0 + 50 * i; eco=ep)
    PN.add_cohort!(s, 2, 12, 900.0 + 10 * i; eco=ep)
  end
  return soa
end
const SCRATCH = PN.new_scratch_pool()    # per-thread working memory (owned by whoever builds the ctx: here the test)
mkctx(clim) = (eco=[ep], cst=PN.PnETConst(), clim=clim, scratch=SCRATCH)

@testset "(d) determinism" begin
  clim = make_climate(PPT)
  A = build_multi(12; seed=5); B = build_multi(12; seed=5); C = build_multi(12; seed=6)
  for t in 1:5
    PC.process_plugin!(A, PN.PnET, t; ctx=mkctx(clim)); PC.process_plugin!(B, PN.PnET, t; ctx=mkctx(clim)); PC.process_plugin!(C, PN.PnET, t; ctx=mkctx(clim))
  end
  same = all(k -> A.csr[k] == B.csr[k], keys(A.csr)) && A.scalar.B == B.scalar.B && A.scalar.soil_water == B.scalar.soil_water && A.refs.cohort == B.refs.cohort
  @test same
  diff = A.scalar.B != C.scalar.B || A.scalar.soil_water != C.scalar.soil_water
  @test diff       # different seeds => different sublayer order / event placement
  println("(d) 5-year state identical for equal seeds: $same ; differs for another seed: $diff ; site 1 B after 5 y = $(A.scalar.B[1])")
end

@testset "(e) allocations of a full process_plugin! year" begin
  clim = make_climate(PPT)
  soa = build_multi(40; seed=5)
  ctx = mkctx(clim)
  old = PC.PARALLEL_SITES[]
  PC.PARALLEL_SITES[] = false
  for t in 1:2; PC.process_plugin!(soa, PN.PnET, t; ctx=ctx); end   # warm-up (compilation + steady CSR capacities)
  @assert all(soa.scalar.live .== 3)
  a_serial = @allocated PC.process_plugin!(soa, PN.PnET, 3; ctx=ctx)
  a_serial2 = @allocated PC.process_plugin!(soa, PN.PnET, 4; ctx=ctx)
  # kernel only (no readjust bookkeeping)
  scr = PN.new_scratch_pool()[1]
  kern(soa, ep, cst, clim, scr, t) = (for i in 1:soa.n; PN.site_year!(soa, i, t, ep, cst, clim, scr, soa.scalar.rng[i]::Random.Xoshiro); end; nothing)
  kern(soa, ep, ctx.cst, ctx.clim, scr, 5)
  a_kernel = @allocated kern(soa, ep, ctx.cst, ctx.clim, scr, 6)
  a_readjust = @allocated PC.readjust_soa!(soa, (cohort=soa.scalar._new_cohort_counts,))
  PC.PARALLEL_SITES[] = true
  for t in 5:6; PC.process_plugin!(soa, PN.PnET, t; ctx=ctx); end
  a_thr = @allocated PC.process_plugin!(soa, PN.PnET, 7; ctx=ctx)
  PC.PARALLEL_SITES[] = old
  println("(e) @allocated serial process_plugin! year (40 sites x 3 cohorts): $a_serial, $a_serial2 bytes; site_year! kernel loop (40 sites): $a_kernel bytes; readjust_soa! (no-change fast path): $a_readjust bytes; threaded ($(Threads.nthreads()) threads): $a_thr bytes")
  @test a_serial == 0 && a_serial2 == 0
  @test a_kernel == 0
end

# ------------------------------------------------------------------------------------------------------------------
# (f) per-plot SOLUS soil (yaml pnet_soil_csv): pressure-head curve from theta33/theta1500, invariants, legacy path untouched
# ------------------------------------------------------------------------------------------------------------------
using DataFrames
@testset "(f) per-plot soil: curve derivation, repairs, legacy path" begin
  K = 0.1019977334     # kPa -> m H2O, same constant as press_head (plugin.jl) / C# CalculateWaterPressure
  # (a) A/B round trip in the plugin's units: tension(theta33) = 33 kPa, tension(theta1500) = 1500 kPa
  for (por, fc, wp) in ((0.45, 0.28, 0.13), (0.38, 0.12, 0.05), (0.52, 0.38, 0.25), (0.40, 0.30, 0.29))
    p, f, w, A, B, nrep = PN.saxton_rawls_theta(por, fc, wp)
    @test p > f > w
    @test A * f^(-B) ≈ 33.0 rtol = 1e-12
    @test A * w^(-B) ≈ 1500.0 rtol = 1e-12
    @test PN.press_head(f, p, A, B) ≈ round(33.0 * K * 100) / 100 atol = 1e-9          # head [m] at field capacity (3.37 m)
    @test PN.press_head(w, p, A, B) ≈ round(1500.0 * K * 100) / 100 atol = 1e-9        # head [m] at wilting point (153.00 m)
    @test PN.press_head(p + 0.01, p, A, B) == 0.0                                       # above porosity -> 0 (C# line 49)
    @test 1.0 - 1e-9 <= B <= 12.0 + 1e-9
    # unchanged inputs <=> no repair
    nrep == 0 && @test (f, w) == (fc, wp)
  end
  # consistency with the texture path: feeding the texture's own (por, fc, wp) reproduces its A, B
  por0, fc0, wp0, A0, B0 = PN.saxton_rawls("LOAM")
  _, _, _, A1, B1, nr = PN.saxton_rawls_theta(por0, fc0, wp0)
  @test nr == 0
  @test A1 ≈ A0 rtol = 1e-9
  @test B1 ≈ B0 rtol = 1e-9
  # (b) repairs: invariants wiltpnt < fieldcap < porosity always hold
  for (por, fc, wp) in ((0.30, 0.30, 0.30), (0.25, 0.35, 0.40), (0.4, 0.2, 0.0), (0.4, 0.2, 0.199), (0.5, 0.45, 0.005))
    p, f, w, A, B, nrep = PN.saxton_rawls_theta(por, fc, wp)
    @test w < f < p
    @test nrep == 1
    @test isfinite(A) && A > 0
    @test 1.0 - 1e-9 <= B <= 12.0 + 1e-9
    @test Float32(A) > 0
    @test isfinite(Float32(A))
  end

  # (c) init_sites!: no soil columns => identical to the legacy texture path; with columns => per-plot values + depth, texture fallback for NaN
  clim = make_climate(PPT)
  mk() = make_soa([3, 3, 3])
  attrs0 = DataFrame(plot_id=1:3, clim_id=Int32[1, 1, 1], cal_year0=Int16[2000, 2000, 2000], lat=Float32[33, 33, 33], soil=["LOAM", "SAND", "LOAM"])
  a = mk(); PN.init_sites!(a, attrs0)
  b = mk(); for i in 1:3; PN.init_site!(b, i; clim_id=1, cal_year0=2000, lat=33.0, soil=attrs0.soil[i]); end
  @test all(k -> getfield(a.scalar, k) == getfield(b.scalar, k), (:soil_por, :soil_fc, :soil_wp, :soil_A, :soil_B, :rooting_depth, :soil_water))
  @test all(a.scalar.rooting_depth .== Float32(PN.PnETConst().root_depth_base))
  attrs1 = copy(attrs0)
  attrs1.soil_por = [0.45, NaN, 0.30]; attrs1.soil_fc = [0.28, NaN, 0.30]; attrs1.soil_wp = [0.13, NaN, 0.30]; attrs1.soil_rd = [650.0, NaN, 1500.0]
  c = mk(); PN.init_sites!(c, attrs1)
  @test c.scalar.soil_fc[1] == Float32(0.28) && c.scalar.soil_wp[1] == Float32(0.13) && c.scalar.soil_por[1] == Float32(0.45)
  @test c.scalar.rooting_depth[1] == 650f0
  @test c.scalar.rooting_depth[3] == 1500f0
  @test c.scalar.soil_water[1] == c.scalar.soil_fc[1]
  @test all(k -> getfield(c.scalar, k)[2] == getfield(b.scalar, k)[2], (:soil_por, :soil_fc, :soil_wp, :soil_A, :soil_B, :rooting_depth))   # NaN row -> texture path (SAND)
  @test all(c.scalar.soil_wp .< c.scalar.soil_fc .< c.scalar.soil_por)
  # effective rooting depth = plot depth * RD_MULT (semantics of the 1000 mm constant preserved)
  @test Float64(c.scalar.rooting_depth[1]) * ep.rd_mult == 650.0 * ep.rd_mult
  # the model runs on per-plot soil and stays finite
  d = mk(); PN.init_sites!(d, attrs1)
  for i in 1:3
    s = PC.getsite(d, i)
    PN.add_cohort!(s, 1, 35 + i, 4000.0 + 100 * i; eco=ep); PN.add_cohort!(s, 2, 30 + i, 3000.0 + 50 * i; eco=ep); PN.add_cohort!(s, 2, 12, 900.0 + 10 * i; eco=ep)
  end
  for t in 1:3; PC.process_plugin!(d, PN.PnET, t; ctx=mkctx(clim)); end
  @test all(isfinite, d.scalar.soil_water)
  @test all(d.scalar.soil_water .>= 0)
  @test all(isfinite, d.scalar.B)
  println("(f) per-plot soil: A/B round trip OK; repairs OK; legacy path identical when pnet_soil_csv unset")
end
