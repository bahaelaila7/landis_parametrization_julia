# Parity gate of the HBAD3P plugin against the REFERENCE kernel of the prototype tree (landis_hbad3/src/plugins/HBAD3/HBAD3P.jl, included from a path relative to this worktree's parent folder;
# the test is skipped with a warning when the prototype tree is absent):
#   (a) the annual/partial-year step: 300 random sites (ties in H, planted stands, partial years, RD > 1 so the self-thinning loop acts, perturbed globals and species traits) -> N, D, H, A, CR;
#   (b) recruitment: 20-year free runs on random sites (class merging, seed presence, rmult incl. 0, ample capacity and a tight capacity that drops classes), cohort by cohort after every year;
#   (c) the same free run through the plugin's own process_plugin! (annual step, CSR growth) on ample capacity.
module UnitHBAD3PParity
using Test, Random, Pan
const H3 = Pan.Plugins.HBAD3PPlugin
const REF_PATH = normpath(joinpath(@__DIR__, "..", "..", "landis_hbad3", "src", "plugins", "HBAD3", "HBAD3P.jl"))

const KEYS = ["ACRU", "LIST2", "LITU", "MAVI2", "NYBI", "PIEL", "PITA", "QULA3", "QUNI", "TAAS", "_H", "_S"]
const NSP = length(KEYS)
@inline _rel(a, b) = a == b ? 0.0 : abs(a - b) / max(abs(a), abs(b), 1e-300)

const RefM = Module(:RefHBAD3P)
isfile(REF_PATH) && Base.include(RefM, REF_PATH)      # (a separate top-level statement: the module's bindings are then visible to the code below)

if !isfile(REF_PATH)
  @warn "unit_hbad3p_parity: the reference kernel $(REF_PATH) is absent; parity tests skipped"
else
  const R = RefM.HBAD3P

  # real fitted parameters (results/hbad3p/P3_GAFL) as plugin objects and as reference objects
  const T = H3.load_3p_tables()
  ref_sp(p) = R.SpPP(p.lnε, p.Hmax, p.ST, p.longevity, p.lnF)
  ref_glob(g) = R.GlobPP((getfield(g, f) for f in H3.GLOB_NAMES)...)
  ref_rec(r) = R.RecGP(r.λ0, r.λ1, r.ε0)
  pcfg = Pan.PluginConfig(H3.HBAD3PContext(KEYS, [collect(1:NSP)]; verbose=false))

  "Perturbed globals and traits (keeps sdimax and mA): exercises every branch of the kernel."
  function perturbed(rng, G::H3.GlobPP, sp::Vector{H3.SpPP})
    G2 = H3.GlobPP(Float64[f in (:sdimax, :mA) ? getfield(G, f) : getfield(G, f) * (1 + 0.3randn(rng)) for f in H3.GLOB_NAMES])
    sp2 = [H3.SpPP(p.lnε + 0.2randn(rng), p.Hmax * (1 + 0.1randn(rng)), p.ST, p.longevity, p.lnF) for p in sp]
    G2, sp2
  end

  function step_parity(; nsites=300, seed=17)
    rng = Random.Xoshiro(seed)
    off = [1]; sp = Int[]; N = Float64[]; D = Float64[]; H = Float64[]; A = Float64[]; zq = Float64[]; CR = Float64[]
    pl = Bool[]; W = Float64[]; Wn = Float64[]; nyr = Int[]; frac = Float64[]; Gs = H3.GlobPP[]; Ps = Vector{H3.SpPP}[]
    nover = 0; nties = 0
    vz = H3.VIG_Z
    for s in 1:nsites
      n = rand(rng, 1:9)
      dens = rand(rng) < 0.35 ? 4000.0 : 700.0
      for i in 1:n
        push!(sp, rand(rng, 1:NSP)); push!(D, 2.54 + 60rand(rng)); push!(N, dens * rand(rng) + 1); push!(A, 1 + 200rand(rng)); push!(CR, 0.05 + 0.9rand(rng)); push!(zq, rand(rng, vz))
        push!(H, 1.4 + 35rand(rng))
        if i > 1 && rand(rng) < 0.3; H[end] = H[end-1]; nties += 1; end         # ties in height
      end
      push!(off, length(sp) + 1)
      push!(pl, rand(rng, Bool)); push!(W, 0.3 + 0.7rand(rng)); push!(Wn, 0.3 + 0.7rand(rng))
      push!(nyr, rand(rng, 0:3)); push!(frac, rand(rng) < 0.5 ? rand(rng) : 0.0)
      nyr[end] == 0 && frac[end] == 0 && (nyr[end] = 1)
      g, p = perturbed(rng, T.glob, T.sp); push!(Gs, g); push!(Ps, p)
      r = off[end-1]:off[end]-1
      nover += sum(N[i] * (D[i] / 25.4)^1.605 for i in r) > T.glob.sdimax
    end
    # one SoA for all sites; the parameters differ per site -> run each site with its own EcoParams
    soa = Pan.make_sites_hbad3p_arrays(pcfg, off, sp, A, N, D, H, zq, CR; rng=Random.Xoshiro(1), no_establishment=true, planted=pl, W=W, Wn=Wn)
    allo = [H3.Allo(0.0, 0.0) for _ in 1:NSP]
    mx = zeros(5); nc = 0
    for s in 1:nsites
      ep = H3.HBAD3PEcoParams(Ps[s], allo, ref_rec_dummy(), Gs[s])
      H3.run_interval!(soa, s, ep, nyr[s], frac[s])
      # reference on this site alone
      lo = off[s]; hi = off[s+1] - 1; n = hi - lo + 1
      S = R.SitesP([1, n + 1], sp[lo:hi], N[lo:hi], D[lo:hi], H[lo:hi], A[lo:hi], [nyr[s]], [frac[s]], [pl[s]], zq[lo:hi], [W[s]], [Wn[s]], CR[lo:hi])
      R.simulate!(S, ref_sp.(Ps[s]), ref_glob(Gs[s]))
      o = Int(soa.refs.cohort[s]) - 1
      for i in 1:n
        mx[1] = max(mx[1], _rel(soa.csr.c_tph[o+i], S.N[i])); mx[2] = max(mx[2], _rel(soa.csr.c_dq[o+i], S.D[i])); mx[3] = max(mx[3], _rel(soa.csr.c_h[o+i], S.H[i]))
        mx[4] = max(mx[4], _rel(soa.csr.c_age[o+i], S.A[i])); mx[5] = max(mx[5], _rel(soa.csr.c_cr[o+i], S.CR[i]))
      end
      nc += n
    end
    (maxrel=maximum(mx), N=mx[1], D=mx[2], H=mx[3], A=mx[4], CR=mx[5], sites=nsites, cohorts=nc, over_sdimax=nover, ties=nties, planted=count(pl), partial=count(>(0), frac))
  end
  ref_rec_dummy() = H3.RecGP(0.0, 0.0, 0.0)

  "Random free-run sites: live cohorts (vigour nodes), seed presence, rmult, W/Wn; capacity = live + extra."
  function free_init(rng, ns; extra=400)
    off = [1]; sp = Int[]; N = Float64[]; D = Float64[]; H = Float64[]; A = Float64[]; zq = Float64[]; CR = Float64[]; live = Int[]
    for s in 1:ns
      nc = s % 6 == 0 ? 0 : rand(rng, 1:4); n = 0
      for _ in 1:nc
        k = rand(rng, 1:NSP); Nc = 20 + 500 * rand(rng); Dc = 2.6 + 45 * rand(rng)^2; Hc = 3 + 0.9 * Dc^0.8; Ac = 1 + 100 * rand(rng); Cc = 0.1 + 0.8rand(rng)
        for q in 1:5; push!(sp, k); push!(N, Nc * H3.VIG_W[q]); push!(D, Dc); push!(H, Hc); push!(A, Ac); push!(zq, H3.VIG_Z[q]); push!(CR, Cc); n += 1; end
      end
      push!(live, n); push!(off, length(sp) + 1)
    end
    seed = BitMatrix(rand(rng, Bool, NSP, ns) .& (rand(rng, NSP, ns) .< 0.4))
    rmult = [rand(rng, (0.0, 0.3, 1.0, 1.0, 2.0)) for _ in 1:ns]
    W = 0.5 .+ 0.5 .* rand(rng, ns); Wn = 0.5 .+ 0.5 .* rand(rng, ns); pl = [rand(rng) < 0.2 for _ in 1:ns]
    (off=off, sp=sp, N=N, D=D, H=H, A=A, zq=zq, CR=CR, live=live, seed=seed, rmult=rmult, W=W, Wn=Wn, pl=pl, extra=extra)
  end

  "Reference free run (recorded after every year) and the plugin's `site_year_free!` on the same initial state; returns the comparison."
  function free_parity(; ns=40, years=20, extra=400, frac_last=0.0, seed=5)
    rng = Random.Xoshiro(seed); I = free_init(rng, ns; extra=extra)
    ep = H3.identity_eco_params(pcfg.ctx)
    # reference store (capacity live + extra per site)
    off = [1]; for s in 1:ns; push!(off, off[end] + I.live[s] + extra); end
    n = off[end] - 1
    spv = ones(Int32, n); Nv = zeros(n); Dv = fill(2.54, n); Hv = fill(1.37, n); Av = zeros(n); zv = zeros(n); Cv = fill(0.4, n)
    for s in 1:ns
      r = I.off[s]:I.off[s+1]-1; d = off[s]:off[s]+length(r)-1
      spv[d] .= I.sp[r]; Nv[d] .= I.N[r]; Dv[d] .= I.D[r]; Hv[d] .= I.H[r]; Av[d] .= I.A[r]; zv[d] .= I.zq[r]; Cv[d] .= I.CR[r]
    end
    R.set_quadrature!(collect(H3.VIG_Z), collect(H3.VIG_W))
    Pr = [R.SpPP(p.lnε, p.Hmax, p.ST, p.longevity, p.lnF) for p in ep.sp]; Gr = ref_glob(ep.G); Rr = ref_rec(ep.rec)
    S = R.SitesP(off, spv, copy(Nv), Dv, Hv, Av, zeros(Int, ns), zeros(ns), copy(I.pl), zv, copy(I.W), copy(I.Wn), Cv)
    copyto!(S.N, Nv)
    F = R.FreeSitesP(S, copy(I.live), zeros(Int, NSP, ns), zeros(Int, NSP, ns), zeros(Int, ns), I.seed, copy(I.rmult))
    ny = years + (frac_last > 0)
    recR = [Vector{NTuple{7,Vector{Float64}}}(undef, ns) for _ in 1:ny]
    snap = (s, y) -> (lo = S.off[s]; r = lo:lo+F.live[s]-1; recR[y][s] = (Float64.(S.sp[r]), S.N[r], S.D[r], S.H[r], S.A[r], S.zq[r], S.CR[r]))
    R.run_free!(F, Pr, Gr, Rr, fill(years, ns), snap; frac=fill(frac_last, ns))
    # plugin
    soa = Pan.make_sites_hbad3p_arrays(pcfg, I.off, I.sp, I.A, I.N, I.D, I.H, I.zq, I.CR; rng=Random.Xoshiro(1), no_establishment=false, planted=I.pl, W=I.W, Wn=I.Wn,
                                       rmult=I.rmult, seedplot=I.seed, extra_capacity=extra)
    recP = [Vector{NTuple{7,Vector{Float64}}}(undef, ns) for _ in 1:ny]; dropped = zeros(Int, ns)
    for s in 1:ns, y in 1:ny
      Δ = y <= years ? 1.0 : frac_last
      dropped[s] += H3.site_year_free!(soa, s, ep, 5, Δ)
      o = Int(soa.refs.cohort[s]) - 1; k = Int(soa.scalar.live[s]); c = soa.csr; r = o+1:o+k
      recP[y][s] = (Float64.(c.c_species[r]), c.c_tph[r], c.c_dq[r], c.c_h[r], c.c_age[r], c.c_zq[r], c.c_cr[r])
    end
    (cmp=compare(recR, recP), dropped_ref=sum(F.dropped), dropped_pan=sum(dropped), same_dropped=F.dropped == dropped, final_cohorts=sum(F.live))
  end

  function compare(a, b)
    mx = zeros(6); nsy = 0; nc = 0; bad = 0
    for y in eachindex(a), s in eachindex(a[y])
      x = a[y][s]; z = b[y][s]; nsy += 1
      if length(x[1]) != length(z[1]) || x[1] != z[1] || x[6] != z[6]
        bad += 1; continue
      end
      for c in eachindex(x[1]), k in 1:5
        mx[k] = max(mx[k], _rel(x[k+1][c], z[k+1][c]))
      end
      mx[6] = max(mx[6], _rel(x[7][1:end] == z[7][1:end] ? 0.0 : 1.0, 0.0))
      for c in eachindex(x[7]); mx[6] = max(mx[6], _rel(x[7][c], z[7][c])); end
      nc += length(x[1])
    end
    (maxrel=maximum(mx), N=mx[1], D=mx[2], H=mx[3], A=mx[4], zq=mx[5], CR=mx[6], site_years=nsy, cohorts=nc, mismatches=bad)
  end

  "The same free run through the plugin's own annual step (process_plugin!: kernel + recruitment, CSR growth) vs the reference with ample capacity."
  function process_parity(; ns=40, years=20, seed=5)
    rng = Random.Xoshiro(seed); I = free_init(rng, ns; extra=0)
    ep = H3.identity_eco_params(pcfg.ctx)
    off = [1]; for s in 1:ns; push!(off, off[end] + I.live[s] + 600); end
    n = off[end] - 1
    spv = ones(Int32, n); Nv = zeros(n); Dv = fill(2.54, n); Hv = fill(1.37, n); Av = zeros(n); zv = zeros(n); Cv = fill(0.4, n)
    for s in 1:ns
      r = I.off[s]:I.off[s+1]-1; d = off[s]:off[s]+length(r)-1
      spv[d] .= I.sp[r]; Nv[d] .= I.N[r]; Dv[d] .= I.D[r]; Hv[d] .= I.H[r]; Av[d] .= I.A[r]; zv[d] .= I.zq[r]; Cv[d] .= I.CR[r]
    end
    R.set_quadrature!(collect(H3.VIG_Z), collect(H3.VIG_W))
    Pr = [R.SpPP(p.lnε, p.Hmax, p.ST, p.longevity, p.lnF) for p in ep.sp]; Gr = ref_glob(ep.G); Rr = ref_rec(ep.rec)
    S = R.SitesP(off, spv, copy(Nv), Dv, Hv, Av, zeros(Int, ns), zeros(ns), copy(I.pl), zv, copy(I.W), copy(I.Wn), Cv); copyto!(S.N, Nv)
    F = R.FreeSitesP(S, copy(I.live), zeros(Int, NSP, ns), zeros(Int, NSP, ns), zeros(Int, ns), I.seed, copy(I.rmult))
    recR = [Vector{NTuple{7,Vector{Float64}}}(undef, ns) for _ in 1:years]
    snap = (s, y) -> (lo = S.off[s]; r = lo:lo+F.live[s]-1; recR[y][s] = (Float64.(S.sp[r]), S.N[r], S.D[r], S.H[r], S.A[r], S.zq[r], S.CR[r]))
    R.run_free!(F, Pr, Gr, Rr, fill(years, ns), snap)
    soa = Pan.make_sites_hbad3p_arrays(pcfg, I.off, I.sp, I.A, I.N, I.D, I.H, I.zq, I.CR; rng=Random.Xoshiro(1), no_establishment=false, planted=I.pl, W=I.W, Wn=I.Wn, rmult=I.rmult, seedplot=I.seed)
    params = Pan.generate_params(pcfg.ctx, KEYS, ["eco"], [collect(1:NSP)])
    ctx = Pan.make_ctx(pcfg.ctx, Pan.generate_eco_params(pcfg.ctx, params))
    SP = Pan.succession_plugin_of(soa)
    recP = [Vector{NTuple{7,Vector{Float64}}}(undef, ns) for _ in 1:years]
    for y in 1:years
      Pan.step!(SP, soa, y, ctx)
      for s in 1:ns
        o = Int(soa.refs.cohort[s]) - 1; k = Int(soa.scalar.live[s]); c = soa.csr; r = o+1:o+k
        recP[y][s] = (Float64.(c.c_species[r]), c.c_tph[r], c.c_dq[r], c.c_h[r], c.c_age[r], c.c_zq[r], c.c_cr[r])
      end
    end
    (cmp=compare(recR, recP), dropped_ref=sum(F.dropped))
  end

  @testset "hbad3p parity vs the reference kernel" begin
    @test H3.VIG_Z == (-2.856970013872806, -1.355626179974266, 0.0, 1.355626179974266, 2.856970013872806)
    @testset "annual / partial-year step (300 random sites)" begin
      r = step_parity()
      @info "step parity" r
      @test r.sites >= 200 && r.over_sdimax > 10 && r.ties > 50 && r.planted > 50 && r.partial > 50
      @test r.maxrel <= 1e-12
    end
    @testset "recruitment: free run, ample capacity" begin
      r = free_parity()
      @info "free-run parity (ample capacity)" r
      @test r.dropped_ref == 0 && r.dropped_pan == 0
      @test r.cmp.mismatches == 0 && r.cmp.maxrel <= 1e-12 && r.cmp.cohorts > 5_000
    end
    @testset "recruitment: tight capacity (dropped classes), partial last year" begin
      r = free_parity(; extra=45, frac_last=0.37, seed=9)
      @info "free-run parity (tight capacity)" r
      @test r.dropped_ref > 0 && r.same_dropped
      @test r.cmp.mismatches == 0 && r.cmp.maxrel <= 1e-12
    end
    @testset "recruitment through process_plugin! (annual step, CSR growth)" begin
      r = process_parity()
      @info "process_plugin! parity" r
      @test r.dropped_ref == 0 && r.cmp.mismatches == 0 && r.cmp.maxrel <= 1e-12 && r.cmp.cohorts > 5_000
    end
  end
end
end
