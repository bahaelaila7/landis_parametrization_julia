# Shared machinery of the HBAD2 parity gate (tools/hbad2_parity.jl, test/unit_hbad2_parity.jl): run the PROTOTYPE kernel (module `PR` = branch hbad-v2's HBAD2.jl) and the Pan plugin
# on identical initial cohort arrays and compare the state of every cohort of every site after every year.
#   init = (off, sp, N0, D0, H0, A0, zq0, live0, planted)   off: CSR over sites WITH capacity (prototype layout); live0: live cohorts per site; sp: key index (= eco-local species index)
using Pan, Random
const HB2 = Pan.Plugins.HBAD2Plugin

"Prototype parameters (module PR) from the three csv files of a parameter directory."
function proto_params(PR, dir)
  t = HB2.load_v2_tables(dir)
  P = [PR.SpP(Float64[getfield(s, f) for f in HB2.SP_PARAM_NAMES]) for s in t.sp]
  g = t.glob
  G = PR.GlobP(g.mD2, g.mA, g.pH, g.pD, g.pM, g.form, g.mSD, g.σv)
  R = [PR.RecP(r.rho0, r.rhoS, r.rhoL, r.D0, r.H0, r.A0) for r in t.rec]
  return P, G, R, t.keys
end

"A recorded state: per year, per site, the live cohorts (sp, N, D, H, A, zq)."
const Rec = Vector{Vector{NTuple{6,Vector{Float64}}}}   # Rec[y][s]

_grab(sp, N, D, H, A, zq, lo, hi) = (Float64.(sp[lo:hi]), N[lo:hi], D[lo:hi], H[lo:hi], A[lo:hi], zq[lo:hi])

"Prototype free run (recruitment) of `init` for `years` years; returns the recorded state after every year."
function proto_free(PR, init, P, G, R, years::Int)
  ns = length(init.live0); nsp = length(P)
  S = PR.Sites(copy(init.off), Int32.(init.sp), copy(init.N0), copy(init.D0), copy(init.H0), copy(init.A0), zeros(Int, ns), zeros(ns), copy(init.planted), copy(init.zq0))
  F = PR.FreeSites(S, copy(init.live0), nsp; vigour=true)
  rec = [Vector{NTuple{6,Vector{Float64}}}(undef, ns) for _ in 1:years]
  snap = (s, y) -> (lo = S.off[s]; rec[y][s] = _grab(S.sp, S.N, S.D, S.H, S.A, S.zq, lo, lo + F.live[s] - 1))
  PR.run_free!(F, P, G, R, fill(years, ns), snap)
  return rec, sum(F.dropped)
end

"Prototype Sim A run (no recruitment): `nyr[s]` whole annual steps of `PR.step_site!` per site, state recorded after every year (rec[y][s] only for y <= nyr[s])."
function proto_simA(PR, init, P, G, nyr::Vector{Int})
  ns = length(init.live0)
  S = PR.Sites(copy(init.off), Int32.(init.sp), copy(init.N0), copy(init.D0), copy(init.H0), copy(init.A0), zeros(Int, ns), zeros(ns), copy(init.planted), copy(init.zq0))
  years = maximum(nyr)
  rec = [Vector{NTuple{6,Vector{Float64}}}(undef, ns) for _ in 1:years]
  for s in 1:ns
    lo = S.off[s]; hi = lo + init.live0[s] - 1
    for y in 1:nyr[s]
      hi >= lo && PR.step_site!(S, P, G, lo, hi, 1.0, S.planted[s])
      rec[y][s] = _grab(S.sp, S.N, S.D, S.H, S.A, S.zq, lo, hi)
    end
  end
  return rec
end

"Pan SoA of `init` (HBAD2 plugin config `pcfg`; species of the context = the prototype's keys, one ecoregion)."
function pan_soa(pcfg, init; no_establishment::Bool)
  live = init.live0
  # CSR over the LIVE cohorts only (Pan grows the capacity itself); cohort c of site s is init index off[s]+c-1
  off = ones(Int, length(live) + 1); for s in eachindex(live); off[s+1] = off[s] + live[s]; end
  idx = reduce(vcat, [collect(init.off[s]:init.off[s]+live[s]-1) for s in eachindex(live)]; init=Int[])
  return Pan.make_sites_hbad2_arrays(pcfg, off, init.sp[idx], init.A0[idx], init.N0[idx], init.D0[idx], init.H0[idx], init.zq0[idx];
                                     rng=Random.Xoshiro(1), no_establishment=no_establishment, planted=init.planted)
end

"Run the Pan plugin `years` annual steps; returns the recorded state after every year (`years_per_site`: nothing = all years for every site)."
function pan_run(pcfg, soa, years::Int; nyr=nothing)
  params = Pan.generate_params(pcfg.ctx, pcfg.ctx.priors.species_list, ["eco"], [collect(1:length(pcfg.ctx.priors.species_list))])
  ctx = Pan.make_ctx(pcfg.ctx, Pan.generate_eco_params(pcfg.ctx, params))
  SP = Pan.succession_plugin_of(soa)
  ns = soa.n
  rec = [Vector{NTuple{6,Vector{Float64}}}(undef, ns) for _ in 1:years]
  for y in 1:years
    Pan.step!(SP, soa, y, ctx)
    for s in 1:ns
      (nyr !== nothing && y > nyr[s]) && continue
      o = Int(soa.refs.cohort[s]) - 1; n = Int(soa.scalar.live[s])
      rec[y][s] = _grab(soa.csr.c_species, soa.csr.c_tph, soa.csr.c_dq, soa.csr.c_h, soa.csr.c_age, soa.csr.c_zq, o + 1, o + n)
    end
  end
  return rec
end

@inline _rel(a, b) = a == b ? 0.0 : abs(a - b) / max(abs(a), abs(b))

"Compare two recordings cohort by cohort: (max rel diff N, D, H, A; sites*years compared; cohorts compared; count of structural mismatches: live count, species or vigour differ)."
function compare_rec(a, b; nyr=nothing)
  mx = zeros(4); nsy = 0; nc = 0; bad = 0
  for y in eachindex(a), s in eachindex(a[y])
    (nyr !== nothing && y > nyr[s]) && continue
    x = a[y][s]; z = b[y][s]; nsy += 1
    if length(x[1]) != length(z[1]) || x[1] != z[1] || x[6] != z[6]
      bad += 1; continue
    end
    for c in eachindex(x[1]), k in 1:4
      mx[k] = max(mx[k], _rel(x[k+1][c], z[k+1][c]))
    end
    nc += length(x[1])
  end
  return (maxrel_N=mx[1], maxrel_D=mx[2], maxrel_H=mx[3], maxrel_A=mx[4], site_years=nsy, cohorts=nc, mismatches=bad, maxrel=maximum(mx))
end

"Synthetic initial state (prototype layout with capacity `cap_extra`): `ns` sites with 0..`nmax` random cohorts (vigour-expanded triplets), some planted."
function synth_init(rng, ns::Int, nsp::Int; nmax::Int=8, cap_extra::Int=200, planted_frac::Float64=0.2)
  off = [1]; sp = Int[]; N0 = Float64[]; D0 = Float64[]; H0 = Float64[]; A0 = Float64[]; zq0 = Float64[]; live = Int[]
  VZ = (-sqrt(3.0), 0.0, sqrt(3.0)); VW = (1 / 6, 2 / 3, 1 / 6)
  for s in 1:ns
    nc = s % 7 == 0 ? 0 : rand(rng, 1:nmax); n = 0
    for _ in 1:nc
      k = rand(rng, 1:nsp); N = 10 + 900 * rand(rng); D = 2.6 + 50 * rand(rng)^2; H = 3 + 0.9 * D^0.8 * (0.8 + 0.4 * rand(rng)); A = 1 + 120 * rand(rng)
      for q in 1:3; push!(sp, k); push!(N0, N * VW[q]); push!(D0, D); push!(H0, H); push!(A0, A); push!(zq0, VZ[q]); n += 1; end
    end
    push!(live, n)
    for _ in 1:cap_extra; push!(sp, 1); push!(N0, 0.0); push!(D0, 2.54); push!(H0, 3.0); push!(A0, 5.0); push!(zq0, 0.0); end
    push!(off, length(sp) + 1)
  end
  (off=off, sp=sp, N0=N0, D0=D0, H0=H0, A0=A0, zq0=zq0, live0=live, planted=[rand(rng) < planted_frac for _ in 1:ns])
end
