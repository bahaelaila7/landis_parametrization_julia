# HBAD4P parameter space (fit4p / eval4p / verify4p): per key ln ε (growth efficiency) and ln F (fecundity); globals (process + observation model).
# Encoding: log-encoded where marked, clamped to the bounds. Fixed by default (HBAD4P_FIX overrides / adds, e.g. "gR:0.3,wT:Inf"):
#   gR = 0 (HBAD3P's crowding gain e^{gR RD} grew with density, a site-quality proxy, docs/hbad3_design.md §8.3), mA = 10, sdimax = 1500 (as HBAD3P),
#   βC = HBAD4P_BETAC (0.58: FACE, +23 % NPP at 550 vs 370 ppm; not identifiable from 2000–2024 FIA).
using CSV, DataFrames
const GSPEC4 = [(:b, 1.75, 1.0, 3.0, false), (:a, 2.35, 0.1, 10.0, true), (:gR, 0.0, -2.0, 1.0, false), (:θ, 8.7, 1.0, 40.0, true), (:ω, 0.96, 0.0, 5.0, false),
                (:s, 1.16, 0.2, 3.0, true), (:c, 1.25, 0.3, 2.0, true), (:η, 0.69, -1.0, 2.0, false), (:λ, 0.0023, 1e-4, 0.3, true), (:μ, 1.22, -0.5, 2.0, false),
                (:kh, 5.1, 0.2, 30.0, true), (:m0, 1.11, 0.01, 10.0, true), (:ms, 0.082, 1e-4, 1.0, true), (:σST, 0.28, -2.0, 2.0, false), (:β, 2.95, 0.1, 20.0, true),
                (:ζ, -0.62, -3.0, 2.0, false), (:mW, 0.12, -2.0, 3.0, false), (:mA, 10.0, 1.0, 50.0, true), (:pD, 0.11, -1.0, 1.0, false), (:pH, 0.26, -1.0, 1.0, false),
                (:pM, -0.06, -2.0, 2.0, false), (:σv, 0.36, 0.0, 1.2, false), (:sdimax, 1500.0, 500.0, 3000.0, true), (:am, 0.83, 0.1, 20.0, true),
                (:ρ, 2.6e-4, 0.0, 0.01, false), (:f0, 0.0, 0.0, 0.5, false), (:wT, 8.0, 1.0, 100.0, true), (:βC, 0.58, 0.0, 2.0, false), (:κ, 0.1, 1e-3, 1.0, true),
                (:λ0, 3.1, 0.01, 20.0, true), (:λ1, -0.07, -2.0, 2.0, false), (:ε0, 0.3, 1e-3, 1.0, true), (:ωR, 0.0, -3.0, 5.0, false),
                (:rM, 1.0, 0.05, 100.0, true), (:rS, 2.0, 0.05, 100.0, true), (:σD, 0.05, 0.005, 0.5, true), (:σH, 0.15, 0.01, 1.0, true)]
const GN4 = [g[1] for g in GSPEC4]
const OBS4 = (:rM, :rS, :σD, :σH)
const FIX_DEFAULT = Dict(:gR => 0.0, :mA => 10.0, :sdimax => 1500.0, :βC => parse(Float64, get(ENV, "HBAD4P_BETAC", "0.58")))
function gfix4()
  f = copy(FIX_DEFAULT)
  for x in filter(!isempty, strip.(split(get(ENV, "HBAD4P_FIX", ""), ",")))
    a_, b_ = split(x, ":"); f[Symbol(a_)] = parse(Float64, b_)
  end
  for x in filter(!isempty, strip.(split(get(ENV, "HBAD4P_FREE", ""), ",")))   # release a default-fixed global, e.g. HBAD4P_FREE=gR
    delete!(f, Symbol(x))
  end
  f
end
genc4(j, v) = GSPEC4[j][5] ? log(v) : v
function gdec4(j, z); lo, hi = GSPEC4[j][3], GSPEC4[j][4]; clamp(GSPEC4[j][5] ? exp(z) : (GSPEC4[j][1] == :σv ? abs(z) : z), lo, hi); end

"Coordinate layout for keys_ and the fixed globals: names, start vector, decoder z → (P, G, Ob)."
function space4(keys_, TR; start_eps=nothing, start_lnF=nothing, start_glob=Dict{Symbol,Float64}())
  nk = length(keys_); fix = gfix4()
  gfree = [j for j in eachindex(GSPEC4) if !haskey(fix, GSPEC4[j][1])]
  gval(j) = (nm = GSPEC4[j][1]; haskey(fix, nm) ? fix[nm] : get(start_glob, nm, GSPEC4[j][2]))
  gv0 = [genc4(j, clamp(gval(j), GSPEC4[j][3], GSPEC4[j][4])) for j in eachindex(GSPEC4)]
  for (nm, v) in fix; j = findfirst(==(nm), GN4); isinf(v) && (gv0[j] = v); end        # (Inf: e.g. wT = Inf switches the temperature response off)
  le = start_eps === nothing ? fill(log(2.0), nk) : log.(start_eps); lf = start_lnF === nothing ? fill(-3.0, nk) : start_lnF
  z0 = vcat(le, lf, gv0[gfree])
  names = vcat(["ε:" * k for k in keys_], ["lnF:" * k for k in keys_], [String(GSPEC4[j][1]) for j in gfree])
  function decode(z)
    gv = copy(gv0); gv[gfree] = z[2nk+1:end]
    g = Dict(GSPEC4[j][1] => (isinf(gv[j]) ? gv[j] : gdec4(j, gv[j])) for j in eachindex(GSPEC4))
    P = [HBAD4P.SpP4(clamp(z[k], -8.0, 6.0), clamp(z[nk+k], -15.0, 5.0), TR.Hmax[k], TR.ST[k], TR.longevity[k], TR.Topt[k]) for k in 1:nk]
    G = HBAD4P.GlobP4(g[:b], g[:a], g[:gR], g[:θ], g[:ω], g[:s], g[:c], g[:η], g[:λ], g[:μ], g[:kh], g[:m0], g[:ms], g[:σST], g[:β], g[:ζ], g[:mW], g[:mA],
                      g[:pD], g[:pH], g[:pM], g[:σv], g[:sdimax], g[:am], g[:ρ], g[:f0], g[:wT], g[:βC], g[:κ], g[:λ0], g[:λ1], g[:ε0], g[:ωR])
    Ob = ObsP(g[:rM], g[:rS], g[:σD], g[:σH])
    P, G, Ob
  end
  (z0=z0, names=names, decode=decode, gfree=gfree, fix=fix, nk=nk)
end
"Warm start from an HBAD3P fit directory (fit3p_<role>): ε per key, lnF per key (recruitment.csv), the shared globals by name."
function hbad3p_start(dir, keys_)
  sp = CSV.read(joinpath(dir, "species.csv"), DataFrame); gl = CSV.read(joinpath(dir, "globals.csv"), DataFrame)
  ed = Dict(String(r.key) => Float64(r.eps) for r in eachrow(sp))
  rc = isfile(joinpath(dir, "recruitment.csv")) ? CSV.read(joinpath(dir, "recruitment.csv"), DataFrame) : DataFrame(name=String[], value=Float64[])
  rd = Dict(String(r.name) => Float64(r.value) for r in eachrow(rc))
  g = Dict(Symbol(r.name) => Float64(r.value) for r in eachrow(gl))
  for nm in (:λ0, :λ1, :ε0); haskey(rd, String(nm)) && (g[nm] = rd[String(nm)]); end
  ([get(ed, k, 2.0) for k in keys_], [get(rd, "lnF:" * k, -3.0) for k in keys_], g)
end
