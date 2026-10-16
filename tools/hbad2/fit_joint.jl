# HBAD v2 joint calibration, two objectives (local vs global, as the Sim A ⊕ Sim B tandem design):
#   fA = Sim A tracked-cohort joint NLL (TRAIN)          fB = Sim B stand-table Poisson NLL + Lorey-H NLL (TRAIN, natural clean plots)
# The two scores share plots, so they are NOT summed as one likelihood: each λ of a weight grid gets its own CMA-ES run minimising the
# augmented Chebyshev scalarisation max(λ fA', (1−λ) fB') + ρ (λ fA' + (1−λ) fB'), f' = (f − f_ideal)/(f_ref − f_ideal) with f_ref = the
# start point and f_ideal = the single-objective optimum estimate (start − 2 % of |start|, re-based when beaten). Recruitment
# (ρ0, ρS, ρL per species) is free (start = the Poisson fit). Every run reports TRAIN and VAL (selection) values of both objectives.
#   HBAD2_FORM=2 HBAD2_SIZE_SPLIT=5,12.7,25,40 ./julia_gdal.sh --project=. --threads=8 tools/hbad2/fit_joint.jl <tag> <start θ.csv> <λ list e.g. 0,0.5,1> [evals]
include(joinpath(@__DIR__, "common.jl")); include(joinpath(@__DIR__, "cmaes.jl")); include(joinpath(@__DIR__, "simB_lik.jl"))
tag, θpath, lams = ARGS[1], ARGS[2], parse.(Float64, split(ARGS[3], ",")); EV = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 12000
const FB_MODE = get(ENV, "HBAD2_FB", "stand")     # Sim B objective: stand (species × class + Lorey H) | size (plot size-class totals + Lorey H) | margins (size + plot species×{sapling,tree} totals + Lorey H) | both
OUTD = joinpath(R2DIR, "joint_" * tag); mkpath(OUTD); write_meta(OUTD); open(io -> (println(io, "fb=", FB_MODE); println(io, "recr_fixed=", get(ENV, "HBAD2_RECR_FIXED", "0"))), joinpath(OUTD, "meta.txt"), "a")
logio = open(joinpath(OUTD, "log.txt"), "a"); lg(s) = (println(s); println(logio, s); flush(logio); flush(stdout))
tr = load_trees(); keys_ = sort(unique(tr.key)); nsp = length(keys_)
lon = CSV.read("/workspace/species_longevity_ref.csv", DataFrame); lond = Dict(String(r.species_symbol) => Float64(r.longevity_yr) for r in eachrow(lon))
longev = [median([get(lond, s, 300.0) for s in unique(tr.sym[tr.key .== k])]) for k in keys_]
rc = CSV.read(joinpath(R2DIR, SITE_SUB ? "recruitment_coefs_subplot.csv" : "recruitment_coefs.csv"), DataFrame); rcd = Dict(String(r.key) => r for r in eachrow(rc))
θg = read_theta(θpath, nsp); ng = length(θg)
rpath = replace(θpath, "theta" => "recr")       # a joint fit's θ comes with its recruitment vector; otherwise start from the Poisson fit
# HBAD2_RECR_FIXED=1: recruitment stays at its own data likelihood optimum (the Poisson fit on observed sapling ingrowth) and is NOT a
# free parameter of the joint fit (otherwise it can compensate structural errors elsewhere — the j6 composition drift, docs §5.14)
const RECR_FIXED = get(ENV, "HBAD2_RECR_FIXED", "0") == "1"
recr_poisson = reduce(vcat, [[rcd[k].rho0, rcd[k].rhoS, rcd[k].rhoL] for k in keys_])
θ0 = RECR_FIXED ? copy(θg) : rpath != θpath && isfile(rpath) ? vcat(θg, vec(CSV.read(rpath, DataFrame).theta)) : vcat(θg, recr_poisson)
decR(z) = [RecP(clamp(z[3k-2], -12, 2), clamp(z[3k-1], -1, 4), clamp(z[3k], -10, 2), 2.54, 3.0, 5.0) for k in 1:nsp]
TRA = build_cohorts(tr, "train", keys_); VAA = build_cohorts(tr, "val", keys_)
TRB = build_simB("train", keys_); VAB = build_simB("val", keys_)
lg(@sprintf("[fB=%s] joint %s: SimA train %d cohorts, SimB train %d sites / val %d sites; params %d (+%d recruitment); λ %s", FB_MODE, tag, length(TRA.S.sp), length(TRB.live0), length(VAB.live0), ng, 3nsp, ARGS[3]))
function both(θ, A, B)
  P, G = decode(view(θ, 1:ng), nsp, longev); R = RECR_FIXED ? decR(recr_poisson) : decR(view(θ, ng+1:length(θ)))
  simulate!(A.S, P, G); a = nll_parts(cohort_view(A), A.obs).total
  r = simB_nll!(B, P, G, R)
  a, (FB_MODE == "size" ? r.size + r.H : FB_MODE == "margins" ? r.size + r.species + r.H : FB_MODE == "plotstand" ? r.plotstand + r.H : FB_MODE == "both" ? r.total + r.size : r.total)
end
a0, b0 = both(θ0, TRA, TRB); va0, vb0 = both(θ0, VAA, VAB)
lg(@sprintf("START train fA %.1f fB %.1f | val fA %.1f fB %.1f", a0, b0, va0, vb0))
# FIXED reference points for the whole run (a moving ideal makes the scalarisation non-stationary: CMA's best-so-far is then compared
# across different objectives and, at λ = 0 or 1, pinned at 0 → the returned point would be the first improvement, not the best)
# ideal = the single-objective optima when known (env HBAD2_IDEAL="fA,fB" on TRAIN, e.g. from the λ = 1 and λ = 0 runs), else start − 5 %|start|
# (only meaningful for λ ∈ {0, 1}, where the scalarisation is monotone in the one objective; a near-zero start makes 5 % a bad scale)
ide = get(ENV, "HBAD2_IDEAL", "")
idealA = Ref(isempty(ide) ? a0 - 0.05abs(a0) : parse(Float64, split(ide, ",")[1])); idealB = Ref(isempty(ide) ? b0 - 0.05abs(b0) : parse(Float64, split(ide, ",")[2])); ρ = 0.05
# nadir (worst value of each objective on the front, e.g. fA at the λ = 0 end, fB at the λ = 1 end; env HBAD2_NADIR="fA,fB"), else the start
nad = get(ENV, "HBAD2_NADIR", "")
nadA = isempty(nad) ? a0 : parse(Float64, split(nad, ",")[1]); nadB = isempty(nad) ? b0 : parse(Float64, split(nad, ",")[2])
(nadA - idealA[] > 0 && nadB - idealB[] > 0) || error("scalarisation needs nadir > ideal for both objectives (fA $(idealA[])→$(nadA), fB $(idealB[])→$(nadB))")
lg(@sprintf("scalarisation: ideal fA %.1f fB %.1f | nadir fA %.1f fB %.1f (start %.1f / %.1f)", idealA[], idealB[], nadA, nadB, a0, b0))
any(l -> 0 < l < 1, lams) && isempty(ide) && @warn "λ strictly between 0 and 1 without HBAD2_IDEAL: the normalisation is arbitrary"
res = DataFrame(lam=Float64[], fA=Float64[], fB=Float64[], vA=Float64[], vB=Float64[])
for λ in lams
  f = function (θ)
    a, b = both(θ, TRA, TRB)
    x = (a - idealA[]) / (nadA - idealA[]); y = (b - idealB[]) / (nadB - idealB[])
    max(λ * x, (1 - λ) * y) + ρ * (λ * x + (1 - λ) * y)
  end
  r = cmaes(f, θ0, parse(Float64, get(ENV, "HBAD2_SIGMA", "0.03")); λ=parse(Int, get(ENV, "HBAD2_LAMBDA", "24")), maxevals=EV, logf=lg, log_every=25)
  local a, b, va, vb
  a, b = both(r.x, TRA, TRB); va, vb = both(r.x, VAA, VAB)
  lg(@sprintf("λ=%.2f  train fA %.1f fB %.1f | VAL fA %.1f fB %.1f  (start val %.1f / %.1f)", λ, a, b, va, vb, va0, vb0))
  push!(res, (λ, a, b, va, vb))
  CSV.write(joinpath(OUTD, @sprintf("theta_l%.2f.csv", λ)), DataFrame(theta=r.x[1:ng])); CSV.write(joinpath(OUTD, @sprintf("recr_l%.2f.csv", λ)), DataFrame(theta=RECR_FIXED ? recr_poisson : r.x[ng+1:end]))
  CSV.write(joinpath(OUTD, "front.csv"), res)
end
