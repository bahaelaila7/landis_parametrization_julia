# HBAD2 calibration problem: the data sets (Sim A and Sim B, train / val / test), the parameter layout and the objective functions of a run, including the augmented
# Chebyshev scalarisation of the two-objective joint fit (prototype tools/hbad2/fit_simA.jl, fit_joint.jl).

"""
    CalibProblem

Everything a calibration run evaluates. `A[set]` / `B[set]` are the Sim A / Sim B data of `:train`, `:val` (and `:test` when requested), `layout` the search-vector layout, `θ0` the full start vector,
`recr_data` the data fit of the recruitment coordinates (3 per key), `objective` = `:sim_a` (minimise the Sim A joint NLL) | `:joint` (augmented Chebyshev scalarisation of fA and fB), `fb` the Sim B
objective mode, `cheb` the scalarisation constants `(lambda, rho, ideal, nadir)`.
"""
struct CalibProblem
  sw::CalibSwitches
  keys::Vector{String}
  longevity::Vector{Float64}
  layout::ThetaLayout
  recr_data::Vector{Float64}
  A::Dict{Symbol,SimAData}
  B::Dict{Symbol,SimBData}
  fb::Symbol
  objective::Symbol
  θ0::Vector{Float64}
  cheb::NamedTuple{(:lambda, :rho, :ideal, :nadir),Tuple{Float64,Float64,NTuple{2,Float64},NTuple{2,Float64}}}
  pc::Any
end

"The recruitment data fit (rho0 rhoS rhoL per key, in key order) from a csv with columns key, rho0, rhoS, rhoL."
function read_recruitment_data(csv::AbstractString, keys_::Vector{String})
  isfile(csv) || error("recruitment table not found: $csv")
  rc = CSV.read(csv, DataFrame); rcd = Dict(String(r.key) => r for r in eachrow(rc))
  miss = [k for k in keys_ if !haskey(rcd, k)]
  isempty(miss) || error("recruitment table $csv lacks rows for the keys $miss")
  return Float64.(reduce(vcat, [[rcd[k].rho0, rcd[k].rhoS, rcd[k].rhoL] for k in keys_]))
end

"""
    init_theta(tr, keys, sw, priors_csv) -> Vector{Float64}

Data-based start of the model coordinates (prototype `init_theta`): growth curves from the TRAIN trees' size quantiles and mean increments, background mortality from the observed death
rate, SDIMAX from the HBAD v1 priors (`priors_csv`: columns key, sdimax_p95_all), competition responses at neutral values, globals at 0.
"""
function init_theta(tr::DataFrame, keys_::Vector{String}, sw::CalibSwitches, priors_csv::AbstractString)
  isfile(priors_csv) || error("HBAD v1 priors not found: $priors_csv")
  pri = CSV.read(priors_csv, DataFrame); pk = Dict(String(r.key) => r for r in eachrow(pri))
  nsp = length(keys_)
  θ = zeros(nsp * NF + NGLOB)
  for (s, k) in enumerate(keys_)
    t = tr[(tr.key .== k) .& (tr.set .== "train"), :]
    f = t[(t.out .== "alive") .& in.(t.cclcd, Ref((1.0, 2.0, 3.0))), :]
    Hmax = 1.1 * Statistics.quantile(t.ht_m, 0.99); Dmax = 1.3 * Statistics.quantile(t.dia_cm, 0.99); c = 1.3
    Dm = Statistics.median(f.dia_cm); Hm = Statistics.median(f.ht_m)
    tD = Statistics.mean(max.(f.dia2_cm .- f.dia_cm, 0) ./ f.dt); tH = Statistics.mean(max.(f.ht2_m .- f.ht_m, 0) ./ f.dt)
    rD = tD / (c * Dm * ((Dmax / Dm)^(1 / c) - 1)); rH = tH / (c * Hm * ((Hmax / Hm)^(1 / c) - 1)); cH = c; cD = c
    if sw.form == 2
      cD = 0.5; Dmax = 40.0; rD = tD / ((Dm / 20)^cD * exp(-Dm / Dmax))
      cH = 0.3; Hmax = 10.0; rH = tH / ((Hm / 15)^cH * exp(-Hm / Hmax))
    end
    mort = sum(t.out .== "dead") / sum(t.out .!= "missing") / Statistics.mean(t.dt)
    sdm = haskey(pk, k) && !ismissing(pk[k].sdimax_p95_all) ? pk[k].sdimax_p95_all : pk["_" * (k[1] == '_' ? k[2:2] : first(t.grp))].sdimax_p95_all
    v = Dict(:Hmax => Hmax, :rH => rH, :cH => cH, :Dmax => Dmax, :rD => rD, :cD => cD, :gOD => 1.0, :gRD => 0.0, :gOH => 0.5, :gRH => 0.2,
             :mbg => log(0.7 * mort), :mD => -0.3, :msup => 0.02, :sdimax => sdm)
    for (j, nm) in enumerate(SP_FREE); θ[(s-1)*NF+j] = enc_param(sw, j, dec_param(sw, j, enc_param(sw, j, v[nm]))); end
  end
  θ[nsp*NF+1:end] .= 0.0
  return θ
end

"The model parameters of a full search vector as the plugin's per-ecoregion struct (`rec` from the recruitment coordinates), plus the observation spread `sigma_w`."
function model_params(prob::CalibProblem, θ::AbstractVector)
  P, G, σw = decode_theta(θ, prob.layout.nsp, prob.sw, prob.longevity)
  R = recruitment_of(θ, prob.layout, prob.recr_data)
  return HBAD2Plugin.HBAD2EcoParams(P, R, HBAD2Plugin.Allo[], G), σw
end

"Sim A score parts of `θ` on the set `set` (`:train` | `:val` | `:test`)."
function simA_scores(prob::CalibProblem, θ::AbstractVector, set::Symbol=:train)
  ep, σw = model_params(prob, θ)
  return eval_simA!(prob.A[set], ep, prob.sw, σw)
end

"Sim B score parts of `θ` on the set `set`."
function simB_scores(prob::CalibProblem, θ::AbstractVector, set::Symbol=:train)
  ep, _ = model_params(prob, θ)
  return eval_simB!(prob.B[set], ep, prob.sw)
end

"`(fA, fB)` of `θ` on `set`: the Sim A total NLL and the scalar Sim B objective of `prob.fb`. fB is `NaN` for a Sim A only problem."
function objectives(prob::CalibProblem, θ::AbstractVector, set::Symbol=:train)
  a = simA_scores(prob, θ, set).total
  haskey(prob.B, set) || return (a, NaN)
  return (a, fb_value(simB_scores(prob, θ, set), prob.fb))
end

"Augmented Chebyshev scalarisation of `(a, b)` with the problem's fixed ideal / nadir, weight `lambda` and `rho`."
@inline function chebyshev(c, a::Real, b::Real)
  x = (a - c.ideal[1]) / (c.nadir[1] - c.ideal[1]); y = (b - c.ideal[2]) / (c.nadir[2] - c.ideal[2])
  return max(c.lambda * x, (1 - c.lambda) * y) + c.rho * (c.lambda * x + (1 - c.lambda) * y)
end

"""
    scalar_objective(prob, z) -> Float64

The minimised scalar of a run on the searched coordinates `z` (the full vector is `expand(layout, θ0, z)`): the Sim A total NLL (`objective = :sim_a`) or the Chebyshev scalarisation of
(fA, fB) (`:joint`), both on the TRAIN set.
"""
function scalar_objective(prob::CalibProblem, z::AbstractVector)
  θ = expand(prob.layout, prob.θ0, z)
  prob.objective == :sim_a && return simA_scores(prob, θ, :train).total
  a, b = objectives(prob, θ, :train)
  return chebyshev(prob.cheb, a, b)
end

"""
    build_calib_problem(; db_path, split, ...) -> CalibProblem

Load the data and build the problem of a calibration run. `split`: DataFrame `statecd unitcd countycd plot set [eco]`; `intervals_cache`: directory of the Arrow cache of the interval product (built when
missing); `longevity_csv`, `recruitment_csv`, `priors_csv`: input tables; `start_theta` / `start_recr`: csv files (column `theta`) of the start (empty = data-based start / the recruitment data fit);
`recr` = :free | :fixed | :kappa, `freeze` = coordinates kept at their start (`mbg`, `G:mD2`, ...); `objective` = :sim_a | :joint (needs Sim B); `fb` the Sim B mode; `sets` the evaluated sets.
`ideal` / `nadir` = `(fA, fB)` reference points of the scalarisation (empty = the start's values: ideal = start - 5 %|start|, nadir = start).
"""
function build_calib_problem(; db_path::AbstractString, split::DataFrame, intervals_cache::AbstractString, sw::CalibSwitches=CalibSwitches(), table::AbstractString="curated_trees",
                             states::AbstractVector{<:Integer}=[12, 13], rebuild::Bool=false, longevity_csv::AbstractString, recruitment_csv::AbstractString, priors_csv::AbstractString="",
                             start_theta::AbstractString="", start_recr::AbstractString="", recr::Symbol=:fixed, freeze::AbstractVector{<:AbstractString}=String[],
                             objective::Symbol=:sim_a, fb::Symbol=:margins, sets::AbstractVector{Symbol}=[:train, :val], lambda::Real=0.5, rho::Real=0.05,
                             ideal::AbstractVector{<:Real}=Float64[], nadir::AbstractVector{<:Real}=Float64[], simb_sigma::Real=0.16, verbose::Bool=true)
  objective in (:sim_a, :joint) || error("hbad2_calib: objective must be sim_a or joint, got $objective")
  :train in sets || error("hbad2_calib: the evaluated sets must include train")
  x = Data.load_hbad2_intervals(intervals_cache, db_path, split; states=states, table=table, rebuild=rebuild)
  tr = load_tracked_trees(x, sw)
  keys_ = sort(unique(tr.key)); nsp = length(keys_)
  longev = species_longevity(longevity_csv, tr, keys_)
  dummy = (keys=keys_, sp=HBAD2Plugin.SpP[HBAD2Plugin.SpP(zeros(HBAD2Plugin.N_SP_PARAMS)) for _ in keys_], rec=HBAD2Plugin.RecP[HBAD2Plugin.RecP(0.0, 0.0, 0.0, 2.54, 3.0, 5.0) for _ in keys_],
           glob=HBAD2Plugin.GlobP(0.0, 10.0, 0.0, 0.0, 0.0, sw.form, 0.0, 0.0), meta=Dict{String,String}())
  pc = Pan.PluginConfig(HBAD2Plugin.HBAD2Context(keys_, [collect(1:nsp)]; v2tables=dummy, vigour=sw.vigour, verbose=false))
  verbose && @info "HBAD2 calibration data" keys = join(keys_, ",") trees = nrow(tr) sets = join(string.(sets), ",") objective = String(objective)
  A = Dict{Symbol,SimAData}(); B = Dict{Symbol,SimBData}()
  for st in sets
    A[st] = build_simA(tr, String(st), keys_, sw, pc)
    objective == :joint && (B[st] = build_simB(x, String(st), keys_, sw, pc; sigma=simb_sigma))
    verbose && @info "set $(st)" simA = string(A[st]) simB = haskey(B, st) ? string(B[st]) : "-"
  end
  recr_data = read_recruitment_data(recruitment_csv, keys_)
  layout = ThetaLayout(keys_; recr=recr, freeze=freeze)
  θg = isempty(start_theta) ? init_theta(tr, keys_, sw, priors_csv) : read_theta_csv(start_theta, nsp)
  θ0 = recr == :free ? vcat(θg, isempty(start_recr) ? recr_data : Float64.(vec(CSV.read(start_recr, DataFrame).theta))) :
       recr == :kappa ? vcat(θg, 0.0) : copy(θg)
  recr == :free && length(θ0) != length(layout) && error("hbad2_calib: start_recr has the wrong length ($(length(θ0) - length(θg)) values, $(3nsp) expected)")
  tmp = CalibProblem(sw, keys_, longev, layout, recr_data, A, B, fb, objective, θ0,
                     (lambda=Float64(lambda), rho=Float64(rho), ideal=(0.0, 0.0), nadir=(1.0, 1.0)), pc)
  cheb = tmp.cheb
  if objective == :joint
    a0, b0 = objectives(tmp, θ0, :train)
    idl = isempty(ideal) ? (a0 - 0.05abs(a0), b0 - 0.05abs(b0)) : (Float64(ideal[1]), Float64(ideal[2]))
    nad = isempty(nadir) ? (a0, b0) : (Float64(nadir[1]), Float64(nadir[2]))
    (nad[1] - idl[1] > 0 && nad[2] - idl[2] > 0) || error("hbad2_calib: the scalarisation needs nadir > ideal for both objectives (fA $(idl[1]) -> $(nad[1]), fB $(idl[2]) -> $(nad[2]))")
    0 < lambda < 1 && isempty(ideal) && @warn "hbad2_calib: lambda strictly between 0 and 1 without an explicit ideal point: the normalisation is arbitrary"
    cheb = (lambda=Float64(lambda), rho=Float64(rho), ideal=idl, nadir=nad)
  end
  return CalibProblem(sw, keys_, longev, layout, recr_data, A, B, fb, objective, θ0, cheb, pc)
end

"The same problem with another scalarisation weight `lambda` (data and reference points shared)."
with_lambda(p::CalibProblem, lambda::Real) =
  CalibProblem(p.sw, p.keys, p.longevity, p.layout, p.recr_data, p.A, p.B, p.fb, p.objective, p.θ0, (lambda=Float64(lambda), rho=p.cheb.rho, ideal=p.cheb.ideal, nadir=p.cheb.nadir), p.pc)
