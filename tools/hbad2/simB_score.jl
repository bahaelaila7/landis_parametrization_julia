# Score a parameter vector (+ optional free recruitment) on the Sim B likelihood terms (stand, size, species margins, Lorey H) for a set.
#   ./julia_gdal.sh --project=. --threads=3 tools/hbad2/simB_score.jl <θ.csv> <recr.csv|-> [set]
include(joinpath(@__DIR__, "common.jl")); include(joinpath(@__DIR__, "simB_lik.jl"))
θpath = ARGS[1]; rpath = ARGS[2] == "-" ? "" : ARGS[2]; SET = length(ARGS) >= 3 ? ARGS[3] : "val"
tr = load_trees(); keys_ = sort(unique(tr.key)); nsp = length(keys_)
lon = CSV.read("/workspace/species_longevity_ref.csv", DataFrame); lond = Dict(String(r.species_symbol) => Float64(r.longevity_yr) for r in eachrow(lon))
longev = [median([get(lond, s, 300.0) for s in unique(tr.sym[tr.key .== k])]) for k in keys_]
P, G = decode(read_theta(θpath, nsp), nsp, longev)
rc = CSV.read(joinpath(R2DIR, SITE_SUB ? "recruitment_coefs_subplot.csv" : "recruitment_coefs.csv"), DataFrame); rcd = Dict(String(r.key) => r for r in eachrow(rc))
R = isempty(rpath) ? [RecP(rcd[k].rho0, rcd[k].rhoS, rcd[k].rhoL, 2.54, 3.0, 5.0) for k in keys_] :
    (z = vec(CSV.read(rpath, DataFrame).theta); [RecP(clamp(z[3k-2], -12, 2), clamp(z[3k-1], -1, 4), clamp(z[3k], -10, 2), 2.54, 3.0, 5.0) for k in 1:nsp])
B = build_simB(SET, keys_); r = simB_nll!(B, P, G, R)
@printf("SIMB %s %s: stand %.1f  size %.1f  species %.1f  plotstand %.1f (trees %.1f)  LoreyH %.1f\n", SET, basename(dirname(θpath)) * "/" * basename(θpath), r.total, r.size, r.species, r.plotstand, r.plotstand_trees, r.H)
