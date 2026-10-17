# Export a fitted HBAD v2 parameter set in decoded, plugin-ready form:
#   <fit>/params_species.csv (key, species members, every SpP field), <fit>/params_global.csv (GlobP fields), <fit>/params_recruitment.csv
#   HBAD2_SIGV_MAX=0.7 HBAD2_VIGOUR=1 HBAD2_SITE=subplot HBAD2_FORM=2 HBAD2_SIZE_SPLIT=5,12.7,25,40 ./julia_gdal.sh --project=. tools/hbad2/export_params.jl v2g
include(joinpath(@__DIR__, "common.jl"))
tag = ARGS[1]; FD = joinpath(R2DIR, "fit_" * tag); check_meta(FD)
tr = load_trees(); keys_ = sort(unique(tr.key)); nsp = length(keys_)
lon = CSV.read("/workspace/species_longevity_ref.csv", DataFrame); lond = Dict(String(r.species_symbol) => Float64(r.longevity_yr) for r in eachrow(lon))
longev = [median([get(lond, s, 300.0) for s in unique(tr.sym[tr.key .== k])]) for k in keys_]
P, G = decode(read_theta(joinpath(FD, "theta.csv"), nsp), nsp, longev)
members = [k[1] == '_' ? "group:" * k[2:2] * " (all other " * (k[2] == 'S' ? "softwoods" : "hardwoods") * ")" : k for k in keys_]
sp = DataFrame([nm => [getfield(P[s], nm) for s in 1:nsp] for nm in HBAD2.SP_PARAM_NAMES]); insertcols!(sp, 1, :key => keys_, :members => members)
CSV.write(joinpath(FD, "params_species.csv"), sp)
CSV.write(joinpath(FD, "params_global.csv"), DataFrame(name=[string(f) for f in fieldnames(HBAD2.GlobP)], value=[Float64(getfield(G, f)) for f in fieldnames(HBAD2.GlobP)]))
rpath = joinpath(FD, "recr.csv")
z = isfile(rpath) ? vec(CSV.read(rpath, DataFrame).theta) : Float64[]
rc = CSV.read(joinpath(R2DIR, SITE_SUB ? "recruitment_coefs_subplot.csv" : "recruitment_coefs.csv"), DataFrame); rcd = Dict(String(r.key) => r for r in eachrow(rc))
rec = DataFrame(key=keys_, rho0=[isempty(z) ? rcd[k].rho0 : z[3i-2] for (i, k) in enumerate(keys_)], rhoS=[isempty(z) ? rcd[k].rhoS : z[3i-1] for (i, k) in enumerate(keys_)],
                rhoL=[isempty(z) ? rcd[k].rhoL : z[3i] for (i, k) in enumerate(keys_)], D0=2.54, H0=3.0, A0=5.0)
CSV.write(joinpath(FD, "params_recruitment.csv"), rec)
big = Set(k for k in keys_ if k[1] != '_')
println("species keys: ", join(keys_, ","), "  | other species map to _S/_H by SFTWD_HRDWD (REF_SPECIES)")
println(G); println(first(sp, 3)); println(first(rec, 3))
