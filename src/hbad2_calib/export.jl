# Plugin-ready export of a calibrated parameter vector: the files `HBAD2Plugin.load_v2_tables` reads (params_species.csv, params_global.csv, params_recruitment.csv, meta.txt)
# plus the search vectors (theta.csv, recr.csv) in the prototype's format, so a calibration output directory can be used as `hbad2_params_dir` and by the prototype tools.

"""
    write_params(dir, prob, θ; note="")

Write the decoded parameter set of the full search vector `θ` to `dir`: `theta.csv` (model block), `recr.csv` (recruitment coordinates actually used: free, or the data fit / kappa-shifted data fit),
`params_species.csv`, `params_global.csv`, `params_recruitment.csv` (plugin-ready, same layout as the prototype's export) and `meta.txt` (structural switches).
"""
function write_params(dir::AbstractString, prob::CalibProblem, θ::AbstractVector; note::AbstractString="")
  mkpath(dir)
  L = prob.layout; sw = prob.sw; nsp = L.nsp; keys_ = prob.keys
  P, G, σw = decode_theta(θ, nsp, sw, prob.longevity)
  R = recruitment_of(θ, L, prob.recr_data)
  zr = L.recr == :free ? Float64.(θ[L.ng+1:end]) : L.recr == :kappa ? kshift(prob.recr_data, clamp(θ[L.ng+1], -3, 1), nsp) : prob.recr_data
  CSV.write(joinpath(dir, "theta.csv"), DataFrame(theta=Float64.(θ[1:L.ng])))
  CSV.write(joinpath(dir, "recr.csv"), DataFrame(theta=zr))
  members = [k[1] == '_' ? "group:" * k[2:2] * " (all other " * (k[2] == 'S' ? "softwoods" : "hardwoods") * ")" : k for k in keys_]
  sp = DataFrame([nm => [getfield(P[s], nm) for s in 1:nsp] for nm in HBAD2Plugin.SP_PARAM_NAMES]); insertcols!(sp, 1, :key => keys_, :members => members)
  CSV.write(joinpath(dir, "params_species.csv"), sp)
  CSV.write(joinpath(dir, "params_global.csv"), DataFrame(name=[string(f) for f in fieldnames(HBAD2Plugin.GlobP)], value=[Float64(getfield(G, f)) for f in fieldnames(HBAD2Plugin.GlobP)]))
  CSV.write(joinpath(dir, "params_recruitment.csv"), DataFrame(key=keys_, rho0=[r.rho0 for r in R], rhoS=[r.rhoS for r in R], rhoL=[r.rhoL for r in R], D0=[r.D0 for r in R], H0=[r.H0 for r in R], A0=[r.A0 for r in R]))
  open(joinpath(dir, "meta.txt"), "w") do io
    println(io, "form=", sw.form); println(io, "size_split=", join(sw.size_split, ",")); println(io, "site=", sw.site_sub ? "subplot" : "plot")
    println(io, "vigour=", sw.vigour ? 1 : 0); sw.vigour && println(io, "sigv_max=", sw.sigv_max); sw.sizecls && println(io, "sizecls=1")
    println(io, "sigma_w=", σw)
    isempty(note) || println(io, "source=", note)
  end
  return dir
end
