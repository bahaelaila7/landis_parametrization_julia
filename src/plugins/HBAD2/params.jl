# HBAD2 parameter containers (isbits). Field names and meaning: docs/hbad_v2.md 4.1 and the module header of the prototype kernel.
const SP_PARAM_NAMES = (:Hmax, :rH, :cH, :Dmax, :rD, :cD, :gOD, :gRD, :gOH, :gRH, :mbg, :mD, :msup, :sdimax, :longevity)
const N_SP_PARAMS = length(SP_PARAM_NAMES)

"""Per species-key parameters (growth curves, competition responses, mortality, SDIMAX cap, LONGEVITY)."""
struct SpP
  Hmax::Float64; rH::Float64; cH::Float64; Dmax::Float64; rD::Float64; cD::Float64
  gOD::Float64; gRD::Float64; gOH::Float64; gRH::Float64; mbg::Float64; mD::Float64; msup::Float64; sdimax::Float64; longevity::Float64
end
SpP(v::AbstractVector) = SpP(ntuple(i -> Float64(v[i]), N_SP_PARAMS)...)

"""Global parameters. `form` = 2 (power-exponential potential growth) for v2g; `mA` = age-mortality shape (Biomass Succession "d"); `σv` = vigour spread."""
struct GlobP
  mD2::Float64; mA::Float64; pH::Float64; pD::Float64; pM::Float64
  form::Int
  mSD::Float64; σv::Float64
end

"""Recruitment per species key: R [trees/ha/yr] = RSCALE exp(rho0 + rhoS ln(1 + BA_sp(D >= DSEED)) + rhoL SDI/RD_REF); recruits enter at (D0, H0, A0)."""
struct RecP
  rho0::Float64; rhoS::Float64; rhoL::Float64; D0::Float64; H0::Float64; A0::Float64
end

"""Reported-AGB allometry of one species: ln(mean tree dry mass in lb) = mq_a + mq_b ln(QMD in inch) (HBAD v1 priors, never used by the dynamics)."""
struct Allo
  mq_a::Float64; mq_b::Float64
end

"""
Fixed v2 parameter set of a run. `KEYS[g]` is the v2 key (own species or the `_S`/`_H` group) of global species g; `SP[g]`, `REC[g]` are the key's
parameters (species sharing a key share the values). Indexed like HBADParams (global species ids; ECO_SPECIES_IDS maps eco-local -> global).
"""
struct HBAD2Params
  ECO_LIST::Vector{String}
  SPECIES_LIST::Vector{String}
  ECO_SPECIES_IDS::Vector{Vector{UIntType}}
  KEYS::Vector{String}
  SP::Vector{SpP}
  REC::Vector{RecP}
  G::GlobP
end

"""Per-ecoregion resolved parameters (eco-local species index; concrete isbits element types)."""
struct HBAD2EcoParams
  sp::Vector{SpP}
  rec::Vector{RecP}
  allo::Vector{Allo}
  G::GlobP
end

"""The v2 tables of a parameter directory (params_{species,global,recruitment}.csv + meta.txt): `(keys, sp, rec, glob, meta)`, per key in file order."""
function load_v2_tables(dir::AbstractString=DEFAULT_PARAMS_DIR)
  fs = joinpath(dir, "params_species.csv"); fg = joinpath(dir, "params_global.csv"); fr = joinpath(dir, "params_recruitment.csv"); fm = joinpath(dir, "meta.txt")
  for f in (fs, fg, fr)
    isfile(f) || error("HBAD2: parameter file not found: $f")
  end
  S = CSV.read(fs, DataFrame); Gd = CSV.read(fg, DataFrame); R = CSV.read(fr, DataFrame)
  keys_ = String.(S.key)
  sp = [SpP([Float64(S[r, nm]) for nm in SP_PARAM_NAMES]) for r in 1:nrow(S)]
  gd = Dict(String(r.name) => Float64(r.value) for r in eachrow(Gd))
  for nm in ("mD2", "mA", "pH", "pD", "pM", "form", "mSD", "σv")
    haskey(gd, nm) || error("HBAD2: $fg lacks the global parameter '$nm'")
  end
  glob = GlobP(gd["mD2"], gd["mA"], gd["pH"], gd["pD"], gd["pM"], round(Int, gd["form"]), gd["mSD"], gd["σv"])
  rd = Dict(String(r.key) => r for r in eachrow(R))
  all(k -> haskey(rd, k), keys_) || error("HBAD2: $fr lacks recruitment rows for $(setdiff(keys_, keys(rd)))")
  rec = [RecP(Float64(rd[k].rho0), Float64(rd[k].rhoS), Float64(rd[k].rhoL), Float64(rd[k].D0), Float64(rd[k].H0), Float64(rd[k].A0)) for k in keys_]
  meta = Dict{String,String}()
  if isfile(fm)
    for l in eachline(fm)
      p = split(l, "="; limit=2)
      length(p) == 2 && (meta[String(p[1])] = String(p[2]))
    end
    haskey(meta, "form") && parse(Int, meta["form"]) != glob.form && error("HBAD2: meta.txt form=$(meta["form"]) but params_global.csv form=$(glob.form)")
  end
  return (keys=keys_, sp=sp, rec=rec, glob=glob, meta=meta)
end

"""Species symbol -> group "S" | "H" (HBAD v1 species_group.csv, = FIA REF_SPECIES SFTWD_HRDWD); unknown symbols and non-group catch-alls -> "H"."""
function _group_of(name::AbstractString, grp_map::AbstractDict{String,String})
  g = HBADPlugin.species_group(name, grp_map)
  return g == "ALL" ? "H" : g
end

"""
    map_species_keys(species_list, keys; grp_map) -> Vector{String}

v2 key of every Pan species: its own symbol when the v2 set has a row for it (the 10 species with their own parameters), else the pooled group key
`_S` (softwood) / `_H` (hardwood) by REF_SPECIES SFTWD_HRDWD. Catch-all Pan names (`_S*`, `_H*`, `_GRP_n_*`) map to their group.
"""
function map_species_keys(species_list::AbstractVector{<:AbstractString}, keys_::AbstractVector{<:AbstractString}; grp_map::AbstractDict{String,String})
  have = Set(String.(keys_))
  return String[(s = String(n); (!startswith(s, "_") && s in have) ? s : ("_" * _group_of(s, grp_map)))
                for n in species_list]
end
