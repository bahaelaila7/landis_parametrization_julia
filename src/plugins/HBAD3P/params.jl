# HBAD3P parameter containers (isbits). Field names and meaning: the module header of the reference kernel (HBAD3P.jl of the prototype) and docs/hbad3p_plugin.md.

"""Per species-key parameters: calibrated growth efficiency lnε and fecundity lnF, traits Hmax [m], shade class ST (1-5), LONGEVITY [yr]."""
struct SpPP
  lnε::Float64; Hmax::Float64; ST::Float64; longevity::Float64; lnF::Float64
end

const GLOB_NAMES = (:b, :a, :gR, :θ, :ω, :s, :c, :η, :λ, :μ, :kh, :m0, :ms, :σST, :β, :ζ, :mW, :mA, :pD, :pH, :pM, :σv, :sdimax, :am, :ρ, :f0, :φ, :crhi, :crlo, :kc, :τr, :τe)
# ASCII names of the yaml roster (the csv files keep the unicode names of the fit)
const GLOB_YAML = ("b", "a", "gR", "theta", "omega", "s", "c", "eta", "lambda", "mu", "kh", "m0", "ms", "sigma_ST", "beta", "zeta", "mW", "mA", "pD", "pH", "pM", "sigma_v", "sdimax",
                   "am", "rho", "f0", "phi", "crhi", "crlo", "kc", "tau_r", "tau_e")

"""Global parameters (32, the field order of the reference `GlobPP`): growth b a gR θ ω, height s c η λ μ kh, mortality m0 ms σST β ζ mW mA, planted pD pH pM, vigour spread σv, SDIMAX, survival light am,
maintenance ρ, light floor f0, crown φ crhi crlo kc τr τe."""
struct GlobPP
  b::Float64; a::Float64; gR::Float64; θ::Float64; ω::Float64
  s::Float64; c::Float64; η::Float64; λ::Float64; μ::Float64; kh::Float64
  m0::Float64; ms::Float64; σST::Float64; β::Float64; ζ::Float64; mW::Float64; mA::Float64
  pD::Float64; pH::Float64; pM::Float64; σv::Float64; sdimax::Float64
  am::Float64; ρ::Float64; f0::Float64; φ::Float64
  crhi::Float64; crlo::Float64; kc::Float64
  τr::Float64; τe::Float64
end
GlobPP(v::AbstractVector) = (length(v) == length(GLOB_NAMES) || error("GlobPP: expected $(length(GLOB_NAMES)) values, got $(length(v))"); GlobPP(ntuple(i -> Float64(v[i]), length(GLOB_NAMES))...))
@assert fieldnames(GlobPP) == GLOB_NAMES

"""Recruitment globals: light window λ0 e^{λ1 (ST - 3)} on the stand RD, background seed rain ε0 where no conspecific seed tree is present."""
struct RecGP
  λ0::Float64; λ1::Float64; ε0::Float64
end

"""Reported-AGB allometry of one species: ln(mean tree dry mass in lb) = mq_a + mq_b ln(QMD in inch) (HBAD v1 priors, never used by the dynamics)."""
struct Allo
  mq_a::Float64; mq_b::Float64
end

"""
Fixed HBAD3P parameter set of a run. `KEYS[g]` is the key (own species or the `_S`/`_H` group) of global species g; `SP[g]` the key's parameters (species sharing a key share the values); `REC`, `G` global;
`FAREA` = density convention of the fit (1: stand densities per ha of the FORESTED subplot, 0: per ha of the whole subplot).
"""
struct HBAD3PParams
  ECO_LIST::Vector{String}
  SPECIES_LIST::Vector{String}
  ECO_SPECIES_IDS::Vector{Vector{UIntType}}
  KEYS::Vector{String}
  SP::Vector{SpPP}
  REC::RecGP
  G::GlobPP
  FAREA::Int
end

"""Per-ecoregion resolved parameters (eco-local species index; concrete isbits element types)."""
struct HBAD3PEcoParams
  sp::Vector{SpPP}
  allo::Vector{Allo}
  rec::RecGP
  G::GlobPP
end

"""The tables of a parameter directory (params_{species,global,recruitment}.csv + meta.txt): `(keys, sp, rec, glob, farea, meta)`, per key in file order. `eps` of the species file is the growth efficiency (lnε = log eps)."""
function load_3p_tables(dir::AbstractString=DEFAULT_PARAMS_DIR)
  fs = joinpath(dir, "params_species.csv"); fg = joinpath(dir, "params_global.csv"); fr = joinpath(dir, "params_recruitment.csv"); fm = joinpath(dir, "meta.txt")
  for f in (fs, fg, fr)
    isfile(f) || error("HBAD3P: parameter file not found: $f")
  end
  S = CSV.read(fs, DataFrame); Gd = CSV.read(fg, DataFrame); R = CSV.read(fr, DataFrame)
  keys_ = String.(S.key)
  gd = Dict(String(r.name) => Float64(r.value) for r in eachrow(Gd))
  for nm in GLOB_NAMES
    haskey(gd, String(nm)) || error("HBAD3P: $fg lacks the global parameter '$nm'")
  end
  glob = GlobPP(Float64[gd[String(nm)] for nm in GLOB_NAMES])
  rd = Dict(String(r.name) => Float64(r.value) for r in eachrow(R))
  for nm in ("λ0", "λ1", "ε0"); haskey(rd, nm) || error("HBAD3P: $fr lacks '$nm'"); end
  for k in keys_; haskey(rd, "lnF:" * k) || error("HBAD3P: $fr lacks 'lnF:$k'"); end
  sp = [SpPP(log(Float64(S.eps[i])), Float64(S.Hmax[i]), Float64(S.ST[i]), Float64(S.longevity[i]), rd["lnF:" * keys_[i]]) for i in eachindex(keys_)]
  rec = RecGP(rd["λ0"], rd["λ1"], rd["ε0"])
  meta = Dict{String,String}()
  if isfile(fm)
    for l in eachline(fm)
      p = split(l, "="; limit=2)
      length(p) == 2 && (meta[String(p[1])] = String(p[2]))
    end
  end
  haskey(meta, "vig_n") && parse(Int, meta["vig_n"]) != length(VIG_Z) && error("HBAD3P: meta.txt vig_n=$(meta["vig_n"]) but the plugin's vigour quadrature has $(length(VIG_Z)) nodes")
  farea = parse(Int, get(meta, "farea", "0"))
  return (keys=keys_, sp=sp, rec=rec, glob=glob, farea=farea, meta=meta)
end

"""Species symbol -> group "S" | "H" (HBAD v1 species_group.csv = FIA REF_SPECIES SFTWD_HRDWD); unknown symbols and non-group catch-alls -> "H"."""
function _group_of(name::AbstractString, grp_map::AbstractDict{String,String})
  g = HBADPlugin.species_group(name, grp_map)
  return g == "ALL" ? "H" : g
end

"""
    map_species_keys(species_list, keys; grp_map) -> Vector{String}

Key of every Pan species: its own symbol when the set has a row for it (the 10 species with their own parameters), else the pooled group key `_S` / `_H` by REF_SPECIES SFTWD_HRDWD.
"""
function map_species_keys(species_list::AbstractVector{<:AbstractString}, keys_::AbstractVector{<:AbstractString}; grp_map::AbstractDict{String,String})
  have = Set(String.(keys_))
  return String[(s = String(n); (!startswith(s, "_") && s in have) ? s : ("_" * _group_of(s, grp_map))) for n in species_list]
end
