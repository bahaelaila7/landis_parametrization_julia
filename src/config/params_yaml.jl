# Parameter roster ⇄ YAML for every succession plugin (docs/params_yaml.md).
#
# One self-describing file: a `pan_params` header (plugin, format_version, name, …) followed by the FULL roster of the
# params struct (every field, mutable or fixed), laid out for humans:
#   species:    one entry per species        (fields whose value is a per-species vector)
#   ecoregions: one entry per eco cell       (per-eco fields) + `species:` list of the eco×species values
#   global:     everything else              (scalars, matrices such as SUFFICIENT_LIGHT)
# plus the mapping metadata a scenario needs to place NEW plots into the parameter space:
#   species_map: FIA species symbol → parameter species (exact, or the catch-all group the training tiering used)
#   eco_mapping: how a plot's (EPA L3, FIA site class) becomes an eco cell "L3|lu=CELL"
# Import is the exact inverse (fields converted to the struct's declared types); unknown/missing fields are errors, so a
# roster is always complete.
import YAML, JSON3
import DataStructures: OrderedDict

const PARAMS_FORMAT_VERSION = 1
plugin_name(::BiomassSuccessionPlugin.BiomassSuccessionParams) = "biomass"
plugin_name(::PnETPlugin.PnETParams) = "pnet"
plugin_name(::HBADPlugin.HBADParams) = "hbad"
plugin_name(::HBAD2Plugin.HBAD2Params) = "hbad2"
plugin_name(::HBAD3PPlugin.HBAD3PParams) = "hbad3p"
params_struct_type(name::AbstractString) =
  lowercase(name) in ("biomass", "biomasssuccession", "bs") ? BiomassSuccessionPlugin.BiomassSuccessionParams :
  lowercase(name) in ("pnet", "pnet-succession") ? PnETPlugin.PnETParams :
  lowercase(name) in ("hbad", "hbad-succession") ? HBADPlugin.HBADParams :
  lowercase(name) in ("hbad2", "hbad-v2") ? HBAD2Plugin.HBAD2Params :
  lowercase(name) in ("hbad3p", "hbad-3p") ? HBAD3PPlugin.HBAD3PParams :
  error("unknown plugin '$name' in pan_params header (biomass | pnet | hbad | hbad2 | hbad3p)")

const _ID_FIELDS = (:ECO_LIST, :SPECIES_LIST, :ECO_SPECIES_IDS)
# per-eco fields (disambiguates when n_species == n_eco); anything else is classified by shape
const _ECO_FIELDS = Set([:RD_MULT, :MIN_REL_BIOMASS, :MIN_REL_RD, :B_MAX_ECO])
# SUFFICIENT_LIGHT layout: [shade_class + 1][shade_tolerance] (6 site shade classes × 5 tolerance classes)
const SUFFICIENT_LIGHT_CORNER = 0.5   # most shade-tolerant species (5) under the deepest site shade (class 5)

function _field_role(p, f::Symbol)
  v = getfield(p, f); ne = length(p.ECO_LIST); ns = length(p.SPECIES_LIST); esi = p.ECO_SPECIES_IDS
  if v isa AbstractVector && !isempty(v) && first(v) isa AbstractVector
    length(v) == ne && all(length(v[e]) == length(esi[e]) for e in 1:ne) && return :eco_species
    length(v) == ne && f in _ECO_FIELDS && return :eco
    length(v) == ne && f != :SUFFICIENT_LIGHT && return :eco
    return :global
  elseif v isa AbstractVector
    f in _ECO_FIELDS && length(v) == ne && return :eco
    length(v) == ns && return :species
    length(v) == ne && return :eco
    return :global
  end
  :global
end

# shortest decimal that round-trips the value's OWN precision (Float32 "0.204", not 0.20399999618530273)
_yamlval(x::AbstractFloat) = isfinite(x) ? parse(Float64, string(x)) : string(x)
_yamlval(x::Bool) = x
_yamlval(x::Integer) = Int(x)
_yamlval(x::AbstractString) = String(x)
_yamlval(x::Symbol) = String(x)
_yamlval(x::AbstractVector) = [_yamlval(y) for y in x]
_yamlval(x) = string(x)

"""
    params_to_yaml(p; name, description="", region=Dict(), provenance=Dict(), species_map=Dict(), eco_mapping=Dict()) -> String

Full roster of a params struct as YAML text (see the file header for the layout).
"""
function params_to_yaml(p; name::AbstractString, description::AbstractString="", region=Dict{String,Any}(),
                        provenance=Dict{String,Any}(), species_map=Dict{String,String}(), eco_mapping=Dict{String,Any}())
  T = typeof(p); fs = [f for f in fieldnames(T) if !(f in _ID_FIELDS)]
  roles = Dict(f => _field_role(p, f) for f in fs)
  sp = String.(p.SPECIES_LIST); eco = String.(p.ECO_LIST); esi = p.ECO_SPECIES_IDS
  doc = OrderedDict{String,Any}()
  doc["pan_params"] = OrderedDict{String,Any}("plugin" => plugin_name(p), "format_version" => PARAMS_FORMAT_VERSION,
    "name" => String(name), "description" => String(description), "created" => string(Dates.today()),
    "region" => region, "provenance" => provenance)
  doc["eco_mapping"] = eco_mapping
  doc["species_map"] = OrderedDict(k => species_map[k] for k in sort(collect(keys(species_map))))
  doc["global"] = OrderedDict{String,Any}(String(f) => _yamlval(getfield(p, f)) for f in fs if roles[f] === :global)
  doc["species"] = [OrderedDict{String,Any}("name" => sp[k], (String(f) => _yamlval(getfield(p, f)[k]) for f in fs if roles[f] === :species)...)
                    for k in eachindex(sp)]
  doc["ecoregions"] = [begin
      d = OrderedDict{String,Any}("name" => eco[e])
      for f in fs; roles[f] === :eco && (d[String(f)] = _yamlval(getfield(p, f)[e])); end
      d["species"] = [OrderedDict{String,Any}("species" => sp[Int(g)], (String(f) => _yamlval(getfield(p, f)[e][j]) for f in fs if roles[f] === :eco_species)...)
                      for (j, g) in enumerate(esi[e])]
      d
    end for e in eachindex(eco)]
  hdr = """# Pan succession parameter set — plugin: $(plugin_name(p)) (format $(PARAMS_FORMAT_VERSION)). Full roster; edit values, keep structure.
# global.SUFFICIENT_LIGHT[shade_class+1][shade_tolerance]: establishment light factor, site shade class 0..5 × species
# shade tolerance 1..5 (corner = most tolerant species under the deepest shade = $(SUFFICIENT_LIGHT_CORNER)).
# species_map: FIA species symbol → parameter species; eco_mapping: (EPA L3, FIA site class) → eco cell.
"""
  io = IOBuffer(); print(io, hdr); _emit(io, doc, 0); String(take!(io))
end

# Compact YAML emitter for the roster document: mappings in block style, scalar vectors (and vectors of scalar vectors)
# inline (`[0.75, 1.0, …]`), lists of mappings as `- key: value` blocks, strings JSON-quoted (valid YAML).
_scalar(x::AbstractString) = JSON3.write(x)
_scalar(x::Bool) = x ? "true" : "false"
_scalar(x::Integer) = string(x)
_scalar(x::AbstractFloat) = isfinite(x) ? string(x) : (isnan(x) ? ".nan" : x > 0 ? ".inf" : "-.inf")
_scalar(x) = JSON3.write(string(x))
_isscalar(x) = x isa Union{AbstractString,Number,Bool}
_key(k) = (s = String(k); occursin(r"^[A-Za-z_][A-Za-z0-9_]*$", s) ? s : JSON3.write(s))
_inline(x) = _isscalar(x) ? _scalar(x) : "[" * join((_inline(y) for y in x), ", ") * "]"
_isinline(x) = _isscalar(x) || (x isa AbstractVector && all(y -> _isscalar(y) || (y isa AbstractVector && all(_isscalar, y)), x))
function _emit(io, d::AbstractDict, ind::Int)
  isempty(d) && return
  for (k, v) in d
    key = _key(k); pad = " "^ind
    if _isinline(v)
      println(io, pad, key, ": ", _inline(v))
    elseif v isa AbstractDict
      if isempty(v); println(io, pad, key, ": {}"); else; println(io, pad, key, ":"); _emit(io, v, ind + 2); end
    else  # list of mappings
      println(io, pad, key, ":")
      for item in v
        if all(_isscalar, values(item)); println(io, " "^(ind + 2), "- ", _flowmap(item)); continue; end
        first_line = true
        for (k2, v2) in item
          lead = first_line ? " "^ind * "  - " : " "^(ind + 4)
          first_line = false
          if _isinline(v2)
            println(io, lead, _key(k2), ": ", _inline(v2))
          else
            println(io, lead, _key(k2), ":")
            v2 isa AbstractDict ? _emit(io, v2, ind + 6) : _emit_list(io, v2, ind + 6)
          end
        end
      end
    end
  end
end
_flowmap(item) = "{" * join((_key(k) * ": " * _inline(v) for (k, v) in item), ", ") * "}"
function _emit_list(io, v, ind)
  for item in v
    if all(_isscalar, values(item)); println(io, " "^ind, "- ", _flowmap(item)); continue; end
    first_line = true
    for (k2, v2) in item
      lead = first_line ? " "^ind * "- " : " "^(ind + 2)
      first_line = false
      _isinline(v2) ? println(io, lead, _key(k2), ": ", _inline(v2)) : (println(io, lead, _key(k2), ":"); _emit(io, v2, ind + 4))
    end
  end
end

_conv(::Type{T}, x) where {T<:AbstractString} = String(x)
_conv(::Type{Bool}, x) = Bool(x)
_conv(::Type{T}, x) where {T<:Integer} = T(round(Int, x))
_conv(::Type{T}, x) where {T<:AbstractFloat} = x isa AbstractString ? T(parse(Float64, x)) : T(x)
_conv(::Type{Vector{T}}, x) where {T} = T[_conv(T, y) for y in x]
_conv(::Type{T}, x) where {T} = convert(T, x)

"""
    params_from_yaml(text) -> (params, meta)

Inverse of `params_to_yaml`. `meta` = (; plugin, name, description, region, provenance, species_map, eco_mapping, warnings).
"""
function params_from_yaml(text::AbstractString)
  doc = YAML.load(text)
  haskey(doc, "pan_params") || error("not a Pan parameter file: missing the `pan_params` header")
  h = doc["pan_params"]; plugin = String(get(h, "plugin", ""))
  T = params_struct_type(plugin)
  fv = Int(get(h, "format_version", 1)); fv <= PARAMS_FORMAT_VERSION || error("format_version $fv is newer than this Pan ($(PARAMS_FORMAT_VERSION))")
  T === HBAD2Plugin.HBAD2Params && return _hbad2_from_yaml(doc, h)
  T === HBAD3PPlugin.HBAD3PParams && return _hbad3p_from_yaml(doc, h)
  spe = doc["species"]; ecos = doc["ecoregions"]; glob = something(get(doc, "global", nothing), Dict{String,Any}())
  species_list = String[String(s["name"]) for s in spe]; spidx = Dict(s => k for (k, s) in enumerate(species_list))
  eco_list = String[String(e["name"]) for e in ecos]
  ft = Dict(f => fieldtype(T, f) for f in fieldnames(T))
  idtype = ft[:ECO_SPECIES_IDS]; idel = eltype(eltype(idtype))
  esi = [idel[idel(spidx[String(r["species"])]) for r in (something(get(e, "species", nothing), Any[]))] for e in ecos]
  kw = Dict{Symbol,Any}(:ECO_LIST => _conv(ft[:ECO_LIST], eco_list), :SPECIES_LIST => _conv(ft[:SPECIES_LIST], species_list),
                        :ECO_SPECIES_IDS => _conv(idtype, esi))
  warnings = String[]; missing_fields = String[]
  for f in fieldnames(T)
    f in _ID_FIELDS && continue
    k = String(f); FT = ft[f]
    if haskey(glob, k)
      kw[f] = _conv(FT, glob[k])
    elseif all(haskey(s, k) for s in spe) && !isempty(spe)
      kw[f] = _conv(FT, [s[k] for s in spe])
    elseif all(haskey(e, k) for e in ecos) && !isempty(ecos)
      kw[f] = _conv(FT, [e[k] for e in ecos])
    elseif all(all(haskey(r, k) for r in something(get(e, "species", nothing), Any[])) for e in ecos)
      kw[f] = _conv(FT, [[r[k] for r in something(get(e, "species", nothing), Any[])] for e in ecos])
    else
      push!(missing_fields, k)   # left to the struct's default (if it has one) — reported below
    end
  end
  p = try
    T(; kw...)
  catch e
    e isa UndefKeywordError || rethrow()
    error("parameter '$(e.var)' missing from the roster (expected under global, species, ecoregions or ecoregions[].species)")
  end
  isempty(missing_fields) || push!(warnings, "parameters not in the file, defaults used: " * join(missing_fields, ", "))
  if hasproperty(p, :SUFFICIENT_LIGHT)
    sl = p.SUFFICIENT_LIGHT
    (length(sl) == 6 && all(length(r) == 5 for r in sl)) || push!(warnings, "SUFFICIENT_LIGHT should be 6 shade classes × 5 tolerance classes")
    c = sl[end][end]
    c == SUFFICIENT_LIGHT_CORNER || push!(warnings, "SUFFICIENT_LIGHT corner (tolerance 5, shade class 5) is $c; the recommended value is $(SUFFICIENT_LIGHT_CORNER) (no growing space under the deepest shade)")
  end
  smap = Dict{String,String}(String(k) => String(v) for (k, v) in something(get(doc, "species_map", nothing), Dict()))
  for (k, v) in smap; haskey(spidx, v) || push!(warnings, "species_map: $k → $v, but $v is not a parameter species"); end
  meta = (; plugin=plugin_name(p), name=String(get(h, "name", "")), description=String(get(h, "description", "")),
          region=get(h, "region", Dict()), provenance=get(h, "provenance", Dict()), species_map=smap,
          eco_mapping=something(get(doc, "eco_mapping", nothing), Dict{String,Any}()), warnings=warnings)
  p, meta
end

"""Exact-equality check of two params structs (all fields); returns a vector of differing field names."""
function params_diff(a, b)
  typeof(a) == typeof(b) || return [:TYPE]
  [f for f in fieldnames(typeof(a)) if !isequal(getfield(a, f), getfield(b, f))]
end


# ---------------------------------------------------------------------------------------------------------------------
# hbad2 roster (HBAD v2; the model name is pan_params.name, e.g. v2g, v2h). The parameters are per species KEY (the 10 own species + the pooled `_S` / `_H` groups), not per ecoregion:
#   global:     mD2 mA pH pD pM form mSD sigma_v
#   species:    one row per KEY: name (= the key), the 15 per-key parameters (SP_PARAM_NAMES) and the recruitment parameters (rho0 rhoS rhoL D0 H0 A0)
#   ecoregions: the eco cells of the region (for the scenario's eco mapping); each lists ALL keys in order, so the eco-local species index IS the key index
# species_map: FIA symbol -> key (own species, else `_S` / `_H` by SFTWD_HRDWD), as the tree loader resolves species. Unknown keys / missing parameters are errors.
# ---------------------------------------------------------------------------------------------------------------------
const _H2_GLOBAL = ("mD2", "mA", "pH", "pD", "pM", "form", "mSD", "sigma_v")
const _H2_REC = ("rho0", "rhoS", "rhoL", "D0", "H0", "A0")

function params_to_yaml(p::HBAD2Plugin.HBAD2Params; name::AbstractString, description::AbstractString="", region=Dict{String,Any}(),
                        provenance=Dict{String,Any}(), species_map=Dict{String,String}(), eco_mapping=Dict{String,Any}())
  sp = String.(p.SPECIES_LIST); eco = String.(p.ECO_LIST); g = p.G
  p.KEYS == sp || error("hbad2 roster: the species of a parameter set must be the v2 keys themselves (got $(p.KEYS) for $(sp))")
  all(ids -> Int.(ids) == collect(1:length(sp)), p.ECO_SPECIES_IDS) || error("hbad2 roster: every ecoregion must list all keys in order")
  doc = OrderedDict{String,Any}()
  doc["pan_params"] = OrderedDict{String,Any}("plugin" => "hbad2", "format_version" => PARAMS_FORMAT_VERSION,
    "name" => String(name), "description" => String(description), "created" => string(Dates.today()),
    "region" => region, "provenance" => provenance)
  doc["eco_mapping"] = eco_mapping
  doc["species_map"] = OrderedDict(k => species_map[k] for k in sort(collect(keys(species_map))))
  doc["global"] = OrderedDict{String,Any}("mD2" => _yamlval(g.mD2), "mA" => _yamlval(g.mA), "pH" => _yamlval(g.pH), "pD" => _yamlval(g.pD), "pM" => _yamlval(g.pM),
                                          "form" => g.form, "mSD" => _yamlval(g.mSD), "sigma_v" => _yamlval(g.σv))
  doc["species"] = [begin
      d = OrderedDict{String,Any}("name" => sp[k])
      for f in HBAD2Plugin.SP_PARAM_NAMES; d[String(f)] = _yamlval(getfield(p.SP[k], f)); end
      for f in _H2_REC; d[f] = _yamlval(getfield(p.REC[k], Symbol(f))); end
      d
    end for k in eachindex(sp)]
  doc["ecoregions"] = [OrderedDict{String,Any}("name" => e, "species" => [OrderedDict{String,Any}("species" => s) for s in sp]) for e in eco]
  hdr = """# Pan succession parameter set - plugin: hbad2 (HBAD v2; format $(PARAMS_FORMAT_VERSION)). Parameters are per species KEY (10 own species + the pooled _S / _H groups).
# species_map: FIA species symbol -> key; eco_mapping: (EPA L3, FIA site class) -> eco cell (the parameters do not depend on the cell).
"""
  io = IOBuffer(); print(io, hdr); _emit(io, doc, 0); String(take!(io))
end

function _hbad2_from_yaml(doc, h)
  haskey(doc, "species") && haskey(doc, "ecoregions") || error("hbad2 roster: needs `species` and `ecoregions`")
  spe = doc["species"]; ecos = doc["ecoregions"]; glob = something(get(doc, "global", nothing), Dict{String,Any}())
  for k in keys(glob); String(k) in _H2_GLOBAL || error("hbad2 roster: unknown global parameter '$k' (expected $(join(_H2_GLOBAL, ", ")))"); end
  for k in _H2_GLOBAL; haskey(glob, k) || error("hbad2 roster: global parameter '$k' missing"); end
  names = String[String(s["name"]) for s in spe]
  allunique(names) || error("hbad2 roster: duplicate species keys")
  spnames = Set(String.(HBAD2Plugin.SP_PARAM_NAMES)); recnames = Set(_H2_REC)
  for s in spe
    for k in keys(s); String(k) == "name" || String(k) in spnames || String(k) in recnames || error("hbad2 roster: unknown parameter '$k' of species $(s["name"])"); end
    for k in union(spnames, recnames); haskey(s, k) || error("hbad2 roster: parameter '$k' missing for species $(s["name"])"); end
  end
  sp = HBAD2Plugin.SpP[HBAD2Plugin.SpP([Float64(s[String(f)]) for f in HBAD2Plugin.SP_PARAM_NAMES]) for s in spe]
  rec = HBAD2Plugin.RecP[HBAD2Plugin.RecP((Float64(s[f]) for f in _H2_REC)...) for s in spe]
  G = HBAD2Plugin.GlobP(Float64(glob["mD2"]), Float64(glob["mA"]), Float64(glob["pH"]), Float64(glob["pD"]), Float64(glob["pM"]),
                        round(Int, glob["form"]), Float64(glob["mSD"]), Float64(glob["sigma_v"]))
  eco_list = String[String(e["name"]) for e in ecos]
  for e in ecos
    lst = String[String(r["species"]) for r in something(get(e, "species", nothing), Any[])]
    lst == names || error("hbad2 roster: ecoregion $(e["name"]) must list all species keys in the order of `species` (the eco-local species index is the key index)")
  end
  esi = [UIntType[UIntType(k) for k in 1:length(names)] for _ in ecos]
  p = HBAD2Plugin.HBAD2Params(eco_list, names, esi, copy(names), sp, rec, G)
  smap = Dict{String,String}(String(k) => String(v) for (k, v) in something(get(doc, "species_map", nothing), Dict()))
  for (k, v) in smap; v in names || error("hbad2 roster: species_map $k -> $v, but $v is not a species key"); end
  meta = (; plugin="hbad2", name=String(get(h, "name", "")), description=String(get(h, "description", "")),
          region=get(h, "region", Dict()), provenance=get(h, "provenance", Dict()), species_map=smap,
          eco_mapping=something(get(doc, "eco_mapping", nothing), Dict{String,Any}()), warnings=String[])
  p, meta
end

# ---------------------------------------------------------------------------------------------------------------------
# hbad3p roster (HBAD3P process model; docs/hbad3p_plugin.md). The parameters are per species KEY (the species with their own parameters + the pooled `_S` / `_H` groups), not per ecoregion:
#   global:     the 32 GlobPP parameters (ASCII names, HBAD3PPlugin.GLOB_YAML), the recruitment globals lam0 lam1 eps0, `farea` (density convention of the fit) and `vig_n` (vigour quadrature nodes, must be 5)
#   species:    one row per KEY: name (= the key), lneps (ln of the growth efficiency ε), Hmax, ST, longevity, lnF (fecundity)
#   ecoregions: the eco cells of the region (scenario eco mapping); each lists ALL keys in order, so the eco-local species index IS the key index
#   data:       `plot_normal_water`: csv (st,u,c,p,wn) of the plot-normal water index AET/PET (path relative to the repository root); the scenario engine sets W = Wn = the plot's normal
# species_map: FIA symbol -> key (own species, else `_S` / `_H` by SFTWD_HRDWD). Unknown keys / missing parameters are errors.
# ---------------------------------------------------------------------------------------------------------------------
const _H3_REC = ("lam0", "lam1", "eps0")
const _H3_SPECIES = ("lneps", "Hmax", "ST", "longevity", "lnF")
const _H3_GLOBAL = (HBAD3PPlugin.GLOB_YAML..., _H3_REC..., "farea", "vig_n")

function params_to_yaml(p::HBAD3PPlugin.HBAD3PParams; name::AbstractString, description::AbstractString="", region=Dict{String,Any}(),
                        provenance=Dict{String,Any}(), species_map=Dict{String,String}(), eco_mapping=Dict{String,Any}(), data=Dict{String,Any}())
  sp = String.(p.SPECIES_LIST); eco = String.(p.ECO_LIST); g = p.G
  p.KEYS == sp || error("hbad3p roster: the species of a parameter set must be the keys themselves (got $(p.KEYS) for $(sp))")
  all(ids -> Int.(ids) == collect(1:length(sp)), p.ECO_SPECIES_IDS) || error("hbad3p roster: every ecoregion must list all keys in order")
  doc = OrderedDict{String,Any}()
  doc["pan_params"] = OrderedDict{String,Any}("plugin" => "hbad3p", "format_version" => PARAMS_FORMAT_VERSION,
    "name" => String(name), "description" => String(description), "created" => string(Dates.today()),
    "region" => region, "provenance" => provenance)
  doc["eco_mapping"] = eco_mapping
  doc["species_map"] = OrderedDict(k => species_map[k] for k in sort(collect(keys(species_map))))
  gl = OrderedDict{String,Any}()
  for (f, y) in zip(HBAD3PPlugin.GLOB_NAMES, HBAD3PPlugin.GLOB_YAML); gl[y] = _yamlval(getfield(g, f)); end
  gl["lam0"] = _yamlval(p.REC.λ0); gl["lam1"] = _yamlval(p.REC.λ1); gl["eps0"] = _yamlval(p.REC.ε0)
  gl["farea"] = p.FAREA; gl["vig_n"] = length(HBAD3PPlugin.VIG_Z)
  doc["global"] = gl
  doc["data"] = data
  doc["species"] = [OrderedDict{String,Any}("name" => sp[k], "lneps" => _yamlval(p.SP[k].lnε), "Hmax" => _yamlval(p.SP[k].Hmax), "ST" => _yamlval(p.SP[k].ST),
                                            "longevity" => _yamlval(p.SP[k].longevity), "lnF" => _yamlval(p.SP[k].lnF)) for k in eachindex(sp)]
  doc["ecoregions"] = [OrderedDict{String,Any}("name" => e, "species" => [OrderedDict{String,Any}("species" => s) for s in sp]) for e in eco]
  hdr = """# Pan succession parameter set - plugin: hbad3p (HBAD3P process model; format $(PARAMS_FORMAT_VERSION)). Parameters are per species KEY (own species + the pooled _S / _H groups).
# species_map: FIA species symbol -> key; eco_mapping: (EPA L3, FIA site class) -> eco cell (the parameters do not depend on the cell); data.plot_normal_water: plot-normal AET/PET table.
"""
  io = IOBuffer(); print(io, hdr); _emit(io, doc, 0); String(take!(io))
end

function _hbad3p_from_yaml(doc, h)
  haskey(doc, "species") && haskey(doc, "ecoregions") || error("hbad3p roster: needs `species` and `ecoregions`")
  spe = doc["species"]; ecos = doc["ecoregions"]; glob = something(get(doc, "global", nothing), Dict{String,Any}())
  for k in keys(glob); String(k) in _H3_GLOBAL || error("hbad3p roster: unknown global parameter '$k' (expected $(join(_H3_GLOBAL, ", ")))"); end
  for k in _H3_GLOBAL; haskey(glob, k) || error("hbad3p roster: global parameter '$k' missing"); end
  Int(glob["vig_n"]) == length(HBAD3PPlugin.VIG_Z) || error("hbad3p roster: vig_n = $(glob["vig_n"]), but the plugin's vigour quadrature has $(length(HBAD3PPlugin.VIG_Z)) nodes")
  Int(glob["farea"]) in (0, 1) || error("hbad3p roster: farea must be 0 or 1")
  names = String[String(s["name"]) for s in spe]
  allunique(names) || error("hbad3p roster: duplicate species keys")
  spnames = Set(_H3_SPECIES)
  for s in spe
    for k in keys(s); String(k) == "name" || String(k) in spnames || error("hbad3p roster: unknown parameter '$k' of species $(s["name"])"); end
    for k in spnames; haskey(s, k) || error("hbad3p roster: parameter '$k' missing for species $(s["name"])"); end
  end
  sp = HBAD3PPlugin.SpPP[HBAD3PPlugin.SpPP(Float64(s["lneps"]), Float64(s["Hmax"]), Float64(s["ST"]), Float64(s["longevity"]), Float64(s["lnF"])) for s in spe]
  G = HBAD3PPlugin.GlobPP(Float64[glob[y] for y in HBAD3PPlugin.GLOB_YAML])
  rec = HBAD3PPlugin.RecGP(Float64(glob["lam0"]), Float64(glob["lam1"]), Float64(glob["eps0"]))
  eco_list = String[String(e["name"]) for e in ecos]
  for e in ecos
    lst = String[String(r["species"]) for r in something(get(e, "species", nothing), Any[])]
    lst == names || error("hbad3p roster: ecoregion $(e["name"]) must list all species keys in the order of `species` (the eco-local species index is the key index)")
  end
  esi = [UIntType[UIntType(k) for k in 1:length(names)] for _ in ecos]
  p = HBAD3PPlugin.HBAD3PParams(eco_list, names, esi, copy(names), sp, rec, G, Int(glob["farea"]))
  smap = Dict{String,String}(String(k) => String(v) for (k, v) in something(get(doc, "species_map", nothing), Dict()))
  for (k, v) in smap; v in names || error("hbad3p roster: species_map $k -> $v, but $v is not a species key"); end
  meta = (; plugin="hbad3p", name=String(get(h, "name", "")), description=String(get(h, "description", "")),
          region=get(h, "region", Dict()), provenance=get(h, "provenance", Dict()), species_map=smap,
          eco_mapping=something(get(doc, "eco_mapping", nothing), Dict{String,Any}()), warnings=String[],
          data=Dict{String,Any}(String(k) => v for (k, v) in something(get(doc, "data", nothing), Dict())))
  p, meta
end
