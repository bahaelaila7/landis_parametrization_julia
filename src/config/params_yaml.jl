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
plugin_name(::HBAD2Plugin.HBAD2Params) = error("hbad2 parameters are the fixed v2g files (results/hbad2/*.csv, stage 1); no yaml roster")
params_struct_type(name::AbstractString) =
  lowercase(name) in ("biomass", "biomasssuccession", "bs") ? BiomassSuccessionPlugin.BiomassSuccessionParams :
  lowercase(name) in ("pnet", "pnet-succession") ? PnETPlugin.PnETParams :
  lowercase(name) in ("hbad", "hbad-succession") ? HBADPlugin.HBADParams :
  error("unknown plugin '$name' in pan_params header (biomass | pnet | hbad)")

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
