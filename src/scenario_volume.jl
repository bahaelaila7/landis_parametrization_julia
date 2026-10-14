# Empirical cohort volumes for scenario output (docs/volume_equations.md, data/volume_eqs_r4.csv, tools/fit_volume_eqs.py).
# Per tree (FIA Region 4 live trees, imperial units):  v = cf · exp(b0) · ba^b1 · H^b2   (ba = 0.005454154·D², ft²; H ft)
#   tcuft : gross total-stem cubic ft (FIA VOLTSGRS)
#   bdft  : gross board feet, International ¼-inch (FIA VOLBFGRS; the Scribner column is empty in this FIADB), only for
#           D ≥ min_dbh (9 in softwood, 11 in hardwood), × p(sawtimber-qualifying) = logistic(sq_a + sq_c · ln D)
# A cohort contributes TPA × v evaluated at its QMD and height. Equation chosen per parameter species: its FIA species, else
# that species' FIA species group, else softwood/hardwood; `_GRP_<n>_*` groups use FIA species group n; `_S*`/`_H*` use S/H.
const SCEN_VOLEQ = Ref(get(ENV, "PAN_VOLUME_EQS", normpath(joinpath(@__DIR__, "..", "data", "volume_eqs_r4.csv"))))

struct VolEq
  b0::Float64; b1::Float64; b2::Float64; cf::Float64
  min_dbh::Float64; sq_a::Float64; sq_c::Float64
end
const _VOLEQ_TABLE = Ref{Union{Nothing,Dict{Tuple{String,String,String},VolEq}}}(nothing)
const _VOLEQ_BY_SET = Dict{String,Vector{NTuple{2,VolEq}}}()      # ParamSet.id → per global species (tcuft, bdft)
const _SPGRP = Ref{Union{Nothing,Dict{Int,Int}}}(nothing)

function _voleq_table()
  _VOLEQ_TABLE[] === nothing || return _VOLEQ_TABLE[]
  isfile(SCEN_VOLEQ[]) || error("volume equation table not found: $(SCEN_VOLEQ[]) (set PAN_VOLUME_EQS)")
  df = CSV.read(SCEN_VOLEQ[], DataFrame; types=Dict(:key => String))
  num(x) = ismissing(x) ? NaN : Float64(x)
  d = Dict{Tuple{String,String,String},VolEq}()
  for r in eachrow(df)
    d[(String(r.model), String(r.level), String(r.key))] =
      VolEq(r.b0, r.b1, r.b2, r.cf, num(r.min_dbh), num(r.sq_a), num(r.sq_c))
  end
  _VOLEQ_TABLE[] = d
end

function _spgrp_codes()
  _SPGRP[] === nothing || return _SPGRP[]
  db = DuckDB.DB(SCEN_FIA_DB[]; readonly=true); con = DuckDB.DBInterface.connect(db)
  d = Dict{Int,Int}()
  try
    # eastern species groups = TREE.SPGRPCD in Region 4 (what the equations were fitted on)
    for r in DuckDB.DBInterface.execute(con, "SELECT SPCD c, E_SPGRPCD g FROM REF_SPECIES WHERE E_SPGRPCD IS NOT NULL")
      d[Int(r.c)] = Int(r.g)
    end
  finally
    DuckDB.DBInterface.close!(con); DuckDB.close(db)
  end
  _SPGRP[] = d
end

function _voleq_for(sp::AbstractString, spcd::Int, model::String)
  T = _voleq_table()
  look(level, key) = get(T, (model, level, string(key)), nothing)
  s = uppercase(String(sp))
  if spcd > 0
    e = look("spcd", spcd); e === nothing || return e
    g = get(_spgrp_codes(), spcd, 0); e = g > 0 ? look("spgrpcd", g) : nothing; e === nothing || return e
    return look("major", spcd < 300 ? "S" : "H")
  end
  m = match(r"^_GRP_(\d+)", s)
  if m !== nothing
    g = parse(Int, m.captures[1]); e = look("spgrpcd", g); e === nothing || return e
    return look("major", g <= 24 ? "S" : "H")            # FIA species groups 1–24 are softwoods
  end
  look("major", startswith(s, "_S") ? "S" : "H")
end

"(tcuft, bdft) equations per global species of a parameter set (cached by set)."
function volume_eqs(ps::ParamSet)
  get!(_VOLEQ_BY_SET, ps.id) do
    [(_voleq_for(String(s), ps.spcd[String(s)], "tcuft"), _voleq_for(String(s), ps.spcd[String(s)], "bdft")) for s in ps.params.SPECIES_LIST]
  end
end

"Cohort volume per acre from trees/ha `t`, QMD `dq` (cm) and height `h` (m): (total cubic ft³/ac, board feet/ac)."
function cohort_volume(eqs::NTuple{2,VolEq}, t::Real, dq::Real, h::Real)
  (t > 0 && dq > 0 && h > 0) || return 0.0, 0.0
  D = dq / 2.54; H = h * 3.28084; ba = 0.005454154 * D^2; tpa = t * SC_HA_PER_AC
  ec, eb = eqs
  v = tpa * ec.cf * exp(ec.b0) * ba^ec.b1 * H^ec.b2
  b = 0.0
  if D >= eb.min_dbh
    p = isfinite(eb.sq_a) ? 1 / (1 + exp(-(eb.sq_a + eb.sq_c * log(D)))) : 1.0
    b = tpa * p * eb.cf * exp(eb.b0) * ba^eb.b1 * H^eb.b2
  end
  v, b
end
