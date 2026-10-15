# HBAD fixed empirical priors: per GLOBAL species, resolved from results/hbad/{priors,translation_coefs}.csv by species name.
#   exact symbol if it has its own row, else the softwood/hardwood pooled row ("_S"/"_H"), else "_ALL" (same rule as
#   tools/hbad_translate_lib.jl::hb_key). Pan catch-all names: "_S*" -> _S, "_H*" -> _H, "_GRP_<n>_*" -> _S if n <= 24 else _H.
# Everything here is plain isbits data (SpPrior) so the annual kernel is allocation-free and dynamic-dispatch free.

const HB_LB = 453.592 / 4046.86            # lb/acre -> g/m2 factor of Pan (per tree: lb -> g/m2 at 1 tree/acre)
const HB_ACRE2HA = 2.47105
const HB_MIN_TPA_HA = 6.01805 * HB_ACRE2HA  # one macroplot tree record (trees/ha) -- smallest density of the FIA cohort table
const HB_X1C = 6.0                          # centring of ln(AGB g/m2) in the translation regression
const HB_X2C = 3.3                          # centring of ln(age)
const HB_DQ_MIN = 0.25                      # cm, floor of the quadratic mean diameter
const HB_DQ_MAX = 170.0                     # cm, ceiling (as hbad_translate)
const HB_REINEKE = 1.605                    # Reineke self-thinning slope
const HB_REINEKE_REF = 25.4                 # cm (10 in)

"""All fixed per-species coefficients (Float64 isbits). One entry per GLOBAL species id."""
struct SpPrior
  # diameter growth  dD = exp(g_b0 + g_b1 ln(D/20) + g_b2 D/20 + g_b3 ln(age/30) - g_k BAL/10)   [cm/yr]
  g_b0::Float64; g_b1::Float64; g_b2::Float64; g_b3::Float64; g_k::Float64
  # annual mortality hazard  exp(m_c0 + m_c1 ln(D/20) + m_c2 D/20 + m_c3 RD + m_c4 BAL/10 + m_c5 RD ln(D/20))
  m_c0::Float64; m_c1::Float64; m_c2::Float64; m_c3::Float64; m_c4::Float64; m_c5::Float64
  # Chapman-Richards height-diameter  H = 1.3 + a (1-exp(-b D))^c  (priors.csv hd_*; used for height growth)
  hd_a::Float64; hd_b::Float64; hd_c::Float64
  # Chapman-Richards of the translation (translation_coefs.csv hcr; used for the initial mean height)
  hc_a::Float64; hc_b::Float64; hc_c::Float64
  # cohort mass-QMD relation  ln m_lb = mq_a + mq_b ln(QMD_in)
  mq_a::Float64; mq_b::Float64
  # ln TPA_ha = t0 + t1 x1 + t2 x2 + t3 x1^2 + t4 x1 x2 + t5 x2^2   (x1 = ln AGB - 6, x2 = ln age - 3.3)
  t0::Float64; t1::Float64; t2::Float64; t3::Float64; t4::Float64; t5::Float64
  # Reineke SDImax quantiles (metric, trees/ha at 25.4 cm)
  sdi_p90::Float64; sdi_p95::Float64; sdi_p99::Float64
end

"""Fixed priors for every global species of the run (index = global species id) + provenance for logging."""
struct HBADPriors
  species_list::Vector{String}
  trans_key::Vector{String}     # translation_coefs.csv row used
  prior_key::Vector{String}     # priors.csv row used (SDImax / allometry)
  growth_key::Vector{String}    # row used for the growth block (differs from prior_key when the sanity guard fired)
  mort_key::Vector{String}
  sp::Vector{SpPrior}
end

"""Eco-local -> global species id table of the run (flat CSR); needed by `add_cohort!` which only sees (eco, local species)."""
struct EcoGids
  off::Vector{Int32}   # eco e owns gid[off[e]+1 : off[e+1]]
  gid::Vector{Int32}
end

"""EcoGids of an `eco_species_ids` table (eco e -> its global species ids)."""
function eco_gids(eco_species_ids::AbstractVector)
  off = Int32[0]
  gid = Int32[]
  for ids in eco_species_ids
    append!(gid, Int32.(ids))
    push!(off, Int32(length(gid)))
  end
  return EcoGids(off, gid)
end
Base.:(==)(a::EcoGids, b::EcoGids) = a.off == b.off && a.gid == b.gid
@inline function global_species(g::EcoGids, eco::Integer, esp::Integer)::Int
  return @inbounds Int(g.gid[Int(g.off[eco]) + Int(esp)])
end

# ---------------------------------------------------------------------------------------------------------------------
# Name -> softwood/hardwood group and table keys
# ---------------------------------------------------------------------------------------------------------------------
"""
    species_group(name, grp_map) -> "S" | "H" | "ALL"

Pan catch-all names: `_S*` softwood, `_H*` hardwood, `_GRP_<n>_*` FIA SPGRPCD n (1-24 softwood, >= 25 hardwood), other `_*` -> ALL.
Plain symbols use `species_group.csv` (SFTWD_HRDWD); unknown symbols default to hardwood (as `hb_group_guess`).
"""
function species_group(name::AbstractString, grp_map::AbstractDict{String,String})
  s = String(name)
  if startswith(s, "_")
    startswith(s, "_S") && return "S"
    startswith(s, "_H") && return "H"
    m = match(r"^_GRP_(\d+)", s)
    m === nothing && return "ALL"
    return parse(Int, m.captures[1]) <= 24 ? "S" : "H"
  end
  return get(grp_map, s, "H")
end
resolve_key(have::AbstractSet{String}, name::AbstractString, grp::AbstractString) =
  name in have ? String(name) : (("_" * grp) in have ? "_" * grp : "_ALL")

_f(df, r, c) = (v = df[r, c]; ismissing(v) ? NaN : Float64(v))

"""Growth block sanity: increments at D in {5,10,30,60} cm, age 40, BAL 0 must be in (0.005, `gmax`]."""
function growth_block_sane(b0, b1, b2, b3, gmax)
  all(isfinite, (b0, b1, b2, b3)) || return false
  for D in (5.0, 10.0, 30.0, 60.0)
    g = exp(b0 + b1 * log(D / 20) + b2 * D / 20 + b3 * log(40 / 30))
    (isfinite(g) && g > 0.005 && g <= gmax) || return false
  end
  return true
end
"""Mortality block sanity: hazard over D in {5,20,50} cm, RD in {0.3,0.8,1.2}, BAL in {0,10,30} m2/ha must be in (1e-4, hmax]."""
function mort_block_sane(c, hmax)
  all(isfinite, c) || return false
  for D in (5.0, 20.0, 50.0), rd in (0.3, 0.8, 1.2), bal in (0.0, 10.0, 30.0)
    l = log(D / 20)
    h = exp(c[1] + c[2] * l + c[3] * D / 20 + c[4] * rd + c[5] * bal / 10 + c[6] * rd * l)
    (isfinite(h) && h > 1e-4 && h <= hmax) || return false
  end
  return true
end

const _TPA_TERMS = ("1", "x1", "x2", "x1sq", "x1x2", "x2sq")

"""
    load_priors(species_list; priors_csv, coefs_csv, selection_csv, group_csv, gmax=3.0, hmax=0.3) -> HBADPriors

Resolve the fixed coefficients of every species. Per-species blocks are taken from the species' own row, else the pooled group row.
Sanity guard (documented deviation): a species' OWN growth / mortality block is replaced by the group block when it is incomplete
(missing) or implies absurd rates (growth > `gmax` cm/yr or hazard > `hmax` /yr on the sanity grid), e.g. SAPA growth explodes for D < 10 cm.
"""
function load_priors(species_list::AbstractVector{<:AbstractString}; priors_csv::AbstractString, coefs_csv::AbstractString,
                     selection_csv::AbstractString=joinpath(dirname(coefs_csv), "translation_selection.csv"),
                     group_csv::AbstractString=joinpath(dirname(coefs_csv), "species_group.csv"),
                     gmax::Float64=3.0, hmax::Float64=0.3)
  for f in (priors_csv, coefs_csv, selection_csv, group_csv)
    isfile(f) || error("HBAD: file not found: $f")
  end
  P = CSV.read(priors_csv, DataFrame)
  prow = Dict{String,Int}(String(k) => i for (i, k) in enumerate(P.key))
  Cdf = CSV.read(coefs_csv, DataFrame)
  C = Dict{String,Dict{String,Dict{String,Float64}}}()
  for r in eachrow(Cdf)
    m = get!(get!(C, String(r.key), Dict{String,Dict{String,Float64}}()), String(r.model), Dict{String,Float64}())
    m[String(r.term)] = r.value
  end
  sel = Dict{String,String}(String(r.item) => String(r.choice) for r in eachrow(CSV.read(selection_csv, DataFrame)))
  (get(sel, "structure", "") == "A" && get(sel, "mass", "") == "mq" && get(sel, "hmodel", "") == "cr") ||
    error("HBAD: translation_selection.csv must select structure=A (TPA-first), mass=mq, hmodel=cr; got $(sel)")
  tpa_model = "tpa_" * get(sel, "v_tpa", "")
  grp_map = Dict{String,String}(String(r.species) => String(r.grp) for r in eachrow(CSV.read(group_csv, DataFrame)))
  ckeys = Set(keys(C)); pkeys = Set(keys(prow))

  n = length(species_list)
  sp = Vector{SpPrior}(undef, n)
  tk = Vector{String}(undef, n); pk = Vector{String}(undef, n); gk = Vector{String}(undef, n); mk = Vector{String}(undef, n)
  gblock(row) = (_f(P, row, :g_b0), _f(P, row, :g_b1_lnD), _f(P, row, :g_b2_D), _f(P, row, :g_b3_lnage), _f(P, row, :g_k_bal))
  mblock(row) = (_f(P, row, :m_c0), _f(P, row, :m_c1_lnD), _f(P, row, :m_c2_D), _f(P, row, :m_c3_RD), _f(P, row, :m_c4_BAL), _f(P, row, :m_c5_RDxlnD))
  for (g, name) in enumerate(species_list)
    grp = species_group(name, grp_map)
    ck = resolve_key(ckeys, name, grp)
    pkey = resolve_key(pkeys, name, grp)
    gkey = ("_" * grp) in pkeys ? "_" * grp : "_ALL"   # pooled group row (the fallback block)
    own = prow[pkey]; grow = prow[gkey]
    gb = gblock(own);  growth_key = pkey
    if !growth_block_sane(gb[1], gb[2], gb[3], gb[4], gmax) || !isfinite(gb[5])
      gb = gblock(grow); growth_key = gkey
    end
    mb = mblock(own);  mort_key = pkey
    if !mort_block_sane(mb, hmax)
      mb = mblock(grow); mort_key = gkey
    end
    hd = (_f(P, own, :hd_a), _f(P, own, :hd_b), _f(P, own, :hd_c))
    all(isfinite, hd) || (hd = (_f(P, grow, :hd_a), _f(P, grow, :hd_b), _f(P, grow, :hd_c)))
    tm = C[ck][tpa_model]; mq = C[ck]["mq"]; hc = C[ck]["hcr"]
    for t in keys(tm)
      t in _TPA_TERMS || t in ("n", "sigma") || error("HBAD: unsupported TPA regression term '$t' in $tpa_model (no SI/planted in the plugin)")
    end
    tg(t) = get(tm, t, 0.0)
    sp[g] = SpPrior(gb[1], gb[2], gb[3], gb[4], gb[5], mb[1], mb[2], mb[3], mb[4], mb[5], mb[6],
      hd[1], hd[2], hd[3], hc["a"], hc["b"], hc["c"], mq["a"], mq["b"],
      tg("1"), tg("x1"), tg("x2"), tg("x1sq"), tg("x1x2"), tg("x2sq"),
      _f(P, own, :sdimax_p90_all), _f(P, own, :sdimax_p95_all), _f(P, own, :sdimax_p99_all))
    tk[g] = ck; pk[g] = pkey; gk[g] = growth_key; mk[g] = mort_key
  end
  return HBADPriors(String.(species_list), tk, pk, gk, mk, sp)
end

# ---------------------------------------------------------------------------------------------------------------------
# Allometry and translation (AGB, age) -> (tph, dq, h)
# ---------------------------------------------------------------------------------------------------------------------
"""Mean tree dry mass (lb) of a cohort with quadratic mean diameter `dq` (cm): ln m = mq_a + mq_b ln(dq/2.54)."""
@inline mean_tree_lb(p::SpPrior, dq::Float64) = exp(p.mq_a + p.mq_b * log(dq / 2.54))

"""Cohort AGB (g/m2) from density `tph` (trees/ha) and quadratic mean diameter `dq` (cm): exact inverse of `translate`."""
@inline agb_of(p::SpPrior, tph::Float64, dq::Float64) = tph / HB_ACRE2HA * mean_tree_lb(p, dq) * HB_LB

@inline chapman_richards(a::Float64, b::Float64, c::Float64, D::Float64) = a * (1.0 - exp(-b * D))^c

"""
    translate(p, age, agb) -> (tph, dq, h)

The selected translation of tools/hbad_translate_lib.jl (structure A, TPA-first; mass-QMD `mq`; Chapman-Richards height; no SI, natural stand):
1. tph = max(exp(quadratic regression in x1 = ln AGB - 6, x2 = ln age - 3.3), one macroplot tree);
2. m_lb = AGB / (tph/2.47105 * LB); QMD_in = exp((ln m_lb - mq_a)/mq_b); dq = 2.54 QMD_in;
3. h = 1.3 + hc_a (1 - exp(-hc_b dq))^hc_c.
`dq` is limited to [0.25, 170] cm (the lib clamps to [2.54, 170]); when a limit binds, `tph` is re-solved so that AGB stays exactly consistent
(identical to the lib wherever the lib does not clamp, i.e. AGB >~ 0.5 g/m2).
"""
@inline function translate(p::SpPrior, age::Float64, agb::Float64)
  lnagb = log(max(agb, 1e-6)); lnage = log(max(age, 1.0))
  x1 = lnagb - HB_X1C; x2 = lnage - HB_X2C
  tph = max(exp(p.t0 + p.t1 * x1 + p.t2 * x2 + p.t3 * x1 * x1 + p.t4 * x1 * x2 + p.t5 * x2 * x2), HB_MIN_TPA_HA)
  mlb = agb / (tph / HB_ACRE2HA * HB_LB)
  dq = exp((log(mlb) - p.mq_a) / p.mq_b) * 2.54
  if !(dq >= HB_DQ_MIN && dq <= HB_DQ_MAX)
    dq = clamp(isfinite(dq) ? dq : HB_DQ_MIN, HB_DQ_MIN, HB_DQ_MAX)
    tph = agb * HB_ACRE2HA / (mean_tree_lb(p, dq) * HB_LB)
  end
  h = 1.3 + chapman_richards(p.hc_a, p.hc_b, p.hc_c, dq)
  return tph, dq, h
end
