# Empirical cohort translation: (species, age, AGB g/m2 [, site index m, planted]) -> (TPA/ha, QMD cm, BA m2/ha, H_mean m, H_dom m).
# Coefficients are read from results/hbad/translation_coefs.csv (long format key,model,term,value) and
# results/hbad/translation_selection.csv (selected model structure).  Stand-alone: only needs CSV + DataFrames.
#   include("tools/hbad_translate_lib.jl");  hbad_translate("PITA", 25, 300.0; si=24.0)
using CSV, DataFrames

const HB_LB = 453.592 / 4046.86
const HB_ACRE2HA = 2.47105
const HB_MIN_TPA_HA = 6.01805 * HB_ACRE2HA   # one macroplot tree record
const HB_X1C = 6.0      # centring of ln(AGB g/m2)
const HB_X2C = 3.3      # centring of ln(age)

hb_term(t::AbstractString, lnagb, lnage, si, pl) =
    t == "1" ? 1.0 :
    t == "x1" ? lnagb - HB_X1C :
    t == "x2" ? lnage - HB_X2C :
    t == "x1sq" ? (lnagb - HB_X1C)^2 :
    t == "x2sq" ? (lnage - HB_X2C)^2 :
    t == "x1x2" ? (lnagb - HB_X1C) * (lnage - HB_X2C) :
    t == "si" ? (si - 25.0) :
    t == "pl" ? pl : error("unknown term $t")

const HBCoefs = Dict{String,Dict{String,Dict{String,Float64}}}   # key -> model -> term -> value

function hb_load_coefs(path::AbstractString)
    df = CSV.read(path, DataFrame)
    C = HBCoefs()
    for r in eachrow(df)
        m = get!(get!(C, String(r.key), Dict{String,Dict{String,Float64}}()), String(r.model), Dict{String,Float64}())
        m[String(r.term)] = r.value
    end
    return C
end
function hb_load_selection(path::AbstractString)
    df = CSV.read(path, DataFrame)
    return Dict(String(r.item) => String(r.choice) for r in eachrow(df))
end

# key resolution: species-specific if fitted, else softwood/hardwood pooled ("_S"/"_H"), else "_ALL"
function hb_key(C::HBCoefs, species::AbstractString, grp::Union{Missing,AbstractString}=missing)
    haskey(C, species) && return species
    ismissing(grp) || (haskey(C, "_" * grp) && return "_" * grp)
    return "_ALL"
end
const HB_GRP = Ref(Dict{String,String}())     # species symbol -> "S"/"H" (SFTWD_HRDWD), loaded by hb_init
hb_group_guess(species) = get(HB_GRP[], String(species), "H")

function hb_reg(C, key, model, lnagb, lnage, si, pl)
    m = C[key][model]; s = 0.0
    for (t, v) in m
        (t == "sigma" || t == "n") && continue
        s += v * hb_term(t, lnagb, lnage, si, pl)
    end
    return s
end

"""
    hbad_translate(species, age, agb_g_m2; si=missing, planted=missing, grp=missing, C=default, sel=default)
Returns a NamedTuple (tpa_ha, qmd_cm, ba_m2ha, h_mean_m, h_dom_m, agb_check_g_m2).
`si` is the site index in metres (FIA cond_sicond * 0.3048), `planted` 0/1 (missing -> natural).
"""
function hbad_translate(species::AbstractString, age::Real, agb::Real; si=missing, planted=missing, grp=missing,
                        C::HBCoefs=HB_COEFS[], sel::Dict{String,String}=HB_SEL[])
    key = hb_key(C, String(species), ismissing(grp) ? hb_group_guess(species) : grp)
    lnagb = log(max(agb, 1e-6)); lnage = log(max(age, 1.0))
    pl = ismissing(planted) ? 0.0 : Float64(planted)
    use_si = !ismissing(si)
    sif = use_si ? Float64(si) : 25.0
    function reg(target)
        v = sel["v_" * target]                             # e.g. "V0"
        mname = target * "_" * v
        if use_si && haskey(C[key], mname * "si"); mname = mname * "si"; end
        return exp(hb_reg(C, key, mname, lnagb, lnage, sif, pl))
    end
    mass = sel["mass"]                                     # "tree" or "mq"
    # m_lb(QMD_in) forward and inverse
    a, b = mass == "mq" ? (C[key]["mq"]["a"], C[key]["mq"]["b"]) : (C[key]["tree"]["alpha"], C[key]["tree"]["beta"])
    structure = sel["structure"]                           # "A" TPA-first, "B" QMD-first, "C" direct
    tpa = qmd = ba = NaN
    if structure == "A"
        tpa = max(reg("tpa"), HB_MIN_TPA_HA)
        mlb = agb / (tpa / HB_ACRE2HA * HB_LB)
        qmd_in = exp((log(mlb) - a) / b); qmd = clamp(qmd_in * 2.54, 2.54, 170.0)
        ba = pi / 4 * (qmd / 100)^2 * tpa
    elseif structure == "B"
        qmd = clamp(reg("qmd"), 2.54, 170.0)
        mlb = exp(a + b * log(qmd / 2.54))
        tpa = max(agb / (mlb * HB_LB) * HB_ACRE2HA, HB_MIN_TPA_HA)
        ba = pi / 4 * (qmd / 100)^2 * tpa
    else
        tpa = max(reg("tpa"), HB_MIN_TPA_HA); ba = reg("ba")
        qmd = clamp(sqrt(4 * ba / (pi * tpa)) * 100, 2.54, 170.0)
    end
    # height of the mean tree
    hm = use_si ? get(sel, "hmodel_si", sel["hmodel"]) : sel["hmodel"]
    h = if hm == "cr"
        c = C[key]["hcr"]; 1.3 + c["a"] * (1 - exp(-c["b"] * qmd))^c["c"]
    elseif hm == "ll"
        c = C[key][use_si && haskey(C[key], "hllsi") ? "hllsi" : "hll"]
        exp(c["a"] + c["b"] * log(qmd) + c["c"] * lnage + (haskey(c, "d") ? c["d"] * (sif - 25.0) : 0.0))
    elseif hm == "cal"
        c = C[key][use_si && haskey(C[key], "hcalsi") ? "hcalsi" : "hcal"]
        exp(c["a"] + c["b"] * log(qmd) + c["c"] * lnage + (haskey(c, "d") ? c["d"] * (sif - 25.0) : 0.0))
    else
        reg("h")
    end
    hdom = h * get(C[key]["hdom"], "ratio", 1.0)
    agb_chk = tpa / HB_ACRE2HA * exp(C[key]["tree"]["alpha"] + C[key]["tree"]["beta"] * log(qmd / 2.54)) * HB_LB
    return (tpa_ha=tpa, qmd_cm=qmd, ba_m2ha=ba, h_mean_m=h, h_dom_m=hdom, agb_check_g_m2=agb_chk)
end

const HB_COEFS = Ref(HBCoefs())
const HB_SEL = Ref(Dict{String,String}())
function hb_init(dir::AbstractString=joinpath(@__DIR__, "..", "results", "hbad"))
    HB_COEFS[] = hb_load_coefs(joinpath(dir, "translation_coefs.csv"))
    HB_SEL[] = hb_load_selection(joinpath(dir, "translation_selection.csv"))
    g = CSV.read(joinpath(dir, "species_group.csv"), DataFrame)
    HB_GRP[] = Dict(String(r.species) => String(r.grp) for r in eachrow(g))
    return nothing
end
