# TreeMap-independent scenario stands: plots that carry their OWN cohort table (e.g. the generative-vegetation stand cohorts of
# standmap/scripts/g05_map_stands.py) instead of an FIA plot CN. Every engine starts from it:
#   biomass / pnet / hbad (v1)  cohorts (species, age, AGB): AGB = trees/ha x the species' mean tree mass at the cohort QMD (the HBAD v1 mass-QMD
#                               allometry fitted to FIA DRYBIO_AG, ln m_lb = mq_a + mq_b ln(QMD_in): the same biomass basis as the calibration cohorts)
#   hbad2 / hbad3p              tree cohorts built from the table directly (Data.tree_cohorts_from_table / tree_cohorts3p_from_table), one site per stand
# The stand starts in GEN_START (the generative state is for 2025); ecoregion, site class, climate and soils come from the stand's location (l4, lon, lat).

const GEN_START = Dates.Date(2025, 6, 15)
const _GEN_GRP = Ref{Union{Nothing,Dict{String,String}}}(nothing)

"Species symbol -> 'S' / 'H' (FIADB REF_SPECIES.SFTWD_HRDWD; group keys _S / _H by their name)."
function _gen_grp(scen::ScenarioConfig, sym::AbstractString)
  startswith(sym, "_S") && return "S"; startswith(sym, "_") && return "H"
  if _GEN_GRP[] === nothing
    m = Dict{String,String}()
    if isfile(scen.fia_db)
      db = DuckDB.DB(scen.fia_db; readonly=true); con = DuckDB.DBInterface.connect(db)
      try
        for r in eachrow(DataFrame(DuckDB.DBInterface.execute(con, "SELECT UPPER(TRIM(SPECIES_SYMBOL)) s, ANY_VALUE(SFTWD_HRDWD) g FROM REF_SPECIES WHERE SPECIES_SYMBOL IS NOT NULL GROUP BY 1")))
          ismissing(r.g) || (m[String(r.s)] = String(r.g))
        end
      finally
        DuckDB.DBInterface.close!(con); DuckDB.close(db)
      end
    end
    _GEN_GRP[] = m
  end
  get(_GEN_GRP[], uppercase(sym), "H")
end

"Mean tree dry mass (lb) at QMD `dq` (cm) per species symbol (HBAD v1 priors; group rows for unknown species), cached per symbol."
const _GEN_MASS = Dict{String,Tuple{Float64,Float64}}()
function _gen_mass_lb(sym::AbstractString, dq::Float64)
  ab = get!(_GEN_MASS, String(sym)) do
    p = HBADPlugin.load_priors([String(sym)]; priors_csv=HBADPlugin.DEFAULT_PRIORS_CSV, coefs_csv=HBADPlugin.DEFAULT_COEFS_CSV).sp[1]
    (p.mq_a, p.mq_b)
  end
  exp(ab[1] + ab[2] * log(dq / 2.54))
end

"The cohort table of a request plot (vector of dicts / named tuples with sym, tph, qmd_cm, ht_m, cr, age) as a DataFrame."
function _gen_table(scen::ScenarioConfig, cohorts)
  g(c, k, d=missing) = c isa AbstractDict ? get(c, String(k), get(c, Symbol(k), d)) : (hasproperty(c, k) ? getproperty(c, k) : d)
  sym = String[uppercase(String(g(c, :sym))) for c in cohorts]
  DataFrame(sym=sym, grp=[_gen_grp(scen, s) for s in sym], tph=Float64[Float64(g(c, :tph)) for c in cohorts],
            qmd_cm=Float64[Float64(g(c, :qmd_cm)) for c in cohorts], ht_m=Float64[Float64(something(g(c, :ht_m, NaN), NaN)) for c in cohorts],
            cr=Float64[Float64(something(g(c, :cr, 0.4), 0.4)) for c in cohorts], age=Float64[Float64(g(c, :age)) for c in cohorts])
end

"""
    gen_units(scen, plots, warnings) -> unit records like `load_scenario_units`, plus `gen` = the cohort table

`plots[i]` = (cn, acres, l4, lon, lat, cohorts). A stand without an L4 location is `no_data`; one with an empty table starts bare (regeneration only).
"""
function gen_units(scen::ScenarioConfig, plots, warnings)
  out = NamedTuple[]; nsc = 0
  for (i, x) in enumerate(plots)
    cn = String(x.cn); l4 = String(something(x.l4, "")); l3 = isempty(l4) ? "" : l3_of_l4(l4)
    t = _gen_table(scen, something(x.cohorts, Any[])); t = t[(t.tph .> 0) .& (t.qmd_cm .> 0), :]
    if isempty(l3)
      push!(out, (cn=cn, key="0_0_0_$i", status="no_data", measdate=GEN_START, l3="", l4="", siteclcd=0, lat=NaN,
                  cohorts=DataFrame(sym=String[], age=Int[], agb=Float64[]), donor="", lon=NaN, gen=t)); continue
    end
    sc = site_class_at(scen, l3, Float64(x.lon), Float64(x.lat)); sc > 0 && (nsc += 1)
    agb = Float64[r.tph * _gen_mass_lb(r.sym, r.qmd_cm) * 453.592 / 10000 for r in eachrow(t)]          # g/m2 = trees/ha x lb/tree x g/lb / (m2/ha)
    co = combine(groupby(DataFrame(sym=t.sym, age=round.(Int, max.(t.age, 1.0)), agb=agb), [:sym, :age]), :agb => sum => :agb)
    push!(out, (cn=cn, key="0_0_0_$i", status="ok", measdate=GEN_START, l3=l3, l4=l4, siteclcd=sc, lat=Float64(x.lat), cohorts=co, donor="",
                lon=Float64(x.lon), gen=t))
  end
  nnl = count(r -> r.status == "no_data", out)
  nnl > 0 && push!(warnings, "$nnl generative stand(s) have no L4 ecoregion location; they were not simulated")
  push!(warnings, "initial vegetation: generative stand cohorts (TreeMap-independent), state of $(Dates.year(GEN_START)); site class from the location for $nsc of $(length(out)) stands")
  out
end

_is_gen(plots) = !isempty(plots) && all(x -> hasproperty(x, :cohorts) && x.cohorts !== nothing, plots)
_any_gen(plots) = any(x -> hasproperty(x, :cohorts) && x.cohorts !== nothing, plots)

"""HBAD v1 on generative stands: its native state is (trees/ha, QMD, height) per cohort, so after the AGB-driven site build the cohorts take the table's
values (rows of one eco species and age pooled: trees/ha summed, quadratic-mean QMD, density-weighted height); c_bio = the allometric AGB of that state.
Symbols that share an eco species (e.g. a catch-all) can leave several site cohorts on one (species, age) key: the pooled trees/ha is shared among them
by their site-build AGB, so the stand's trees/ha and basal area match the table."""
function _gen_hbad1_state!(ps::ParamSet, soa, meta)
  pm = Dict(m.plot_id => m for m in meta); nset = 0
  for i in 1:soa.n
    s = getsite(soa, i); m = get(pm, Int(s.mapcode), nothing); (m === nothing || _gen_of(m) === nothing) && continue
    acc = Dict{Tuple{Int,Int},NTuple{3,Float64}}()
    for r in eachrow(m.gen)
      j = scenario_species(ps, m.eco, r.sym); j == 0 && continue
      k = (j, round(Int, max(r.age, 1.0))); a = get(acc, k, (0.0, 0.0, 0.0))
      h = isfinite(r.ht_m) ? r.ht_m : 0.0
      acc[k] = (a[1] + r.tph, a[2] + r.tph * r.qmd_cm^2, a[3] + r.tph * h)
    end
    ckey(c) = (Int(s.c_species[c]), round(Int, Float64(s.c_age[c])))
    kb = Dict{Tuple{Int,Int},Tuple{Float64,Int}}()                 # key -> (site-build AGB, cohorts) over the site cohorts sharing it
    for c in 1:Int(s.live)
      b = get(kb, ckey(c), (0.0, 0)); kb[ckey(c)] = (b[1] + max(Float64(s.c_bio[c]), 0.0), b[2] + 1)
    end
    for c in 1:Int(s.live)
      v = get(acc, ckey(c), nothing); v === nothing && continue
      b = kb[ckey(c)]; f = b[1] > 0 ? max(Float64(s.c_bio[c]), 0.0) / b[1] : 1.0 / b[2]
      tph, dq = f * v[1], sqrt(v[2] / v[1]); h = v[3] / v[1]
      pr = ps.hb.priors.sp[_gsp(ps, s.eco_id, s.c_species[c])]
      s.c_tph[c] = FloatType(tph); s.c_dq[c] = FloatType(dq); h > 1.3 && (s.c_h[c] = FloatType(h))
      s.c_bio[c] = FloatType(tph * HBADPlugin.mean_tree_lb(pr, dq) * 453.592 / 10000); nset += 1
    end
  end
  nset
end

"The generative cohort table of a scenario plot record (nothing for FIA plots and records built without the field)."
_gen_of(m) = hasproperty(m, :gen) ? m.gen : nothing
