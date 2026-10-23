# Tree-based cohort loader for the HBAD3P plugin: live trees of given plot visits -> subplot sites -> (species key, birth decade, diameter class) cohorts, each expanded to the 5 vigour nodes,
# with crown ratio and the forest-area convention of the HBAD3P fit (tools/hbad3/lib3p.jl of the prototype: `subplot_cohorts`, `area_of`, `build_free`; docs/hbad3p_plugin.md).

"Gauss-Hermite (5-point) vigour nodes of HBAD3P (= HBAD3PPlugin.VIG_Z / VIG_W; checked by test/unit_hbad3p.jl)."
const TREE3P_VIG_Z = (-2.856970013872806, -1.355626179974266, 0.0, 1.355626179974266, 2.856970013872806)
const TREE3P_VIG_W = (0.011257411327721, 0.222075922005613, 0.533333333333333, 0.222075922005613, 0.011257411327721)
"Smallest effective forested fraction (a forest sliver is not a closed stand at 1 / f its tree density); prototype `FMIN`."
const FOREST_FMIN = 0.25

"Effective forested fraction a of a STOCKED site (status unknown or 0 although trees were tallied: 1; else max(f, FMIN)); prototype `area_of`."
area_of(f::Real) = (isnan(f) || f <= 0) ? 1.0 : max(Float64(f), FOREST_FMIN)

"""
    TreeCohorts3P

Initial HBAD3P state of subplot sites. Site `s` (visit `site_visit[s]`, subplot `site_subp[s]`) owns the cohorts `off[s]:off[s+1]-1` (CSR; empty subplots own none). Per cohort: `key`, `N` trees/ha of the forested
subplot (x4 / `area[s]`), `D` (QMD cm), `H` (m, >= 1.37), `A` (yr), `zq` vigour node, `CR` crown ratio. Per site: `area` = a (1 without the farea convention), `rmult` = recruitment area multiplier (forested fraction / a; 0 = a
nonforest subplot), `f` = forested fraction of the subplot (NaN unknown); `seed[k, s]` = a conspecific tree with D >= 12.7 cm on the PLOT (any subplot of the visit). `forest_ok` = the forested fractions were obtained from the database.
"""
struct TreeCohorts3P
  keys::Vector{String}
  visits::DataFrame
  site_visit::Vector{Int}
  site_subp::Vector{Int}
  off::Vector{Int}
  key::Vector{Int32}
  N::Vector{Float64}
  D::Vector{Float64}
  H::Vector{Float64}
  A::Vector{Float64}
  zq::Vector{Float64}
  CR::Vector{Float64}
  area::Vector{Float64}
  rmult::Vector{Float64}
  f::Vector{Float64}
  seed::BitMatrix
  vigour::Bool
  farea::Bool
  forest_ok::Bool
  n_trees::Int
end
Base.length(tc::TreeCohorts3P) = length(tc.site_visit)

"""
    forest_fractions(con, vis) -> Dict{(statecd, unitcd, countycd, plot, measdate, subp) => f} | nothing

Forested fraction of every subplot visit of `vis` (accessible forest land: COND_STATUS_CD = 1) = sum of SUBPCOND_PROP over the forest conditions (SUBP_COND x COND x PLOT of the FIADB attached as `src`; visit date =
PLOT MEASYEAR / MEASMON / MEASDAY). NaN when the proportions of the subplot do not sum to 1. `nothing` when the tables cannot be read.
"""
function forest_fractions(con, vis::DataFrame)
  try
    DuckDB.register_data_frame(con, DataFrame(st=Int.(vis.statecd), un=Int.(vis.unitcd), co=Int.(vis.countycd), pl=Int.(vis.plot), md=Dates.Date.(vis.measdate)), "vis_f")
    df = DataFrame(DuckDB.execute(con, """
      SELECT p.STATECD AS st, p.UNITCD AS un, p.COUNTYCD AS co, p.PLOT AS pl, make_date(p.MEASYEAR::INT, p.MEASMON::INT, p.MEASDAY::INT) AS md, s.SUBP AS sb,
             sum(CASE WHEN k.COND_STATUS_CD = 1 THEN s.SUBPCOND_PROP ELSE 0 END) AS f, sum(s.SUBPCOND_PROP) AS tot
      FROM src.SUBP_COND s JOIN src.PLOT p ON p.CN = s.PLT_CN JOIN src.COND k ON k.PLT_CN = s.PLT_CN AND k.CONDID = s.CONDID
      JOIN vis_f v ON v.st = p.STATECD AND v.un = p.UNITCD AND v.co = p.COUNTYCD AND v.pl = p.PLOT
      WHERE p.MEASYEAR IS NOT NULL AND p.MEASMON IS NOT NULL AND p.MEASDAY IS NOT NULL
      GROUP BY ALL"""))
    out = Dict{NTuple{6,Any},Float64}()
    vset = Set((Int(r.statecd), Int(r.unitcd), Int(r.countycd), Int(r.plot), Dates.Date(r.measdate)) for r in eachrow(vis))
    for r in eachrow(df)
      k = (Int(r.st), Int(r.un), Int(r.co), Int(r.pl), Dates.Date(r.md))
      k in vset || continue
      ok = !ismissing(r.tot) && abs(Float64(r.tot) - 1) <= 1e-6 && !ismissing(r.f)
      out[(k..., Int(r.sb))] = ok ? clamp(Float64(r.f), 0.0, 1.0) : NaN
    end
    return out
  catch e
    @warn "HBAD3P: forested fractions could not be read from the database (SUBP_COND / COND / PLOT): $(sprint(showerror, e))"
    return nothing
  end
end

"""
    load_tree_cohorts3p(db_path, visits, keys; birth_w=10, size_breaks=[5, 12.7, 25, 40], vigour=true, farea=true, subplots=1:4,
                        table="curated_trees", resolve=default_key_resolver(keys)) -> TreeCohorts3P

Live trees (STATUSCD 1, DIA > 0, HT > 0, estimated_age > 0, TPA_UNADJ > 0) of the plot visits in `visits` (statecd, unitcd, countycd, plot, measdate) from the cohort database `db_path` (READ_ONLY, bound parameters,
`ORDER BY` the full key). Sites = (visit, SUBP) for `subplots` (empty subplots are sites with zero cohorts). Per tree: w = TPA_UNADJ x 2.47105 (trees/ha), D = DIA x 2.54 cm, H = HT x 0.3048 m, CR = CR / 100. Cohort =
(species key, birth decade `floor((year(measdate) - age) / birth_w)`, diameter class `searchsortedlast(size_breaks, D) + 1` = classes CLOSED BELOW [b_i, b_i+1)); N = 4 sum(w) x node weight / a, D = w-weighted quadratic mean,
H = max(w-weighted mean, 1.37), A = w-weighted mean, CR = w-weighted mean over trees with a recorded crown ratio > 0 (else 0.4) -- the prototype's `subplot_cohorts`. With `farea` (the fit convention) a = `area_of(f)`
from the subplot's forested fraction f; without it (or when the fractions cannot be read: `forest_ok` false, a warning) a = 1. The recruitment multiplier is f / a for a site with a known f (0 for a nonforest empty subplot, 1 for f
unknown or 0 with trees tallied). Cohorts are sorted by (key, birth class, diameter class).
"""
function load_tree_cohorts3p(db_path::AbstractString, visits::DataFrame, keys::AbstractVector{<:AbstractString};
                             birth_w::Real=10, size_breaks::AbstractVector{<:Real}=[5.0, 12.7, 25.0, 40.0], vigour::Bool=true, farea::Bool=true,
                             subplots::AbstractVector{<:Integer}=1:4, table::AbstractString="curated_trees", resolve=default_key_resolver(keys))
  for c in (:statecd, :unitcd, :countycd, :plot, :measdate)
    hasproperty(visits, c) || error("load_tree_cohorts3p: visits needs the column $c")
  end
  isfile(db_path) || error("load_tree_cohorts3p: database not found: $db_path")
  keys_ = String.(keys); kix = Dict(k => Int32(i) for (i, k) in enumerate(keys_)); nsp = length(keys_)
  vis = sort!(unique(select(visits, :statecd, :unitcd, :countycd, :plot, :measdate)), [:statecd, :unitcd, :countycd, :plot, :measdate])
  nv = nrow(vis)
  breaks = Float64.(size_breaks); bw = Float64(birth_w)
  sub = collect(Int, subplots)
  con = DuckDB.connect(DuckDB.DB())
  DuckDB.execute(con, "ATTACH '$(_sql_literal_path(db_path))' AS src (READ_ONLY);")
  tab = _require_table(con, table, "load_tree_cohorts3p table")
  DuckDB.register_data_frame(con, DataFrame(st=Int.(vis.statecd), un=Int.(vis.unitcd), co=Int.(vis.countycd), pl=Int.(vis.plot), md=Dates.Date.(vis.measdate)), "vis_in")
  df = DataFrame(DuckDB.execute(con, """
    SELECT t.STATECD AS st, t.UNITCD AS un, t.COUNTYCD AS co, t.PLOT AS pl, t.MEASDATE AS md, t.SUBP AS sb,
           r.SPECIES_SYMBOL AS sym, r.SFTWD_HRDWD AS grp, t.DIA AS dia, t.HT AS ht, t.TPA_UNADJ AS tpa, t.estimated_age AS age, t.CR AS cr
    FROM src.$(tab) t
    JOIN vis_in v ON v.st = t.STATECD AND v.un = t.UNITCD AND v.co = t.COUNTYCD AND v.pl = t.PLOT AND v.md = t.MEASDATE
    LEFT JOIN src.REF_SPECIES r ON r.SPCD = t.spcd_resolved
    WHERE t.STATUSCD = ? AND t.DIA > ? AND t.HT > ? AND t.estimated_age > ? AND t.TPA_UNADJ > ? AND t.SUBP >= ? AND t.SUBP <= ?
    ORDER BY t.STATECD, t.UNITCD, t.COUNTYCD, t.PLOT, t.MEASDATE, t.SUBP, t.TREE, t.CN""",
    [1, 0.0, 0.0, 0.0, 0.0, minimum(sub), maximum(sub)]))
  ff = forest_fractions(con, vis)
  forest_ok = ff !== nothing && !isempty(ff)
  nq = vigour ? length(TREE3P_VIG_Z) : 1
  vix = Dict((Int(r.statecd), Int(r.unitcd), Int(r.countycd), Int(r.plot), Dates.Date(r.measdate)) => i for (i, r) in enumerate(eachrow(vis)))
  nrows = nrow(df)
  nsites = nv * length(sub)
  site_visit = Int[]; site_subp = Int[]; off = Int[1]
  ckey = Int32[]; cN = Float64[]; cD = Float64[]; cH = Float64[]; cA = Float64[]; czq = Float64[]; cCR = Float64[]
  area = Float64[]; rmult = Float64[]; fv = Float64[]; seed = falses(nsp, nsites)
  r0 = 1; s_ = 0
  for v in 1:nv
    # the rows of this visit (all its subplots): plot-level seed presence
    r1 = r0
    while r1 <= nrows && vix[(Int(df.st[r1]), Int(df.un[r1]), Int(df.co[r1]), Int(df.pl[r1]), Dates.Date(df.md[r1]))] == v; r1 += 1; end
    vseed = falses(nsp)
    for r in r0:r1-1
      Float64(df.dia[r]) * 2.54 >= 12.7 || continue
      k = resolve(String(coalesce(df.sym[r], "UNK")), String(coalesce(df.grp[r], "H")))
      haskey(kix, k) || error("load_tree_cohorts3p: species $(df.sym[r]) resolves to key $(repr(k)), which is not among the keys $(keys_)")
      vseed[kix[k]] = true
    end
    y1 = Dates.year(Dates.Date(vis.measdate[v]))
    for sb in sub
      s_ += 1
      push!(site_visit, v); push!(site_subp, sb)
      rows = Int[]
      while r0 <= nrows
        rv = vix[(Int(df.st[r0]), Int(df.un[r0]), Int(df.co[r0]), Int(df.pl[r0]), Dates.Date(df.md[r0]))]
        (rv < v || (rv == v && df.sb[r0] < sb)) && error("load_tree_cohorts3p: unexpected row order")
        (rv == v && df.sb[r0] == sb) || break
        push!(rows, r0); r0 += 1
      end
      f = forest_ok ? get(ff, (Int(vis.statecd[v]), Int(vis.unitcd[v]), Int(vis.countycd[v]), Int(vis.plot[v]), Dates.Date(vis.measdate[v]), sb), NaN) : NaN
      a = farea && forest_ok ? area_of(f) : 1.0
      fr = (isnan(f) || (f == 0 && !isempty(rows))) ? 1.0 : f                 # true forested fraction (recruitment area); a nonforest empty subplot: 0
      push!(area, a); push!(fv, f); push!(rmult, farea && forest_ok ? fr / a : (forest_ok ? fr : 1.0))
      seed[:, s_] .= vseed
      if !isempty(rows)
        m = length(rows)
        w = Vector{Float64}(undef, m); dia = similar(w); ht = similar(w); age = similar(w); crv = similar(w); kk = Vector{Int32}(undef, m); bc = Vector{Int}(undef, m)
        for (j, r) in enumerate(rows)
          w[j] = Float64(df.tpa[r]) * 2.47105; dia[j] = Float64(df.dia[r]) * 2.54; ht[j] = Float64(df.ht[r]) * 0.3048; age[j] = Float64(df.age[r])
          crv[j] = ismissing(df.cr[r]) ? 0.0 : Float64(df.cr[r])
          k = resolve(String(coalesce(df.sym[r], "UNK")), String(coalesce(df.grp[r], "H")))
          haskey(kix, k) || error("load_tree_cohorts3p: species $(df.sym[r]) resolves to key $(repr(k)), which is not among the keys $(keys_)")
          kk[j] = kix[k]
          bc[j] = floor(Int, (y1 - age[j]) / bw) * 100 + (searchsortedlast(breaks, dia[j]) + 1)
        end
        order = sortperm(collect(zip(kk, bc, 1:m)))
        i = 1
        while i <= m
          j = i
          while j < m && kk[order[j+1]] == kk[order[i]] && bc[order[j+1]] == bc[order[i]]; j += 1; end
          idx = order[i:j]
          ww = w[idx]; sw = sum(ww)
          N = 4 * sw; D = sqrt(sum(ww .* dia[idx] .^ 2) / sw); H = max(sum(ww .* ht[idx]) / sw, 1.37); A = sum(ww .* age[idx]) / sw
          cm = crv[idx] .> 0
          CR = any(cm) ? sum(ww[cm] .* crv[idx][cm]) / sum(ww[cm]) / 100 : 0.4
          for q in 1:nq
            push!(ckey, kk[order[i]]); push!(cN, N * (vigour ? TREE3P_VIG_W[q] : 1.0) / a); push!(cD, D); push!(cH, H); push!(cA, A)
            push!(czq, vigour ? TREE3P_VIG_Z[q] : 0.0); push!(cCR, CR)
          end
          i = j + 1
        end
      end
      push!(off, length(ckey) + 1)
    end
    r0 == r1 || error("load_tree_cohorts3p: unexpected row order (visit $v)")
  end
  r0 > nrows || error("load_tree_cohorts3p: $(nrows - r0 + 1) rows were not assigned to a site")
  return TreeCohorts3P(keys_, vis, site_visit, site_subp, off, ckey, cN, cD, cH, cA, czq, cCR, area, rmult, fv, seed, vigour, farea, forest_ok, nrows)
end

"""The sites `idx` of `tc` in that order (repeats allowed), with their cohorts and per-site attributes; `keys`, `visits`, flags are kept."""
function subset_sites(tc::TreeCohorts3P, idx::AbstractVector{<:Integer})
  off = Int[1]; key = Int32[]; N = Float64[]; D = Float64[]; H = Float64[]; A = Float64[]; zq = Float64[]; CR = Float64[]
  for s in idx
    r = tc.off[s]:tc.off[s+1]-1
    append!(key, tc.key[r]); append!(N, tc.N[r]); append!(D, tc.D[r]); append!(H, tc.H[r]); append!(A, tc.A[r]); append!(zq, tc.zq[r]); append!(CR, tc.CR[r])
    push!(off, length(key) + 1)
  end
  return TreeCohorts3P(tc.keys, tc.visits, tc.site_visit[idx], tc.site_subp[idx], off, key, N, D, H, A, zq, CR, tc.area[idx], tc.rmult[idx], tc.f[idx], tc.seed[:, idx], tc.vigour, tc.farea, tc.forest_ok, tc.n_trees)
end

"""
    tree_cohorts3p_from_table(keys, stands; vigour=true, resolve=default_key_resolver(keys)) -> TreeCohorts3P

TreeMap-independent HBAD3P state: one site per stand from a cohort table (e.g. the generative-vegetation stand cohorts). `stands[i]` = DataFrame with
columns sym (species symbol or group key `_H` / `_S`), grp ("H" / "S"), tph (trees per forested ha), qmd_cm, ht_m, cr (fraction), age (yr). Each row is one
cohort (D = qmd_cm, H = max(ht_m, 1.37), A = age, CR = cr, else 0.4), expanded to the vigour nodes like `load_tree_cohorts3p`; area = 1 (densities are
already per forested ha), recruitment multiplier 1, forested fraction 1; seed = a cohort of the key with QMD >= 12.7 cm in the stand.
"""
function tree_cohorts3p_from_table(keys::AbstractVector{<:AbstractString}, stands::AbstractVector; vigour::Bool=true, resolve=default_key_resolver(keys))
  keys_ = String.(keys); kix = Dict(k => Int32(i) for (i, k) in enumerate(keys_)); nsp = length(keys_); ns = length(stands)
  nq = vigour ? length(TREE3P_VIG_Z) : 1
  off = Int[1]; ckey = Int32[]; cN = Float64[]; cD = Float64[]; cH = Float64[]; cA = Float64[]; czq = Float64[]; cCR = Float64[]
  seed = falses(nsp, ns); nrows = 0
  for (s, t) in enumerate(stands)
    for r in eachrow(t)
      (r.tph > 0 && r.qmd_cm > 0) || continue
      k = resolve(String(r.sym), String(r.grp))
      haskey(kix, k) || error("tree_cohorts3p_from_table: species $(r.sym) resolves to key $(repr(k)), which is not among the keys $(keys_)")
      D = Float64(r.qmd_cm); D >= 12.7 && (seed[kix[k], s] = true); nrows += 1
      cr = ismissing(r.cr) || !(r.cr > 0) ? 0.4 : Float64(r.cr)
      for q in 1:nq
        push!(ckey, kix[k]); push!(cN, Float64(r.tph) * (vigour ? TREE3P_VIG_W[q] : 1.0)); push!(cD, D); push!(cH, max(Float64(r.ht_m), 1.37))
        push!(cA, max(Float64(r.age), 1.0)); push!(czq, vigour ? TREE3P_VIG_Z[q] : 0.0); push!(cCR, cr)
      end
    end
    push!(off, length(ckey) + 1)
  end
  vis = DataFrame(statecd=zeros(Int, ns), unitcd=zeros(Int, ns), countycd=zeros(Int, ns), plot=collect(1:ns), measdate=fill(Dates.Date(2025, 6, 15), ns))
  return TreeCohorts3P(keys_, vis, collect(1:ns), ones(Int, ns), off, ckey, cN, cD, cH, cA, czq, cCR, ones(ns), ones(ns), ones(ns), seed, vigour, false, false, nrows)
end
