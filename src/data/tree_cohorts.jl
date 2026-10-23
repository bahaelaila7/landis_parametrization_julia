# Tree-based cohort loader for the HBAD2 plugin: live trees of given plot visits -> subplot sites -> (species key, birth class, size class) cohorts,
# each expanded to the 3 vigour sub-cohorts. The exact construction of the prototype's Sim B initial state (tools/hbad2/simB_lik.jl `build_simB`).

"Gauss-Hermite (3-point) vigour sub-cohorts (= HBAD2Plugin.VIG_Z / VIG_W; checked by test/unit_hbad2.jl)."
const TREE_VIG_Z = (-sqrt(3.0), 0.0, sqrt(3.0))
const TREE_VIG_W = (1 / 6, 2 / 3, 1 / 6)

"""
    TreeCohorts

Initial HBAD2 state of subplot sites. Site `s` (visit `site_visit[s]` of the `visits` table, subplot `site_subp[s]`) owns the cohorts `off[s]:off[s+1]-1`
(CSR; empty subplots own none). Per cohort: `key` (index into `keys`), `N` trees/ha (x`site_mult`), `D` quadratic mean diameter (cm), `H` mean height (m), `A` mean age (yr, >= 1),
`zq` vigour quantile. With `vigour` every cohort appears as 3 consecutive sub-cohorts (shares 1/6, 2/3, 1/6 of N).
"""
struct TreeCohorts
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
  vigour::Bool
  n_trees::Int
end
Base.length(tc::TreeCohorts) = length(tc.site_visit)

"""Default species -> key rule: the species' own symbol when it is one of `keys` (and not a `_` group key), else the pooled group key `_S` / `_H` (FIA REF_SPECIES SFTWD_HRDWD; unknown -> `_H`)."""
function default_key_resolver(keys::AbstractVector{<:AbstractString})
  have = Set(String.(keys)); big = Set(k for k in have if !startswith(k, "_"))
  return (sym::AbstractString, grp::AbstractString) -> (sym in big ? String(sym) : "_" * String(grp))
end

"""
    load_tree_cohorts(db_path, visits, keys; birth_w=10, size_breaks=[5, 12.7, 25, 40], vigour=true, subplots=1:4, site_mult=4.0,
                      table="curated_trees", resolve=default_key_resolver(keys)) -> TreeCohorts

Read the live trees (STATUSCD 1, DIA > 0, HT > 0, estimated_age > 0, TPA_UNADJ > 0) of the plot visits in `visits` (DataFrame with columns `statecd, unitcd, countycd, plot, measdate`)
from the cohort database `db_path` (opened READ_ONLY; all values are bound parameters; the rows are read `ORDER BY` the full key STATECD, UNITCD, COUNTYCD, PLOT, MEASDATE, SUBP, TREE, CN).
Sites = (visit, SUBP) for every subplot of `subplots` (empty subplots are sites with zero cohorts), sorted by visit key then SUBP. Per tree: density `w = TPA_UNADJ * 2.47105`
(trees/ha), `D = DIA * 2.54` cm, `H = HT * 0.3048` m, age = `estimated_age`. Cohort = (species key via `resolve(symbol, SFTWD_HRDWD)`, birth class `floor(-age / birth_w)`, size class
`searchsortedfirst(size_breaks, D)`): `N = sum(w) * site_mult`, `D` = quadratic mean (w-weighted), `H`, `A` = w-weighted means, `A = max(A, 1)`. Cohorts are sorted by (key index, birth class, size class).
A species the resolver maps to a key outside `keys` is an error.
"""
function load_tree_cohorts(db_path::AbstractString, visits::DataFrame, keys::AbstractVector{<:AbstractString};
                           birth_w::Real=10, size_breaks::AbstractVector{<:Real}=[5.0, 12.7, 25.0, 40.0], vigour::Bool=true,
                           subplots::AbstractVector{<:Integer}=1:4, site_mult::Real=4.0, table::AbstractString="curated_trees",
                           resolve=default_key_resolver(keys))
  for c in (:statecd, :unitcd, :countycd, :plot, :measdate)
    hasproperty(visits, c) || error("load_tree_cohorts: visits needs the column $c")
  end
  isfile(db_path) || error("load_tree_cohorts: database not found: $db_path")
  keys_ = String.(keys); kix = Dict(k => Int32(i) for (i, k) in enumerate(keys_))
  vis = sort!(unique(select(visits, :statecd, :unitcd, :countycd, :plot, :measdate)), [:statecd, :unitcd, :countycd, :plot, :measdate])
  nv = nrow(vis)
  breaks = Float64.(size_breaks); bw = Float64(birth_w); mult = Float64(site_mult)
  sub = collect(Int, subplots)
  con = DuckDB.connect(DuckDB.DB())
  df = try
    DuckDB.execute(con, "ATTACH '$(_sql_literal_path(db_path))' AS src (READ_ONLY);")
    tab = _require_table(con, table, "load_tree_cohorts table")
    DuckDB.register_data_frame(con, DataFrame(st=Int.(vis.statecd), un=Int.(vis.unitcd), co=Int.(vis.countycd), pl=Int.(vis.plot), md=Dates.Date.(vis.measdate)), "vis_in")
    DataFrame(DuckDB.execute(con, """
      SELECT t.STATECD AS st, t.UNITCD AS un, t.COUNTYCD AS co, t.PLOT AS pl, t.MEASDATE AS md, t.SUBP AS sb,
             r.SPECIES_SYMBOL AS sym, r.SFTWD_HRDWD AS grp, t.DIA AS dia, t.HT AS ht, t.TPA_UNADJ AS tpa, t.estimated_age AS age
      FROM src.$(tab) t
      JOIN vis_in v ON v.st = t.STATECD AND v.un = t.UNITCD AND v.co = t.COUNTYCD AND v.pl = t.PLOT AND v.md = t.MEASDATE
      LEFT JOIN src.REF_SPECIES r ON r.SPCD = t.spcd_resolved
      WHERE t.STATUSCD = ? AND t.DIA > ? AND t.HT > ? AND t.estimated_age > ? AND t.TPA_UNADJ > ? AND t.SUBP >= ? AND t.SUBP <= ?
      ORDER BY t.STATECD, t.UNITCD, t.COUNTYCD, t.PLOT, t.MEASDATE, t.SUBP, t.TREE, t.CN""",
      [1, 0.0, 0.0, 0.0, 0.0, minimum(sub), maximum(sub)]))
  finally
    nothing   # the in-memory connection is released with the object (as in data/utils.jl)
  end
  nq = vigour ? 3 : 1
  vix = Dict((Int(r.statecd), Int(r.unitcd), Int(r.countycd), Int(r.plot), Dates.Date(r.measdate)) => i for (i, r) in enumerate(eachrow(vis)))
  nrows = nrow(df)
  # rows are ordered by visit, SUBP: walk them site by site
  site_visit = Int[]; site_subp = Int[]; off = Int[1]
  ckey = Int32[]; cN = Float64[]; cD = Float64[]; cH = Float64[]; cA = Float64[]; czq = Float64[]
  r0 = 1
  for v in 1:nv, sb in sub
    push!(site_visit, v); push!(site_subp, sb)
    rows = Int[]
    while r0 <= nrows
      rv = vix[(Int(df.st[r0]), Int(df.un[r0]), Int(df.co[r0]), Int(df.pl[r0]), Dates.Date(df.md[r0]))]
      (rv < v || (rv == v && df.sb[r0] < sb)) && error("load_tree_cohorts: unexpected row order")
      (rv == v && df.sb[r0] == sb) || break
      push!(rows, r0); r0 += 1
    end
    if !isempty(rows)
      m = length(rows)
      w = Vector{Float64}(undef, m); dia = similar(w); ht = similar(w); age = similar(w); kk = Vector{Int32}(undef, m); bc = Vector{Int}(undef, m)
      for (a, r) in enumerate(rows)
        w[a] = Float64(df.tpa[r]) * 2.47105; dia[a] = Float64(df.dia[r]) * 2.54; ht[a] = Float64(df.ht[r]) * 0.3048; age[a] = Float64(df.age[r])
        sym = coalesce(df.sym[r], "UNK"); grp = coalesce(df.grp[r], "H")
        k = resolve(String(sym), String(grp))
        haskey(kix, k) || error("load_tree_cohorts: species $(sym) (group $(grp)) resolves to key $(repr(k)), which is not among the keys $(keys_)")
        kk[a] = kix[k]
        bc[a] = floor(Int, -age[a] / bw) * 100 + searchsortedfirst(breaks, dia[a])
      end
      order = sortperm(collect(zip(kk, bc, 1:m)))          # (key, bc, row): deterministic cohort order, rows keep their SQL order inside a cohort
      i = 1
      while i <= m
        j = i
        while j < m && kk[order[j+1]] == kk[order[i]] && bc[order[j+1]] == bc[order[i]]; j += 1; end
        idx = order[i:j]
        ww = w[idx]; sw = sum(ww)
        N = sw * mult; D = sqrt(sum(ww .* dia[idx] .^ 2) / sw); H = sum(ww .* ht[idx]) / sw; A = max(sum(ww .* age[idx]) / sw, 1.0)
        for q in 1:nq
          push!(ckey, kk[order[i]]); push!(cN, N * (vigour ? TREE_VIG_W[q] : 1.0)); push!(cD, D); push!(cH, H); push!(cA, A); push!(czq, vigour ? TREE_VIG_Z[q] : 0.0)
        end
        i = j + 1
      end
    end
    push!(off, length(ckey) + 1)
  end
  r0 > nrows || error("load_tree_cohorts: $(nrows - r0 + 1) rows were not assigned to a site")
  return TreeCohorts(keys_, vis, site_visit, site_subp, off, ckey, cN, cD, cH, cA, czq, vigour, nrows)
end

"""
    subset_sites(tc, idx) -> TreeCohorts

The sites `idx` of `tc` in that order (repeats allowed, e.g. two scenario plots sharing a donor visit), with their cohorts; `keys`, `visits` and `vigour` are kept.
"""
function subset_sites(tc::TreeCohorts, idx::AbstractVector{<:Integer})
  off = Int[1]; key = Int32[]; N = Float64[]; D = Float64[]; H = Float64[]; A = Float64[]; zq = Float64[]
  for s in idx
    r = tc.off[s]:tc.off[s+1]-1
    append!(key, tc.key[r]); append!(N, tc.N[r]); append!(D, tc.D[r]); append!(H, tc.H[r]); append!(A, tc.A[r]); append!(zq, tc.zq[r])
    push!(off, length(key) + 1)
  end
  return TreeCohorts(tc.keys, tc.visits, tc.site_visit[idx], tc.site_subp[idx], off, key, N, D, H, A, zq, tc.vigour, tc.n_trees)
end

"""
    tree_cohorts_from_table(keys, stands; vigour=true, resolve=default_key_resolver(keys)) -> TreeCohorts

TreeMap-independent HBAD2 state: one site per stand from a cohort table (columns sym, grp, tph = trees/ha, qmd_cm, ht_m, age), one cohort per row expanded
to the 3 vigour sub-cohorts like `load_tree_cohorts` (A = max(age, 1)).
"""
function tree_cohorts_from_table(keys::AbstractVector{<:AbstractString}, stands::AbstractVector; vigour::Bool=true, resolve=default_key_resolver(keys))
  keys_ = String.(keys); kix = Dict(k => Int32(i) for (i, k) in enumerate(keys_)); ns = length(stands)
  nq = vigour ? length(TREE_VIG_Z) : 1
  off = Int[1]; ckey = Int32[]; cN = Float64[]; cD = Float64[]; cH = Float64[]; cA = Float64[]; czq = Float64[]; nrows = 0
  for t in stands
    for r in eachrow(t)
      (r.tph > 0 && r.qmd_cm > 0) || continue
      k = resolve(String(r.sym), String(r.grp))
      haskey(kix, k) || error("tree_cohorts_from_table: species $(r.sym) resolves to key $(repr(k)), which is not among the keys $(keys_)")
      nrows += 1
      for q in 1:nq
        push!(ckey, kix[k]); push!(cN, Float64(r.tph) * (vigour ? TREE_VIG_W[q] : 1.0)); push!(cD, Float64(r.qmd_cm)); push!(cH, Float64(r.ht_m))
        push!(cA, max(Float64(r.age), 1.0)); push!(czq, vigour ? TREE_VIG_Z[q] : 0.0)
      end
    end
    push!(off, length(ckey) + 1)
  end
  vis = DataFrame(statecd=zeros(Int, ns), unitcd=zeros(Int, ns), countycd=zeros(Int, ns), plot=collect(1:ns), measdate=fill(Dates.Date(2025, 6, 15), ns))
  return TreeCohorts(keys_, vis, collect(1:ns), ones(Int, ns), off, ckey, cN, cD, cH, cA, czq, vigour, nrows)
end
