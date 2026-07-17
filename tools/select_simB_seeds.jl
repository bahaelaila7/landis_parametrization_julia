# Select Sim-B seeds from a finished Sim-A run: the 5 p101-front positions + K crowding-weighted-random
# picks from the rest of the p101 front, on TRAIN or VAL objectives. Writes a DuckDB `sobol_results`
# table (same schema tools/extract_archive_seeds.jl emits) that a Sim-B igelmo run loads via
# `sobol_candidates_db` with `sobol_top_frac: 1.0` and `igel_mu: <N>`. Each seed's params carry its
# Sim-A growth → with `fix_growth: true` + `igel_freeze_seed_growth: true` that growth is frozen
# per-lineage in Sim B, and `igel_seed_maturity: true` gives the phase-1 maturing window.
#
# Front math (nondom / positions / knee / crowding) mirrors tools/analyze_run.jl so seeds line up with
# the analysis' p101 positions.
#
# Usage:
#   ./julia_gdal.sh --project=. tools/select_simB_seeds.jl <sim_a_run_dir> <out_seeds.duckdb> [options]
#     --val       : select on the held-out VALIDATION front (needs cv_val_cache.csv). Default: TRAIN.
#     --mu N      : total seeds to emit (default 25) → 5 positions + (N-5) crowding-weighted picks.
#     --warmup N  : ignore candidates from gens <= N (default 10).
#     --seed S    : RNG seed for the crowding-weighted sampling (default 1).
using Pan
import JLD2, CSV, DuckDB, DataFrames, Random, Serialization
const DF = DataFrames

_argval(flag, dflt) = (i = findfirst(==(flag), ARGS); i === nothing ? dflt : ARGS[i+1])
_hasflag(flag) = any(==(flag), ARGS)

length(ARGS) >= 2 || error("usage: select_simB_seeds.jl <sim_a_run_dir> <out_seeds.duckdb> [--val] [--mu 25] [--warmup 10] [--seed 1]")
run_dir = ARGS[1]; out_db = ARGS[2]
isdir(run_dir) || error("run_dir not found: $run_dir")
use_val = _hasflag("--val")
mu      = parse(Int, _argval("--mu", "25"))
warmup  = parse(Int, _argval("--warmup", "10"))
rng     = Random.MersenneTwister(parse(Int, _argval("--seed", "1")))
mu > 5 || error("--mu must be > 5 (5 positions + K crowding picks)")
K = mu - 5
split = use_val ? "val" : "train"
stride   = parse(Int, _argval("--stride", "1"))     # load every Nth checkpoint (1 = all)
maxckpts = parse(Int, _argval("--max-ckpts", "0"))  # 0 = no cap; else keep only the last K (the final archive already accumulates the non-dominated set, so a cap loses little)

# ─────────────── load checkpoints (each has .archive of MOCandidates: c.x=params, c.fx=MOFitness) ───────────────
ckpt_files = filter(f -> occursin(r"^search_state@\d+\.jld2$", f), readdir(run_dir))
isempty(ckpt_files) && error("no search_state@<gen>.jld2 checkpoints in $run_dir")
gen_of(f) = parse(Int, match(r"@(\d+)\.jld2$", f).captures[1])
ckpt_files = sort(ckpt_files, by=gen_of)
ntotal = length(ckpt_files)
# SAFETY: the archive is cumulative, so the final checkpoint already holds the non-dominated front. Loading
# thousands of large archives blows past memory (e.g. 10k ckpts × 1k candidates ≈ 150+ GB). If the caller
# set no stride/cap and there are many checkpoints, default to the FINAL archive only (override with the flags).
if stride == 1 && maxckpts == 0 && ntotal > 1500
    println("NOTE: $ntotal checkpoints — loading only the FINAL archive (it already accumulates the non-dominated front).")
    println("      Override with --max-ckpts K (last K) or --stride N to widen the pool."); flush(stdout)
    maxckpts = 1
end
stride > 1 && (ckpt_files = ckpt_files[1:stride:end])
maxckpts > 0 && length(ckpt_files) > maxckpts && (ckpt_files = ckpt_files[end-maxckpts+1:end])
println("loading $(length(ckpt_files))/$ntotal checkpoints (stride=$stride, max-ckpts=$(maxckpts == 0 ? "all" : maxckpts)) …"); flush(stdout)
gens = Vector{Tuple{Int,Any}}(undef, length(ckpt_files))
for (i, f) in enumerate(ckpt_files)
    gens[i] = (gen_of(f), JLD2.load_object(joinpath(run_dir, f)).archive)
    (i % 100 == 0 || i == length(ckpt_files)) && (println("  loaded $i/$(length(ckpt_files)) checkpoints"); flush(stdout))
end
println("loaded $(length(gens)) checkpoints: gens $(gen_of(ckpt_files[1]))..$(gen_of(ckpt_files[end]))")

train_pt(c) = (Float64(sum(@view c.fx.objectives[1:2:end])), Float64(sum(@view c.fx.objectives[2:2:end])))
ckey(c) = Tuple(round.(Float64.(collect(c.fx.objectives)), digits=10))
_rk(w, a) = (round(w, digits=6), round(a, digits=6))

# ─────────────── point map on the chosen split ───────────────
pf = if use_val
  valcache = Dict{Tuple{Float64,Float64},Tuple{Float64,Float64}}()
  f = joinpath(run_dir, "cv_val_cache.csv")
  isfile(f) || error("cv_val_cache.csv not found in $run_dir (needed for --val)")
  for r in DF.eachrow(CSV.read(f, DF.DataFrame))
    valcache[_rk(Float64(r.A_W_train), Float64(r.A_AGB_train))] = (Float64(r.A_W), Float64(r.A_AGB))
  end
  println("val cache: $(length(valcache)) scored candidates")
  c -> get(valcache, _rk(train_pt(c)...), nothing)
else
  c -> train_pt(c)
end

# unique candidates across all eligible gens (gen > warmup) that HAVE a point on this split
seen = Set{NTuple}(); pairs = Tuple{Any,Tuple{Float64,Float64}}[]
for (g, arch) in gens
  g <= warmup && continue
  for c in arch
    k = ckey(c); k in seen && continue
    p = pf(c); p === nothing && continue
    push!(seen, k); push!(pairs, (c, p))
  end
end
isempty(pairs) && error("no candidates with a $split point after warmup=$warmup")
println("eligible unique candidates on $split: $(length(pairs))")

# ─────────────── front math (mirrors analyze_run.jl) ───────────────
nondom(prs) = filter(p -> !any(q -> q[2][1] <= p[2][1] && q[2][2] <= p[2][2] && q[2] != p[2], prs), prs)
function knee_dist_pts(pts)
  length(pts) < 3 && return zeros(length(pts))
  ws = [q[1] for q in pts]; as = [q[2] for q in pts]
  nw(w) = (hi = maximum(ws); lo = minimum(ws); hi > lo ? (w-lo)/(hi-lo) : 0.0)
  na(a) = (hi = maximum(as); lo = minimum(as); hi > lo ? (a-lo)/(hi-lo) : 0.0)
  eW = pts[argmin(ws)]; eA = pts[argmin(as)]; p1 = (nw(eW[1]), na(eW[2])); pk = (nw(eA[1]), na(eA[2]))
  d12 = hypot(pk[1]-p1[1], pk[2]-p1[2])
  d12 <= 0 ? zeros(length(pts)) :
    [abs((pk[1]-p1[1])*(p1[2]-na(q[2])) - (p1[1]-nw(q[1]))*(pk[2]-p1[2]))/d12 for q in pts]
end
# NSGA-II crowding distance (summed over both objectives, boundaries → Inf) — same as IgelMOCMAES._crowding.
function crowding(pts)
  m = length(pts); m <= 2 && return fill(Inf, m)
  cd = zeros(m)
  for o in 1:2
    vals = [Float64(p[o]) for p in pts]
    ord = sortperm(vals); span = vals[ord[end]] - vals[ord[1]]
    cd[ord[1]] = Inf; cd[ord[end]] = Inf
    span > 0 && for r in 2:m-1; cd[ord[r]] += (vals[ord[r+1]] - vals[ord[r-1]]) / span; end
  end
  cd
end
# 5 positions on the non-dominated front: extreme-W, extreme-AGB, knee, median, best-aggregate.
function positions(prs)
  nd = sort(nondom(prs), by = p -> (p[2][1], -p[2][2]))
  cs = [p[1] for p in nd]; vp = [p[2] for p in nd]; n = length(nd)
  n == 0 && return Any[]
  ws = [q[1] for q in vp]; as = [q[2] for q in vp]
  iW = argmin(ws); iA = argmin(as); iB = argmin(ws .+ as)
  n == 1 && return [cs[1]]
  kd = knee_dist_pts(vp); ik = argmax(kd)
  # median = closest to the midpoint of the two extremes in normalized space
  nw(w) = (hi = maximum(ws); lo = minimum(ws); hi > lo ? (w-lo)/(hi-lo) : 0.0)
  na(a) = (hi = maximum(as); lo = minimum(as); hi > lo ? (a-lo)/(hi-lo) : 0.0)
  M = ((nw(vp[iW][1])+nw(vp[iA][1]))/2, (na(vp[iW][2])+na(vp[iA][2]))/2)
  im = argmin(hypot(nw(vp[i][1])-M[1], na(vp[i][2])-M[2]) for i in 1:n)
  [cs[iW], cs[iA], cs[ik], cs[im], cs[iB]]
end

p101 = nondom(pairs)
p101 = sort(p101, by = p -> (p[2][1], -p[2][2]))
println("p101 front size on $split: $(length(p101))")

# 5 positions (deduped by objective key — coincide when the front is tiny)
pos_cands = positions(pairs)
pos_keys = Set(ckey(c) for c in pos_cands)
pos_cands = unique(c -> ckey(c), pos_cands)
println("positions selected: $(length(pos_cands)) (extreme-W, extreme-AGB, knee, median, best-agg; deduped)")

# K crowding-weighted-random from the REST of the p101 front (higher crowding ⇒ higher probability)
p101_cands = [p[1] for p in p101]; p101_pts = [p[2] for p in p101]
cd = crowding(p101_pts)
pool_idx = [i for i in eachindex(p101_cands) if !(ckey(p101_cands[i]) in pos_keys)]
maxfin = maximum([cd[i] for i in pool_idx if isfinite(cd[i])]; init=1.0)
pool_w = [isfinite(cd[i]) ? cd[i] : maxfin for i in pool_idx]   # any residual Inf (shouldn't occur) ⇒ top finite weight
function wsample!(rng, idxs, weights, k)
  idxs = collect(idxs); weights = collect(Float64.(weights)); picked = Int[]
  for _ in 1:min(k, length(idxs))
    s = sum(weights)
    j = (s <= 0 || !isfinite(s)) ? rand(rng, 1:length(idxs)) : begin
      r = rand(rng) * s; acc = 0.0; jj = length(idxs)
      for t in eachindex(weights); acc += weights[t]; if acc >= r; jj = t; break; end; end; jj
    end
    push!(picked, idxs[j]); deleteat!(idxs, j); deleteat!(weights, j)
  end
  picked
end
k_idx = wsample!(rng, pool_idx, pool_w, K)
crowd_cands = [p101_cands[i] for i in k_idx]
if length(crowd_cands) < K
  @warn "only $(length(crowd_cands)) crowding picks available (pool=$(length(pool_idx))); emitting $(length(pos_cands)+length(crowd_cands)) seeds < mu=$mu"
end

# ─────────────── write the seed DB (sobol_results schema; sumW/sumAGB = split point) ───────────────
seeds = vcat(pos_cands, crowd_cands)
seeds = unique(c -> ckey(c), seeds)     # a crowding pick can't equal a position (pool excludes them), but be safe
println("total seeds: $(length(seeds)) ($(length(pos_cands)) positions + $(length(crowd_cands)) crowding)")

isfile(out_db) && rm(out_db)
db = DuckDB.DB(out_db); con = DuckDB.connect(db)
DuckDB.execute(con, "CREATE TABLE sobol_results (run_id VARCHAR, sobol_idx INTEGER, mean_loss DOUBLE, std_loss DOUBLE, median_loss DOUBLE, sumW DOUBLE, sumAGB DOUBLE, params_blob BLOB)")
for (i, c) in enumerate(seeds)
  w, a = pf(c); agg = Float64(w + a)
  buf = IOBuffer(); Serialization.serialize(buf, c.x)
  DuckDB.execute(con, "INSERT INTO sobol_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
    ["simB_seeds_$split", i, agg, 0.0, agg, Float64(w), Float64(a), take!(buf)])
end
close(db)
println("SEEDS=$(length(seeds))")   # machine-readable count for the submit script (→ igel_mu override)
println("wrote $out_db — $(length(seeds)) Sim-B seeds on the $split front")
println("  Sim-B yaml: sobol_candidates_db: $out_db  |  sobol_top_frac: 1.0  |  igel_mu: $(length(seeds))")
println("              fix_growth: true  igel_freeze_seed_growth: true  igel_seed_maturity: true  igel_maturity: <N>")
