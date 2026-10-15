# Evaluation of one Sim A (+ Sim B) result against the pass criteria of docs/simAB_options_experiment.md.
#   julia tools/simab_eval.jl <cfgA.yml> <simA_params.jld2> [<simB_params.jld2>] <out.md>
# env: SIMAB_CFG_B (default: configs/region4/simab_B.yml next to cfgA), SIMAB_ARCHIVE (Sim A gdigel output dir with mo_archive/ for criterion 6),
#      SIMAB_NAME (label in the report), SIMAB_THREADS (default 8), SIMAB_REF (BS coordinate test reference "15,28,46" = ±10 %, ±20 % passes, groups).
# Each check runs in its own Julia process (tools/simab_eval_tost.jl and tools/simab_eval_ctx.jl; tost_common.jl is unchanged).
using JLD2, DataFrames, CSV, Statistics, Printf
args = ARGS; length(args) in (3, 4) || error("usage: simab_eval.jl <cfgA.yml> <simA.jld2> [<simB.jld2>] <out.md>")
cfgA, pA = abspath(args[1]), abspath(args[2]); pB = length(args) == 4 ? abspath(args[3]) : nothing; out = abspath(args[end])
root = dirname(@__DIR__); cfgB = abspath(get(ENV, "SIMAB_CFG_B", joinpath(dirname(cfgA), "simab_B.yml")))
name = get(ENV, "SIMAB_NAME", splitext(basename(out))[1]); thr = get(ENV, "SIMAB_THREADS", "8")
ref10_t, ref20_t, refng_t = parse.(Int, split(get(ENV, "SIMAB_REF", "15,28,46"), ","))   # the TOST-targeted reference's TEST numbers: information only
refparams = abspath(get(ENV, "SIMAB_REF_PARAMS", "/workspace/landis_scen/results/gafl/params_gafl_biomass_real.jld2"))   # reference set: its VAL numbers decide criterion 1
work = joinpath(dirname(out), "eval_" * name); mkpath(work); archive = get(ENV, "SIMAB_ARCHIVE", "")
jl(script, a...) = run(Cmd(String[joinpath(root, "julia_gdal.sh"), "--project=$root", "--threads=$thr", joinpath(root, "tools", script), String.(collect(a))...]))
res = Dict{String,Any}()
step(f, tag) = (o = joinpath(work, f); isfile(o) || jl(tag...); JLD2.load_object(o))   # resumable: a finished check is reused
refcache = abspath(get(ENV, "SIMAB_REF_CACHE", joinpath(root, "runs_local", "simab_ref_tost.jld2")))
tR = nothing
if isfile(refparams); isfile(refcache) || jl("simab_eval_tost.jl", cfgA, refparams, refcache); tR = JLD2.load_object(refcache); end
tA = (o = joinpath(work, "tost_A.jld2"); isfile(o) || jl("simab_eval_tost.jl", cfgA, pA, o); JLD2.load_object(o))
cA = (o = joinpath(work, "ctx_A.jld2"); isfile(o) || jl("simab_eval_ctx.jl", "A", cfgA, pA, o, archive); JLD2.load_object(o))
tB = cB = nothing
if pB !== nothing
  tB = (o = joinpath(work, "tost_B.jld2"); isfile(o) || jl("simab_eval_tost.jl", cfgB, pB, o); JLD2.load_object(o))
  cB = (o = joinpath(work, "ctx_B.jld2"); isfile(o) || jl("simab_eval_ctx.jl", "B", cfgB, pB, o); JLD2.load_object(o))
end
io = IOBuffer(); P(s...) = println(io, s...)
verdict = Dict{Int,Union{Bool,Missing}}()
P("# Sim A/B evaluation: $name\n"); P("Sim A config `$(basename(cfgA))`, params `$(pA)`", pB === nothing ? "" : "; Sim B config `$(basename(cfgB))`, params `$(pB)`", "\n")
# ---- 1. growth, held-out (VAL decides; TEST is reported only)
P("## 1. Growth (Sim A held-out TOST): VAL decides, TEST reported\n"); P("| split | groups | ±5 % | ±10 % | ±20 % | median abs bias % |"); P("|---|---|---|---|---|---|")
for sp in ("val", "test"); r = tA[sp]; P("| $sp$(sp == "test" ? " (information only)" : "") | $(r["ng"]) | $(r["pass05"]) | $(r["pass10"]) | $(r["pass20"]) | $(@sprintf("%.1f", r["median_abs_bias"])) |"); end
if tR !== nothing
  for sp in ("val", "test"); r = tR[sp]; P("| reference (TOST-targeted BS coordinate set), $sp | $(r["ng"]) | $(r["pass05"]) | $(r["pass10"]) | $(r["pass20"]) | $(@sprintf("%.1f", r["median_abs_bias"])) |"); end
end
P("\nThe reference's TEST numbers from the design: ±10 % $ref10_t/$refng_t, ±20 % $ref20_t/$refng_t (information only).")
r = tA["val"]
if tR !== nothing
  rv = tR["val"]; sc = r["ng"] / rv["ng"]; t10 = rv["pass10"] * sc; t20 = rv["pass20"] * sc
  ok1 = r["pass10"] >= t10 - 1e-9 && r["pass20"] >= t20 - 1e-9 && r["median_abs_bias"] <= 10.0
  P("\nRule (decided on VAL): ±10 % passes ≥ $(round(t10, digits=1)) and ±20 % ≥ $(round(t20, digits=1)) (the reference's val numbers $(rv["pass10"]) / $(rv["pass20"]) of $(rv["ng"]) groups, scaled to $(r["ng"]) groups) and median |bias| ≤ 10 %: **$(ok1 ? "PASS" : "FAIL")**\n")
else
  ok1 = missing; P("\nNo reference parameter set found (`SIMAB_REF_PARAMS`): criterion 1 not decided.\n")
end
verdict[1] = ok1
# ---- 2. bounds
P("## 2. Bound hugging (u ≤ 0.01 or ≥ 0.99), % of fitted dimensions\n"); P("| stage | family | dims | hugging | % | low / high |"); P("|---|---|---|---|---|---|")
ok2 = true
for (st, c) in (("Sim A", cA), ("Sim B", cB)); c === nothing && continue
  for (f, b) in sort(collect(c["bounds"]); by=first); fr = 100 * b.hug / max(b.n, 1); global ok2 &= fr <= 5.0
    P("| $st | $f | $(b.n) | $(b.hug) | $(@sprintf("%.1f", fr)) | $(b.lo) / $(b.hi) |"); end
end
verdict[2] = ok2; P("\nRule: ≤ 5 % per family: **$(ok2 ? "PASS" : "FAIL")**\n")
# ---- 3. mortality
P("## 3. Mortality: implied vs empirical natural-death rate (train)\n")
emp = cA["empirical"]; pm = (cB !== nothing ? cB["cells"] : cA["cells"]); src = cB !== nothing ? "Sim B final params" : "Sim A params"
j = innerjoin(emp, select(pm, :eco, :species, :pmort), on=[:eco, :species])
j = filter(r -> r.n_cohort_intervals >= 20 && !isnan(r.pmort), j)
if nrow(j) == 0
  P("No (eco, species) cell with a fitted PROB_MORT and ≥ 20 cohort-intervals ($src)."); verdict[3] = missing
else
  j.ratio = [(max(r.pmort, 1e-6) / max(r.p_emp, 1e-6)) for r in eachrow(j)]
  within = mean(0.5 .<= j.ratio .<= 2.0); expo = j.exposure_yr
  pool_imp = sum(expo .* j.pmort) / sum(expo); pool_emp = sum(j.n_deaths) / sum(expo); pb = 100 * (pool_imp / pool_emp - 1)
  ok3 = within >= 0.8 && abs(pb) <= 20
  verdict[3] = ok3
  P("Implied annual rate p from $src; empirical = per-cell MLE of the survive/die likelihood over the same cohort-intervals (cells with ≥ 20 intervals: $(nrow(j))).\n")
  P("- cells within ×2 of the empirical rate: $(@sprintf("%.0f", 100 * within)) % (rule ≥ 80 %)"); P("- pooled implied $(@sprintf("%.4f", pool_imp)) vs empirical $(@sprintf("%.4f", pool_emp)) per year: bias $(@sprintf("%+.0f", pb)) % (rule |bias| ≤ 20 %)")
  P("\n**$(ok3 ? "PASS" : "FAIL")**\n"); P("| eco | species | intervals | deaths | p_emp | p_implied |"); P("|---|---|---|---|---|---|")
  for r in eachrow(first(sort(j, :n_cohort_intervals; rev=true), 15)); P("| $(r.eco) | $(r.species) | $(r.n_cohort_intervals) | $(r.n_deaths) | $(@sprintf("%.4f", r.p_emp)) | $(@sprintf("%.4f", r.pmort)) |"); end
  P("\n(15 largest cells shown.)\n")
end
# ---- 4. establishment
P("## 4. Establishment (Sim B PROB_ESTAB)\n")
if cB === nothing || !haskey(cB, "estab"); P("Not evaluated (no Sim B params, or no data-window table for the config).\n"); verdict[4] = missing
else
  e = cB["estab"]; allin = e.in_window == e.cells_with_data; pinfrac = 100 * e.pinned_low / max(e.cells_with_data, 1); ok4 = allin && pinfrac <= 25
  verdict[4] = ok4
  P("- cells with a data value: $(e.cells_with_data) of $(e.total_cells) (the table covers only the L3s it lists); inside the window [mult × data, 1]: $(e.in_window); pinned at the lower bound: $(e.pinned_low) ($(@sprintf("%.0f", pinfrac)) %, rule ≤ 25 %)")
  P("\n**$(ok4 ? "PASS" : "FAIL")**\n")
end
# ---- 5. Sim B held-out (VAL decides; TEST reported)
P("## 5. Sim B held-out (plot-cluster bootstrap on log Σsim/Σobs): VAL decides, TEST reported\n")
if tB === nothing; P("Not evaluated (no Sim B params).\n"); verdict[5] = missing
else
  P("| split | groups | ±5 % | ±10 % | ±20 % | median abs bias % | cycle bias % |"); P("|---|---|---|---|---|---|---|")
  for sp in ("val", "test"); r = tB[sp]; cb = get(r, "cycle_bias", Dict())
    P("| $sp$(sp == "test" ? " (information only)" : "") | $(r["ng"]) | $(r["pass05"]) | $(r["pass10"]) | $(r["pass20"]) | $(@sprintf("%.1f", r["median_abs_bias"])) | $(join(["c$c: " * @sprintf("%+.1f", v.pct) for (c, v) in sort(collect(cb); by=first)], ", ")) |")
  end
  cbv = get(tB["val"], "cycle_bias", Dict()); b12 = [haskey(cbv, c) ? cbv[c].pct : NaN for c in (1, 2)]
  ok5 = all(abs.(b12) .<= 15)
  verdict[5] = ok5
  P("\nRule (decided on VAL): cycle-1 and cycle-2 total AGB bias within ±15 % (val: $(join([@sprintf("%+.1f", x) for x in b12], ", "))): **$(ok5 ? "PASS" : "FAIL")**; the ±20 % pass counts are the held-out score (no fixed threshold in the design).\n")
end
# ---- 6. identifiability
P("## 6. Identifiability (archive spread, reported)\n")
if haskey(cA, "identifiability")
  P("Behavioural points: ref-normalised F ≤ 1.5 × knee F ($(@sprintf("%.3f", cA["archive_F_knee"])) at the knee).\n"); P("| family | points | dims | median max/min | share of dims > 5× | flag |"); P("|---|---|---|---|---|---|")
  for (f, s) in sort(collect(cA["identifiability"]); by=first); P("| $f | $(s.npoints) | $(s.ndims) | $(@sprintf("%.2f", s.median_ratio)) | $(@sprintf("%.0f", 100 * s.frac_gt5)) % | $(s.median_ratio > 5 ? "not identified" : "") |"); end
  P("")
else P("No archive directory given (`SIMAB_ARCHIVE`).\n"); end
P("## Verdict\n"); allv = [verdict[k] for k in 1:5 if haskey(verdict, k)]
P("| criterion | 1 growth | 2 bounds | 3 mortality | 4 establishment | 5 Sim B |"); P("|---|---|---|---|---|---|")
P("| result | " * join([ismissing(get(verdict, k, missing)) ? "n/a" : verdict[k] ? "PASS" : "FAIL" for k in 1:5], " | ") * " |")
P("\nProper parameterization (1–5 all pass): **$(any(ismissing, allv) ? "incomplete (criteria not evaluated)" : all(allv) ? "YES" : "NO")**")
write(out, String(take!(io))); println("wrote $out")
