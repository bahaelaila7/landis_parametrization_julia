# Fit the empirical cohort translation (species, age, AGB[, SI, planted]) -> (TPA, QMD, BA, H) on TRAIN plots,
# select model forms on VAL, report VAL and TEST.  Usage:
#   ./julia_gdal.sh --project=. --threads=2 tools/hbad_translation.jl
# Output: results/hbad/translation_{coefs,selection,metrics}.csv, species_group.csv
include(joinpath(@__DIR__, "hbad_common.jl"))
include(joinpath(@__DIR__, "hbad_translate_lib.jl"))

const TERMSETS = Dict(
    "V0" => ["1", "x1", "x2"],
    "V2" => ["1", "x1", "x2", "pl"],
    "V3" => ["1", "x1", "x2", "x1sq", "x1x2", "x2sq"],
    "V4" => ["1", "x1", "x2", "x1sq", "x1x2", "x2sq", "pl"])
const TARGETS = Dict("tpa" => :lntpa, "qmd" => :lnqmd, "ba" => :lnba, "h" => :lnh)

feat_matrix(df, terms) = hcat([[hb_term(t, df.lnagb[i], df.lnage[i], coalesce(df.si_m[i], 25.0), df.planted[i]) for i in 1:nrow(df)] for t in terms]...)

function fit_terms(df, ycol, terms)
    X = feat_matrix(df, terms); y = df[!, ycol]
    β = ols(X, y); res = y .- X * β
    d = Dict{String,Float64}(t => β[j] for (j, t) in enumerate(terms))
    d["sigma"] = std(res); d["n"] = Float64(length(y))
    return d
end

function chapman_richards(D, H)
    hmax = maximum(H)
    f(θ) = (θ[1] .* (1 .- exp.(-abs(θ[2]) .* D)) .^ abs(θ[3])) .- (H .- 1.3)
    θ, sse = lm_fit(f, [1.2 * hmax, 0.03, 1.3])
    return Dict("a" => θ[1], "b" => abs(θ[2]), "c" => abs(θ[3]), "rmse" => sqrt(sse / length(H)), "n" => Float64(length(H)))
end

function main()
    @info "loading data"
    coh = DataFrame(Arrow.Table(joinpath(HBAD_OUT, "cohorts_gafl.arrow")))
    trees = hbad_load_trees(); dropmissing!(trees, :sym)
    mkpath(HBAD_OUT)
    # species -> group map (all species seen)
    sg = unique(select(trees, :sym => :species, :grp))
    sg = sort!(combine(groupby(sg, :species), :grp => first => :grp), :species)   # B37: deterministic row order
    CSV.write(joinpath(HBAD_OUT, "species_group.csv"), sg)
    ## cohort features
    coh.lnagb = log.(coh.agb_g_m2); coh.lnage = log.(Float64.(coh.age))
    coh.lntpa = log.(coh.tpa_ha); coh.lnqmd = log.(coh.qmd_cm); coh.lnba = log.(coh.ba_m2ha); coh.lnh = log.(coh.h_mean_m)
    coh.lnm = log.(coh.agb_g_m2 ./ (coh.tpa_ha ./ ACRE2HA .* LB))     # ln mean tree mass (lb)
    coh.lnqmd_in = log.(coh.qmd_cm ./ 2.54)
    coh.planted = Float64.(coh.planted)
    # keys: top-30 species by train cohort count + pooled S/H
    tr = coh[coh.set .== "train", :]
    cnt = combine(groupby(tr, :species), nrow => :n); sort!(cnt, [order(:n, rev=true), :species])   # B37: ties by name
    top = cnt.species[1:30]; topset = Set(top)
    coh.key = [s in topset ? s : "_" * g for (s, g) in zip(coh.species, coh.grp)]
    trees.lndia = log.(trees.dia); trees.lnbio = log.(trees.bio); trees.lnage = log.(trees.age)
    trees.si_m = trees.si_ft .* FT2M
    trees.key = [s in topset ? s : "_" * g for (s, g) in zip(trees.sym, trees.grp)]
    ttr = trees[trees.set .== "train", :]
    C = HBCoefs()
    keysets = Dict{String,Tuple{DataFrame,DataFrame}}()
    for s in top
        keysets[s] = (tr[tr.species .== s, :], ttr[ttr.sym .== s, :])
    end
    for g in ["S", "H"]
        keysets["_" * g] = (tr[tr.grp .== g, :], ttr[ttr.grp .== g, :])
    end
    keysets["_ALL"] = (tr, ttr)
    @info "fitting" nkeys=length(keysets)
    for (key, (c, t)) in keysets
        M = get!(C, key, Dict{String,Dict{String,Float64}}())
        nc = nrow(c)
        # --- tree allometry ln(bio lb) = alpha + beta ln(dia in)
        X = hcat(ones(nrow(t)), t.lndia); β = ols(X, t.lnbio); res = t.lnbio .- X * β
        M["tree"] = Dict("alpha" => β[1], "beta" => β[2], "sigma" => std(res), "r2" => r2(t.lnbio, X * β), "n" => Float64(nrow(t)))
        # --- cohort-level mass-QMD relation ln m_lb = a + b ln QMD_in  (absorbs Jensen effect)
        X = hcat(ones(nc), c.lnqmd_in); β = ols(X, c.lnm)
        M["mq"] = Dict("a" => β[1], "b" => β[2], "sigma" => std(c.lnm .- X * β), "n" => Float64(nc))
        # --- tree height-diameter: Chapman-Richards (D cm, H m)
        D = t.dia .* IN2CM; H = t.ht .* FT2M
        M["hcr"] = chapman_richards(D, H)
        # --- log-linear H-D with age (and SI)
        X = hcat(ones(nrow(t)), log.(D), t.lnage); lh = log.(H); β = ols(X, lh)
        M["hll"] = Dict("a" => β[1], "b" => β[2], "c" => β[3], "sigma" => std(lh .- X * β), "n" => Float64(nrow(t)))
        ts = t[.!ismissing.(t.si_m), :]
        if nrow(ts) >= 200
            Ds = ts.dia .* IN2CM; si = Float64.(ts.si_m) .- 25.0
            X = hcat(ones(nrow(ts)), log.(Ds), ts.lnage, si); lh = log.(ts.ht .* FT2M); β = ols(X, lh)
            M["hllsi"] = Dict("a" => β[1], "b" => β[2], "c" => β[3], "d" => β[4], "sigma" => std(lh .- X * β), "n" => Float64(nrow(ts)))
        end
        # --- cohort-calibrated H (true QMD as regressor)
        X = hcat(ones(nc), c.lnqmd, c.lnage); β = ols(X, c.lnh)
        M["hcal"] = Dict("a" => β[1], "b" => β[2], "c" => β[3], "sigma" => std(c.lnh .- X * β), "n" => Float64(nc))
        cs = c[.!ismissing.(c.si_m), :]
        if nrow(cs) >= 200
            X = hcat(ones(nrow(cs)), cs.lnqmd, cs.lnage, Float64.(cs.si_m) .- 25.0); β = ols(X, cs.lnh)
            M["hcalsi"] = Dict("a" => β[1], "b" => β[2], "c" => β[3], "d" => β[4], "sigma" => std(cs.lnh .- X * β), "n" => Float64(nrow(cs)))
        end
        # --- dominant/mean height ratio
        mt = c[c.n_trees .> 1, :]
        M["hdom"] = Dict("ratio" => nrow(mt) > 50 ? median(mt.h_dom_m ./ mt.h_mean_m) : 1.0)
        # --- cohort regressions for all targets x variants (and +SI variants)
        for (tg, ycol) in TARGETS, (vn, terms) in TERMSETS
            M[tg * "_" * vn] = fit_terms(c, ycol, terms)
            if nrow(cs) >= 200
                M[tg * "_" * vn * "si"] = fit_terms(cs, ycol, vcat(terms, ["si"]))
            end
        end
    end
    @info "fit done; selecting on val"

    # ---------- evaluation helpers ----------
    val = coh[coh.set .== "val", :]; tst = coh[coh.set .== "test", :]
    function predict(df, C, sel; use_si=true)
        out = DataFrame(tpa=zeros(nrow(df)), qmd=zeros(nrow(df)), ba=zeros(nrow(df)), h=zeros(nrow(df)), hdom=zeros(nrow(df)), agbchk=zeros(nrow(df)))
        for i in 1:nrow(df)
            r = hbad_translate(df.species[i], df.age[i], df.agb_g_m2[i]; si=(use_si ? df.si_m[i] : missing), planted=df.planted[i], grp=df.grp[i], C=C, sel=sel)
            out.tpa[i] = r.tpa_ha; out.qmd[i] = r.qmd_cm; out.ba[i] = r.ba_m2ha; out.h[i] = r.h_mean_m; out.hdom[i] = r.h_dom_m; out.agbchk[i] = r.agb_check_g_m2
        end
        return out
    end
    HB_GRP[] = Dict(String(r.species) => String(r.grp) for r in eachrow(sg))
    # R2 on log scale / median abs % error for each variable
    function metr(df, p)
        d = Dict{String,Tuple{Float64,Float64}}()
        d["tpa"] = (r2(log.(df.tpa_ha), log.(p.tpa)), mdape(df.tpa_ha, p.tpa))
        d["ba"]  = (r2(log.(df.ba_m2ha), log.(p.ba)), mdape(df.ba_m2ha, p.ba))
        d["qmd"] = (r2(log.(df.qmd_cm), log.(p.qmd)), mdape(df.qmd_cm, p.qmd))
        d["h"]   = (r2(log.(df.h_mean_m), log.(p.h)), mdape(df.h_mean_m, p.h))
        d["h_dom"] = (r2(log.(df.h_dom_m), log.(p.hdom)), mdape(df.h_dom_m, p.hdom))
        d["agb_consistency"] = (NaN, mdape(df.agb_g_m2, p.agbchk))
        return d
    end

    # (1) per-target regression-variant choice by standalone val MdAPE (no structure effects)
    sel = Dict{String,String}("structure" => "C", "mass" => "tree", "hmodel" => "dir")
    choice = Dict{String,String}(); sicoef_keep = Dict{String,Bool}()
    for tg in ["tpa", "qmd", "ba", "h"]
        best = ("", Inf)
        for vn in keys(TERMSETS)
            vals = Float64[]
            ycol = TARGETS[tg]; truth = exp.(val[!, ycol])
            pred = [exp(hb_reg(C, val.key[i], tg * "_" * vn, val.lnagb[i], val.lnage[i], 25.0, val.planted[i])) for i in 1:nrow(val)]
            tg == "tpa" && (pred = max.(pred, HB_MIN_TPA_HA))
            m = mdape(truth, pred)
            @info "variant" tg vn mdape=round(m, digits=3)
            m < best[2] && (best = (vn, m))
        end
        choice[tg] = best[1]
        # SI add-on check on val rows that have SI
        vs = val[.!ismissing.(val.si_m), :]
        ycol = TARGETS[tg]; truth = exp.(vs[!, ycol]); vn = best[1]
        p0 = [exp(hb_reg(C, vs.key[i], tg * "_" * vn, vs.lnagb[i], vs.lnage[i], 25.0, vs.planted[i])) for i in 1:nrow(vs)]
        p1 = [begin mn = tg * "_" * vn; haskey(C[vs.key[i]], mn * "si") ? exp(hb_reg(C, vs.key[i], mn * "si", vs.lnagb[i], vs.lnage[i], vs.si_m[i], vs.planted[i])) : p0[i] end for i in 1:nrow(vs)]
        if tg == "tpa"; p0 = max.(p0, HB_MIN_TPA_HA); p1 = max.(p1, HB_MIN_TPA_HA); end
        m0, m1 = mdape(truth, p0), mdape(truth, p1)
        sicoef_keep[tg] = m1 < 0.99 * m0
        @info "SI add-on" tg m_noSI=round(m0, digits=3) m_SI=round(m1, digits=3) keep=sicoef_keep[tg]
    end
    # (2) structure + mass relation + H model, chosen on val with chosen variants
    for tg in ["tpa","qmd","ba","h"]; sel["v_" * tg] = choice[tg]; end; sel["hmodel"] = "dir"
    # drop SI variants of targets for which they did not help (so lib ignores them)
    for (key, M) in C, tg in keys(sicoef_keep), vn in keys(TERMSETS)
        sicoef_keep[tg] || delete!(M, tg * "_" * vn * "si")
    end
    vsub = val[1:min(nrow(val), 120_000), :]   # speed
    results = Dict{String,Any}()
    for st in ["A", "B", "C"], mass in (st == "C" ? ["tree"] : ["tree", "mq"])
        s = copy(sel); s["structure"] = st; s["mass"] = mass; s["hmodel"] = "cr"
        p = predict(vsub, C, s; use_si=false)
        m = metr(vsub, p)
        score = m["tpa"][2] + m["ba"][2] + m["qmd"][2]
        results[st * "/" * mass] = (m, score)
        @info "structure" st mass tpa=round(m["tpa"][2], digits=2) ba=round(m["ba"][2], digits=2) qmd=round(m["qmd"][2], digits=2) agbc=round(m["agb_consistency"][2], digits=2) score=round(score, digits=2)
    end
    consistent = [k for k in keys(results) if !startswith(k, "C")]
    bestc = consistent[argmin([results[k][2] for k in consistent])]
    chosen = results["C/tree"][2] < 0.9 * results[bestc][2] ? "C/tree" : bestc
    st, mass = split(chosen, "/")
    sel["structure"] = String(st); sel["mass"] = String(mass)
    @info "chosen structure" chosen consistent_best=bestc
    # (3) H model
    hbest = ("", Inf)
    for hm in ["cr", "ll", "cal", "dir"]
        s = copy(sel); s["hmodel"] = hm
        p = predict(vsub, C, s; use_si=false); m = metr(vsub, p)
        @info "H model" hm mdape=round(m["h"][2], digits=3) r2=round(m["h"][1], digits=4)
        m["h"][2] < hbest[2] && (hbest = (hm, m["h"][2]))
    end
    sel["hmodel"] = hbest[1]
    # (4) height model to use when a site index is supplied: best of {cr(no SI), ll+SI, cal+SI, dir+SI} on val rows with SI
    vs = vsub[.!ismissing.(vsub.si_m), :]
    hs = Dict{String,Float64}()
    p0 = predict(vs, C, sel; use_si=false); hs["cr_or_chosen_noSI"] = mdape(vs.h_mean_m, p0.h)
    for hm in ["ll", "cal", "dir"]
        s2 = copy(sel); s2["hmodel"] = hm; s2["hmodel_si"] = hm
        p1 = predict(vs, C, s2; use_si=true); hs[hm * "+SI"] = mdape(vs.h_mean_m, p1.h)
    end
    @info "H with SI (val rows with SI)" hs
    bestsi = argmin(hs)
    if bestsi == "cr_or_chosen_noSI"
        sel["hmodel_si"] = sel["hmodel"]
    else
        sel["hmodel_si"] = replace(bestsi, "+SI" => "")
    end
    # prune coefficient sets that the final selection does not use
    used = Set(["tree", "mq", "hcr", "hdom"])
    for tg in ["tpa", "qmd", "ba", "h"]
        push!(used, tg * "_" * sel["v_" * tg]); sicoef_keep[tg] && push!(used, tg * "_" * sel["v_" * tg] * "si")
    end
    for hm in (sel["hmodel"], sel["hmodel_si"])
        hm == "ll" && (push!(used, "hll"); push!(used, "hllsi"))
        hm == "cal" && (push!(used, "hcal"); push!(used, "hcalsi"))
    end
    # keep the full regression library in a separate file for reference, prune the working one
    allrows = NamedTuple[]
    for (key, M) in sort(collect(C), by=first), (mod, d) in sort(collect(M), by=first), (term, v) in sort(collect(d), by=first)
        push!(allrows, (key=key, model=mod, term=term, value=v))
    end
    CSV.write(joinpath(HBAD_OUT, "translation_coefs_all_variants.csv"), DataFrame(allrows))
    for (key, M) in C, mod in collect(keys(M))
        mod in used || delete!(M, mod)
    end
    @info "final selection" sel

    ## ---------- write coefficients ----------
    rows = NamedTuple[]
    for (key, M) in sort(collect(C), by=first), (mod, d) in sort(collect(M), by=first), (term, v) in sort(collect(d), by=first)
        push!(rows, (key=key, model=mod, term=term, value=v))
    end
    CSV.write(joinpath(HBAD_OUT, "translation_coefs.csv"), DataFrame(rows))
    CSV.write(joinpath(HBAD_OUT, "translation_selection.csv"), DataFrame(item=collect(keys(sel)), choice=collect(values(sel))))

    ## ---------- final metrics (full val & test; with and without SI/planted info) ----------
    mrows = NamedTuple[]
    function add_metrics!(setname, df, p, grpname, variant)
        for (v, (rr, md)) in metr(df, p)
            push!(mrows, (set=setname, subset=grpname, variant=variant, n=nrow(df), var=v, r2_log=rr, mdape_pct=md))
        end
    end
    HB_COEFS[] = C; HB_SEL[] = sel
    for (setname, df) in (("val", val), ("test", tst))
        p = predict(df, C, sel; use_si=false)
        add_metrics!(setname, df, p, "overall", "no_SI")
        hasSI = .!ismissing.(df.si_m)
        ps = predict(df[hasSI, :], C, sel; use_si=true)
        add_metrics!(setname, df[hasSI, :], ps, "rows_with_SI", "with_SI")
        add_metrics!(setname, df[hasSI, :], p[hasSI, :], "rows_with_SI", "no_SI")
        add_metrics!(setname, df[df.n_trees .> 1, :], p[df.n_trees .> 1, :], "multi_tree_cohorts", "no_SI")
        add_metrics!(setname, df[df.n_trees .== 1, :], p[df.n_trees .== 1, :], "single_tree_cohorts", "no_SI")
        add_metrics!(setname, df[df.sap_frac .== 0, :], p[df.sap_frac .== 0, :], "no_saplings", "no_SI")
        add_metrics!(setname, df[df.sap_frac .> 0, :], p[df.sap_frac .> 0, :], "with_saplings", "no_SI")
        add_metrics!(setname, df[df.planted .== 1, :], p[df.planted .== 1, :], "planted", "no_SI")
        add_metrics!(setname, df[df.planted .== 0, :], p[df.planted .== 0, :], "natural", "no_SI")
        for k in unique(df.key)
            idx = df.key .== k
            sum(idx) >= 100 && add_metrics!(setname, df[idx, :], p[idx, :], "sp:" * k, "no_SI")
        end
        if setname == "val"
            for st in ["A", "B", "C"], mass in (st == "C" ? ["tree"] : ["tree", "mq"])
                s = copy(sel); s["structure"] = st; s["mass"] = mass
                pa = predict(df, C, s; use_si=false)
                add_metrics!("val", df, pa, "overall", "alt_structure_$(st)_$(mass)")
            end
        end
    end
    CSV.write(joinpath(HBAD_OUT, "translation_metrics.csv"), DataFrame(mrows))
    # tree-level allometry diagnostics on val trees
    vt = trees[trees.set .== "val", :]
    ar = Float64[]; hr = Float64[]
    prd = [C[vt.key[i]]["tree"] for i in 1:nrow(vt)]
    pa = [d["alpha"] + d["beta"] * vt.lndia[i] for (i, d) in enumerate(prd)]
    println("VAL tree allometry: R2(log bio) = ", round(r2(vt.lnbio, pa), digits=4), "  MdAPE bio = ", round(mdape(vt.bio, exp.(pa)), digits=2), "%")
    ph = [begin c = C[vt.key[i]]["hcr"]; 1.3 + c["a"] * (1 - exp(-c["b"] * vt.dia[i] * IN2CM))^c["c"] end for i in 1:nrow(vt)]
    println("VAL tree H-D (Chapman-Richards): R2(log H) = ", round(r2(log.(vt.ht .* FT2M), log.(ph)), digits=4), "  MdAPE H = ", round(mdape(vt.ht .* FT2M, ph), digits=2), "%")
    m = DataFrame(mrows)
    for s in ("val", "test")
        println("\n=== overall ", s, " (no SI) ==="); show(stdout, m[(m.set .== s) .& (m.subset .== "overall") .& (m.variant .== "no_SI"), [:var, :n, :r2_log, :mdape_pct]]; allrows=true); println()
        println("=== ", s, " rows with site index: no SI vs with SI ==="); show(stdout, m[(m.set .== s) .& (m.subset .== "rows_with_SI"), [:variant, :var, :n, :r2_log, :mdape_pct]]; allrows=true); println()
    end
    println("\n=== val structure comparison ==="); show(stdout, m[startswith.(m.variant, "alt_") .& (m.var .∈ Ref(["tpa", "ba", "qmd", "agb_consistency"])), [:variant, :var, :r2_log, :mdape_pct]]; allrows=true); println()
    println("\n=== subsets, test ==="); show(stdout, m[(m.set .== "test") .& (m.variant .== "no_SI") .& (.!startswith.(m.subset, "sp:")) .& (m.subset .!= "overall"), [:subset, :var, :n, :r2_log, :mdape_pct]]; allrows=true); println()
    sp = m[(m.set .== "test") .& startswith.(m.subset, "sp:") .& (m.variant .== "no_SI") .& (m.var .∈ Ref(["tpa", "ba", "h"])), :]
    sp = unstack(sp[:, [:subset, :n, :var, :mdape_pct]], [:subset, :n], :var, :mdape_pct)
    println("\n=== per species TEST MdAPE% ==="); show(stdout, sort(sp, :n, rev=true); allrows=true); println()
end
main()
