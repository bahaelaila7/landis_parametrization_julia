# Priors for a cohort-level HBAD (height / basal area / density) growth model, per species (top-30 + pooled S/H/ALL).
# Writes results/hbad/priors.csv (+ priors_dgrowth_by_dbh.csv, priors_reineke_slope.csv).
#   ./julia_gdal.sh --project=. --threads=2 tools/hbad_priors.jl
include(joinpath(@__DIR__, "hbad_common.jl"))
include(joinpath(@__DIR__, "hbad_translate_lib.jl"))

const REINEKE = 1.605
sdi_metric(tpa_ha, qmd_cm) = tpa_ha * (qmd_cm / 25.4)^REINEKE      # trees/ha at QMD of 25.4 cm (10 in), Reineke slope 1.605

function auc(y::AbstractVector{Bool}, s::AbstractVector{<:Real})
    n1 = sum(y); n0 = length(y) - n1; (n1 == 0 || n0 == 0) && return NaN
    r = zeros(length(s)); o = sortperm(s); i = 1
    while i <= length(s)
        j = i; while j < length(s) && s[o[j+1]] == s[o[i]]; j += 1; end
        r[o[i:j]] .= (i + j) / 2; i = j + 1
    end
    return (sum(r[y]) - n1 * (n1 + 1) / 2) / (n1 * n0)
end

"""Binomial regression with complementary log-log link and log(dt) offset (annual hazard h = exp(Xβ); P(die in dt)=1-exp(-dt h))."""
function cloglog_fit(X::Matrix{Float64}, y::Vector{Float64}, off::Vector{Float64}; maxit=100, ridge=1e-3)
    p = size(X, 2)
    R = ridge * Diagonal([0.0; ones(p - 1)])            # mild ridge on slopes only (intercept free)
    function dev(β)
        η = X * β .+ off; e = exp.(clamp.(η, -30, 5)); μ = clamp.(1 .- exp.(-e), 1e-12, 1 - 1e-12)
        -2 * sum(y .* log.(μ) .+ (1 .- y) .* log.(1 .- μ)) + β' * R * β
    end
    β = zeros(p); β[1] = log(max(sum(y) / sum(exp.(off)), 1e-6)); d0 = dev(β)
    for it in 1:maxit
        η = X * β .+ off; e = exp.(clamp.(η, -30, 5)); μ = clamp.(1 .- exp.(-e), 1e-12, 1 - 1e-12)
        dμ = e .* exp.(-e); w = dμ .^ 2 ./ (μ .* (1 .- μ)); z = (η .- off) .+ (y .- μ) ./ max.(dμ, 1e-12)
        A = X' * (X .* w) + R; βn = A \ (X' * (w .* z))
        step = 1.0; dn = dev(βn)
        while !(dn <= d0) && step > 1e-4; step /= 2; βn = β .+ step .* (βn .- β); dn = dev(βn); end
        δ = maximum(abs.(βn .- β)); β = βn; dchg = abs(d0 - dn); d0 = dn
        (δ < 1e-7 || dchg < 1e-8 * abs(d0)) && break
    end
    return β
end
cloglog_pred(X, β, dt) = 1 .- exp.(-dt .* exp.(clamp.(X * β, -30, 5)))     # P(death in dt years)

function main()
    ## ---------- load ----------
    @info "loading"
    T = hbad_load_trees(); dropmissing!(T, :sym)
    con = hbad_connect()
    D = DataFrame(DBInterface.execute(con, """
      SELECT t.STATECD statecd, t.UNITCD unitcd, t.COUNTYCD countycd, t.PLOT plot, t.SUBP subp, t.TREE tree, t.INVYR invyr, t.MEASDATE measdate, t.tree_fate fate
      FROM s.curated_trees t JOIN sp ON sp.statecd=t.STATECD AND sp.unitcd=t.UNITCD AND sp.countycd=t.COUNTYCD AND sp.plot=t.PLOT
      WHERE t.STATECD IN (12,13) AND t.STATUSCD = 2"""))
    DBInterface.close!(con); D.measdate = Date.(D.measdate)
    coefs = hb_load_coefs(joinpath(HBAD_OUT, "translation_coefs.csv"))
    top = sort([k for k in keys(coefs) if !startswith(k, "_")])
    topset = Set(top)
    T.key = [s in topset ? s : "_" * g for (s, g) in zip(T.sym, T.grp)]
    T.dia_cm = T.dia .* IN2CM; T.h_m = T.ht .* FT2M; T.tpa_ha = T.tpa .* ACRE2HA
    T.ba_ha = (pi / 4) .* (T.dia_cm ./ 100) .^ 2 .* T.tpa_ha
    T.ge5 = T.dia .>= 5.0
    ## ---------- plot-visit stats ----------
    gpv = groupby(T, [:statecd, :unitcd, :countycd, :plot, :invyr]); T.pv = groupindices(gpv)
    npv = maximum(T.pv)
    pv_ba5 = zeros(npv); pv_tpa5 = zeros(npv); pv_n5 = zeros(Int, npv); pv_ba_all = zeros(npv); pv_tpa_all = zeros(npv)
    for i in 1:nrow(T)
        p = T.pv[i]; pv_ba_all[p] += T.ba_ha[i]; pv_tpa_all[p] += T.tpa_ha[i]
        if T.ge5[i]; pv_ba5[p] += T.ba_ha[i]; pv_tpa5[p] += T.tpa_ha[i]; pv_n5[p] += 1; end
    end
    qmd5 = sqrt.(4 .* pv_ba5 ./ (pi .* max.(pv_tpa5, 1e-9))) .* 100
    qmd_all = sqrt.(4 .* pv_ba_all ./ (pi .* max.(pv_tpa_all, 1e-9))) .* 100
    sdi5 = [pv_tpa5[p] > 0 ? sdi_metric(pv_tpa5[p], qmd5[p]) : 0.0 for p in 1:npv]
    sdi_all = [sdi_metric(pv_tpa_all[p], qmd_all[p]) for p in 1:npv]
    # BAL (m2/ha of strictly larger live trees on the plot-visit)
    o = sortperm(collect(zip(T.pv, -T.dia))); T.bal = zeros(nrow(T))
    i = 1
    while i <= length(o)
        j = i; p = T.pv[o[i]]
        while j < length(o) && T.pv[o[j+1]] == p; j += 1; end
        cum = 0.0; k = i
        while k <= j
                m = k; while m < j && T.dia[o[m+1]] == T.dia[o[k]]; m += 1; end
                grp_ba = 0.0
                for q in k:m; T.bal[o[q]] = cum; grp_ba += T.ba_ha[o[q]]; end
                cum += grp_ba; k = m + 1
        end
        i = j + 1
    end
    T.pv_ba5 = pv_ba5[T.pv]
    ## ---------- (c) SDImax by species ----------
    # species BA share (trees >= 5 in) per plot-visit
    shr = combine(groupby(T[T.ge5, :], [:pv, :key]), :ba_ha => sum => :ba)
    shr.tot = pv_ba5[shr.pv]; shr.share = shr.ba ./ shr.tot
    shr.n5 = pv_n5[shr.pv]
    grp_of = Dict(zip(T.sym, T.grp))
    shr_g = combine(groupby(T[T.ge5, :], [:pv, :grp]), :ba_ha => sum => :ba)
    shr_g.share = shr_g.ba ./ pv_ba5[shr_g.pv]; shr_g.n5 = pv_n5[shr_g.pv]
    pv_set = combine(groupby(T, :pv), :set => first => :set).set
    pv_plot = combine(groupby(T, :pv), :plot => first => :plot, :statecd => first => :statecd).plot
    P = Dict{String,Dict{String,Any}}()
    allkeys = vcat(top, ["_S", "_H", "_ALL"])
    for k in allkeys; P[k] = Dict{String,Any}("key" => k); end
    function sdistats!(k, pvs, setname)
        v = sdi5[pvs]
        P[k]["sdimax_p90_$setname"] = quantile(v, 0.90); P[k]["sdimax_p95_$setname"] = quantile(v, 0.95)
        P[k]["sdimax_p99_$setname"] = quantile(v, 0.99); P[k]["sdimax_max_$setname"] = maximum(v)
        P[k]["sdimax_nvisits_$setname"] = length(v)
    end
    for k in top
        sub = shr[(shr.key .== k) .& (shr.share .>= 0.6) .& (shr.n5 .>= 5), :]
        length(sub.pv) >= 20 && (sdistats!(k, sub.pv, "all"); sdistats!(k, sub.pv[pv_set[sub.pv] .== "train"], "train"))
    end
    for g in ["S", "H"]
        sub = shr_g[(shr_g.grp .== g) .& (shr_g.share .>= 0.8) .& (shr_g.n5 .>= 5), :]
        sdistats!("_" * g, sub.pv, "all"); sdistats!("_" * g, sub.pv[pv_set[sub.pv] .== "train"], "train")
    end
    ok = findall(pv_n5 .>= 5); sdistats!("_ALL", ok, "all"); sdistats!("_ALL", ok[pv_set[ok] .== "train"], "train")
    for k in top
        haskey(P[k], "sdimax_p95_all") || (for c in ["p90","p95","p99","max"]; P[k]["sdimax_$(c)_all"] = P["_" * (grp_of[k] == "S" ? "S" : "H")]["sdimax_$(c)_all"]; end;
                                         P[k]["sdimax_nvisits_all"] = 0; P[k]["sdimax_fallback"] = "group")
        P[k]["sdimax_ratio_to_all_p95"] = P[k]["sdimax_p95_all"] / P["_ALL"]["sdimax_p95_all"]
    end
    # empirical Reineke slope: binned P95 of ln(TPA>=5in) vs ln(QMD>=5in) over dominated plots (all species pooled)
    okv = ok[(qmd5[ok] .> 12.7)]
    lq = log.(qmd5[okv]); lt = log.(pv_tpa5[okv]); edges = quantile(lq, range(0, 1, length=16))
    bx = Float64[]; by = Float64[]
    for b in 1:15
        idx = (lq .>= edges[b]) .& (lq .<= edges[b+1]); sum(idx) < 100 && continue
        push!(bx, mean(lq[idx])); push!(by, quantile(lt[idx], 0.95))
    end
    βr = ols(hcat(ones(length(bx)), bx), by)
    println("Empirical self-thinning slope (binned P95 ln TPA vs ln QMD, plots with >=5 trees >=5in): ", round(βr[2], digits=3), "  (Reineke = -1.605)")
    CSV.write(joinpath(HBAD_OUT, "priors_reineke_slope.csv"), DataFrame(bin_lnqmd=bx, p95_lntpa=by, slope=fill(βr[2], length(bx)), intercept=fill(βr[1], length(bx))))
    # plot-level SDImax (species-share weighted P95) -> relative density
    sdimax_key(k) = (haskey(P, k) && haskey(P[k], "sdimax_p95_all")) ? P[k]["sdimax_p95_all"] : P["_ALL"]["sdimax_p95_all"]
    smax_pv = zeros(npv)
    for r in eachrow(shr)
        kk = haskey(P, r.key) ? r.key : "_ALL"
        smax_pv[r.pv] += r.share * sdimax_key(kk)
    end
    smax_pv[smax_pv .== 0] .= P["_ALL"]["sdimax_p95_all"]
    T.rd = sdi5[T.pv] ./ smax_pv[T.pv]
    ## ---------- (a),(b) from the translation fits ----------
    for k in allkeys
        haskey(coefs, k) || continue
        t = coefs[k]["tree"]; P[k]["alloc_alpha"] = t["alpha"]; P[k]["alloc_beta"] = t["beta"]; P[k]["alloc_sigma_ln"] = t["sigma"]
        P[k]["alloc_r2"] = t["r2"]; P[k]["n_trees"] = t["n"]
        h = coefs[k]["hcr"]; P[k]["hd_a"] = h["a"]; P[k]["hd_b"] = h["b"]; P[k]["hd_c"] = h["c"]; P[k]["hd_rmse_m"] = h["rmse"]
    end
    ## ---------- remeasurement pairs ----------
    @info "building remeasurement pairs"
    A = T[:, [:statecd, :unitcd, :countycd, :plot, :subp, :tree, :measdate, :spcd, :dia, :dia_cm, :h_m, :age, :key, :set, :bal, :pv_ba5, :rd, :tpa_ha, :si_ft]]
    A.alive = trues(nrow(A)); A.dead = falses(nrow(A)); A.harv = falses(nrow(A))
    Dd = D[:, [:statecd, :unitcd, :countycd, :plot, :subp, :tree, :measdate]]
    for c in [:spcd, :dia, :dia_cm, :h_m, :age, :bal, :pv_ba5, :rd, :tpa_ha, :si_ft]; Dd[!, c] = fill(NaN, nrow(Dd)); end
    Dd.key = fill("", nrow(Dd)); Dd.set = fill("", nrow(Dd))
    Dd.alive = falses(nrow(Dd)); Dd.dead = D.fate .== "dead"; Dd.harv = D.fate .!= "dead"
    Dd.spcd = Float64.(Dd.spcd); A.spcd = Float64.(A.spcd); A.si_ft = Float64.(coalesce.(A.si_ft, NaN)); Dd.si_ft = Float64.(Dd.si_ft)
    B = vcat(A, Dd[:, names(A)])
    sort!(B, [:statecd, :unitcd, :countycd, :plot, :subp, :tree, :measdate])
    n = nrow(B); i1 = Int[]; i2 = Int[]
    for i in 1:n-1
        B.alive[i] || continue
        (B.statecd[i] == B.statecd[i+1] && B.unitcd[i] == B.unitcd[i+1] && B.countycd[i] == B.countycd[i+1] && B.plot[i] == B.plot[i+1] &&
         B.subp[i] == B.subp[i+1] && B.tree[i] == B.tree[i+1]) || continue
        push!(i1, i); push!(i2, i + 1)
    end
    dt = [(B.measdate[b] - B.measdate[a]).value / 365.25 for (a, b) in zip(i1, i2)]
    keep = (dt .>= 2.0) .& (dt .<= 12.0) .& .!B.harv[i2]
    i1 = i1[keep]; i2 = i2[keep]; dt = dt[keep]
    Pr = B[i1, [:key, :set, :spcd, :dia_cm, :h_m, :age, :bal, :pv_ba5, :rd, :si_ft]]
    Pr.dt = dt; Pr.died = B.dead[i2]; Pr.alive2 = B.alive[i2]
    Pr.dia2_cm = B.dia_cm[i2]; Pr.h2_m = B.h_m[i2]; Pr.spcd2 = B.spcd[i2]
    @info "pairs" n=nrow(Pr) died=sum(Pr.died) alive2=sum(Pr.alive2)
    ## ---------- (d) diameter increment ----------
    G = Pr[Pr.alive2 .&& (Pr.spcd .== Pr.spcd2), :]
    G.g = (G.dia2_cm .- G.dia_cm) ./ G.dt
    G = G[(G.g .>= -0.8) .& (G.g .<= 4.0), :]                  # drop gross measurement errors (> 0.8 cm/yr shrink, > 4 cm/yr growth)
    G.g = max.(G.g, 0.0)
    G.lnD = log.(G.dia_cm ./ 20); G.D20 = G.dia_cm ./ 20; G.lna = log.(G.age ./ 30); G.balx = G.bal ./ 10; G.bax = G.pv_ba5 ./ 10
    gpred(θ, d) = exp.(θ[1] .+ θ[2] .* d.lnD .+ θ[3] .* d.D20 .+ θ[4] .* d.lna .- θ[5] .* d.balx .- θ[6] .* d.bax)
    rows_dbh = NamedTuple[]
    for k in allkeys
        sel = k == "_ALL" ? trues(nrow(G)) : startswith(k, "_") ? [grp_of_key(x, grp_of) == k[2:2] for x in G.key] : (G.key .== k)
        gk = G[sel, :]; tr = gk[gk.set .== "train", :]; va = gk[gk.set .== "val", :]
        P[k]["growth_n_train"] = nrow(tr)
        nrow(tr) < 300 && continue
        if nrow(tr) > 150_000; tr = tr[randperm(MersenneTwister(1), nrow(tr))[1:150_000], :]; end
        # full model (BAL and stand BA) for reference, then the reduced BAL-only model used as the prior (k_ba collinear with k_bal)
        θf, _ = lm_fit(θ -> gpred(θ, tr) .- tr.g, [log(max(mean(tr.g), 0.05)), 0.3, -0.3, 0.0, 0.1, 0.1])
        P[k]["g_full_k_ba"] = θf[6]; P[k]["g_full_k_bal"] = θf[5]
        nrow(va) > 50 && (P[k]["g_full_val_r2"] = r2(va.g, gpred(θf, va)))
        gred(θ, d) = gpred([θ; 0.0], d)
        θr, sse = lm_fit(θ -> gred(θ, tr) .- tr.g, θf[1:5])
        θ = [θr; 0.0]
        for (j, nm) in enumerate(["b0", "b1_lnD", "b2_D", "b3_lnage", "k_bal"]); P[k]["g_" * nm] = θ[j]; end
        pv = gpred(θ, va); pt = gpred(θ, tr)
        pos = tr.g .> 0.02
        P[k]["g_zero_frac"] = mean(tr.g .<= 0.02)
        P[k]["g_sigma_ln_pos"] = std(log.(tr.g[pos]) .- log.(pt[pos]))    # multiplicative scatter among positive increments
        P[k]["g_mean_cm_yr"] = mean(tr.g); P[k]["g_p90_cm_yr"] = quantile(tr.g, 0.9)
        P[k]["g_val_r2"] = nrow(va) > 50 ? r2(va.g, pv) : NaN
        P[k]["g_val_rmse"] = nrow(va) > 50 ? sqrt(mean((va.g .- pv) .^ 2)) : NaN
        P[k]["g_val_n"] = nrow(va)
        for (lo, hi) in ((2.5, 12.7), (12.7, 25.4), (25.4, 38.1), (38.1, 200.0))
            s2 = (gk.dia_cm .>= lo) .& (gk.dia_cm .< hi)
            sum(s2) >= 30 && push!(rows_dbh, (key=k, dbh_lo_cm=lo, dbh_hi_cm=hi, n=sum(s2), mean=mean(gk.g[s2]), median=median(gk.g[s2]), p10=quantile(gk.g[s2], .1), p90=quantile(gk.g[s2], .9), zero_frac=mean(gk.g[s2] .<= 0.02)))
        end
    end
    CSV.write(joinpath(HBAD_OUT, "priors_dgrowth_by_dbh.csv"), DataFrame(rows_dbh))
    ## ---------- (e) mortality ----------
    M = Pr[(Pr.alive2 .|| Pr.died), :]
    M.lnD = log.(M.dia_cm ./ 20); M.D20 = M.dia_cm ./ 20; M.balx = M.bal ./ 10
    mx(d) = hcat(ones(nrow(d)), d.lnD, d.D20, d.rd, d.balx, d.rd .* d.lnD)
    for k in allkeys
        sel = k == "_ALL" ? trues(nrow(M)) : startswith(k, "_") ? [grp_of_key(x, grp_of) == k[2:2] for x in M.key] : (M.key .== k)
        mk = M[sel, :]; tr = mk[mk.set .== "train", :]; va = mk[mk.set .== "val", :]
        P[k]["mort_n_train"] = nrow(tr); P[k]["mort_deaths_train"] = sum(tr.died)
        sum(tr.died) < 100 && continue
        β = cloglog_fit(mx(tr), Float64.(tr.died), log.(tr.dt))
        for (j, nm) in enumerate(["c0", "c1_lnD", "c2_D", "c3_RD", "c4_BAL", "c5_RDxlnD"]); P[k]["m_" * nm] = β[j]; end
        P[k]["mort_annual_rate_train"] = sum(tr.died) / sum(tr.dt)
        if sum(va.died) > 20
            pp = cloglog_pred(mx(va), β, va.dt)
            P[k]["mort_val_auc"] = auc(va.died, pp)
            P[k]["mort_val_obs_over_pred"] = sum(va.died) / sum(pp)
            P[k]["mort_val_deaths"] = sum(va.died)
        end
    end
    # empirical annual mortality hazard by relative density x DIA class (all plots, deaths/(tree-years))
    mt = NamedTuple[]
    rdb = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 10.0]; dcl = [0.0, 12.7, 25.4, 38.1, 500.0]
    for k in allkeys
        sel = k == "_ALL" ? trues(nrow(M)) : startswith(k, "_") ? [grp_of_key(x, grp_of) == k[2:2] for x in M.key] : (M.key .== k)
        mk = M[sel, :]
        for a in 1:length(rdb)-1, b in 1:length(dcl)-1
            c = (mk.rd .>= rdb[a]) .& (mk.rd .< rdb[a+1]) .& (mk.dia_cm .>= dcl[b]) .& (mk.dia_cm .< dcl[b+1])
            sum(c) >= 50 && push!(mt, (key=k, rd_lo=rdb[a], rd_hi=rdb[a+1], dbh_lo_cm=dcl[b], dbh_hi_cm=dcl[b+1], n=sum(c), deaths=sum(mk.died[c]),
                                       annual_hazard=sum(mk.died[c]) / sum(mk.dt[c])))
        end
    end
    CSV.write(joinpath(HBAD_OUT, "priors_mortality_table.csv"), DataFrame(mt))
    ## ---------- (f) height increment ----------
    H = Pr[Pr.alive2 .&& (Pr.spcd .== Pr.spcd2), :]
    H.gh = (H.h2_m .- H.h_m) ./ H.dt
    H = H[(H.gh .>= -2.0) .& (H.gh .<= 3.0), :]
    H.lnH = log.(H.h_m ./ 15); H.lna = log.(H.age ./ 30); H.balx = H.bal ./ 10
    hp(θ, d) = exp.(θ[1] .+ θ[2] .* d.lnH .+ θ[3] .* d.lna .- θ[4] .* d.balx)
    for k in allkeys
        sel = k == "_ALL" ? trues(nrow(H)) : startswith(k, "_") ? [grp_of_key(x, grp_of) == k[2:2] for x in H.key] : (H.key .== k)
        hk = H[sel, :]; tr = hk[hk.set .== "train", :]; va = hk[hk.set .== "val", :]
        P[k]["hg_n_train"] = nrow(tr)
        nrow(tr) < 300 && continue
        if nrow(tr) > 150_000; tr = tr[randperm(MersenneTwister(2), nrow(tr))[1:150_000], :]; end
        θ, sse = lm_fit(θ -> hp(θ, tr) .- tr.gh, [log(max(mean(tr.gh), 0.05)), 0.0, 0.0, 0.05])
        for (j, nm) in enumerate(["h0", "h1_lnH", "h2_lnage", "k_bal"]); P[k]["hg_" * nm] = θ[j]; end
        P[k]["hg_mean_m_yr"] = mean(tr.gh)
        if nrow(va) > 50
            P[k]["hg_val_r2"] = r2(va.gh, hp(θ, va))
            # H-D implied: dH/dD(CR) * observed dD
            if haskey(coefs, k)
                c = coefs[k]["hcr"]; dHdD = c["a"] * c["c"] * c["b"] .* exp.(-c["b"] .* va.dia_cm) .* (1 .- exp.(-c["b"] .* va.dia_cm)) .^ (c["c"] - 1)
                P[k]["hg_val_r2_via_HD"] = r2(va.gh, dHdD .* (va.dia2_cm .- va.dia_cm) ./ va.dt)
            end
        end
    end
    ## ---------- write ----------
    cols = unique(vcat(["key"], reduce(vcat, [collect(keys(P[k])) for k in allkeys])))
    cols = vcat(["key"], sort(setdiff(cols, ["key"])))
    rows = [[get(P[k], c, missing) for c in cols] for k in allkeys]
    df = DataFrame(permutedims(reduce(hcat, rows)), cols)
    CSV.write(joinpath(HBAD_OUT, "priors.csv"), df)
    # console summary for the main species
    want = ["PITA", "PIEL", "PIPA2", "LIST2", "QUNI", "ACRU", "LITU", "QUAL", "NYBI", "QULA3", "_S", "_H", "_ALL"]
    show_cols = ["alloc_alpha", "alloc_beta", "hd_a", "hd_b", "hd_c", "sdimax_p95_all", "sdimax_p99_all", "sdimax_max_all", "sdimax_nvisits_all",
                 "g_b0", "g_b1_lnD", "g_b2_D", "g_b3_lnage", "g_k_bal", "g_full_k_ba", "g_full_val_r2", "g_sigma_ln_pos", "g_zero_frac", "g_mean_cm_yr", "g_val_r2",
                 "m_c0", "m_c1_lnD", "m_c2_D", "m_c3_RD", "m_c4_BAL", "m_c5_RDxlnD", "mort_annual_rate_train", "mort_val_auc", "mort_val_obs_over_pred",
                 "hg_h0", "hg_h1_lnH", "hg_h2_lnage", "hg_k_bal", "hg_val_r2", "hg_val_r2_via_HD"]
    for k in want
        println("\n== ", k)
        for c in show_cols
            v = get(P[k], c, missing); v isa Number && print(c, "=", round(v, sigdigits=4), "  ")
        end
        println()
    end
end

grp_of_key(key, grp_of) = startswith(key, "_") ? key[2:2] : grp_of[key]
main()
