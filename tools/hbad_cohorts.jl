# Build the cohort table (plot, subplot, species, round(age), visit) for GA+FL from FIA tree data.
# Usage: ./julia_gdal.sh --project=. --threads=2 tools/hbad_cohorts.jl
include(joinpath(@__DIR__, "hbad_common.jl"))
using Dates

function build_cohorts(trees::DataFrame)
    t = trees
    t.age_c = Int.(round.(t.age))
    t.age_c = max.(t.age_c, 1)
    t.tpa_ha = t.tpa .* ACRE2HA
    t.ba_ha  = (pi/4) .* (t.dia .* 0.0254) .^ 2 .* t.tpa_ha
    t.agb    = t.bio .* t.tpa .* LB
    t.h_m    = t.ht .* FT2M
    t.hw     = t.h_m .* t.tpa_ha
    t.si_m   = t.si_ft .* FT2M
    t.pl     = (coalesce.(t.stdorg, 0.0) .== 1.0)
    t.sap    = t.dia .< 5.0
    t.sap_tpa = t.tpa_ha .* t.sap
    keys = [:statecd, :unitcd, :countycd, :plot, :subp, :invyr, :sym, :age_c]
    g = groupby(t, keys)
    # dominant ("tallest 40%" by stem count) height: tallest trees, TPA-weighted, up to 40% of the cohort's stems
    function hdom(h, w)
        n = length(h); n == 1 && return h[1]
        o = sortperm(h; rev=true); W = sum(w); target = 0.4W; acc = 0.0; s = 0.0
        for i in o
            take = min(w[i], target - acc); take <= 0 && break
            s += take * h[i]; acc += take
        end
        return s / acc
    end
    out = combine(g,
        :measdate => maximum => :measdate,
        :grp => first => :grp, :spcd => first => :spcd,
        :agb => sum => :agb_g_m2,
        :tpa_ha => sum => :tpa_ha,
        :ba_ha => sum => :ba_m2ha,
        :hw => sum => :hw,
        [:h_m, :tpa_ha] => hdom => :h_dom_m,
        :dia => length => :n_trees,
        :dia => maximum => :dia_max_in,
        :si_m => (x -> (v = collect(skipmissing(x)); isempty(v) ? missing : mean(v))) => :si_m,
        :siteclcd => first => :siteclcd,
        :epa_l3 => first => :epa_l3,
        :pl => maximum => :planted,
        :sap_tpa => sum => :sap_tpa,
        :set => first => :set, :eco => first => :eco)
    out.h_mean_m = out.hw ./ out.tpa_ha
    out.qmd_cm = sqrt.(4 .* out.ba_m2ha ./ (pi .* out.tpa_ha)) .* 100
    out.sap_frac = out.sap_tpa ./ out.tpa_ha
    select!(out, Not([:hw, :sap_tpa]))
    rename!(out, :sym => :species, :age_c => :age)
    out.planted = Int.(out.planted)
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    @info "loading trees"
    trees = hbad_load_trees()
    @info "trees" n=nrow(trees) plots=length(unique(zip(trees.statecd, trees.unitcd, trees.countycd, trees.plot)))
    dropmissing!(trees, [:sym])
    coh = build_cohorts(trees)
    mkpath(HBAD_OUT)
    Arrow.write(joinpath(HBAD_OUT, "cohorts_gafl.arrow"), coh; compress=:zstd)
    CSV.write(joinpath(HBAD_OUT, "cohorts_gafl_head.csv"), first(coh, 2000))
    println("cohort rows: ", nrow(coh), "  per set: ", combine(groupby(coh, :set), nrow => :n))
    println("cohorts with n_trees==1: ", round(mean(coh.n_trees .== 1), digits=3), "  with saplings: ", round(mean(coh.sap_frac .> 0), digits=3))
    println(describe(coh[:, [:agb_g_m2, :tpa_ha, :ba_m2ha, :qmd_cm, :h_mean_m, :h_dom_m, :age, :si_m]], :min, :q25, :median, :q75, :max, :nmissing))
    sm = combine(groupby(coh, :species), nrow => :n_cohorts, :agb_g_m2 => sum => :agb_total,
                 :age => median => :age_med, :tpa_ha => median => :tpa_med, :qmd_cm => median => :qmd_med, :h_mean_m => median => :h_med)
    sort!(sm, :agb_total, rev=true); sm.agb_share = sm.agb_total ./ sum(sm.agb_total)
    println("top 30 species by total cohort AGB"); show(stdout, first(sm, 30), allrows=true); println()
    println("distinct species: ", nrow(sm))
end
