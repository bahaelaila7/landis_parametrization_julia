# Shared helpers for the HBAD (height / basal area / density) empirical translation tools.
# Read-only: attaches slim_r4.duckdb READ_ONLY and reads the GA+FL train/val/test split.
using DuckDB, DataFrames, CSV, Statistics, Arrow, LinearAlgebra, Random, Dates

const HBAD_DB    = "/workspace/landis_region4/data_r4/slim_r4.duckdb"
const HBAD_SPLIT = "/workspace/landis_region4/runs_local/gaflst_split/split_plots.csv"
const HBAD_OUT   = joinpath(@__DIR__, "..", "results", "hbad")
const LB   = 453.592 / 4046.86         # lb/acre -> g/m2 (Pan's unit factor)
const ACRE2HA = 2.47105                # trees/acre -> trees/ha
const FT2M = 0.3048
const IN2CM = 2.54

"""DuckDB connection with slim_r4 attached READ_ONLY as `s` and the split table as `sp`."""
function hbad_connect()
    con = DBInterface.connect(DuckDB.DB, ":memory:")
    DBInterface.execute(con, "ATTACH '$(HBAD_DB)' AS s (READ_ONLY)")
    sp = CSV.read(HBAD_SPLIT, DataFrame)
    DBInterface.execute(con, "CREATE TABLE sp(statecd BIGINT, unitcd BIGINT, countycd BIGINT, plot BIGINT, eco VARCHAR, \"set\" VARCHAR, dom VARCHAR)")
    app = DuckDB.Appender(con, "sp")
    for r in eachrow(sp)
        DuckDB.append(app, r.statecd); DuckDB.append(app, r.unitcd); DuckDB.append(app, r.countycd); DuckDB.append(app, r.plot)
        DuckDB.append(app, String(r.eco)); DuckDB.append(app, String(r.set)); DuckDB.append(app, String(r.dom))
        DuckDB.end_row(app)
    end
    DuckDB.close(app)
    return con
end

"""Live-tree table for GA(13)+FL(12): one row per tree-visit with DIA>0, HT>0, estimated_age>0.
Adds set (train/val/test, from split_plots.csv; plots not in the split are dropped), species symbol and group."""
function hbad_load_trees(; alive_only=true)
    con = hbad_connect()
    st = alive_only ? "t.STATUSCD = 1" : "TRUE"
    df = DataFrame(DBInterface.execute(con, """
      SELECT t.STATECD statecd, t.UNITCD unitcd, t.COUNTYCD countycd, t.PLOT plot, t.SUBP subp, t.TREE tree,
             t.INVYR invyr, t.MEASDATE measdate, t.STATUSCD statuscd, t.tree_fate fate,
             t.spcd_resolved spcd, r.SPECIES_SYMBOL sym, r.SFTWD_HRDWD grp,
             t.DIA dia, t.HT ht, t.CR cr, t.CCLCD cclcd, t.DRYBIO_AG bio, t.TPA_UNADJ tpa, t.estimated_age age,
             t.cond_sicond si_ft, t.cond_siteclcd siteclcd, t.cond_stdorgcd stdorg, t.epa_l3, sp.eco, sp."set" "set", sp.dom
      FROM s.curated_trees t
      JOIN sp ON sp.statecd=t.STATECD AND sp.unitcd=t.UNITCD AND sp.countycd=t.COUNTYCD AND sp.plot=t.PLOT
      LEFT JOIN s.REF_SPECIES r ON r.SPCD = t.spcd_resolved
      WHERE t.STATECD IN (12,13) AND $st AND t.DIA > 0 AND t.HT > 0 AND t.estimated_age > 0
    """))
    DBInterface.close!(con)
    df.measdate = Date.(df.measdate)
    return df
end

# ---------- tiny least-squares helpers (no GLM dependency) ----------
"""OLS via QR on column-stacked X (n×p) and y; returns coef. Optional ridge λ on non-intercept columns."""
function ols(X::AbstractMatrix, y::AbstractVector; ridge=1e-8)
    A = Matrix{Float64}(X); b = Vector{Float64}(y)
    XtX = A' * A; Xty = A' * b
    for j in 2:size(A, 2); XtX[j, j] += ridge * max(1.0, XtX[j, j]); end
    return XtX \ Xty
end
r2(y, yhat) = 1 - sum(abs2, y .- yhat) / sum(abs2, y .- mean(y))
"""Levenberg–Marquardt non-linear least squares with a forward-difference Jacobian. f(θ)->residual vector."""
function lm_fit(f, θ0; maxit=200, tol=1e-10)
    θ = Float64.(θ0); λ = 1e-3
    r = f(θ); sse = sum(abs2, r)
    for it in 1:maxit
        J = zeros(length(r), length(θ))
        for j in eachindex(θ)
            h = 1e-6 * max(abs(θ[j]), 1e-3); θp = copy(θ); θp[j] += h
            J[:, j] = (f(θp) .- r) ./ h
        end
        JtJ = J' * J; g = J' * r
        improved = false
        for _ in 1:20
            δ = -((JtJ + λ * Diagonal(max.(diag(JtJ), 1e-12))) \ g)
            θn = θ .+ δ; rn = f(θn); ssen = sum(abs2, rn)
            if isfinite(ssen) && ssen < sse
                rel = (sse - ssen) / sse
                θ, r, sse = θn, rn, ssen; λ = max(λ / 3, 1e-12); improved = true
                rel < tol && return θ, sse
                break
            else
                λ *= 4
            end
        end
        improved || break
    end
    return θ, sse
end
mdape(y, yh) = median(abs.(yh .- y) ./ abs.(y)) * 100   # median absolute % error
