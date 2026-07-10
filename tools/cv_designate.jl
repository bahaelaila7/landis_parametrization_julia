# Reconstruct one fold's cv_reselect_metrics.csv from cv_val_fronts.csv — NO re-simulation. Ranks every
# checkpoint's archive by its VAL front-area (shared-rectangle rule: min area, tie max knee, tie max count) to
# pick p100 (best on val); last = final checkpoint. On each archive's val-NON-DOMINATED front, designates 5
# positions: extreme_w, extreme_agb, knee, "median", best_aggregate (min A_W+A_AGB). Writes 10 rows (val+train).
#   ./julia_gdal.sh --project=. tools/cv_designate.jl <fold_dir>
using CSV, DataFrames
FOLD = ARGS[1]
vf = CSV.read(joinpath(FOLD, "cv_val_fronts.csv"), DataFrame)
nval = length(unique(CSV.read(joinpath(FOLD, "cv_front_plots.csv"), DataFrame).plot_id))
Wm = minimum(vf.A_W); Am = minimum(vf.A_AGB)                       # full val rectangle floor (for area rule)

# rows of a gen's archive, reduced to its val-non-dominated front, sorted by val A_W asc
function vnd(sub)
    n = nrow(sub); keep = trues(n)
    for i in 1:n
        for j in 1:n
            (i == j) && continue
            if sub.A_W[j] <= sub.A_W[i] && sub.A_AGB[j] <= sub.A_AGB[i] && (sub.A_W[j], sub.A_AGB[j]) != (sub.A_W[i], sub.A_AGB[i])
                keep[i] = false; break     # inner break only — marks THIS point dominated, continues to next i
            end
        end
    end
    sort(sub[keep, :], [:A_W, order(:A_AGB, rev=true)])
end
function area_of(s)
    p1w, p1a = s.A_W[1], s.A_AGB[1]; a = (p1w - Wm) * (p1a - Am)
    for i in 1:nrow(s)-1; a += (s.A_W[i+1] - s.A_W[i]) * ((s.A_AGB[i] - Am) + (s.A_AGB[i+1] - Am)) / 2; end
    a
end
nrm(x, lo, hi) = hi > lo ? (x - lo) / (hi - lo) : 0.0
function knee_of(s)
    nrow(s) < 3 && return 0.0
    wl,wh = extrema(s.A_W); al,ah = extrema(s.A_AGB)
    p1 = (nrm(s.A_W[1],wl,wh), nrm(s.A_AGB[1],al,ah)); pk = (nrm(s.A_W[end],wl,wh), nrm(s.A_AGB[end],al,ah))
    d12 = hypot(pk[1]-p1[1], pk[2]-p1[2]); d12 <= 0 && return 0.0
    maximum(abs((pk[1]-p1[1])*(p1[2]-nrm(s.A_AGB[i],al,ah)) - (p1[1]-nrm(s.A_W[i],wl,wh))*(pk[2]-p1[2]))/d12 for i in 1:nrow(s))
end

fronts = Dict{Int,DataFrame}()
for sub in groupby(vf, :gen); fronts[sub.gen[1]] = vnd(DataFrame(sub)); end
gens = sort(collect(keys(fronts)))
scored = [(g=g, area=area_of(fronts[g]), knee=knee_of(fronts[g]), n=nrow(fronts[g])) for g in gens]
order = sort(scored, by = s -> (s.area, -s.knee, -s.n))
p100_gen = order[1].g
last_gen  = gens[end]

function positions(s)      # -> [(name, rowindex)] on the val-non-dominated front s
    iW = argmin(s.A_W); iA = argmin(s.A_AGB); iB = argmin(s.A_W .+ s.A_AGB)   # extremes + best aggregate
    nrow(s) == 1 && return [("extreme_w",1),("extreme_agb",1),("knee",1),("median",1),("best_aggregate",1)]  # all = the one candidate
    wl,wh = extrema(s.A_W); al,ah = extrema(s.A_AGB)
    p1 = (nrm(s.A_W[iW],wl,wh), nrm(s.A_AGB[iW],al,ah)); pk = (nrm(s.A_W[iA],wl,wh), nrm(s.A_AGB[iA],al,ah))
    M = ((p1[1]+pk[1])/2, (p1[2]+pk[2])/2)
    segd(i) = (qn=(nrm(s.A_W[i],wl,wh),nrm(s.A_AGB[i],al,ah)); ABx=-M[1];ABy=-M[2];dd=ABx^2+ABy^2; t=dd<=0 ? 0.0 : clamp(((qn[1]-M[1])*ABx+(qn[2]-M[2])*ABy)/dd,0,1); hypot(qn[1]-(M[1]+t*ABx), qn[2]-(M[2]+t*ABy)))
    im = argmin(segd(i) for i in 1:nrow(s))                                    # "median" = closest to midpoint→corner line
    if nrow(s) == 2                                                            # knee undefined with 2 pts → knee = median
        return [("extreme_w",iW),("extreme_agb",iA),("knee",im),("median",im),("best_aggregate",iB)]
    end
    d12 = hypot(pk[1]-p1[1], pk[2]-p1[2])
    perp(i) = d12<=0 ? 0.0 : abs((pk[1]-p1[1])*(p1[2]-nrm(s.A_AGB[i],al,ah)) - (p1[1]-nrm(s.A_W[i],wl,wh))*(pk[2]-p1[2]))/d12
    ik = argmax(perp(i) for i in 1:nrow(s))
    [("extreme_w",iW),("extreme_agb",iA),("knee",ik),("median",im),("best_aggregate",iB)]
end

rows = NamedTuple[]
for (atype, g) in (("p100", p100_gen), ("last", last_gen))
    s = fronts[g]
    for (nm, i) in positions(s)
        push!(rows, (archive=atype, position=nm, A_W=s.A_W[i], A_AGB=s.A_AGB[i],
                     A_W_train=s.A_W_train[i], A_AGB_train=s.A_AGB_train[i], n_val=nval, sel_gen=g, front_size=nrow(s)))
    end
end
CSV.write(joinpath(FOLD, "cv_reselect_metrics.csv"), DataFrame(rows))
println("$(basename(FOLD)): p100=@$p100_gen last=@$last_gen n_val=$nval  → 5 positions × 2 archives")
