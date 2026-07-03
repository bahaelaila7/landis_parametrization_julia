# Standalone replica of Pan's tier-3 + AGB-hinge + cell-normalized loss, to score FVS cohorts against
# the curated ground truth on the IDENTICAL objective Pan optimizes. Per (eco×lu×species) cell:
#   W   = Σ bw·|cdf_sim − cdf_ref|  (AGB-weighted, binned, cumsum-normalized age-CDF; L1)  / W_SCALE  × RANKW
#   AGB = lambda·softplus_relu(|agb_sim − agb_ref| − thresh)^p                              / AGB_SCALE × RANKW
#   scalar = (alpha·ΣW + ΣAGB) / num_obs        (the 2 Sim-A objectives summed)
# W_SCALE = Σ bw·max(cdf_ref,1−cdf_ref) (per-ref max-W1); AGB_SCALE = Σ obs AGB; RANKW = (1/log(rank+1))/Σ by AGB.
# Cohorts come in as (plot, year, eff, agebin → agb). Validation: feed obs as sim ⇒ W=0, AGB inside deadband ⇒ ~0.
module FVSLoss
using DataFrames, Statistics

const BINS = [10, 20, 30, 40, 50, 60, 80, 100, 120, 150]
const NB = length(BINS) + 1                                   # 11 bins, last open
bin_widths() = Float64[BINS[1]; diff(BINS)]                   # length 10; W1 sums these (last open bin contributes 0)
binidx(a) = (for (i, b) in enumerate(BINS); a < b && return i; end; NB)   # age → bin 1..NB
# hinge params (match the run config: L1, 5% deadband clamped [2,50], softplus β=0.1, λ=1, α=1)
const HINGE_PCT=0.05; const HINGE_MIN=2.0; const HINGE_MAX=50.0; const HINGE_BETA=0.1; const HINGE_P=1.0
const LAMBDA=1.0; const ALPHA=1.0
hinge_thresh(obs) = clamp(obs*HINGE_PCT, HINGE_MIN, HINGE_MAX)
softplus_relu(x) = HINGE_BETA<=0 ? max(0.0,x) : max(0.0,x) + log1p(exp(-abs(x)*HINGE_BETA))/HINGE_BETA

# cdf over the NB bins from (agebin → agb): cumsum(agb per bin)/total, length NB (last entry 1).
function cell_cdf(agebins::Vector{Int}, agbs::Vector{Float64})
  h = zeros(Float64, NB); for (b,a) in zip(agebins,agbs); h[clamp(b,1,NB)] += a; end
  s = sum(h); s<=0 && return nothing
  cumsum(h) ./ s
end

# Build per-(plot,year,eff) → (agb_sum, cdf, agebins, agbs) from a cohort frame with cols plot,year,eff,agebin,agb.
function index_cohorts(df::DataFrame)
  d = Dict{Tuple{Any,Int,String}, NamedTuple}()
  for g in groupby(df, [:plotkey, :year, :eff])
    abv = Int.(g.agebin); av = Float64.(g.agb); cdf = cell_cdf(abv, av)
    cdf === nothing && continue
    d[(first(g.plotkey), Int(first(g.year)), String(first(g.eff)))] = (agb=sum(av), cdf=cdf, abv=abv, av=av)
  end
  d
end

# Cell scales from the observed cells: per (eco, eff) — W_SCALE, AGB_SCALE, RANKW (rank by total obs AGB).
function cell_scales(obs::Dict, cell_of::Dict)   # cell_of: (plot,year,eff)->(eco). returns Dict (eco,eff)->(ws,as,agbsum)
  bw = bin_widths()
  agg = Dict{Tuple{String,String}, NamedTuple{(:ws,:as),Tuple{Float64,Float64}}}()
  acc = Dict{Tuple{String,String}, Vector{Float64}}()   # (eco,eff) -> [Σmax-W1, ΣAGB]
  for (k, v) in obs
    eco = get(cell_of, k, nothing); eco === nothing && continue
    eff = k[3]; maxw = sum(bw .* max.(v.cdf[1:length(bw)], 1 .- v.cdf[1:length(bw)]))
    a = get!(acc, (eco, eff), [0.0, 0.0]); a[1] += maxw; a[2] += v.agb
  end
  for (ce, a) in acc; agg[ce] = (ws=a[1], as=a[2]); end
  # RANKW: rank cells by ΣAGB desc within all, 1/log(rank+1) normalized to Σ=1
  cells = sort(collect(keys(agg)); by=ce -> -agg[ce].as)
  rw = Dict{Tuple{String,String},Float64}(); tot = 0.0
  for (i, ce) in enumerate(cells); rw[ce] = 1/log(i+1); tot += rw[ce]; end
  for ce in cells; rw[ce] /= tot; end
  agg, rw
end

# Tier-3 + hinge + cell-norm scalar over the (plot,year) pairs present in both obs and sim.
function loss(obs::Dict, sim::Dict, cell_of::Dict)
  scales, rankw = cell_scales(obs, cell_of)
  bw = bin_widths()
  W = 0.0; A = 0.0; nobs = 0
  pys = unique([(k[1], k[2]) for k in keys(obs)])
  for (plot, yr) in pys
    effs = Set{String}()
    for k in keys(obs); k[1]==plot && k[2]==yr && push!(effs, k[3]); end
    for k in keys(sim); k[1]==plot && k[2]==yr && push!(effs, k[3]); end
    isempty(effs) && continue
    eco = get(cell_of, (plot, yr, first(effs)), nothing); eco === nothing && continue
    for eff in effs
      o = get(obs, (plot,yr,eff), nothing); s = get(sim, (plot,yr,eff), nothing)
      sc = get(scales, (eco,eff), (ws=1.0,as=1.0)); rw = get(rankw, (eco,eff), 0.0)
      ocdf = o===nothing ? zeros(NB) : o.cdf; scdf = s===nothing ? zeros(NB) : s.cdf
      oagb = o===nothing ? 0.0 : o.agb; sagb = s===nothing ? 0.0 : s.agb
      w = sum(bw .* abs.(scdf[1:length(bw)] .- ocdf[1:length(bw)]))
      ws = sc.ws>0 ? sc.ws : 1.0; as = sc.as>0 ? sc.as : 1.0
      W += rw * (w/ws)
      A += rw * (LAMBDA * softplus_relu(abs(sagb-oagb) - hinge_thresh(oagb))^HINGE_P / as)
    end
    nobs += 1
  end
  (total=(ALPHA*W + A)/max(nobs,1), W=ALPHA*W/max(nobs,1), AGB=A/max(nobs,1), nobs=nobs)
end
end # module
