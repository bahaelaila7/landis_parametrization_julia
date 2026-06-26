# Overlay the 4 sobol×mo CMA-MAE variants' best train & validation loss trajectories,
# no age-binning / no smoothing (cmame_lonb_*). Set HINGE=true to plot the hinge set (cmame_lonbh_*).
import CairoMakie
const MK = CairoMakie
const HINGE = get(ENV, "HINGE", "0") == "1"
const PFX = HINGE ? "cmame_lonbh" : "cmame_lonb"
variants = [("base", :off, :off), ("sobol", :on, :off), ("mo", :off, :on), ("sobolmo", :on, :on)]
cols = Dict("base"=>:gray40, "sobol"=>:steelblue, "mo"=>:firebrick, "sobolmo"=>:seagreen)
function load(v)
  f = "runs/$(PFX)_$(v)_outputs/metrics.csv"
  it=Int[]; tr=Float64[]; vl=Float64[]
  for (li,ln) in enumerate(eachline(f)); li==1 && continue
    c=split(ln,","); push!(it,parse(Int,c[1])); push!(tr,parse(Float64,c[2]))
    push!(vl, isempty(c[3]) ? NaN : parse(Float64,c[3]))
  end
  it,tr,vl
end
fig = MK.Figure(size=(1100,460))
ax1 = MK.Axis(fig[1,1]; xlabel="generation", ylabel="best train loss", title="train")
ax2 = MK.Axis(fig[1,2]; xlabel="generation", ylabel="best val loss", title="validation")
for (v,so,mo) in variants
  it,tr,vl = load(v)
  MK.lines!(ax1, it, tr; color=cols[v], linewidth=2, label="$(v) (sobol=$(so), mo=$(mo))")
  MK.lines!(ax2, it, vl; color=cols[v], linewidth=2, label="$(v) (sobol=$(so), mo=$(mo))")
end
MK.axislegend(ax1; position=:rt, framevisible=true, labelsize=9)
ttl = HINGE ? "CMA-MAE 8.3.5.65o — no binning/smoothing + AGB hinge (thr=10) — Sobol × MO-rank" :
              "CMA-MAE 8.3.5.65o — no binning/smoothing — Sobol × MO-rank"
MK.Label(fig[0,1:2], ttl; fontsize=13, font=:bold)
out = HINGE ? "runs/cmame_lonbh_comparison.png" : "runs/cmame_lonb_comparison.png"
MK.save(out, fig)
println("wrote ", out)
