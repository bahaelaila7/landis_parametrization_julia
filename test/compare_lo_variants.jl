# Overlay the 4 sobol×mo CMA-MAE variants' best train & validation loss trajectories.
import CairoMakie
const MK = CairoMakie
variants = [("base", :off, :off), ("sobol", :on, :off), ("mo", :off, :on), ("sobolmo", :on, :on)]
cols = Dict("base"=>:gray40, "sobol"=>:steelblue, "mo"=>:firebrick, "sobolmo"=>:seagreen)
function load(v)
  f = "runs/cmame_lo_$(v)_outputs/metrics.csv"
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
  lab = "$(v) (sobol=$(so), mo=$(mo))"
  MK.lines!(ax1, it, tr; color=cols[v], linewidth=2, label=lab)
  MK.lines!(ax2, it, vl; color=cols[v], linewidth=2, label=lab)
end
MK.axislegend(ax1; position=:rt, framevisible=true, labelsize=9)
MK.Label(fig[0,1:2], "CMA-MAE on 8.3.5.65o (longevity pinned) — Sobol × MO-rank"; fontsize=14, font=:bold)
MK.ylims!(ax1, 100, 260); MK.ylims!(ax2, 100, 280)
MK.save("runs/cmame_lo_comparison.png", fig)
println("wrote runs/cmame_lo_comparison.png")
