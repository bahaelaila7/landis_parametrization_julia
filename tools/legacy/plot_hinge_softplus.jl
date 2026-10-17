# Illustration only (not tied to any run): the AGB penalty's SOFTPLUS smoothing of the hard hinge, with a
# SLACK (threshold) of 1 — penalty stays ~0 while the deviation |sim−obs| is within the slack, then rises.
# softplus_β(x−slack) = log(1+e^{β(x−slack)})/β (= _hinge_relu in utils.jl); β→∞ recovers the hard hinge.
#   Run: ./julia_gdal.sh --project=. tools/plot_hinge_softplus.jl
import CairoMakie
const MK = CairoMakie

const SLACK = 1.0
hinge(x) = max(0.0, x - SLACK)
softplus(x, β) = (z = x - SLACK; max(0.0, z) + log1p(exp(-β * abs(z))) / β)   # stable log(1+e^{βz})/β

xs = collect(range(0.0, 4.0; length=601))
betas = [0.5, 1.0, 2.0, 5.0]
cols = [MK.RGBf(0.20, 0.55, 0.85), MK.RGBf(0.15, 0.60, 0.35), MK.RGBf(0.90, 0.55, 0.10), MK.RGBf(0.75, 0.20, 0.25)]

fig = MK.Figure(size=(640, 470))
ax = MK.Axis(fig[1, 1]; title="AGB penalty: softplus smoothing of the hinge  (slack = $(SLACK))",
  xlabel="x = |sim − obs|", ylabel="penalty")
MK.vspan!(ax, 0, SLACK; color=(:gray, 0.13))
MK.vlines!(ax, [SLACK]; color=(:gray, 0.6), linestyle=:dash, linewidth=0.9)
MK.lines!(ax, xs, hinge.(xs); color=:black, linestyle=:dash, linewidth=1.7, label="hinge (β→∞)")
for (β, c) in zip(betas, cols)
  MK.lines!(ax, xs, softplus.(xs, β); color=c, linewidth=2.4, label="β = $β")
end
MK.text!(ax, SLACK / 2, 2.7; text="slack\n(≈0 penalty)", color=:gray25, fontsize=11, align=(:center, :top))
MK.axislegend(ax; position=:lt, framevisible=true, labelsize=11)
out = "tools/hinge_softplus_illustration.png"
MK.save(out, fig); println("wrote $out")
