# Stocking chart (log trees/ha vs log QMD): observed FIA plot-visits (natural / planted) + HBAD v2 free-run trajectories from bare ground
# + Reineke lines (N = SDI (QMD/25.4)^-1.605).   python -I tools/hbad2/plot_stocking.py <out.png> <scale plot|subplot> <longrun.csv> [label] ...
import sys, os, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "hbad2")
out, scale = sys.argv[1], sys.argv[2]; runs = sys.argv[3:]
o = pd.read_csv(os.path.join(root, "obs_stocking.csv")); o = o[(o.scale == scale) & (o.N > 0) & (o.QMD > 0)]
fig, ax = plt.subplots(figsize=(7.5, 6))
for pl, col, lab in ((False, "grey", "FIA natural"), (True, "tab:orange", "FIA planted")):
    x = o[o.planted == pl]; ax.scatter(x.QMD, x.N, s=2, alpha=0.08, color=col, label=f"{lab} ({scale}-visits)", rasterized=True)
q = np.geomspace(2.5, 80, 50)
for sdi, ls in ((400, ":"), (700, "--"), (1000, "-")):
    ax.plot(q, sdi * (q / 25.4) ** -1.605, "k" + ls, lw=0.9, label=f"Reineke SDI {sdi}")
for i in range(0, len(runs), 2):
    r = pd.read_csv(runs[i]); lab = runs[i + 1] if i + 1 < len(runs) else os.path.basename(runs[i])
    ax.plot(r.QMD_all, r.N_all, "-o", ms=2.5, lw=1.6, label="HBAD v2 free run: " + lab)
    for _, rr in r[r.yr.isin([10, 30, 60, 100, 150])].iterrows(): ax.annotate(f"{int(rr.yr)}y", (rr.QMD_all, rr.N_all), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.set(xscale="log", yscale="log", xlabel="QMD of trees ≥ 2.54 cm (cm)", ylabel="trees ≥ 2.54 cm per ha", xlim=(2.5, 80), ylim=(20, 2e4),
       title="Stocking chart: FIA stands vs HBAD v2 trajectory from bare ground")
ax.legend(fontsize=7, loc="lower left", markerscale=4); fig.tight_layout(); fig.savefig(out, dpi=140); print("wrote", out)
