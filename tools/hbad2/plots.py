# HBAD v2 figures from the CSV outputs of timing_curves.jl / simB.jl / longrun.jl.
#   /workspace/PERSEUS_PRESENTATION/.venv/bin/python -I tools/hbad2/plots.py <tag>
import sys, os, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "hbad2")
tag = sys.argv[1]; D = os.path.join(root, "fit_" + tag); F = os.path.join(root, "figures"); os.makedirs(F, exist_ok=True)
cur = pd.read_csv(os.path.join(D, "timing_curves.csv")); vs = pd.read_csv(os.path.join(D, "timing_vs_obs.csv"))
keys = sorted(cur.key.unique()); cm = plt.get_cmap("tab20")
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for i, k in enumerate(keys):
    c = cur[cur.key == k]; ls = "-" if k.startswith("P") or k == "_S" or k == "TAAS" else "--"
    ax[0].plot(c.age, c.H, ls, color=cm(i), label=k); ax[1].plot(c.age, c.D, ls, color=cm(i))
    ax[2].plot(c.age, c.dH / c.dD.clip(lower=1e-6), ls, color=cm(i))
ax[0].set(xlabel="cohort age (yr)", ylabel="mean height (m)", title="free-growth height"); ax[1].set(xlabel="cohort age (yr)", ylabel="QMD (cm)", title="free-growth diameter")
ax[2].set(xlabel="cohort age (yr)", ylabel="dH (m) : dD (cm)", title="allocation height vs diameter", ylim=(0, 2.5))
ax[0].legend(ncol=2, fontsize=7); fig.suptitle(f"HBAD v2 ({tag}): species free-growth curves (open-grown cohort; dashed = hardwoods)"); fig.tight_layout()
fig.savefig(os.path.join(F, f"{tag}_species_curves.png"), dpi=130); plt.close(fig)
fig, axs = plt.subplots(3, 4, figsize=(14, 9), sharex=True, sharey=True)
for a, k in zip(axs.flat, keys):
    v = vs[vs.key == k]
    a.plot(v.acl, v.dH_obs / v.dD_obs, "ko-", label="obs (CCLCD 1-3)"); a.plot(v.acl, v.dH_mod / v.dD_mod, "s--", color="tab:green", label="model")
    a.set_title(k); a.set_ylim(0, 1.8)
axs[0, 0].legend(fontsize=8); [a.set_xlabel("age class (yr)") for a in axs[-1]]; [a.set_ylabel("dH:dD") for a in axs[:, 0]]
fig.suptitle(f"Height vs diameter growth allocation over age, free-growing trees: data vs HBAD v2 ({tag})"); fig.tight_layout()
fig.savefig(os.path.join(F, f"{tag}_allocation_vs_obs.png"), dpi=130); plt.close(fig)
sb = os.path.join(D, "simB_val.csv")
if os.path.exists(sb):
    s = pd.read_csv(sb)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    for a, (x, nm) in zip(ax, (("BA", "BA (m²/ha)"), ("N5", "trees ≥ 5 in /ha"), ("HL", "Lorey height (m)"))):
        o, m, p = s[x + "_o"], s[x + "_m"], s[x + "_0"]
        a.scatter(o, p, s=5, alpha=.3, color="grey", label="persistence"); a.scatter(o, m, s=5, alpha=.4, color="tab:green", label="HBAD v2 free run")
        lim = np.nanpercentile(np.r_[o, m], 99.5); a.plot([0, lim], [0, lim], "k-", lw=.8); a.set(xlim=(0, lim), ylim=(0, lim), xlabel="observed " + nm, ylabel="simulated")
    ax[0].legend(); fig.suptitle(f"Sim B: free stand runs from the first visit (VAL, natural, undisturbed plots), {tag}"); fig.tight_layout()
    fig.savefig(os.path.join(F, f"{tag}_simB_scatter.png"), dpi=130); plt.close(fig)
print("figures in", F)
