"""Density-sweep figure.  Usage: python make_density_figure.py [results_density_dir]"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import make_figures as mf

d = sys.argv[1] if len(sys.argv) > 1 else "../results_density"
S = json.load(open(os.path.join(d, "density_results.json")))
keys = sorted(S, key=lambda k: int(k[1:]))
n = np.array([int(k[1:]) for k in keys])
M = [("ens_init", "Random-init ensemble", "#2a78d6", "-"),
     ("ens_init_rc", "Random-init + label recalibration", "#1baf7a", "-"),
     ("ens_boot", "Bootstrap ensemble", "#6250d6", "-"),
     ("gp", "Gradient GP", "#eda100", "-")]

fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
for m, lab, c, ls in M:
    axs[0].plot(n, [S[k][m]["F_z_rms"] for k in keys], ls, marker="o", color=c, lw=2, ms=7, label=lab)
    if not m.endswith("_rc"):
        axs[1].plot(n, [S[k][m]["F_rmse"] for k in keys], ls, marker="o", color=c, lw=2, ms=7, label=lab)
for m, lab, c, _ in M:
    if m.endswith("_rc"):
        continue
    axs[2].plot(n, [S[k]["timing"][m] for k in keys], marker="o", color=c, lw=2, ms=7, label=lab)
axs[0].axhline(1, color=mf.INK2, ls="--", lw=1)
axs[0].set_yscale("log"); axs[0].set_yticks([0.5, 1, 2, 4, 8, 16])
axs[0].set_yticklabels(["0.5", "1", "2", "4", "8", "16"]); axs[0].minorticks_off()
axs[0].set_ylabel("RMS z-score (1 = honest error bars)")
axs[0].set_title("Honesty of free-energy error bars", color=mf.INK, fontsize=12)
axs[1].set_ylabel("Free-energy RMSE [kT]"); axs[1].set_ylim(0, None)
axs[1].set_title("Accuracy", color=mf.INK, fontsize=12)
axs[2].set_ylabel("Training / fitting time per replica [s]"); axs[2].set_yscale("log")
axs[2].set_title("Cost (one CPU)", color=mf.INK, fontsize=12)
for ax in axs:
    ax.set_xscale("log"); ax.set_xticks(n); ax.set_xticklabels([str(x) for x in n]); ax.minorticks_off()
    ax.set_xlabel("Restraint windows (4D)"); ax.grid(color=mf.GRID, lw=0.6)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=10.5)
fig.tight_layout(rect=(0, 0.08, 1, 1))
p = os.path.join(d, "fig9_density_sweep.png")
fig.savefig(p); print("written", p)
