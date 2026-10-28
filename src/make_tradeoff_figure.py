"""Accuracy vs honesty of error bars, 4D, untuned vs tuned networks.

Usage: python make_tradeoff_figure.py <untuned_dir> <tuned_dir> [out_dir]
x = free-energy RMSE (kT, lower = more accurate)
y = RMS z-score (1 = honest error bars, higher = overconfident)
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import make_figures as mf

du, dt = sys.argv[1], sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else dt
U = json.load(open(os.path.join(du, "pilot_results.json")))
T = json.load(open(os.path.join(dt, "pilot_results.json")))
M = [("ens_init", "Random-init ensemble", "#2a78d6"), ("ens_boot", "Bootstrap ensemble", "#6250d6"),
     ("ens_noise", "Noise-injected ensemble", "#eb6834"), ("gp", "Gradient GP", "#eda100")]

fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
for ax, regime in zip(axs, ("short", "long")):
    for m, lab, c in M:
        xu, yu = U[regime][m]["F_rmse"], U[regime][m]["F_z_rms"]
        xt, yt = T[regime][m]["F_rmse"], T[regime][m]["F_z_rms"]
        if m == "gp":
            ax.plot(xu, yu, "D", color=c, ms=10, label=lab)
        else:
            ax.annotate("", xy=(xt, yt), xytext=(xu, yu),
                        arrowprops=dict(arrowstyle="->", color=c, lw=1.6))
            ax.plot(xu, yu, "o", mfc="white", mec=c, mew=2, ms=9)
            ax.plot(xt, yt, "o", color=c, ms=9, label=lab)
    ax.axhline(1, color=mf.INK2, ls="--", lw=1)
    ax.set_yscale("log"); ax.set_yticks([0.5, 1, 2, 4, 8, 16])
    ax.set_yticklabels(["0.5", "1", "2", "4", "8", "16"]); ax.minorticks_off()
    ax.set_xlim(0, None); ax.grid(color=mf.GRID, lw=0.6)
    ax.set_xlabel("Free-energy error, RMSE [kT]  (left = more accurate)")
    ax.set_title(f"4D, {regime} trajectories", color=mf.INK, fontsize=12)
axs[0].set_ylabel("RMS z-score  (1 = honest error bars)")
axs[0].text(0.02, 0.03, "open circle = default network   filled = tuned on held-out labels",
            transform=axs[0].transAxes, fontsize=9.5, color=mf.INK2)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=10.5)
fig.tight_layout(rect=(0, 0.07, 1, 1))
p = os.path.join(out, "fig8_accuracy_vs_honesty_4d.png")
fig.savefig(p); print("written", p)
for regime in ("short", "long"):
    print(regime)
    for m, lab, _ in M:
        print(f"  {lab:26s} RMSE {U[regime][m]['F_rmse']:.3f} -> {T[regime][m]['F_rmse']:.3f}   "
              f"z {U[regime][m]['F_z_rms']:.2f} -> {T[regime][m]['F_z_rms']:.2f}   "
              f"targets cov95 {U[regime][m]['targets_cov95']:.2f} -> {T[regime][m]['targets_cov95']:.2f}")
