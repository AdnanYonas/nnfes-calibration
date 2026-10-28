"""4D figures: calibration curves, z-scores, and a 2D-vs-4D comparison.

Usage: python make_figures_4d.py <results_4d_dir> <results_2d_dir>
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import make_figures as mf

d4 = sys.argv[1] if len(sys.argv) > 1 else "../results_4d"
d2 = sys.argv[2] if len(sys.argv) > 2 else "../results"
S4 = json.load(open(os.path.join(d4, "pilot_results.json")))
S2 = json.load(open(os.path.join(d2, "pilot_results.json")))

mf.RES = d4
mf.fig_calibration(S4)
os.replace(os.path.join(d4, "fig2_calibration.png"), os.path.join(d4, "fig5_calibration_4d.png"))
mf.fig_zrms(S4)
os.replace(os.path.join(d4, "fig3_zscore.png"), os.path.join(d4, "fig6_zscore_4d.png"))

methods = ["ens_init", "ens_boot", "ens_noise", "ens_noise_rc", "gp"]
names = ["Random-init\nensemble", "Bootstrap\nensemble", "Noise-injected\nensemble",
         "Noise-inj. +\nrecalibration", "Gradient\nGP"]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, regime in zip(axs, ("short", "long")):
    for k, (S, lab, c) in enumerate(((S2, "2D", "#2a78d6"), (S4, "4D", "#eb6834"))):
        v = [S[regime][m]["F_z_rms"] for m in methods]
        xs = np.arange(5) + (k - 0.5) * 0.25
        ax.plot(xs, v, "o", color=c, ms=9, label=lab)
        for x, y in zip(xs, v):
            ax.text(x, y * 1.07, f"{y:.1f}", ha="center", fontsize=9, color=mf.INK)
    ax.axhline(1, color=mf.INK2, ls="--", lw=1)
    ax.set_yscale("log"); ax.set_yticks([0.5, 1, 2, 4, 8, 16])
    ax.set_yticklabels(["0.5", "1", "2", "4", "8", "16"]); ax.minorticks_off()
    ax.set_xticks(range(5)); ax.set_xticklabels(names, fontsize=9)
    ax.grid(axis="y", color=mf.GRID, lw=0.6)
    ax.set_title(f"Free energy z-score — {regime} trajectories", color=mf.INK, fontsize=12)
axs[0].set_ylabel("RMS z-score (1 = calibrated)")
axs[0].legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(d4, "fig7_2d_vs_4d.png")); plt.close(fig)

raw = json.load(open(os.path.join(d4, "pilot_raw.json")))
for r in raw:
    t = {k: np.mean([o["timing"][k] for o in raw[r]]) for k in raw[r][0]["timing"]}
    h = raw[r][0]["gp_hyper"]
    print(r, "mean seconds:", {k: round(v) for k, v in t.items()},
          "| GP hyperfit %.0fs, factor %.1fs, N_train=%d" % (
              np.mean([o["gp_hyper"]["t_fit"] for o in raw[r]]),
              np.mean([o["gp_hyper"]["t_factor"] for o in raw[r]]), raw[r][0]["n_train"]))
print("figures written to", d4)
