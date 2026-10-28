"""Figures for the pilot. Reads ../results/pilot_results.json."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from system import free_energy, MINIMA, SADDLES, REF_POINT, DOMAIN
from sampling import window_centres

RES = sys.argv[1] if len(sys.argv) > 1 else "../results"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SER = {"ens_init": "#2a78d6", "ens_boot": "#6250d6", "ens_noise": "#eb6834", "ens_noise_rc": "#1baf7a", "gp": "#eda100"}
LABEL = {"ens_init": "Random-init ensemble (RiD-style)", "ens_boot": "Bootstrap ensemble", "ens_noise": "Noise-injected ensemble",
         "ens_noise_rc": "Noise-injected + label recalibration", "gp": "Gradient GP"}
plt.rcParams.update({"font.size": 12, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200})


def fig_system():
    xs = np.linspace(DOMAIN["xmin"], DOMAIN["xmax"], 300)
    ys = np.linspace(DOMAIN["ymin"], DOMAIN["ymax"], 300)
    X, Y = np.meshgrid(xs, ys)
    F = free_energy(X, Y)[0] - free_energy(*REF_POINT)[0]
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    cs = ax.contourf(X, Y, np.minimum(F, 14), levels=np.arange(0, 14.5, 1), cmap="Blues_r")
    ax.contour(X, Y, F, levels=np.arange(0, 14, 2), colors="white", linewidths=0.6)
    C = window_centres()
    ax.plot(C[:, 0], C[:, 1], ".", color=INK, ms=3, label="Restraint centres (245)")
    for k, p in MINIMA.items():
        ax.annotate(k, p, color=INK, fontsize=13, fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec=INK2, lw=0.8))
    for k, p in SADDLES.items():
        ax.plot(*p, marker="X", color="#e34948", ms=10, mec="white", mew=1)
        ax.annotate(k, p, xytext=(6, 6), textcoords="offset points", color=INK, fontsize=11)
    cb = fig.colorbar(cs, ax=ax); cb.set_label("Exact F(x, y) − F(A)  [kT]")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(loc="lower left", fontsize=10, frameon=True)
    ax.set_title("Test system: exact free-energy surface", color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig1_system.png")); plt.close(fig)


def fig_calibration(summ):
    lv = np.linspace(0.05, 0.95, 19)
    fig, axs = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    for i, regime in enumerate(("short", "long")):
        for j, (lvl, ttl) in enumerate((("force", "Mean force ∇F"), ("F", "Free energy F − F(A)"))):
            ax = axs[i, j]
            ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
            ax.text(0.97, 0.90, "ideal", color=INK2, fontsize=10, ha="right")
            for m, c in SER.items():
                ax.plot(lv, summ[regime][m][f"{lvl}_curve"], color=c, lw=2, marker="o", ms=4,
                        label=LABEL[m])
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(color=GRID, lw=0.6)
            ax.set_title(f"{ttl} — {regime} trajectories", color=INK, fontsize=12)
            if i == 1: ax.set_xlabel("Nominal interval coverage")
            if j == 0: ax.set_ylabel("Observed coverage")
    axs[0, 0].text(0.03, 0.97, "Below the diagonal = overconfident", transform=axs[0, 0].transAxes,
                   va="top", fontsize=10, color=INK2)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(os.path.join(RES, "fig2_calibration.png"))
    plt.close(fig)


def fig_zrms(summ):
    methods = ["ens_init", "ens_boot", "ens_noise", "ens_noise_rc", "gp"]
    names = ["Random-init\nensemble", "Bootstrap\nensemble", "Noise-injected\nensemble",
             "Noise-inj. +\nrecalibration", "Gradient\nGP"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    for ax, (key, ttl) in zip(axs, (("force_z_rms", "Mean force (grid)"),
                                    ("F_z_rms", "Free energy (grid)"),
                                    ("targets_z_rms", "ΔF and barrier (across replicas)"))):
        for k, (regime, c, off) in enumerate((("short", "#2a78d6", -0.12), ("long", "#eb6834", 0.12))):
            v = [summ[regime][m][key] for m in methods]
            ax.plot(np.arange(5) + off, v, "o", color=c, ms=9, label=f"{regime} trajectories")
            for x, y in zip(np.arange(5) + off, v):
                ax.text(x, y * 1.06, f"{y:.1f}", ha="center", fontsize=9, color=INK)
        ax.axhline(1, color=INK2, ls="--", lw=1)
        ax.set_yscale("log"); ax.set_yticks([0.5, 1, 2, 4, 8]); ax.set_yticklabels(["0.5", "1", "2", "4", "8"]); ax.minorticks_off(); ax.set_xticks(range(5)); ax.set_xticklabels(names, fontsize=9)
        ax.set_title(ttl, color=INK, fontsize=12); ax.grid(axis="y", color=GRID, lw=0.6)
    axs[0].set_ylabel("RMS z-score  (1 = calibrated, >1 = overconfident)")
    axs[0].legend(frameon=False, fontsize=10, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig3_zscore.png")); plt.close(fig)


if __name__ == "__main__":
    summ = json.load(open(os.path.join(RES, "pilot_results.json")))
    fig_system(); fig_calibration(summ); fig_zrms(summ)
    print("figures written to", RES)
