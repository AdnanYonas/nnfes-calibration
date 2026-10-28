"""Alanine dipeptide figures.  Usage: python make_ala2_figures.py [results_dir]"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import make_figures as mf
from ala2_reference import Reference, GRID
from run_ala2 import setup

d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results_ala2")
S = json.load(open(os.path.join(d, "ala2_results.json")))
R = Reference(); st = setup(R)

# Fig A: reference FES with windows and targets
fig, ax = plt.subplots(figsize=(6.6, 5.6))
deg = np.degrees(GRID)
cs = ax.contourf(deg, deg, np.minimum(R.F.T, 16), levels=np.arange(0, 16.5, 1), cmap="Blues_r")
ax.contour(deg, deg, R.F.T, levels=np.arange(0, 16, 2), colors="white", linewidths=0.5)
C = np.degrees(st["centres"])
ax.plot(C[:, 0], C[:, 1], ".", color=mf.INK, ms=2.5, label=f"Restraint centres ({len(C)})")
for name, p in zip(("C7eq", "C7ax", "TS"), np.degrees(st["targets"])):
    ax.annotate(name, p, color=mf.INK, fontsize=11, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=mf.INK2, lw=0.8))
p = np.degrees(st["ref_pt"])
ax.annotate("C5", p, color=mf.INK, fontsize=11, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=mf.INK2, lw=0.8))
cb = fig.colorbar(cs, ax=ax); cb.set_label("Reference F(φ, ψ) [kT]")
ax.set_xlabel("φ [deg]"); ax.set_ylabel("ψ [deg]"); ax.set_aspect("equal")
ax.set_title("Alanine dipeptide (vacuum, 300 K): metadynamics reference", color=mf.INK, fontsize=11)
ax.legend(loc="lower left", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(d, "fig10_ala2_reference.png")); plt.close(fig)

# Fig B: z-scores (F and targets) and RMSE, both regimes
methods = ["ens_init", "ens_boot", "ens_noise", "ens_init_rc", "gp"]
names = ["Random-init\nensemble", "Bootstrap\nensemble", "Noise-injected\nensemble",
         "Random-init +\nrescaling", "Gradient\nGP"]
fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
for k, (regime, c, off) in enumerate((("short", "#2a78d6", -0.12), ("long", "#eb6834", 0.12))):
    if regime not in S:
        continue
    for ax, key in zip(axs, ("F_z_rms", "targets_z_rms", "F_rmse")):
        v = [S[regime][m][key] for m in methods]
        xs = np.arange(len(methods)) + off
        ax.plot(xs, v, "o", color=c, ms=9, label=f"{regime} ({'10' if regime == 'short' else '50'} ps/window)")
        for x, y in zip(xs, v):
            ax.text(x, y * (1.07 if key != "F_rmse" else 1.03), f"{y:.2f}" if key == "F_rmse" else f"{y:.1f}",
                    ha="center", fontsize=8.5, color=mf.INK)
for ax, ttl in zip(axs, ("Free energy (reference grid, F<8 kT)", "ΔF and barrier (across replicas)",
                         "Free-energy RMSE [kT]")):
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(names, fontsize=9)
    ax.set_title(ttl, color=mf.INK, fontsize=12); ax.grid(axis="y", color=mf.GRID, lw=0.6)
for ax in axs[:2]:
    ax.axhline(1, color=mf.INK2, ls="--", lw=1); ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 4, 8]); ax.set_yticklabels(["0.5", "1", "2", "4", "8"]); ax.minorticks_off()
axs[0].set_ylabel("RMS z-score (1 = honest error bars)")
axs[2].set_ylim(0, axs[2].get_ylim()[1] * 1.12)
axs[0].legend(frameon=False, fontsize=9.5, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(d, "fig11_ala2_calibration.png")); plt.close(fig)
print("figures written to", d)
