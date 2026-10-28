"""Label-error ablation figure.

Usage: python make_ablation_figure.py <md_dir> <debiased_dir> <oracle_dir> [out_dir]
Each dir must contain pilot_results.json from run_pilot.py.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
MODES = [("md", "Raw MD labels", "#2a78d6"),
         ("debiased", "Bias removed", "#eb6834"),
         ("oracle", "Bias removed + correct label error bars", "#1baf7a")]
METHODS = ["ens_init", "ens_boot", "ens_noise", "ens_noise_rc", "gp"]
NAMES = ["Random-init\nensemble", "Bootstrap\nensemble", "Noise-injected\nensemble",
         "Noise-inj. +\nrecalibration", "Gradient\nGP"]
plt.rcParams.update({"font.size": 12, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200})


def main():
    dirs = sys.argv[1:4]
    out = sys.argv[4] if len(sys.argv) > 4 else dirs[0]
    S = {m: json.load(open(os.path.join(d, "pilot_results.json"))) for (m, _, _), d in zip(MODES, dirs)}
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for i, (key, ttl) in enumerate((("F_z_rms", "Free energy on grid"),
                                    ("targets_z_rms", "ΔF and barrier across replicas"))):
        for j, regime in enumerate(("short", "long")):
            ax = axs[i, j]
            for k, (m, lab, c) in enumerate(MODES):
                v = [S[m][regime][x][key] for x in METHODS]
                xs = np.arange(len(METHODS)) + (k - 1) * 0.22
                ax.plot(xs, v, "o", color=c, ms=9, label=lab)
                for x, y in zip(xs, v):
                    ax.text(x, y * 1.08, f"{y:.1f}", ha="center", fontsize=8.5, color=INK)
            ax.axhline(1, color=INK2, ls="--", lw=1)
            ax.set_yscale("log"); ax.set_yticks([0.5, 1, 2, 4, 8, 16])
            ax.set_yticklabels(["0.5", "1", "2", "4", "8", "16"]); ax.minorticks_off()
            ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(NAMES, fontsize=9)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_title(f"{ttl} — {regime} trajectories", color=INK, fontsize=12)
            if j == 0:
                ax.set_ylabel("RMS z-score (1 = calibrated)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    p = os.path.join(out, "fig4_label_ablation.png")
    fig.savefig(p); print("written", p)

    rows = []
    for regime in ("short", "long"):
        for x in METHODS:
            rows.append([regime, x] + [f"{S[m][regime][x]['F_z_rms']:.2f}/{S[m][regime][x]['targets_cov95']:.2f}"
                                       for m, _, _ in MODES])
    print("regime  method  | F z_rms / targets cov95 for: md  debiased  oracle")
    for r in rows:
        print("  ".join(r))
    for m, _, _ in MODES:
        print(m, "label cov95:", {r: round(S[m][r]["labels_cov95"]["labels_block"], 2) for r in ("short", "long")})


if __name__ == "__main__":
    main()
