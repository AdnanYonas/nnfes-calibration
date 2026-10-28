"""All manuscript figures, regenerated from the saved result files.

Run from the paper/ folder:  python make_paper_figures.py
Writes figures/fig*.pdf and .png (vector PDF for the journal, PNG for preview).
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "ala2"))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#d9d8d4", "#ffffff"
C = dict(ens_init="#2a78d6", ens_boot="#eb6834", ens_noise="#1baf7a",
         ens_init_rc="#eda100", gp="#6250d6", ens_noise_rc="#008300", ens_boot_rc="#e87ba4")
MK = dict(ens_init="o", ens_boot="s", ens_noise="^", ens_init_rc="D", gp="P", ens_noise_rc="v", ens_boot_rc="X")
NAME = dict(ens_init="Random-init ensemble", ens_boot="Bootstrap ensemble",
            ens_noise="Noise-injected ensemble", ens_init_rc="Random-init + rescaling",
            ens_noise_rc="Noise-injected + rescaling", ens_boot_rc="Bootstrap + rescaling", gp="Gradient GP")
SHORT = dict(ens_init="Random\ninit", ens_boot="Boot-\nstrap", ens_noise="Noise-\ninjected",
             ens_init_rc="Random init\n+ rescaling", ens_noise_rc="Noise-inj.\n+ rescaling", gp="Gradient\nGP")
METHODS = ["ens_init", "ens_init_rc", "ens_boot", "ens_noise", "gp"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "savefig.dpi": 300, "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42})
W1, W2 = 3.37, 6.69          # JCP single / double column widths (inches)


def load(p):
    return json.load(open(os.path.join(ROOT, p)))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"))
    plt.close(fig)
    print("wrote", name)


def panel(ax, s, x=-0.14, y=1.04):
    ax.text(x, y, f"({s})", transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, va="bottom")


def zaxis(ax, lo=0.4, hi=20):
    ax.set_yscale("log"); ax.set_ylim(lo, hi)
    t = [v for v in (0.5, 1, 2, 4, 8, 16) if lo <= v <= hi]
    ax.set_yticks(t); ax.set_yticklabels([f"{v:g}" for v in t]); ax.minorticks_off()
    for v in t:
        if v != 1:
            ax.axhline(v, color=GRID, lw=0.5, zorder=0)
    ax.axhline(1, color=INK2, ls="--", lw=0.9, zorder=1)


# ------------------------------------------------------------------ Fig 1
def fig_workflow():
    fig, ax = plt.subplots(figsize=(W2, 3.35))
    ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec=INK2, lw=0.8))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=7.3, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 6.8, body, ha="center", va="top", fontsize=6.9, color=INK, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, lw=1.0, color=INK2))

    box(0.5, 26, 22, 24, "1  Restrained MD",
        "$W$ windows at $c_w$\n$U_w=U+\\frac{\\kappa}{2}|s-c_w|^2$\nshort or long\nrun per window", "#eef4fc")
    box(25.3, 26, 23.5, 24, "2  Mean-force labels",
        "$\\hat g_w=\\kappa(c_w-\\langle s\\rangle_w)$\nat $\\hat s_w=\\langle s\\rangle_w$\nblock standard\nerror $\\hat\\epsilon_w$", "#eef4fc")
    box(51.6, 26, 23.5, 24, "3  Model + spread",
        "$F_\\theta(s)$ with\n$\\nabla_sF_\\theta\\approx\\hat g$; ensembles\n(random init, bootstrap,\nnoise) or gradient GP", "#fdf1ea")
    box(77.9, 26, 21.4, 24, "4  Prediction",
        "$\\Delta F(s)=F(s)-F(s_0)$\n$\\mu(s)\\pm\\sigma(s)$\nalso $\\Delta F_{AB}$\nand barriers", "#fdf1ea")
    for x0, x1 in ((22.9, 24.9), (49.2, 51.2), (75.5, 77.5)):
        arrow(x0, 38, x1, 38)
    # optional rescaling loop
    ax.add_patch(FancyBboxPatch((51.85, 15.5), 23, 6.5, boxstyle="round,pad=0.3,rounding_size=1",
                                fc="#fff6dc", ec=INK2, lw=0.8, ls="--"))
    ax.text(63.35, 18.7, "optional: rescale $\\sigma$ by $\\alpha$\nfrom 15% held-out labels",
            ha="center", va="center", fontsize=6.8, color=INK)
    arrow(63.35, 22.3, 63.35, 25.6)
    # scoring row
    ax.add_patch(FancyBboxPatch((0.5, 1), 99, 10.5, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc="#f3f3f1", ec=INK2, lw=0.8))
    ax.text(50, 10.6, "5  Calibration test against a known truth (exact FES or metadynamics reference)",
            ha="center", va="top", fontsize=7.8, fontweight="bold", color=INK)
    ax.text(50, 5.6, "$z=(\\mu-F_{\\rm true})/\\sigma$ at three levels: mean force $\\nabla F$;  free energy $\\Delta F(s)$;\n"
            "$\\Delta F$ between minima and barriers (across replicas).  RMS $z=1$: honest error bars",
            ha="center", va="top", fontsize=7.0, color=INK, linespacing=1.4)
    arrow(88.6, 25.6, 88.6, 12.2)
    save(fig, "fig1_workflow")


# ------------------------------------------------------------------ Fig 2
def fig_models():
    """Schematic of the gradient-matching network and where each ensemble's spread comes from."""
    fig = plt.figure(figsize=(W2, 2.55))
    ax = fig.add_axes([0, 0, 0.47, 1]); ax.axis("off"); ax.set_xlim(0, 50); ax.set_ylim(0, 30)
    layers = [(4, [11, 19]), (15, [6, 11, 16, 21, 26]), (26, [6, 11, 16, 21, 26]), (37, [16])]
    for (xa, ya), (xb, yb) in zip(layers[:-1], layers[1:]):
        for p in ya:
            for q in yb:
                ax.plot([xa, xb], [p, q], color=GRID, lw=0.5, zorder=1)
    for i, (x, ys) in enumerate(layers):
        for y in ys:
            ax.add_patch(plt.Circle((x, y), 1.5, fc=["#cde2fb", "#ffffff", "#ffffff", "#fbd9c9"][i], ec=INK2, lw=0.8, zorder=2))
    ax.text(4, 25, "input\n$s$", ha="center", fontsize=7.5, color=INK)
    ax.text(4, 3.0, "periodic CVs:\n$(\\cos s,\\sin s)$", ha="center", fontsize=6.8, color=INK2)
    ax.text(20.5, 29, "2 tanh layers × 64", ha="center", fontsize=7.5, color=INK)
    ax.text(37, 20, "$F_\\theta(s)$", ha="center", fontsize=8.5, color=INK)
    ax.annotate("", xy=(47, 16), xytext=(39.5, 16), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))
    ax.text(43.5, 17.3, "autograd", ha="center", fontsize=6.8, color=INK2)
    ax.text(43.5, 10.5, "$\\nabla_sF_\\theta(\\hat s_w)$\nvs. label $\\hat g_w$", ha="center", fontsize=7.5, color=INK)
    ax.text(25, 0.6, "Loss $\\mathcal{L}_m=\\sum_w\\omega_{mw}\\,|\\nabla_sF_{\\theta_m}(\\hat s_w)-\\tilde g_{mw}|^2+\\lambda\\|\\theta_m\\|^2$",
            ha="center", fontsize=7.5, color=INK)
    ax2 = fig.add_axes([0.5, 0.05, 0.5, 0.9]); ax2.axis("off"); ax2.set_xlim(0, 50); ax2.set_ylim(0, 30)
    rows = [("ens_init", "Random init", "same labels; members differ only\nin initial weights $\\theta_m^{(0)}$"),
            ("ens_boot", "Bootstrap", "$\\omega_{mw}$ = resampling counts of\nwindows (with replacement)"),
            ("ens_noise", "Noise-injected", "labels $\\hat g_w+\\hat\\epsilon_w\\xi_{mw}$, $\\xi\\sim\\mathcal{N}(0,1)$;\nweights $\\omega_{mw}\\propto\\hat\\epsilon_w^{-2}$"),
            ("gp", "Gradient GP", "exact posterior; label noise\n$\\hat\\epsilon_w^2$ on the diagonal")]
    for i, (m, t, d) in enumerate(rows):
        y = 26 - 7.2 * i
        ax2.plot(2, y - 1.2, MK[m], color=C[m], ms=8, mec="white", mew=0.6)
        ax2.text(5, y, t, fontsize=8, fontweight="bold", color=INK, va="top")
        ax2.text(21.5, y, d, fontsize=7.2, color=INK, va="top", linespacing=1.3)
    fig.text(0.01, 0.97, "(a)", fontsize=10, fontweight="bold"); fig.text(0.5, 0.97, "(b)", fontsize=10, fontweight="bold")
    save(fig, "fig2_models")


# ------------------------------------------------------------------ Fig 3
def fig_systems():
    import system as S2
    import sampling as SP
    import system4d as S4
    from ala2_reference import Reference, GRID as AGRID
    from run_ala2 import setup
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.35), gridspec_kw=dict(width_ratios=[1.12, 1.12, 1], wspace=0.42))
    # (a) 2D
    ax = axs[0]
    xs = np.linspace(-1.6, 1.1, 220); ys = np.linspace(-0.4, 2.0, 220)
    X, Y = np.meshgrid(xs, ys)
    F = S2.free_energy(X, Y)[0] - S2.free_energy(*S2.REF_POINT)[0]
    cs = ax.contourf(X, Y, np.minimum(F, 14), levels=np.arange(0, 14.5, 1), cmap="Blues_r")
    ax.contour(X, Y, F, levels=np.arange(0, 14, 2), colors="white", linewidths=0.4)
    Cw = SP.window_centres()
    ax.plot(Cw[:, 0], Cw[:, 1], ".", color=INK, ms=1.3)
    for k, p in S2.MINIMA.items():
        ax.text(*p, k, ha="center", va="center", fontsize=7.5, fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec=INK2, lw=0.6))
    ax.plot(*S2.SADDLES["S1"], "x", color="#e34948", ms=5, mew=1.5)
    ax.text(S2.SADDLES["S1"][0] + 0.06, S2.SADDLES["S1"][1] + 0.08, "S1", fontsize=7, color=INK)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_title("(a) 2D, exact $F$", loc="left", fontweight="bold")
    # (b) 4D projection: min over (u, v) on an (x, y) grid
    ax = axs[1]
    xs4 = np.linspace(-1.6, 1.1, 90); ys4 = np.linspace(-0.4, 2.0, 90)
    us = np.linspace(-1.9, 1.9, 41)
    X4, Y4 = np.meshgrid(xs4, ys4)
    best = np.full(X4.shape, np.inf)
    for u in us:
        v = 0.5 * np.sin(2 * X4) + 0.3 * u            # bottom of the curved v valley
        for dv in (-0.15, 0.0, 0.15):
            P = np.stack([X4.ravel(), Y4.ravel(), np.full(X4.size, u), (v + dv).ravel()], 1)
            best = np.minimum(best, S4.free_energy(P)[0].reshape(X4.shape))
    best -= S4.free_energy(S4.REF_POINT[None])[0][0]
    ax.contourf(X4, Y4, np.minimum(best, 14), levels=np.arange(0, 14.5, 1), cmap="Blues_r")
    ax.contour(X4, Y4, best, levels=np.arange(0, 14, 2), colors="white", linewidths=0.4)
    C4 = S4.window_centres(1000)
    ax.plot(C4[:, 0], C4[:, 1], ".", color=INK, ms=0.8, alpha=0.6)
    for k, p in S4.MINIMA.items():
        ax.text(p[0], p[1], k, ha="center", va="center", fontsize=7.5, fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec=INK2, lw=0.6))
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_title("(b) 4D, min over $(u,v)$", loc="left", fontweight="bold")
    # (c) alanine dipeptide
    ax = axs[2]
    R = Reference(); st = setup(R)
    deg = np.degrees(AGRID)
    cs = ax.contourf(deg, deg, np.minimum(R.F.T, 14), levels=np.arange(0, 14.5, 1), cmap="Blues_r")
    ax.contour(deg, deg, R.F.T, levels=np.arange(0, 14, 2), colors="white", linewidths=0.4)
    Ca = np.degrees(st["centres"])
    ax.plot(Ca[:, 0], Ca[:, 1], ".", color=INK, ms=1.3)
    for nm, p in zip(("C5", "C7eq", "C7ax", "TS"), np.degrees(np.vstack([st["ref_pt"], st["targets"]]))):
        ax.text(p[0], p[1], nm, ha="center", va="center", fontsize=6.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=INK2, lw=0.6))
    ax.set_xlabel("$\\phi$ (deg)"); ax.set_ylabel("$\\psi$ (deg)"); ax.set_aspect("equal")
    ax.set_xticks([-180, 0, 180]); ax.set_yticks([-180, 0, 180])
    ax.set_title("(c) Ala2", loc="left", fontweight="bold")
    cb = fig.colorbar(cs, ax=axs, fraction=0.025, pad=0.035); cb.set_label("$F-F_{\\rm ref}$ (kT)")
    save(fig, "fig3_systems")


# ------------------------------------------------------------------ helpers
def dotplot(ax, data, keys, methods, regimes=("short", "long"), lo=0.4, hi=16, label_vals=True):
    """data[regime][method][key] -> one marker per method; filled = short, open = long."""
    for j, regime in enumerate(regimes):
        for i, m in enumerate(methods):
            v = data[regime][m][keys]
            x = i + (-0.17 if j == 0 else 0.17)
            ax.plot(x, v, MK[m], ms=6.5, color=C[m], mfc=C[m] if j == 0 else "white", mew=1.4, zorder=3)
            if label_vals:
                ax.text(x, v * 1.13, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(range(len(methods))); ax.set_xticklabels([SHORT[m] for m in methods], fontsize=7)
    ax.set_xlim(-0.6, len(methods) - 0.4)
    zaxis(ax, lo, hi)


def method_legend(fig, ms, ncol=5, y=1.05):
    h = [Line2D([], [], marker=MK[m], ls="", color=C[m], ms=6, label=NAME[m]) for m in ms]
    fig.legend(handles=h, loc="upper center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, y),
               fontsize=6.8, handletextpad=0.2, columnspacing=0.9)


def regime_legend(ax, loc="upper right", labels=("short runs", "long runs")):
    h = [Line2D([], [], marker="o", ls="", color=INK2, mfc=INK2, ms=6, label=labels[0]),
         Line2D([], [], marker="o", ls="", color=INK2, mfc="white", mew=1.3, ms=6, label=labels[1])]
    ax.legend(handles=h, frameon=False, loc=loc, handletextpad=0.3)


# ------------------------------------------------------------------ Fig 4
def fig_2d_curves():
    S = load("results/pilot_results.json")
    lv = np.linspace(0.05, 0.95, 19)
    fig, axs = plt.subplots(1, 4, figsize=(W2, 2.0), sharey=True)
    ms = ["ens_init", "ens_init_rc", "ens_boot", "ens_noise", "gp"]
    for k, (regime, lvl) in enumerate([("short", "force"), ("short", "F"), ("long", "force"), ("long", "F")]):
        ax = axs[k]
        ax.plot([0, 1], [0, 1], color=INK2, ls="--", lw=0.9)
        for m in ms:
            ax.plot(lv, S[regime][m][f"{lvl}_curve"], color=C[m], lw=1.6, label=NAME[m])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel("nominal coverage")
        ax.set_title(f"({'abcd'[k]}) {regime}, {'force' if lvl == 'force' else 'free en.'}", loc="left", fontsize=7.8)
        ax.grid(color=GRID, lw=0.4)
    axs[0].set_ylabel("observed coverage")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.04), fontsize=7, handlelength=1.4, columnspacing=1.0)
    save(fig, "fig4_2d_calibration_curves")


# ------------------------------------------------------------------ Fig 5
def fig_2d_z():
    S = load("results/pilot_results.json")
    ms = ["ens_init", "ens_init_rc", "ens_boot", "ens_noise", "ens_noise_rc", "gp"]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.3), sharey=True)
    for ax, key, t, c in zip(axs, ("force_z_rms", "F_z_rms", "targets_z_rms"),
                             ("mean force", "free energy $\\Delta F(s)$", "$\\Delta F$ and barrier"), "abc"):
        dotplot(ax, S, key, ms, lo=0.4, hi=16, label_vals=False)
        ax.set_xticklabels([]); ax.tick_params(axis="x", length=0)
        ax.set_title(f"({c}) {t}", loc="left")
    axs[0].set_ylabel("RMS $z$ (1 = honest)")
    method_legend(fig, ms, ncol=3, y=1.13)
    regime_legend(axs[2])
    save(fig, "fig5_2d_zscores")


# ------------------------------------------------------------------ Fig 6
def fig_ablation():
    modes = [("results", "MD labels"), ("results_debiased", "bias removed"), ("results_oracle", "oracle labels")]
    D = {lab: load(f"{p}/pilot_results.json") for p, lab in modes}
    ms = ["ens_init", "ens_boot", "ens_noise", "gp"]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.3), sharey=True)
    for ax, regime, c in zip(axs, ("short", "long"), "ab"):
        for i, m in enumerate(ms):
            vals = [D[lab][regime][m]["F_z_rms"] for _, lab in modes]
            xs = i + np.array([-0.3, 0, 0.3])
            ax.plot(xs, vals, "-", color=C[m], lw=1.0, alpha=0.6, zorder=2)
            for x, v, fill in zip(xs, vals, ("full", "left", "none")):
                ax.plot(x, v, MK[m], ms=6.5, color=C[m], fillstyle=fill, mfc=C[m] if fill != "none" else "white",
                        mew=1.3, zorder=3)
                ax.text(x, v * 1.12, f"{v:.1f}", ha="center", fontsize=5.8, color=INK)
        ax.set_xticks(range(len(ms))); ax.set_xticklabels([SHORT[m] for m in ms], fontsize=7)
        zaxis(ax, 0.4, 16); ax.set_title(f"({c}) 2D, {regime} runs: free-energy RMS $z$", loc="left")
        lab = [D[l][regime]["labels_cov95"]["labels_block"] for _, l in modes]
        ax.text(0.02, 0.03, "label 95% coverage: " + " / ".join(f"{v:.0%}" for v in lab),
                transform=ax.transAxes, fontsize=6.8, color=INK2, va="bottom")
    axs[0].set_ylabel("RMS $z$ (1 = honest)")
    h = [Line2D([], [], marker="o", ls="", color=INK2, fillstyle=f, mfc=INK2 if f != "none" else "white", ms=6, label=l)
         for f, l in zip(("full", "left", "none"), ("MD labels", "estimator bias removed", "oracle labels"))]
    axs[1].legend(handles=h, frameon=False, loc="upper right", fontsize=7, bbox_to_anchor=(1.0, 0.9))
    save(fig, "fig6_label_ablation")


# ------------------------------------------------------------------ Fig 7
def fig_4d_tradeoff():
    A = load("results_4d/pilot_results.json"); B = load("results_4d_tuned/pilot_results.json")
    ms = ["ens_init", "ens_init_rc", "ens_boot", "ens_noise", "gp"]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.55), sharey=True)
    for ax, regime, c in zip(axs, ("short", "long"), "ab"):
        for m in ms:
            x0, y0 = A[regime][m]["F_rmse"], A[regime][m]["F_z_rms"]
            x1, y1 = B[regime][m]["F_rmse"], B[regime][m]["F_z_rms"]
            if m != "gp":
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="-|>", color=C[m], lw=1.0, alpha=0.8, shrinkA=4, shrinkB=4))
                ax.plot(x0, y0, MK[m], ms=6, color=C[m], mfc="white", mew=1.2)
            ax.plot(x1, y1, MK[m], ms=6.5, color=C[m], mew=0.5, mec="white", label=NAME[m])
        ax.set_xscale("log"); ax.set_xlim(0.05, 0.9)
        ax.set_xticks([0.05, 0.1, 0.2, 0.4, 0.8]); ax.set_xticklabels(["0.05", "0.1", "0.2", "0.4", "0.8"])
        ax.minorticks_off()
        zaxis(ax, 0.5, 12)
        ax.set_xlabel("free-energy RMSE (kT)")
        ax.set_title(f"({c}) 4D, {regime} runs", loc="left")
        ax.text(0.97, 0.04, "accurate and honest:\nbottom left", transform=ax.transAxes, ha="right", fontsize=6.5, color=INK2)
    axs[0].set_ylabel("free-energy RMS $z$")
    axs[1].legend(frameon=False, fontsize=6.8, loc="upper right", handletextpad=0.2)
    save(fig, "fig7_4d_tradeoff")


# ------------------------------------------------------------------ Fig 8
def fig_density():
    D = load("results_density_lbfgs/density_results.json")
    R = load("results_density_lbfgs/density_raw.json")
    ns = [500, 1000, 2000, 4000]
    ms = ["ens_init", "ens_init_rc", "ens_boot", "gp"]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2))
    for m in ms:
        axs[0].plot(ns, [D[f"n{n}"][m]["F_z_rms"] for n in ns], "-", marker=MK[m], color=C[m], ms=5, lw=1.4, label=NAME[m])
        axs[1].plot(ns, [D[f"n{n}"][m]["F_rmse"] for n in ns], "-", marker=MK[m], color=C[m], ms=5, lw=1.4)
    for m, key in (("ens_init", "ens_init"), ("ens_boot", "ens_boot"), ("gp", "gp")):
        t = [np.mean([r["timing"][key] for r in R[f"n{n}"]]) for n in ns]
        axs[2].plot(ns, t, "-", marker=MK[m], color=C[m], ms=5, lw=1.4)
    for ax in axs:
        ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns]); ax.minorticks_off()
        ax.set_xlabel("restraint windows $W$"); ax.grid(axis="x" if ax is axs[0] else "both", color=GRID, lw=0.4)
    zaxis(axs[0], 0.3, 25); axs[0].set_yticks([0.5, 1, 2, 4, 8, 16]); axs[0].set_yticklabels(["0.5", "1", "2", "4", "8", "16"])
    axs[0].set_ylabel("free-energy RMS $z$"); axs[0].set_title("honesty")
    axs[1].set_ylim(0, 0.2); axs[1].set_ylabel("free-energy RMSE (kT)"); axs[1].set_title("accuracy")
    axs[2].set_yscale("log"); axs[2].set_ylabel("wall time per fit (s)"); axs[2].set_title("cost (CPU)")
    axs[2].set_yticks([50, 100, 200, 500, 1000]); axs[2].set_yticklabels(["50", "100", "200", "500", "1000"]); axs[2].minorticks_off()
    for ax, c in zip(axs, "abc"):
        panel(ax, c, -0.3)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.1), fontsize=7)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig8_density")


# ------------------------------------------------------------------ Fig 9
def fig_ala2():
    S = load("ala2/ala2_pooled_results.json")
    a = load("ala2/results_ala2/ala2_raw.json"); b = load("ala2/results_ala2_user/ala2_raw.json")
    raw = {r: a[r] + b[r] for r in a}
    ms = ["ens_init", "ens_init_rc", "ens_boot", "ens_noise", "gp"]
    fig = plt.figure(figsize=(W2, 4.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.62, wspace=0.32)
    ax = fig.add_subplot(gs[0, 0:2])
    dotplot(ax, S, "F_z_rms", ms, lo=0.5, hi=6)
    ax.set_ylabel("free-energy RMS $z$"); ax.set_title("(a) free energy on the reference grid", loc="left")
    regime_legend(ax, labels=("short (10 ps/window)", "long (50 ps/window)"))
    ax = fig.add_subplot(gs[0, 2])
    lv = np.linspace(0.05, 0.95, 19)
    ax.plot([0, 1], [0, 1], color=INK2, ls="--", lw=0.9)
    for m in ms:
        ax.plot(lv, S["long"][m]["F_curve"], color=C[m], lw=1.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.grid(color=GRID, lw=0.4)
    ax.set_xlabel("nominal coverage"); ax.set_ylabel("observed coverage"); ax.set_title("(b) long runs, $\\Delta F(s)$", loc="left")
    # (c-e) target errors with 95% intervals per replica
    tnames = ["$\\Delta F$(C7eq − C5)", "$\\Delta F$(C7ax − C5)", "barrier C7eq→C7ax"]
    for k in range(3):
        ax = fig.add_subplot(gs[1, k])
        for i, m in enumerate(["ens_init", "ens_boot", "gp"]):
            for j, regime in enumerate(("short", "long")):
                e = np.array([o[m]["targets"]["est"][k] - o[m]["targets"]["true"][k] for o in raw[regime]])
                s = np.array([o[m]["targets"]["sig"][k] for o in raw[regime]])
                xs = i * 2.6 + j * 1.15 + np.linspace(-0.45, 0.45, len(e))
                miss = np.abs(e) > 1.96 * s
                for x, ee, ss, mm in zip(xs, e, s, miss):
                    ax.plot([x, x], [ee - 1.96 * ss, ee + 1.96 * ss], color=C[m], lw=1.0, alpha=0.9 if mm else 0.45)
                ax.plot(xs, e, MK[m], ms=3.2, color=C[m], mfc=C[m] if j == 0 else "white", mew=0.9, ls="")
                ax.plot(xs[miss], e[miss], "x", color="#e34948", ms=3.5, mew=1.0, ls="")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks([0.55, 3.15, 5.75]); ax.set_xticklabels(["Random\ninit", "Boot-\nstrap", "Gradient\nGP"], fontsize=7)
        ax.set_title(f"({'cde'[k]}) " + tnames[k], fontsize=8.5, loc="left"); ax.grid(axis="y", color=GRID, lw=0.4)
        if k == 0:
            ax.set_ylabel("estimate − reference (kT)")
    method_legend(fig, ms, ncol=5, y=0.995)
    fig.text(0.5, 0.005, "Each bar: one replica, 95% interval (left group short runs, right group long runs); "
             "red × = interval misses the reference.", ha="center", fontsize=7, color=INK2)
    save(fig, "fig9_ala2")


# ------------------------------------------------------------------ Fig 10
def fig_transfer():
    """H2: does force-level calibration transfer to F? force z vs F z, all systems/regimes."""
    sets = [("2D", load("results/pilot_results.json"), "o"),
            ("4D", load("results_4d_tuned/pilot_results.json"), "s"),
            ("Ala2", load("ala2/ala2_pooled_results.json"), "^")]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.7))
    ax = axs[0]
    for lab, S, mk in sets:
        for regime in ("short", "long"):
            for m in ["ens_init", "ens_init_rc", "ens_boot", "ens_noise", "ens_noise_rc", "gp"]:
                x, y = S[regime][m]["force_z_rms"], S[regime][m]["F_z_rms"]
                ax.plot(x, y, mk, ms=5.5, color=C[m], mfc=C[m] if regime == "short" else "white", mew=1.1)
    ax.plot([0.5, 12], [0.5, 12], color=INK2, ls="--", lw=0.9)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.5, 12); ax.set_ylim(0.4, 12)
    for a_ in (ax.xaxis, ax.yaxis):
        a_.set_major_locator(matplotlib.ticker.FixedLocator([0.5, 1, 2, 4, 8]))
        a_.set_major_formatter(matplotlib.ticker.FixedFormatter(["0.5", "1", "2", "4", "8"]))
    ax.minorticks_off(); ax.grid(color=GRID, lw=0.4)
    ax.set_xlabel("mean-force RMS $z$"); ax.set_ylabel("free-energy RMS $z$")
    ax.set_title("(a) force vs free-energy calibration", loc="left")
    h = [Line2D([], [], marker=mk, ls="", color=INK2, ms=5, label=lab) for lab, _, mk in sets]
    ax.legend(handles=h, frameon=False, loc="upper left", fontsize=7)
    # (b) rescaling effect: before vs after, F level, per system/regime
    ax = axs[1]
    rows = []
    for lab, S, mk in sets:
        for regime in ("short", "long"):
            rows.append((f"{lab} {regime}", S[regime]["ens_init"]["F_z_rms"], S[regime]["ens_init_rc"]["F_z_rms"],
                         S[regime]["ens_init"]["force_z_rms"], S[regime]["ens_init_rc"]["force_z_rms"]))
    y = np.arange(len(rows))[::-1]
    for yy, (lab, f0, f1, g0, g1) in zip(y, rows):
        ax.annotate("", xy=(f1, yy + 0.12), xytext=(f0, yy + 0.12), arrowprops=dict(arrowstyle="-|>", color=C["ens_init"], lw=1.1))
        ax.annotate("", xy=(g1, yy - 0.12), xytext=(g0, yy - 0.12), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, ls="--"))
        ax.text(f1 * 0.92, yy + 0.12, f"{f1:.1f}", ha="right", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xscale("log"); ax.set_xlim(0.3, 14)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.5, 1, 2, 4, 8]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["0.5", "1", "2", "4", "8"]))
    ax.minorticks_off(); ax.axvline(1, color=INK2, ls="--", lw=0.8); ax.grid(axis="x", color=GRID, lw=0.4)
    ax.set_xlabel("RMS $z$ of the random-init ensemble")
    ax.set_title("(b) held-out-label rescaling", loc="left")
    h = [Line2D([], [], color=C["ens_init"], lw=1.2, label="free energy"),
         Line2D([], [], color=INK2, lw=1, ls="--", label="mean force")]
    ax.legend(handles=h, frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(w_pad=2)
    save(fig, "fig10_transfer")


if __name__ == "__main__":
    which = sys.argv[1:] or ["workflow", "models", "systems", "2d_curves", "2d_z", "ablation",
                             "4d_tradeoff", "density", "ala2", "transfer"]
    for w in which:
        globals()["fig_" + w]()
