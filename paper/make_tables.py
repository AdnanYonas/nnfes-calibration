"""Write the LaTeX result tables (tables/*.tex) straight from the saved results,
so every number in the manuscript comes from the result files."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(HERE, "tables")
os.makedirs(OUT, exist_ok=True)
L = lambda p: json.load(open(os.path.join(ROOT, p)))
NAME = dict(ens_init="Random-init ensemble", ens_init_rc="\\quad + held-out rescaling",
            ens_boot="Bootstrap ensemble", ens_boot_rc="\\quad + held-out rescaling",
            ens_noise="Noise-injected ensemble", ens_noise_rc="\\quad + held-out rescaling",
            gp="Gradient GP")


def f(v, d=2):
    return f"{v:.{d}f}"


def pct(v):
    return f"{100 * v:.0f}"


def write(name, s):
    open(os.path.join(OUT, name + ".tex"), "w").write(s)
    print("wrote", name)


def main_table(S, methods, caption, label, extra=""):
    rows = []
    for m in methods:
        a, b = S["short"][m], S["long"][m]
        rows.append(f"{NAME[m]} & {f(a['force_z_rms'])} & {f(a['F_z_rms'])} & {pct(a['F_cov95'])} & {f(a['F_rmse'], 3)} & {pct(a['targets_cov95'])}"
                    f" & {f(b['force_z_rms'])} & {f(b['F_z_rms'])} & {pct(b['F_cov95'])} & {f(b['F_rmse'], 3)} & {pct(b['targets_cov95'])} \\\\")
    lab = (f"Labels (block errors) & \\multicolumn{{5}}{{c}}{{95\\% coverage {pct(S['short']['labels_cov95']['labels_block'])}\\%}}"
           f" & \\multicolumn{{5}}{{c}}{{95\\% coverage {pct(S['long']['labels_cov95']['labels_block'])}\\%}} \\\\")
    return f"""\\begin{{table*}}
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{ruledtabular}}
\\begin{{tabular}}{{lcccccccccc}}
 & \\multicolumn{{5}}{{c}}{{Short runs}} & \\multicolumn{{5}}{{c}}{{Long runs}} \\\\
\\cline{{2-6}}\\cline{{7-11}}
Method & $z_{{\\nabla F}}$ & $z_F$ & cov$_{{95}}^F$ (\\%) & RMSE$_F$ (kT) & cov$_{{95}}^{{\\rm tgt}}$ (\\%) & $z_{{\\nabla F}}$ & $z_F$ & cov$_{{95}}^F$ (\\%) & RMSE$_F$ (kT) & cov$_{{95}}^{{\\rm tgt}}$ (\\%) \\\\
\\hline
{chr(10).join(rows)}
\\hline
{lab}
\\end{{tabular}}
\\end{{ruledtabular}}
{extra}\\end{{table*}}
"""


M6 = ["ens_init", "ens_init_rc", "ens_boot", "ens_boot_rc", "ens_noise", "ens_noise_rc", "gp"]

# Table III: 2D
S2 = L("results/pilot_results.json")
write("tab_2d", main_table(S2, [m for m in M6 if m in S2["short"]],
      "Two-dimensional system with exact $F$: calibration and accuracy (10 replicas per regime, 245 windows). "
      "$z$ is the RMS $z$-score (1 = calibrated, $>1$ = overconfident) for the mean force ($z_{\\nabla F}$) and the anchored "
      "free energy ($z_F$) on the test grid; cov$_{95}$ is the fraction of nominal 95\\% intervals that contain the truth, "
      "for $F$ on the grid and for the three scalar targets across replicas (30 estimates per cell).", "tab:2d"))

# Table IV: ablation
modes = [("results", "MD"), ("results_debiased", "bias removed"), ("results_oracle", "oracle")]
D = {lab: L(f"{p}/pilot_results.json") for p, lab in modes}
rows = []
for m in ["ens_init", "ens_init_rc", "ens_boot", "ens_noise", "ens_noise_rc", "gp"]:
    rows.append(f"{NAME[m]} & " + " & ".join(f(D[l]['short'][m]['F_z_rms']) for _, l in modes) + " & "
                + " & ".join(f(D[l]['long'][m]['F_z_rms']) for _, l in modes) + " \\\\")
lab = ("Label 95\\% coverage (\\%) & " + " & ".join(pct(D[l]['short']['labels_cov95']['labels_block']) for _, l in modes) + " & "
       + " & ".join(pct(D[l]['long']['labels_cov95']['labels_block']) for _, l in modes) + " \\\\")
write("tab_ablation", f"""\\begin{{table*}}
\\caption{{Label ablation on the 2D system (same MD seeds, paired): free-energy RMS $z$-score $z_F$ with raw MD labels,
with the exact estimator bias subtracted, and with oracle labels (exact gradient plus Gaussian noise whose standard
deviation equals the reported block error).}}
\\label{{tab:ablation}}
\\begin{{ruledtabular}}
\\begin{{tabular}}{{lcccccc}}
 & \\multicolumn{{3}}{{c}}{{Short runs}} & \\multicolumn{{3}}{{c}}{{Long runs}} \\\\
\\cline{{2-4}}\\cline{{5-7}}
Method & MD & debiased & oracle & MD & debiased & oracle \\\\
\\hline
{chr(10).join(rows)}
\\hline
{lab}
\\end{{tabular}}
\\end{{ruledtabular}}
\\end{{table*}}
""")

# Table V: 4D default vs tuned
A, B = L("results_4d/pilot_results.json"), L("results_4d_tuned/pilot_results.json")
rows = []
for m in ["ens_init", "ens_init_rc", "ens_boot", "ens_noise", "ens_noise_rc", "gp"]:
    r = [f"{f(A[g][m]['F_rmse'])}$\\to${f(B[g][m]['F_rmse'])}" if m != "gp" else f(A[g][m]['F_rmse']) for g in ("short", "long")]
    z = [f"{f(A[g][m]['F_z_rms'])}$\\to${f(B[g][m]['F_z_rms'])}" if m != "gp" else f(A[g][m]['F_z_rms']) for g in ("short", "long")]
    rows.append(f"{NAME[m]} & {r[0]} & {z[0]} & {r[1]} & {z[1]} \\\\")
write("tab_4d", f"""\\begin{{table*}}
\\caption{{Four-dimensional system (8 replicas, 1000 Sobol windows): free-energy RMSE (kT) and RMS $z$-score for the
default network settings $\\to$ the settings selected on held-out labels only ($\\lambda=3\\times10^{{-3}}$, 8000 steps).
The GP is unchanged.}}
\\label{{tab:4d}}
\\begin{{ruledtabular}}
\\begin{{tabular}}{{lcccc}}
 & \\multicolumn{{2}}{{c}}{{Short runs}} & \\multicolumn{{2}}{{c}}{{Long runs}} \\\\
\\cline{{2-3}}\\cline{{4-5}}
Method & RMSE$_F$ & $z_F$ & RMSE$_F$ & $z_F$ \\\\
\\hline
{chr(10).join(rows)}
\\end{{tabular}}
\\end{{ruledtabular}}
\\end{{table*}}
""")

# Table VI: density sweep
Dn = L("results_density_lbfgs/density_results.json"); Rn = L("results_density_lbfgs/density_raw.json")
rows = []
for n in (500, 1000, 2000, 4000):
    x = Dn[f"n{n}"]; t = {k: np.mean([r["timing"][k] for r in Rn[f"n{n}"]]) for k in ("ens_init", "gp")}
    rows.append(f"{n} ({x['n_train']}) & {f(x['ens_init']['F_z_rms'])} & {f(x['ens_init_rc']['F_z_rms'])} & {f(x['ens_boot']['F_z_rms'])} & {f(x['gp']['F_z_rms'])}"
                f" & {f(x['ens_init']['F_rmse'], 3)} & {f(x['ens_boot']['F_rmse'], 3)} & {f(x['gp']['F_rmse'], 3)}"
                f" & {pct(x['ens_boot']['targets_cov95'])} & {pct(x['gp']['targets_cov95'])} & {t['ens_init']:.0f} & {t['gp']:.0f} \\\\")
write("tab_density", f"""\\begin{{table*}}
\\caption{{Data-density sweep on the 4D system (nested Sobol window sets, short runs, fixed test set of 3000 points with
$F<10$ kT, 3 replicas per density, networks with L-BFGS polish). Times are wall-clock seconds per fit on one CPU
(8 ensemble members trained together).}}
\\label{{tab:density}}
\\begin{{ruledtabular}}
\\begin{{tabular}}{{lccccccccccc}}
 & \\multicolumn{{4}}{{c}}{{$z_F$}} & \\multicolumn{{3}}{{c}}{{RMSE$_F$ (kT)}} & \\multicolumn{{2}}{{c}}{{cov$_{{95}}^{{\\rm tgt}}$ (\\%)}} & \\multicolumn{{2}}{{c}}{{time (s)}} \\\\
\\cline{{2-5}}\\cline{{6-8}}\\cline{{9-10}}\\cline{{11-12}}
$W$ (train) & init & init+resc. & boot & GP & init & boot & GP & boot & GP & ens. & GP \\\\
\\hline
{chr(10).join(rows)}
\\end{{tabular}}
\\end{{ruledtabular}}
\\end{{table*}}
""")

# Table VII: alanine dipeptide pooled
SA = L("ala2/ala2_pooled_results.json")
lc = L("ala2/label_check.json")
extra = (f"\\footnotetext{{}}")
write("tab_ala2", main_table(SA, M6,
      "Alanine dipeptide (vacuum, 300 K) against the metadynamics reference: 10 replicas per regime (5 on each of two "
      "machines), 372 windows, 1204 test points with $F_{\\rm ref}<8$ kT. All $z$-scores include the reference's own "
      "standard error in quadrature. Columns as in Table~\\ref{tab:2d}.", "tab:ala2"))

# label diagnosis numbers for the text
open(os.path.join(OUT, "numbers.json"), "w").write(json.dumps(lc, indent=1))
print(json.dumps(lc, indent=1))
