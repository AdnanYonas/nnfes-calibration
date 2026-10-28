"""Pilot: are neural-network free-energy error bars calibrated?

For each label regime (short / long restrained trajectories) and each of R
independent replicas (new MD seeds), every method is trained on the same
labels and scored against the exact F of system.py:

  force level : predicted gradient +- sigma vs exact grad F   (on a test grid)
  F level     : anchored F(s) - F(ref) +- sigma vs exact       (on a test grid)
  targets     : dF_BA, dF_CA, barrier_AS1 +- sigma vs exact     (across replicas)

Methods: ens_init, ens_boot, ens_noise, the last two recalibrated on held-out
labels (suffix _rc), and gp. Results -> results/pilot_results.json and
results/pilot_raw.npz.

Usage:  python run_pilot.py --replicas 10 [--quick] [--label-mode md|debiased|oracle]
"""
import argparse, json, os, time
import numpy as np
from scipy.stats import spearmanr
from system import free_energy, MINIMA, SADDLES, REF_POINT, DOMAIN
from sampling import window_centres, sample_labels, apply_label_mode, estimator_bias
from models import (train_ensemble, ensemble_predict, GradGP,
                    label_scale_factor, DEVICE)

REGIMES = {"short": 0.4, "long": 4.0}          # trajectory length per window (time units)
DT = 2.5e-5
TARGET_PTS = np.array([MINIMA["B"], MINIMA["C"], SADDLES["S1"]])
TARGET_NAMES = ["dF_BA", "dF_CA", "barrier_AS1"]
LEVELS = np.linspace(0.05, 0.95, 19)            # nominal coverage levels


def test_grid(labels_s, n=60, f_cut=10.0, max_dist=0.07):
    """Grid points in the low-F region that lie within max_dist of a label
    (interpolation regime)."""
    xs = np.linspace(DOMAIN["xmin"], DOMAIN["xmax"], n)
    ys = np.linspace(DOMAIN["ymin"], DOMAIN["ymax"], n)
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X.ravel(), Y.ravel()], 1)
    F = free_energy(P[:, 0], P[:, 1])[0] - free_energy(*REF_POINT)[0]
    d = np.sqrt(((P[:, None] - labels_s[None]) ** 2).sum(-1)).min(1)
    keep = (F < f_cut) & (d < max_dist)
    return P[keep]


def calib(err, sig):
    """Calibration summary for Gaussian intervals."""
    err, sig = np.ravel(err), np.ravel(sig)
    z = err / np.maximum(sig, 1e-12)
    from scipy.stats import norm
    obs = np.array([(np.abs(z) < norm.ppf(0.5 + p / 2)).mean() for p in LEVELS])
    rho = spearmanr(sig, np.abs(err)).correlation if sig.std() > 0 else np.nan
    return dict(cov68=float((np.abs(z) < 1.0).mean()),
                cov95=float((np.abs(z) < 1.96).mean()),
                z_rms=float(np.sqrt(np.mean(z**2))),        # 1 if calibrated
                rmse=float(np.sqrt(np.mean(err**2))),
                mean_sigma=float(np.sqrt(np.mean(sig**2))),
                spearman=float(rho),
                miscal_area=float(np.mean(np.abs(obs - LEVELS))),
                curve=obs.tolist())


def run_replica(regime, rep, centres, M, epochs, holdout=0.15, label_mode="md", bias=None):
    seed = 1000 * (list(REGIMES).index(regime) + 1) + rep
    rng = np.random.default_rng(seed)
    d = sample_labels(centres, int(REGIMES[regime] / DT), seed=seed)
    d = apply_label_mode(d, label_mode, seed=seed, bias=bias)
    idx = rng.permutation(len(centres))
    n_ho = int(holdout * len(idx))
    ho, tr = idx[:n_ho], idx[n_ho:]
    s, g, se = d["s"][tr], d["g"][tr], d["se"][tr]

    grid = test_grid(s)
    pts = np.vstack([grid, TARGET_PTS])
    F0 = free_energy(*REF_POINT)[0]
    Ft, gx, gy = free_energy(pts[:, 0], pts[:, 1])
    Ft = Ft - F0
    gt = np.stack([gx, gy], 1)
    ng = len(grid)

    # label-level calibration (the inputs every method sees)
    _, lgx, lgy = free_energy(d["s"][:, 0], d["s"][:, 1])
    lab_err = d["g"] - np.stack([lgx, lgy], 1)
    out = {"labels_block": calib(lab_err, d["se"]),
           "labels_iid": calib(lab_err, d["se_iid"])}
    preds = {}

    for kind in ("ens_init", "ens_boot", "ens_noise"):
        net = train_ensemble(s, g, se, kind, M=M, epochs=epochs, seed=seed)
        Fm, gm = ensemble_predict(net, pts, REF_POINT)
        muF, sF = Fm.mean(0), Fm.std(0, ddof=1)
        mug, sg = gm.mean(0), gm.std(0, ddof=1)
        preds[kind] = (muF, sF, mug, sg)
        # recalibration from held-out labels only (no ground truth)
        Fh, gh = ensemble_predict(net, d["s"][ho], REF_POINT)
        a = label_scale_factor(gh.mean(0), gh.std(0, ddof=1), d["g"][ho], d["se"][ho])
        preds[kind + "_rc"] = (muF, a * sF, mug, a * sg)
        out[kind + "_rc_alpha"] = float(a)

    gp = GradGP(s, g, se)
    preds["gp"] = gp.predict(pts, REF_POINT)
    out["gp_hyper"] = dict(sf=float(gp.sf), l=float(gp.l))

    for name, (muF, sF, mug, sg) in preds.items():
        out[name] = dict(force=calib(mug[:ng] - gt[:ng], sg[:ng]),
                         F=calib(muF[:ng] - Ft[:ng], sF[:ng]),
                         targets=dict(est=muF[ng:].tolist(), sig=sF[ng:].tolist(),
                                      true=Ft[ng:].tolist()))
    out["n_train"], out["n_grid"] = int(len(tr)), int(ng)
    return out


def summarise(all_out):
    summ = {}
    for regime, reps in all_out.items():
        summ[regime] = {}
        lab = {k: np.mean([r[k]["cov95"] for r in reps]) for k in ("labels_block", "labels_iid")}
        summ[regime]["labels_cov95"] = lab
        methods = [k for k in reps[0] if isinstance(reps[0][k], dict) and "force" in reps[0][k]]
        for m in methods:
            row = {}
            for lvl in ("force", "F"):
                for key in ("cov68", "cov95", "z_rms", "rmse", "mean_sigma", "spearman", "miscal_area"):
                    row[f"{lvl}_{key}"] = float(np.mean([r[m][lvl][key] for r in reps]))
                row[f"{lvl}_curve"] = np.mean([r[m][lvl]["curve"] for r in reps], 0).tolist()
            est = np.array([r[m]["targets"]["est"] for r in reps])
            sig = np.array([r[m]["targets"]["sig"] for r in reps])
            tru = np.array([r[m]["targets"]["true"] for r in reps])
            z = (est - tru) / sig
            row["targets_cov68"] = float((np.abs(z) < 1).mean())
            row["targets_cov95"] = float((np.abs(z) < 1.96).mean())
            row["targets_z_rms"] = float(np.sqrt((z**2).mean()))
            row["targets_rmse_each"] = dict(zip(TARGET_NAMES, np.sqrt(((est - tru)**2).mean(0)).tolist()))
            row["targets_sigma_each"] = dict(zip(TARGET_NAMES, np.sqrt((sig**2).mean(0)).tolist()))
            if m + "_alpha" in reps[0]:
                pass
            summ[regime][m] = row
        summ[regime]["rc_alpha"] = {k: float(np.mean([r[k] for r in reps]))
                                    for k in reps[0] if k.endswith("_rc_alpha")}
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicas", type=int, default=10)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--quick", action="store_true", help="1 replica, 1500 epochs smoke test")
    ap.add_argument("--label-mode", default="md", choices=["md", "debiased", "oracle"],
                    help="md = raw MD labels; debiased = exact estimator bias removed; "
                         "oracle = exact gradient + noise matching the reported error bars")
    ap.add_argument("--out", default="../results")
    a = ap.parse_args()
    if a.quick:
        a.replicas, a.epochs = 1, 1500
    os.makedirs(a.out, exist_ok=True)
    centres = window_centres()
    bias = estimator_bias(centres) if a.label_mode == "debiased" else None
    print(f"device={DEVICE} windows={len(centres)} replicas={a.replicas} "
          f"members={a.members} epochs={a.epochs} label_mode={a.label_mode}", flush=True)
    all_out = {r: [] for r in REGIMES}
    t0 = time.time()
    for regime in REGIMES:
        for rep in range(a.replicas):
            o = run_replica(regime, rep, centres, a.members, a.epochs, label_mode=a.label_mode, bias=bias)
            all_out[regime].append(o)
            print(f"[{time.time()-t0:6.0f}s] {regime} rep {rep} labels_cov95={o['labels_block']['cov95']:.2f}: "
                  + " ".join(f"{m}:Fcov95={o[m]['F']['cov95']:.2f}"
                             for m in ("ens_init", "ens_noise", "ens_noise_rc", "gp")), flush=True)
            json.dump(all_out, open(os.path.join(a.out, "pilot_raw.json"), "w"))
    summ = summarise(all_out)
    json.dump(summ, open(os.path.join(a.out, "pilot_results.json"), "w"), indent=1)
    print(json.dumps({r: {m: {k: round(v, 3) for k, v in summ[r][m].items()
                              if k in ("force_cov95", "F_cov95", "F_z_rms", "targets_cov95", "F_rmse")}
                          for m in summ[r] if isinstance(summ[r][m], dict) and "F_cov95" in summ[r][m]}
                      for r in summ}, indent=1))


if __name__ == "__main__":
    main()
