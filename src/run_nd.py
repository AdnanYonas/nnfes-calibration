"""4D pilot: same protocol as run_pilot.py, on system4d.

Usage:  python run_nd.py --replicas 10 [--quick] [--label-mode md|debiased|oracle] [--out DIR]
"""
import argparse, json, os, time
import numpy as np
import system4d as S
from sampling_nd import sample_labels, apply_label_mode, estimator_bias
from models import (train_ensemble, ensemble_predict, GradGPARD,
                    label_scale_factor, DEVICE)
import run_pilot
from run_pilot import calib, REGIMES, DT, summarise
run_pilot.TARGET_NAMES = ["dF_BA", "dF_CA", "barrier_SBC"]

TARGET_PTS = np.array([S.MINIMA["B"], S.MINIMA["C"], S.SADDLES["S_BC"]])
CEN, SCL = (S.LO + S.HI) / 2, (S.HI - S.LO) / 2


def test_points(labels_s, n=3000, f_cut=10.0, max_dist=0.12, seed=0):
    """Quasi-random points in the low-F region within max_dist of a label."""
    from scipy.stats import qmc
    sob = qmc.Sobol(S.DIM, scramble=True, seed=seed)
    F0 = S.free_energy(S.REF_POINT[None])[0][0]
    out = []
    while sum(len(o) for o in out) < n:
        p = S.LO + (S.HI - S.LO) * sob.random(8192)
        p = p[S.free_energy(p)[0] - F0 < f_cut]
        d = np.sqrt(((p[:, None] - labels_s[None]) ** 2).sum(-1)).min(1)
        out.append(p[d < max_dist])
    return np.vstack(out)[:n]


def run_replica(regime, rep, centres, M, epochs, label_mode, bias, holdout=0.15, wd=1e-6, gp_prev=None):
    seed = 5000 + 1000 * list(REGIMES).index(regime) + rep
    rng = np.random.default_rng(seed)
    d = sample_labels(S, centres, int(REGIMES[regime] / DT), seed=seed)
    d = apply_label_mode(S, d, label_mode, seed=seed, bias=bias)
    idx = rng.permutation(len(centres))
    n_ho = int(holdout * len(idx))
    ho, tr = idx[:n_ho], idx[n_ho:]
    s, g, se = d["s"][tr], d["g"][tr], d["se"][tr]

    grid = test_points(s, seed=seed)
    pts = np.vstack([grid, TARGET_PTS])
    F0 = S.free_energy(S.REF_POINT[None])[0][0]
    Ft, gt = S.free_energy(pts)
    Ft = Ft - F0
    ng = len(grid)
    _, lg = S.free_energy(d["s"])
    out = {"labels_block": calib(d["g"] - lg, d["se"]), "labels_iid": calib(d["g"] - lg, d["se"])}
    preds, timing = {}, {}
    for kind in ("ens_init", "ens_boot", "ens_noise"):
        t0 = time.time()
        net = train_ensemble(s, g, se, kind, M=M, epochs=epochs, wd=wd, seed=seed, cen=CEN, scl=SCL)
        timing[kind] = time.time() - t0
        Fm, gm = ensemble_predict(net, pts, S.REF_POINT)
        muF, sF, mug, sg = Fm.mean(0), Fm.std(0, ddof=1), gm.mean(0), gm.std(0, ddof=1)
        preds[kind] = (muF, sF, mug, sg)
        _, gh = ensemble_predict(net, d["s"][ho], S.REF_POINT)
        a = label_scale_factor(gh.mean(0), gh.std(0, ddof=1), d["g"][ho], d["se"][ho])
        preds[kind + "_rc"] = (muF, a * sF, mug, a * sg)
        out[kind + "_rc_alpha"] = float(a)
    if gp_prev is None:
        t0 = time.time()
        gp = GradGPARD(s, g, se, seed=seed)
        preds["gp"] = gp.predict(pts, S.REF_POINT)
        timing["gp"] = time.time() - t0
        out["gp_hyper"] = dict(sf=float(gp.sf), l=gp.l.tolist(), t_fit=gp.t_fit, t_factor=gp.t_factor)
    else:                       # identical seeds -> identical GP; reuse earlier result
        out["gp"], out["gp_hyper"] = gp_prev["gp"], gp_prev["gp_hyper"]
        timing["gp"] = gp_prev["timing"]["gp"]
    for name, (muF, sF, mug, sg) in preds.items():
        out[name] = dict(force=calib(mug[:ng] - gt[:ng], sg[:ng]),
                         F=calib(muF[:ng] - Ft[:ng], sF[:ng]),
                         targets=dict(est=muF[ng:].tolist(), sig=sF[ng:].tolist(),
                                      true=Ft[ng:].tolist()))
    out["n_train"], out["n_grid"], out["timing"] = int(len(tr)), int(ng), timing
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicas", type=int, default=8)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--windows", type=int, default=1000)
    ap.add_argument("--quick", action="store_true", help="1 replica, 1500 epochs smoke test")
    ap.add_argument("--label-mode", default="md", choices=["md", "debiased", "oracle"])
    ap.add_argument("--wd", type=float, default=1e-6, help="weight decay (tuned 4D value: 3e-3 with --epochs 8000)")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run: keep replicas already saved in OUT/pilot_raw.json")
    ap.add_argument("--gp-from", default=None, help="pilot_raw.json of an earlier run with the same seeds; reuse its GP results")
    ap.add_argument("--out", default="../results_4d")
    a = ap.parse_args()
    if a.quick:
        a.replicas, a.epochs = 1, 1500
    os.makedirs(a.out, exist_ok=True)
    centres = S.window_centres(a.windows)
    bias = estimator_bias(S, centres) if a.label_mode == "debiased" else None
    print(f"4D device={DEVICE} windows={len(centres)} replicas={a.replicas} members={a.members} "
          f"epochs={a.epochs} wd={a.wd} label_mode={a.label_mode}", flush=True)
    prev = json.load(open(a.gp_from)) if a.gp_from else None
    all_out = {r: [] for r in REGIMES}
    raw_path = os.path.join(a.out, "pilot_raw.json")
    if a.resume and os.path.exists(raw_path):
        all_out.update(json.load(open(raw_path)))
        print("resuming: already done " + ", ".join(f"{r}={len(all_out[r])}" for r in REGIMES), flush=True)
    t0 = time.time()
    for regime in REGIMES:
        for rep in range(len(all_out[regime]), a.replicas):
            o = run_replica(regime, rep, centres, a.members, a.epochs, a.label_mode, bias, wd=a.wd,
                            gp_prev=prev[regime][rep] if prev else None)
            all_out[regime].append(o)
            print(f"[{time.time()-t0:6.0f}s] {regime} rep {rep} labels_cov95={o['labels_block']['cov95']:.2f}: "
                  + " ".join(f"{m}:Fcov95={o[m]['F']['cov95']:.2f}" for m in ("ens_init", "ens_boot", "ens_noise", "gp"))
                  + f" | train s/ens {np.mean([o['timing'][k] for k in ('ens_init','ens_boot','ens_noise')]):.0f}"
                  + f" gp {o['timing']['gp']:.0f}", flush=True)
            json.dump(all_out, open(os.path.join(a.out, "pilot_raw.json"), "w"))
    summ = summarise(all_out)
    json.dump(summ, open(os.path.join(a.out, "pilot_results.json"), "w"), indent=1)
    print(json.dumps({r: {m: {k: round(v, 3) for k, v in summ[r][m].items()
                              if k in ("force_z_rms", "F_z_rms", "targets_cov95", "F_rmse")}
                          for m in summ[r] if isinstance(summ[r][m], dict) and "F_cov95" in summ[r][m]}
                      for r in summ}, indent=1))


if __name__ == "__main__":
    main()
