"""4D data-density sweep: how do accuracy and honesty of error bars change
with the number of restraint windows?

- Nested Sobol window sets (the 500 windows are the first 500 of the 1000, ...).
- One FIXED test set (3000 points with F < 10 kT) for every density.
- Short trajectories, raw MD labels, tuned networks (wd 3e-3, 8000 Adam steps
  on mini-batches of at most 1024 labels, then 3000 full-batch L-BFGS
  iterations; without the L-BFGS polish networks under-converge at 4000
  windows because gradient matching penalises long-wavelength F errors weakly).
- Methods: random-init ensemble, bootstrap ensemble (+ held-out-label
  recalibration of both) and the ARD gradient GP. Compute time is recorded.

Usage:  python run_density.py [--replicas 5] [--densities 500,1000,2000,4000]
                              [--out DIR] [--resume] [--quick]
"""
import argparse, json, os, time
import numpy as np
from scipy.stats import qmc
import system4d as S
import run_pilot
from run_pilot import calib, summarise, DT
from sampling_nd import sample_labels
from models import train_ensemble, ensemble_predict, GradGPARD, label_scale_factor, DEVICE
from run_nd import TARGET_PTS, CEN, SCL

run_pilot.TARGET_NAMES = ["dF_BA", "dF_CA", "barrier_SBC"]
T_SHORT = 0.4


def fixed_test_set(n=3000, f_cut=10.0, seed=424242):
    sob = qmc.Sobol(S.DIM, scramble=True, seed=seed)
    F0 = S.free_energy(S.REF_POINT[None])[0][0]
    out = []
    while sum(len(o) for o in out) < n:
        p = S.LO + (S.HI - S.LO) * sob.random(8192)
        out.append(p[S.free_energy(p)[0] - F0 < f_cut])
    return np.vstack(out)[:n]


def run_one(n_win, di, rep, centres_all, test, M, epochs, wd, batch, holdout=0.15, lbfgs=0, gp_prev=None):
    seed = 9000 + 100 * di + rep
    C = centres_all[:n_win]
    d = sample_labels(S, C, int(T_SHORT / DT), seed=seed)
    idx = np.random.default_rng(seed).permutation(n_win)
    n_ho = int(holdout * n_win)
    ho, tr = idx[:n_ho], idx[n_ho:]
    s, g, se = d["s"][tr], d["g"][tr], d["se"][tr]
    pts = np.vstack([test, TARGET_PTS])
    F0 = S.free_energy(S.REF_POINT[None])[0][0]
    Ft, gt = S.free_energy(pts)
    Ft = Ft - F0
    ng = len(test)
    _, lg = S.free_energy(d["s"])
    out = {"labels_block": calib(d["g"] - lg, d["se"]), "labels_iid": calib(d["g"] - lg, d["se"])}
    preds, timing = {}, {}
    for kind in ("ens_init", "ens_boot"):
        t0 = time.time()
        net = train_ensemble(s, g, se, kind, M=M, epochs=epochs, wd=wd, seed=seed,
                             cen=CEN, scl=SCL, batch=batch, lbfgs_steps=lbfgs)
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
    else:                      # same seeds -> identical GP; reuse
        out["gp"], out["gp_hyper"] = gp_prev["gp"], gp_prev["gp_hyper"]
        timing["gp"] = gp_prev["timing"]["gp"]
    for name, (muF, sF, mug, sg) in preds.items():
        out[name] = dict(force=calib(mug[:ng] - gt[:ng], sg[:ng]),
                         F=calib(muF[:ng] - Ft[:ng], sF[:ng]),
                         targets=dict(est=muF[ng:].tolist(), sig=sF[ng:].tolist(),
                                      true=Ft[ng:].tolist()))
    out["n_train"], out["timing"] = int(len(tr)), timing
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicas", type=int, default=5)
    ap.add_argument("--densities", default="500,1000,2000,4000")
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8000)
    ap.add_argument("--wd", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--out", default="../results_density")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--lbfgs", type=int, default=3000,
                    help="full-batch L-BFGS polish iterations after Adam (0 = off, as in the first sweep)")
    ap.add_argument("--gp-from", default=None, help="density_raw.json of an earlier sweep with the same seeds; reuse its GP")
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 replica, 2 small densities, 300 steps")
    a = ap.parse_args()
    if a.quick:
        a.replicas, a.densities, a.epochs = 1, "200,400", 300
    dens = [int(x) for x in a.densities.split(",")]
    os.makedirs(a.out, exist_ok=True)
    centres_all = S.window_centres(max(dens))
    test = fixed_test_set()
    raw_path = os.path.join(a.out, "density_raw.json")
    all_out = {f"n{n}": [] for n in dens}
    if a.resume and os.path.exists(raw_path):
        all_out.update(json.load(open(raw_path)))
        print("resuming: already done " + ", ".join(f"{k}={len(v)}" for k, v in all_out.items()), flush=True)
    print(f"density sweep device={DEVICE} densities={dens} replicas={a.replicas} "
          f"steps={a.epochs} wd={a.wd} batch={a.batch} lbfgs={a.lbfgs}", flush=True)
    prev = json.load(open(a.gp_from)) if a.gp_from else None
    t0 = time.time()
    for rep in range(a.replicas):                 # replica-major: every density gets data early
        for di, n in enumerate(dens):
            key = f"n{n}"
            if len(all_out[key]) > rep:
                continue
            gp_prev = None
            if prev is not None and len(prev.get(key, [])) > rep:
                gp_prev = prev[key][rep]
            o = run_one(n, di, rep, centres_all, test, a.members, a.epochs, a.wd, a.batch,
                        lbfgs=a.lbfgs, gp_prev=gp_prev)
            all_out[key].append(o)
            json.dump(all_out, open(raw_path, "w"))
            print(f"[{time.time()-t0:6.0f}s] windows={n:5d} rep {rep}: "
                  + " ".join(f"{m}: z={o[m]['F']['z_rms']:.2f} rmse={o[m]['F']['rmse']:.3f}"
                             for m in ("ens_init", "ens_boot", "gp"))
                  + " | sec " + " ".join(f"{k}={v:.0f}" for k, v in o["timing"].items()), flush=True)
    done = {k: v for k, v in all_out.items() if len(v) > 0}
    summ = summarise(done)
    for k, v in done.items():
        summ[k]["timing"] = {m: float(np.mean([o["timing"][m] for o in v])) for m in v[0]["timing"]}
        summ[k]["gp_t_factor"] = float(np.mean([o["gp_hyper"]["t_factor"] for o in v]))
        summ[k]["n_train"] = v[0]["n_train"]
    json.dump(summ, open(os.path.join(a.out, "density_results.json"), "w"), indent=1)
    print("\nwindows | F z-score: init / init_rc / boot / gp | F RMSE: init / boot / gp")
    for k in done:
        x = summ[k]
        print(f"{k:6s} | {x['ens_init']['F_z_rms']:.2f} / {x['ens_init_rc']['F_z_rms']:.2f} / "
              f"{x['ens_boot']['F_z_rms']:.2f} / {x['gp']['F_z_rms']:.2f} | "
              f"{x['ens_init']['F_rmse']:.3f} / {x['ens_boot']['F_rmse']:.3f} / {x['gp']['F_rmse']:.3f}")


if __name__ == "__main__":
    main()
