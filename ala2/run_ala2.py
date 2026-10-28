"""Alanine dipeptide calibration study (vacuum, 300 K, CVs phi/psi).

Same protocol as the toy systems, with a metadynamics reference as "truth":
  windows   15-degree grid on the torus where F_ref < 15 kT (covers the C7eq-C7ax barrier, 13 kT)
  labels    restrained MD (k = 1000 kJ/mol/rad^2), umbrella-integration mean force,
            5-block standard errors; regimes: short (10 ps) and long (50 ps) per window
  models    periodic networks (cos/sin inputs; wd 3e-3, 8000 Adam + 3000 L-BFGS),
            random-init / bootstrap / noise-injected ensembles (+ held-out-label
            rescaling) and a periodic-kernel gradient GP
  scoring   reference grid points with F_ref < 8 kT; z-scores include the reference's
            own standard error, so the test is conservative
  targets   dF(C7eq - C5), dF(C7ax - C5), barrier C7eq->C7ax (lowest saddle), anchored at C5

Usage: python run_ala2.py [--replicas 5] [--out DIR] [--resume] [--quick] [--procs N]
"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from ala2_reference import Reference, GRID, N
from ala2_labels import sample_labels
from models import train_ensemble, ensemble_predict, GradGPPeriodic, label_scale_factor, DEVICE
import run_pilot
from run_pilot import calib, summarise

REGIMES = {"short": 10.0, "long": 50.0}          # ps of production per window
run_pilot.TARGET_NAMES = ["dF_C7eq", "dF_C7ax", "barrier_C7eq_C7ax"]


def setup(ref):
    mins = ref.minima()
    def nearest(phi, psi):
        return min(mins, key=lambda m: ((m[0] - phi) ** 2 + (m[1] - psi) ** 2))
    c5 = np.array(nearest(np.radians(-150), np.radians(155))[:2])
    c7eq = np.array(nearest(np.radians(-80), np.radians(75))[:2])
    c7ax = np.array(nearest(np.radians(70), np.radians(-70))[:2])
    idx = lambda p: tuple(np.round((p + np.pi) / (2 * np.pi / N)).astype(int) % N)
    sad, _ = ref.minimax_saddle(idx(c7eq), idx(c7ax))
    g15 = np.radians(np.arange(-180, 180, 15.0))
    C = np.array([(a, b) for a in g15 for b in g15])
    C = C[ref.free_energy(C)[0] < 15.0]
    P = np.array([(GRID[i], GRID[j]) for i in range(N) for j in range(N) if ref.F[i, j] < 8.0])
    return dict(ref_pt=c5, targets=np.array([c7eq, c7ax, sad]), centres=C, test=P)


def anchored_ref_precision(ref, pts, anchor):
    vals = np.array([ref._eval(s, pts) - ref._eval(s, anchor[None])[0] for s in ref._run_spl])
    return vals.std(0, ddof=1) / np.sqrt(ref.n_runs)


def run_replica(regime, rep, S, ref, args):
    seed = 7000 + 1000 * list(REGIMES).index(regime) + rep
    d = sample_labels(S["centres"], REGIMES[regime], seed=seed, n_proc=args.procs)
    n = len(d["s"])
    idx = np.random.default_rng(seed).permutation(n)
    ho, tr = idx[:int(0.15 * n)], idx[int(0.15 * n):]
    s, g, se = d["s"][tr], d["g"][tr], d["se"][tr]
    pts = np.vstack([S["test"], S["targets"]])
    ng = len(S["test"])
    Ft, Gt = ref.free_energy(pts)
    Ft = Ft - ref.free_energy(S["ref_pt"][None])[0][0]
    sF_ref = anchored_ref_precision(ref, pts, S["ref_pt"])
    _, sG_ref = ref.precision(pts)
    _, lg = ref.free_energy(d["s"])
    out = {"labels_block": calib(d["g"] - lg, d["se"]), "labels_iid": calib(d["g"] - lg, d["se"])}
    preds, timing = {}, {}
    for kind in ("ens_init", "ens_boot", "ens_noise"):
        t0 = time.time()
        net = train_ensemble(s, g, se, kind, M=args.members, epochs=args.epochs, wd=args.wd,
                             seed=seed, lbfgs_steps=args.lbfgs, periodic=True)
        timing[kind] = time.time() - t0
        Fm, gm = ensemble_predict(net, pts, S["ref_pt"])
        preds[kind] = (Fm.mean(0), Fm.std(0, ddof=1), gm.mean(0), gm.std(0, ddof=1))
        _, gh = ensemble_predict(net, d["s"][ho], S["ref_pt"])
        a = label_scale_factor(gh.mean(0), gh.std(0, ddof=1), d["g"][ho], d["se"][ho])
        muF, sF, mug, sg = preds[kind]
        preds[kind + "_rc"] = (muF, a * sF, mug, a * sg)
        out[kind + "_rc_alpha"] = float(a)
    t0 = time.time()
    gp = GradGPPeriodic(s, g, se, seed=seed)
    preds["gp"] = gp.predict(pts, S["ref_pt"])
    timing["gp"] = time.time() - t0
    out["gp_hyper"] = dict(sf=float(gp.sf), l=gp.l.tolist())
    for name, (muF, sF, mug, sg) in preds.items():
        tot_F = np.sqrt(sF**2 + sF_ref**2)               # model + reference uncertainty
        tot_G = np.sqrt(sg**2 + sG_ref**2)
        out[name] = dict(force=calib(mug[:ng] - Gt[:ng], tot_G[:ng]),
                         F=calib(muF[:ng] - Ft[:ng], tot_F[:ng]),
                         F_model_only=calib(muF[:ng] - Ft[:ng], sF[:ng]),
                         targets=dict(est=muF[ng:].tolist(), sig=tot_F[ng:].tolist(),
                                      true=Ft[ng:].tolist()))
    out["n_train"], out["timing"] = int(len(tr)), timing
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicas", type=int, default=5)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8000)
    ap.add_argument("--wd", type=float, default=3e-3)
    ap.add_argument("--lbfgs", type=int, default=3000)
    ap.add_argument("--procs", type=int, default=None, help="parallel MD processes (default: all cores)")
    ap.add_argument("--out", default=os.path.join(HERE, "results_ala2"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 replica, short only, few steps")
    a = ap.parse_args()
    ref = Reference()
    S = setup(ref)
    regimes = list(REGIMES)
    if a.quick:
        a.replicas, a.epochs, a.lbfgs, regimes = 1, 300, 50, ["short"]
    os.makedirs(a.out, exist_ok=True)
    raw_path = os.path.join(a.out, "ala2_raw.json")
    all_out = {r: [] for r in regimes}
    if a.resume and os.path.exists(raw_path):
        all_out.update(json.load(open(raw_path)))
        print("resuming: already done " + ", ".join(f"{k}={len(v)}" for k, v in all_out.items()), flush=True)
    t = ref.free_energy(S["targets"])[0] - ref.free_energy(S["ref_pt"][None])[0][0]
    print(f"alanine dipeptide: reference runs={ref.n_runs} windows={len(S['centres'])} test points={len(S['test'])} "
          f"targets(kT)={np.round(t, 2).tolist()} device={DEVICE}", flush=True)
    t0 = time.time()
    for regime in regimes:
        for rep in range(len(all_out[regime]), a.replicas):
            o = run_replica(regime, rep, S, ref, a)
            all_out[regime].append(o)
            json.dump(all_out, open(raw_path, "w"))
            print(f"[{time.time()-t0:6.0f}s] {regime} rep {rep} labels_cov95={o['labels_block']['cov95']:.2f}: "
                  + " ".join(f"{m}: z={o[m]['F']['z_rms']:.2f} rmse={o[m]['F']['rmse']:.3f}"
                             for m in ("ens_init", "ens_boot", "ens_noise", "gp")), flush=True)
    summ = summarise({k: v for k, v in all_out.items() if v})
    json.dump(summ, open(os.path.join(a.out, "ala2_results.json"), "w"), indent=1)
    print("\nregime | F z-score: init / init_rc / boot / noise / gp | F RMSE: init / boot / gp | targets cov95 init/boot/gp")
    for r in summ:
        x = summ[r]
        print(f"{r:6s} | {x['ens_init']['F_z_rms']:.2f} / {x['ens_init_rc']['F_z_rms']:.2f} / {x['ens_boot']['F_z_rms']:.2f} / "
              f"{x['ens_noise']['F_z_rms']:.2f} / {x['gp']['F_z_rms']:.2f} | {x['ens_init']['F_rmse']:.3f} / "
              f"{x['ens_boot']['F_rmse']:.3f} / {x['gp']['F_rmse']:.3f} | {x['ens_init']['targets_cov95']:.2f}/"
              f"{x['ens_boot']['targets_cov95']:.2f}/{x['gp']['targets_cov95']:.2f}")


if __name__ == "__main__":
    main()
