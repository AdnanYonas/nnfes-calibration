"""Hyperparameter search for the 4D ensembles using held-out labels only.

For each dataset (independent MD seed) and each configuration, an ensemble is
trained on 85% of labels; the selection criterion is the inverse-variance
weighted squared error on the 15% held-out labels (no ground truth).
The true F error is also recorded, only to test whether the criterion is a
valid proxy. Usage: python tune_4d.py [--out DIR]
"""
import argparse, itertools, json, os, sys, time
import numpy as np
import system4d as S
from sampling_nd import sample_labels
from models import train_ensemble, ensemble_predict, SE_FLOOR
from run_nd import test_points, CEN, SCL, TARGET_PTS
from run_pilot import REGIMES, DT

GRID = dict(wd=[1e-6, 1e-4, 1e-3, 3e-3], epochs=[4000, 8000], width=[64])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="5000,5001,5003,5005")
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--out", default="../results_tune4d")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    C = S.window_centres(1000)
    F0 = S.free_energy(S.REF_POINT[None])[0][0]
    rows = []
    t0 = time.time()
    for seed in map(int, a.seeds.split(",")):
        d = sample_labels(S, C, int(REGIMES["short"] / DT), seed=seed)
        idx = np.random.default_rng(seed).permutation(len(C))
        ho, tr = idx[:150], idx[150:]
        s, g, se = d["s"][tr], d["g"][tr], d["se"][tr]
        P = test_points(s, n=3000, seed=seed)
        Ft = S.free_energy(P)[0] - F0
        Tt = S.free_energy(TARGET_PTS)[0] - F0
        w_ho = 1.0 / (d["se"][ho] ** 2 + SE_FLOOR**2)
        for wd, ep, width in itertools.product(GRID["wd"], GRID["epochs"], GRID["width"]):
            net = train_ensemble(s, g, se, "ens_init", M=a.members, epochs=ep, wd=wd,
                                 width=width, seed=seed, cen=CEN, scl=SCL)
            _, gh = ensemble_predict(net, d["s"][ho], S.REF_POINT)
            ho_loss = float((w_ho * (gh.mean(0) - d["g"][ho]) ** 2).mean())
            Fm, _ = ensemble_predict(net, np.vstack([P, TARGET_PTS]), S.REF_POINT)
            mu, sd = Fm.mean(0), Fm.std(0, ddof=1)
            row = dict(seed=seed, wd=wd, epochs=ep, width=width, ho_loss=ho_loss,
                       F_rmse=float(np.sqrt(((mu[:-3] - Ft) ** 2).mean())),
                       F_zrms=float(np.sqrt((((mu[:-3] - Ft) / sd[:-3]) ** 2).mean())),
                       target_abs_err=np.abs(mu[-3:] - Tt).tolist())
            rows.append(row)
            print(f"[{time.time()-t0:5.0f}s] seed {seed} wd {wd:g} ep {ep} w {width}: "
                  f"held-out loss {ho_loss:.3f}  F rmse {row['F_rmse']:.3f}  F z {row['F_zrms']:.2f}  "
                  f"|dF_BA err| {row['target_abs_err'][0]:.2f}", flush=True)
            json.dump(rows, open(os.path.join(a.out, "tune_rows.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
