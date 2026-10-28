"""Why do networks lose accuracy at high window density?

Same labels as the density sweep (replica 0). For each configuration a
4-member random-init ensemble is trained; we report F error on the fixed test
set, the error left after removing the best constant offset, and the
gradient error at the training points.  Usage: python diag_density.py
"""
import sys, time
import numpy as np
import system4d as S
from sampling_nd import sample_labels
from models import train_ensemble, ensemble_predict
from run_nd import CEN, SCL
from run_density import fixed_test_set, T_SHORT
from run_pilot import DT

test = fixed_test_set()
F0 = S.free_energy(S.REF_POINT[None])[0][0]
Ft = S.free_energy(test)[0] - F0
Call = S.window_centres(4000)
CONFIGS = {   # name: (width, weight decay, steps)
    "w64 wd3e-3 (current)": (64, 3e-3, 8000),
    "w64 wd scaled 850/N": (64, None, 8000),
    "w128 wd3e-3": (128, 3e-3, 8000),
    "w128 wd scaled 850/N": (128, None, 8000),
}
for n, di in ((2000, 2), (4000, 3)):
    seed = 9000 + 100 * di
    d = sample_labels(S, Call[:n], int(T_SHORT / DT), seed=seed)
    idx = np.random.default_rng(seed).permutation(n)
    tr = idx[int(0.15 * n):]
    s, g, se = d["s"][tr], d["g"][tr], d["se"][tr]
    _, Gs = S.free_energy(s)
    for name, (w, wd, steps) in CONFIGS.items():
        wd_ = 3e-3 * 850 / len(tr) if wd is None else wd
        t0 = time.time()
        net = train_ensemble(s, g, se, "ens_init", M=4, epochs=steps, wd=wd_, width=w,
                             seed=seed, cen=CEN, scl=SCL, batch=1024)
        Fm, _ = ensemble_predict(net, test, S.REF_POINT)
        _, gtr = ensemble_predict(net, s, S.REF_POINT)
        e = Fm.mean(0) - Ft
        print(f"windows {n} | {name:22s} wd={wd_:.1e} | {time.time()-t0:4.0f}s | F rmse {np.sqrt((e**2).mean()):.3f} "
              f"| after offset {np.sqrt(((e-e.mean())**2).mean()):.3f} | grad err {np.sqrt(((gtr.mean(0)-Gs)**2).mean()):.3f} "
              f"| z {np.sqrt(((e/Fm.std(0,ddof=1))**2).mean()):.2f}", flush=True)
