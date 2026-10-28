"""Restrained overdamped Langevin labels for an N-dimensional system module
(same estimator and error model as sampling.py, which is the 2D version).

`sysm` must provide DIM, full_force(s, z) -> (fs (W,D), fz (W,)),
kz_of(s), free_energy(s) -> (F, gradF).
"""
import numpy as np


def sample_labels(sysm, centres, n_steps, k_res=4000.0, dt=2.5e-5, n_blocks=5,
                  burn=2000, seed=0):
    rng = np.random.default_rng(seed)
    W, D = centres.shape
    s = centres.copy()
    z = rng.normal(0, np.sqrt(1.0 / sysm.kz_of(s)))
    noise = np.sqrt(2 * dt)
    L = n_steps // n_blocks
    bs = np.zeros((n_blocks, W, 2 * D))
    for t in range(burn + L * n_blocks):
        fs, fz = sysm.full_force(s, z)
        if t >= burn:
            b = (t - burn) // L
            bs[b, :, :D] += fs
            bs[b, :, D:] += s
        xi = rng.standard_normal((W, D + 1))
        s = s + dt * (fs + k_res * (centres - s)) + noise * xi[:, :D]
        z = z + dt * fz + noise * xi[:, D]
    bm = bs / L
    mean = bm.mean(0)
    se = bm[:, :, :D].std(0, ddof=1) / np.sqrt(n_blocks)
    return dict(s=mean[:, D:], g=-mean[:, :D], se=se, centres=centres)


def estimator_bias(sysm, centres, k_res=4000.0, n_mc=4000, seed=123):
    """Exact-in-expectation bias of the average-force label, by importance
    sampling: s ~ N(c, I/k) weighted by exp(-(F(s) - F(c)))."""
    rng = np.random.default_rng(seed)
    W, D = centres.shape
    b = np.zeros((W, D))
    for i, c in enumerate(centres):
        s = c + rng.standard_normal((n_mc, D)) / np.sqrt(k_res)
        F, G = sysm.free_energy(s)
        w = np.exp(-(F - F.min()))
        w /= w.sum()
        ms = (w[:, None] * s).sum(0)
        _, g0 = sysm.free_energy(ms[None])
        b[i] = (w[:, None] * G).sum(0) - g0[0]
    return b


def apply_label_mode(sysm, d, mode, seed=0, bias=None):
    d = dict(d)
    if mode == "md":
        return d
    if mode == "debiased":
        d["g"] = d["g"] - (estimator_bias(sysm, d["centres"]) if bias is None else bias)
        return d
    if mode == "oracle":
        rng = np.random.default_rng(seed + 7919)
        _, G = sysm.free_energy(d["s"])
        d["g"] = G + rng.standard_normal(G.shape) * d["se"]
        return d
    raise ValueError(mode)
