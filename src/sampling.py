"""Restrained overdamped Langevin sampling -> mean-force labels.

Each window restrains (x, y) harmonically around a centre c with stiffness k;
the hidden coordinate z is free. The label for a window is the
average-force estimate of the FES gradient,

    g_hat = -< f_phys(x, y, z) >,    located at  s_hat = < (x, y) >,

where f_phys excludes the restraint. Because <z^2 | x,y> = kT/kz(x,y),
E[g_hat] -> grad F evaluated (to second order) at s_hat. Label noise comes
from finite, time-correlated trajectories (z relaxes on 1/kz, up to ~4000
steps here), so naive error bars are optimistic. Each label carries a
block-averaged standard error, which is what a practitioner would report.

All windows are integrated simultaneously (vectorised).
"""
import numpy as np
from system import full_force, free_energy, DOMAIN, KT, log_kz


def window_centres(spacing=0.1, f_cut=12.0):
    """Regular grid of restraint centres inside the low-F region."""
    xs = np.arange(DOMAIN["xmin"], DOMAIN["xmax"] + 1e-9, spacing)
    ys = np.arange(DOMAIN["ymin"], DOMAIN["ymax"] + 1e-9, spacing)
    X, Y = np.meshgrid(xs, ys)
    F, _, _ = free_energy(X.ravel(), Y.ravel())
    F = F - F.min()
    keep = F < f_cut
    return np.stack([X.ravel()[keep], Y.ravel()[keep]], 1)


def sample_labels(centres, n_steps, k_res=4000.0, dt=2.5e-5, n_blocks=5,
                  burn=2000, seed=0):
    """Run all windows; return label locations, gradient labels, errors.

    Returns dict with
      s      (W,2)  mean restrained position (label location)
      g      (W,2)  average-force gradient estimate
      se     (W,2)  block-averaged standard error of g  (n_blocks blocks)
      se_iid (W,2)  naive standard error assuming independent steps
    """
    rng = np.random.default_rng(seed)
    W = len(centres)
    x, y = centres[:, 0].copy(), centres[:, 1].copy()
    lk, _, _ = log_kz(x, y)
    z = rng.normal(0, np.sqrt(KT / np.exp(lk)))      # start z at equilibrium
    noise = np.sqrt(2 * KT * dt)
    L = n_steps // n_blocks
    bs = np.zeros((n_blocks, W, 4))                   # block sums of fx,fy,x,y
    sq = np.zeros((W, 2))
    total = burn + L * n_blocks
    for t in range(total):
        fx, fy, fz = full_force(x, y, z)
        if t >= burn:
            b = (t - burn) // L
            bs[b, :, 0] += fx
            bs[b, :, 1] += fy
            bs[b, :, 2] += x
            bs[b, :, 3] += y
            sq[:, 0] += fx * fx
            sq[:, 1] += fy * fy
        rx = k_res * (centres[:, 0] - x)
        ry = k_res * (centres[:, 1] - y)
        xi = rng.standard_normal((3, W))
        x = x + dt * (fx + rx) + noise * xi[0]
        y = y + dt * (fy + ry) + noise * xi[1]
        z = z + dt * fz + noise * xi[2]
    bm = bs / L                                       # block means
    mean = bm.mean(0)
    g = -mean[:, :2]
    s = mean[:, 2:]
    se = bm[:, :, :2].std(0, ddof=1) / np.sqrt(n_blocks)
    n = L * n_blocks
    var_iid = sq / n - mean[:, :2] ** 2
    se_iid = np.sqrt(np.maximum(var_iid, 0) / n)
    return dict(s=s, g=g, se=se, se_iid=se_iid, centres=centres)


def estimator_bias(centres, k_res=4000.0, half_width=0.08, n=121):
    """Exact systematic bias of the average-force label for each window.

    Under the continuous restrained distribution p(s) ~ exp(-F(s) - k|s-c|^2/2)
    the label's expected value is E_p[grad F], but it is placed at E_p[s].
    Bias b = E_p[grad F] - grad F(E_p[s]) is computed by quadrature (z is
    already integrated out in F). Independent of trajectory length.
    """
    h = np.linspace(-half_width, half_width, n)
    HX, HY = np.meshgrid(h, h)
    b = np.zeros((len(centres), 2))
    for i, c in enumerate(centres):
        X, Y = c[0] + HX, c[1] + HY
        F, Fx, Fy = free_energy(X, Y)
        w = np.exp(-(F - F.min()) - 0.5 * k_res * (HX**2 + HY**2))
        w /= w.sum()
        sx, sy = (w * X).sum(), (w * Y).sum()
        _, gx, gy = free_energy(sx, sy)
        b[i] = [(w * Fx).sum() - gx, (w * Fy).sum() - gy]
    return b


def apply_label_mode(d, mode, seed=0, bias=None):
    """Return a copy of the label dict d transformed for an ablation.

    md        unchanged MD labels (systematic bias + real, correlated noise;
              block errors underestimate the noise)
    debiased  MD labels minus the exact estimator bias: same noise, same
              block errors, zero systematic error
    oracle    labels = exact grad F at the label location + Gaussian noise
              whose std equals the reported block error: error bars are
              correct by construction, no bias
    """
    d = dict(d)
    if mode == "md":
        return d
    if mode == "debiased":
        b = estimator_bias(d["centres"]) if bias is None else bias
        d["g"] = d["g"] - b
        return d
    if mode == "oracle":
        rng = np.random.default_rng(seed + 7919)
        _, gx, gy = free_energy(d["s"][:, 0], d["s"][:, 1])
        d["g"] = np.stack([gx, gy], 1) + rng.standard_normal(d["g"].shape) * d["se"]
        return d
    raise ValueError(mode)


if __name__ == "__main__":
    import time
    C = window_centres()
    print("windows:", len(C))
    for T in (0.4, 4.0):
        t0 = time.time()
        n = int(T / 2.5e-5)
        d = sample_labels(C, n, seed=1)
        _, gx, gy = free_energy(d["s"][:, 0], d["s"][:, 1])
        err = d["g"] - np.stack([gx, gy], 1)
        z = err / d["se"]
        zi = err / d["se_iid"]
        print(f"n={n}: {time.time()-t0:.1f}s | RMS err {np.sqrt((err**2).mean()):.3f} kT/unit"
              f" | mean err {err.mean(0).round(3)} | std(z_block) {z.std():.2f}"
              f" | std(z_iid) {zi.std():.2f} | median se {np.median(d['se']):.3f}")
