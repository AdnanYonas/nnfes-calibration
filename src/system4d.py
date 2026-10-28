"""4D test system with an exactly known free-energy surface.

Collective variables s = (x, y, u, v); hidden coordinate z is integrated out.

  F(s) = a*MB(x,y)                                  Mueller-Brown backbone
       + c*(u^2 - 1)^2 + b*u*(y - Y0)                double well in u, tilted by y
       + 0.5*kv*(v - 0.5*sin(2x) - 0.3*u)^2          v slaved to x and u (curved valley)
       + kT*[eps*sin(3x)cos(3y) + eps2*cos(2u)sin(2v)]   entropic term from z

The entropic term arises exactly as in the 2D system: the hidden z has
stiffness kz(s) = k0*exp(2*[...]) and integrating it out adds 0.5 kT ln kz.
So the physical force on s fluctuates with z^2 and labels are noisy and
time-correlated. The coupling makes F genuinely non-separable: the basin
in u that is preferred flips between minimum A (high y) and B (low y).
All functions take s as an (N, 4) array.  Units kT = 1.
"""
import numpy as np
from scipy.optimize import minimize
from system import mb, A_SCALE, K0, EPS, KT

DIM = 2 + 2
C_U, B_U, Y0 = 2.5, 3.0, 0.7
KV = 20.0
EPS2 = 0.7
NAMES = ("x", "y", "u", "v")
LO = np.array([-1.6, -0.4, -1.9, -1.6])
HI = np.array([1.1, 2.0, 1.9, 1.6])


def _lnk(s):
    """ln kz (relative) and its gradient."""
    x, y, u, v = s.T
    e = EPS * np.sin(3 * x) * np.cos(3 * y) + EPS2 * np.cos(2 * u) * np.sin(2 * v)
    g = np.stack([EPS * 3 * np.cos(3 * x) * np.cos(3 * y),
                  -EPS * 3 * np.sin(3 * x) * np.sin(3 * y),
                  -EPS2 * 2 * np.sin(2 * u) * np.sin(2 * v),
                  EPS2 * 2 * np.cos(2 * u) * np.cos(2 * v)], 1)
    return 2 * e, 2 * g


def potential_part(s):
    """Everything in F except the entropic term, and its gradient."""
    s = np.atleast_2d(s)
    x, y, u, v = s.T
    m, mx, my = mb(x, y)
    w = v - 0.5 * np.sin(2 * x) - 0.3 * u
    U = A_SCALE * m + C_U * (u**2 - 1) ** 2 + B_U * u * (y - Y0) + 0.5 * KV * w**2
    g = np.stack([A_SCALE * mx - KV * w * np.cos(2 * x),
                  A_SCALE * my + B_U * u,
                  4 * C_U * u * (u**2 - 1) + B_U * (y - Y0) - 0.3 * KV * w,
                  KV * w], 1)
    return U, g


def free_energy(s):
    """Exact F and grad F at s (N,4)."""
    s = np.atleast_2d(s)
    U, gU = potential_part(s)
    lk, glk = _lnk(s)
    return U + 0.5 * KT * lk, gU + 0.5 * KT * glk


def full_force(s, z):
    """Physical force on s and on z in the full (s, z) system."""
    _, gU = potential_part(s)
    lk, glk = _lnk(s)
    kz = K0 * np.exp(lk)
    fs = -(gU + 0.5 * (z**2 * kz)[:, None] * glk)
    fz = -kz * z
    return fs, fz


def kz_of(s):
    return K0 * np.exp(_lnk(s)[0])


def _stationary():
    f = lambda p: free_energy(p[None])[0][0]
    jac = lambda p: free_energy(p[None])[1][0]
    g2 = lambda p: (free_energy(p[None])[1][0] ** 2).sum()
    mins = {}
    for name, p0 in dict(A=(-0.55, 1.45, -1.0, 0.3), B=(0.62, 0.03, 1.0, 0.8),
                         C=(-0.05, 0.47, 1.0, 0.3)).items():
        mins[name] = minimize(f, p0, jac=jac, method="BFGS", options=dict(gtol=1e-10)).x
    # B-C saddle (index 1, ~8 kT above A, inside the sampled region).
    # The A-C saddles lie at 12.4 / 13.1 kT, above the window cut-off.
    from scipy.optimize import root
    h = 1e-5
    hess = lambda p: np.stack([(jac(p + h * e) - jac(p - h * e)) / (2 * h) for e in np.eye(DIM)])
    sad = root(jac, (0.21, 0.29, 1.0, 0.5), jac=hess, method="hybr").x
    return mins, {"S_BC": sad}


MINIMA, SADDLES = _stationary()
REF_POINT = MINIMA["A"]


def window_centres(n_target=1500, f_cut=12.0, seed=0):
    """Quasi-random (Sobol) restraint centres in the low-F region."""
    from scipy.stats import qmc
    sob = qmc.Sobol(DIM, scramble=True, seed=seed)
    pts = []
    F0 = free_energy(REF_POINT[None])[0][0]
    while sum(len(p) for p in pts) < n_target:
        c = LO + (HI - LO) * sob.random(4096)
        F = free_energy(c)[0] - F0
        pts.append(c[F < f_cut])
    return np.vstack(pts)[:n_target]


if __name__ == "__main__":
    F0 = free_energy(REF_POINT[None])[0][0]
    for k, p in {**MINIMA, **SADDLES}.items():
        F, g = free_energy(p[None])
        print(k, p.round(4), "F-F(A) = %.3f" % (F[0] - F0), "|grad| = %.1e" % np.abs(g).max())
    # numerical check of gradient
    rng = np.random.default_rng(0)
    s = LO + (HI - LO) * rng.random((5, 4)); h = 1e-6
    _, g = free_energy(s)
    fd = np.stack([(free_energy(s + h * e)[0] - free_energy(s - h * e)[0]) / (2 * h) for e in np.eye(4)], 1)
    print("grad check max err", np.abs(g - fd).max())
    # check that <full force> over z equals -grad F
    z = rng.standard_normal((200000,)) / np.sqrt(kz_of(np.repeat(s[:1], 200000, 0)))
    fs, _ = full_force(np.repeat(s[:1], 200000, 0), z)
    print("mean force vs -gradF:", fs.mean(0).round(3), (-g[0]).round(3))
    C = window_centres()
    from scipy.stats import qmc
    frac = (free_energy(LO + (HI - LO) * qmc.Sobol(4, seed=1).random(8192))[0] - F0 < 12).mean()
    print("windows", len(C), "| fraction of box with F<12:", round(frac, 3))
