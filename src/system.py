"""Test system with an exactly known free-energy surface (FES).

Collective variables s = (x, y). A hidden coordinate z is integrated out:

    U(x, y, z) = a * MB(x, y) + 0.5 * kz(x, y) * z**2
    kz(x, y)   = k0 * exp(2 * eps * sin(3x) * cos(3y))

Integrating exp(-U/kT) over z analytically gives the exact FES

    F(x, y) = a * MB(x, y) + 0.5 * kT * ln kz(x, y) + const
            = a * MB(x, y) + kT * eps * sin(3x) cos(3y) + const

so F differs from the bare potential by an entropic term, and the
instantaneous force on (x, y) fluctuates with z (z**2 enters the force),
which makes the mean-force labels genuinely noisy and time-correlated.
Units: kT = 1.
"""
import numpy as np
from scipy.optimize import minimize

KT = 1.0
A_SCALE = 0.1          # Mueller-Brown scaling -> barriers ~10 kT
K0 = 20.0              # base stiffness of hidden coordinate
EPS = 1.0              # entropic modulation amplitude (in kT)

_A = np.array([-200.0, -100.0, -170.0, 15.0])
_a = np.array([-1.0, -1.0, -6.5, 0.7])
_b = np.array([0.0, 0.0, 11.0, 0.6])
_c = np.array([-10.0, -10.0, -6.5, 0.7])
_x0 = np.array([1.0, 0.0, -0.5, -1.0])
_y0 = np.array([0.0, 0.5, 1.5, 1.0])


def mb(x, y):
    """Mueller-Brown potential and gradient (vectorised over x, y)."""
    x = np.asarray(x, float)[..., None]
    y = np.asarray(y, float)[..., None]
    dx, dy = x - _x0, y - _y0
    e = _A * np.exp(_a * dx**2 + _b * dx * dy + _c * dy**2)
    v = e.sum(-1)
    gx = (e * (2 * _a * dx + _b * dy)).sum(-1)
    gy = (e * (_b * dx + 2 * _c * dy)).sum(-1)
    return v, gx, gy


def log_kz(x, y):
    """ln kz and its gradient."""
    lk = np.log(K0) + 2 * EPS * np.sin(3 * x) * np.cos(3 * y)
    dlx = 2 * EPS * 3 * np.cos(3 * x) * np.cos(3 * y)
    dly = -2 * EPS * 3 * np.sin(3 * x) * np.sin(3 * y)
    return lk, dlx, dly


def free_energy(x, y):
    """Exact F(x, y) (kT units, arbitrary constant) and its gradient."""
    v, gx, gy = mb(x, y)
    lk, dlx, dly = log_kz(x, y)
    F = A_SCALE * v + 0.5 * KT * (lk - np.log(K0))
    Fx = A_SCALE * gx + 0.5 * KT * dlx
    Fy = A_SCALE * gy + 0.5 * KT * dly
    return F, Fx, Fy


def full_force(x, y, z):
    """Physical force -dU/d(x,y,z) of the full system."""
    _, gx, gy = mb(x, y)
    lk, dlx, dly = log_kz(x, y)
    kz = np.exp(lk)
    fx = -(A_SCALE * gx + 0.5 * z**2 * kz * dlx)
    fy = -(A_SCALE * gy + 0.5 * z**2 * kz * dly)
    fz = -kz * z
    return fx, fy, fz


DOMAIN = dict(xmin=-1.6, xmax=1.1, ymin=-0.4, ymax=2.0)


def _stationary_points():
    """Locate the three minima and two saddles of the exact F."""
    f = lambda p: free_energy(p[0], p[1])[0]
    g2 = lambda p: np.hypot(*free_energy(p[0], p[1])[1:])**2
    mins = {}
    for name, p0 in dict(A=(-0.56, 1.44), B=(0.62, 0.03), C=(-0.05, 0.47)).items():
        mins[name] = minimize(f, p0, method="Nelder-Mead",
                              options=dict(xatol=1e-9, fatol=1e-12)).x
    sads = {}
    for name, p0 in dict(S1=(-0.82, 0.62), S2=(0.21, 0.29)).items():
        sads[name] = minimize(g2, p0, method="Nelder-Mead",
                              options=dict(xatol=1e-10, fatol=1e-16)).x
    return mins, sads


MINIMA, SADDLES = _stationary_points()
REF_POINT = MINIMA["A"]          # F is anchored to 0 at the global minimum


def target_quantities(Ffun):
    """Scalar targets used for coverage tests, from any F(x,y) callable.

    dF_BA, dF_CA : minimum-to-minimum free-energy differences
    barrier_AS1  : barrier from A to the A-C saddle
    """
    pts = np.array([MINIMA["A"], MINIMA["B"], MINIMA["C"], SADDLES["S1"]])
    F = Ffun(pts[:, 0], pts[:, 1])
    return dict(dF_BA=F[1] - F[0], dF_CA=F[2] - F[0], barrier_AS1=F[3] - F[0])


if __name__ == "__main__":
    print("minima:", {k: np.round(v, 4) for k, v in MINIMA.items()})
    print("saddles:", {k: np.round(v, 4) for k, v in SADDLES.items()})
    tq = target_quantities(lambda x, y: free_energy(x, y)[0])
    print("true targets (kT):", {k: round(float(v), 3) for k, v in tq.items()})
    for k, p in {**SADDLES}.items():
        print(k, "|grad F| =", np.hypot(*free_energy(*p)[1:]))
