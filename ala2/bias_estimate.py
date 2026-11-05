"""Leading-order umbrella-integration estimator bias (manuscript eq 4) from the metadynamics reference:
b_i = 1/2 sum_jk d_ijk F Sigma_jk, Sigma = (kappa I + Hess F)^-1, RMS per component over F_ref < 15 kT.
Usage: python bias_estimate.py      (prints vacuum and water)"""
import numpy as np
from ala2_reference import Reference, GRID

KAPPA = 401.0  # kT / rad^2


def rms_bias(solvent, f_cut=15.0):
    R = Reference(solvent=solvent); F = R.F; h = GRID[1] - GRID[0]
    D = lambda A, ax: (np.roll(A, -1, ax) - np.roll(A, 1, ax)) / (2 * h)
    g = [D(F, i) for i in (0, 1)]
    H = [[D(g[i], j) for j in (0, 1)] for i in (0, 1)]
    S = np.linalg.inv(KAPPA * np.eye(2) + np.array(H).transpose(2, 3, 0, 1))
    b = np.zeros((2,) + F.shape)
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                b[i] += 0.5 * D(H[i][j], k) * S[..., j, k]
    m = F < f_cut
    return np.sqrt(np.mean((b[0][m] ** 2 + b[1][m] ** 2) / 2))


if __name__ == "__main__":
    for s in ("vacuum", "water"):
        print(f"{s}: RMS leading-order bias {rms_bias(s):.3f} kT/rad per component")
