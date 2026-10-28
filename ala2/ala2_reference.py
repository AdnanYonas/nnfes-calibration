"""Reference ("truth") F(phi, psi) from the independent metadynamics runs.

OpenMM stores the bias grid with variables reversed ([psi, phi]) and includes
both periodic end points (linspace(0, 1, 72)), so we transpose and drop the
duplicated last point -> 71 x 71 grid with spacing 2*pi/71.
The reference mean over runs is used as truth; its precision is the standard
error over runs (per point, for F and for grad F).
"""
import glob, os
import numpy as np
from scipy.interpolate import RectBivariateSpline
from ala2_system import HERE

N = 71
GRID = -np.pi + 2 * np.pi * np.arange(N) / N


def _periodic_spline(F):
    pad = 8
    Fp = np.pad(F, pad, mode="wrap")
    g = -np.pi + 2 * np.pi * np.arange(-pad, N + pad) / N
    return RectBivariateSpline(g, g, Fp, kx=3, ky=3, s=0)


class Reference:
    def __init__(self, pattern=os.path.join(HERE, "ref_run*.npz")):
        runs = []
        for f in sorted(glob.glob(pattern)):
            F = np.load(f)["fes"][-1].T[:N, :N]          # -> [phi, psi]
            runs.append(F - F.min())
        self.runs = np.array(runs)
        self.n_runs = len(runs)
        # align runs by their mean over the low-F region before averaging
        m = self.runs.mean(0)
        low = m < 8.0
        for r in self.runs:
            r -= (r[low] - m[low]).mean()
        self.F = self.runs.mean(0)
        self.F -= self.F.min()
        self._spl = _periodic_spline(self.F)
        self._run_spl = [_periodic_spline(r) for r in self.runs]

    def _eval(self, spl, p, dx=0, dy=0):
        p = np.atleast_2d(p)
        q = (p + np.pi) % (2 * np.pi) - np.pi
        return spl.ev(q[:, 0], q[:, 1], dx=dx, dy=dy)

    def free_energy(self, p):
        """Reference F (kT) and grad F (kT/rad) at points p (N,2)."""
        F = self._eval(self._spl, p)
        G = np.stack([self._eval(self._spl, p, dx=1), self._eval(self._spl, p, dy=1)], 1)
        return F, G

    def precision(self, p):
        """Standard error of the reference F (relative to the anchor-free mean) and grad F."""
        Fr = np.array([self._eval(s, p) for s in self._run_spl])
        Gr = np.array([np.stack([self._eval(s, p, dx=1), self._eval(s, p, dy=1)], 1) for s in self._run_spl])
        sF = Fr.std(0, ddof=1) / np.sqrt(self.n_runs)
        sG = Gr.std(0, ddof=1) / np.sqrt(self.n_runs)
        return sF, sG

    def minima(self, f_max=10.0):
        from scipy.ndimage import minimum_filter
        m = (self.F == minimum_filter(self.F, size=7, mode="wrap")) & (self.F < f_max)
        return [(GRID[i], GRID[j], self.F[i, j]) for i, j in zip(*np.nonzero(m))]

    def minimax_saddle(self, a, b):
        """Lowest-barrier connection between grid points a and b (indices):
        the grid point at which the two basins first join (union-find over
        points sorted by F)."""
        order = np.argsort(self.F, axis=None)
        parent = -np.ones(N * N, int)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        ia, ib = np.ravel_multi_index(a, (N, N)), np.ravel_multi_index(b, (N, N))
        for idx in order:
            parent[idx] = idx
            i, j = divmod(idx, N)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = ((i + di) % N) * N + (j + dj) % N
                if parent[nb] >= 0:
                    parent[find(nb)] = find(idx)
            if parent[ia] >= 0 and parent[ib] >= 0 and find(ia) == find(ib):
                return np.array([GRID[i], GRID[j]]), self.F[i, j]
        return None


if __name__ == "__main__":
    R = Reference()
    print("runs:", R.n_runs)
    for phi, psi, f in R.minima():
        print(f"minimum phi {np.degrees(phi):6.1f} psi {np.degrees(psi):6.1f}  F {f:.2f} kT")
    lowP = np.array([[GRID[i], GRID[j]] for i in range(N) for j in range(N) if R.F[i, j] < 8])
    sF, sG = R.precision(lowP)
    print(f"reference precision on F<8 region: F {np.sqrt((sF**2).mean()):.3f} kT, grad {np.sqrt((sG**2).mean()):.3f} kT/rad")
    print("fraction of torus with F < 12 kT:", round((R.F < 12).mean(), 3))
