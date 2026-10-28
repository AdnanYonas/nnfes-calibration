"""Free-energy models trained on mean-force labels, with uncertainty.

Neural ensembles: a scalar MLP F_theta(s) whose input-gradient is fitted to
the mean-force labels (gradient matching guarantees an integrable field).
All M members are trained simultaneously as one batched network.

  ens_init   - members differ only in random initialisation (RiD-style
               "uncertainty indicator", Zhang, Wang & E 2018)
  ens_boot   - bootstrap resampling of labels + random init
  ens_noise  - randomise-then-optimise: each member sees labels perturbed by
               N(0, se^2) and an inverse-variance weighted loss

GP baseline: Gaussian process on F with gradient observations and the
labels' own (heteroscedastic) noise, hyperparameters by marginal likelihood
(Stecher, Bernstein & Csanyi 2014 style).

Every model returns F anchored at a reference point (F(ref) = 0), because a
free energy is only defined up to a constant.
"""
import numpy as np
import torch
from scipy.optimize import minimize
from system import DOMAIN

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SE_FLOOR = 0.05   # kT/unit; floor on label errors so near-zero se cannot dominate

_CEN = torch.tensor([(DOMAIN["xmin"] + DOMAIN["xmax"]) / 2,
                     (DOMAIN["ymin"] + DOMAIN["ymax"]) / 2])
_SCL = torch.tensor([(DOMAIN["xmax"] - DOMAIN["xmin"]) / 2,
                     (DOMAIN["ymax"] - DOMAIN["ymin"]) / 2])


class EnsembleMLP(torch.nn.Module):
    """M independent tanh MLPs R^2 -> R evaluated in one batched pass."""

    def __init__(self, M, width=64, depth=2, seed=0, cen=None, scl=None, periodic_dim=None):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.periodic_dim = periodic_dim      # if set: inputs are angles -> (cos, sin) features
        if periodic_dim is None:
            self.cen = _CEN if cen is None else torch.as_tensor(cen, dtype=torch.float64)
            self.scl = _SCL if scl is None else torch.as_tensor(scl, dtype=torch.float64)
            d_in = len(self.cen)
        else:
            d_in = 2 * periodic_dim
        dims = [d_in] + [width] * depth + [1]
        self.W = torch.nn.ParameterList()
        self.b = torch.nn.ParameterList()
        for i, (a, c) in enumerate(zip(dims[:-1], dims[1:])):
            std = np.sqrt(1.0 / a)                      # LeCun init for tanh
            self.W.append(torch.nn.Parameter(torch.randn(M, a, c, generator=g) * std))
            self.b.append(torch.nn.Parameter(torch.randn(M, 1, c, generator=g) * 0.1))
        self.M = M

    def forward(self, s):                               # s: (M, N, d) physical units
        if self.periodic_dim is None:
            h = (s - self.cen.to(s)) / self.scl.to(s)
        else:
            h = torch.cat([torch.cos(s), torch.sin(s)], -1)
        for i in range(len(self.W)):
            h = torch.baddbmm(self.b[i], h, self.W[i])
            if i < len(self.W) - 1:
                h = torch.tanh(h)
        return h.squeeze(-1) * 5.0                      # output scale ~ kT range

    def grad(self, s, create_graph=False):
        s = s.detach().requires_grad_(True)
        F = self.forward(s)
        g, = torch.autograd.grad(F.sum(), s, create_graph=create_graph)
        return F, g


def train_ensemble(s, g, se, kind, M=8, epochs=4000, lr=3e-3, wd=1e-6, seed=0,
                   cen=None, scl=None, width=64, batch=None, lbfgs_steps=0, periodic=False):
    """Fit an ensemble to labels. s,g,se: (N,d) numpy. Returns trained model.
    cen/scl: input centring/scaling (default: the 2D system's domain).
    batch: if set and N > batch, each step uses a random subset of `batch`
    labels (same optimiser budget regardless of N); default = full batch.
    lbfgs_steps: optional full-batch L-BFGS polish after Adam (same loss plus
    the equivalent L2 term); fixes slow, long-wavelength errors in F that a
    gradient-matching loss penalises only weakly. Default 0 = off."""
    rng = np.random.default_rng(seed)
    N, D = g.shape
    se_f = np.sqrt(se**2 + SE_FLOOR**2)
    G = np.repeat(g[None], M, 0)                         # (M,N,2) targets
    Wt = np.ones((M, N, D))                              # per-component weights
    if kind == "ens_boot":
        counts = np.stack([np.bincount(rng.integers(0, N, N), minlength=N) for _ in range(M)])
        Wt = np.repeat(counts[:, :, None], D, 2).astype(float)
    elif kind == "ens_noise":
        G = G + rng.standard_normal(G.shape) * se_f[None]
        Wt = np.repeat((1.0 / se_f**2)[None], M, 0)
    elif kind != "ens_init":
        raise ValueError(kind)
    Wt = Wt / Wt.sum(axis=(1, 2), keepdims=True)
    S = torch.tensor(np.repeat(s[None], M, 0), device=DEVICE)
    G = torch.tensor(G, device=DEVICE)
    Wt = torch.tensor(Wt, device=DEVICE)
    net = EnsembleMLP(M, width=width, seed=seed, cen=cen, scl=scl,
                      periodic_dim=(g.shape[1] if periodic else None)).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr * 0.01)
    brng = torch.Generator().manual_seed(seed + 17)
    for _ in range(epochs):
        opt.zero_grad()
        if batch is not None and N > batch:
            idx = torch.randperm(N, generator=brng)[:batch].to(DEVICE)
            _, pg = net.grad(S[:, idx], create_graph=True)
            w = Wt[:, idx]
            loss = (w / w.sum(dim=(1, 2), keepdim=True) * (pg - G[:, idx]) ** 2).sum()
        else:
            _, pg = net.grad(S, create_graph=True)
            loss = (Wt * (pg - G) ** 2).sum()
        loss.backward()
        opt.step()
        sched.step()
    if lbfgs_steps > 0:
        lb = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=lbfgs_steps,
                               history_size=50, line_search_fn="strong_wolfe",
                               tolerance_grad=1e-10, tolerance_change=1e-14)

        def closure():
            lb.zero_grad()
            _, pg = net.grad(S, create_graph=True)
            l2 = sum((p ** 2).sum() for p in net.parameters())
            loss = (Wt * (pg - G) ** 2).sum() + 0.5 * wd * l2
            loss.backward()
            return loss
        lb.step(closure)
    return net


def ensemble_predict(net, pts, ref):
    """Anchored F and gradient per member at pts (P,2). Returns numpy arrays
    F_m (M,P) with F_m(ref)=0, and grad_m (M,P,2)."""
    M = net.M
    P = np.vstack([pts, ref[None]])
    S = torch.tensor(np.repeat(P[None], M, 0), device=DEVICE)
    F, g = net.grad(S)
    F = F.detach().cpu().numpy()
    g = g.detach().cpu().numpy()
    return F[:, :-1] - F[:, -1:], g[:, :-1]


# ---------------------------------------------------------------- GP ------
class GradGP:
    """GP prior F ~ GP(0, sf^2 exp(-r^2 / 2 l^2)) observed through noisy
    gradients y_n = grad F(s_n) + eps_n, eps_n ~ N(0, diag(se_n^2))."""

    def __init__(self, s, g, se):
        self.s, self.y = s, g.reshape(-1)                 # y ordered (n, component)
        self.noise = (se**2 + SE_FLOOR**2).reshape(-1)
        res = minimize(self._nlml, x0=np.log([5.0, 0.2]), method="L-BFGS-B",
                       bounds=[(np.log(0.1), np.log(100.0)), (np.log(0.02), np.log(2.0))])
        self.sf, self.l = np.exp(res.x)
        self._factor()

    @staticmethod
    def _Kgg(a, b, sf, l):
        d = a[:, None, :] - b[None, :, :]                 # (A,B,2)
        e = sf**2 * np.exp(-(d**2).sum(-1) / (2 * l**2))
        I = np.eye(2)
        K = e[:, :, None, None] * (I / l**2 - d[:, :, :, None] * d[:, :, None, :] / l**4)
        return K.transpose(0, 2, 1, 3).reshape(2 * len(a), 2 * len(b))

    @staticmethod
    def _Kfg(a, b, sf, l):
        d = a[:, None, :] - b[None, :, :]
        e = sf**2 * np.exp(-(d**2).sum(-1) / (2 * l**2))
        return (e[:, :, None] * d / l**2).reshape(len(a), 2 * len(b))

    def _nlml(self, p):
        sf, l = np.exp(p)
        K = self._Kgg(self.s, self.s, sf, l) + np.diag(self.noise)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        a = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
        return 0.5 * self.y @ a + np.log(np.diag(L)).sum()

    def _factor(self):
        K = self._Kgg(self.s, self.s, self.sf, self.l) + np.diag(self.noise)
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))

    def predict(self, pts, ref):
        """Anchored mean/std of F at pts, and mean/std of gradient."""
        sf, l = self.sf, self.l
        kp = self._Kfg(pts, self.s, sf, l)
        kr = self._Kfg(ref[None], self.s, sf, l)
        v = kp - kr                                      # cov(F(p)-F(ref), y)
        mF = v @ self.alpha
        d2 = ((pts - ref[None]) ** 2).sum(-1)
        prior = 2 * sf**2 * (1 - np.exp(-d2 / (2 * l**2)))
        W = np.linalg.solve(self.L, v.T)
        sF = np.sqrt(np.maximum(prior - (W**2).sum(0), 1e-12))
        Kgp = self._Kgg(pts, self.s, sf, l)               # (2P, 2N)
        mg = (Kgp @ self.alpha).reshape(-1, 2)
        Wg = np.linalg.solve(self.L, Kgp.T)
        sg = np.sqrt(np.maximum(sf**2 / l**2 - (Wg**2).sum(0), 1e-12)).reshape(-1, 2)
        return mF, sF, mg, sg


def label_scale_factor(mu, sm, y, se):
    """Practitioner-side recalibration: find alpha so that held-out labels
    y satisfy std[(y - mu) / sqrt(alpha^2 sm^2 + se^2)] = 1. No ground truth
    used. Returns alpha (0 if labels alone already explain the residuals)."""
    r2 = ((y - mu) ** 2).ravel()
    sm2, se2 = (sm**2).ravel(), (se**2 + SE_FLOOR**2).ravel()
    f = lambda a: np.mean(r2 / (a**2 * sm2 + se2)) - 1.0
    if f(0.0) <= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    while f(hi) > 0 and hi < 1e4:
        hi *= 2
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


class GradGPARD:
    """N-dimensional gradient-observation GP with one length scale per
    dimension (ARD). Hyperparameters are fitted by marginal likelihood on a
    random subset of at most n_fit labels (full exact posterior afterwards),
    because the gradient Gram matrix is (N*d) x (N*d)."""

    def __init__(self, s, g, se, n_fit=400, seed=0):
        import time
        self.s, self.y = s, g.reshape(-1)
        self.N, self.d = g.shape
        self.noise = (se**2 + SE_FLOOR**2).reshape(-1)
        rng = np.random.default_rng(seed)
        sub = rng.permutation(self.N)[:min(n_fit, self.N)]
        ys = g[sub].reshape(-1)
        ns = (se[sub]**2 + SE_FLOOR**2).reshape(-1)
        x0 = np.log(np.r_[5.0, 0.3 * np.ones(self.d)])
        bnds = [(np.log(0.1), np.log(100.0))] + [(np.log(0.02), np.log(3.0))] * self.d
        t0 = time.time()
        res = minimize(lambda p: self._nlml(p, s[sub], ys, ns), x0=x0,
                       method="L-BFGS-B", bounds=bnds)
        self.t_fit = time.time() - t0
        self.sf, self.l = np.exp(res.x[0]), np.exp(res.x[1:])
        t0 = time.time()
        K = self._Kgg(s, s, self.sf, self.l) + np.diag(self.noise)
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))
        self.t_factor = time.time() - t0

    @staticmethod
    def _Kgg(a, b, sf, l, block=256):
        """Gradient-gradient Gram matrix, built in row blocks to bound memory."""
        D = a.shape[1]
        out = np.empty((D * len(a), D * len(b)))
        L2 = np.diag(1 / l**2)[None, None]
        for i in range(0, len(a), block):
            ab = a[i:i + block]
            diff = ab[:, None, :] - b[None, :, :]
            d = diff / l**2
            e = sf**2 * np.exp(-0.5 * ((diff / l) ** 2).sum(-1))
            K = e[:, :, None, None] * (L2 - d[:, :, :, None] * d[:, :, None, :])
            out[D * i:D * (i + len(ab))] = K.transpose(0, 2, 1, 3).reshape(D * len(ab), D * len(b))
        return out

    @staticmethod
    def _Kfg(a, b, sf, l):
        diff = a[:, None, :] - b[None, :, :]
        e = sf**2 * np.exp(-0.5 * ((diff / l) ** 2).sum(-1))
        return (e[:, :, None] * diff / l**2).reshape(len(a), a.shape[1] * len(b))

    def _nlml(self, p, s, y, noise):
        sf, l = np.exp(p[0]), np.exp(p[1:])
        K = self._Kgg(s, s, sf, l) + np.diag(noise)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        a = np.linalg.solve(L.T, np.linalg.solve(L, y))
        return 0.5 * y @ a + np.log(np.diag(L)).sum()

    def predict(self, pts, ref, chunk=200):
        sf, l, D = self.sf, self.l, self.d
        mF, sF, mg, sg = [], [], [], []
        kr = self._Kfg(ref[None], self.s, sf, l)
        for i in range(0, len(pts), chunk):
            p = pts[i:i + chunk]
            v = self._Kfg(p, self.s, sf, l) - kr
            mF.append(v @ self.alpha)
            r2 = (((p - ref[None]) / l) ** 2).sum(-1)
            prior = 2 * sf**2 * (1 - np.exp(-0.5 * r2))
            W = np.linalg.solve(self.L, v.T)
            sF.append(np.sqrt(np.maximum(prior - (W**2).sum(0), 1e-12)))
            Kgp = self._Kgg(p, self.s, sf, l)
            mg.append((Kgp @ self.alpha).reshape(-1, D))
            Wg = np.linalg.solve(self.L, Kgp.T)
            pv = np.tile(sf**2 / l**2, len(p))
            sg.append(np.sqrt(np.maximum(pv - (Wg**2).sum(0), 1e-12)).reshape(-1, D))
        return np.concatenate(mF), np.concatenate(sF), np.vstack(mg), np.vstack(sg)


class GradGPPeriodic(GradGPARD):
    """Gradient-observation GP on a torus (angles in radians) with the periodic
    kernel k = sf^2 exp(-sum_d 2 sin^2((a_d - b_d)/2) / l_d^2).
    Derivatives: dk/db_j = k sin(a_j - b_j)/l_j^2 ;
    d2k/da_i db_j = k [delta_ij cos(D_i)/l_i^2 - sin(D_i) sin(D_j)/(l_i^2 l_j^2)]."""

    @staticmethod
    def _k(diff, sf, l):
        return sf**2 * np.exp(-(2 * np.sin(diff / 2) ** 2 / l**2).sum(-1))

    @staticmethod
    def _Kgg(a, b, sf, l, block=256):
        D = a.shape[1]
        out = np.empty((D * len(a), D * len(b)))
        for i in range(0, len(a), block):
            ab = a[i:i + block]
            diff = ab[:, None, :] - b[None, :, :]
            k = GradGPPeriodic._k(diff, sf, l)
            s = np.sin(diff) / l**2
            c = np.cos(diff) / l**2
            K = k[:, :, None, None] * (np.eye(D)[None, None] * c[:, :, :, None] - s[:, :, :, None] * s[:, :, None, :])
            out[D * i:D * (i + len(ab))] = K.transpose(0, 2, 1, 3).reshape(D * len(ab), D * len(b))
        return out

    @staticmethod
    def _Kfg(a, b, sf, l):
        diff = a[:, None, :] - b[None, :, :]
        k = GradGPPeriodic._k(diff, sf, l)
        return (k[:, :, None] * np.sin(diff) / l**2).reshape(len(a), a.shape[1] * len(b))

    def predict(self, pts, ref, chunk=200):
        sf, l, D = self.sf, self.l, self.d
        mF, sF, mg, sg = [], [], [], []
        kr = self._Kfg(ref[None], self.s, sf, l)
        for i in range(0, len(pts), chunk):
            p = pts[i:i + chunk]
            v = self._Kfg(p, self.s, sf, l) - kr
            mF.append(v @ self.alpha)
            prior = 2 * (sf**2 - self._k(p - ref[None], sf, l))
            W = np.linalg.solve(self.L, v.T)
            sF.append(np.sqrt(np.maximum(prior - (W**2).sum(0), 1e-12)))
            Kgp = self._Kgg(p, self.s, sf, l)
            mg.append((Kgp @ self.alpha).reshape(-1, D))
            Wg = np.linalg.solve(self.L, Kgp.T)
            pv = np.tile(sf**2 / l**2, len(p))
            sg.append(np.sqrt(np.maximum(pv - (Wg**2).sum(0), 1e-12)).reshape(-1, D))
        return np.concatenate(mF), np.concatenate(sF), np.vstack(mg), np.vstack(sg)
