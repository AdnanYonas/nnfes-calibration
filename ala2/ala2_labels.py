"""Restrained-MD mean-force labels for alanine dipeptide in (phi, psi).

Each window restrains (phi, psi) harmonically (k = 1000 kJ/mol/rad^2) around a
centre c. Umbrella-integration label:
    grad F(<s>) ~= k (c - <s>)      (periodic differences; converted to kT/rad)
placed at the circular mean <s>, with a 5-block standard error, exactly like
the toy systems. Windows are visited along a serpentine path so each starts
from its neighbour's final structure; work is split over processes.
"""
import os, time
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u
from multiprocessing import Pool
from ala2_system import (load, base_system, add_restraint, dihedrals, wrap,
                         T, KT_KJ, DT_FS)

K_RES = 1000.0                       # kJ/mol/rad^2
SAMPLE_EVERY = 10                    # steps between recorded (phi, psi)


def serpentine(centres, spacing):
    i = np.round((centres[:, 0] + np.pi) / spacing).astype(int)
    j = np.round((centres[:, 1] + np.pi) / spacing).astype(int)
    key = np.where(i % 2 == 0, j, -j)
    return np.lexsort((key, i))


def _run_chunk(args):
    centres, n_prod, n_eq_first, n_eq, seed, threads = args
    prm, crd = load()
    system = base_system(prm)
    add_restraint(system, centres[0], K_RES)
    integ = mm.LangevinMiddleIntegrator(T * u.kelvin, 1 / u.picosecond, DT_FS * u.femtosecond)
    integ.setRandomNumberSeed(int(seed))
    sim = app.Simulation(prm.topology, system, integ, mm.Platform.getPlatformByName("CPU"),
                         {"Threads": str(threads)})
    sim.context.setPositions(crd.positions)
    sim.context.setParameter("phi0", centres[0][0]); sim.context.setParameter("psi0", centres[0][1])
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(T * u.kelvin, int(seed))
    out = []
    cur = np.array(centres[0], float)
    for w, c in enumerate(centres):
        # move the restraint centre gradually (<= 0.15 rad per 0.5 ps) to avoid huge forces
        delta = wrap(np.asarray(c) - cur)
        n_sub = int(np.ceil(np.abs(delta).max() / 0.15))
        for k in range(1, n_sub + 1):
            p = wrap(cur + delta * k / n_sub)
            sim.context.setParameter("phi0", float(p[0])); sim.context.setParameter("psi0", float(p[1]))
            sim.step(250)
        cur = np.array(c, float)
        sim.context.setParameter("phi0", float(c[0]))
        sim.context.setParameter("psi0", float(c[1]))
        sim.step(n_eq_first if w == 0 else n_eq)
        n_rec = n_prod // SAMPLE_EVERY
        traj = np.empty((n_rec, 2))
        for r in range(n_rec):
            sim.step(SAMPLE_EVERY)
            pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)
            traj[r] = dihedrals(np.asarray(pos))
        d = wrap(traj - c)                                  # displacement from centre
        f = -K_RES * d / KT_KJ                              # k (c - s) in kT/rad, per frame
        nb = 5
        L = n_rec // nb
        bm = f[:L * nb].reshape(nb, L, 2).mean(1)
        s_mean = wrap(c + d.mean(0))
        out.append((s_mean, f.mean(0), bm.std(0, ddof=1) / np.sqrt(nb)))
    return out


def sample_labels(centres, prod_ps, seed=0, n_proc=None, eq_first_ps=20.0, eq_ps=4.0, spacing=None):
    """Run all windows. Returns dict(s, g, se, centres) in the input order."""
    n_proc = n_proc or max(1, (os.cpu_count() or 2))
    spacing = spacing or 2 * np.pi / 18
    order = serpentine(centres, spacing)
    chunks = np.array_split(order, n_proc)
    steps = lambda ps: int(round(ps * 1000 / DT_FS))
    jobs = [(centres[ch], steps(prod_ps), steps(eq_first_ps), steps(eq_ps), seed * 1000 + k + 1, 1)
            for k, ch in enumerate(chunks) if len(ch)]
    if n_proc == 1:
        res = [_run_chunk(j) for j in jobs]
    else:
        with Pool(len(jobs)) as pool:
            res = pool.map(_run_chunk, jobs)
    s = np.empty((len(centres), 2)); g = np.empty_like(s); se = np.empty_like(s)
    for ch, r in zip([c for c in chunks if len(c)], res):
        for idx, (sm, gm, e) in zip(ch, r):
            s[idx], g[idx], se[idx] = sm, gm, e
    return dict(s=s, g=g, se=se, centres=centres)
