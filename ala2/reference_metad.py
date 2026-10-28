"""Reference free-energy surface F(phi, psi) for alanine dipeptide by
well-tempered metadynamics (OpenMM). Run several independent copies; their
spread estimates the precision of the reference.

Usage: python reference_metad.py --seed 1 --ns 30 --out ref_run1.npz
Saves the FES (kT, minimum at 0) on a 72x72 periodic grid every 5 ns.
"""
import argparse, time
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u
from openmm.app.metadynamics import Metadynamics, BiasVariable
from ala2_system import load, base_system, torsion_cv, PHI_ATOMS, PSI_ATOMS, T, KT_KJ, DT_FS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ns", type=float, default=30.0)
    ap.add_argument("--out", default="ref_run1.npz")
    ap.add_argument("--threads", default="1")
    a = ap.parse_args()
    prm, crd = load()
    system = base_system(prm)
    cvs = [BiasVariable(torsion_cv(at), -np.pi, np.pi, 0.35, periodic=True, gridWidth=72)
           for at in (PHI_ATOMS, PSI_ATOMS)]
    meta = Metadynamics(system, cvs, T * u.kelvin, biasFactor=6.0,
                        height=1.2 * u.kilojoule_per_mole, frequency=500)
    integ = mm.LangevinMiddleIntegrator(T * u.kelvin, 1 / u.picosecond, DT_FS * u.femtosecond)
    integ.setRandomNumberSeed(a.seed)
    sim = app.Simulation(prm.topology, system, integ, mm.Platform.getPlatformByName("CPU"),
                         {"Threads": a.threads})
    sim.context.setPositions(crd.positions)
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(T * u.kelvin, a.seed)
    steps_per_chunk = int(5.0e6 / DT_FS)              # 5 ns
    n_chunks = int(round(a.ns / 5.0))
    snaps = []
    t0 = time.time()
    for c in range(n_chunks):
        meta.step(sim, steps_per_chunk)
        F = np.array(meta.getFreeEnergy().value_in_unit(u.kilojoule_per_mole)) / KT_KJ
        F = F - F.min()
        snaps.append(F)
        np.savez(a.out, fes=np.array(snaps), ns=5.0 * np.arange(1, len(snaps) + 1))
        print(f"[{time.time()-t0:6.0f}s] seed {a.seed}: {5*(c+1)} ns done", flush=True)


if __name__ == "__main__":
    main()
