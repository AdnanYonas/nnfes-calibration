"""Reference free-energy surface F(phi, psi) for alanine dipeptide by
well-tempered metadynamics (OpenMM). Run several independent copies; their
spread estimates the precision of the reference.

Usage:
  python reference_metad.py --seed 1 --ns 25 --out ref_run1.npz                  (vacuum, as published)
  python reference_metad.py --solvent water --seed 1 --ns 50 --resume            (explicit water)
Saves the FES (kT, minimum at 0) on a 72x72 periodic grid every 5 ns.
With --resume (water runs), a checkpoint and the deposited bias are saved after every
5 ns block, so an interrupted run continues from the last finished block.
"""
import argparse, glob, os, time
import numpy as np
import openmm as mm
import openmm.unit as u
from openmm.app.metadynamics import Metadynamics, BiasVariable
from ala2_system import (HERE, make_model, make_simulation, torsion_cv, PHI_ATOMS, PSI_ATOMS,
                         T, KT_KJ, DT_FS)

BLOCK_NS = 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solvent", default="vacuum", choices=["vacuum", "water"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ns", type=float, default=30.0)
    ap.add_argument("--out", default=None, help="default: ref_run<seed>.npz (vacuum) / ref_water_run<seed>.npz")
    ap.add_argument("--threads", default="1")
    ap.add_argument("--platform", default=None, help="force CUDA / OpenCL / CPU (water only)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--block-ns", type=float, default=BLOCK_NS, help="save interval (testing only)")
    a = ap.parse_args()
    tag = "" if a.solvent == "vacuum" else "water_"
    out = a.out or os.path.join(HERE, f"ref_{tag}run{a.seed}.npz")
    model = make_model(a.solvent)
    cvs = [BiasVariable(torsion_cv(at), -np.pi, np.pi, 0.35, periodic=True, gridWidth=72)
           for at in (PHI_ATOMS, PSI_ATOMS)]
    blk = a.block_ns
    steps_per_block = int(blk * 1e6 / DT_FS)
    kw = {}
    bias_dir = chk = None
    if a.solvent != "vacuum":                       # bias + checkpoint so long runs can resume
        bias_dir = os.path.splitext(out)[0] + "_bias"
        chk = os.path.splitext(out)[0] + ".chk"
        os.makedirs(bias_dir, exist_ok=True)
        if not a.resume:
            for f in glob.glob(os.path.join(bias_dir, "*")):
                os.remove(f)
        kw = dict(saveFrequency=steps_per_block, biasDir=bias_dir)
    meta = Metadynamics(model["system"], cvs, T * u.kelvin, biasFactor=6.0,
                        height=1.2 * u.kilojoule_per_mole, frequency=500, **kw)
    integ = mm.LangevinMiddleIntegrator(T * u.kelvin, 1 / u.picosecond, DT_FS * u.femtosecond)
    integ.setRandomNumberSeed(a.seed)
    sim = make_simulation(model, integ, solvent=a.solvent, threads=int(a.threads), prefer=a.platform)
    print(f"{a.solvent}: {model['topology'].getNumAtoms()} atoms, platform "
          f"{sim.context.getPlatform().getName()}, output {out}", flush=True)
    snaps = []
    if a.resume and chk and os.path.exists(chk) and os.path.exists(out):
        snaps = list(np.load(out)["fes"])
        sim.loadCheckpoint(chk)
        print(f"resuming after {blk * len(snaps):g} ns", flush=True)
    else:
        sim.minimizeEnergy()
        sim.context.setVelocitiesToTemperature(T * u.kelvin, a.seed)
    n_blocks = int(round(a.ns / blk))
    t0, start = time.time(), len(snaps)
    for c in range(start, n_blocks):
        meta.step(sim, steps_per_block)
        F = np.array(meta.getFreeEnergy().value_in_unit(u.kilojoule_per_mole)) / KT_KJ
        snaps.append(F - F.min())
        np.savez(out, fes=np.array(snaps), ns=blk * np.arange(1, len(snaps) + 1))
        if chk:
            sim.saveCheckpoint(chk)
        el = time.time() - t0
        rate = blk * (c + 1 - start) / el * 86400
        print(f"[{el:6.0f}s] seed {a.seed}: {blk * (c + 1):g} of {blk * n_blocks:g} ns done "
              f"({rate:.0f} ns/day)", flush=True)


if __name__ == "__main__":
    main()
