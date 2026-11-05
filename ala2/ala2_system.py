"""Alanine dipeptide (ACE-ALA-NME) at 300 K, in vacuum or in explicit water.
Collective variables: backbone dihedrals phi, psi (radians).

  vacuum : Amber prmtop/crd from the openmmtools test-system collection, no cutoff
  water  : Amber14 (ff14SB) + TIP3P, cubic box with 1.2 nm padding, PME (0.9 nm cutoff),
           equilibrated once at 1 bar (NPT) and then simulated at fixed volume (NVT)

Free energies are reported in kT; mean-force labels in kT/rad.
"""
import os
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
T = 300.0
KT_KJ = (u.MOLAR_GAS_CONSTANT_R * T * u.kelvin).value_in_unit(u.kilojoule_per_mole)  # 2.494
PHI_ATOMS = (4, 6, 8, 14)      # C(ACE) - N - CA - C
PSI_ATOMS = (6, 8, 14, 16)     # N - CA - C - N(NME)
DT_FS = 2.0


SOLVENTS = ("vacuum", "water")
WATER_PDB = os.path.join(DATA, "ala2_water_equil.pdb")


def best_platform(threads=None, prefer=None):
    """Fastest available OpenMM platform: CUDA > OpenCL > CPU (or `prefer`)."""
    names = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
    order = [prefer] if prefer else ["CUDA", "OpenCL", "CPU"]
    for n in order:
        if n in names:
            props = {"Precision": "mixed"} if n in ("CUDA", "OpenCL") else ({"Threads": str(threads)} if threads else {})
            return mm.Platform.getPlatformByName(n), props
    return mm.Platform.getPlatformByName("CPU"), {}


def _water_forcefield():
    return app.ForceField("amber14-all.xml", "amber14/tip3p.xml")


def build_water_box(padding_nm=1.2, equil_ps=200.0, seed=1, prefer=None):
    """Solvate, minimise and equilibrate at 300 K / 1 bar; save to WATER_PDB."""
    pdb = app.PDBFile(os.path.join(DATA, "alanine-dipeptide.pdb"))
    ff = _water_forcefield()
    mod = app.Modeller(pdb.topology, pdb.positions)
    mod.addSolvent(ff, model="tip3p", padding=padding_nm * u.nanometer)
    system = ff.createSystem(mod.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=0.9 * u.nanometer, constraints=app.HBonds)
    system.addForce(mm.MonteCarloBarostat(1 * u.bar, T * u.kelvin))
    integ = mm.LangevinMiddleIntegrator(T * u.kelvin, 1 / u.picosecond, DT_FS * u.femtosecond)
    integ.setRandomNumberSeed(seed)
    plat, props = best_platform(prefer=prefer)
    sim = app.Simulation(mod.topology, system, integ, plat, props)
    sim.context.setPositions(mod.positions)
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(T * u.kelvin, seed)
    sim.step(int(equil_ps * 1000 / DT_FS))
    st = sim.context.getState(getPositions=True, enforcePeriodicBox=False)
    top = mod.topology
    top.setPeriodicBoxVectors(st.getPeriodicBoxVectors())
    with open(WATER_PDB, "w") as fh:
        app.PDBFile.writeFile(top, st.getPositions(), fh, keepIds=True)
    print(f"water box: {top.getNumAtoms()} atoms, box {st.getPeriodicBoxVectors()[0][0]} -> {WATER_PDB}", flush=True)


def make_model(solvent="vacuum"):
    """Return dict(topology, positions, system, box) for the chosen environment.
    The system has no restraint or bias; callers add those."""
    if solvent == "vacuum":
        prm, crd = load()
        return dict(topology=prm.topology, positions=crd.positions, system=base_system(prm), box=None)
    if solvent == "water":
        if not os.path.exists(WATER_PDB):
            build_water_box()
        pdb = app.PDBFile(WATER_PDB)
        system = _water_forcefield().createSystem(pdb.topology, nonbondedMethod=app.PME,
                                                  nonbondedCutoff=0.9 * u.nanometer, constraints=app.HBonds)
        return dict(topology=pdb.topology, positions=pdb.positions, system=system,
                    box=pdb.topology.getPeriodicBoxVectors())
    raise ValueError(solvent)


def make_simulation(model, integrator, solvent="vacuum", threads=1, prefer=None):
    """Simulation on CPU (vacuum, as in the published runs) or the fastest platform (water)."""
    if solvent == "vacuum":
        plat, props = mm.Platform.getPlatformByName("CPU"), {"Threads": str(threads)}
    else:
        plat, props = best_platform(threads=threads, prefer=prefer)
    sim = app.Simulation(model["topology"], model["system"], integrator, plat, props)
    if model["box"] is not None:
        sim.context.setPeriodicBoxVectors(*model["box"])
    sim.context.setPositions(model["positions"])
    return sim


def add_cv_readout(system):
    """Zero-energy force whose collective variables are (phi, psi): reading them with
    getCollectiveVariableValues avoids copying all positions off the GPU."""
    f = mm.CustomCVForce("0*phi + 0*psi")
    for name, atoms in (("phi", PHI_ATOMS), ("psi", PSI_ATOMS)):
        f.addCollectiveVariable(name, torsion_cv(atoms))
    system.addForce(f)
    return f


def load():
    prm = app.AmberPrmtopFile(os.path.join(DATA, "alanine-dipeptide.prmtop"))
    crd = app.AmberInpcrdFile(os.path.join(DATA, "alanine-dipeptide.crd"))
    return prm, crd


def base_system(prm):
    return prm.createSystem(nonbondedMethod=app.NoCutoff, constraints=app.HBonds)


def torsion_cv(atoms):
    f = mm.CustomTorsionForce("theta")
    f.addTorsion(*atoms)
    return f


def add_restraint(system, centre, k_kj):
    """Harmonic restraint on (phi, psi) around centre (radians), stiffness k_kj
    in kJ/mol/rad^2, periodic-aware. Returns the force (parameters phi0, psi0 can
    be changed per window)."""
    expr = ("0.5*k*d^2; d = min(dd, 2*pi - dd); dd = abs(theta - theta0); pi = 3.141592653589793")
    forces = []
    for atoms, c0, name in ((PHI_ATOMS, centre[0], "phi0"), (PSI_ATOMS, centre[1], "psi0")):
        f = mm.CustomTorsionForce(expr.replace("theta0", name))
        f.addGlobalParameter("k", k_kj)
        f.addGlobalParameter(name, c0)
        f.addTorsion(*atoms)
        system.addForce(f)
        forces.append(f)
    return forces


def dihedrals(pos_nm):
    """phi, psi from an (N,3) position array in nm."""
    def dih(p0, p1, p2, p3):
        b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
        b1n = b1 / np.linalg.norm(b1)
        v = b0 - np.dot(b0, b1n) * b1n
        w = b2 - np.dot(b2, b1n) * b1n
        x = np.dot(v, w)
        y = np.dot(np.cross(b1n, v), w)
        return np.arctan2(y, x)
    p = pos_nm
    return np.array([dih(*p[list(PHI_ATOMS)]), dih(*p[list(PSI_ATOMS)])])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
