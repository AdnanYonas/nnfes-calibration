"""Alanine dipeptide (ACE-ALA-NME, Amber ff96-style prmtop from openmmtools) in
vacuum at 300 K. Collective variables: backbone dihedrals phi, psi (radians).

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
