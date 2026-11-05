"""Quick check of the OpenMM platform and speed for solvated alanine dipeptide.
Usage: python benchmark_platform.py [--platform CUDA|OpenCL|CPU]"""
import argparse, time
import openmm as mm, openmm.unit as u
from ala2_system import make_model, make_simulation, T, DT_FS

ap = argparse.ArgumentParser(); ap.add_argument("--platform", default=None); a = ap.parse_args()
names = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
print("OpenMM", mm.__version__, "platforms available:", names)
model = make_model("water")
integ = mm.LangevinMiddleIntegrator(T * u.kelvin, 1 / u.picosecond, DT_FS * u.femtosecond)
sim = make_simulation(model, integ, solvent="water", threads=0, prefer=a.platform)
print("using:", sim.context.getPlatform().getName())
sim.minimizeEnergy(); sim.step(1000)
n = 10000; t = time.time(); sim.step(n); dt = time.time() - t
print(f"speed: {n * DT_FS * 1e-6 / dt * 86400:.0f} ns/day")
