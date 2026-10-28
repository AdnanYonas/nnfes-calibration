"""Pool the alanine-dipeptide replicas from the two machines into one summary.

Usage: python pool_replicas.py [dirA] [dirB]   (default: results_ala2 results_ala2_user)
Writes ala2_pooled_results.json (10 replicas per regime), used by the paper figures/tables.
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import run_pilot
run_pilot.TARGET_NAMES = ["dF_C7eq", "dF_C7ax", "barrier_C7eq_C7ax"]
from run_pilot import summarise

a, b = (sys.argv[1:3] if len(sys.argv) > 2 else ("results_ala2", "results_ala2_user"))
A = json.load(open(os.path.join(HERE, a, "ala2_raw.json")))
B = json.load(open(os.path.join(HERE, b, "ala2_raw.json")))
pooled = {r: A[r] + B[r] for r in A}
json.dump(summarise(pooled), open(os.path.join(HERE, "ala2_pooled_results.json"), "w"), indent=1)
print({r: len(v) for r, v in pooled.items()}, "-> ala2_pooled_results.json")
