"""Label-error diagnosis for the long regime: are labels' 95% intervals
missed because of reference error or because of label error?"""
import json, sys, numpy as np
from ala2_reference import Reference
from ala2_labels import sample_labels
from run_ala2 import setup, REGIMES
R = Reference(); S = setup(R)
out = {}
for regime in ("short", "long"):
    d = sample_labels(S["centres"], REGIMES[regime], seed=7000 + 1000 * list(REGIMES).index(regime) + 0)
    _, lg = R.free_energy(d["s"]); _, sG = R.precision(d["s"])
    e = d["g"] - lg
    z1 = e / np.maximum(d["se"], 1e-12)
    z2 = e / np.sqrt(d["se"]**2 + sG**2)
    out[regime] = dict(cov95_label_se=float((abs(z1) < 1.96).mean()),
                       cov95_label_plus_ref=float((abs(z2) < 1.96).mean()),
                       rms_err=float(np.sqrt((e**2).mean())), rms_se=float(np.sqrt((d["se"]**2).mean())),
                       rms_ref=float(np.sqrt((sG**2).mean())), mean_err=e.mean(0).tolist())
    print(regime, out[regime], flush=True)
json.dump(out, open("label_check.json", "w"), indent=1)
