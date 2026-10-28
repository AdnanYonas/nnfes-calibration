# nnfes-calibration

**Do neural-network free-energy surfaces know when they are wrong?**
Code and data for a calibration study of uncertainty estimates for free-energy surfaces (FES) learned from mean forces.

Adnan Younas (Department of Computer Science, Bahria University, Islamabad) and
Habiba Adnan (Department of Computer Science, Virtual University of Pakistan, Islamabad).

[![DOI](https://zenodo.org/badge/DOI/ZENODO_DOI_HERE.svg)](https://doi.org/ZENODO_DOI_HERE)

---

## What this repository contains

Every model is a scalar function F(s) whose gradient is fitted to mean-force labels from restrained simulations.
Each model's error bars are tested against a known truth:

| System | Truth | Folder |
|---|---|---|
| 2D model (Müller–Brown plus a hidden coordinate) | exact F | `src/system.py`, `results*/` |
| 4D non-separable model | exact F | `src/system4d.py`, `results_4d*/`, `results_density*/` |
| Alanine dipeptide in vacuum, 300 K (OpenMM), CVs φ/ψ | mean of 4 × 25 ns well-tempered metadynamics runs | `ala2/` |

The uncertainty methods tested (`src/models.py`) are:

| Name in code | Method |
|---|---|
| `ens_init` | 8 networks that differ only by random initialization (the reinforced-dynamics "uncertainty indicator") |
| `ens_boot` | 8 networks trained on bootstrap resamples of the windows |
| `ens_noise` | 8 networks trained on labels perturbed by their error bars, with inverse-variance weights |
| `*_rc` | the same, with the spread rescaled to fit 15% held-out labels (no ground truth used) |
| `gp` | Gaussian process with gradient observations and the labels' own noise (isotropic, ARD or periodic kernel) |

Calibration is scored as RMS z-score (1 = honest error bars, >1 = overconfident) and coverage of nominal intervals.
It is scored for the mean force, for the anchored free energy, and for free-energy differences and barriers across
independent replicas.

## Repository layout

```
src/                     model systems, samplers, models, experiment runners, original analysis figures
  system.py, system4d.py      exact 2D and 4D free-energy surfaces
  sampling.py, sampling_nd.py restrained overdamped Langevin -> mean-force labels (+ exact estimator bias)
  models.py                   gradient-matching network ensembles, gradient GPs, held-out rescaling
  run_pilot.py                2D study (label modes: md / debiased / oracle)
  run_nd.py                   4D study
  tune_4d.py                  4D hyperparameter search on held-out labels only
  run_density.py              4D data-density sweep (500-4000 windows)
ala2/                    alanine dipeptide
  ala2_system.py              OpenMM system, dihedral CVs, restraints
  reference_metad.py          well-tempered metadynamics reference run
  ref_run1..4.npz             the four 25 ns reference FES (saved every 5 ns)
  ala2_reference.py           reference F, its gradient and its standard error
  ala2_labels.py              restrained-MD umbrella-integration labels
  run_ala2.py                 the calibration study (short 10 ps / long 50 ps per window)
  pool_replicas.py            pools the two machines' replicas -> ala2_pooled_results.json
  label_check.py              label-error diagnosis (block error vs reference error)
  data/                       topology/coordinates (from the openmmtools test-system collection)
  results_ala2/               5 replicas per regime, Linux workstation
  results_ala2_user/          5 replicas per regime, Windows desktop (independent trajectories)
results/                 2D, raw MD labels          (10 replicas per regime)
results_debiased/        2D, estimator bias removed (same seeds)
results_oracle/          2D, oracle labels          (same seeds)
results_4d/              4D, default network settings (8 replicas)
results_tune4d/          4D hyperparameter search
results_4d_tuned/        4D, tuned settings
results_density/         4D density sweep, preliminary (no L-BFGS polish, 3 replicas)
results_density_lbfgs/   4D density sweep, final (with L-BFGS polish, 3 replicas)
results/PILOT_FINDINGS.md   running log of all findings (1-31)
paper/                   scripts that regenerate every manuscript figure and table from the result files
  make_paper_figures.py -> paper/figures/fig1..fig10 (.pdf, .png)
  make_tables.py        -> paper/tables/*.tex
logs/                    console logs of every run
```

### Result files

Every experiment writes two JSON files:

* `*_raw.json`: one entry per replica and regime. For each method it holds calibration summaries at the force and
  free-energy levels (`cov68`, `cov95`, `z_rms`, `rmse`, `mean_sigma`, `spearman`, `miscal_area`, calibration
  `curve`) and the target estimates (`est`, `sig`, `true`). It also records label coverage, rescaling factors,
  GP hyperparameters and timings.
* `*_results.json`: the same metrics averaged over replicas. Target coverage and z-scores are pooled across
  replicas.

Free energies are in units of k_BT. Mean forces are in k_BT per CV unit (per radian for alanine dipeptide).

## Installation

```
conda create -n fescal python=3.11
conda activate fescal
pip install -r requirements.txt
```

A CPU is enough. If a CUDA GPU is present, PyTorch uses it automatically. OpenMM is needed only for `ala2/`.

## Reproducing the results

Run each command from the folder shown. Approximate wall times are on one desktop CPU. All long runs except the 2D
ones accept `--resume`, which continues after an interruption.

| Experiment | Command | Time |
|---|---|---|
| 2D, MD labels | `cd src` then `python run_pilot.py --replicas 10` | ~20 min |
| 2D, bias removed | `python run_pilot.py --replicas 10 --label-mode debiased --out ../results_debiased` | ~20 min |
| 2D, oracle labels | `python run_pilot.py --replicas 10 --label-mode oracle --out ../results_oracle` | ~20 min |
| 4D, default | `python run_nd.py --resume` | ~1.5 h |
| 4D, tuning | `python tune_4d.py` | ~35 min |
| 4D, tuned | `python run_nd.py --wd 3e-3 --epochs 8000 --out ../results_4d_tuned --gp-from ../results_4d/pilot_raw.json --resume` | ~2.5 h |
| 4D density sweep | `python run_density.py --replicas 3 --out ../results_density_lbfgs --resume` | ~2 h |
| Ala2 reference (optional; the four runs are included) | `cd ala2` then `python reference_metad.py --seed 1 --ns 25 --out ref_run1.npz` (seeds 1–4) | ~20 min each |
| Ala2 study | `python run_ala2.py --resume --out results_ala2_mine` | 1.5–3 h |
| Pool Ala2 replicas | `python pool_replicas.py` | seconds |
| Manuscript figures and tables | `cd paper` then `python make_paper_figures.py` and `python make_tables.py` | ~1 min |

Each runner also has a `--quick` smoke test. Seeds are fixed per system, regime and replica, so reruns on the same
machine reproduce the results. On a different machine, network training with L-BFGS can differ by a few percent in
z-score because of floating-point differences. RMSE agrees, and OpenMM trajectories differ, which gives independent
replicas.

## Key numbers (free-energy RMS z-score; 1 = calibrated)

| | 2D short / long | 4D tuned short / long | Alanine dipeptide short / long |
|---|---|---|---|
| Random-init ensemble | 7.27 / 10.66 | 3.67 / 2.53 | 2.93 / 3.26 |
| Bootstrap ensemble | 0.98 / 1.97 | 1.18 / 1.55 | 1.05 / 1.76 |
| Noise-injected ensemble | 4.11 / 2.90 | 9.16 / 2.58 | 1.58 / 2.13 |
| Gradient GP | 4.11 / 3.07 | 8.59 / 3.42 | 1.65 / 2.11 |

In the 4D density sweep, the GP's z-score rises from 4.8 (500 windows) to 18.4 (4000 windows) while its accuracy
stays flat.

## Citation

If you use this code or data, please cite the archived version (see `CITATION.cff`):

> A. Younas and H. Adnan, *Code and data for: Do neural-network free-energy surfaces know when they are wrong?
> Calibrated uncertainty for free-energy landscapes learned from mean forces*, version 1.0.0, Zenodo (2026),
> doi:ZENODO_DOI_HERE.

## License

Code: MIT (see `LICENSE`). Data and results in this repository: CC BY 4.0. The alanine-dipeptide topology and
coordinates in `ala2/data/` come from the openmmtools test-system collection (MIT licence).
