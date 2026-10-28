# Pilot findings (10 replicas x 2 regimes, exact FES known)

RMS z-score: 1 = calibrated, >1 = error bars too small. Coverage = share of 95% intervals containing the truth.

| Method | F z-score short / long | ΔF & barrier 95% coverage short / long |
|---|---|---|
| Random-init ensemble (RiD-style) | 7.3 / 10.7 | 27% / 17% |
| Bootstrap ensemble | 1.0 / 2.0 | 100% / 70% |
| Noise-injected ensemble | 4.1 / 2.9 | 67% / 50% |
| Noise-injected + label recalibration | 2.0 / 2.7 | 93% / 57% |
| Random-init + label recalibration | 2.5 / 4.0 | 60% / 50% |
| Gradient GP (label noise from block averaging) | 4.1 / 3.1 | 63% / 43% |

Labels themselves: block-averaged 95% error bars cover the true mean force only 71% (short) / 65% (long);
naive i.i.d. error bars cover 8% / 4%.

1. H1 supported: random-init spread is 7-11x too small for F and ΔF.
2. H2 supported: recalibrating the force-level spread on held-out labels brings force z to 1.3/1.7 but F stays 2.5/4.0.
   Force-level calibration does not carry over to integrated free energies.
3. Methods that trust the labels' own error bars (GP, noise-injected) inherit the labels' overconfidence.
4. Bootstrap is the only method near calibration, and only with short trajectories. With long trajectories the
   remaining error is dominated by systematic label bias (~0.11 kT/unit RMS from the average-force estimator),
   which no resampling scheme can see.

Caveats: one toy 2D system; 30 target estimates per cell (coverage SE ~ +/-8 points); grid points are correlated;
window placement used the true FES; SE floor 0.05 and network size/epochs not yet varied.

# Label ablation (same MD seeds, paired)

Label modes: md = raw labels; debiased = exact estimator bias (RMS 0.11 kT/unit) subtracted;
oracle = exact gradient + noise whose std equals the reported block error (label error bars correct by construction).
Label 95% coverage: md 71%/65%, debiased 77%/87%, oracle 95%/95% (short/long).

F z-score (1 = calibrated), short / long:

| Method | md | debiased | oracle |
|---|---|---|---|
| Random-init ensemble | 7.3 / 10.7 | 7.5 / 5.9 | 6.5 / 5.6 |
| Bootstrap ensemble | 1.0 / 2.0 | 1.1 / 1.0 | 1.0 / 1.0 |
| Noise-injected ensemble | 4.1 / 2.9 | 4.3 / 1.3 | 0.7 / 1.0 |
| Noise-injected + recalibration | 2.0 / 2.7 | 2.2 / 1.3 | 1.1 / 2.2 |
| Gradient GP | 4.1 / 3.1 | 4.4 / 1.3 | 0.7 / 0.9 |

5. Bootstrap's long-run slip (2.0) is caused by label bias: removing it gives 1.0. Bootstrap does not use the
   labels' error bars, so it is calibrated whenever labels are unbiased, even when those error bars are wrong.
6. GP and noise-injected ensembles are calibrated only when the labels' error bars are correct (oracle).
   In short runs, block averaging underestimates noise and these methods stay 4x overconfident even with bias removed.
7. The RiD-style random-init ensemble stays 5.6-7.5x overconfident even with perfect labels: it ignores label noise
   by construction. This is the most robust finding.
8. Held-out-label recalibration is fragile: with oracle labels in long runs it shrank the error bars (2.2 on F,
   3.4 on targets). Residuals on noisy held-out labels are a weak signal for model error.

# 4D pilot (x, y, u, v; 1000 Sobol windows, 850 training labels, 8 replicas x 2 regimes, raw MD labels)

Exact targets: dF_BA 4.07, dF_CA 7.44, F(S_BC)-F(A) 8.01 kT.

| Method | F z-score 4D short / long (2D) | F RMSE kT 4D short / long | ΔF & barrier 95% cov. 4D short / long |
|---|---|---|---|
| Random-init ensemble | 2.3 / 1.7 (7.3 / 10.7) | 0.30 / 0.26 | 88% / 100% |
| Bootstrap ensemble | 1.1 / 1.5 (1.0 / 2.0) | 0.64 / 0.57 | 88% / 92% |
| Noise-injected ensemble | 5.6 / 1.8 (4.1 / 2.9) | 0.46 / 0.42 | 79% / 62% |
| Random-init + recalibration | 0.9 / 1.8 (2.5 / 4.0) | 0.30 / 0.26 | 100% / 100% |
| Gradient GP (ARD) | 8.6 / 3.4 (4.1 / 3.1) | 0.15 / 0.07 | 75% / 33% |

9. Overconfidence of the random-init ensemble depends strongly on data density: 7-11x in 2D (dense labels),
   only 1.7-2.3x in 4D (sparse labels), because members disagree more where data are thin.
10. The GP is the most accurate model in 4D (2-8x lower F error than the ensembles) and the most overconfident
    (3.4-8.6x). Ensembles are honest-ish but inaccurate; the ensemble anchor-point error (near minimum A)
    dominates their ΔF errors (0.5-1.3 kT).
11. H2 is weaker in 4D: recalibrating the random-init ensemble at force level also fixes F in the short regime.
12. GP cost was not prohibitive at this size (exact factorisation 1.4 s for a 3400 x 3400 Gram matrix,
    hyperparameters fitted on 400 labels in ~67 s). "Beyond GP" needs larger N*d.

Caveats: network hyperparameters were not re-tuned for 4D (accuracy gap vs GP may be partly a tuning issue);
24 target estimates per cell (coverage SE ~ +/-9 points); no 4D label ablation yet.

# 4D network tuning (held-out labels only; same seeds as the 4D pilot)

Grid: weight decay {1e-6, 1e-4, 1e-3, 3e-3} x epochs {4000, 8000}, 4 independent datasets, M=4.
Selection by mean rank of inverse-variance weighted held-out label loss -> wd 3e-3, 8000 epochs
(the true-error optimum was wd 1e-3, 8000 epochs; both are the two top-ranked settings).
Within one dataset, held-out label loss barely tracks true F error (Spearman 0.00 / 0.07 / 0.88 / 0.10):
label noise swamps model differences, so tuning must pool several datasets.

| Method (4D) | F RMSE kT default -> tuned, short / long | F z-score default -> tuned, short / long |
|---|---|---|
| Random-init ensemble | 0.30 -> 0.20 / 0.26 -> 0.13 | 2.3 -> 3.7 / 1.7 -> 2.5 |
| Bootstrap ensemble | 0.64 -> 0.33 / 0.57 -> 0.29 | 1.1 -> 1.2 / 1.5 -> 1.6 |
| Noise-injected ensemble | 0.46 -> 0.28 / 0.42 -> 0.23 | 5.6 -> 9.2 / 1.8 -> 2.6 |
| Random-init + label recalibration | 0.30 -> 0.20 / 0.26 -> 0.13 | 0.9 -> 0.8 / 1.8 -> 1.4 |
| Gradient GP (unchanged) | 0.15 / 0.07 | 8.6 / 3.4 |

13. Tuning halves the ensembles' error (now 1.3-2x the GP's), but ensembles whose spread comes from random
    initialisation become MORE overconfident (spread shrinks ~3x as members agree): accuracy and honesty trade off.
14. Bootstrap is the exception: error halves while calibration is unchanged (z ~1.2 / 1.6).
15. Rescaling the random-init spread on held-out labels gives the best combined result for networks in 4D
    (RMSE 0.20 / 0.13 kT with z 0.8 / 1.4), unlike 2D where it failed at the F level.
16. The GP remains the most accurate and the most overconfident model at this size.

# 4D data-density sweep (nested Sobol windows, fixed test set, short runs, tuned nets, 3 replicas)

| Windows (train labels) | F z-score: random-init / +rescaled / bootstrap / GP | F RMSE kT: random-init / bootstrap / GP | GP fit time |
|---|---|---|---|
| 500 (425) | 2.8 / 0.9 / 1.0 / 4.8 | 0.09 / 0.29 / 0.17 | 74 s |
| 1000 (850) | 3.1 / 0.7 / 1.2 / 7.4 | 0.15 / 0.44 / 0.16 | 116 s |
| 2000 (1700) | 2.9 / 0.8 / 1.4 / 13.1 | 0.17 / 0.33 / 0.16 | 172 s |
| 4000 (3400) | 2.0 / 0.5 / 1.5 / 18.4 | 0.40 / 0.47 / 0.16 | 722 s |

17. GP overconfidence grows steadily with data density (4.8 -> 18.4x) while its accuracy stays flat (~0.16 kT):
    its posterior shrinks as 1/N but the label errors it trusts are underestimated and correlated. Headline result.
18. Ensemble calibration is roughly density-independent: random-init stays 2-3x overconfident, bootstrap 1.0-1.5x,
    rescaled random-init 0.5-0.9x (slightly under-confident).
19. GP cost grows steeply (74 s -> 722 s per fit on one CPU; 13600x13600 Gram matrix factorised in 52 s, ~4 GB RAM);
    ensembles ~constant (fixed 8000-step budget, mini-batch 1024).
20. OPEN ISSUE: network accuracy degrades at 4000 windows (0.40 kT). Tripling steps or full-batch training
    only reduces it to 0.33 kT; local gradient errors are unchanged (~0.18), so errors accumulate on integration.
    Needs resolving before network accuracy claims at high density.
Note: the test set here is fixed over the whole F<10 kT region (not only near labels), so RMSEs differ from the 4D pilot.
Coverage of targets: 9 estimates per cell - indicative only.

# Density sweep, fixed network training (Adam 8000 mini-batch steps + 3000 full-batch L-BFGS; GP unchanged)

Diagnosis: at 4000 windows the networks were under-converged, not over-regularised. Scaling weight decay with 1/N
or widening the network did not help; more Adam steps helped slowly (3000 -> 8000 -> 24000 steps: 0.6-0.9 -> 0.45 -> 0.33 kT).
A gradient-matching loss penalises long-wavelength F errors only weakly (an error of amplitude A over length L
changes gradients by ~A/L), so first-order optimisers remove them slowly. A full-batch L-BFGS polish fixes it
(4000 windows, one dataset: 0.46 -> 0.064 kT) without hurting lower densities.

| Windows | F z-score: random-init / +rescaled / bootstrap / GP | F RMSE kT: random-init / bootstrap / GP |
|---|---|---|
| 500 | 3.9 / 1.0 / 0.9 / 4.8 | 0.15 / 0.13 / 0.17 |
| 1000 | 4.4 / 0.8 / 1.1 / 7.4 | 0.07 / 0.08 / 0.16 |
| 2000 | 4.0 / 0.7 / 1.4 / 13.1 | 0.10 / 0.14 / 0.16 |
| 4000 | 3.0 / 0.5 / 1.4 / 18.4 | 0.12 / 0.17 / 0.16 |

21. The high-density accuracy loss is resolved: networks are now as accurate as or more accurate than the GP at
    every density (remaining variation is within replica noise, per-replica RMSE 0.04-0.25 kT).
22. Better-converged random-init ensembles are again more overconfident (3-4.4x), confirming finding 13.
23. Bootstrap stays near-honest (0.9-1.4x) AND accurate: best single method across densities.
24. ΔF/barrier 95% coverage (9 estimates per cell): bootstrap 100% at all densities; GP falls 67% -> 33%.
25. The GP is now the least accurate, most overconfident and (at 4000 windows) slowest model.

# Alanine dipeptide (real molecule: OpenMM, vacuum, 300 K, CVs phi/psi; 5 replicas per regime)

Reference ("truth"): mean of 4 independent 25 ns well-tempered metadynamics runs; precision (standard error over
runs) 0.046 kT on F and 0.15 kT/rad on grad F in the F<8 kT region. All z-scores below add the reference's own
standard error to the model's, so they are conservative (they slightly favour the models).
Labels: 372 restraint windows (15-degree grid, F_ref<15 kT), k = 1000 kJ/mol/rad^2, umbrella-integration mean force.
Targets anchored at C5: dF(C7eq) 0.20 kT, dF(C7ax) 3.60 kT, C7eq->C7ax barrier 13.07 kT.

| Regime | F z-score: random-init / +rescaled / bootstrap / noise / GP | F RMSE kT: init / boot / GP | targets 95% cov: init / boot / GP | label 95% cov |
|---|---|---|---|---|
| short (10 ps) | 2.75 / 1.70 / 1.02 / 1.55 / 1.61 | 0.162 / 0.168 / 0.164 | 0.73 / 0.93 / 0.80 | 0.76-0.81 |
| long (50 ps)  | 3.33 / 2.31 / 1.92 / 2.14 / 2.20 | 0.113 / 0.121 / 0.122 | 0.80 / 0.93 / 0.87 | 0.62-0.64 |

26. H1 holds on a real molecule: the RiD-style random-init ensemble is overconfident in every one of 10 replicas
    (F z 2.2-4.0), and its ΔF/barrier error bars cover only 73-80% at nominal 95%.
27. Bootstrap is again the best method: honest on F with short runs (z 1.02) and on ΔF/barrier in both regimes
    (z 0.8-0.9, 93% coverage), at the same accuracy as all other methods.
28. Longer runs make every method more overconfident (all F z-scores rise by ~0.6-0.9) while accuracy improves
    (0.16 -> 0.12 kT). Same pattern as the 2D toy (bootstrap 1.0 -> 2.0): as statistical error shrinks, systematic
    label error (umbrella-integration curvature bias, plus reference error on gradients) takes over and no method
    that trusts the labels' block errors can see it. Label 95% coverage falling 0.79 -> 0.63 shows this directly.
    Caveat: label coverage is computed against the reference gradient WITHOUT its 0.15 kT/rad error, so part of
    the drop is reference error; a longer reference or a debiased-label ablation would separate the two.
29. Force-level held-out rescaling helps random-init (2.75 -> 1.70, 3.33 -> 2.31) but does not reach calibration
    on F — H2 again: calibration on forces does not transfer to F.
30. At this data size (316 training labels) the GP is accurate and its cost is negligible; its overconfidence is
    moderate (1.6 / 2.2), consistent with the low-density end of the 4D density sweep.

Independent replication on the user's PC (Windows, same code; MD trajectories differ, so these are 5 new
replicas per regime, not a rerun of the same ones):
| Regime | F z-score: init / +rescaled / boot / noise / GP | F RMSE: init / boot / GP | targets 95% cov: init / boot / GP |
|---|---|---|---|
| short | 3.11 / 1.90 / 1.07 / 1.62 / 1.69 | 0.161 / 0.167 / 0.168 | 0.60 / 0.93 / 0.87 |
| long  | 3.19 / 2.04 / 1.61 / 2.12 / 2.02 | 0.108 / 0.112 / 0.107 | 0.87 / 1.00 / 0.93 |
31. Findings 26-30 replicate: same ranking, z-scores within ~0.3, RMSE within 0.01 kT, label 95% coverage
    0.79 (short) and 0.62 (long) on both machines. The paper can pool 10 replicas per regime.
