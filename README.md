# CGMD-Silicon-cutting

## Overview
This repository accompanies the manuscript:
“Bridging atomistic and large-scale simulations of silicon cutting via a bottom-up coarse-grained molecular dynamics framework”

This repository provides the code and data supporting a data-driven coarse-grained molecular dynamics (CGMD) framework for nanometric cutting of single-crystal silicon. It includes parameter optimization  workflows to ensure reproducibility of the reported results.

---
## 🔧 Software and Dependencies

The differential evolution (DE) optimization is implemented in Python using SciPy.

All molecular dynamics (MD) simulations are performed using the Large-scale Atomic/Molecular Massively Parallel Simulator (LAMMPS). Structural configurations and atomic-scale evolution are visualized using OVITO.

- SciPy (optimization): https://scipy.org/
- LAMMPS (MD simulation): https://www.lammps.org/
- OVITO (visualization): https://www.ovito.org/

## 🔧 Tersoff Potential Optimization (Si–Si Interaction)

Six key parameters are optimized:

Ω_T = {n_lambda2, n_B, n_R, n_D, n_lambda1, n_A}

The optimization is performed using differential evolution.

The objective function is defined as the weighted sum of normalized RMSE values between AA and CG stress–strain responses under compression and shear.

---

## 🔧 Morse Potential Optimization (Si–C Interaction)

The Morse potential parameters (De, alpha, r0) are optimized by fitting the CG interaction energy–distance curve to AA reference data.

The optimization is performed using differential evolution.

The objective function is defined as the RMSE between AA and CG energy–distance curves.

---

Together, the optimized Tersoff (Si–Si) and Morse (Si–C) potentials enable a consistent CGMD framework for nanometric cutting simulations.
