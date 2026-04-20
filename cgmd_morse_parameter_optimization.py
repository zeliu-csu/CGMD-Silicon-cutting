#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import shutil
import subprocess as sp
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution


# =========================================================
# Configuration
# =========================================================

# ---------- LAMMPS executable ----------
# Recommended: set this to "lmp" if LAMMPS is available in PATH.
LMP_CMD = ["lmp"]

# ---------- Input / reference files ----------
TEMPLATE_IN = "create_cg_sic_morse.in"
REF_FILE = "pe.txt"

# ---------- Working directory ----------
WORKDIR = "morse_optimization_runs"

# ---------- Parallel execution ----------
NPROC = 120

# ---------- Morse parameters ----------
# (De, alpha, r0)
INIT_PARAMS = np.array([0.4350, 4.6487, 1.9475], dtype=float)

# Parameter bounds used by differential evolution
BOUNDS = [
    (0, 2.0),      # De
    (3.0, 6.0),      # alpha
    (1.9475, 4.0),   # r0
]


# =========================================================
# Utilities
# =========================================================

def read_pe_file(path):
    """
    Read interaction energy-distance data.

    Expected usable columns:
    - v_dis : separation distance
    - c_1   : interaction energy

    This function assumes the actual data are stored in columns 2 and 3
    (0-based indexing: 1 and 2) after excluding comments.
    """
    path = Path(path)

    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)

    if df.shape[1] < 3:
        raise ValueError(f"File {path} does not contain at least 3 columns.")

    df = df.iloc[:, 1:3].copy()
    df.columns = ["v_dis", "c_1"]

    df["v_dis"] = pd.to_numeric(df["v_dis"], errors="coerce")
    df["c_1"] = pd.to_numeric(df["c_1"], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    return df


def replace_morse_params_infile(template_path, out_path, params):
    """
    Replace the Morse parameters in the LAMMPS input file.

    pair_coeff 1 2 morse De alpha r0
    """
    template_path = Path(template_path)
    out_path = Path(out_path)

    with template_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    new_line = (
        f"pair_coeff 1 2 morse "
        f"{params[0]:.6f} {params[1]:.6f} {params[2]:.6f}\n"
    )

    pattern = re.compile(r"^pair_coeff\s+1\s+2\s+morse.*$")
    replaced = False

    for i, line in enumerate(lines):
        if pattern.match(line.strip()):
            lines[i] = new_line
            replaced = True
            break

    if not replaced:
        lines.append(new_line)

    with out_path.open("w", encoding="utf-8") as f:
        f.writelines(lines)


def run_lammps(infile_name, rundir):
    """
    Run LAMMPS in the given directory.
    Returns 0 if successful, otherwise a nonzero return code.
    """
    rundir = Path(rundir)
    cmd = ["mpirun", "-n", str(NPROC)] + LMP_CMD + ["-in", infile_name]

    proc = sp.Popen(cmd, cwd=rundir)
    proc.wait()

    return proc.returncode


def compute_rmse(ref_df, test_df):
    """
    Compute RMSE between the AA reference curve and CG curve after merging
    by the sampled separation distance v_dis.
    """
    merged = pd.merge(
        ref_df,
        test_df,
        on="v_dis",
        suffixes=("_ref", "_test"),
        how="inner"
    )

    if merged.empty:
        return None, None

    diff = merged["c_1_test"].to_numpy() - merged["c_1_ref"].to_numpy()
    rmse = np.sqrt(np.mean(diff ** 2))

    return float(rmse), merged


def prepare_run_directory(rundir):
    """
    Create a clean run directory.
    """
    rundir = Path(rundir)
    rundir.mkdir(parents=True, exist_ok=True)


def copy_required_files(rundir):
    """
    Copy files required by the LAMMPS run into the working directory.

    Adjust this list if your input script depends on additional files.
    """
    required_files = ["Si.tersoff"]

    for fname in required_files:
        src = Path(fname)
        if src.exists():
            shutil.copy(src, Path(rundir) / src.name)
        else:
            raise FileNotFoundError(
                f"Required file not found: {fname}. "
                f"Please place it in the script directory or update copy_required_files()."
            )


# =========================================================
# Objective function
# =========================================================

def loss_function(params, ref_df, template_in, workdir, iteration):
    """
    Objective function for Morse parameter optimization.

    Loss = RMSE between AA and CG interaction energy-distance curves.
    """
    rundir = Path(workdir) / (
        f"iter_{iteration:04d}_"
        f"De_{params[0]:.4f}_"
        f"alpha_{params[1]:.4f}_"
        f"r0_{params[2]:.4f}"
    )
    prepare_run_directory(rundir)

    infile = rundir / "in.lammps"
    replace_morse_params_infile(template_in, infile, params)

    copy_required_files(rundir)

    ret = run_lammps(infile.name, rundir)
    if ret != 0:
        print(f"[WARN] LAMMPS run failed with return code {ret}")
        return 1e12, None, None

    test_file = rundir / "pe_test.txt"
    if not test_file.exists():
        print("[WARN] Output file pe_test.txt was not generated.")
        return 1e12, None, None

    test_df = read_pe_file(test_file)

    rmse, merged = compute_rmse(ref_df, test_df)
    if rmse is None:
        print("[WARN] No overlapping v_dis values between reference and test data.")
        return 1e12, None, None

    print(
        f"[ITER {iteration:04d}] "
        f"De={params[0]:.6f}, alpha={params[1]:.6f}, r0={params[2]:.6f}, "
        f"RMSE={rmse:.6e}"
    )

    return float(rmse), test_df, merged


# =========================================================
# Main optimization
# =========================================================

def main():
    workdir = Path(WORKDIR)

    # Clean previous runs
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # Load AA reference data
    ref_df = read_pe_file(REF_FILE)

    # ---------- Figure 1: fitting curves ----------
    plt.ion()
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ref_line, = ax1.plot(
        ref_df["v_dis"].to_numpy(),
        ref_df["c_1"].to_numpy(),
        label="AA reference",
        color="black",
        linewidth=2.0
    )
    ax1.set_xlabel("Distance")
    ax1.set_ylabel("Interaction energy")
    ax1.set_title("Morse potential fitting")
    recent_lines = []

    ax1.legend()
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)

    # ---------- Figure 2: loss history ----------
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    loss_values = []
    iter_values = []

    (loss_line,) = ax2.plot([], [], marker="o", label="RMSE")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("RMSE")
    ax2.set_title("Optimization history")
    ax2.legend()

    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)

    iteration = {"count": 0}
    history = []

    def objective(params):
        iteration["count"] += 1
        it = iteration["count"]

        loss, test_df, merged = loss_function(
            params=params,
            ref_df=ref_df,
            template_in=TEMPLATE_IN,
            workdir=WORKDIR,
            iteration=it
        )

        history.append({
            "iter": it,
            "De": params[0],
            "alpha": params[1],
            "r0": params[2],
            "rmse": loss,
        })

        # ---- update fit curve figure ----
        rundir = Path(WORKDIR) / (
            f"iter_{it:04d}_"
            f"De_{params[0]:.4f}_"
            f"alpha_{params[1]:.4f}_"
            f"r0_{params[2]:.4f}"
        )
        test_file = rundir / "pe_test.txt"

        if test_file.exists():
            test_df_plot = read_pe_file(test_file)

            # keep only the most recent 5 curves
            if len(recent_lines) >= 5:
                old_line = recent_lines.pop(0)
                try:
                    old_line.remove()
                except Exception:
                    pass

            (new_line,) = ax1.plot(
                test_df_plot["v_dis"].to_numpy(),
                test_df_plot["c_1"].to_numpy(),
                label=f"Iter {it}",
                alpha=0.7
            )
            recent_lines.append(new_line)

            handles = [ref_line] + recent_lines
            labels = [h.get_label() for h in handles]
            ax1.legend(handles, labels, loc="best")
            ax1.relim()
            ax1.autoscale_view()

            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)

        # ---- update loss history ----
        iter_values.append(it)
        loss_values.append(loss)
        loss_line.set_data(iter_values, loss_values)
        ax2.relim()
        ax2.autoscale_view()

        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)

        return loss

    # ---------- Baseline evaluation ----------
    print("[INFO] Running baseline evaluation...")
    baseline_loss = objective(INIT_PARAMS.copy())
    print(f"[INFO] Baseline RMSE = {baseline_loss:.6e}")

    # ---------- Differential evolution ----------
    result = differential_evolution(
        objective,
        bounds=BOUNDS,
        maxiter=5000,
        tol=1e-2,
        disp=True,
        polish=False,
    )

    print("\nOptimization finished.")
    print("Best parameters (De, alpha, r0):")
    print(result.x)
    print("Minimum RMSE:")
    print(result.fun)

    # ---------- Save optimization history ----------
    history_df = pd.DataFrame(history)
    history_df.to_csv(workdir / "optimization_history.csv", index=False)
    print(f"Optimization history saved to: {workdir / 'optimization_history.csv'}")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()