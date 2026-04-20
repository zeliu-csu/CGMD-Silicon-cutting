#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import shutil
import subprocess as sp
import threading
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution


# =========================================================
# Configuration
# =========================================================

# ---------- CG mapping level ----------
# For CG8-to-1, N = 2 because 2×2×2 = 8 atoms are mapped to 1 CG bead.
N_CG = 2

# ---------- Tersoff template ----------
TEMPLATE_TERSOFF = "SiC_Erhart-Albe.tersoff"

# ---------- Working directory ----------
WORKDIR = "workdir_tersoff_de"

# ---------- LAMMPS executable ----------
LMP_CMD = ["lmp"]   # 建议改成你的实际可执行文件
NPROC = 120

# ---------- Enabled calibration cases ----------
# To be fully consistent with the Methods description, only compression and shear are used.
SIM_CASES = {
    "compress": {
        "infile": "AA-8w-compress/compress.in",
        "outfile": "stress-strain-CG8.txt",
        "ref": "AA-8w-compress/Compress-AA.txt",
        "strain_range": (0.0, 0.3),
        "weight": 0.5,
    },
    "shear": {
        "infile": "AA-8w-shear/shear.in",
        "outfile": "stress-strain-CG8.txt",
        "ref": "AA-8w-shear/Shear-AA.txt",
        "strain_range": (0.0, 0.6),
        "weight": 0.5,
    },
}

ENABLED_CASES = ["compress", "shear"]

# ---------- Temperature monitoring ----------
TEMP_THRESHOLD = 400.0
TEMP_N_CONSEC = 10

# ---------- DE search bounds ----------
# Methods: n_i in [0.9N, 1.1N]
DEFAULT_PARAMS = np.array([N_CG] * 6, dtype=float)
BOUNDS = [(0.9 * N_CG, 1.1 * N_CG) for _ in range(6)]

# ---------- Normalization baselines ----------
# Each loading mode should have its own normalization baseline.
BASELINE_RMSE = {case_name: None for case_name in ENABLED_CASES}


# =========================================================
# Utilities
# =========================================================

def read_stress_strain(path):
    """
    Read stress-strain data file with columns:
    timestep, strain, v_p1
    """
    path = Path(path)
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
    if df.shape[1] < 3:
        raise ValueError(f"File {path} does not contain at least 3 columns.")
    df = df.iloc[:, :3].copy()
    df.columns = ["timestep", "strain", "v_p1"]
    return df


def monitor_log(logfile: Path, stop_flag: dict, threshold: float, n_consec: int):
    """
    Monitor log.lammps in real time.
    During the first thermo section, if Temp exceeds threshold for n_consec
    consecutive outputs, stop the simulation.
    """
    logfile = Path(logfile)
    last_size = 0
    consec_count = 0
    temp_col = None
    phase = 0  # 0: not entered, 1: detect temperature, 2: stop detection

    while not stop_flag["stop"]:
        if logfile.exists():
            with logfile.open("r", encoding="utf-8", errors="ignore") as f:
                f.seek(last_size)
                for line in f:
                    if not line.strip():
                        continue
                    parts = line.split()

                    if parts[0] == "Step":
                        if phase == 0:
                            phase = 1
                            temp_col = parts.index("Temp") if "Temp" in parts else None
                            continue
                        elif phase == 1:
                            phase = 2
                            temp_col = None
                            continue

                    if phase == 1 and temp_col is not None:
                        try:
                            temp = float(parts[temp_col])
                        except Exception:
                            continue

                        if temp > threshold:
                            consec_count += 1
                            if consec_count >= n_consec:
                                print(f"[HALT] Temp > {threshold} for {n_consec} consecutive outputs.")
                                stop_flag["stop"] = True
                                return
                        else:
                            consec_count = 0

                last_size = logfile.stat().st_size

        time.sleep(1.0)


def generate_cg_tersoff(template_path, out_path,
                        n_lambda2, n_B, n_R, n_D, n_lambda1, n_A,
                        decimals=8):
    """
    Generate a CG Tersoff file from the AA template.
    Scaling rules for CG8-to-1 are controlled by the coefficients:
    {n_lambda2, n_B, n_R, n_D, n_lambda1, n_A}

    lambda2_CG = lambda2_AA / n_lambda2
    B_CG       = B_AA * n_B^3
    R_CG       = R_AA * n_R
    D_CG       = D_AA * n_D
    lambda1_CG = lambda1_AA / n_lambda1
    A_CG       = A_AA * n_A^3
    """
    template_path = Path(template_path)
    out_path = Path(out_path)

    with template_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    idx = None
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) >= 3 and parts[:3] == ["Si", "Si", "Si"]:
            idx = i
            break

    if idx is None:
        raise RuntimeError("Cannot find the 'Si Si Si' entry in the Tersoff file.")

    line1 = lines[idx].rstrip("\n")
    line2 = lines[idx + 1].strip()

    vals = [float(x) for x in line2.split()[:7]]
    # order: beta, lambda2, B, R, D, lambda1, A

    vals[1] = vals[1] / n_lambda2
    vals[2] = vals[2] * (n_B ** 3)
    vals[3] = vals[3] * n_R
    vals[4] = vals[4] * n_D
    vals[5] = vals[5] / n_lambda1
    vals[6] = vals[6] * (n_A ** 3)

    fmt = f"{{:.{decimals}f}}"
    new_line2 = "             " + "  ".join(fmt.format(v) for v in vals) + "\n"

    out_lines = lines.copy()
    out_lines[idx] = line1 + "\n"
    out_lines[idx + 1] = new_line2

    with out_path.open("w", encoding="utf-8") as f:
        f.writelines(out_lines)


def modify_infile_and_copy(in_src, in_dst):
    """
    Modify the original AA-based LAMMPS input to generate the CG8 input.
    The modifications are made to match the CG8 setup used in the manuscript.

    Rules:
    1. mass 1 28.0855 -> multiplied by 8
    2. pair_coeff * * SiC_Erhart-Albe.tersoff Si -> Si_8.tersoff
    3. output file after 'file' -> stress-strain-CG8.txt
    4. dump ... lammpstrj -> commented out
    5. thermo_style custom ... temp ... -> move temp to the last column
    6. lattice diamond a -> lattice diamond (a * 2) for CG8
    """
    in_src = Path(in_src)
    in_dst = Path(in_dst)

    with in_src.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        stripped = line.strip()

        # 1) mass scaling for CG8
        if stripped.startswith("mass") and "28.0855" in stripped:
            parts = line.split()
            try:
                mass_val = float(parts[2]) * 8.0
                line = f"{parts[0]}\t{parts[1]} {mass_val:.4f}\n"
            except Exception:
                pass

        # 2) pair_coeff file replacement
        elif stripped.startswith("pair_coeff") and "SiC_Erhart-Albe.tersoff" in stripped:
            line = line.replace("SiC_Erhart-Albe.tersoff", "Si_8.tersoff")

        # 3) output filename replacement
        elif stripped.startswith("fix") and "file" in stripped:
            line = re.sub(r"(file\s+)(\S+\.txt)", r"\1stress-strain-CG8.txt", line)

        # 4) comment out dump trajectory
        elif stripped.startswith("dump") and "lammpstrj" in stripped:
            line = "# " + line if not line.startswith("#") else line

        # 5) move temp to the last thermo_style column
        elif stripped.startswith("thermo_style") and "custom" in stripped:
            parts = line.split()
            if "temp" in parts:
                parts.remove("temp")
                parts.append("temp")
                line = " ".join(parts) + "\n"

        # 6) lattice constant scaling for CG8 (2x in each direction)
        elif stripped.startswith("lattice") and "diamond" in stripped:
            parts = line.split()
            try:
                lat_val = float(parts[2]) * N_CG
                line = f"{parts[0]}\t{parts[1]} {lat_val:.4f}\n"
            except Exception:
                pass

        new_lines.append(line)

    with in_dst.open("w", encoding="utf-8") as f:
        f.writelines(new_lines)


def run_lammps_with_log_monitor(infile_name, rundir, threshold=400.0, n_consec=10):
    """
    Run LAMMPS in rundir while monitoring log.lammps.
    If temperature exceeds threshold for n_consec consecutive outputs,
    terminate the run and return a penalty value.
    """
    rundir = Path(rundir)
    logfile = rundir / "log.lammps"

    stop_flag = {"stop": False}
    monitor_thread = threading.Thread(
        target=monitor_log,
        args=(logfile, stop_flag, threshold, n_consec),
        daemon=True
    )
    monitor_thread.start()

    cmd = ["mpirun", "-n", str(NPROC)] + LMP_CMD + ["-in", infile_name]
    proc = sp.Popen(cmd, cwd=rundir)

    while proc.poll() is None:
        if stop_flag["stop"]:
            proc.kill()
            print("[WARN] LAMMPS killed due to excessive temperature.")
            return 1e6
        time.sleep(2.0)

    return 0.0 if proc.returncode == 0 else 1e6


def compute_case_rmse(ref_df, test_df, lo, hi):
    """
    Compute RMSE between AA and CG curves within the prescribed strain interval.
    """
    ref = ref_df[(ref_df["strain"] >= lo) & (ref_df["strain"] <= hi)].copy()
    test = test_df[(test_df["strain"] >= lo) & (test_df["strain"] <= hi)].copy()

    if ref.empty or test.empty:
        return None

    ref["strain_round"] = ref["strain"].round(5)
    test["strain_round"] = test["strain"].round(5)

    merged = pd.merge(
        ref[["strain_round", "v_p1"]],
        test[["strain_round", "v_p1"]],
        on="strain_round",
        how="inner",
        suffixes=("_ref", "_test")
    )

    if merged.empty:
        return None

    diff = merged["v_p1_test"].to_numpy() - merged["v_p1_ref"].to_numpy()
    rmse = np.sqrt(np.mean(diff ** 2))
    return float(rmse)


def run_one_case(case_name, case_cfg, params, iteration):
    """
    Run one calibration case and return:
    (normalized_rmse, raw_rmse, test_df)

    On failure, return:
    (penalty, None, None)
    """
    n_lambda2, n_B, n_R, n_D, n_lambda1, n_A = params

    rundir = Path(WORKDIR) / (
        f"iter_{iteration:04d}_"
        f"nL2_{n_lambda2:.4f}_"
        f"nB_{n_B:.4f}_"
        f"nR_{n_R:.4f}_"
        f"nD_{n_D:.4f}_"
        f"nL1_{n_lambda1:.4f}_"
        f"nA_{n_A:.4f}"
    )
    casedir = rundir / case_name
    casedir.mkdir(parents=True, exist_ok=True)

    # 1. Generate CG tersoff file
    tersoff_out = casedir / "Si_8.tersoff"
    generate_cg_tersoff(
        TEMPLATE_TERSOFF,
        tersoff_out,
        n_lambda2, n_B, n_R, n_D, n_lambda1, n_A
    )

    # 2. Prepare case-specific input file
    in_src = Path(case_cfg["infile"])
    in_dst = casedir / in_src.name
    modify_infile_and_copy(in_src, in_dst)

    # 3. Run LAMMPS
    ret = run_lammps_with_log_monitor(
        infile_name=in_dst.name,
        rundir=casedir,
        threshold=TEMP_THRESHOLD,
        n_consec=TEMP_N_CONSEC
    )
    if ret != 0.0:
        print(f"[WARN] {case_name}: simulation failed.")
        return 1e6, None, None

    # 4. Read CG result
    out_file = casedir / case_cfg["outfile"]
    if not out_file.exists():
        print(f"[WARN] {case_name}: output file not found -> {out_file}")
        return 1e6, None, None

    test_df = read_stress_strain(out_file)

    # 5. Read AA reference
    ref_df = read_stress_strain(case_cfg["ref"])
    lo, hi = case_cfg["strain_range"]

    # 6. RMSE
    rmse = compute_case_rmse(ref_df, test_df, lo, hi)
    if rmse is None:
        print(f"[WARN] {case_name}: failed to compute RMSE.")
        return 1e6, None, None

    # 7. Per-case normalization baseline
    if BASELINE_RMSE[case_name] is None:
        BASELINE_RMSE[case_name] = rmse if rmse != 0 else 1.0

    normalized_rmse = rmse / BASELINE_RMSE[case_name]

    return float(normalized_rmse), float(rmse), test_df


def loss_function(params, iteration):
    """
    Total loss = weighted sum of normalized RMSE values for compression and shear.
    """
    total_loss = 0.0
    per_case_info = {}

    for case_name in ENABLED_CASES:
        case_cfg = SIM_CASES[case_name]
        weight = case_cfg["weight"]

        case_loss, raw_rmse, test_df = run_one_case(case_name, case_cfg, params, iteration)

        per_case_info[case_name] = {
            "normalized_rmse": case_loss,
            "raw_rmse": raw_rmse,
            "test_df": test_df,
        }

        total_loss += weight * case_loss

    print(f"[ITER {iteration:04d}] Total Loss = {total_loss:.6e}")

    return float(total_loss), per_case_info


# =========================================================
# Main optimization
# =========================================================

def main():
    # Clean working directory
    workdir = Path(WORKDIR)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # Interactive plotting
    plt.ion()
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))

    ax_map = {
        "compress": axs[0, 0],
        "shear": axs[0, 1],
    }
    ax_loss = axs[1, 0]
    axs[1, 1].axis("off")

    recent_lines_map = {}

    # Plot AA reference curves
    for case_name in ENABLED_CASES:
        ax = ax_map[case_name]
        ref_df = read_stress_strain(SIM_CASES[case_name]["ref"])
        ax.plot(
            ref_df["strain"].to_numpy(),
            ref_df["v_p1"].to_numpy(),
            label=f"AA reference ({case_name})",
            color="black",
            linewidth=2.0
        )
        ax.set_xlabel("Strain")
        ax.set_ylabel("Stress")
        ax.set_title(f"{case_name.capitalize()} response")
        ax.legend()
        recent_lines_map[case_name] = deque(maxlen=5)

    loss_values = []
    iter_values = []

    (loss_line,) = ax_loss.plot([], [], marker="o", label="Total loss")

    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Optimization history")
    ax_loss.legend()

    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)

    iteration_counter = {"count": 0}
    history = []

    def objective(params):
        iteration_counter["count"] += 1
        iteration = iteration_counter["count"]

        loss, per_case_info = loss_function(params, iteration)

        history.append({
            "iter": iteration,
            "n_lambda2": params[0],
            "n_B": params[1],
            "n_R": params[2],
            "n_D": params[3],
            "n_lambda1": params[4],
            "n_A": params[5],
            "loss": loss,
        })

        # Update response curves
        run_glob = (
            f"iter_{iteration:04d}_"
            f"nL2_{params[0]:.4f}_"
            f"nB_{params[1]:.4f}_"
            f"nR_{params[2]:.4f}_"
            f"nD_{params[3]:.4f}_"
            f"nL1_{params[4]:.4f}_"
            f"nA_{params[5]:.4f}"
        )

        for case_name in ENABLED_CASES:
            case_cfg = SIM_CASES[case_name]
            casedir = Path(WORKDIR) / run_glob / case_name
            test_file = casedir / case_cfg["outfile"]

            if not test_file.exists():
                continue

            test_df = read_stress_strain(test_file)
            ax = ax_map[case_name]
            recent_lines = recent_lines_map[case_name]

            if len(recent_lines) == recent_lines.maxlen:
                old_line = recent_lines.popleft()
                try:
                    old_line.remove()
                except Exception:
                    pass

            (new_line,) = ax.plot(
                test_df["strain"].to_numpy(),
                test_df["v_p1"].to_numpy(),
                label=f"Iter {iteration}",
                alpha=0.7
            )
            recent_lines.append(new_line)

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc="best")
            ax.relim()
            ax.autoscale_view()

        # Update loss history
        iter_values.append(iteration)
        loss_values.append(min(loss, 1e6))

        loss_line.set_data(iter_values, loss_values)

        ax_loss.relim()
        ax_loss.autoscale_view()

        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)

        return loss

    # -----------------------------------------------------
    # Baseline evaluation
    # -----------------------------------------------------
    print("[INFO] Running baseline with default CG scaling coefficients...")
    baseline_loss = objective(DEFAULT_PARAMS.copy())
    print(f"[INFO] Baseline total loss = {baseline_loss:.6e}")

    # -----------------------------------------------------
    # Differential evolution
    # -----------------------------------------------------
    result = differential_evolution(
        objective,
        bounds=BOUNDS,
        maxiter=20000,
        tol=1e-2,
        disp=True,
        polish=False,
    )

    print("\nOptimization finished.")
    print("Best parameters (n_lambda2, n_B, n_R, n_D, n_lambda1, n_A):")
    print(result.x)
    print("Minimum loss:")
    print(result.fun)

    # Save optimization history
    history_df = pd.DataFrame(history)
    history_df.to_csv(Path(WORKDIR) / "optimization_history.csv", index=False)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()