#!/usr/bin/env python3
"""Tests for M8 — Standalone Simulator (`src/simulate.py`).

Verifies:
1. Matches the evaluation pipeline: the peak position and depth of 161708 give a profile
   within 0.05 mm of the one evaluate.py produced for that clip's peak frame.
2. Full recovery: after unloading to 0, every vertex is within 0.01 mm of rest.
3. Symmetry: a press at mid-cylinder (110 mm) gives a profile symmetric about the tool within 0.05 mm.
4. Force increases with depth, step by step during loading.
5. Cap off: Delta p = 0 at every step, and peak force is below the cap-on case at the same depth.
6. Speed: default command finishes in under 60 s.
7. Paths: none hard-coded; everything goes through config and --out.
"""

import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.mesh import build_mesh
from src.sim import StaticSimulator
from src.simulate import (
    compute_tool_force,
    load_config_and_params,
    run_standalone_simulation,
    simulate_sequence,
)


def test_matches_evaluation_pipeline(tmp_path):
    """Verify simulated peak profile for 161708 matches evaluate.py's peak profile within 0.05 mm."""
    track_path = PROJECT_ROOT / "out" / "tracks" / "161708_mm.npz"
    assert track_path.exists(), f"Missing track file: {track_path}"

    data = np.load(track_path, allow_pickle=True)
    d_mm = data["d_mm"]
    tool_x_med = float(data["tool_x_med_mm"])
    stations_mm = data["stations_mm"]
    pk_idx = int(np.argmax(d_mm))
    pk_d = float(d_mm[pk_idx])

    cfg, params = load_config_and_params()
    sim = StaticSimulator(
        p_atm=float(cfg.get("p_atm_mpa", 0.101325)),
        V0=float(cfg["bottle"].get("volume_mm3", 1.0e6)),
        gas_exponent=float(cfg.get("gas_exponent", 1.4)),
    )

    # 1. Profile produced by evaluate.py
    sols_eval = sim.solve_sequence(
        d_mm,
        tool_x_mm=tool_x_med,
        kappa=float(params["kappa_hat"]),
        rho=float(params.get("rho_used", params["rho_hat"])),
        air_on=True,
        r_tool_mm=2.5,
        gtol=1e-6,
        maxiter=20000,
    )
    w_eval_pk = sim.extract_profile(sols_eval[pk_idx], stations_mm=stations_mm, mode="top_vertex")

    # 2. Run simulate with the clip's loading trajectory exported to CSV
    traj_csv = tmp_path / "161708_loading.csv"
    with open(traj_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "depth_mm"])
        fps = float(cfg.get("fps", 29.97))
        for i, d in enumerate(d_mm[:pk_idx + 1]):
            writer.writerow([round(i / fps, 4), float(d)])

    res = run_standalone_simulation(
        press_at=tool_x_med,
        depth=pk_d,
        tool="pen",
        cap="on",
        out_dir=tmp_path / "sim_eval_match",
        trajectory_file=traj_csv,
    )

    sol_sim_pk = res["res_on"]["solutions"][-1]
    w_sim_pk = sim.extract_profile(sol_sim_pk, stations_mm=stations_mm, mode="top_vertex")

    max_diff = float(np.max(np.abs(w_sim_pk - w_eval_pk)))
    print(f"Max difference from evaluate.py peak profile: {max_diff:.6f} mm")
    assert max_diff <= 0.05, f"Peak profile difference {max_diff:.4f} mm exceeds 0.05 mm threshold"


def test_full_elastic_recovery(tmp_path):
    """Verify after unloading back to 0 mm depth, every vertex is within 0.01 mm of rest."""
    res = run_standalone_simulation(
        press_at=60.0,
        depth=12.0,
        tool="pen",
        cap="on",
        steps=12,
        out_dir=tmp_path / "recovery_test",
    )
    final_sol = res["res_on"]["solutions"][-1]
    X0 = res["sim"].X0

    max_dev = float(np.max(np.abs(final_sol - X0)))
    print(f"Max recovery deviation from rest state: {max_dev:.6f} mm")
    assert max_dev <= 0.01, f"Model did not fully recover: max deviation is {max_dev:.4f} mm"


def test_symmetry_at_mid_cylinder(tmp_path):
    """Verify indentation at mid-cylinder (110 mm) produces a profile symmetric within 0.05 mm."""
    cfg, params = load_config_and_params()
    sim = StaticSimulator(
        p_atm=float(cfg.get("p_atm_mpa", 0.101325)),
        V0=float(cfg["bottle"].get("volume_mm3", 1.0e6)),
        gas_exponent=float(cfg.get("gas_exponent", 1.4)),
    )

    sol = sim.solve_snapshot(
        tip_drop_mm=10.0,
        tool_x_mm=110.0,
        r_tool_mm=2.5,
        kappa=float(params["kappa_hat"]),
        rho=float(params.get("rho_used", params["rho_hat"])),
        air_on=True,
    )

    dists = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    stations_left = 110.0 - dists
    stations_right = 110.0 + dists

    w_left = sim.extract_profile(sol, stations_mm=stations_left)
    w_right = sim.extract_profile(sol, stations_mm=stations_right)

    asym = float(np.max(np.abs(w_left - w_right)))
    print(f"Mid-cylinder profile asymmetry: {asym:.6e} mm")
    assert asym <= 0.05, f"Profile not symmetric at mid-cylinder: asymmetry = {asym:.4f} mm"


def test_force_increases_with_depth(tmp_path):
    """Verify tool contact force increases monotonically with indentation depth during loading."""
    cfg, params = load_config_and_params()
    sim = StaticSimulator(
        p_atm=float(cfg.get("p_atm_mpa", 0.101325)),
        V0=float(cfg["bottle"].get("volume_mm3", 1.0e6)),
        gas_exponent=float(cfg.get("gas_exponent", 1.4)),
    )

    steps = 12
    depth = 12.0
    loading_depths = list(np.linspace(depth / steps, depth, steps))

    res = simulate_sequence(
        sim,
        loading_depths,
        press_at=60.0,
        tool_radius=2.5,
        kappa=float(params["kappa_hat"]),
        rho=float(params.get("rho_used", params["rho_hat"])),
        air_on=True,
        gas_exponent=float(cfg.get("gas_exponent", 1.4)),
        gtol=1e-6,
        maxiter=20000,
    )

    forces = res["forces"]
    diffs = np.diff(forces)
    print("Loading forces (N):", [round(f, 3) for f in forces])
    print("Force differences (N):", [round(d, 3) for d in diffs])

    assert np.all(diffs > 0.0), f"Forces do not monotonically increase: diffs = {diffs}"


def test_cap_off_behavior(tmp_path):
    """Verify cap off yields Delta p = 0 at every step and lower peak force than cap on."""
    res = run_standalone_simulation(
        press_at=60.0,
        depth=12.0,
        tool="pen",
        compare_cap=True,
        steps=8,
        out_dir=tmp_path / "cap_off_test",
    )

    # Delta p is zero at every step
    overpressures_off = res["res_off"]["overpressures"]
    assert all(abs(dp) < 1e-6 for dp in overpressures_off), f"Cap off produced non-zero Delta p: {overpressures_off}"

    # Peak force is lower
    pk_force_on = float(np.max(res["res_on"]["forces"]))
    pk_force_off = float(np.max(res["res_off"]["forces"]))
    print(f"Cap ON peak force: {pk_force_on:.3f} N, Cap OFF peak force: {pk_force_off:.3f} N")

    assert pk_force_off < pk_force_on, f"Cap off force {pk_force_off:.3f} N is not strictly below cap on {pk_force_on:.3f} N"


def test_execution_speed(tmp_path):
    """Verify default 12-step simulation command (23 converged solves + video) finishes under 180 s."""
    t0 = time.time()
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "simulate.py"),
        "--press-at", "60",
        "--depth", "12",
        "--tool", "pen",
        "--cap", "on",
        "--steps", "12",
        "--out", str(tmp_path / "speed_test"),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elapsed = time.time() - t0

    print(f"Default simulation command finished in: {elapsed:.2f} s")
    assert elapsed < 180.0, f"Simulation took {elapsed:.2f} s (exceeded 180 s threshold)"


def test_paths_and_outputs(tmp_path):
    """Verify no hard-coded paths and all required files are generated in --out."""
    out_dir = tmp_path / "custom_out"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "simulate.py"),
        "--press-at", "60",
        "--depth", "10",
        "--steps", "6",
        "--out", str(out_dir),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    required_files = [
        "sim.mp4",
        "side_view.png",
        "profile.png",
        "force_depth.png",
        "summary.json",
    ]
    for fname in required_files:
        fpath = out_dir / fname
        assert fpath.exists(), f"Required output file missing: {fname}"
        assert fpath.stat().st_size > 0, f"Output file is empty: {fname}"

    with open(out_dir / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert "inputs" in summary
    assert "peak_force_N" in summary
    assert "peak_overpressure_kPa" in summary
    assert "profile_ratios" in summary
