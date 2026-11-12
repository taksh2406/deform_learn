#!/usr/bin/env python3
"""Tests for M5 — Fitting (`src/fit.py`).

Verifies:
1. Synthetic data recovery with air off: rho recovered within 15%, loss flat in kappa within 1%.
2. Synthetic data recovery with air on: kappa recovered within 30%.
3. Real fit Stage 1 and Stage 2 losses are below 50% of baseline B0.
4. Output parameters and plots exist and are non-empty.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.fit import huber_loss
from src.sim import StaticSimulator


def test_synthetic_air_off_recovery_and_invariance():
    """Verify rho is recovered within 15% from noisy synthetic profiles, and loss is flat in kappa."""
    sim = StaticSimulator()
    true_rho = 1.4
    tx = 92.0
    depths = [3.0, 6.0, 9.0, 12.0]
    stations = np.arange(10, 115, 3)
    mask = np.abs(stations - tx) >= 8.0

    # Generate synthetic profiles at true_rho, air_on=False
    sols_true = sim.solve_sequence(depths, tool_x_mm=tx, kappa=1.0, rho=true_rho, air_on=False)
    rng = np.random.default_rng(42)
    synth_obs = [
        sim.extract_profile(sol, stations_mm=stations, mode="top_vertex") + rng.normal(0, 0.05, size=len(stations))
        for sol in sols_true
    ]

    def synth_loss_rho(rho: float) -> float:
        sols = sim.solve_sequence(depths, tool_x_mm=tx, kappa=1.0, rho=rho, air_on=False)
        return sum(
            np.sum(huber_loss((sim.extract_profile(s, stations_mm=stations, mode="top_vertex") - w)[mask]))
            for s, w in zip(sols, synth_obs)
        )

    res = minimize_scalar(synth_loss_rho, bounds=(0.5, 3.0), method="bounded", options={"xatol": 0.05})
    recovered_rho = float(res.x)
    rel_err = abs(recovered_rho - true_rho) / true_rho
    print(f"Air-off recovery: true={true_rho}, rec={recovered_rho:.3f}, rel_err={rel_err*100:.1f}%")
    assert rel_err <= 0.15, f"Expected relative error <= 15%, got {rel_err*100:.2f}%"

    # Exact / within 1% scale invariance in kappa
    loss_k1 = synth_loss_rho(recovered_rho)
    sols_k3 = sim.solve_sequence(depths, tool_x_mm=tx, kappa=3.0, rho=recovered_rho, air_on=False)
    loss_k3 = sum(
        np.sum(huber_loss((sim.extract_profile(s, stations_mm=stations, mode="top_vertex") - w)[mask]))
        for s, w in zip(sols_k3, synth_obs)
    )
    rel_diff_k = abs(loss_k1 - loss_k3) / max(1e-12, loss_k1)
    assert rel_diff_k <= 0.01, f"Loss across kappa not flat within 1%: {rel_diff_k*100:.4f}%"


def test_synthetic_air_on_recovery():
    """Verify kappa is recovered within 30% from noisy synthetic profiles with air on."""
    sim = StaticSimulator()
    true_rho = 1.4
    true_kappa = 1.5
    tx = 92.0
    depths = [3.0, 6.0, 9.0, 12.0]
    stations = np.arange(10, 115, 3)
    mask = np.abs(stations - tx) >= 8.0

    # Generate synthetic profiles at true_kappa, true_rho, air_on=True
    sols_true = sim.solve_sequence(depths, tool_x_mm=tx, kappa=true_kappa, rho=true_rho, air_on=True)
    rng = np.random.default_rng(42)
    synth_obs = [
        sim.extract_profile(sol, stations_mm=stations, mode="top_vertex") + rng.normal(0, 0.05, size=len(stations))
        for sol in sols_true
    ]

    def synth_loss_kappa(log_k: float) -> float:
        k = float(np.exp(log_k))
        sols = sim.solve_sequence(depths, tool_x_mm=tx, kappa=k, rho=true_rho, air_on=True)
        return sum(
            np.sum(huber_loss((sim.extract_profile(s, stations_mm=stations, mode="top_vertex") - w)[mask]))
            for s, w in zip(sols, synth_obs)
        )

    res = minimize_scalar(synth_loss_kappa, bounds=(np.log(0.3), np.log(5.0)), method="bounded", options={"xatol": 0.05})
    recovered_k = float(np.exp(res.x))
    rel_err_k = abs(recovered_k - true_kappa) / true_kappa
    print(f"Air-on recovery: true={true_kappa}, rec={recovered_k:.3f}, rel_err={rel_err_k*100:.1f}%")
    assert rel_err_k <= 0.30, f"Expected relative error <= 30%, got {rel_err_k*100:.2f}%"


def test_stage1_and_stage2_losses_below_50_percent_of_b0():
    """Verify Stage 1 and Stage 2 model losses are below 50% of baseline B0 on fit clips."""
    params_file = Path("out/fit/params.json")
    assert params_file.exists(), "params.json missing"

    with open(params_file) as f:
        params = json.load(f)

    l1 = float(params["loss_stage1"])
    b0_1 = float(params["b0_loss_stage1"])
    l2 = float(params["loss_stage2"])
    b0_2 = float(params["b0_loss_stage2"])

    assert l1 < 0.5 * b0_1, f"Stage 1 loss {l1:.4f} not < 50% of B0 {b0_1:.4f} (ratio: {l1/b0_1*100:.1f}%)"
    assert l2 < 0.5 * b0_2, f"Stage 2 loss {l2:.4f} not < 50% of B0 {b0_2:.4f} (ratio: {l2/b0_2*100:.1f}%)"


def test_fit_params_and_plots_written():
    """Verify all required fitting deliverables, solver block, and interval consistency exist."""
    fit_dir = Path("out/fit")
    assert (fit_dir / "params.json").exists()
    assert (fit_dir / "loss_vs_rho.csv").stat().st_size > 100
    assert (fit_dir / "loss_vs_kappa.csv").stat().st_size > 100
    assert (fit_dir / "loss_vs_rho.png").stat().st_size > 1000
    assert (fit_dir / "loss_vs_kappa.png").stat().st_size > 1000
    assert (fit_dir / "depth_fingerprint.png").stat().st_size > 1000
    assert (fit_dir / "fit_profiles.png").stat().st_size > 1000

    with open(fit_dir / "params.json") as f:
        data = json.load(f)
    for key in [
        "solver",
        "rho_identified",
        "rho_used",
        "rho_hat",
        "kappa_hat",
        "loss_stage1",
        "loss_stage2",
        "kappa_ci_68",
        "kappa_ci_95",
        "peak_force_N",
        "peak_overpressure_kPa",
    ]:
        assert key in data, f"Key {key} missing from params.json"
    assert data["solver"]["gtol"] == 1e-6
    assert data["solver"]["converged"] is True
    assert data["solver"]["contact"] == "held"
    ci_68 = data["kappa_ci_68"]
    ci_95 = data["kappa_ci_95"]
    assert ci_95[0] <= ci_68[0] and ci_95[1] >= ci_68[1]

