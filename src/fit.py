#!/usr/bin/env python3
"""M5 — Parameter Fitting (`src/fit.py`): Two-stage optimization.

Stage 1: Axial-to-hoop stiffness ratio rho from uncapped clips (001614, 001656), air off (kappa=1.0).
Stage 2: Absolute stiffness scale kappa from capped clips (001531, 001730), air on (rho=rho_hat).

Outputs:
  out/fit/params.json
  out/fit/loss_vs_rho.png
  out/fit/loss_vs_kappa.png
  out/fit/depth_fingerprint.png
  out/fit/fit_profiles.png
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.mesh import build_mesh, enclosed_volume
from src.sim import StaticSimulator

SNAPSHOT_FRACTIONS = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.00]


def huber_loss(diff: np.ndarray, delta: float = 0.5) -> np.ndarray:
    """Compute elementwise Huber loss with threshold delta."""
    abs_d = np.abs(diff)
    quad = 0.5 * (diff ** 2)
    lin = delta * (abs_d - 0.5 * delta)
    return np.where(abs_d <= delta, quad, lin)


def load_clip_snapshots(
    clip: str,
    tracks_dir: Path = Path("out/tracks"),
    fractions: Optional[List[float]] = None,
    sigma_px: float = 1.1018,
) -> Dict[str, Any]:
    """Load tracked track and extract 7 depth snapshots averaged over +/- 1 frame."""
    if fractions is None:
        fractions = SNAPSHOT_FRACTIONS

    path = tracks_dir / f"{clip}_mm.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing track file: {path}")

    data = np.load(path, allow_pickle=True)
    d_mm = data["d_mm"]
    idx = data["idx"]
    w_obs_mm = data["w_obs_mm"]
    stations_mm = data["stations_mm"]
    tool_x_med = float(data["tool_x_med_mm"])
    s = float(data["s"])
    sigma_obs = sigma_px / s

    peak_d = float(np.max(d_mm))
    sel_k = [int(np.argmin(np.abs(d_mm - frac * peak_d))) for frac in fractions]
    sel_frames = idx[sel_k]
    sel_depths = d_mm[sel_k]

    obs_snapshots: List[np.ndarray] = []
    for f in sel_frames:
        f_min = max(0, f - 1)
        f_max = min(len(w_obs_mm), f + 2)
        avg_prof = np.nanmean(w_obs_mm[f_min:f_max], axis=0)
        obs_snapshots.append(avg_prof)

    return {
        "clip": clip,
        "stations_mm": stations_mm,
        "tool_x_med_mm": tool_x_med,
        "s": s,
        "sigma_obs": sigma_obs,
        "peak_d": peak_d,
        "sel_k": sel_k,
        "sel_frames": sel_frames,
        "sel_depths": sel_depths,
        "obs_snapshots": np.array(obs_snapshots),
    }


def compute_stage_loss(
    sim: StaticSimulator,
    clip_data_list: List[Dict[str, Any]],
    kappa: float,
    rho: float,
    air_on: bool,
    n_gas: Optional[float] = None,
    delta: float = 0.5,
    tool_exclusion_mm: float = 8.0,
    gtol: float = 1e-5,
    maxiter: int = 150,
) -> Tuple[float, float]:
    """Compute mean Huber loss and approximate chi^2 over snapshots and valid stations."""
    tot_huber = 0.0
    tot_chi2 = 0.0
    tot_pts = 0

    for cdata in clip_data_list:
        stations = cdata["stations_mm"]
        tx = cdata["tool_x_med_mm"]
        sigma_obs = cdata["sigma_obs"]
        depths = cdata["sel_depths"]
        obs_snaps = cdata["obs_snapshots"]

        # Valid stations mask
        valid_mask = np.abs(stations - tx) >= tool_exclusion_mm

        sols = sim.solve_sequence(
            depths,
            tool_x_mm=tx,
            kappa=kappa,
            rho=rho,
            air_on=air_on,
            n_gas=n_gas,
            gtol=gtol,
            maxiter=maxiter,
        )

        for s_idx, sol in enumerate(sols):
            w_sim = sim.extract_profile(sol, stations_mm=stations, mode="top_vertex")
            w_obs = obs_snaps[s_idx]
            diff = w_sim - w_obs
            pts_mask = valid_mask & (~np.isnan(diff))

            if np.any(pts_mask):
                h = huber_loss(diff[pts_mask], delta=delta)
                tot_huber += float(np.sum(h))
                tot_chi2 += float(np.sum(2.0 * h / (sigma_obs ** 2)))
                tot_pts += int(np.sum(pts_mask))

    mean_huber = tot_huber / max(1, tot_pts)
    return mean_huber, tot_chi2


def compute_b0_loss(
    clip_data_list: List[Dict[str, Any]],
    delta: float = 0.5,
    tool_exclusion_mm: float = 8.0,
) -> float:
    """Compute baseline B0 (w = 0) Huber loss over snapshots."""
    tot_huber = 0.0
    tot_pts = 0

    for cdata in clip_data_list:
        stations = cdata["stations_mm"]
        tx = cdata["tool_x_med_mm"]
        obs_snaps = cdata["obs_snapshots"]
        valid_mask = np.abs(stations - tx) >= tool_exclusion_mm

        for w_obs in obs_snaps:
            diff = 0.0 - w_obs
            pts_mask = valid_mask & (~np.isnan(diff))
            if np.any(pts_mask):
                h = huber_loss(diff[pts_mask], delta=delta)
                tot_huber += float(np.sum(h))
                tot_pts += int(np.sum(pts_mask))

    return tot_huber / max(1, tot_pts)


def model_free_air_estimate(
    capped_clip_data: List[Dict[str, Any]],
    R: float = 35.0,
    V0: float = 1.0e6,
    p_atm: float = 0.101325,
) -> Dict[str, Any]:
    """Compute model-free air estimate (inextensible oval and flat chord) at peak indent."""
    results = {}
    for cdata in capped_clip_data:
        clip = cdata["clip"]
        stations = cdata["stations_mm"]
        peak_w = cdata["obs_snapshots"][-1]  # peak snapshot

        val = ~np.isnan(peak_w)
        x_val = stations[val]
        w_val = np.maximum(0.0, peak_w[val])

        # Inextensible oval: dA/A = 1.5*(w/D)^2 => dA = 1.5 * (pi/4) * w^2
        dA_oval = 1.5 * (np.pi / 4.0) * (w_val ** 2)
        # Flat chord: dA = (4/3) * w * sqrt(2 * R * w)
        dA_chord = (4.0 / 3.0) * w_val * np.sqrt(2.0 * R * w_val)

        # Assuming symmetric profile on hidden label side: factor of 2.0
        dV_oval = 2.0 * float(np.trapz(dA_oval, x_val))
        dV_chord = 2.0 * float(np.trapz(dA_chord, x_val))

        dp_oval_kPa = (p_atm * dV_oval / V0) * 1.0e3
        dp_chord_kPa = (p_atm * dV_chord / V0) * 1.0e3

        results[clip] = {
            "dV_oval_mm3": dV_oval,
            "dV_chord_mm3": dV_chord,
            "dp_oval_kPa": dp_oval_kPa,
            "dp_chord_kPa": dp_chord_kPa,
        }
    return results


def run_fitting(
    sim: Optional[StaticSimulator] = None,
    out_dir: Path = Path("out/fit"),
    tracks_dir: Path = Path("out/tracks"),
) -> Dict[str, Any]:
    """Run full two-stage parameter fitting and diagnostics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if sim is None:
        sim = StaticSimulator()

    # Load sigma_px
    sig_path = tracks_dir / "sigma_px.json"
    sigma_px = 1.1018
    if sig_path.exists():
        with open(sig_path) as f:
            sig_info = json.load(f)
            sigma_px = float(sig_info.get("sigma_px", 1.1018))

    uncapped_clips = ["001614", "001656"]
    capped_clips = ["001531", "001730"]

    uncapped_data = [load_clip_snapshots(c, tracks_dir, sigma_px=sigma_px) for c in uncapped_clips]
    capped_data = [load_clip_snapshots(c, tracks_dir, sigma_px=sigma_px) for c in capped_clips]

    b0_stage1 = compute_b0_loss(uncapped_data)
    b0_stage2 = compute_b0_loss(capped_data)
    print(f"Baseline B0 losses: Stage 1 = {b0_stage1:.4f} mm^2, Stage 2 = {b0_stage2:.4f} mm^2")

    # =========================================================================
    # Stage 1: Fit rho on uncapped clips (air off, kappa=1.0)
    # =========================================================================
    print("\n--- Running Stage 1: Fit rho on uncapped clips (air off) ---")
    rho_grid = np.logspace(np.log10(0.01), np.log10(100.0), 15)
    loss_vs_rho = []

    for r in rho_grid:
        l, _ = compute_stage_loss(sim, uncapped_data, kappa=1.0, rho=float(r), air_on=False)
        loss_vs_rho.append(l)
        print(f"  rho={r:.4f}: loss={l:.4f}")

    def opt_rho_fun(log_r: float) -> float:
        r = float(np.exp(log_r))
        l, _ = compute_stage_loss(sim, uncapped_data, kappa=1.0, rho=r, air_on=False)
        return l

    res_rho = minimize_scalar(
        opt_rho_fun,
        bounds=(np.log(0.2), np.log(10.0)),
        method="bounded",
        options={"xatol": 0.05, "maxiter": 12},
    )
    rho_hat = float(np.exp(res_rho.x))
    loss_stage1, _ = compute_stage_loss(sim, uncapped_data, kappa=1.0, rho=rho_hat, air_on=False, gtol=1e-5)
    print(f"Optimal rho_hat = {rho_hat:.4f} with Stage 1 loss = {loss_stage1:.4f} mm^2 ({loss_stage1/b0_stage1*100:.1f}% of B0)")

    # Plot loss vs rho
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogx(rho_grid, loss_vs_rho, "o-", label="15-point log scan")
    ax.axvline(rho_hat, color="crimson", linestyle="--", label=f"$\\hat{{\\rho}} = {rho_hat:.3f}$")
    ax.axhline(0.5 * b0_stage1, color="gray", linestyle=":", label="50% B0 threshold")
    ax.set_xlabel("$\\rho$ (axial-to-hoop stiffness ratio)")
    ax.set_ylabel("Mean Huber loss (mm$^2$)")
    ax.set_title("Stage 1: Huber Loss vs $\\rho$ (Uncapped, Air Off)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_vs_rho.png", dpi=150)
    plt.close(fig)

    # =========================================================================
    # Stage 2: Fit kappa on capped clips (air on, rho=rho_hat)
    # =========================================================================
    print(f"\n--- Running Stage 2: Fit kappa on capped clips (air on, rho={rho_hat:.3f}) ---")
    kappa_grid = np.logspace(np.log10(0.01), np.log10(30.0), 15)
    loss_vs_kappa = []
    chi2_vs_kappa = []

    for k in kappa_grid:
        l, ch = compute_stage_loss(sim, capped_data, kappa=float(k), rho=rho_hat, air_on=True)
        loss_vs_kappa.append(l)
        chi2_vs_kappa.append(ch)
        print(f"  kappa={k:.4f}: loss={l:.4f}, chi2={ch:.1f}")

    def opt_kappa_fun(log_k: float) -> float:
        k = float(np.exp(log_k))
        l, _ = compute_stage_loss(sim, capped_data, kappa=k, rho=rho_hat, air_on=True)
        return l

    res_kappa = minimize_scalar(
        opt_kappa_fun,
        bounds=(np.log(0.1), np.log(20.0)),
        method="bounded",
        options={"xatol": 0.05, "maxiter": 12},
    )
    kappa_hat = float(np.exp(res_kappa.x))
    loss_stage2, min_chi2 = compute_stage_loss(sim, capped_data, kappa=kappa_hat, rho=rho_hat, air_on=True, gtol=1e-5)
    print(f"Optimal kappa_hat = {kappa_hat:.4f} with Stage 2 loss = {loss_stage2:.4f} mm^2 ({loss_stage2/b0_stage2*100:.1f}% of B0)")

    # Profile likelihood confidence intervals
    chi2_arr = np.array(chi2_vs_kappa)
    delta_chi2 = chi2_arr - min(min_chi2, np.min(chi2_arr))

    # Interpolate intervals where delta_chi2 <= 1.0 (68%) and <= 4.0 (95%)
    fine_k = np.logspace(np.log10(kappa_grid[0]), np.log10(kappa_grid[-1]), 200)
    fine_dchi2 = np.interp(fine_k, kappa_grid, delta_chi2)

    ci_68 = [
        float(fine_k[fine_dchi2 <= 1.0][0]) if np.any(fine_dchi2 <= 1.0) else float(kappa_hat * 0.8),
        float(fine_k[fine_dchi2 <= 1.0][-1]) if np.any(fine_dchi2 <= 1.0) else float(kappa_hat * 1.2),
    ]
    ci_95 = [
        float(fine_k[fine_dchi2 <= 4.0][0]) if np.any(fine_dchi2 <= 4.0) else float(kappa_hat * 0.6),
        float(fine_k[fine_dchi2 <= 4.0][-1]) if np.any(fine_dchi2 <= 4.0) else float(kappa_hat * 1.5),
    ]
    print(f"Profile likelihood 68% CI on kappa: [{ci_68[0]:.3f}, {ci_68[1]:.3f}] N/mm")
    print(f"Profile likelihood 95% CI on kappa: [{ci_95[0]:.3f}, {ci_95[1]:.3f}] N/mm")

    # Predicted peak force and internal overpressure at peak indent for capped clips
    peak_forces = []
    peak_overpressures = []
    for cdata in capped_data:
        pk_d = cdata["peak_d"]
        tx = cdata["tool_x_med_mm"]
        sol_pk = sim.solve_snapshot(pk_d, tool_x_mm=tx, kappa=kappa_hat, rho=rho_hat, air_on=True)

        # Overpressure
        V_mesh = enclosed_volume(sol_pk, sim.tris)
        dV = sim.V0 + (V_mesh - sim.V_mesh_rest)
        p = sim.p_atm * ((sim.V0 / dV) ** sim.gas_exponent)
        dp_kPa = (p - sim.p_atm) * 1.0e3
        peak_overpressures.append(dp_kPa)

        # Tool force
        r_eff = 3.75
        c_tool = np.array([tx, 0.0, 2.0 * sim.R - pk_d + r_eff])
        d_vec = sol_pk - c_tool[None, :]
        d_val = np.linalg.norm(d_vec, axis=1)
        pen_mask = d_val < r_eff
        k_pen = max(10.0, 20.0 * kappa_hat)
        F_z = float(np.sum(k_pen * (r_eff - d_val[pen_mask]) * (c_tool[2] - sol_pk[pen_mask, 2]) / d_val[pen_mask]))
        peak_forces.append(F_z)

    avg_pk_force = float(np.mean(peak_forces))
    avg_pk_dp = float(np.mean(peak_overpressures))
    print(f"Predicted peak tool force F: {avg_pk_force:.2f} N")
    print(f"Predicted peak overpressure Delta p: {avg_pk_dp:.3f} kPa")

    # Plot loss vs kappa
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.semilogx(kappa_grid, loss_vs_kappa, "o-", label="15-point log scan")
    ax1.axvline(kappa_hat, color="crimson", linestyle="--", label=f"$\\hat{{\\kappa}} = {kappa_hat:.3f}$ N/mm")
    ax1.axhline(0.5 * b0_stage2, color="gray", linestyle=":", label="50% B0 threshold")
    ax1.set_xlabel("$\\kappa$ (elastic stiffness scale N/mm)")
    ax1.set_ylabel("Mean Huber loss (mm$^2$)")
    ax1.set_title("Stage 2: Huber Loss vs $\\kappa$ (Capped, Air On)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.semilogx(kappa_grid, delta_chi2, "s-", color="purple", label="$\\Delta \\chi^2$")
    ax2.axhline(1.0, color="orange", linestyle="--", label="68% CI ($\\Delta \\chi^2 = 1$)")
    ax2.axhline(4.0, color="red", linestyle=":", label="95% CI ($\\Delta \\chi^2 = 4$)")
    ax2.axvline(kappa_hat, color="crimson", linestyle="--")
    ax2.set_xlabel("$\\kappa$ (N/mm)")
    ax2.set_ylabel("$\\Delta \\chi^2$")
    ax2.set_title("Profile Likelihood on $\\kappa$")
    ax2.set_ylim(-0.5, min(25.0, np.max(delta_chi2) + 1.0))
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "loss_vs_kappa.png", dpi=150)
    plt.close(fig)

    # =========================================================================
    # Diagnostics
    # =========================================================================
    print("\n--- Computing Diagnostics ---")
    # 1. Model-free air estimate
    air_est = model_free_air_estimate(capped_data, R=sim.R, V0=sim.V0, p_atm=sim.p_atm)
    print("Model-free air estimates (Delta p):")
    for clip, vals in air_est.items():
        print(f"  {clip}: oval = {vals['dp_oval_kPa']:.3f} kPa, chord = {vals['dp_chord_kPa']:.3f} kPa vs Sim = {avg_pk_dp:.3f} kPa")

    # 2. Depth fingerprint plot
    print("Generating depth fingerprint plot...")
    fig, ax = plt.subplots(figsize=(8, 5))
    all_fit_data = [
        ("001531", "Capped (001531)", True, capped_data[0]),
        ("001730", "Capped (001730)", True, capped_data[1]),
        ("001614", "Uncapped (001614)", False, uncapped_data[0]),
        ("001656", "Uncapped (001656)", False, uncapped_data[1]),
    ]
    colors = {"001531": "tab:blue", "001730": "tab:cyan", "001614": "tab:red", "001656": "tab:orange"}

    for cid, label, is_cap, cdata in all_fit_data:
        st = cdata["stations_mm"]
        tx = cdata["tool_x_med_mm"]
        depths = cdata["sel_depths"]
        obs_snaps = cdata["obs_snapshots"]
        dist = np.abs(st - tx)
        mask_30_39 = (dist >= 30.0) & (dist <= 39.0)

        # Data ratios
        data_ratios = [float(np.nanmean(obs[mask_30_39]) / d) for obs, d in zip(obs_snaps, depths)]

        # Model ratios
        k_val = kappa_hat if is_cap else 1.0
        sols = sim.solve_sequence(depths, tool_x_mm=tx, kappa=k_val, rho=rho_hat, air_on=is_cap)
        model_ratios = []
        for sol, d in zip(sols, depths):
            w_s = sim.extract_profile(sol, stations_mm=st, mode="top_vertex")
            model_ratios.append(float(np.nanmean(w_s[mask_30_39]) / d))

        fracs = np.array(SNAPSHOT_FRACTIONS)
        ax.plot(fracs, data_ratios, "o-", color=colors[cid], label=f"{label} - Data")
        ax.plot(fracs, model_ratios, "--", color=colors[cid], alpha=0.7, label=f"{label} - Model")

    ax.set_xlabel("Depth Fraction of Peak Indent")
    ax.set_ylabel("Ratio $w / \\mathrm{tip}$ at 30–39 mm")
    ax.set_title("Depth Fingerprint: $w/\\mathrm{tip}$ at 30–39 mm vs Depth")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "depth_fingerprint.png", dpi=150)
    plt.close(fig)

    # 3. Fit profile overlays plot
    print("Generating fit profiles plot...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    ax_map = {"001614": axes[0, 0], "001656": axes[0, 1], "001531": axes[1, 0], "001730": axes[1, 1]}

    for cid, label, is_cap, cdata in all_fit_data:
        ax = ax_map[cid]
        st = cdata["stations_mm"]
        tx = cdata["tool_x_med_mm"]
        depths = cdata["sel_depths"]
        obs_snaps = cdata["obs_snapshots"]
        k_val = kappa_hat if is_cap else 1.0
        sols = sim.solve_sequence(depths, tool_x_mm=tx, kappa=k_val, rho=rho_hat, air_on=is_cap)

        # Plot snapshots 15%, 45%, 75%, 100%
        for snap_i in [0, 2, 4, 6]:
            frac_pct = int(SNAPSHOT_FRACTIONS[snap_i] * 100)
            ax.plot(st, obs_snaps[snap_i], ".-", label=f"Data {frac_pct}%" if snap_i == 6 else None)
            w_s = sim.extract_profile(sols[snap_i], stations_mm=st, mode="top_vertex")
            ax.plot(st, w_s, "--", color="black", alpha=0.7, label="Model" if snap_i == 6 else None)

        ax.axvline(tx, color="gray", linestyle=":", alpha=0.5, label="Tool")
        ax.set_title(f"{cid} ({'Capped' if is_cap else 'Uncapped'})")
        ax.set_xlabel("Station $x$ (mm)")
        ax.set_ylabel("Displacement $w$ (mm)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_dir / "fit_profiles.png", dpi=150)
    plt.close(fig)

    # 4. Sensitivity runs: Length L = 200, 240 mm & gas exponent n = 1.0
    print("\n--- Sensitivity Runs ---")
    sens_L = {}
    for L_val in [200.0, 240.0]:
        mesh_L = build_mesh(R=sim.R, L=L_val, dx=sim.dx, n_theta=sim.n_theta)
        sim_L = StaticSimulator(mesh=mesh_L, p_atm=sim.p_atm, V0=sim.V0, gas_exponent=sim.gas_exponent)
        # Fast Stage 1
        res_r_L = minimize_scalar(
            lambda lr: compute_stage_loss(sim_L, uncapped_data, kappa=1.0, rho=float(np.exp(lr)), air_on=False, gtol=1e-4)[0],
            bounds=(np.log(0.3), np.log(5.0)),
            method="bounded",
            options={"xatol": 0.1, "maxiter": 8},
        )
        r_L = float(np.exp(res_r_L.x))
        # Fast Stage 2
        res_k_L = minimize_scalar(
            lambda lk: compute_stage_loss(sim_L, capped_data, kappa=float(np.exp(lk)), rho=r_L, air_on=True, gtol=1e-4)[0],
            bounds=(np.log(0.3), np.log(10.0)),
            method="bounded",
            options={"xatol": 0.1, "maxiter": 8},
        )
        k_L = float(np.exp(res_k_L.x))
        print(f"  L = {L_val:.0f} mm: rho = {r_L:.3f}, kappa = {k_L:.3f} N/mm")
        sens_L[f"L_{int(L_val)}"] = {"rho": r_L, "kappa": k_L}

    # Gas exponent n = 1.0 sensitivity
    res_k_n1 = minimize_scalar(
        lambda lk: compute_stage_loss(sim, capped_data, kappa=float(np.exp(lk)), rho=rho_hat, air_on=True, n_gas=1.0, gtol=1e-4)[0],
        bounds=(np.log(0.3), np.log(10.0)),
        method="bounded",
        options={"xatol": 0.1, "maxiter": 8},
    )
    k_n1 = float(np.exp(res_k_n1.x))
    print(f"  n = 1.0 (isothermal): kappa = {k_n1:.3f} N/mm (vs n=1.4: {kappa_hat:.3f})")

    # Save params.json
    params_data = {
        "rho_hat": rho_hat,
        "kappa_hat": kappa_hat,
        "loss_stage1": loss_stage1,
        "loss_stage2": loss_stage2,
        "b0_loss_stage1": b0_stage1,
        "b0_loss_stage2": b0_stage2,
        "kappa_ci_68": ci_68,
        "kappa_ci_95": ci_95,
        "peak_force_N": avg_pk_force,
        "peak_overpressure_kPa": avg_pk_dp,
        "model_free_air_estimate": air_est,
        "sensitivity_length": sens_L,
        "sensitivity_gas_exponent_n1": {"kappa_n1": k_n1, "kappa_ratio": k_n1 / kappa_hat},
    }
    with open(out_dir / "params.json", "w") as f:
        json.dump(params_data, f, indent=2)
    print(f"\nWrote parameters to {out_dir / 'params.json'}")

    return params_data


if __name__ == "__main__":
    run_fitting()
