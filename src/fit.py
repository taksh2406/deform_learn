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

import csv
from concurrent.futures import ProcessPoolExecutor
import json
import os
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
DEFAULT_GTOL = 1e-6
DEFAULT_MAXITER = 20000
_WORKER_SIMS: Dict[Tuple[Any, ...], StaticSimulator] = {}


def _get_worker_sim(sim_spec: Tuple[Any, ...]) -> StaticSimulator:
    """Lazily construct and cache a StaticSimulator inside each worker process."""
    if sim_spec not in _WORKER_SIMS:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        R, L, dx, n_theta, p_atm, V0, gas_exponent, log_path = sim_spec
        mesh = build_mesh(R=R, L=L, dx=dx, n_theta=n_theta)
        _WORKER_SIMS[sim_spec] = StaticSimulator(
            mesh=mesh,
            p_atm=p_atm,
            V0=V0,
            gas_exponent=gas_exponent,
            log_path=log_path,
        )
    return _WORKER_SIMS[sim_spec]


def _worker_solve_sequence(args: Tuple[Any, ...]) -> Tuple[int, List[np.ndarray]]:
    """Worker entrypoint for depth-sequential solve of one clip at an initial grid point."""
    (sim_spec, clip_idx, depths, tx, kappa, rho, air_on, n_gas, gtol, maxiter) = args
    sim = _get_worker_sim(sim_spec)
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
    return clip_idx, sols


def _worker_solve_snapshot(args: Tuple[Any, ...]) -> Tuple[int, int, np.ndarray]:
    """Worker entrypoint for warm-started single snapshot solve across grid neighbors."""
    (sim_spec, clip_idx, snap_idx, tip_drop, tx, kappa, rho, air_on, n_gas, X_init, gtol, maxiter) = args
    sim = _get_worker_sim(sim_spec)
    sol = sim.solve_snapshot(
        tip_drop_mm=float(tip_drop),
        tool_x_mm=float(tx),
        kappa=kappa,
        rho=rho,
        air_on=air_on,
        n_gas=n_gas,
        X_init=X_init,
        gtol=gtol,
        maxiter=maxiter,
    )
    return clip_idx, snap_idx, sol


def _sim_spec(sim: StaticSimulator) -> Tuple[Any, ...]:
    """Return picklable descriptor tuple for a StaticSimulator instance."""
    return (
        sim.R,
        sim.L,
        sim.dx,
        sim.n_theta,
        sim.p_atm,
        sim.V0,
        sim.gas_exponent,
        str(sim.log_path),
    )


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


def _score_stage_solutions(
    sim: StaticSimulator,
    clip_data_list: List[Dict[str, Any]],
    sols_by_clip: List[List[np.ndarray]],
    delta: float = 0.5,
    tool_exclusion_mm: float = 8.0,
) -> Tuple[float, float, int]:
    """Compute mean Huber loss, approximate chi^2, and valid point count."""
    tot_huber = 0.0
    tot_chi2 = 0.0
    tot_pts = 0
    for cdata, sols in zip(clip_data_list, sols_by_clip):
        stations = cdata["stations_mm"]
        tx = cdata["tool_x_med_mm"]
        sigma_obs = cdata["sigma_obs"]
        obs_snaps = cdata["obs_snapshots"]
        valid_mask = np.abs(stations - tx) >= tool_exclusion_mm
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
    return mean_huber, tot_chi2, tot_pts


def _evaluate_stage_parallel(
    sim: StaticSimulator,
    clip_data_list: List[Dict[str, Any]],
    kappa: float,
    rho: float,
    air_on: bool,
    n_gas: Optional[float] = None,
    delta: float = 0.5,
    tool_exclusion_mm: float = 8.0,
    gtol: float = DEFAULT_GTOL,
    maxiter: int = DEFAULT_MAXITER,
    warm_starts: Optional[List[List[np.ndarray]]] = None,
    executor: Optional[ProcessPoolExecutor] = None,
) -> Tuple[float, float, int, List[List[np.ndarray]]]:
    """Solve all snapshots in clip_data_list (optionally warm-started and parallel) and score."""
    spec = _sim_spec(sim)
    sols_by_clip: List[List[np.ndarray]] = [
        [None] * len(cdata["sel_depths"])  # type: ignore[list-item]
        for cdata in clip_data_list
    ]

    if warm_starts is None:
        tasks = [
            (spec, c_idx, cdata["sel_depths"], cdata["tool_x_med_mm"], kappa, rho, air_on, n_gas, gtol, maxiter)
            for c_idx, cdata in enumerate(clip_data_list)
        ]
        if executor is not None and len(tasks) > 1:
            for c_idx, sols in executor.map(_worker_solve_sequence, tasks):
                sols_by_clip[c_idx] = sols
        else:
            for t in tasks:
                c_idx, sols = _worker_solve_sequence(t)
                sols_by_clip[c_idx] = sols
    else:
        tasks = []
        for c_idx, cdata in enumerate(clip_data_list):
            tx = cdata["tool_x_med_mm"]
            for s_idx, tip_drop in enumerate(cdata["sel_depths"]):
                tasks.append(
                    (
                        spec,
                        c_idx,
                        s_idx,
                        float(tip_drop),
                        float(tx),
                        kappa,
                        rho,
                        air_on,
                        n_gas,
                        warm_starts[c_idx][s_idx],
                        gtol,
                        maxiter,
                    )
                )
        if executor is not None and len(tasks) > 1:
            for c_idx, s_idx, sol in executor.map(_worker_solve_snapshot, tasks):
                sols_by_clip[c_idx][s_idx] = sol
        else:
            for t in tasks:
                c_idx, s_idx, sol = _worker_solve_snapshot(t)
                sols_by_clip[c_idx][s_idx] = sol

    mean_huber, tot_chi2, tot_pts = _score_stage_solutions(
        sim,
        clip_data_list,
        sols_by_clip,
        delta=delta,
        tool_exclusion_mm=tool_exclusion_mm,
    )
    return mean_huber, tot_chi2, tot_pts, sols_by_clip


def compute_stage_loss(
    sim: StaticSimulator,
    clip_data_list: List[Dict[str, Any]],
    kappa: float,
    rho: float,
    air_on: bool,
    n_gas: Optional[float] = None,
    delta: float = 0.5,
    tool_exclusion_mm: float = 8.0,
    gtol: float = DEFAULT_GTOL,
    maxiter: int = DEFAULT_MAXITER,
    warm_starts: Optional[List[List[np.ndarray]]] = None,
    executor: Optional[ProcessPoolExecutor] = None,
) -> Tuple[float, float]:
    """Compute mean Huber loss and approximate chi^2 over snapshots and valid stations."""
    mean_huber, tot_chi2, _, _ = _evaluate_stage_parallel(
        sim,
        clip_data_list,
        kappa=kappa,
        rho=rho,
        air_on=air_on,
        n_gas=n_gas,
        delta=delta,
        tool_exclusion_mm=tool_exclusion_mm,
        gtol=gtol,
        maxiter=maxiter,
        warm_starts=warm_starts,
        executor=executor,
    )
    return mean_huber, tot_chi2


def _scan_and_refine_1d(
    sim: StaticSimulator,
    clip_data_list: List[Dict[str, Any]],
    param_name: str,
    grid: np.ndarray,
    fixed_other: float,
    air_on: bool,
    n_gas: Optional[float] = None,
    executor: Optional[ProcessPoolExecutor] = None,
    gtol: float = DEFAULT_GTOL,
    maxiter: int = DEFAULT_MAXITER,
    refine_maxiter: int = 12,
    refine_xatol: float = 0.02,
) -> Dict[str, Any]:
    """Scan a log-spaced grid outward from the midpoint with across-grid warm starts, then refine."""
    n = len(grid)
    losses = np.zeros(n, dtype=float)
    chi2s = np.zeros(n, dtype=float)
    tot_pts = 0
    cache: Dict[float, Dict[str, Any]] = {}

    def _eval_at(val: float, warm: Optional[List[List[np.ndarray]]]) -> Dict[str, Any]:
        val = float(val)
        rho = val if param_name == "rho" else float(fixed_other)
        kappa = float(fixed_other) if param_name == "rho" else val
        if warm is None and cache:
            nearest = min(cache.keys(), key=lambda x: abs(np.log(x) - np.log(val)))
            warm = cache[nearest]["sols"]
        loss, chi2, pts, sols = _evaluate_stage_parallel(
            sim,
            clip_data_list,
            kappa=kappa,
            rho=rho,
            air_on=air_on,
            n_gas=n_gas,
            gtol=gtol,
            maxiter=maxiter,
            warm_starts=warm,
            executor=executor,
        )
        entry = {"val": val, "loss": loss, "chi2": chi2, "pts": pts, "sols": sols}
        cache[val] = entry
        return entry

    mid = n // 2
    mid_entry = _eval_at(float(grid[mid]), None)
    losses[mid] = mid_entry["loss"]
    chi2s[mid] = mid_entry["chi2"]
    tot_pts = mid_entry["pts"]
    print(f"  {param_name}={grid[mid]:.4f}: loss={losses[mid]:.4f}, chi2={chi2s[mid]:.1f}")

    for idx in range(mid + 1, n):
        e = _eval_at(float(grid[idx]), cache[float(grid[idx - 1])]["sols"])
        losses[idx] = e["loss"]
        chi2s[idx] = e["chi2"]
        print(f"  {param_name}={grid[idx]:.4f}: loss={losses[idx]:.4f}, chi2={chi2s[idx]:.1f}")

    for idx in range(mid - 1, -1, -1):
        e = _eval_at(float(grid[idx]), cache[float(grid[idx + 1])]["sols"])
        losses[idx] = e["loss"]
        chi2s[idx] = e["chi2"]
        print(f"  {param_name}={grid[idx]:.4f}: loss={losses[idx]:.4f}, chi2={chi2s[idx]:.1f}")

    i_best = int(np.argmin(chi2s))
    i_lo = max(0, i_best - 2)
    i_hi = min(n - 1, i_best + 2)
    res = minimize_scalar(
        lambda lv: _eval_at(float(np.exp(lv)), None)["chi2"],
        bounds=(float(np.log(grid[i_lo])), float(np.log(grid[i_hi]))),
        method="bounded",
        options={"xatol": refine_xatol, "maxiter": refine_maxiter},
    )
    opt_val = float(np.exp(res.x))
    opt_entry = _eval_at(opt_val, None)
    return {
        "grid": grid,
        "losses": losses,
        "chi2s": chi2s,
        "tot_pts": tot_pts,
        "dof": max(1, tot_pts - 1),
        "opt_val": opt_val,
        "opt_loss": float(opt_entry["loss"]),
        "opt_chi2": float(opt_entry["chi2"]),
        "opt_sols": opt_entry["sols"],
        "cache": cache,
    }


def _compute_rescaled_profile(scan: Dict[str, Any]) -> Dict[str, Any]:
    """Rescale Delta chi^2 by chi^2_min / dof and extract 68% and 95% regions/bounds."""
    grid = np.asarray(scan["grid"], dtype=float)
    chi2_grid = np.asarray(scan["chi2s"], dtype=float)
    dof = int(scan["dof"])
    opt_val = float(scan["opt_val"])
    opt_chi2 = float(scan["opt_chi2"])

    chi2_min = min(float(np.min(chi2_grid)), opt_chi2)
    chi2_red = chi2_min / max(1, dof)
    dchi2_raw_grid = chi2_grid - chi2_min
    dchi2_rescaled_grid = dchi2_raw_grid / max(1e-12, chi2_red)

    # Merge grid with continuous optimum so fine curve passes through Delta chi^2 = 0
    all_vals = np.append(grid, opt_val)
    all_dchi2 = np.append(dchi2_rescaled_grid, (opt_chi2 - chi2_min) / max(1e-12, chi2_red))
    order = np.argsort(all_vals)
    all_vals = all_vals[order]
    all_dchi2 = all_dchi2[order]

    fine_log = np.linspace(np.log(grid[0]), np.log(grid[-1]), 2000)
    fine_vals = np.exp(fine_log)
    fine_dchi2 = np.interp(fine_log, np.log(all_vals), all_dchi2)

    m68 = fine_dchi2 <= 1.0
    m95 = fine_dchi2 <= 4.0
    ci_68 = [float(fine_vals[m68][0]), float(fine_vals[m68][-1])]
    ci_95 = [float(fine_vals[m95][0]), float(fine_vals[m95][-1])]
    assert ci_95[0] <= ci_68[0] + 1e-12 and ci_95[1] + 1e-12 >= ci_68[1], (
        f"Inconsistent intervals: 95% {ci_95} does not contain 68% {ci_68}"
    )

    closed_low_68 = bool(dchi2_rescaled_grid[0] > 1.0)
    closed_high_68 = bool(dchi2_rescaled_grid[-1] > 1.0)
    closed_low_95 = bool(dchi2_rescaled_grid[0] > 4.0)
    closed_high_95 = bool(dchi2_rescaled_grid[-1] > 4.0)

    return {
        "chi2_min": chi2_min,
        "dof": dof,
        "chi2_red": chi2_red,
        "dchi2_raw_grid": dchi2_raw_grid,
        "dchi2_rescaled_grid": dchi2_rescaled_grid,
        "max_dchi2_rescaled": float(np.max(dchi2_rescaled_grid)),
        "fine_vals": fine_vals,
        "fine_dchi2": fine_dchi2,
        "ci_68": ci_68,
        "ci_95": ci_95,
        "closed_low_68": closed_low_68,
        "closed_high_68": closed_high_68,
        "closed_low_95": closed_low_95,
        "closed_high_95": closed_high_95,
        "interval_closed_68": bool(closed_low_68 and closed_high_68),
        "interval_closed_95": bool(closed_low_95 and closed_high_95),
    }


def _write_loss_csv(
    path: Path,
    param_name: str,
    scan: Dict[str, Any],
    prof: Dict[str, Any],
) -> None:
    """Save grid scan loss, chi^2, and rescaled Delta chi^2 table to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([param_name, "mean_huber_mm2", "chi2", "dof", "chi2_red", "delta_chi2_raw", "delta_chi2_rescaled"])
        for val, loss, ch, draw, dres in zip(
            scan["grid"],
            scan["losses"],
            scan["chi2s"],
            prof["dchi2_raw_grid"],
            prof["dchi2_rescaled_grid"],
        ):
            writer.writerow([
                f"{float(val):.6g}",
                f"{float(loss):.6f}",
                f"{float(ch):.4f}",
                int(prof["dof"]),
                f"{float(prof['chi2_red']):.6f}",
                f"{float(draw):.4f}",
                f"{float(dres):.6f}",
            ])


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


def _run_stage2_for_rho(
    sim: StaticSimulator,
    capped_data: List[Dict[str, Any]],
    rho_val: float,
    kappa_grid: np.ndarray,
    executor: Optional[ProcessPoolExecutor],
    csv_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run Stage 2 (kappa scan + refinement + rescaled CI + peak force/dp) for a fixed rho."""
    scan_k = _scan_and_refine_1d(
        sim,
        capped_data,
        "kappa",
        kappa_grid,
        fixed_other=float(rho_val),
        air_on=True,
        executor=executor,
        gtol=DEFAULT_GTOL,
        maxiter=DEFAULT_MAXITER,
        refine_maxiter=12,
        refine_xatol=0.02,
    )
    prof_k = _compute_rescaled_profile(scan_k)
    if csv_path is not None:
        _write_loss_csv(csv_path, "kappa", scan_k, prof_k)

    kappa_hat = float(scan_k["opt_val"])
    peak_forces = []
    peak_overpressures = []
    for c_idx, cdata in enumerate(capped_data):
        pk_d = float(cdata["peak_d"])
        tx = float(cdata["tool_x_med_mm"])
        if abs(float(cdata["sel_depths"][-1]) - pk_d) < 1e-9:
            sol_pk = scan_k["opt_sols"][c_idx][-1]
        else:
            sol_pk = sim.solve_snapshot(
                pk_d,
                tool_x_mm=tx,
                r_tool_mm=2.5,
                kappa=kappa_hat,
                rho=float(rho_val),
                air_on=True,
                gtol=DEFAULT_GTOL,
                maxiter=DEFAULT_MAXITER,
            )

        V_mesh = enclosed_volume(sol_pk, sim.tris)
        V_air = sim.V0 + (V_mesh - sim.V_mesh_rest)
        p = sim.p_atm * ((sim.V0 / V_air) ** sim.gas_exponent)
        dp_kPa = (p - sim.p_atm) * 1.0e3
        peak_overpressures.append(float(dp_kPa))

        fz = sim.reaction_force(
            sol_pk,
            tip_drop_mm=pk_d,
            tool_x_mm=tx,
            r_tool_mm=2.5,
            kappa=kappa_hat,
            rho=float(rho_val),
            air_on=True,
        )
        peak_forces.append(float(fz))

    return {
        "rho": float(rho_val),
        "scan": scan_k,
        "prof": prof_k,
        "kappa_hat": kappa_hat,
        "loss_stage2": float(scan_k["opt_loss"]),
        "chi2_stage2": float(scan_k["opt_chi2"]),
        "peak_force_N": float(np.mean(peak_forces)),
        "peak_overpressure_kPa": float(np.mean(peak_overpressures)),
        "peak_forces_per_clip": {c["clip"]: float(f) for c, f in zip(capped_data, peak_forces)},
        "peak_dp_per_clip": {c["clip"]: float(dp) for c, dp in zip(capped_data, peak_overpressures)},
    }


def _fast_opt_warm(
    sim: StaticSimulator,
    clip_data_list: List[Dict[str, Any]],
    param_name: str,
    fixed_other: float,
    air_on: bool,
    bounds_val: Tuple[float, float],
    n_gas: Optional[float] = None,
    executor: Optional[ProcessPoolExecutor] = None,
    init_val: Optional[float] = None,
    init_sols: Optional[List[List[np.ndarray]]] = None,
) -> Dict[str, Any]:
    """Warm-started parallel 1D optimization helper for sensitivity checks (gtol=1e-6)."""
    cache: Dict[float, Dict[str, Any]] = {}
    if init_val is not None and init_sols is not None:
        cache[float(init_val)] = {"sols": init_sols}

    def _eval(v: float) -> Dict[str, Any]:
        v = float(v)
        rho = v if param_name == "rho" else float(fixed_other)
        kappa = float(fixed_other) if param_name == "rho" else v
        warm = None
        if cache:
            nearest = min(cache.keys(), key=lambda x: abs(np.log(x) - np.log(v)))
            warm = cache[nearest]["sols"]
        loss, chi2, pts, sols = _evaluate_stage_parallel(
            sim,
            clip_data_list,
            kappa=kappa,
            rho=rho,
            air_on=air_on,
            n_gas=n_gas,
            gtol=DEFAULT_GTOL,
            maxiter=DEFAULT_MAXITER,
            warm_starts=warm,
            executor=executor,
        )
        cache[v] = {"loss": loss, "chi2": chi2, "pts": pts, "sols": sols}
        return cache[v]

    # Seed midpoint if no initial state was provided
    if not cache:
        seed_val = float(np.sqrt(bounds_val[0] * bounds_val[1]))
        _eval(seed_val)

    res = minimize_scalar(
        lambda lv: _eval(float(np.exp(lv)))["chi2"],
        bounds=(float(np.log(bounds_val[0])), float(np.log(bounds_val[1]))),
        method="bounded",
        options={"xatol": 0.05, "maxiter": 10},
    )
    opt_v = float(np.exp(res.x))
    opt_e = _eval(opt_v)
    return {"opt_val": opt_v, "opt_loss": float(opt_e["loss"]), "opt_chi2": float(opt_e["chi2"]), "opt_sols": opt_e["sols"]}


def run_fitting(
    sim: Optional[StaticSimulator] = None,
    out_dir: Path = Path("out/fit"),
    tracks_dir: Path = Path("out/tracks"),
) -> Dict[str, Any]:
    """Run full two-stage converged parameter fitting, decision rules, sensitivities, and diagnostics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if sim is None:
        sim = StaticSimulator()

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

    solve_log_path = sim.log_path
    solves_before = 0
    if solve_log_path.exists():
        with open(solve_log_path) as f:
            solves_before = max(0, sum(1 for _ in f) - 1)

    max_workers = min(8, max(1, os.cpu_count() or 1))
    print(f"Executing fitting with {max_workers} parallel workers (gtol={DEFAULT_GTOL}, maxiter={DEFAULT_MAXITER})")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # =====================================================================
        # Stage 1: Fit / test identifiability of rho on uncapped clips (air off)
        # =====================================================================
        print("\n--- Running Stage 1: rho grid scan on uncapped clips (air off) ---")
        rho_grid = np.logspace(np.log10(0.01), np.log10(100.0), 15)
        scan_rho = _scan_and_refine_1d(
            sim,
            uncapped_data,
            "rho",
            rho_grid,
            fixed_other=1.0,
            air_on=False,
            executor=executor,
            gtol=DEFAULT_GTOL,
            maxiter=DEFAULT_MAXITER,
            refine_maxiter=12,
            refine_xatol=0.02,
        )
        prof_rho = _compute_rescaled_profile(scan_rho)
        _write_loss_csv(out_dir / "loss_vs_rho.csv", "rho", scan_rho, prof_rho)

        # Pre-registered decision rule 0.5 / 3.2:
        # rho is unidentified if rescaled Delta chi^2 stays < 4 across the whole rho grid.
        rho_opt = float(scan_rho["opt_val"])
        max_rho_dchi2 = float(prof_rho["max_dchi2_rescaled"])
        rho_identified = bool(max_rho_dchi2 >= 4.0)
        if rho_identified:
            rho_used = rho_opt
            loss_stage1 = float(scan_rho["opt_loss"])
            chi2_stage1 = float(scan_rho["opt_chi2"])
            uncapped_sols_final = scan_rho["opt_sols"]
        else:
            rho_used = 1.0
            warm_1 = scan_rho["cache"].get(float(rho_grid[7]), {}).get("sols")
            loss_stage1, chi2_stage1, _, uncapped_sols_final = _evaluate_stage_parallel(
                sim,
                uncapped_data,
                kappa=1.0,
                rho=1.0,
                air_on=False,
                gtol=DEFAULT_GTOL,
                maxiter=DEFAULT_MAXITER,
                warm_starts=warm_1,
                executor=executor,
            )

        print(
            f"Stage 1 summary: rho_opt={rho_opt:.4f}, max rescaled Delta chi^2={max_rho_dchi2:.3f}, "
            f"chi2_min={prof_rho['chi2_min']:.1f}, dof={prof_rho['dof']}, chi2_red={prof_rho['chi2_red']:.3f}"
        )
        print(
            f"Pre-registered rule -> rho_identified={rho_identified}, rho_used={rho_used:.4f}, "
            f"Stage 1 loss={loss_stage1:.4f} mm^2 ({loss_stage1/b0_stage1*100:.1f}% of B0)"
        )

        # Plot loss vs rho (both Huber loss and rescaled Delta chi^2)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        ax1.semilogx(rho_grid, scan_rho["losses"], "o-", label="15-point converged scan")
        ax1.axvline(rho_used, color="crimson", linestyle="--", label=f"$\\rho_{{\\mathrm{{used}}}} = {rho_used:.3f}$")
        if abs(rho_opt - rho_used) > 1e-3:
            ax1.axvline(rho_opt, color="navy", linestyle=":", label=f"$\\rho_{{\\mathrm{{opt}}}} = {rho_opt:.3f}$")
        ax1.axhline(0.5 * b0_stage1, color="gray", linestyle=":", label="50% B0 threshold")
        ax1.set_xlabel("$\\rho$ (axial-to-hoop stiffness ratio)")
        ax1.set_ylabel("Mean Huber loss (mm$^2$)")
        ax1.set_title("Stage 1: Huber Loss vs $\\rho$ (Uncapped, Air Off)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.semilogx(rho_grid, prof_rho["dchi2_rescaled_grid"], "s-", color="purple", label="Rescaled $\\Delta \\chi^2$")
        ax2.axhline(4.0, color="red", linestyle="--", label="Identifiability threshold ($\\Delta\\chi^2 = 4$)")
        ax2.axhline(1.0, color="orange", linestyle=":", label="68% threshold ($\\Delta\\chi^2 = 1$)")
        ax2.set_xlabel("$\\rho$ (axial-to-hoop stiffness ratio)")
        ax2.set_ylabel("Rescaled $\\Delta \\chi^2$ ($\\Delta\\chi^2 / \\chi^2_{\\mathrm{red}}$)")
        ax2.set_title(f"Stage 1 Identifiability (max = {max_rho_dchi2:.2f})")
        ax2.set_ylim(-0.2, max(5.0, max_rho_dchi2 * 1.15))
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        fig.tight_layout()
        fig.savefig(out_dir / "loss_vs_rho.png", dpi=150)
        plt.close(fig)

        # =====================================================================
        # Stage 2: Fit kappa on capped clips (air on, rho=rho_used)
        # =====================================================================
        print(f"\n--- Running Stage 2: Fit kappa on capped clips (air on, rho={rho_used:.3f}) ---")
        kappa_grid = np.logspace(np.log10(0.01), np.log10(30.0), 15)
        s2_main = _run_stage2_for_rho(
            sim,
            capped_data,
            rho_val=rho_used,
            kappa_grid=kappa_grid,
            executor=executor,
            csv_path=out_dir / "loss_vs_kappa.csv",
        )
        scan_k = s2_main["scan"]
        prof_k = s2_main["prof"]
        kappa_hat = s2_main["kappa_hat"]
        loss_stage2 = s2_main["loss_stage2"]
        ci_68 = prof_k["ci_68"]
        ci_95 = prof_k["ci_95"]
        avg_pk_force = s2_main["peak_force_N"]
        avg_pk_dp = s2_main["peak_overpressure_kPa"]

        print(
            f"Optimal kappa_hat = {kappa_hat:.4f} N/mm, Stage 2 loss = {loss_stage2:.4f} mm^2 "
            f"({loss_stage2/b0_stage2*100:.1f}% of B0), chi2_red = {prof_k['chi2_red']:.3f}"
        )
        print(
            f"Rescaled 68% kappa region: [{ci_68[0]:.3f}, {ci_68[1]:.3f}] "
            f"(closed_low={prof_k['closed_low_68']}, closed_high={prof_k['closed_high_68']})"
        )
        print(
            f"Rescaled 95% kappa region: [{ci_95[0]:.3f}, {ci_95[1]:.3f}] "
            f"(closed_low={prof_k['closed_low_95']}, closed_high={prof_k['closed_high_95']}, "
            f"interval_closed={prof_k['interval_closed_95']})"
        )
        print(f"Predicted peak tool force F: {avg_pk_force:.2f} N (held reaction force)")
        print(f"Predicted peak overpressure Delta p: {avg_pk_dp:.3f} kPa")

        # Plot loss vs kappa (Huber + rescaled Delta chi^2)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        ax1.semilogx(kappa_grid, scan_k["losses"], "o-", label="15-point converged scan")
        ax1.axvline(kappa_hat, color="crimson", linestyle="--", label=f"$\\hat{{\\kappa}} = {kappa_hat:.3f}$ N/mm")
        ax1.axhline(0.5 * b0_stage2, color="gray", linestyle=":", label="50% B0 threshold")
        ax1.set_xlabel("$\\kappa$ (elastic stiffness scale N/mm)")
        ax1.set_ylabel("Mean Huber loss (mm$^2$)")
        ax1.set_title(f"Stage 2: Huber Loss vs $\\kappa$ ($\\rho={rho_used:.2g}$, Air On)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.semilogx(kappa_grid, prof_k["dchi2_rescaled_grid"], "s-", color="purple", label="Rescaled $\\Delta \\chi^2$")
        ax2.axhline(1.0, color="orange", linestyle="--", label="68% ($\\Delta \\chi^2 = 1$)")
        ax2.axhline(4.0, color="red", linestyle=":", label="95% ($\\Delta \\chi^2 = 4$)")
        ax2.axvline(kappa_hat, color="crimson", linestyle="--")
        ax2.set_xlabel("$\\kappa$ (N/mm)")
        ax2.set_ylabel("Rescaled $\\Delta \\chi^2$")
        ax2.set_title("Rescaled Profile Likelihood on $\\kappa$")
        ax2.set_ylim(-0.3, min(25.0, max(6.0, float(np.max(prof_k["dchi2_rescaled_grid"])) * 1.1)))
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        fig.tight_layout()
        fig.savefig(out_dir / "loss_vs_kappa.png", dpi=150)
        plt.close(fig)

        # =====================================================================
        # Phase 3.5: Sensitivity across rho = {0.5, 1.0, 7.0}
        # =====================================================================
        sens_rho: Dict[str, Any] = {}
        sens_targets = [r_s for r_s in [0.5, 1.0, 7.0] if abs(r_s - rho_used) / max(1.0, rho_used) > 0.02]
        if sens_targets:
            print(f"\n--- Stage 2 Sensitivity Runs across rho = {sens_targets} ---")
            for r_s in sens_targets:
                tag = str(r_s).replace(".", "_")
                s2_s = _run_stage2_for_rho(
                    sim,
                    capped_data,
                    rho_val=r_s,
                    kappa_grid=kappa_grid,
                    executor=executor,
                    csv_path=out_dir / f"loss_vs_kappa_rho_{tag}.csv",
                )
                print(
                    f"  rho={r_s}: kappa_hat={s2_s['kappa_hat']:.4f} N/mm, "
                    f"95%=[{s2_s['prof']['ci_95'][0]:.3f}, {s2_s['prof']['ci_95'][1]:.3f}] "
                    f"(closed_95={s2_s['prof']['interval_closed_95']}), "
                    f"loss={s2_s['loss_stage2']:.4f}, F={s2_s['peak_force_N']:.2f} N, dp={s2_s['peak_overpressure_kPa']:.3f} kPa"
                )
                sens_rho[f"rho_{tag}"] = {
                    "rho": r_s,
                    "kappa_hat": s2_s["kappa_hat"],
                    "loss_stage2": s2_s["loss_stage2"],
                    "chi2_min": s2_s["prof"]["chi2_min"],
                    "chi2_red": s2_s["prof"]["chi2_red"],
                    "kappa_ci_68": s2_s["prof"]["ci_68"],
                    "kappa_ci_95": s2_s["prof"]["ci_95"],
                    "kappa_interval_closed_95": s2_s["prof"]["interval_closed_95"],
                    "kappa_closed_low_95": s2_s["prof"]["closed_low_95"],
                    "kappa_closed_high_95": s2_s["prof"]["closed_high_95"],
                    "peak_force_N": s2_s["peak_force_N"],
                    "peak_overpressure_kPa": s2_s["peak_overpressure_kPa"],
                }

        # =====================================================================
        # Diagnostics (reusing converged solutions — zero extra solves)
        # =====================================================================
        print("\n--- Computing Diagnostics ---")
        air_est = model_free_air_estimate(capped_data, R=sim.R, V0=sim.V0, p_atm=sim.p_atm)
        print("Model-free air estimates (Delta p):")
        for clip, vals in air_est.items():
            print(f"  {clip}: oval = {vals['dp_oval_kPa']:.3f} kPa, chord = {vals['dp_chord_kPa']:.3f} kPa vs Sim = {avg_pk_dp:.3f} kPa")

        all_fit_data = [
            ("001531", "Capped (001531)", True, capped_data[0], scan_k["opt_sols"][0]),
            ("001730", "Capped (001730)", True, capped_data[1], scan_k["opt_sols"][1]),
            ("001614", "Uncapped (001614)", False, uncapped_data[0], uncapped_sols_final[0]),
            ("001656", "Uncapped (001656)", False, uncapped_data[1], uncapped_sols_final[1]),
        ]
        colors = {"001531": "tab:blue", "001730": "tab:cyan", "001614": "tab:red", "001656": "tab:orange"}

        fig, ax = plt.subplots(figsize=(8, 5))
        for cid, label, is_cap, cdata, sols in all_fit_data:
            st = cdata["stations_mm"]
            tx = cdata["tool_x_med_mm"]
            depths = cdata["sel_depths"]
            obs_snaps = cdata["obs_snapshots"]
            dist = np.abs(st - tx)
            mask_30_39 = (dist >= 30.0) & (dist <= 39.0)

            data_ratios = [float(np.nanmean(obs[mask_30_39]) / d) for obs, d in zip(obs_snaps, depths)]
            model_ratios = [
                float(np.nanmean(sim.extract_profile(sol, stations_mm=st, mode="top_vertex")[mask_30_39]) / d)
                for sol, d in zip(sols, depths)
            ]

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

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        ax_map = {"001614": axes[0, 0], "001656": axes[0, 1], "001531": axes[1, 0], "001730": axes[1, 1]}
        for cid, label, is_cap, cdata, sols in all_fit_data:
            ax = ax_map[cid]
            st = cdata["stations_mm"]
            tx = cdata["tool_x_med_mm"]
            obs_snaps = cdata["obs_snapshots"]
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

        # =====================================================================
        # Phase 3.6 & Gas sensitivity (gtol=1e-6, maxiter=20000, warm-started)
        # =====================================================================
        print("\n--- Sensitivity Runs (converged, gtol=1e-6) ---")
        sens_L: Dict[str, Any] = {}
        for L_val in [200.0, 240.0]:
            mesh_L = build_mesh(R=sim.R, L=L_val, dx=sim.dx, n_theta=sim.n_theta)
            sim_L = StaticSimulator(
                mesh=mesh_L,
                p_atm=sim.p_atm,
                V0=sim.V0,
                gas_exponent=sim.gas_exponent,
                log_path=sim.log_path,
            )
            opt_r_L = _fast_opt_warm(
                sim_L,
                uncapped_data,
                "rho",
                fixed_other=1.0,
                air_on=False,
                bounds_val=(0.1, 10.0),
                executor=executor,
            )
            r_L = opt_r_L["opt_val"]
            opt_k_at_rL = _fast_opt_warm(
                sim_L,
                capped_data,
                "kappa",
                fixed_other=r_L,
                air_on=True,
                bounds_val=(0.05, 30.0),
                executor=executor,
            )
            opt_k_at_rho_used = _fast_opt_warm(
                sim_L,
                capped_data,
                "kappa",
                fixed_other=rho_used,
                air_on=True,
                bounds_val=(0.05, 30.0),
                executor=executor,
            )
            print(
                f"  L = {L_val:.0f} mm: rho_opt = {r_L:.3f} (kappa={opt_k_at_rL['opt_val']:.3f}), "
                f"at rho={rho_used:.1f}: kappa={opt_k_at_rho_used['opt_val']:.3f} N/mm"
            )
            sens_L[f"L_{int(L_val)}"] = {
                "rho_opt": r_L,
                "loss_stage1_at_rho_opt": opt_r_L["opt_loss"],
                "kappa_at_rho_opt": opt_k_at_rL["opt_val"],
                "kappa_at_rho_used": opt_k_at_rho_used["opt_val"],
                "loss_stage2_at_rho_used": opt_k_at_rho_used["opt_loss"],
            }

        opt_k_n1 = _fast_opt_warm(
            sim,
            capped_data,
            "kappa",
            fixed_other=rho_used,
            air_on=True,
            bounds_val=(0.05, 30.0),
            n_gas=1.0,
            executor=executor,
            init_val=kappa_hat,
            init_sols=scan_k["opt_sols"],
        )
        k_n1 = opt_k_n1["opt_val"]
        print(f"  n = 1.0 (isothermal): kappa = {k_n1:.3f} N/mm (vs n=1.4: {kappa_hat:.3f})")

    # Verify zero unrecovered failed solves added during fitting (attempt 2 or 3 recovery clears pending failure)
    n_solves_fit = 0
    n_retries_fit = 0
    n_unrecovered = 0
    zero_failed = True
    if solve_log_path.exists():
        with open(solve_log_path) as f:
            rows = list(csv.DictReader(f))
        new_rows = rows[solves_before:]
        n_solves_fit = len(new_rows)
        pending_failures: Dict[Tuple[str, ...], int] = {}
        for r in new_rows:
            ok_r = r["success"] == "True" and float(r["max_grad_free"]) < 1e-4
            att = int(r["attempt"])
            key = (r["caller"], r["tip_drop_mm"], r["tool_x_mm"], r["kappa"], r["rho"], r["air_on"], r["n_gas"])
            if att == 1:
                if not ok_r:
                    pending_failures[key] = pending_failures.get(key, 0) + 1
            else:
                if att == 2:
                    n_retries_fit += 1
                if ok_r:
                    pending_failures[key] = max(0, pending_failures.get(key, 0) - 1)
        n_unrecovered = sum(pending_failures.values())
        zero_failed = n_unrecovered == 0
        assert zero_failed, f"G3.1 failed: {n_unrecovered} unrecovered failed solves in solve_log.csv!"

    params_data = {
        "solver": {
            "gtol": DEFAULT_GTOL,
            "maxiter": DEFAULT_MAXITER,
            "max_grad_tol_N": 1e-4,
            "converged": zero_failed,
            "contact": "held",
            "n_solves_fit": n_solves_fit,
            "n_retries_fit": n_retries_fit,
            "zero_failed_solves": zero_failed,
        },
        "rho_identified": rho_identified,
        "rho_opt_stage1": rho_opt,
        "rho_used": rho_used,
        "rho_hat": rho_used,
        "rho_max_delta_chi2_rescaled": max_rho_dchi2,
        "rho_chi2_min": prof_rho["chi2_min"],
        "rho_dof": prof_rho["dof"],
        "rho_chi2_red": prof_rho["chi2_red"],
        "kappa_hat": kappa_hat,
        "loss_stage1": loss_stage1,
        "loss_stage1_opt": float(scan_rho["opt_loss"]),
        "loss_stage2": loss_stage2,
        "b0_loss_stage1": b0_stage1,
        "b0_loss_stage2": b0_stage2,
        "kappa_chi2_min": prof_k["chi2_min"],
        "kappa_dof": prof_k["dof"],
        "kappa_chi2_red": prof_k["chi2_red"],
        "kappa_ci_68": ci_68,
        "kappa_ci_95": ci_95,
        "kappa_interval_closed_68": prof_k["interval_closed_68"],
        "kappa_interval_closed_95": prof_k["interval_closed_95"],
        "kappa_closed_low_68": prof_k["closed_low_68"],
        "kappa_closed_high_68": prof_k["closed_high_68"],
        "kappa_closed_low_95": prof_k["closed_low_95"],
        "kappa_closed_high_95": prof_k["closed_high_95"],
        "peak_force_N": avg_pk_force,
        "peak_overpressure_kPa": avg_pk_dp,
        "peak_forces_per_clip": s2_main["peak_forces_per_clip"],
        "peak_dp_per_clip": s2_main["peak_dp_per_clip"],
        "model_free_air_estimate": air_est,
        "sensitivity_rho": sens_rho,
        "sensitivity_length": sens_L,
        "sensitivity_gas_exponent_n1": {"kappa_n1": k_n1, "kappa_ratio": k_n1 / kappa_hat},
    }
    with open(out_dir / "params.json", "w") as f:
        json.dump(params_data, f, indent=2)
    print(f"\nWrote parameters to {out_dir / 'params.json'}")

    return params_data


if __name__ == "__main__":
    run_fitting()
