#!/usr/bin/env python3
"""M6 — Evaluation (`src/evaluate.py`): Model evaluation on held-out test clips.

Evaluates the calibrated shell model on held-out test clips:
  001711 (uncapped, pen)
  161655, 161708, 161747, 161802, 161904 (capped, pen)
  161824, 161853 (capped, finger)

Predictions:
  Quasi-static solve at each loading frame, warm-started sequentially.
Baselines:
  B0: Zero deformation (w = 0)
  B1: Normalized empirical template from fit clips of the same cap state scaled by tip drop.
Metrics:
  RMSE over loading frames and valid stations.
  Peak profile ratios (15-21 mm, 30-39 mm, 45-60 mm).
  Post-release residual.
  Stratification by drift_timing (clean/at_release vs during_loading).
  Headline aggregate covering all test clips with nothing dropped.

Outputs:
  out/eval/metrics.csv
  out/eval/<clip>_error_vs_time.png
  out/eval/<clip>_peak_profile.png
"""

import csv
import json
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sim import StaticSimulator

TEST_CLIPS = ["001711", "161655", "161708", "161747", "161802", "161904", "161824", "161853"]
FIT_CLIPS_UNCAPPED = ["001614", "001656"]
FIT_CLIPS_CAPPED = ["001531", "001730"]


def build_b1_templates(
    tracks_dir: Path = Path("out/tracks"),
) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Build empirical template functions w/tip as a function of dx = x - tool_x for capped and uncapped."""
    def _make_template(clips: List[str]) -> Callable[[np.ndarray], np.ndarray]:
        dx_all: List[float] = []
        ratio_all: List[float] = []
        for c in clips:
            p = tracks_dir / f"{c}_mm.npz"
            if not p.exists():
                continue
            data = np.load(p, allow_pickle=True)
            d_mm = data["d_mm"]
            u_mm = data["u_mm"]
            st = data["stations_mm"]
            tx = float(data["tool_x_med_mm"])
            dx = st - tx
            for k in range(len(d_mm)):
                if d_mm[k] > 1.0:
                    dx_all.extend(dx)
                    ratio_all.extend(u_mm[k] / d_mm[k])

        dx_arr = np.array(dx_all, dtype=np.float64)
        r_arr = np.array(ratio_all, dtype=np.float64)
        val = ~np.isnan(r_arr)
        order = np.argsort(dx_arr[val])
        x_s = dx_arr[val][order]
        y_s = r_arr[val][order]

        x_uniq, inv = np.unique(np.round(x_s, 1), return_inverse=True)
        y_binned = np.array([np.median(y_s[inv == i]) for i in range(len(x_uniq))])

        # Interp1d with flat extension
        f_interp = interp1d(
            x_uniq,
            y_binned,
            bounds_error=False,
            fill_value=(float(y_binned[0]), float(y_binned[-1])),
        )
        return lambda dx_query: np.maximum(0.0, f_interp(dx_query))

    f_uncapped = _make_template(FIT_CLIPS_UNCAPPED)
    f_capped = _make_template(FIT_CLIPS_CAPPED)
    return f_uncapped, f_capped


def evaluate_clip(
    clip: str,
    sim: StaticSimulator,
    rho: float,
    kappa: float,
    b1_template: Callable[[np.ndarray], np.ndarray],
    clip_cfg: Dict[str, Any],
    tracks_dir: Path = Path("out/tracks"),
    out_dir: Path = Path("out/eval"),
) -> Dict[str, Any]:
    """Evaluate model and baselines on a single test clip."""
    out_dir.mkdir(parents=True, exist_ok=True)
    track_path = tracks_dir / f"{clip}_mm.npz"
    data = np.load(track_path, allow_pickle=True)

    d_mm = data["d_mm"]
    u_mm = data["u_mm"]
    stations_mm = data["stations_mm"]
    tool_x_med = float(data["tool_x_med_mm"])
    s = float(data["s"])
    flags = data["flags"].item() if hasattr(data["flags"], "item") else data["flags"]
    drift_timing = flags.get("drift_timing", "none")

    is_cap = bool(clip_cfg.get("cap", False))
    tool_type = str(clip_cfg.get("tool", "pen"))
    camera_moved = bool(clip_cfg.get("camera_moved", False))

    r_tool = 8.0 if tool_type == "finger" else 2.5
    excl_mm = 15.0 if tool_type == "finger" else 8.0

    # 1. Model prediction: static solve at each loading frame
    air_on = is_cap
    eff_kappa = kappa if air_on else 1.0
    sols = sim.solve_sequence(
        d_mm,
        tool_x_mm=tool_x_med,
        kappa=eff_kappa,
        rho=rho,
        air_on=air_on,
        r_tool_mm=r_tool,
        gtol=1e-4,
        maxiter=120,
    )
    w_sim_seq = np.array([sim.extract_profile(sol, stations_mm=stations_mm, mode="top_vertex") for sol in sols])

    # 2. Baseline B1 prediction
    dx_st = stations_mm - tool_x_med
    template_shape = b1_template(dx_st)
    w_b1_seq = np.array([float(d) * template_shape for d in d_mm])

    # 3. Valid stations mask
    dist = np.abs(stations_mm - tool_x_med)
    valid_st_mask = dist >= excl_mm

    # 4. Overall RMSE calculation
    val_grid = valid_st_mask[None, :] & (~np.isnan(u_mm))
    err_model = w_sim_seq - u_mm
    err_b0 = 0.0 - u_mm
    err_b1 = w_b1_seq - u_mm

    rmse_model = float(np.sqrt(np.mean(err_model[val_grid] ** 2)))
    rmse_b0 = float(np.sqrt(np.mean(err_b0[val_grid] ** 2)))
    rmse_b1 = float(np.sqrt(np.mean(err_b1[val_grid] ** 2)))

    # Frame-by-frame RMSE for error_vs_time plot
    frame_rmse_model = []
    frame_rmse_b1 = []
    for k in range(len(d_mm)):
        f_val = valid_st_mask & (~np.isnan(u_mm[k]))
        if np.any(f_val):
            frame_rmse_model.append(np.sqrt(np.mean(err_model[k, f_val] ** 2)))
            frame_rmse_b1.append(np.sqrt(np.mean(err_b1[k, f_val] ** 2)))
        else:
            frame_rmse_model.append(0.0)
            frame_rmse_b1.append(0.0)

    # 5. Peak frame ratios
    k_pk = int(np.argmax(d_mm))
    pk_d = float(d_mm[k_pk])
    obs_pk = u_mm[k_pk]
    sim_pk = w_sim_seq[k_pk]

    m15 = (dist >= 15.0) & (dist <= 21.0) & (~np.isnan(obs_pk))
    m30 = (dist >= 30.0) & (dist <= 39.0) & (~np.isnan(obs_pk))
    m45 = (dist >= 45.0) & (dist <= 60.0) & (~np.isnan(obs_pk))

    r15_data = float(np.nanmean(obs_pk[m15]) / pk_d) if np.any(m15) else np.nan
    r15_sim = float(np.nanmean(sim_pk[m15]) / pk_d) if np.any(m15) else np.nan

    if camera_moved or clip == "161853":
        # Skip mid/far metrics on 161853 (camera moved)
        r30_data = np.nan
        r30_sim = np.nan
        r45_data = np.nan
        r45_sim = np.nan
    else:
        r30_data = float(np.nanmean(obs_pk[m30]) / pk_d) if np.any(m30) else np.nan
        r30_sim = float(np.nanmean(sim_pk[m30]) / pk_d) if np.any(m30) else np.nan
        r45_data = float(np.nanmean(obs_pk[m45]) / pk_d) if np.any(m45) else np.nan
        r45_sim = float(np.nanmean(sim_pk[m45]) / pk_d) if np.any(m45) else np.nan

    # 6. Post-release residual
    res_px = float(flags.get("signed_residual_px", flags.get("residual_px", 0.0)))
    res_mm = float(res_px / s)

    # 7. Generate plots
    # (a) Error vs time plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(d_mm, frame_rmse_model, "o-", color="crimson", label=f"Model (mean RMSE = {rmse_model:.2f} mm)")
    ax.plot(d_mm, frame_rmse_b1, "s--", color="gray", label=f"B1 Template (mean RMSE = {rmse_b1:.2f} mm)")
    ax.set_xlabel("Tip drop depth $d(t)$ (mm)")
    ax.set_ylabel("Spatial RMSE across valid stations (mm)")
    ax.set_title(f"Clip {clip}: Error vs Loading Depth ({'Capped' if is_cap else 'Uncapped'}, {tool_type})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{clip}_error_vs_time.png", dpi=150)
    plt.close(fig)

    # (b) Peak profile plot: Data, Model, B1 on same axes
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(stations_mm, obs_pk, "o-", color="black", label=f"Data (peak d = {pk_d:.2f} mm)")
    ax.plot(stations_mm, sim_pk, "s-", color="crimson", label="Model")
    ax.plot(stations_mm, w_b1_seq[k_pk], "^--", color="dodgerblue", label="B1 Template")
    ax.axvline(tool_x_med, color="purple", linestyle=":", label=f"Tool $x = {tool_x_med:.1f}$ mm")
    ax.axvspan(tool_x_med - excl_mm, tool_x_med + excl_mm, color="gray", alpha=0.15, label="Tool exclusion zone")
    ax.set_xlabel("Station $x$ along bottle (mm)")
    ax.set_ylabel("Drop displacement $w$ (mm)")
    ax.set_title(f"Clip {clip} Peak Profile Comparison ({'Capped' if is_cap else 'Uncapped'}, {tool_type})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{clip}_peak_profile.png", dpi=150)
    plt.close(fig)

    return {
        "clip": clip,
        "cap": "on" if is_cap else "off",
        "tool": tool_type,
        "drift_timing": drift_timing,
        "rmse_model_mm": rmse_model,
        "rmse_b0_mm": rmse_b0,
        "rmse_b1_mm": rmse_b1,
        "ratio_15_21_data": r15_data,
        "ratio_15_21_model": r15_sim,
        "ratio_30_39_data": r30_data,
        "ratio_30_39_model": r30_sim,
        "ratio_45_60_data": r45_data,
        "ratio_45_60_model": r45_sim,
        "residual_mm": res_mm,
    }


def run_evaluation(
    params_path: Path = Path("out/fit/params.json"),
    clips_cfg_path: Path = Path("config/clips.yaml"),
    tracks_dir: Path = Path("out/tracks"),
    out_dir: Path = Path("out/eval"),
) -> List[Dict[str, Any]]:
    """Run evaluation over all held-out test clips and generate metrics.csv and plots."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if not params_path.exists():
        raise FileNotFoundError(f"Missing params file: {params_path}")
    with open(params_path) as f:
        params = json.load(f)

    with open(clips_cfg_path) as f:
        cfg = yaml.safe_load(f)

    rho = float(params["rho_hat"])
    kappa = float(params["kappa_hat"])
    print(f"Loaded calibrated parameters: rho_hat = {rho:.4f}, kappa_hat = {kappa:.4f} N/mm")

    f_uncap, f_cap = build_b1_templates(tracks_dir)
    sim = StaticSimulator()

    results: List[Dict[str, Any]] = []
    print("\n--- Running Evaluation on Test Clips ---")
    for clip in TEST_CLIPS:
        c_cfg = cfg["clips"][clip]
        is_cap = bool(c_cfg.get("cap", False))
        b1_fun = f_cap if is_cap else f_uncap

        res = evaluate_clip(
            clip,
            sim,
            rho=rho,
            kappa=kappa,
            b1_template=b1_fun,
            clip_cfg=c_cfg,
            tracks_dir=tracks_dir,
            out_dir=out_dir,
        )
        results.append(res)
        print(f"  {clip} ({'cap' if is_cap else 'uncap'}, {c_cfg['tool']}): RMSE model={res['rmse_model_mm']:.2f} mm, B0={res['rmse_b0_mm']:.2f} mm, B1={res['rmse_b1_mm']:.2f} mm, drift={res['drift_timing']}")

    # =========================================================================
    # Aggregation & Stratification
    # =========================================================================
    # 1. Headline aggregate over ALL test clips (nothing dropped)
    headline_rmse_model = float(np.mean([r["rmse_model_mm"] for r in results]))
    headline_rmse_b0 = float(np.mean([r["rmse_b0_mm"] for r in results]))
    headline_rmse_b1 = float(np.mean([r["rmse_b1_mm"] for r in results]))

    # 2. Stratification by drift_timing:
    # Group A: Clean or at_release ('none', 'at_release')
    group_clean = [r for r in results if r["drift_timing"] in ["none", "at_release"]]
    # Group B: During loading ('during_loading')
    group_drift = [r for r in results if r["drift_timing"] == "during_loading"]

    clean_rmse_model = float(np.mean([r["rmse_model_mm"] for r in group_clean]))
    clean_rmse_b0 = float(np.mean([r["rmse_b0_mm"] for r in group_clean]))
    clean_rmse_b1 = float(np.mean([r["rmse_b1_mm"] for r in group_clean]))

    drift_rmse_model = float(np.mean([r["rmse_model_mm"] for r in group_drift]))
    drift_rmse_b0 = float(np.mean([r["rmse_b0_mm"] for r in group_drift]))
    drift_rmse_b1 = float(np.mean([r["rmse_b1_mm"] for r in group_drift]))

    # 3. Cap effect summary (observed vs model-predicted difference in 15-21 and 30-39 ratios)
    # Between capped pen test clips and uncapped pen test clip (001711)
    uncapped_test = [r for r in results if r["cap"] == "off" and r["tool"] == "pen"][0]
    capped_pen_test = [r for r in results if r["cap"] == "on" and r["tool"] == "pen"]

    cap_obs_15 = float(np.nanmean([r["ratio_15_21_data"] for r in capped_pen_test])) - uncapped_test["ratio_15_21_data"]
    cap_sim_15 = float(np.nanmean([r["ratio_15_21_model"] for r in capped_pen_test])) - uncapped_test["ratio_15_21_model"]

    cap_obs_30 = float(np.nanmean([r["ratio_30_39_data"] for r in capped_pen_test])) - uncapped_test["ratio_30_39_data"]
    cap_sim_30 = float(np.nanmean([r["ratio_30_39_model"] for r in capped_pen_test])) - uncapped_test["ratio_30_39_model"]

    print("\n--- Aggregate & Stratified Summary ---")
    print(f"HEADLINE (all {len(results)} test clips): RMSE model={headline_rmse_model:.2f} mm, B0={headline_rmse_b0:.2f} mm, B1={headline_rmse_b1:.2f} mm")
    print(f"STRATIFIED Clean/At-Release ({len(group_clean)} clips): RMSE model={clean_rmse_model:.2f} mm, B0={clean_rmse_b0:.2f} mm, B1={clean_rmse_b1:.2f} mm")
    print(f"STRATIFIED During Loading ({len(group_drift)} clips): RMSE model={drift_rmse_model:.2f} mm, B0={drift_rmse_b0:.2f} mm, B1={drift_rmse_b1:.2f} mm")
    print(f"CAP EFFECT 15-21mm: Obs diff = {cap_obs_15:.3f}, Model diff = {cap_sim_15:.3f}")
    print(f"CAP EFFECT 30-39mm: Obs diff = {cap_obs_30:.3f}, Model diff = {cap_sim_30:.3f}")

    # =========================================================================
    # Write metrics.csv
    # =========================================================================
    csv_path = out_dir / "metrics.csv"
    fieldnames = [
        "clip",
        "cap",
        "tool",
        "drift_timing",
        "rmse_model_mm",
        "rmse_b0_mm",
        "rmse_b1_mm",
        "ratio_15_21_data",
        "ratio_15_21_model",
        "ratio_30_39_data",
        "ratio_30_39_model",
        "ratio_45_60_data",
        "ratio_45_60_model",
        "residual_mm",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Individual test clip rows
        for r in results:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else ("" if isinstance(v, float) and np.isnan(v) else v)) for k, v in r.items()})

        # Headline aggregate row
        writer.writerow({
            "clip": "HEADLINE_ALL_TEST_CLIPS",
            "cap": "all",
            "tool": "all",
            "drift_timing": "all",
            "rmse_model_mm": f"{headline_rmse_model:.4f}",
            "rmse_b0_mm": f"{headline_rmse_b0:.4f}",
            "rmse_b1_mm": f"{headline_rmse_b1:.4f}",
            "ratio_15_21_data": "",
            "ratio_15_21_model": "",
            "ratio_30_39_data": "",
            "ratio_30_39_model": "",
            "ratio_45_60_data": "",
            "ratio_45_60_model": "",
            "residual_mm": "",
        })

        # Stratified clean or at release row
        writer.writerow({
            "clip": "STRATIFIED_CLEAN_OR_AT_RELEASE",
            "cap": "stratified",
            "tool": "stratified",
            "drift_timing": "none_or_at_release",
            "rmse_model_mm": f"{clean_rmse_model:.4f}",
            "rmse_b0_mm": f"{clean_rmse_b0:.4f}",
            "rmse_b1_mm": f"{clean_rmse_b1:.4f}",
            "ratio_15_21_data": "",
            "ratio_15_21_model": "",
            "ratio_30_39_data": "",
            "ratio_30_39_model": "",
            "ratio_45_60_data": "",
            "ratio_45_60_model": "",
            "residual_mm": "",
        })

        # Stratified during loading row
        writer.writerow({
            "clip": "STRATIFIED_DURING_LOADING",
            "cap": "stratified",
            "tool": "stratified",
            "drift_timing": "during_loading",
            "rmse_model_mm": f"{drift_rmse_model:.4f}",
            "rmse_b0_mm": f"{drift_rmse_b0:.4f}",
            "rmse_b1_mm": f"{drift_rmse_b1:.4f}",
            "ratio_15_21_data": "",
            "ratio_15_21_model": "",
            "ratio_30_39_data": "",
            "ratio_30_39_model": "",
            "ratio_45_60_data": "",
            "ratio_45_60_model": "",
            "residual_mm": "",
        })

        # Cap effect summary row
        writer.writerow({
            "clip": "CAP_EFFECT_SUMMARY",
            "cap": "delta_capped_minus_uncapped",
            "tool": "pen",
            "drift_timing": "all",
            "rmse_model_mm": "",
            "rmse_b0_mm": "",
            "rmse_b1_mm": "",
            "ratio_15_21_data": f"{cap_obs_15:.4f}",
            "ratio_15_21_model": f"{cap_sim_15:.4f}",
            "ratio_30_39_data": f"{cap_obs_30:.4f}",
            "ratio_30_39_model": f"{cap_sim_30:.4f}",
            "ratio_45_60_data": "",
            "ratio_45_60_model": "",
            "residual_mm": "",
        })

    print(f"\nWrote metrics to {csv_path}")
    return results


if __name__ == "__main__":
    run_evaluation()

