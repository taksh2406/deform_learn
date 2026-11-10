#!/usr/bin/env python3
"""M7 — Demo Video Pipeline (`src/demo.py`).

Produces the required demonstration videos and concatenates them with title cards:
1. `out/demo/fresh_instance_161747.mp4` (257 frames) & `out/demo/fresh_instance_001711.mp4` (221 frames):
   - Left: video crop with observed edge (green) and predicted edge (red: y = rest_px + w_sim * s).
   - Middle: cylinder mesh in Matplotlib 3D, fixed view, colored by displacement.
   - Right: spatial RMSE vs time with moving cursor.
2. `out/demo/fresh_instance.mp4`:
   - Concatenation of the two fresh instance test clips (161747 and 001711).
3. `out/demo/cap_effect.mp4` (279 frames):
   - 001531 (capped) and 001656 (uncapped) side by side with predicted edges overlaid.
   - Below them: normalized peak profiles (data and model, capped and uncapped on the same axes).
4. `out/demo/all.mp4`:
   - Concatenates the fresh instance videos and cap effect video with high-contrast title cards.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

# Ensure Matplotlib uses writable temp directory
os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sim import StaticSimulator


def load_inputs(
    params_path: Path = Path("out/fit/params.json"),
    clips_cfg_path: Path = Path("config/clips.yaml"),
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not params_path.exists():
        raise FileNotFoundError(f"Missing parameters file: {params_path}")
    if not clips_cfg_path.exists():
        raise FileNotFoundError(f"Missing clips config file: {clips_cfg_path}")

    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)
    with open(clips_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return params, cfg


def precompute_simulation_grid(
    sim: StaticSimulator,
    max_d_mm: float,
    tool_x_mm: float,
    kappa: float,
    rho: float,
    air_on: bool,
    r_tool_mm: float = 2.5,
    n_pts: int = 25,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """Solve static equilibrium on an increasing depth grid warm-started sequentially."""
    grid_d = np.linspace(0.0, max(0.5, float(max_d_mm)), n_pts)
    eff_kappa = float(kappa) if air_on else 1.0
    sols = sim.solve_sequence(
        grid_d,
        tool_x_mm=float(tool_x_mm),
        kappa=eff_kappa,
        rho=float(rho),
        air_on=air_on,
        r_tool_mm=float(r_tool_mm),
        gtol=1e-4,
        maxiter=120,
    )
    # Extract top profile on ring_x for each solution
    profs = np.array([sim.extract_profile(s, stations_mm=sim.ring_x, mode="top_vertex") for s in sols])
    return grid_d, sols, profs


def interpolate_state(
    d_val: float,
    grid_d: np.ndarray,
    sols: List[np.ndarray],
    profs: np.ndarray,
    sim: StaticSimulator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate deformed mesh vertex positions and ring profile for a given depth."""
    if d_val <= 0.0:
        return sim.X0.copy(), np.zeros(sim.n_rings, dtype=np.float32)

    d_clamped = float(np.clip(d_val, grid_d[0], grid_d[-1]))
    idx = int(np.searchsorted(grid_d, d_clamped))
    if idx == 0:
        return sols[0].copy(), profs[0].copy()
    if idx >= len(grid_d):
        return sols[-1].copy(), profs[-1].copy()

    d0, d1 = grid_d[idx - 1], grid_d[idx]
    alpha = (d_clamped - d0) / max(1e-6, (d1 - d0))

    X_interp = (1.0 - alpha) * sols[idx - 1] + alpha * sols[idx]
    prof_interp = (1.0 - alpha) * profs[idx - 1] + alpha * profs[idx]
    return X_interp, prof_interp


def create_title_card(
    title: str,
    subtitle: str,
    bullets: Optional[List[str]] = None,
    badge: Optional[str] = None,
    width: int = 1920,
    height: int = 720,
) -> np.ndarray:
    """Render a crisp, high-contrast title card frame."""
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor("#0b0f19")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Decorative header line
    ax.plot([0.15, 0.85], [0.85, 0.85], color="#38bdf8", lw=3)

    if badge:
        ax.text(
            0.5,
            0.89,
            badge.upper(),
            fontsize=13,
            fontweight="bold",
            color="#38bdf8",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e293b", edgecolor="#38bdf8", lw=1.5),
        )

    # Title
    ax.text(
        0.5,
        0.73,
        title,
        fontsize=28,
        fontweight="bold",
        color="#f8fafc",
        ha="center",
        va="center",
    )

    # Subtitle
    ax.text(
        0.5,
        0.61,
        subtitle,
        fontsize=18,
        color="#94a3b8",
        ha="center",
        va="center",
    )

    # Bullet points
    if bullets:
        y = 0.48
        for b in bullets:
            ax.text(
                0.5,
                y,
                f"•   {b}",
                fontsize=15,
                color="#e2e8f0",
                ha="center",
                va="center",
            )
            y -= 0.08

    # Bottom accent
    ax.plot([0.3, 0.7], [0.12, 0.12], color="#475569", lw=1)

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return cv2.resize(img, (width, height))


def convert_to_h264(input_path: Path, output_path: Path, crf: int = 20) -> None:
    """Re-encode video to browser-compatible H.264 (yuv420p) using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-preset",
        "medium",
        str(output_path),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        # Fallback: keep the input file if ffmpeg fails
        print(f"Warning: ffmpeg conversion failed, keeping raw video: {res.stderr.decode('utf-8')[:200]}")
        if input_path != output_path:
            input_path.rename(output_path)
    else:
        if input_path != output_path and input_path.exists():
            input_path.unlink()


def render_fresh_instance_clip(
    clip_id: str,
    out_path: Path,
    sim: StaticSimulator,
    rho: float,
    kappa: float,
    cfg: Dict[str, Any],
    tracks_dir: Path = Path("out/tracks"),
    fps: float = 29.97,
) -> Path:
    """Render 3-panel demonstration video for a single fresh instance test clip."""
    print(f"Rendering Fresh Instance Demo for Clip {clip_id}...")
    c_cfg = cfg["clips"][clip_id]
    is_cap = bool(c_cfg.get("cap", False))
    tool_type = str(c_cfg.get("tool", "pen"))
    r_tool = float(cfg.get("finger_radius_mm", 8.0) if tool_type == "finger" else cfg.get("pen_radius_mm", 2.5))
    excl_mm = float(cfg["tool_exclusion_mm"][tool_type])

    # Load tracks
    tr = np.load(tracks_dir / f"{clip_id}.npz", allow_pickle=True)
    tr_mm = np.load(tracks_dir / f"{clip_id}_mm.npz", allow_pickle=True)

    edge_px = tr["edge_px"]
    rest_px = tr["rest_px"]
    tip_px = tr["tip_px"]
    tool_x_px = tr["tool_x_px"]
    contact = tr["contact"]

    s = float(tr_mm["s"])
    x_base_px = float(tr_mm["x_base_px"])
    stations_mm = tr_mm["stations_mm"]
    tool_x_med_mm = float(tr_mm["tool_x_med_mm"])
    tip_drop_mm = tr_mm["tip_drop_mm"]

    n_frames = edge_px.shape[0]
    bx0, bx1 = int(c_cfg["body_x"][0]), int(c_cfg["body_x"][1])

    # Pre-solve simulation depth grid
    c_idx = np.where(contact)[0]
    c_start = int(c_idx[0]) if len(c_idx) > 0 else 0
    c_end = int(c_idx[-1]) if len(c_idx) > 0 else n_frames - 1
    t_peak = int(c_start + np.nanargmax(tip_drop_mm[c_start : c_end + 1])) if len(c_idx) > 0 else n_frames // 2

    max_d = float(np.nanmax(tip_drop_mm[c_idx])) if len(c_idx) > 0 else 5.0
    grid_d, sols, profs = precompute_simulation_grid(
        sim,
        max_d_mm=max_d,
        tool_x_mm=tool_x_med_mm,
        kappa=kappa,
        rho=rho,
        air_on=is_cap,
        r_tool_mm=r_tool,
        n_pts=25,
    )

    # Precalculate station coordinates in pixels
    st_px = np.round(x_base_px - stations_mm * s).astype(np.int32)
    valid_st_dist = np.abs(stations_mm - tool_x_med_mm) >= excl_mm

    # Precalculate per-frame model profile and RMSE
    all_w_sim = []
    frame_rmse = np.zeros(n_frames, dtype=np.float32)

    # Columns corresponding to stations along bottle
    cols_x = np.arange(bx0, bx1)
    cols_x_mm = (x_base_px - cols_x) / s

    for t in range(n_frames):
        in_c = bool(contact[t])
        d_val = float(tip_drop_mm[t]) if in_c and not np.isnan(tip_drop_mm[t]) else 0.0
        d_val = max(0.0, d_val)

        X_t, prof_t = interpolate_state(d_val, grid_d, sols, profs, sim)
        # Interpolate onto columns
        w_cols = np.interp(cols_x_mm, sim.ring_x, prof_t).astype(np.float32)
        all_w_sim.append((X_t, w_cols, d_val))

        # Station error
        w_sim_st = np.interp(stations_mm, sim.ring_x, prof_t)
        obs_st = (edge_px[t, st_px] - rest_px[st_px]) / s
        val = valid_st_dist & (~np.isnan(obs_st))
        if np.any(val):
            frame_rmse[t] = float(np.sqrt(np.mean((w_sim_st[val] - obs_st[val]) ** 2)))
        else:
            frame_rmse[t] = 0.0

    # Video reader
    raw_video_path = Path("data/raw") / c_cfg["file"]
    cap_vid = cv2.VideoCapture(str(raw_video_path))
    assert cap_vid.isOpened(), f"Cannot open video: {raw_video_path}"

    # Setup video writer (temp path)
    temp_video_path = out_path.with_name(f"temp_{out_path.name}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (1920, 720))

    # Matplotlib figure for Middle (3D) and Right (RMSE) panels
    fig = plt.figure(figsize=(12, 7.2), dpi=100)
    fig.patch.set_facecolor("#0b0f19")
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)

    # Cylinder surface connectivity for plotting
    body_N = sim.n_rings * sim.n_theta
    norm_disp = plt.Normalize(vmin=0.0, vmax=max(5.0, max_d))
    cmap = plt.cm.get_cmap("coolwarm")

    # Crop boundaries for video
    y0_crop = max(0, int(c_cfg["edge_rows"][0]) - 80)
    y1_crop = min(int(cap_vid.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(c_cfg["edge_rows"][1]) + 80)
    x0_crop = max(0, bx0 - 30)
    x1_crop = min(int(cap_vid.get(cv2.CAP_PROP_FRAME_WIDTH)), bx1 + 30)

    print(f"Processing {n_frames} frames for {clip_id}...")
    for t in range(n_frames):
        ret, frame_bgr = cap_vid.read()
        if not ret:
            break

        X_t, w_cols, d_val = all_w_sim[t]

        # ---------------- 1. Left Panel (720 x 720): Video with Overlays ----------------
        crop = frame_bgr[y0_crop:y1_crop, x0_crop:x1_crop].copy()

        # Draw resting edge (cyan faint dashed)
        for i, col in enumerate(cols_x):
            yr = rest_px[col]
            if not np.isnan(yr) and y0_crop <= yr < y1_crop:
                cv2.circle(crop, (int(col - x0_crop), int(round(yr - y0_crop))), 1, (160, 160, 160), -1)

        # Draw observed edge (green)
        for i, col in enumerate(cols_x):
            yo = edge_px[t, col]
            if not np.isnan(yo) and y0_crop <= yo < y1_crop:
                cv2.circle(crop, (int(col - x0_crop), int(round(yo - y0_crop))), 2, (0, 230, 0), -1)

        # Draw predicted edge (red: y = rest_px + w_sim * s)
        for i, col in enumerate(cols_x):
            yr = rest_px[col]
            if not np.isnan(yr):
                yp = yr + w_cols[i] * s
                if y0_crop <= yp < y1_crop:
                    cv2.circle(crop, (int(col - x0_crop), int(round(yp - y0_crop))), 2, (0, 0, 255), -1)

        # Draw tool tip marker
        if not np.isnan(tip_px[t]) and not np.isnan(tool_x_px[t]):
            tx_i, ty_i = int(round(tool_x_px[t])), int(round(tip_px[t]))
            if x0_crop <= tx_i < x1_crop and y0_crop <= ty_i < y1_crop:
                cv2.circle(crop, (tx_i - x0_crop, ty_i - y0_crop), 6, (0, 255, 255), -1)
                cv2.circle(crop, (tx_i - x0_crop, ty_i - y0_crop), 9, (0, 165, 255), 2)

        # Resize video crop to 720x720
        panel_left = cv2.resize(crop, (720, 720), interpolation=cv2.INTER_AREA)

        # Clean informational overlay card on left panel
        cv2.rectangle(panel_left, (15, 15), (420, 150), (15, 23, 42), -1)
        cv2.rectangle(panel_left, (15, 15), (420, 150), (56, 189, 248), 2)

        cap_str = "CAPPED (Air Pressure On)" if is_cap else "UNCAPPED (Pure Bending)"
        cv2.putText(panel_left, f"Clip: {clip_id} ({cap_str})", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(panel_left, f"Frame: {t + 1}/{n_frames}  |  Tip drop: {d_val:.2f} mm", (25, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (203, 213, 225), 1)

        # Legend
        cv2.circle(panel_left, (35, 95), 5, (0, 230, 0), -1)
        cv2.putText(panel_left, "Observed Edge", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 230, 0), 2)
        cv2.circle(panel_left, (215, 95), 5, (0, 0, 255), -1)
        cv2.putText(panel_left, "Model Predicted Edge", (230, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)

        cv2.circle(panel_left, (35, 125), 4, (160, 160, 160), -1)
        cv2.putText(panel_left, "Rest Edge", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
        cv2.circle(panel_left, (215, 125), 5, (0, 255, 255), -1)
        cv2.putText(panel_left, "Tracked Indenter Tip", (230, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # ---------------- 2. Middle Panel (600 x 720): 3D Mesh ----------------
        ax3d.cla()
        ax3d.set_facecolor("#0b0f19")

        # Color vertices by displacement magnitude
        disp_mag = np.linalg.norm(X_t[:body_N] - sim.X0[:body_N], axis=1)
        body_pts = X_t[:body_N].reshape(sim.n_rings, sim.n_theta, 3)
        body_disp = disp_mag.reshape(sim.n_rings, sim.n_theta)

        # Circular closure
        body_pts_c = np.concatenate([body_pts, body_pts[:, :1, :]], axis=1)
        body_disp_c = np.concatenate([body_disp, body_disp[:, :1]], axis=1)

        X_grid = body_pts_c[:, :, 0]
        Y_grid = body_pts_c[:, :, 1]
        Z_grid = body_pts_c[:, :, 2]
        face_colors = cmap(norm_disp(body_disp_c))

        ax3d.plot_surface(
            X_grid,
            Y_grid,
            Z_grid,
            facecolors=face_colors,
            rstride=1,
            cstride=1,
            shade=False,
            alpha=0.92,
            edgecolor="gray",
            linewidth=0.15,
        )

        # Plot tool contact location
        c_tool = [tool_x_med_mm, 0.0, 2.0 * sim.R - d_val + r_tool]
        ax3d.scatter([c_tool[0]], [c_tool[1]], [c_tool[2]], color="#facc15", s=80, edgecolors="white", lw=1.5)

        ax3d.set_box_aspect([220, 70, 70])
        ax3d.view_init(elev=25, azim=-60)
        ax3d.set_xlim(0, 220)
        ax3d.set_ylim(-40, 40)
        ax3d.set_zlim(0, 75)
        ax3d.axis("off")
        ax3d.set_title(
            f"3D Shell Mesh Displacement\nIndentation $d = {d_val:.1f}$ mm",
            color="#f8fafc",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

        # ---------------- 3. Right Panel (600 x 720): RMSE vs Time ----------------
        ax2d.cla()
        ax2d.set_facecolor("#0f172a")

        # Shade regions outside loading (pre-contact approach & post-release)
        ax2d.axvspan(0, c_start, color="#1e293b", alpha=0.65, label="Outside Loading")
        ax2d.axvspan(t_peak, n_frames, color="#1e293b", alpha=0.65)
        ax2d.axvspan(c_start, t_peak, color="#0284c7", alpha=0.15, label="Loading Phase (Scored)")

        t_axis = np.arange(n_frames)
        ax2d.plot(t_axis, frame_rmse, color="#38bdf8", lw=2.2, label="Model RMSE (mm)")
        ax2d.axvline(x=t, color="#f43f5e", lw=2.5, linestyle="--", label=f"Cursor (t={t})")
        ax2d.scatter([t], [frame_rmse[t]], color="#f43f5e", s=60, zorder=5)

        ax2d.set_xlim(0, n_frames)
        ax2d.set_ylim(0, max(2.5, float(np.max(frame_rmse)) * 1.25))
        ax2d.set_xlabel("Frame Index $t$", color="#e2e8f0", fontsize=11)
        ax2d.set_ylabel("Spatial RMSE along Bottle (mm)", color="#e2e8f0", fontsize=11)
        in_loading = c_start <= t <= t_peak
        phase_str = "Loading Phase" if in_loading else "Outside Loading"
        ax2d.set_title(
            f"Spatial RMSE vs Time ({phase_str}: t={c_start}..{t_peak})\nCurrent RMSE: {frame_rmse[t]:.2f} mm",
            color="#f8fafc",
            fontsize=11,
            fontweight="bold",
            pad=10,
        )
        ax2d.tick_params(colors="#94a3b8", labelsize=10)
        ax2d.grid(True, color="#334155", linestyle=":", alpha=0.6)
        for spine in ax2d.spines.values():
            spine.set_color("#475569")
        ax2d.legend(loc="upper right", facecolor="#1e293b", edgecolor="#475569", labelcolor="white", fontsize=8)

        fig.canvas.draw()
        fig_img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        fig_bgr = cv2.cvtColor(fig_img, cv2.COLOR_RGB2BGR)
        mid_right = cv2.resize(fig_bgr, (1200, 720), interpolation=cv2.INTER_AREA)

        # Assemble composite frame
        composite = np.hstack([panel_left, mid_right])
        writer.write(composite)

    cap_vid.release()
    writer.release()
    plt.close(fig)

    # Convert to browser-compatible H.264
    convert_to_h264(temp_video_path, out_path)
    print(f"Successfully generated: {out_path} ({n_frames} frames)")
    return out_path


def render_cap_effect(
    out_path: Path,
    sim: StaticSimulator,
    rho: float,
    kappa: float,
    cfg: Dict[str, Any],
    tracks_dir: Path = Path("out/tracks"),
    fps: float = 29.97,
) -> Path:
    """Render demonstration video comparing 001531 (capped) and 001656 (uncapped)."""
    print("Rendering Cap Effect Demo (001531 vs 001656)...")
    clip_cap = "001531"
    clip_uncap = "001656"

    c_cap = cfg["clips"][clip_cap]
    c_uncap = cfg["clips"][clip_uncap]

    tr_cap = np.load(tracks_dir / f"{clip_cap}.npz", allow_pickle=True)
    tr_mm_cap = np.load(tracks_dir / f"{clip_cap}_mm.npz", allow_pickle=True)
    tr_uncap = np.load(tracks_dir / f"{clip_uncap}.npz", allow_pickle=True)
    tr_mm_uncap = np.load(tracks_dir / f"{clip_uncap}_mm.npz", allow_pickle=True)

    n_frames_cap = tr_cap["edge_px"].shape[0]
    n_frames_uncap = tr_uncap["edge_px"].shape[0]
    total_frames = max(n_frames_cap, n_frames_uncap)

    # Peak normalization data
    pk_d_cap = float(np.nanmax(tr_mm_cap["d_mm"]))
    pk_d_uncap = float(np.nanmax(tr_mm_uncap["d_mm"]))

    tx_cap = float(tr_mm_cap["tool_x_med_mm"])
    tx_uncap = float(tr_mm_uncap["tool_x_med_mm"])

    st_cap = tr_mm_cap["stations_mm"]
    st_uncap = tr_mm_uncap["stations_mm"]

    dx_cap = st_cap - tx_cap
    dx_uncap = st_uncap - tx_uncap

    # Peak model solve
    sol_pk_cap = sim.solve_snapshot(pk_d_cap, tool_x_mm=tx_cap, kappa=kappa, rho=rho, air_on=True, r_tool_mm=2.5)
    w_model_cap_pk = sim.extract_profile(sol_pk_cap, stations_mm=st_cap, mode="top_vertex")
    norm_model_cap = w_model_cap_pk / pk_d_cap

    sol_pk_uncap = sim.solve_snapshot(pk_d_uncap, tool_x_mm=tx_uncap, kappa=1.0, rho=rho, air_on=False, r_tool_mm=2.5)
    w_model_uncap_pk = sim.extract_profile(sol_pk_uncap, stations_mm=st_uncap, mode="top_vertex")
    norm_model_uncap = w_model_uncap_pk / pk_d_uncap

    # Peak data
    k_pk_cap = np.argmax(tr_mm_cap["d_mm"])
    norm_data_cap = tr_mm_cap["u_mm"][k_pk_cap] / pk_d_cap

    k_pk_uncap = np.argmax(tr_mm_uncap["d_mm"])
    norm_data_uncap = tr_mm_uncap["u_mm"][k_pk_uncap] / pk_d_uncap

    # Pre-solve sequence for both clips to overlay model predictions
    grid_d_cap, sols_cap, profs_cap = precompute_simulation_grid(
        sim, pk_d_cap, tx_cap, kappa=kappa, rho=rho, air_on=True, r_tool_mm=2.5, n_pts=20
    )
    grid_d_uncap, sols_uncap, profs_uncap = precompute_simulation_grid(
        sim, pk_d_uncap, tx_uncap, kappa=1.0, rho=rho, air_on=False, r_tool_mm=2.5, n_pts=20
    )

    cap_vid1 = cv2.VideoCapture(str(Path("data/raw") / c_cap["file"]))
    cap_vid2 = cv2.VideoCapture(str(Path("data/raw") / c_uncap["file"]))

    temp_video_path = out_path.with_name(f"temp_{out_path.name}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (1920, 720))

    # Pre-render bottom panel (Normalized peak profiles comparison) ONCE
    fig_bot = plt.figure(figsize=(19.2, 3.6), dpi=100)
    fig_bot.patch.set_facecolor("#0b0f19")
    ax_bot = fig_bot.add_subplot(111)
    ax_bot.set_facecolor("#0f172a")

    # Capped curves (Blue)
    ax_bot.plot(dx_cap, norm_data_cap, "o-", color="#38bdf8", lw=2.2, ms=6, label="Capped Data (001531)")
    ax_bot.plot(dx_cap, norm_model_cap, "--", color="#0284c7", lw=2.5, label="Capped Model (\\kappa=0.76, Air On)")

    # Uncapped curves (Orange)
    ax_bot.plot(dx_uncap, norm_data_uncap, "s-", color="#fb923c", lw=2.2, ms=6, label="Uncapped Data (001656)")
    ax_bot.plot(dx_uncap, norm_model_uncap, "--", color="#ea580c", lw=2.5, label="Uncapped Model (\\kappa=1.00, Air Off)")

    ax_bot.set_xlim(-60, 60)
    ax_bot.set_ylim(-0.05, 1.15)
    ax_bot.set_xlabel("Axial Distance from Tool Tip $\\Delta x = x - x_{\\mathrm{tool}}$ (mm)", color="#e2e8f0", fontsize=11)
    ax_bot.set_ylabel("Normalised Drop $w / d_{\\mathrm{peak}}$", color="#e2e8f0", fontsize=11)
    ax_bot.set_title(
        "Cap Effect: Normalized Peak Profiles (Air Narrows the Profile: Near-Tool 15-21 mm Ratio 0.64 Capped vs 0.79 Uncapped)",
        color="#f8fafc",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    ax_bot.tick_params(colors="#94a3b8", labelsize=10)
    ax_bot.grid(True, color="#334155", linestyle=":", alpha=0.6)
    for spine in ax_bot.spines.values():
        spine.set_color("#475569")
    ax_bot.legend(loc="upper right", ncol=2, facecolor="#1e293b", edgecolor="#475569", labelcolor="white", fontsize=9)

    fig_bot.tight_layout()
    fig_bot.canvas.draw()
    bot_img = np.asarray(fig_bot.canvas.buffer_rgba())[:, :, :3]
    bot_bgr = cv2.cvtColor(bot_img, cv2.COLOR_RGB2BGR)
    panel_bot = cv2.resize(bot_bgr, (1920, 360), interpolation=cv2.INTER_AREA)
    plt.close(fig_bot)

    s_cap = float(tr_mm_cap["s"])
    s_uncap = float(tr_mm_uncap["s"])
    bx0_c, bx1_c = int(c_cap["body_x"][0]), int(c_cap["body_x"][1])
    bx0_u, bx1_u = int(c_uncap["body_x"][0]), int(c_uncap["body_x"][1])
    cols_c = np.arange(bx0_c, bx1_c)
    cols_u = np.arange(bx0_u, bx1_u)
    cols_c_mm = (float(tr_mm_cap["x_base_px"]) - cols_c) / s_cap
    cols_u_mm = (float(tr_mm_uncap["x_base_px"]) - cols_u) / s_uncap

    y0_c, y1_c = int(c_cap["edge_rows"][0]) - 50, int(c_cap["edge_rows"][1]) + 50
    y0_u, y1_u = int(c_uncap["edge_rows"][0]) - 50, int(c_uncap["edge_rows"][1]) + 50

    last_f2 = None
    print(f"Processing {total_frames} frames for Cap Effect...")

    for t in range(total_frames):
        # Read video 1 (capped)
        ret1, f1 = cap_vid1.read()
        if not ret1:
            f1 = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Read video 2 (uncapped)
        ret2, f2 = cap_vid2.read()
        if ret2:
            last_f2 = f2.copy()
        else:
            f2 = last_f2.copy() if last_f2 is not None else np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Simulation drop for current frame
        d_c = float(tr_mm_cap["tip_drop_mm"][t]) if t < n_frames_cap and tr_cap["contact"][t] else 0.0
        d_c = max(0.0, d_c)
        _, prof_c = interpolate_state(d_c, grid_d_cap, sols_cap, profs_cap, sim)
        w_cols_c = np.interp(cols_c_mm, sim.ring_x, prof_c)

        t_u = min(t, n_frames_uncap - 1)
        d_u = float(tr_mm_uncap["tip_drop_mm"][t_u]) if tr_uncap["contact"][t_u] else 0.0
        d_u = max(0.0, d_u)
        _, prof_u = interpolate_state(d_u, grid_d_uncap, sols_uncap, profs_uncap, sim)
        w_cols_u = np.interp(cols_u_mm, sim.ring_x, prof_u)

        # ---------------- Panel Top-Left (960 x 360): Capped 001531 ----------------
        crop1 = f1[y0_c:y1_c, bx0_c - 20 : bx1_c + 20].copy()
        # Draw observed edge (green)
        if t < n_frames_cap:
            yo_c = tr_cap["edge_px"][t, cols_c]
            mask_obs = ~np.isnan(yo_c) & (yo_c >= y0_c) & (yo_c < y1_c)
            if np.any(mask_obs):
                pts_obs = np.column_stack([
                    (cols_c[mask_obs] - (bx0_c - 20)).astype(np.int32),
                    np.round(yo_c[mask_obs] - y0_c).astype(np.int32)
                ]).reshape((-1, 1, 2))
                cv2.polylines(crop1, [pts_obs], isClosed=False, color=(0, 230, 0), thickness=2)

        # Draw predicted edge (red)
        yr_c = tr_cap["rest_px"][cols_c]
        mask_pred = ~np.isnan(yr_c)
        if np.any(mask_pred):
            yp_c = yr_c[mask_pred] + w_cols_c[mask_pred] * s_cap
            valid_p = (yp_c >= y0_c) & (yp_c < y1_c)
            if np.any(valid_p):
                pts_pred = np.column_stack([
                    (cols_c[mask_pred][valid_p] - (bx0_c - 20)).astype(np.int32),
                    np.round(yp_c[valid_p] - y0_c).astype(np.int32)
                ]).reshape((-1, 1, 2))
                cv2.polylines(crop1, [pts_pred], isClosed=False, color=(0, 0, 255), thickness=2)

        panel_top_left = cv2.resize(crop1, (960, 360), interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel_top_left, (10, 10), (450, 60), (15, 23, 42), -1)
        cv2.rectangle(panel_top_left, (10, 10), (450, 60), (56, 189, 248), 2)
        cv2.putText(panel_top_left, "001531 (CAPPED) - Internal Air (gamma=1.4)", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        cv2.putText(panel_top_left, f"Frame {t+1}/{n_frames_cap} | Tip Drop: {d_c:.2f} mm", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (203, 213, 225), 1)

        # ---------------- Panel Top-Right (960 x 360): Uncapped 001656 ----------------
        crop2 = f2[y0_u:y1_u, bx0_u - 20 : bx1_u + 20].copy()
        # Draw observed edge (green)
        yo_u = tr_uncap["edge_px"][t_u, cols_u]
        mask_obs_u = ~np.isnan(yo_u) & (yo_u >= y0_u) & (yo_u < y1_u)
        if np.any(mask_obs_u):
            pts_obs_u = np.column_stack([
                (cols_u[mask_obs_u] - (bx0_u - 20)).astype(np.int32),
                np.round(yo_u[mask_obs_u] - y0_u).astype(np.int32)
            ]).reshape((-1, 1, 2))
            cv2.polylines(crop2, [pts_obs_u], isClosed=False, color=(0, 230, 0), thickness=2)

        # Draw predicted edge (red)
        yr_u = tr_uncap["rest_px"][cols_u]
        mask_pred_u = ~np.isnan(yr_u)
        if np.any(mask_pred_u):
            yp_u = yr_u[mask_pred_u] + w_cols_u[mask_pred_u] * s_uncap
            valid_pu = (yp_u >= y0_u) & (yp_u < y1_u)
            if np.any(valid_pu):
                pts_pred_u = np.column_stack([
                    (cols_u[mask_pred_u][valid_pu] - (bx0_u - 20)).astype(np.int32),
                    np.round(yp_u[valid_pu] - y0_u).astype(np.int32)
                ]).reshape((-1, 1, 2))
                cv2.polylines(crop2, [pts_pred_u], isClosed=False, color=(0, 0, 255), thickness=2)

        panel_top_right = cv2.resize(crop2, (960, 360), interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel_top_right, (10, 10), (450, 60), (15, 23, 42), -1)
        cv2.rectangle(panel_top_right, (10, 10), (450, 60), (251, 146, 60), 2)
        cv2.putText(panel_top_right, "001656 (UNCAPPED) - Pure Shell Bending (Air Off)", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        cv2.putText(panel_top_right, f"Frame {t_u+1}/{n_frames_uncap} | Tip Drop: {d_u:.2f} mm", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (203, 213, 225), 1)

        panel_top = np.hstack([panel_top_left, panel_top_right])

        # Assemble composite frame
        composite = np.vstack([panel_top, panel_bot])
        writer.write(composite)

    cap_vid1.release()
    cap_vid2.release()
    writer.release()
    plt.close(fig_bot)

    # Convert to browser-compatible H.264
    convert_to_h264(temp_video_path, out_path)
    print(f"Successfully generated: {out_path} ({total_frames} frames)")
    return out_path


def render_all_concatenated(
    out_path: Path,
    fresh_161747_path: Path,
    fresh_001711_path: Path,
    cap_effect_path: Path,
    fps: float = 29.97,
) -> Path:
    """Concatenate demo videos with high-contrast informative title cards into `all.mp4`."""
    print("Generating all.mp4 with title cards...")
    out_dir = out_path.parent
    cards_dir = out_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)

    # 1. Main intro title card
    card1 = create_title_card(
        title="Learning PET Bottle Deformation from Phone Video",
        subtitle="Part 2 Brief: Physics-Calibrated Shell-Springs + Gas Overpressure",
        bullets=[
            "Data: 30 fps phone video of capped & uncapped 1 L PET bottles",
            "Model: Discrete triangle-mesh shell with anisotropic bending springs",
            "Internal Air: Adiabatic relation p * V^1.4 = const (gamma = 1.4)",
            "Predicting 3D deformation field from tracked tool tip trajectory",
        ],
        badge="Evaluation Pipeline",
    )

    # 2. Fresh instance title card
    card2 = create_title_card(
        title="Part 1: Fresh Instance Generalization",
        subtitle="Held-out Clip 161747 (Capped, Pen Indenter, New Position)",
        bullets=[
            "Tracking input: Pen tip drop d(t) driving top spherical contact",
            "Left: Observed silhouette edge (green) vs Model predicted edge (red)",
            "Middle: 3D cylinder mesh colored by radial displacement magnitude",
            "Right: Real-time spatial RMSE vs time with tracking cursor",
        ],
        badge="Part 1: Test Clip 161747",
    )

    # 3. Transition card for 001711
    card3 = create_title_card(
        title="Part 1 (Continued): Uncapped Generalization",
        subtitle="Held-out Clip 001711 (Uncapped, Pen Indenter, Zero Gauge Pressure)",
        bullets=[
            "Validates model under pure bending without internal air stiffening",
            "Model achieves 1.41 mm RMSE vs 5.43 mm baseline B0 (74% error reduction)",
            "Quasi-static equilibrium solver tracks continuous deformation branch",
        ],
        badge="Part 1: Test Clip 001711",
    )

    # 4. Cap effect title card
    card4 = create_title_card(
        title="Part 2: Cap Effect Comparison",
        subtitle="001531 (Capped, Air On) vs 001656 (Uncapped, Pure Shell Bending)",
        bullets=[
            "Side-by-side video overlays: observed edge (green) & predicted edge (red)",
            "Bottom panel: Normalized peak profiles data vs model on same axes",
            "Internal air pressure narrows the deformation profile",
            "Near-tool drop ratio (15-21 mm): 0.64 capped vs 0.79 uncapped in data",
        ],
        badge="Part 2: Cap Effect",
    )

    # 5. Standalone simulation card
    card_sim = create_title_card(
        title="Part 3: Standalone Simulation",
        subtitle="Standalone simulation: press at 40 mm from base, 14 mm depth, cap on vs off (model prediction, no video).",
        bullets=[
            "Quasi-static loading from 0 to 14 mm and elastic recovery back to 0 mm",
            "Left: Capped (Air On, peak force 11.60 N, Delta p = 1.52 kPa)",
            "Right: Uncapped (Air Off, peak force 8.93 N, Delta p = 0.0 kPa)",
            "Internal air stiffens bottle and narrows profile near the tool",
        ],
        badge="Part 3: Simulation",
    )

    # 6. Conclusion card
    card5 = create_title_card(
        title="Evaluation Summary & Key Results",
        subtitle="Benchmark on Held-Out Test Suite",
        bullets=[
            "Headline: Model 1.95 mm vs B1 1.84 mm over all test clips",
            "B1 (template) wins on all 4 pen clips",
            "The physics model wins on both finger clips (new indenter)",
            "161853 has camera shift and its near-tool shape is still missed (0.73 sim vs 0.45 data)",
        ],
        badge="Conclusions",
    )

    def write_card_video(card_img: np.ndarray, duration_sec: float, card_path: Path) -> Path:
        n_f = int(round(duration_sec * fps))
        temp_card = card_path.with_name(f"temp_{card_path.name}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = cv2.VideoWriter(str(temp_card), fourcc, fps, (1920, 720))
        for _ in range(n_f):
            w.write(card_img)
        w.release()
        convert_to_h264(temp_card, card_path)
        return card_path

    c1_path = write_card_video(card1, 3.0, cards_dir / "card1.mp4")
    c2_path = write_card_video(card2, 2.5, cards_dir / "card2.mp4")
    c3_path = write_card_video(card3, 2.5, cards_dir / "card3.mp4")
    c4_path = write_card_video(card4, 3.0, cards_dir / "card4.mp4")
    c_sim_path = write_card_video(card_sim, 3.0, cards_dir / "card_sim.mp4")
    c5_path = write_card_video(card5, 3.5, cards_dir / "card5.mp4")

    # Check for standalone simulation segment
    sim_comp_src = Path("out/sim/compare_press_x40_d14_pen/sim.mp4")
    sim_segment_path = cards_dir / "sim_segment.mp4"
    if sim_comp_src.exists() and not (sim_segment_path.exists() and sim_segment_path.stat().st_size > 10000):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-stream_loop",
                "3",
                "-i",
                str(sim_comp_src),
                "-vf",
                "scale=1920:720:force_original_aspect_ratio=decrease,pad=1920:720:(ow-iw)/2:(oh-ih)/2:color=#0f172a",
                "-r",
                str(fps),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-t",
                "10.0",
                str(sim_segment_path),
            ],
            check=True,
        )

    # Build concat file for ffmpeg
    def rel_to_cards(p: Path) -> str:
        return os.path.relpath(p, cards_dir)

    concat_list_path = cards_dir / "concat_list.txt"
    with open(concat_list_path, "w", encoding="utf-8") as f:
        f.write(f"file '{rel_to_cards(c1_path)}'\n")
        f.write(f"file '{rel_to_cards(c2_path)}'\n")
        f.write(f"file '{rel_to_cards(fresh_161747_path)}'\n")
        f.write(f"file '{rel_to_cards(c3_path)}'\n")
        f.write(f"file '{rel_to_cards(fresh_001711_path)}'\n")
        f.write(f"file '{rel_to_cards(c4_path)}'\n")
        f.write(f"file '{rel_to_cards(cap_effect_path)}'\n")
        if sim_segment_path.exists():
            f.write(f"file '{rel_to_cards(c_sim_path)}'\n")
            f.write(f"file '{rel_to_cards(sim_segment_path)}'\n")
        f.write(f"file '{rel_to_cards(c5_path)}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        "-preset",
        "medium",
        str(out_path),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg concat failed: {res.stderr.decode('utf-8')}")

    print(f"Successfully generated master demo video: {out_path}")
    return out_path


def run_demo_pipeline(out_dir: Path = Path("out/demo"), force: bool = False) -> Dict[str, Path]:
    """Execute complete Module M7 demo video pipeline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    params, cfg = load_inputs()

    rho = float(params["rho_hat"])
    kappa = float(params["kappa_hat"])
    sim = StaticSimulator()
    fps = float(cfg.get("fps", 29.97))

    # 1. Fresh instance individual videos
    p_161747 = out_dir / "fresh_instance_161747.mp4"
    if force or not (p_161747.exists() and p_161747.stat().st_size > 100000):
        render_fresh_instance_clip("161747", p_161747, sim, rho, kappa, cfg, fps=fps)
    else:
        print(f"Clip 161747 video already exists: {p_161747}")

    p_001711 = out_dir / "fresh_instance_001711.mp4"
    if force or not (p_001711.exists() and p_001711.stat().st_size > 100000):
        render_fresh_instance_clip("001711", p_001711, sim, rho, kappa, cfg, fps=fps)
    else:
        print(f"Clip 001711 video already exists: {p_001711}")

    # 2. Fresh instance concatenated video
    p_fresh = out_dir / "fresh_instance.mp4"
    if force or not (p_fresh.exists() and p_fresh.stat().st_size > 100000):
        concat_txt = out_dir / "concat_fresh.txt"
        with open(concat_txt, "w") as f:
            f.write(f"file '{p_161747.resolve()}'\n")
            f.write(f"file '{p_001711.resolve()}'\n")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt), "-c", "copy", str(p_fresh)],
            check=True,
        )
        if concat_txt.exists():
            concat_txt.unlink()
    else:
        print(f"Fresh instance concatenated video already exists: {p_fresh}")

    # 3. Cap effect comparison video
    p_cap = out_dir / "cap_effect.mp4"
    if force or not (p_cap.exists() and p_cap.stat().st_size > 100000):
        render_cap_effect(p_cap, sim, rho, kappa, cfg, fps=fps)
    else:
        print(f"Cap effect video already exists: {p_cap}")

    # 4. Master video with title cards
    p_all = out_dir / "all.mp4"
    if force or not (p_all.exists() and p_all.stat().st_size > 100000):
        render_all_concatenated(p_all, p_161747, p_001711, p_cap, fps=fps)
    else:
        print(f"Master video already exists: {p_all}")

    return {
        "fresh_instance_161747": p_161747,
        "fresh_instance_001711": p_001711,
        "fresh_instance": p_fresh,
        "cap_effect": p_cap,
        "all": p_all,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M7 Demo Video Pipeline")
    parser.add_argument("--out-dir", type=Path, default=Path("out/demo"), help="Output directory")
    parser.add_argument("--force", action="store_true", help="Force re-rendering of existing videos")
    args = parser.parse_args()

    paths = run_demo_pipeline(out_dir=args.out_dir, force=args.force)
    print("\n--- Module M7 Demo Complete ---")
    for name, path in paths.items():
        print(f"  {name}: {path} ({path.stat().st_size / (1024*1024):.2f} MB)")
