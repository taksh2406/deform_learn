#!/usr/bin/env python3
"""M8 — Standalone simulator (`src/simulate.py`).

Allows pressing the fitted virtual bottle at any position and depth with no video involved,
simulating quasi-static loading and elastic recovery, and comparing cap-on vs cap-off behaviour.

Usage:
    python src/simulate.py --press-at 60 --depth 12 --tool pen --cap on
    python src/simulate.py --press-at 40 --depth 14 --compare-cap
    python src/simulate.py --trajectory my_press.csv --press-at 60
    python src/simulate.py --interactive
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure Matplotlib config directory is writable
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/mpl"

import matplotlib
if "--interactive" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import yaml

# Relative project imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.mesh import enclosed_volume
from src.sim import StaticSimulator


def load_config_and_params(
    config_path: Path = PROJECT_ROOT / "config" / "clips.yaml",
    params_path: Path = PROJECT_ROOT / "out" / "fit" / "params.json",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load clips configuration and fitted physical parameters."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)
    return cfg, params


def compute_extrapolation_warnings(press_at: float, depth: float) -> List[str]:
    """Check input press parameters against observed experimental calibration data."""
    warnings = []
    if depth > 17.0:
        warnings.append(
            f"Indentation depth {depth:.1f} mm exceeds 17.0 mm (deepest observed press in dataset). "
            "Results represent mathematical model extrapolation."
        )
    if press_at < 56.0 or press_at > 98.0:
        warnings.append(
            f"Press position {press_at:.1f} mm falls outside 56–98 mm (range of observed presses in dataset). "
            "Results represent spatial extrapolation."
        )
    if press_at < 15.0 or press_at > (220.0 - 15.0):
        warnings.append(
            f"Press position {press_at:.1f} mm is within 15 mm of cylinder end ([0, 220] mm). "
            "End-ring boundary effects dominate."
        )
    return warnings


def compute_tool_force(
    sol: np.ndarray,
    press_at: float,
    depth: float,
    r_eff: float,
    R: float,
    kappa: float,
    air_on: bool,
    sim: Optional[StaticSimulator] = None,
    rho: float = 1.0,
    gas_exponent: float = 1.4,
) -> float:
    """Compute upward vertical reaction force F_z exerted by the bottle shell on the held indenter."""
    if depth <= 0.0:
        return 0.0
    sim_obj = StaticSimulator() if sim is None else sim
    F_z = sim_obj.reaction_force(
        sol,
        tip_drop_mm=float(depth),
        tool_x_mm=float(press_at),
        r_tool_mm=float(r_eff),
        kappa=float(kappa),
        rho=float(rho),
        air_on=bool(air_on),
        n_gas=float(gas_exponent),
    )
    return max(0.0, float(F_z))


def simulate_sequence(
    sim: StaticSimulator,
    depths: List[float],
    press_at: float,
    tool_radius: float,
    kappa: float,
    rho: float,
    air_on: bool,
    gas_exponent: float,
    gtol: float = 1e-6,
    maxiter: int = 20000,
) -> Dict[str, Any]:
    """Solve static equilibrium sequentially along a sequence of indentation depths."""
    sols = []
    forces = []
    overpressures = []
    volumes_lost = []

    depths_arr = np.array(depths, dtype=np.float64)
    k_pk = int(np.argmax(depths_arr))

    # Loading phase: use sim.solve_sequence for exact consistency with evaluate.py
    loading_depths = depths_arr[:k_pk + 1]
    nonzero_mask = loading_depths > 0.0
    if np.any(nonzero_mask):
        first_nonzero = int(np.where(nonzero_mask)[0][0])
        nz_depths = loading_depths[first_nonzero:]
        nz_sols = sim.solve_sequence(
            nz_depths,
            tool_x_mm=press_at,
            r_tool_mm=tool_radius,
            kappa=float(kappa),
            rho=rho,
            air_on=air_on,
            n_gas=gas_exponent,
            gtol=gtol,
            maxiter=maxiter,
        )
        load_sols = [sim.X0.copy() for _ in range(first_nonzero)] + nz_sols
    else:
        load_sols = [sim.X0.copy() for _ in range(k_pk + 1)]

    # Unloading phase: warm-start from previous solution, returning to rest state at d=0
    unload_sols = []
    x_curr = load_sols[-1] if len(load_sols) > 0 else sim.X0.copy()
    unloading_depths = depths_arr[k_pk + 1:]
    for d in unloading_depths:
        if d <= 0.0:
            sol = sim.X0.copy()
        else:
            sol = sim.solve_snapshot(
                tip_drop_mm=float(d),
                tool_x_mm=press_at,
                r_tool_mm=tool_radius,
                kappa=float(kappa),
                rho=rho,
                air_on=air_on,
                n_gas=gas_exponent,
                X_init=x_curr,
                gtol=gtol,
                maxiter=maxiter,
            )
        x_curr = sol
        unload_sols.append(sol)

    all_sols = load_sols + unload_sols

    for d, sol in zip(depths, all_sols):
        if d <= 0.0:
            F_z = 0.0
            dp = 0.0
            dV = 0.0
        else:
            F_z = compute_tool_force(
                sol,
                press_at,
                float(d),
                tool_radius,
                sim.R,
                kappa,
                air_on,
                sim=sim,
                rho=rho,
                gas_exponent=gas_exponent,
            )
            V_mesh = enclosed_volume(sol, sim.tris)
            dV = float(sim.V_mesh_rest - V_mesh)
            if air_on:
                V_gas = sim.V0 + (V_mesh - sim.V_mesh_rest)
                p = sim.p_atm * ((sim.V0 / V_gas) ** gas_exponent)
                dp = float((p - sim.p_atm) * 1.0e3)
            else:
                dp = 0.0

        sols.append(sol)
        forces.append(F_z)
        overpressures.append(dp)
        volumes_lost.append(dV)

    return {
        "depths": depths,
        "solutions": sols,
        "forces": forces,
        "overpressures": overpressures,
        "volumes_lost": volumes_lost,
    }


def compute_profile_ratios(
    w_profile: np.ndarray,
    stations_mm: np.ndarray,
    tool_x_mm: float,
    peak_depth: float,
) -> Dict[str, Optional[float]]:
    """Compute normalized profile drop ratios at 15–21, 30–39, and 45–60 mm bands."""
    if peak_depth <= 1e-6:
        return {"ratio_15_21": None, "ratio_30_39": None, "ratio_45_60": None}

    dist = np.abs(stations_mm - tool_x_mm)
    m15 = (dist >= 15.0) & (dist <= 21.0)
    m30 = (dist >= 30.0) & (dist <= 39.0)
    m45 = (dist >= 45.0) & (dist <= 60.0)

    r15 = float(np.mean(w_profile[m15]) / peak_depth) if np.any(m15) else None
    r30 = float(np.mean(w_profile[m30]) / peak_depth) if np.any(m30) else None
    r45 = float(np.mean(w_profile[m45]) / peak_depth) if np.any(m45) else None

    return {
        "ratio_15_21": round(r15, 4) if r15 is not None else None,
        "ratio_30_39": round(r30, 4) if r30 is not None else None,
        "ratio_45_60": round(r45, 4) if r45 is not None else None,
    }


def plot_side_view(
    out_path: Path,
    sim: StaticSimulator,
    press_at: float,
    depth: float,
    res_on: Dict[str, Any],
    res_off: Optional[Dict[str, Any]] = None,
) -> None:
    """Plot simulated top silhouette at peak indentation with rest line."""
    k_pk = int(np.argmax(res_on["depths"]))
    sol_pk_on = res_on["solutions"][k_pk]
    w_on = sim.extract_profile(sol_pk_on, mode="top_vertex")
    x_rings = sim.ring_x

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    ax.axhline(2.0 * sim.R, color="gray", linestyle="--", linewidth=1.5, label="Rest top generator (z = 70 mm)")
    ax.axhline(0.0, color="black", linestyle="-", linewidth=2.0, label="Floor support (z = 0 mm)")

    # Deformed silhouette
    z_on = 2.0 * sim.R - w_on
    ax.plot(x_rings, z_on, color="crimson", linewidth=2.5, label=f"Simulated top (Cap ON, peak d = {depth:.1f} mm)")

    if res_off is not None:
        sol_pk_off = res_off["solutions"][k_pk]
        w_off = sim.extract_profile(sol_pk_off, mode="top_vertex")
        z_off = 2.0 * sim.R - w_off
        ax.plot(x_rings, z_off, color="dodgerblue", linestyle="-.", linewidth=2.0, label="Simulated top (Cap OFF)")

    # Indenter position marker
    ax.plot([press_at], [2.0 * sim.R - depth], "v", color="purple", markersize=10, label=f"Tool position (x = {press_at:.1f} mm)")

    ax.set_xlim(-5, sim.L + 5)
    ax.set_ylim(-5, 2.0 * sim.R + 10)
    ax.set_xlabel("Station x along bottle from base (mm)")
    ax.set_ylabel("Elevation z (mm)")
    ax.set_title(f"Simulated Silhouette at Peak Indentation ({press_at:.1f} mm from base, {depth:.1f} mm depth)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_profile(
    out_path: Path,
    sim: StaticSimulator,
    press_at: float,
    depth: float,
    res_on: Dict[str, Any],
    res_off: Optional[Dict[str, Any]] = None,
) -> None:
    """Plot normalized drop w/tip vs distance from tool at peak indentation."""
    k_pk = int(np.argmax(res_on["depths"]))
    sol_pk_on = res_on["solutions"][k_pk]
    stations = np.linspace(max(0, press_at - 80), min(sim.L, press_at + 80), 200)
    w_on = sim.extract_profile(sol_pk_on, stations_mm=stations, mode="top_vertex")
    dist = np.abs(stations - press_at)

    # Sort by distance for a clean 1D profile curve
    order = np.argsort(dist)
    dist_sorted = dist[order]
    norm_w_on = (w_on / max(depth, 1e-6))[order]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    # Shaded metric evaluation zones
    ax.axvspan(15, 21, color="gold", alpha=0.2, label="15–21 mm band")
    ax.axvspan(30, 39, color="lightgreen", alpha=0.2, label="30–39 mm band")
    ax.axvspan(45, 60, color="lightblue", alpha=0.2, label="45–60 mm band")

    ax.plot(dist_sorted, norm_w_on, color="crimson", linewidth=2.2, label="Cap ON (air narrows profile)")

    if res_off is not None:
        sol_pk_off = res_off["solutions"][k_pk]
        w_off = sim.extract_profile(sol_pk_off, stations_mm=stations, mode="top_vertex")
        norm_w_off = (w_off / max(depth, 1e-6))[order]
        ax.plot(dist_sorted, norm_w_off, color="dodgerblue", linestyle="--", linewidth=2.2, label="Cap OFF (air off, wider profile)")

    ax.set_xlim(0, 80)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Distance from tool |x - x_tool| (mm)")
    ax.set_ylabel("Normalized drop w / d_tip")
    ax.set_title(f"Normalized Indentation Profile at Peak ({depth:.1f} mm depth)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_force_depth(
    out_path: Path,
    res_on: Dict[str, Any],
    res_off: Optional[Dict[str, Any]] = None,
) -> None:
    """Plot predicted indenter contact force vs depth."""
    k_pk = int(np.argmax(res_on["depths"]))
    load_depths = res_on["depths"][:k_pk + 1]
    load_forces_on = res_on["forces"][:k_pk + 1]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(load_depths, load_forces_on, "o-", color="crimson", linewidth=2.0, label="Cap ON (shell + air overpressure)")

    if res_off is not None:
        load_forces_off = res_off["forces"][:k_pk + 1]
        ax.plot(load_depths, load_forces_off, "s--", color="dodgerblue", linewidth=2.0, label="Cap OFF (shell only)")

    ax.set_xlabel("Indentation depth (mm)")
    ax.set_ylabel("Contact force (N)")
    ax.set_title("Predicted Tool Force vs Depth\n[model prediction, unverified (no force sensor)]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def render_sim_video(
    out_video_path: Path,
    sim: StaticSimulator,
    press_at: float,
    peak_depth: float,
    tool_type: str,
    res_on: Dict[str, Any],
    res_off: Optional[Dict[str, Any]] = None,
    fps: int = 10,
) -> None:
    """Render 3D mesh animation through loading and unloading snapshots."""
    temp_dir = out_video_path.parent / "temp_frames"
    temp_dir.mkdir(parents=True, exist_ok=True)

    tris = sim.tris
    X0 = sim.X0
    n_frames = len(res_on["depths"])

    max_disp = max(peak_depth, 1.0)
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=max_disp)

    frame_paths = []

    for i in range(n_frames):
        d_val = res_on["depths"][i]
        sol_on = res_on["solutions"][i]
        f_on = res_on["forces"][i]
        dp_on = res_on["overpressures"][i]
        disp_on = np.linalg.norm(sol_on - X0, axis=1)

        if res_off is not None:
            # Side-by-side comparison figure
            fig = plt.figure(figsize=(14, 6), dpi=100)
            ax1 = fig.add_subplot(121, projection="3d")
            ax2 = fig.add_subplot(122, projection="3d")

            # Left: Cap ON
            s1 = ax1.plot_trisurf(sol_on[:, 0], sol_on[:, 1], sol_on[:, 2], triangles=tris, cmap="plasma", norm=norm, edgecolor="none")
            s1.set_array(disp_on)
            ax1.set_xlim(0, sim.L)
            ax1.set_ylim(-45, 45)
            ax1.set_zlim(0, 80)
            ax1.view_init(elev=20, azim=-65)
            ax1.set_title(f"Cap ON (Air Overpressure)\nDepth: {d_val:.1f} mm | Force: {f_on:.2f} N* | Δp: {dp_on:.2f} kPa")
            ax1.set_xlabel("X (mm)")
            ax1.set_ylabel("Y (mm)")
            ax1.set_zlabel("Z (mm)")

            # Right: Cap OFF
            sol_off = res_off["solutions"][i]
            f_off = res_off["forces"][i]
            dp_off = res_off["overpressures"][i]
            disp_off = np.linalg.norm(sol_off - X0, axis=1)

            s2 = ax2.plot_trisurf(sol_off[:, 0], sol_off[:, 1], sol_off[:, 2], triangles=tris, cmap="plasma", norm=norm, edgecolor="none")
            s2.set_array(disp_off)
            ax2.set_xlim(0, sim.L)
            ax2.set_ylim(-45, 45)
            ax2.set_zlim(0, 80)
            ax2.view_init(elev=20, azim=-65)
            ax2.set_title(f"Cap OFF (Air Off)\nDepth: {d_val:.1f} mm | Force: {f_off:.2f} N* | Δp: 0.00 kPa")
            ax2.set_xlabel("X (mm)")
            ax2.set_ylabel("Y (mm)")
            ax2.set_zlabel("Z (mm)")

            fig.suptitle(f"Standalone Simulation: {tool_type.capitalize()} Indentation at {press_at:.1f} mm (*unverified force)", fontsize=12)
        else:
            # Single view
            fig = plt.figure(figsize=(9, 6), dpi=100)
            ax = fig.add_subplot(111, projection="3d")
            s = ax.plot_trisurf(sol_on[:, 0], sol_on[:, 1], sol_on[:, 2], triangles=tris, cmap="plasma", norm=norm, edgecolor="none")
            s.set_array(disp_on)
            ax.set_xlim(0, sim.L)
            ax.set_ylim(-45, 45)
            ax.set_zlim(0, 80)
            ax.view_init(elev=20, azim=-65)
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.set_zlabel("Z (mm)")
            status = "Loading" if i <= n_frames // 2 else "Unloading (Elastic Recovery)"
            ax.set_title(
                f"Virtual Press ({status})\n"
                f"Position: {press_at:.1f} mm | Depth: {d_val:.1f} mm\n"
                f"Contact Force: {f_on:.2f} N* | Δp: {dp_on:.2f} kPa\n"
                f"*force is unverified model prediction"
            )

        fpath = temp_dir / f"frame_{i:04d}.png"
        fig.savefig(fpath)
        plt.close(fig)
        frame_paths.append(fpath)

    # Encode with ffmpeg
    cmd = [
        "ffmpeg",
        "-y",
        "-r", str(fps),
        "-i", str(temp_dir / "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(out_video_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Clean up temp frames
    for p in frame_paths:
        try:
            p.unlink()
        except OSError:
            pass
    try:
        temp_dir.rmdir()
    except OSError:
        pass


def run_interactive(
    sim: StaticSimulator,
    kappa: float,
    rho: float,
    gas_exponent: float,
    tool_radius: float = 2.5,
) -> None:
    """Launch interactive matplotlib window with sliders for position, depth, and cap state."""
    if not os.environ.get("DISPLAY"):
        print("[INFO] Interactive mode requires a display (DISPLAY environment variable not set).")
        print("To run simulations headlessly, use the standard command-line options (e.g. --press-at, --depth, --compare-cap).")
        return

    from matplotlib.widgets import CheckButtons, Slider

    try:
        fig = plt.figure(figsize=(11, 7))
    except Exception as e:
        print(f"[ERROR] Could not open GUI display window: {e}")
        return
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)
    plt.subplots_adjust(bottom=0.25)

    # Initial values
    init_x = 60.0
    init_d = 12.0
    init_cap = True

    sol_state = [sim.solve_snapshot(init_d, init_x, r_tool_mm=tool_radius, kappa=kappa, rho=rho, air_on=init_cap)]

    # Draw initial plots
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=20.0)

    def redraw(x_val: float, d_val: float, cap_val: bool) -> None:
        ax3d.clear()
        ax2d.clear()

        sol = sol_state[0]
        disp = np.linalg.norm(sol - sim.X0, axis=1)

        s = ax3d.plot_trisurf(sol[:, 0], sol[:, 1], sol[:, 2], triangles=sim.tris, cmap="plasma", norm=norm, edgecolor="none")
        s.set_array(disp)
        ax3d.set_xlim(0, sim.L)
        ax3d.set_ylim(-45, 45)
        ax3d.set_zlim(0, 80)
        ax3d.view_init(elev=20, azim=-65)
        ax3d.set_title(f"3D Mesh (x={x_val:.1f} mm, d={d_val:.1f} mm, Cap={'ON' if cap_val else 'OFF'})")

        # 2D Side view
        w = sim.extract_profile(sol, mode="top_vertex")
        z = 2.0 * sim.R - w
        ax2d.axhline(2.0 * sim.R, color="gray", linestyle="--", label="Rest (70 mm)")
        ax2d.axhline(0.0, color="black", linestyle="-", label="Floor")
        ax2d.plot(sim.ring_x, z, color="crimson", linewidth=2.0, label="Top profile")
        ax2d.plot([x_val], [2.0 * sim.R - d_val], "v", color="purple", markersize=8, label="Tool")
        ax2d.set_xlim(0, sim.L)
        ax2d.set_ylim(-5, 80)
        ax2d.set_xlabel("x (mm)")
        ax2d.set_ylabel("z (mm)")
        ax2d.set_title("Side Profile")
        ax2d.legend(loc="lower right")
        ax2d.grid(True, alpha=0.3)

        fig.canvas.draw_idle()

    redraw(init_x, init_d, init_cap)

    # Sliders
    ax_x = plt.axes([0.2, 0.12, 0.5, 0.03])
    ax_d = plt.axes([0.2, 0.06, 0.5, 0.03])
    ax_cap = plt.axes([0.78, 0.06, 0.15, 0.09])

    slider_x = Slider(ax_x, "Position (mm)", 15.0, 205.0, valinit=init_x, valstep=1.0)
    slider_d = Slider(ax_d, "Depth (mm)", 0.0, 20.0, valinit=init_d, valstep=0.5)
    check_cap = CheckButtons(ax_cap, ["Cap ON"], [init_cap])

    def update(val: Any = None) -> None:
        x_val = float(slider_x.val)
        d_val = float(slider_d.val)
        cap_val = bool(check_cap.get_status()[0])
        sol_state[0] = sim.solve_snapshot(
            d_val,
            x_val,
            r_tool_mm=tool_radius,
            kappa=kappa,
            rho=rho,
            air_on=cap_val,
            X_init=sol_state[0],
            gtol=1e-6,
            maxiter=20000,
        )
        redraw(x_val, d_val, cap_val)

    slider_x.on_changed(update)
    slider_d.on_changed(update)
    check_cap.on_clicked(update)

    plt.show()


def _compare_cap_worker(args: Tuple[Any, ...]) -> Dict[str, Any]:
    """Worker solving one cap state (on or off) for --compare-cap."""
    p_atm, V0, gas_exponent, depths, press_at, tool_radius, kappa, rho, air_on, gtol, maxiter = args
    sim_local = StaticSimulator(p_atm=p_atm, V0=V0, gas_exponent=gas_exponent)
    return simulate_sequence(
        sim_local,
        depths,
        press_at,
        tool_radius,
        kappa,
        rho,
        air_on=air_on,
        gas_exponent=gas_exponent,
        gtol=gtol,
        maxiter=maxiter,
    )


def run_standalone_simulation(
    press_at: float = 60.0,
    depth: float = 12.0,
    tool: str = "pen",
    cap: str = "on",
    steps: int = 12,
    out_dir: Optional[Path] = None,
    compare_cap: bool = False,
    trajectory_file: Optional[Path] = None,
    config_path: Path = PROJECT_ROOT / "config" / "clips.yaml",
    params_path: Path = PROJECT_ROOT / "out" / "fit" / "params.json",
) -> Dict[str, Any]:
    """Execute standalone simulation, generate plots, video, and summary JSON."""
    cfg, params = load_config_and_params(config_path, params_path)

    tool_radius = float(cfg["finger_radius_mm"] if tool == "finger" else cfg["pen_radius_mm"])
    p_atm = float(cfg.get("p_atm_mpa", 0.101325))
    V0 = float(cfg["bottle"].get("volume_mm3", 1.0e6))
    gas_exponent = float(cfg.get("gas_exponent", 1.4))

    kappa_hat = float(params["kappa_hat"])
    rho_hat = float(params.get("rho_used", params["rho_hat"]))

    # Establish output directory
    if out_dir is None:
        if compare_cap:
            folder_name = f"compare_press_x{press_at:.0f}_d{depth:.0f}_{tool}"
        else:
            folder_name = f"press_x{press_at:.0f}_d{depth:.0f}_{tool}_cap_{cap}"
        out_dir = PROJECT_ROOT / "out" / "sim" / folder_name
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extrapolation warnings
    warnings = compute_extrapolation_warnings(press_at, depth)
    for w in warnings:
        print(f"[WARNING] {w}")

    # Build sequence of depths
    if trajectory_file is not None:
        traj_path = Path(trajectory_file)
        if not traj_path.exists():
            raise FileNotFoundError(f"Trajectory file {traj_path} not found.")
        depths = []
        with open(traj_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                depths.append(float(row["depth_mm"]))
        peak_depth = float(np.max(depths))
    else:
        peak_depth = float(depth)
        d_loading = np.linspace(peak_depth / steps, peak_depth, steps)
        d_unloading = np.linspace(peak_depth, 0.0, steps + 1)[1:]
        depths = [0.0] + list(d_loading) + list(d_unloading)

    # Initialize simulator
    sim = StaticSimulator(p_atm=p_atm, V0=V0, gas_exponent=gas_exponent)

    # Optimization tolerances (held-contact equilibrium)
    gtol = 1e-6
    maxiter = 20000

    # Execute simulation
    if compare_cap:
        print(f"Simulating Cap ON and Cap OFF comparison (press_at={press_at:.1f} mm, peak depth={peak_depth:.1f} mm, steps={steps})...")
        tasks = [
            (p_atm, V0, gas_exponent, depths, press_at, tool_radius, kappa_hat, rho_hat, True, gtol, maxiter),
            (p_atm, V0, gas_exponent, depths, press_at, tool_radius, kappa_hat, rho_hat, False, gtol, maxiter),
        ]
        with ProcessPoolExecutor(max_workers=2) as ex:
            res_on, res_off = list(ex.map(_compare_cap_worker, tasks))
    else:
        air_on = (cap.lower() == "on")
        print(f"Simulating Cap {cap.upper()} (press_at={press_at:.1f} mm, peak depth={peak_depth:.1f} mm, steps={steps})...")
        res_on = simulate_sequence(sim, depths, press_at, tool_radius, kappa_hat, rho_hat, air_on=air_on, gas_exponent=gas_exponent, gtol=gtol, maxiter=maxiter)
        res_off = None

    # Calculate profile ratios at peak
    k_pk = int(np.argmax(res_on["depths"]))
    sol_pk_on = res_on["solutions"][k_pk]
    stations = np.arange(10.0, sim.L - 5.0, 3.0)
    w_pk_on = sim.extract_profile(sol_pk_on, stations_mm=stations, mode="top_vertex")
    ratios_on = compute_profile_ratios(w_pk_on, stations, press_at, peak_depth)

    if res_off is not None:
        sol_pk_off = res_off["solutions"][k_pk]
        w_pk_off = sim.extract_profile(sol_pk_off, stations_mm=stations, mode="top_vertex")
        ratios_off = compute_profile_ratios(w_pk_off, stations, press_at, peak_depth)
    else:
        ratios_off = None

    # Generate output plots
    side_view_path = out_dir / "side_view.png"
    profile_path = out_dir / "profile.png"
    force_depth_path = out_dir / "force_depth.png"
    video_path = out_dir / "sim.mp4"
    summary_path = out_dir / "summary.json"

    print("Generating plots...")
    plot_side_view(side_view_path, sim, press_at, peak_depth, res_on, res_off)
    plot_profile(profile_path, sim, press_at, peak_depth, res_on, res_off)
    plot_force_depth(force_depth_path, res_on, res_off)

    print("Rendering video...")
    render_sim_video(video_path, sim, press_at, peak_depth, tool, res_on, res_off)

    # Compile summary JSON
    pk_force_on = float(np.max(res_on["forces"]))
    pk_dp_on = float(np.max(res_on["overpressures"]))
    pk_dV_on = float(np.max(res_on["volumes_lost"]))

    summary: Dict[str, Any] = {
        "inputs": {
            "press_at_mm": press_at,
            "depth_mm": peak_depth,
            "tool": tool,
            "cap": cap if not compare_cap else "compare",
            "steps": steps,
            "trajectory": str(trajectory_file) if trajectory_file else None,
        },
        "peak_force_N": round(pk_force_on, 3),
        "peak_overpressure_kPa": round(pk_dp_on, 3),
        "delta_V_mm3": round(pk_dV_on, 1),
        "profile_ratios": ratios_on,
        "extrapolation_warnings": warnings,
    }

    if compare_cap and ratios_off is not None:
        summary["cap_on"] = {
            "peak_force_N": round(pk_force_on, 3),
            "peak_overpressure_kPa": round(pk_dp_on, 3),
            "delta_V_mm3": round(pk_dV_on, 1),
            "profile_ratios": ratios_on,
        }
        summary["cap_off"] = {
            "peak_force_N": round(float(np.max(res_off["forces"])), 3),
            "peak_overpressure_kPa": 0.0,
            "delta_V_mm3": round(float(np.max(res_off["volumes_lost"])), 1),
            "profile_ratios": ratios_off,
        }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Simulation completed successfully. Outputs saved to: {out_dir}")
    return {
        "out_dir": out_dir,
        "summary": summary,
        "res_on": res_on,
        "res_off": res_off,
        "sim": sim,
        "peak_profile_on": w_pk_on,
        "stations": stations,
    }


def main() -> None:
    """CLI entry point for standalone simulator."""
    parser = argparse.ArgumentParser(description="Standalone bottle deformation simulator (M8)")
    parser.add_argument("--press-at", type=float, default=60.0, help="Press position in mm from the base (default: 60)")
    parser.add_argument("--depth", type=float, default=12.0, help="Peak indentation in mm (default: 12)")
    parser.add_argument("--tool", type=str, default="pen", choices=["pen", "finger"], help="Tool type ('pen' or 'finger', default: pen)")
    parser.add_argument("--cap", type=str, default="on", choices=["on", "off"], help="Cap state ('on' or 'off', default: on)")
    parser.add_argument("--steps", type=int, default=12, help="Snapshots for loading; same number for unloading (default: 12)")
    parser.add_argument("--out", type=str, default=None, help="Output folder (default: out/sim/<auto-name>/)")
    parser.add_argument("--compare-cap", action="store_true", help="Compare cap on and cap off side by side")
    parser.add_argument("--trajectory", type=str, default=None, help="Path to custom CSV depth profile (columns: t_s, depth_mm)")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive matplotlib slider GUI")

    args = parser.parse_args()

    cfg, params = load_config_and_params()
    tool_radius = float(cfg["finger_radius_mm"] if args.tool == "finger" else cfg["pen_radius_mm"])

    if args.interactive:
        sim = StaticSimulator(
            p_atm=float(cfg.get("p_atm_mpa", 0.101325)),
            V0=float(cfg["bottle"].get("volume_mm3", 1.0e6)),
            gas_exponent=float(cfg.get("gas_exponent", 1.4)),
        )
        run_interactive(
            sim,
            kappa=float(params["kappa_hat"]),
            rho=float(params.get("rho_used", params["rho_hat"])),
            gas_exponent=float(cfg.get("gas_exponent", 1.4)),
            tool_radius=tool_radius,
        )
    else:
        run_standalone_simulation(
            press_at=args.press_at,
            depth=args.depth,
            tool=args.tool,
            cap=args.cap,
            steps=args.steps,
            out_dir=Path(args.out) if args.out else None,
            compare_cap=args.compare_cap,
            trajectory_file=Path(args.trajectory) if args.trajectory else None,
        )


if __name__ == "__main__":
    main()
