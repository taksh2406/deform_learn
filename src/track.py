#!/usr/bin/env python3
"""M1 — Video Tracking (raw pixels).

Extracts per-frame top silhouette edge `edge_px[t, x]`, resting edge `rest_px[x]`,
post-release edge `post_px[x]`, tool tip coordinates `(tool_x_px[t], tip_px[t])`,
contact mask `contact[t]`, and quality flags for each clip in `config/clips.yaml`.
Also renders `out/tracks/<clip>_overlay.mp4` and visual gate `out/tracks/<clip>_gate.png`.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml


def load_config(config_path: Path = Path("config/clips.yaml")) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_manual_checks(path: Path = Path("config/manual_checks.yaml")) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("clips", {}) if data else {}


def assert_orientation(frame_bgr: np.ndarray, clip_id: str) -> None:
    """Assert bottle/table is at bottom (bright white table) and green backdrop at top."""
    v_channel = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    h = v_channel.shape[0]
    top_15 = float(np.mean(v_channel[: int(0.15 * h), :]))
    bot_15 = float(np.mean(v_channel[int(0.85 * h) :, :]))
    assert bot_15 > top_15, (
        f"Clip {clip_id}: orientation check failed (bottom 15% mean V={bot_15:.1f} "
        f"<= top 15% mean V={top_15:.1f})"
    )


def detect_edge_and_tool_frame(
    frame_bgr: np.ndarray,
    bx0: int,
    bx1: int,
    y0: int,
    y1: int,
    otsu_s: int,
    tool_kind: str,
    allow_tool: bool = True,
) -> Tuple[np.ndarray, float, float]:
    """Run M1 HSV edge and tool detector on a single BGR frame within ROI."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    s_roi = hsv[y0:y1, bx0:bx1, 1]
    h_roi, w_roi = s_roi.shape

    gt = (s_roi > otsu_s).astype(np.int32)
    le = (s_roi <= otsu_s).astype(np.int32)
    cs_gt = np.vstack([np.zeros((1, w_roi), dtype=np.int32), np.cumsum(gt, axis=0)])
    cs_le = np.vstack([np.zeros((1, w_roi), dtype=np.int32), np.cumsum(le, axis=0)])

    # Candidate row index r in [20, h_roi - 15] corresponds to image row y = y0 + r
    above = cs_gt[20 : h_roi - 14] - cs_gt[0 : h_roi - 34]
    below = cs_le[35 : h_roi + 1] - cs_le[20 : h_roi - 14]
    cond = (above >= 17) & (below >= 13)
    has_match = cond.any(axis=0)
    first_idx = np.argmax(cond, axis=0)
    ey = np.where(has_match, y0 + 20 + first_idx, np.nan).astype(np.float64)

    # 9-px moving nanmedian along x
    padded = np.pad(ey, (4, 4), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 9)
    with np.errstate(all="ignore"):
        ey = np.nanmedian(windows, axis=1)

    tx = np.nan
    tip = np.nan
    if allow_tool and tool_kind in ("pen", "finger"):
        if tool_kind == "pen":
            v_roi = hsv[y0:y1, bx0:bx1, 2]
            tmask = v_roi < 30
            half_width = 60
        else:
            h_ch = hsv[y0:y1, bx0:bx1, 0]
            v_roi = hsv[y0:y1, bx0:bx1, 2]
            tmask = ((h_ch < 22) | (h_ch > 170)) & (s_roi > 50) & (v_roi > 90)
            half_width = 110

        col_cnt = tmask.sum(axis=0)
        tcols = np.where(col_cnt >= 40)[0]
        if len(tcols) > 0:
            tx = float(bx0 + np.median(tcols))
            lowest = y0 + (h_roi - 1 - np.argmax(tmask[:, tcols][::-1, :], axis=0))
            tip = float(np.percentile(lowest, 95))
            c0 = max(0, int(round(tx - half_width)) - bx0)
            c1 = min(w_roi, int(round(tx + half_width)) - bx0 + 1)
            ey[c0:c1] = np.nan

    return ey.astype(np.float32), tx, tip


def track_clip(
    clip_id: str,
    cfg: Dict[str, Any],
    out_dir: Path,
    manual_checks: Optional[List[Dict[str, Any]]] = None,
    write_video: bool = True,
) -> Dict[str, Any]:
    """Track a single clip, verify assertions, and save .npz, overlay.mp4, and gate.png."""
    video_dir = Path(cfg["video_dir"])
    fps = float(cfg["fps"])
    c = cfg["clips"][clip_id]
    video_path = video_dir / c["file"]
    assert video_path.exists(), f"Missing video file: {video_path}"

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Failed to open video: {video_path}"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames_expected = int(c["n_frames"])

    bx0, bx1 = int(c["body_x"][0]), int(c["body_x"][1])
    y0, y1 = int(c["edge_rows"][0]), int(c["edge_rows"][1])
    otsu_s = int(c["otsu_s"])
    tool_kind = str(c["tool"])
    ignore_before = int(c.get("ignore_tool_before", 0))
    rf0, rf1 = int(c["rest_frames"][0]), int(c["rest_frames"][1])

    edge_px = np.full((n_frames_expected, w), np.nan, dtype=np.float32)
    tool_x_px = np.full(n_frames_expected, np.nan, dtype=np.float32)
    tip_px = np.full(n_frames_expected, np.nan, dtype=np.float32)

    frames_cache: Dict[int, np.ndarray] = {}
    gate_frame_indices = set()
    if manual_checks:
        for pt in manual_checks:
            gate_frame_indices.add(int(pt["frame"]))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx == 0:
            assert_orientation(frame, clip_id)
        if idx in gate_frame_indices:
            frames_cache[idx] = frame.copy()

        ey, tx, tip = detect_edge_and_tool_frame(
            frame,
            bx0=bx0,
            bx1=bx1,
            y0=y0,
            y1=y1,
            otsu_s=otsu_s,
            tool_kind=tool_kind,
            allow_tool=(idx >= ignore_before),
        )
        edge_px[idx, bx0:bx1] = ey
        tool_x_px[idx] = tx
        tip_px[idx] = tip
        idx += 1

    cap.release()
    assert idx == n_frames_expected, f"Clip {clip_id}: read {idx} frames, expected {n_frames_expected}"

    t_sec = (np.arange(n_frames_expected, dtype=np.float64) / fps).astype(np.float32)

    assert rf1 - rf0 + 1 >= 10, (
        f"Clip {clip_id}: rest_frames [{rf0}, {rf1}] length {rf1 - rf0 + 1} < 10 frames"
    )

    with np.errstate(all="ignore"):
        rest_px = np.nanmedian(edge_px[rf0 : rf1 + 1], axis=0).astype(np.float32)
        rf_col_std = np.nanstd(edge_px[rf0 : rf1 + 1, bx0:bx1], axis=0)
        rf_median_std = float(np.nanmedian(rf_col_std))

    # Interpolate only interior NaN gaps in rest_px[bx0:bx1] (where hovering tool was masked),
    # preserving exterior NaNs (such as container occlusion at the left end of 001614).
    sub_rest = rest_px[bx0:bx1].copy()
    valid_idx = np.where(~np.isnan(sub_rest))[0]
    if len(valid_idx) >= 2:
        first_v, last_v = int(valid_idx[0]), int(valid_idx[-1])
        interior_slice = sub_rest[first_v : last_v + 1]
        int_nans = np.isnan(interior_slice)
        if np.any(int_nans):
            x_int = np.arange(len(interior_slice))
            interior_slice[int_nans] = np.interp(
                x_int[int_nans], x_int[~int_nans], interior_slice[~int_nans]
            )
            sub_rest[first_v : last_v + 1] = interior_slice
            rest_px[bx0:bx1] = sub_rest

    # Assertion: rest_frames has edge std <= 2 px over body_x (with tool columns masked)
    assert rf_median_std <= 2.0, (
        f"Clip {clip_id}: rest_frames [{rf0}, {rf1}] edge std {rf_median_std:.2f} px > 2.0 px"
    )

    # Contact mask: tip_px - rest_px[tool_x_px] > 5 px
    contact = np.zeros(n_frames_expected, dtype=bool)
    with np.errstate(all="ignore"):
        rest_body_median = float(np.nanmedian(rest_px[bx0:bx1]))
    for f_idx in range(n_frames_expected):
        tx = tool_x_px[f_idx]
        if not np.isnan(tx):
            col_i = int(np.clip(round(float(tx)), bx0, bx1 - 1))
            ref_y = float(rest_px[col_i])
            if np.isnan(ref_y):
                ref_y = rest_body_median
            if float(tip_px[f_idx]) - ref_y > 5.0:
                contact[f_idx] = True

    # Assertion: no contact during rest_frames
    contact_in_rest = int(np.sum(contact[rf0 : rf1 + 1]))
    assert contact_in_rest == 0, (
        f"Clip {clip_id}: rest_frames [{rf0}, {rf1}] has {contact_in_rest} contact frames"
    )

    c_indices = np.where(contact)[0]
    if len(c_indices) > 0:
        c_start, c_end = int(c_indices[0]), int(c_indices[-1])
        # Assertion: rest_frames ends before contact start
        assert rf1 < c_start, (
            f"Clip {clip_id}: rest_frames end {rf1} is not before contact start {c_start}"
        )
        post_start = min(n_frames_expected - 1, c_end + 15)
        post_end = min(n_frames_expected, c_end + 46)
        with np.errstate(all="ignore"):
            post_px = np.nanmedian(edge_px[post_start:post_end], axis=0).astype(np.float32)
            residual_px = float(np.nanmedian(np.abs(post_px[bx0:bx1] - rest_px[bx0:bx1])))
            signed_residual_px = float(np.nanmedian(post_px[bx0:bx1] - rest_px[bx0:bx1]))

            # End-point time series for drift_timing classification
            valid_cols = np.where(np.mean(~np.isnan(edge_px[:, bx0:bx1]), axis=0) > 0.5)[0]
            l_cols = bx0 + valid_cols[:100]
            r_cols = bx0 + valid_cols[-150:-50]
            lbl_ts = np.nanmedian(edge_px[:, l_cols] - rest_px[None, l_cols], axis=1)
            base_ts = np.nanmedian(edge_px[:, r_cols] - rest_px[None, r_cols], axis=1)
            f_peak = int(c_start + np.nanargmax(tip_px[c_start : c_end + 1]))
            end_ts_summary = {
                "lbl_early_load_px": float(np.nanmedian(lbl_ts[c_start : min(c_end + 1, c_start + 5)])),
                "lbl_peak_px": float(lbl_ts[f_peak]),
                "lbl_contact_end_px": float(np.nanmedian(lbl_ts[max(c_start, c_end - 2) : c_end + 1])),
                "lbl_just_released_px": float(np.nanmedian(lbl_ts[c_end + 1 : min(n_frames_expected, c_end + 9)])),
                "lbl_post_px": float(np.nanmedian(lbl_ts[post_start:post_end])),
                "base_early_load_px": float(np.nanmedian(base_ts[c_start : min(c_end + 1, c_start + 5)])),
                "base_peak_px": float(base_ts[f_peak]),
                "base_contact_end_px": float(np.nanmedian(base_ts[max(c_start, c_end - 2) : c_end + 1])),
                "base_just_released_px": float(np.nanmedian(base_ts[c_end + 1 : min(n_frames_expected, c_end + 9)])),
                "base_post_px": float(np.nanmedian(base_ts[post_start:post_end])),
            }
    else:
        c_start, c_end = -1, -1
        post_px = rest_px.copy()
        residual_px = 0.0
        signed_residual_px = 0.0
        end_ts_summary = {}

    is_residual = bool(residual_px > 3.0)
    if not is_residual:
        drift_timing = "none"
    else:
        # Classify residual timing from end-point time series:
        # If the bottle stayed positive during loading and sprang upward (negative) only upon release
        # (or camera wobbled at release with near-zero signed body drift), classify as 'at_release';
        # if the label/base end shifted downward in the loading direction and stayed settled, 'during_loading'.
        if signed_residual_px <= 2.0 or end_ts_summary.get("base_post_px", 0.0) < -5.0:
            drift_timing = "at_release"
        else:
            drift_timing = "during_loading"

    flags: Dict[str, Any] = {
        "residual": is_residual,
        "drift_timing": drift_timing,
        "residual_px": residual_px,
        "signed_residual_px": signed_residual_px,
        "rest_std_px": rf_median_std,
        "contact_window": [c_start, c_end] if c_start >= 0 else None,
        "camera_moved": bool(c.get("camera_moved", False)),
        "end_timeseries": end_ts_summary,
    }

    if clip_id == "161638":
        # Compute idle edge noise sigma_px excluding bump at frames 55-75
        # and camera grab at frames 110..112; de-step the two stationary segments.
        with np.errstate(all="ignore"):
            seg1 = edge_px[:55, bx0:bx1] - np.nanmedian(edge_px[:55, bx0:bx1], axis=0, keepdims=True)
            seg2 = edge_px[76:110, bx0:bx1] - np.nanmedian(
                edge_px[76:110, bx0:bx1], axis=0, keepdims=True
            )
            comb = np.vstack([seg1, seg2])
            sigma_px = float(np.nanmedian(np.nanstd(comb, axis=0)))
            sigma_pre_bump_px = float(np.nanmedian(np.nanstd(edge_px[:55, bx0:bx1], axis=0)))
        flags["sigma_px"] = sigma_px
        flags["sigma_pre_bump_px"] = sigma_pre_bump_px
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "sigma_px.json", "w", encoding="utf-8") as f:
            json.dump({"sigma_px": sigma_px, "sigma_pre_bump_px": sigma_pre_bump_px}, f, indent=2)

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{clip_id}.npz"
    np.savez_compressed(
        npz_path,
        t=t_sec,
        edge_px=edge_px,
        rest_px=rest_px,
        post_px=post_px,
        tool_x_px=tool_x_px,
        tip_px=tip_px,
        contact=contact,
        flags=np.array(flags, dtype=object),
    )

    if manual_checks:
        render_gate_png(
            clip_id=clip_id,
            frames_cache=frames_cache,
            edge_px=edge_px,
            rest_px=rest_px,
            tool_x_px=tool_x_px,
            tip_px=tip_px,
            manual_checks=manual_checks,
            bx0=bx0,
            bx1=bx1,
            out_path=out_dir / f"{clip_id}_gate.png",
        )

    if write_video:
        render_overlay_video(
            video_path=video_path,
            out_path=out_dir / f"{clip_id}_overlay.mp4",
            fps=fps,
            w=w,
            h=h,
            bx0=bx0,
            bx1=bx1,
            edge_px=edge_px,
            rest_px=rest_px,
            tool_x_px=tool_x_px,
            tip_px=tip_px,
            contact=contact,
            clip_id=clip_id,
        )

    return {
        "clip_id": clip_id,
        "npz_path": npz_path,
        "flags": flags,
    }


def render_gate_png(
    clip_id: str,
    frames_cache: Dict[int, np.ndarray],
    edge_px: np.ndarray,
    rest_px: np.ndarray,
    tool_x_px: np.ndarray,
    tip_px: np.ndarray,
    manual_checks: List[Dict[str, Any]],
    bx0: int,
    bx1: int,
    out_path: Path,
) -> None:
    """Render 3-panel visual gate PNG (rest, mid, peak) with detected & manual points."""
    by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for pt in manual_checks:
        fn = int(pt["frame"])
        by_frame.setdefault(fn, []).append(pt)

    panels = []
    xs = np.arange(bx0, bx1, dtype=np.int32)
    for fn in sorted(by_frame.keys()):
        if fn not in frames_cache:
            continue
        vis = frames_cache[fn].copy()
        # Draw rest_px in cyan
        ry = rest_px[bx0:bx1]
        valid_r = ~np.isnan(ry)
        if np.any(valid_r):
            pts_r = np.column_stack([xs[valid_r], np.round(ry[valid_r]).astype(np.int32)])
            cv2.polylines(vis, [pts_r], isClosed=False, color=(255, 255, 0), thickness=1)

        # Draw detected edge_px in bright green
        ey = edge_px[fn, bx0:bx1]
        valid_e = ~np.isnan(ey)
        if np.any(valid_e):
            pts_e = np.column_stack([xs[valid_e], np.round(ey[valid_e]).astype(np.int32)])
            cv2.polylines(vis, [pts_e], isClosed=False, color=(0, 255, 0), thickness=2)

        # Draw tool tip if present
        tx, tip = float(tool_x_px[fn]), float(tip_px[fn])
        if not np.isnan(tx) and not np.isnan(tip):
            cv2.drawMarker(
                vis,
                (int(round(tx)), int(round(tip))),
                color=(0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=2,
            )

        # Draw manual check points and error text
        stage = by_frame[fn][0].get("stage", "")
        for pt in by_frame[fn]:
            col = int(pt["col"])
            man_y = int(pt["y_px"])
            det_y = float(edge_px[fn, col])
            err = det_y - man_y
            cv2.circle(vis, (col, man_y), 6, (0, 165, 255), 2)
            cv2.putText(
                vis,
                f"x={col} err={err:+.1f}px",
                (col - 60, max(30, man_y - 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            vis,
            f"Clip {clip_id} | {stage} (frame {fn})",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        # Downscale by 0.5 for compact multi-panel gate image
        small = cv2.resize(vis, (vis.shape[1] // 2, vis.shape[0] // 2))
        panels.append(small)

    if panels:
        combined = np.vstack(panels)
        cv2.imwrite(str(out_path), combined)


def render_overlay_video(
    video_path: Path,
    out_path: Path,
    fps: float,
    w: int,
    h: int,
    bx0: int,
    bx1: int,
    edge_px: np.ndarray,
    rest_px: np.ndarray,
    tool_x_px: np.ndarray,
    tip_px: np.ndarray,
    contact: np.ndarray,
    clip_id: str,
) -> None:
    """Render overlay MP4 showing tracked edge, rest reference, and tool tip."""
    out_w, out_h = w // 2, h // 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
    cap = cv2.VideoCapture(str(video_path))
    xs = np.arange(bx0, bx1, dtype=np.int32)
    ry = rest_px[bx0:bx1]
    valid_r = ~np.isnan(ry)
    pts_r = (
        np.column_stack([xs[valid_r] // 2, np.round(ry[valid_r] / 2.0).astype(np.int32)])
        if np.any(valid_r)
        else None
    )

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (out_w, out_h))
        if pts_r is not None:
            cv2.polylines(small, [pts_r], isClosed=False, color=(255, 255, 0), thickness=1)

        ey = edge_px[idx, bx0:bx1]
        valid_e = ~np.isnan(ey)
        if np.any(valid_e):
            pts_e = np.column_stack(
                [xs[valid_e] // 2, np.round(ey[valid_e] / 2.0).astype(np.int32)]
            )
            cv2.polylines(small, [pts_e], isClosed=False, color=(0, 255, 0), thickness=2)

        tx, tip = float(tool_x_px[idx]), float(tip_px[idx])
        if not np.isnan(tx) and not np.isnan(tip):
            color = (0, 0, 255) if contact[idx] else (0, 165, 255)
            cv2.circle(small, (int(round(tx / 2.0)), int(round(tip / 2.0))), 5, color, -1)

        status = "CONTACT" if contact[idx] else "IDLE"
        cv2.putText(
            small,
            f"{clip_id} f={idx} [{status}]",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(small)
        idx += 1

    cap.release()
    writer.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="M1: Track bottle edge and tool tip from videos.")
    parser.add_argument("--config", type=Path, default=Path("config/clips.yaml"))
    parser.add_argument("--manual-checks", type=Path, default=Path("config/manual_checks.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("out/tracks"))
    parser.add_argument("--clips", type=str, default=None, help="Comma-separated clip IDs")
    parser.add_argument("--all", action="store_true", help="Track all clips in config")
    parser.add_argument("--no-video", action="store_true", help="Skip overlay MP4 rendering")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manual_all = load_manual_checks(args.manual_checks)

    if args.clips:
        clip_ids = [c.strip() for c in args.clips.split(",") if c.strip()]
    else:
        clip_ids = list(cfg["clips"].keys())

    for cid in clip_ids:
        res = track_clip(
            clip_id=cid,
            cfg=cfg,
            out_dir=args.out_dir,
            manual_checks=manual_all.get(cid),
            write_video=not args.no_video,
        )
        fl = res["flags"]
        print(
            f"OK {cid}: contact={fl['contact_window']} rest_std={fl['rest_std_px']:.2f}px "
            f"res={fl['residual_px']:.2f}px (flag={fl['residual']}, timing={fl['drift_timing']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

