#!/usr/bin/env python3
"""M2 — Scale & Observation (`src/scale.py`).

Measures per-clip pixel-to-mm scale `s` from rest frames (bottle diameter
between top silhouette edge and bottom table-contact edge, cross-checked against
clear cylindrical span), maps coordinates to mm from the bottle base, computes
full-frame `w_obs_mm[T, S]`, `tool_x_mm[T]`, `tip_drop_mm[T]`, and subsampled
loading-phase observations over all unoccluded stations in `[10, 112] mm`.
"""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.track import load_config


def measure_scale_and_pen(
    clip_id: str,
    cfg: Dict[str, Any],
    rest_px: np.ndarray,
    tool_x_px: np.ndarray,
    tip_px: np.ndarray,
    contact: np.ndarray,
) -> Dict[str, Any]:
    """Measure pixel-to-mm scale s from bottle diameter on a rest frame and cross-check length."""
    video_dir = Path(cfg["video_dir"])
    c = cfg["clips"][clip_id]
    bx0, bx1 = int(c["body_x"][0]), int(c["body_x"][1])
    rf0, rf1 = int(c["rest_frames"][0]), int(c["rest_frames"][1])
    diam_mm = float(cfg["bottle"]["diameter_mm"])
    clear_len_mm = float(cfg["bottle"]["label_mm_from_base"][0])  # 115.0 mm clear body span

    cap = cv2.VideoCapture(str(video_dir / c["file"]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, (rf0 + rf1) // 2)
    ret, frame = cap.read()
    assert ret, f"Clip {clip_id}: failed to read rest frame {(rf0 + rf1) // 2}"
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Inner [20%, 80%] of the cylindrical clear-body span
    c0 = int(bx0 + 0.2 * (bx1 - bx0))
    c1 = int(bx0 + 0.8 * (bx1 - bx0))
    with np.errstate(all="ignore"):
        y_top_med = float(np.nanmedian(rest_px[c0:c1]))

    d_est = float(c.get("diameter_px", 490))
    r0 = max(0, int(round(y_top_med + d_est - 15)))
    r1 = min(frame.shape[0], int(round(y_top_med + d_est + 16)))
    v_patch = hsv[r0:r1, c0:c1, 2]
    s_patch = hsv[r0:r1, c0:c1, 1]
    comb = np.median(v_patch - 0.8 * s_patch, axis=1)
    idx_reach = np.where(comb >= np.percentile(comb, 80))[0]
    y_bot = float(r0 + int(idx_reach[0])) if len(idx_reach) > 0 else float(y_top_med + d_est)

    with np.errstate(all="ignore"):
        d_px = float(np.nanmedian(y_bot - rest_px[c0:c1]))
    s_diam = d_px / diam_mm

    full_clear_span_clips = {
        "001531",
        "001614",
        "001656",
        "001711",
        "001730",
        "161638",
        "161708",
        "161747",
        "161802",
    }
    both_bounds_visible = clip_id in full_clear_span_clips
    s_len = (bx1 - bx0) / clear_len_mm if both_bounds_visible else np.nan

    # Measure pen contacting tip radius (5 px above tip_px at mid-contact) if pen clip
    pen_radius_mm = float(cfg["pen_radius_mm"])
    c_idx = np.where(contact)[0]
    if c["tool"] == "pen" and len(c_idx) > 0:
        f_mid = int(c_idx[len(c_idx) // 2])
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_mid)
        ret_m, frame_m = cap.read()
        if ret_m and not np.isnan(tip_px[f_mid]) and not np.isnan(tool_x_px[f_mid]):
            hsv_m = cv2.cvtColor(frame_m, cv2.COLOR_BGR2HSV)
            r_tip = max(0, int(round(float(tip_px[f_mid]))) - 5)
            tx_i = int(round(float(tool_x_px[f_mid])))
            v_row = hsv_m[r_tip, max(0, tx_i - 60) : min(frame_m.shape[1], tx_i + 61), 2] < 30
            cols = np.where(v_row)[0]
            if len(cols) > 0:
                w_px = float(cols[-1] - cols[0] + 1)
                pen_radius_mm = float(w_px / (2.0 * s_diam))

    cap.release()
    return {
        "s_px_per_mm": float(s_diam),
        "s_len_px_per_mm": float(s_len) if not np.isnan(s_len) else np.nan,
        "both_shoulders_visible": bool(both_bounds_visible),
        "diameter_px_measured": float(d_px),
        "y_bot_px": float(y_bot),
        "pen_radius_mm_measured": float(pen_radius_mm),
    }


def scale_clip(clip_id: str, cfg: Dict[str, Any], tracks_dir: Path) -> Dict[str, Any]:
    """Convert raw pixel tracks for `clip_id` into mm coordinates and station observations."""
    npz_path = tracks_dir / f"{clip_id}.npz"
    assert npz_path.exists(), f"Missing M1 track file: {npz_path}"
    tr = np.load(npz_path, allow_pickle=True)

    c = cfg["clips"][clip_id]
    bx0, bx1 = int(c["body_x"][0]), int(c["body_x"][1])
    rf0, rf1 = int(c["rest_frames"][0]), int(c["rest_frames"][1])

    edge_px = tr["edge_px"].copy()
    tool_x_px = tr["tool_x_px"]
    tip_px = tr["tip_px"]
    contact = tr["contact"]
    flags = tr["flags"].item()
    n_frames = edge_px.shape[0]

    # Spatial outlier filter: reject narrow specular/background glitches deviating > 15 px
    # from the 121-px spatial moving median along the bottle silhouette.
    sub = edge_px[:, bx0:bx1]
    padded = np.pad(sub, ((0, 0), (60, 60)), mode="edge")
    with np.errstate(all="ignore"):
        med121 = np.nanmedian(
            np.lib.stride_tricks.sliding_window_view(padded, 121, axis=1), axis=2
        )
    sub[np.abs(sub - med121) > 15.0] = np.nan
    edge_px[:, bx0:bx1] = sub

    with np.errstate(all="ignore"):
        rest_px = np.nanmedian(edge_px[rf0 : rf1 + 1], axis=0).astype(np.float32)

    # Interpolate interior NaNs in rest_px (where hovering tool was masked), preserving
    # exterior NaNs (such as the foreground container occlusion on 001614).
    sub_r = rest_px[bx0:bx1].copy()
    valid_r = np.where(~np.isnan(sub_r))[0]
    if len(valid_r) >= 2:
        f_v, l_v = int(valid_r[0]), int(valid_r[-1])
        isl = sub_r[f_v : l_v + 1]
        nans = np.isnan(isl)
        if np.any(nans):
            x_i = np.arange(len(isl))
            isl[nans] = np.interp(x_i[nans], x_i[~nans], isl[~nans])
            sub_r[f_v : l_v + 1] = isl
            rest_px[bx0:bx1] = sub_r

    scale_info = measure_scale_and_pen(
        clip_id=clip_id,
        cfg=cfg,
        rest_px=rest_px,
        tool_x_px=tool_x_px,
        tip_px=tip_px,
        contact=contact,
    )
    s = scale_info["s_px_per_mm"]
    x_base_px = float(bx1)
    tool_x_mm_series = ((x_base_px - tool_x_px) / s).astype(np.float32)

    c_idx = np.where(contact)[0]
    if len(c_idx) == 0:
        out_path = tracks_dir / f"{clip_id}_mm.npz"
        np.savez_compressed(
            out_path,
            s=np.float32(s),
            s_px_per_mm=np.float32(s),
            s_len_px_per_mm=np.float32(scale_info["s_len_px_per_mm"]),
            both_shoulders_visible=np.bool_(scale_info["both_shoulders_visible"]),
            x_base_px=np.float32(x_base_px),
            x_mm=np.zeros(0, dtype=np.float32),
            w_obs_mm=np.zeros((n_frames, 0), dtype=np.float32),
            tool_x_mm=tool_x_mm_series,
            tip_drop_mm=np.full(n_frames, np.nan, dtype=np.float32),
            stations_mm=np.zeros(0, dtype=np.float32),
            u_mm=np.zeros((0, 0), dtype=np.float32),
            d_mm=np.zeros(0, dtype=np.float32),
            tool_x_med_mm=np.float32(np.nan),
            idx=np.zeros(0, dtype=np.int32),
            flags=np.array(flags, dtype=object),
        )
        return {"clip_id": clip_id, "s": s, "K": 0, "M": 0, "tool_x_mm": np.nan}

    tx_med_px = float(np.nanmedian(tool_x_px[c_idx]))
    tool_x_med_mm = float((x_base_px - tx_med_px) / s)

    col_tool = int(np.clip(round(tx_med_px), bx0, bx1 - 1))
    ref_at_tool = float(rest_px[col_tool])
    if np.isnan(ref_at_tool):
        with np.errstate(all="ignore"):
            ref_at_tool = float(np.nanmedian(rest_px[bx0:bx1]))

    tip_drop_mm = ((tip_px - ref_at_tool) / s).astype(np.float32)

    # Loading-phase subsample (K <= 25): from contact start up to t_peak = argmax(tip_drop_mm)
    # where tip_drop_mm >= 1.5 mm, sorted by tip_drop_mm and thinned by >= 0.3 mm.
    c_start, c_end = int(c_idx[0]), int(c_idx[-1])
    t_peak = int(c_start + np.nanargmax(tip_drop_mm[c_start : c_end + 1]))
    cand_frames = [t for t in range(c_start, t_peak + 1) if float(tip_drop_mm[t]) >= 1.5]
    cand_frames.sort(key=lambda t: float(tip_drop_mm[t]))

    thinned: List[int] = []
    for t in cand_frames:
        if len(thinned) == 0 or (float(tip_drop_mm[t]) - float(tip_drop_mm[thinned[-1]])) >= 0.3:
            thinned.append(t)
    if cand_frames and thinned[-1] != cand_frames[-1]:
        if (
            len(thinned) >= 2
            and (float(tip_drop_mm[cand_frames[-1]]) - float(tip_drop_mm[thinned[-2]])) >= 0.3
        ):
            thinned[-1] = cand_frames[-1]
        else:
            thinned.append(cand_frames[-1])
    if len(thinned) > 25:
        sel = np.round(np.linspace(0, len(thinned) - 1, 25)).astype(int)
        thinned = [thinned[i] for i in sel]

    idx = np.array(thinned, dtype=np.int32)
    d_mm = tip_drop_mm[idx].astype(np.float32)

    # Fixed stations: x_mm in [10, 112] step 3 mm, dropping |x_mm - tool_x_med_mm| < tool_exclusion_mm.
    # Keep all unoccluded stations over [10, 112] mm using the same rule for every clip.
    excl_mm = float(cfg["tool_exclusion_mm"][c["tool"]])
    hw_px = max(3, int(round(1.5 * s)))  # +/- 1.5 mm window
    cand_stations = [xm for xm in np.arange(10.0, 115.0, 3.0) if abs(xm - tool_x_med_mm) >= excl_mm]

    kept_stations: List[float] = []
    u_cols: List[np.ndarray] = []
    w_full_cols: List[np.ndarray] = []
    for xm in cand_stations:
        x_px = int(round(x_base_px - xm * s))
        if x_px - hw_px < bx0 or x_px + hw_px + 1 > bx1:
            continue
        r_slice = rest_px[x_px - hw_px : x_px + hw_px + 1]
        if np.all(np.isnan(r_slice)):
            continue
        diff_all = (edge_px[:, x_px - hw_px : x_px + hw_px + 1] - r_slice[None, :]) / s
        with np.errstate(all="ignore"):
            w_all = np.nanmedian(diff_all, axis=1).astype(np.float32)
        u_k = w_all[idx].copy()
        # Unoccluded station rule: valid at peak and on >= 80% of loading frames
        if not np.isnan(u_k[-1]) and np.mean(~np.isnan(u_k)) >= 0.80:
            if np.any(np.isnan(u_k)):
                nans = np.isnan(u_k)
                u_k[nans] = np.interp(d_mm[nans], d_mm[~nans], u_k[~nans])
            kept_stations.append(float(xm))
            u_cols.append(u_k)
            w_full_cols.append(w_all)

    x_mm = np.array(kept_stations, dtype=np.float32)
    u_mm = (
        np.column_stack(u_cols).astype(np.float32)
        if u_cols
        else np.zeros((len(idx), 0), dtype=np.float32)
    )
    w_obs_mm = (
        np.column_stack(w_full_cols).astype(np.float32)
        if w_full_cols
        else np.zeros((n_frames, 0), dtype=np.float32)
    )

    out_path = tracks_dir / f"{clip_id}_mm.npz"
    np.savez_compressed(
        out_path,
        s=np.float32(s),
        s_px_per_mm=np.float32(s),
        s_len_px_per_mm=np.float32(scale_info["s_len_px_per_mm"]),
        both_shoulders_visible=np.bool_(scale_info["both_shoulders_visible"]),
        x_base_px=np.float32(x_base_px),
        x_mm=x_mm,
        w_obs_mm=w_obs_mm,
        tool_x_mm=tool_x_mm_series,
        tip_drop_mm=tip_drop_mm,
        stations_mm=x_mm,
        u_mm=u_mm,
        d_mm=d_mm,
        tool_x_med_mm=np.float32(tool_x_med_mm),
        idx=idx,
        flags=np.array(flags, dtype=object),
    )

    return {
        "clip_id": clip_id,
        "s": s,
        "K": len(idx),
        "M": len(x_mm),
        "tool_x_mm": tool_x_med_mm,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="M2: Scale tracks to mm and extract stations.")
    parser.add_argument("--config", type=Path, default=Path("config/clips.yaml"))
    parser.add_argument("--tracks-dir", type=Path, default=Path("out/tracks"))
    parser.add_argument("--clips", type=str, default=None, help="Comma-separated clip IDs")
    parser.add_argument("--all", action="store_true", help="Process all clips")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.clips:
        clip_ids = [c.strip() for c in args.clips.split(",") if c.strip()]
    else:
        clip_ids = list(cfg["clips"].keys())

    for cid in clip_ids:
        res = scale_clip(cid, cfg, args.tracks_dir)
        print(
            f"OK {cid}: s={res['s']:.3f} px/mm, tool_x={res['tool_x_mm']:.1f} mm, "
            f"K={res['K']}, M={res['M']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
