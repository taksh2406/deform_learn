"""M1 unit tests for video tracking (`src/track.py`)."""

from pathlib import Path
import cv2
import numpy as np
import yaml

from src.track import assert_orientation, load_config, load_manual_checks


CONFIG_PATH = Path("config/clips.yaml")
MANUAL_PATH = Path("config/manual_checks.yaml")
TRACKS_DIR = Path("out/tracks")


def _load_track(clip_id: str):
    path = TRACKS_DIR / f"{clip_id}.npz"
    assert path.exists(), f"Missing track output: {path}"
    data = np.load(path, allow_pickle=True)
    return {
        "t": data["t"],
        "edge_px": data["edge_px"],
        "rest_px": data["rest_px"],
        "post_px": data["post_px"],
        "tool_x_px": data["tool_x_px"],
        "tip_px": data["tip_px"],
        "contact": data["contact"],
        "flags": data["flags"].item(),
    }


def test_orientation_all_clips():
    """Orientation assertion passes on every clip (bottom 15% mean V > top 15% mean V)."""
    cfg = load_config(CONFIG_PATH)
    video_dir = Path(cfg["video_dir"])
    for cid, c in cfg["clips"].items():
        cap = cv2.VideoCapture(str(video_dir / c["file"]))
        ret, frame = cap.read()
        cap.release()
        assert ret, f"Could not read first frame of {cid}"
        assert_orientation(frame, cid)


def test_idle_noise_161638():
    """On 161638 (idle, excluding bump at frames 55-75), edge std < 2 px and recorded as sigma_px."""
    tr = _load_track("161638")
    sigma_px = float(tr["flags"]["sigma_px"])
    assert 0.0 < sigma_px < 2.0, f"161638 sigma_px={sigma_px:.2f} not < 2.0 px"
    assert (TRACKS_DIR / "sigma_px.json").exists()


def test_rest_frames_assertions_all_clips():
    """Verify rest_frames >= 10 frames, no contact, tool cols masked, edge std <= 2 px, ends before contact."""
    cfg = load_config(CONFIG_PATH)
    for cid, c in cfg["clips"].items():
        tr = _load_track(cid)
        bx0, bx1 = c["body_x"]
        rf0, rf1 = c["rest_frames"]
        edge_px = tr["edge_px"]
        tool_x_px = tr["tool_x_px"]
        contact = tr["contact"]

        # 1. Window length >= 10 frames and no contact during rest_frames; tool columns masked if hovering
        assert rf1 - rf0 + 1 >= 10, f"{cid}: rest_frames [{rf0},{rf1}] shorter than 10 frames"
        assert int(np.sum(contact[rf0 : rf1 + 1])) == 0, f"{cid}: contact detected in rest_frames"
        hw = 60 if c["tool"] == "pen" else (110 if c["tool"] == "finger" else 0)
        for f_idx in range(rf0, rf1 + 1):
            tx = tool_x_px[f_idx]
            if not np.isnan(tx) and hw > 0:
                c0 = max(bx0, int(round(float(tx) - hw)))
                c1 = min(bx1, int(round(float(tx) + hw)) + 1)
                assert np.all(np.isnan(edge_px[f_idx, c0:c1])), (
                    f"{cid} f={f_idx}: tool columns [{c0},{c1}] not masked to NaN"
                )

        # 2. Edge std <= 2 px over body_x within the window
        with np.errstate(all="ignore"):
            col_std = np.nanstd(edge_px[rf0 : rf1 + 1, bx0:bx1], axis=0)
            med_std = float(np.nanmedian(col_std))
        assert med_std <= 2.0, f"{cid}: rest_frames edge std {med_std:.2f} px > 2.0 px"

        # 3. Ends before contact start
        c_idx = np.where(contact)[0]
        if len(c_idx) > 0:
            assert rf1 < int(c_idx[0]), (
                f"{cid}: rest_frames end {rf1} not before contact start {int(c_idx[0])}"
            )


def test_contact_windows_and_peak_px():
    """Verify contact start/end within +-5 frames and peak tip displacement within +-6 px."""
    cfg = load_config(CONFIG_PATH)
    for cid, c in cfg["clips"].items():
        tr = _load_track(cid)
        contact = tr["contact"]
        if cid == "161638":
            assert int(contact.sum()) == 0, "161638 must have 0 contact frames"
            continue

        c_idx = np.where(contact)[0]
        assert len(c_idx) > 0, f"{cid}: no contact frames detected"
        c_start, c_end = int(c_idx[0]), int(c_idx[-1])
        exp_start, exp_end = c["contact"]
        assert abs(c_start - exp_start) <= 5, (
            f"{cid}: contact start {c_start} not within +-5 of {exp_start}"
        )
        assert abs(c_end - exp_end) <= 5, (
            f"{cid}: contact end {c_end} not within +-5 of {exp_end}"
        )

        if c["tool"] == "pen" and "peak_px" in c:
            bx0, bx1 = c["body_x"]
            tx_med = int(round(float(np.nanmedian(tr["tool_x_px"][c_idx]))))
            ref_y = float(tr["rest_px"][tx_med])
            if np.isnan(ref_y):
                ref_y = float(np.nanmedian(tr["rest_px"][bx0:bx1]))
            peak_disp = float(np.nanmax(tr["tip_px"][c_idx]) - ref_y)
            exp_peak = float(c["peak_px"])
            assert abs(peak_disp - exp_peak) <= 6.0, (
                f"{cid}: peak disp {peak_disp:.1f} px not within +-6 px of {exp_peak}"
            )


def test_non_nan_edge_fraction():
    """Non-NaN fraction inside body_x (outside tool mask) >= 0.85 (>= 0.55 on 001614)."""
    cfg = load_config(CONFIG_PATH)
    for cid, c in cfg["clips"].items():
        tr = _load_track(cid)
        bx0, bx1 = c["body_x"]
        edge_px = tr["edge_px"]
        tool_x_px = tr["tool_x_px"]
        hw = 60 if c["tool"] == "pen" else (110 if c["tool"] == "finger" else 0)

        valid_cnt = 0
        total_cnt = 0
        for f_idx in range(edge_px.shape[0]):
            row = edge_px[f_idx, bx0:bx1]
            mask = np.ones(bx1 - bx0, dtype=bool)
            tx = tool_x_px[f_idx]
            if not np.isnan(tx) and hw > 0:
                c0 = max(0, int(round(float(tx) - hw)) - bx0)
                c1 = min(bx1 - bx0, int(round(float(tx) + hw)) - bx0 + 1)
                mask[c0:c1] = False
            total_cnt += int(mask.sum())
            valid_cnt += int(np.sum(~np.isnan(row[mask])))

        frac = valid_cnt / max(1, total_cnt)
        min_frac = 0.55 if cid == "001614" else 0.85
        assert frac >= min_frac, f"{cid}: non-NaN fraction {frac:.3f} < {min_frac}"


def test_post_release_residuals():
    """Verify post-release residual <= 3 px on clean clips, flags['residual'], and flags['drift_timing'].

    Note (User Decision A1): 161747 and 161802 exhibit genuine rigid tilt / settling during loading
    (> 3 px over full body_x, primarily on the label side while pivoting near the base contact),
    so they keep the 3 px threshold and set flags['residual'] == True rather than asserting <= 3 px.
    """
    for cid in ["001531", "001614", "001656", "001730", "161638", "161708"]:
        tr = _load_track(cid)
        res_px = float(tr["flags"]["residual_px"])
        assert res_px <= 3.0, f"{cid}: residual {res_px:.2f} px > 3.0 px"
        assert tr["flags"]["residual"] is False
        assert tr["flags"]["drift_timing"] == "none"

    for cid in ["001711", "161853"]:
        tr = _load_track(cid)
        assert tr["flags"]["residual"] is True, f"{cid}: expected flags['residual'] == True"
        assert tr["flags"]["drift_timing"] == "at_release"

    for cid in ["161655", "161747", "161802", "161824", "161904"]:
        tr = _load_track(cid)
        assert tr["flags"]["residual"] is True, f"{cid}: expected flags['residual'] == True"
        assert tr["flags"]["drift_timing"] == "during_loading"


def test_manual_checks_within_4px():
    """All 9 manual check points per clip agree within 4 px of edge_px."""
    manual_all = load_manual_checks(MANUAL_PATH)
    cfg = load_config(CONFIG_PATH)
    assert set(manual_all.keys()) == set(cfg["clips"].keys())

    for cid, pts in manual_all.items():
        assert len(pts) == 9, f"{cid}: expected 9 manual check points, got {len(pts)}"
        tr = _load_track(cid)
        edge_px = tr["edge_px"]
        for pt in pts:
            fn = int(pt["frame"])
            col = int(pt["col"])
            man_y = float(pt["y_px"])
            det_y = float(edge_px[fn, col])
            assert not np.isnan(det_y), f"{cid} f={fn} col={col}: detected edge_px is NaN"
            err = abs(det_y - man_y)
            assert err <= 4.0, (
                f"{cid} f={fn} col={col}: |det ({det_y:.1f}) - manual ({man_y:.1f})| = {err:.2f} > 4.0 px"
            )

