"""M2 unit tests for scale and station observation extraction (`src/scale.py`)."""

from pathlib import Path
import numpy as np

from src.track import load_config


CONFIG_PATH = Path("config/clips.yaml")
TRACKS_DIR = Path("out/tracks")


def _load_mm(clip_id: str):
    path = TRACKS_DIR / f"{clip_id}_mm.npz"
    assert path.exists(), f"Missing M2 output: {path}"
    data = np.load(path, allow_pickle=True)
    return {
        "s": float(data["s"]),
        "s_px_per_mm": float(data["s_px_per_mm"]),
        "s_len_px_per_mm": float(data["s_len_px_per_mm"]),
        "both_shoulders_visible": bool(data["both_shoulders_visible"]),
        "x_base_px": float(data["x_base_px"]),
        "x_mm": data["x_mm"],
        "w_obs_mm": data["w_obs_mm"],
        "tool_x_mm": data["tool_x_mm"],
        "tip_drop_mm": data["tip_drop_mm"],
        "stations_mm": data["stations_mm"],
        "u_mm": data["u_mm"],
        "d_mm": data["d_mm"],
        "tool_x_med_mm": float(data["tool_x_med_mm"]),
        "idx": data["idx"],
        "flags": data["flags"].item(),
    }


def test_scale_ranges_and_length_crosscheck():
    """s is in 6.5-7.5 px/mm on 9 Nov clips and 7.0-8.8 on 10 Nov clips; agrees with length cross-check within 5%."""
    cfg = load_config(CONFIG_PATH)
    for cid in cfg["clips"]:
        mm = _load_mm(cid)
        s = mm["s"]
        if cid.startswith("00"):
            assert 7.0 <= s <= 8.8, f"{cid}: s={s:.3f} not in [7.0, 8.8]"
        else:
            assert 6.5 <= s <= 7.5, f"{cid}: s={s:.3f} not in [6.5, 7.5]"

        if mm["both_shoulders_visible"]:
            s_len = mm["s_len_px_per_mm"]
            rel_err = abs(s - s_len) / s
            assert rel_err <= 0.05, (
                f"{cid}: s_diam={s:.3f} vs s_len={s_len:.3f} differ by {rel_err*100:.1f}% > 5%"
            )


def test_tool_x_mm_targets():
    """Peak tool_x_mm matches BUILD_SPEC targets within +/-6 mm."""
    expected_tx = {
        "001531": 92.0,
        "001614": 98.0,
        "001656": 92.0,
        "001711": 94.0,
        "001730": 86.0,
        "161655": 81.0,
        "161708": 63.0,
        "161747": 76.0,
        "161802": 57.0,
        "161904": 56.0,
    }
    for cid, exp in expected_tx.items():
        mm = _load_mm(cid)
        tx = mm["tool_x_med_mm"]
        assert abs(tx - exp) <= 6.0, (
            f"{cid}: tool_x_mm={tx:.1f} differs from expected {exp:.1f} by > 6 mm"
        )


def test_normalised_peak_profiles_fact4():
    """Normalised peak profiles (w/tip at 15-21, 30-39, 45-60 mm) reproduce Section 0 fact 4 table within +/-0.08."""
    # Section 0, fact 4 table:
    # Capped:   15-21: [0.56, 0.71], 30-39: [0.36, 0.49], 45-60: [0.27, 0.33]
    # Uncapped: 15-21: [0.76, 0.84], 30-39: [0.48, 0.54], 45-60: [0.34, 0.38]
    tol = 0.08
    capped_bounds = [(0.56 - tol, 0.71 + tol), (0.36 - tol, 0.49 + tol), (0.27 - tol, 0.33 + tol)]
    uncapped_bounds = [(0.76 - tol, 0.84 + tol), (0.48 - tol, 0.54 + tol), (0.34 - tol, 0.38 + tol)]

    for cid, is_capped in [
        ("001531", True),
        ("001730", True),
        ("001614", False),
        ("001656", False),
        ("001711", False),
        ("161708", True),
        ("161747", True),
        ("161904", True),
    ]:
        mm = _load_mm(cid)
        x_mm = mm["x_mm"]
        u_mm = mm["u_mm"]
        d_mm = mm["d_mm"]
        tx = mm["tool_x_med_mm"]

        k_peak = int(np.argmax(d_mm))
        tip_pk = float(d_mm[k_peak])
        w_pk = u_mm[k_peak, :]
        dist = np.abs(x_mm - tx)

        b1 = float(np.nanmean(w_pk[(dist >= 15.0) & (dist <= 21.0)]) / tip_pk)
        b2 = float(np.nanmean(w_pk[(dist >= 30.0) & (dist <= 39.0)]) / tip_pk)
        b3 = float(np.nanmean(w_pk[(dist >= 45.0) & (dist <= 60.0)]) / tip_pk)

        bounds = capped_bounds if is_capped else uncapped_bounds
        for val, (lo, hi), label in zip([b1, b2, b3], bounds, ["15-21", "30-39", "45-60"]):
            assert lo <= val <= hi, (
                f"{cid} ({label} mm): w/tip={val:.3f} not in [{lo:.2f}, {hi:.2f}]"
            )


def test_peak_displacement_positive_and_decreasing_trend():
    """At every station m, u_mm[argmax(d_mm), m] > 0 and decreases in trend with |x_mm[m] - tool_x_mm|."""
    cfg = load_config(CONFIG_PATH)
    for cid in cfg["clips"]:
        if cid == "161638":
            continue
        mm = _load_mm(cid)
        d_mm = mm["d_mm"]
        u_mm = mm["u_mm"]
        stations_mm = mm["stations_mm"]
        tx = mm["tool_x_med_mm"]

        assert 1 <= len(d_mm) <= 25, f"{cid}: expected 1..25 loading frames, got {len(d_mm)}"
        k_peak = int(np.argmax(d_mm))
        u_peak = u_mm[k_peak, :]
        assert np.all(u_peak > 0.0), (
            f"{cid}: min u_peak={float(np.min(u_peak)):.2f} mm <= 0"
        )

        dist = np.abs(stations_mm - tx)
        slope = np.polyfit(dist, u_peak, 1)[0]
        assert slope < 0.0, f"{cid}: u_peak does not decrease in trend with |x_mm - tool_x_mm| (slope={slope:.4f})"


def test_161904_by_eye_ratios():
    """On 161904 at peak frame, u_mm(tool_x_mm - 20)/d_mm in [0.45, 0.75] and at -40 in [0.20, 0.48]."""
    mm = _load_mm("161904")
    d_mm = mm["d_mm"]
    u_mm = mm["u_mm"]
    stations_mm = mm["stations_mm"]
    tx = mm["tool_x_med_mm"]

    k_peak = int(np.argmax(d_mm))
    d_peak = float(d_mm[k_peak])
    u_peak = u_mm[k_peak, :]

    r20 = float(np.interp(tx - 20.0, stations_mm, u_peak)) / d_peak
    r40 = float(np.interp(tx - 40.0, stations_mm, u_peak)) / d_peak
    assert 0.45 <= r20 <= 0.75, f"161904 u(tx-20)/d = {r20:.3f} not in [0.45, 0.75]"
    assert 0.20 <= r40 <= 0.48, f"161904 u(tx-40)/d = {r40:.3f} not in [0.20, 0.48]"


def test_zero_nans_on_fit_clips():
    """Zero NaN in u_mm and d_mm on 001531, 001730, 001614, 001656."""
    for cid in ["001531", "001730", "001614", "001656"]:
        mm = _load_mm(cid)
        assert not np.any(np.isnan(mm["u_mm"])), f"{cid}: NaNs found in u_mm"
        assert not np.any(np.isnan(mm["d_mm"])), f"{cid}: NaNs found in d_mm"
        assert mm["u_mm"].shape == (len(mm["d_mm"]), len(mm["stations_mm"]))
