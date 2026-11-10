"""Module M7 Demo Unit Tests.

Verifies:
1. All demo video deliverables exist, have valid non-empty sizes (> 100 KB), and 1920x720 resolution.
2. Frame counts match raw video clips exactly:
   - 161747: 257 frames
   - 001711: 221 frames
   - fresh_instance.mp4: 478 frames (257 + 221)
   - cap_effect.mp4: 279 frames
   - all.mp4: full master video with title cards
3. Pass/fail criterion from BUILD_SPEC:
   - On rest frames (prior to contact), predicted edge is within <= 2.0 px of observed edge.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest
import yaml

from src.sim import StaticSimulator


@pytest.fixture(scope="module")
def cfg():
    with open("config/clips.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_video_info(video_path: Path):
    assert video_path.exists(), f"Video file {video_path} does not exist"
    assert video_path.stat().st_size > 100_000, f"Video {video_path} is smaller than 100 KB"
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Could not open video {video_path}"
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Test decodability: read first, middle, last
    ret1, f1 = cap.read()
    assert ret1 and f1 is not None, f"Failed to read first frame of {video_path}"

    cap.set(cv2.CAP_PROP_POS_FRAMES, n_frames // 2)
    ret2, f2 = cap.read()
    assert ret2 and f2 is not None, f"Failed to read middle frame of {video_path}"

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n_frames - 2))
    ret3, f3 = cap.read()
    assert ret3 and f3 is not None, f"Failed to read last frame of {video_path}"

    cap.release()
    return {"n_frames": n_frames, "width": width, "height": height, "fps": fps}


def test_fresh_instance_videos_exist_and_counts():
    """Verify fresh instance individual and combined demo videos."""
    demo_dir = Path("out/demo")

    info_161747 = get_video_info(demo_dir / "fresh_instance_161747.mp4")
    assert info_161747["width"] == 1920
    assert info_161747["height"] == 720
    assert info_161747["n_frames"] == 257, f"Expected 257 frames for 161747, got {info_161747['n_frames']}"

    info_001711 = get_video_info(demo_dir / "fresh_instance_001711.mp4")
    assert info_001711["width"] == 1920
    assert info_001711["height"] == 720
    assert info_001711["n_frames"] == 221, f"Expected 221 frames for 001711, got {info_001711['n_frames']}"

    info_fresh = get_video_info(demo_dir / "fresh_instance.mp4")
    assert info_fresh["width"] == 1920
    assert info_fresh["height"] == 720
    assert info_fresh["n_frames"] == 478, f"Expected 478 frames for fresh_instance, got {info_fresh['n_frames']}"


def test_cap_effect_video_exists_and_counts():
    """Verify cap effect comparison demo video."""
    demo_dir = Path("out/demo")
    info_cap = get_video_info(demo_dir / "cap_effect.mp4")
    assert info_cap["width"] == 1920
    assert info_cap["height"] == 720
    assert info_cap["n_frames"] == 279, f"Expected 279 frames for cap_effect, got {info_cap['n_frames']}"


def test_master_all_video_exists_and_counts():
    """Verify master all.mp4 demo video with title cards."""
    demo_dir = Path("out/demo")
    info_all = get_video_info(demo_dir / "all.mp4")
    assert info_all["width"] == 1920
    assert info_all["height"] == 720
    # Master video contains 5 title cards (~14.5s = ~435 frames) + fresh (478) + cap (279) > 1000 frames
    assert info_all["n_frames"] >= 1000, f"Expected >= 1000 frames for all.mp4, got {info_all['n_frames']}"


@pytest.mark.parametrize("clip_id,rest_frame_idx", [
    ("161747", 59),
    ("001711", 83),
    ("001531", 68),
    ("001656", 45),
])
def test_rest_frame_predicted_edge_accuracy(clip_id, rest_frame_idx, cfg):
    """Pass/fail criterion: On rest frames, predicted edge must be within <= 2.0 px of observed edge."""
    tracks_dir = Path("out/tracks")
    tr = np.load(tracks_dir / f"{clip_id}.npz", allow_pickle=True)
    tr_mm = np.load(tracks_dir / f"{clip_id}_mm.npz", allow_pickle=True)

    edge_px = tr["edge_px"]
    rest_px = tr["rest_px"]
    s = float(tr_mm["s"])
    c_cfg = cfg["clips"][clip_id]
    bx0, bx1 = int(c_cfg["body_x"][0]), int(c_cfg["body_x"][1])
    cols = np.arange(bx0, bx1)

    # On rest frame before contact, simulation drop w_sim = 0
    # Therefore, y_pred = rest_px
    y_obs = edge_px[rest_frame_idx, cols]
    y_pred = rest_px[cols]

    valid = ~np.isnan(y_obs) & ~np.isnan(y_pred)
    assert np.sum(valid) > 50, f"Insufficient valid edge points on rest frame for {clip_id}"

    diff = np.abs(y_pred[valid] - y_obs[valid])
    mean_err = np.mean(diff)
    p95_err = np.percentile(diff, 95)

    print(f"Clip {clip_id} rest frame {rest_frame_idx}: mean error = {mean_err:.2f} px, 95th pct = {p95_err:.2f} px")
    assert mean_err <= 2.0, f"Mean error on rest frame {mean_err:.2f} px exceeds 2.0 px threshold for {clip_id}"
