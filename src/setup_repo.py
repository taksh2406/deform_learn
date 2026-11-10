"""M0: Repository setup and video verification."""

import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "clips.yaml"


def load_config(config_path: Path = DEFAULT_CONFIG) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_orientation_frame0(frame: np.ndarray) -> bool:
    """Assert that mean V of the bottom 15% of rows > mean V of the top 15%."""
    if frame is None or frame.size == 0:
        return False
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    h = v.shape[0]
    band = int(0.15 * h)
    top_mean = float(np.mean(v[:band, :]))
    bottom_mean = float(np.mean(v[h - band :, :]))
    return bottom_mean > top_mean


def verify_repo(config_path: Path = DEFAULT_CONFIG) -> list[str]:
    """Run all M0 verification checks in exact order and return 13 OK lines."""
    cfg = load_config(config_path)
    video_dir = REPO_ROOT / cfg["video_dir"]
    clips = cfg["clips"]

    # 1. Every file: in the config exists in video_dir
    expected_files = set()
    for clip_id, cinfo in clips.items():
        fname = cinfo["file"]
        expected_files.add(fname)
        fpath = video_dir / fname
        if not fpath.is_file():
            raise FileNotFoundError(
                f"Clip {clip_id}: file '{fname}' missing in {video_dir}"
            )

    # 2. video_dir (data/raw/) contains no extra .mp4 the config doesn't list
    actual_mp4s = {p.name for p in video_dir.glob("*.mp4")}
    extra_mp4s = sorted(actual_mp4s - expected_files)
    if extra_mp4s:
        raise AssertionError(
            f"Extra .mp4 files in {video_dir} not listed in config: {extra_mp4s}"
        )

    ok_lines = []
    for clip_id, cinfo in clips.items():
        fname = cinfo["file"]
        fpath = video_dir / fname
        cap = cv2.VideoCapture(str(fpath))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Clip {clip_id}: failed to open {fpath}")

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            ret, frame0 = cap.read()
        finally:
            cap.release()

        # 3. Each file opens with OpenCV at 1920x1080
        if (width, height) != (1920, 1080):
            raise AssertionError(
                f"Clip {clip_id} ({fname}): expected 1920x1080, got {width}x{height}"
            )

        # 4. Each file's frame count equals its n_frames
        expected_n = int(cinfo["n_frames"])
        if n_frames != expected_n:
            raise AssertionError(
                f"Clip {clip_id} ({fname}): expected n_frames={expected_n}, got {n_frames}"
            )

        # 5. Each file's frame rate is 29.97 +- 0.01
        if abs(fps - 29.97) > 0.01:
            raise AssertionError(
                f"Clip {clip_id} ({fname}): expected fps 29.97 +- 0.01, got {fps:.5f}"
            )

        # 6. Floor-at-bottom orientation check passes on frame 0
        if not ret or not check_orientation_frame0(frame0):
            raise AssertionError(
                f"Clip {clip_id} ({fname}): floor-at-bottom orientation check failed on frame 0"
            )

        ok_lines.append(
            f"OK {clip_id}: {fname} (1920x1080, {n_frames} frames, {fps:.4f} fps, floor=bottom)"
        )

    return ok_lines


def ensure_dirs() -> None:
    dirs = [
        REPO_ROOT / "data" / "raw",
        REPO_ROOT / "config",
        REPO_ROOT / "src",
        REPO_ROOT / "tests",
        REPO_ROOT / "out" / "tracks",
        REPO_ROOT / "out" / "fit",
        REPO_ROOT / "out" / "eval",
        REPO_ROOT / "out" / "demo",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Setup and verify repository.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify videos and configuration.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to clips.yaml.",
    )
    args = parser.parse_args()

    if not args.check:
        ensure_dirs()

    ok_lines = verify_repo(args.config)
    for line in ok_lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

