"""Tests for M0 repository setup and data verification."""

from pathlib import Path
import subprocess
import sys
import cv2
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import setup_repo  # noqa: E402


def test_directories_and_files_exist():
    assert (REPO_ROOT / "data" / "raw").is_dir()
    assert (REPO_ROOT / "config" / "clips.yaml").is_file()
    assert (REPO_ROOT / "requirements.txt").is_file()
    assert (REPO_ROOT / ".gitignore").is_file()
    assert not (REPO_ROOT / "videos").exists()

    gitignore_text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "out/" in gitignore_text
    assert "data/raw" not in gitignore_text


def test_verify_repo_checks():
    cfg = setup_repo.load_config()
    video_dir = REPO_ROOT / cfg["video_dir"]
    clips = cfg["clips"]
    assert len(clips) == 13

    # 1. Every file: in the config exists in video_dir
    expected_files = set()
    for clip_id, cinfo in clips.items():
        fname = cinfo["file"]
        expected_files.add(fname)
        assert (video_dir / fname).is_file(), f"Missing {fname} for clip {clip_id}"

    # 2. No extra .mp4 in data/raw/
    actual_mp4s = {p.name for p in video_dir.glob("*.mp4")}
    assert actual_mp4s == expected_files

    # 3-6. OpenCV checks per clip
    for clip_id, cinfo in clips.items():
        fpath = video_dir / cinfo["file"]
        cap = cv2.VideoCapture(str(fpath))
        assert cap.isOpened(), f"Failed to open {fpath}"
        try:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            ret, frame0 = cap.read()
        finally:
            cap.release()

        assert (w, h) == (1920, 1080), f"Clip {clip_id}: got {w}x{h}"
        assert n == cinfo["n_frames"], f"Clip {clip_id}: got {n} frames"
        assert abs(fps - 29.97) <= 0.01, f"Clip {clip_id}: got fps={fps}"
        assert ret and setup_repo.check_orientation_frame0(frame0), (
            f"Clip {clip_id}: orientation check failed"
        )


def test_setup_repo_cli_check():
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "src" / "setup_repo.py"), "--check"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    lines = [line.strip() for line in proc.stdout.strip().splitlines() if line.strip()]
    assert len(lines) == 13
    assert all(line.startswith("OK ") for line in lines)

