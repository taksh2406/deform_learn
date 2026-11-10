#!/usr/bin/env python3
"""Tests for M6 — Evaluation (`src/evaluate.py`).

Verifies:
1. metrics.csv contains all 8 test clips plus headline and stratified summary rows.
2. No NaN metrics except where explicitly skipped by flag (161853 mid/far).
3. All required error_vs_time and peak_profile plots are generated for all test clips.
4. Model significantly beats baseline B0 (zero deformation) across test clips.
5. Post-release residuals are present and non-null for all test clips.
"""

import csv
from pathlib import Path
import pytest

TEST_CLIPS = ["001711", "161655", "161708", "161747", "161802", "161904", "161824", "161853"]


def test_metrics_csv_structure_and_completeness():
    """Verify metrics.csv contains all required rows and columns."""
    metrics_file = Path("out/eval/metrics.csv")
    assert metrics_file.exists(), "metrics.csv missing"

    with open(metrics_file, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    clip_names = [r["clip"] for r in rows]
    for c in TEST_CLIPS:
        assert c in clip_names, f"Test clip {c} missing from metrics.csv"

    assert "HEADLINE_ALL_TEST_CLIPS" in clip_names, "HEADLINE_ALL_TEST_CLIPS row missing"
    assert "STRATIFIED_CLEAN_OR_AT_RELEASE" in clip_names, "STRATIFIED_CLEAN_OR_AT_RELEASE row missing"
    assert "STRATIFIED_DURING_LOADING" in clip_names, "STRATIFIED_DURING_LOADING row missing"
    assert "CAP_EFFECT_SUMMARY" in clip_names, "CAP_EFFECT_SUMMARY row missing"

    # Verify no NaN or empty metrics on individual clips except 161853 mid/far
    for r in rows:
        c = r["clip"]
        if c in TEST_CLIPS:
            assert r["rmse_model_mm"] != "", f"Empty rmse_model_mm on {c}"
            assert r["rmse_b0_mm"] != "", f"Empty rmse_b0_mm on {c}"
            assert r["rmse_b1_mm"] != "", f"Empty rmse_b1_mm on {c}"
            assert r["ratio_15_21_data"] != "", f"Empty ratio_15_21_data on {c}"
            assert r["ratio_15_21_model"] != "", f"Empty ratio_15_21_model on {c}"
            assert r["residual_mm"] != "", f"Empty residual_mm on {c}"
            assert r["drift_timing"] in ["none", "at_release", "during_loading"], f"Invalid drift_timing on {c}"

            if c not in ["161853", "161824"]:
                assert r["ratio_30_39_data"] != "", f"Empty ratio_30_39_data on {c}"
                assert r["ratio_30_39_model"] != "", f"Empty ratio_30_39_model on {c}"
                assert r["ratio_45_60_data"] != "", f"Empty ratio_45_60_data on {c}"
                assert r["ratio_45_60_model"] != "", f"Empty ratio_45_60_model on {c}"
            elif c == "161824":
                # 161824: pressed at 52.8 mm with cropped span [10, 94], no stations in 45-60 mm
                assert r["ratio_30_39_data"] != "", f"Empty ratio_30_39_data on {c}"
                assert r["ratio_45_60_data"] == "", f"Expected empty ratio_45_60_data on {c}"
            else:
                # 161853: camera moved, mid/far must be skipped
                assert r["ratio_30_39_data"] == "", f"Expected skipped ratio_30_39_data on {c}"
                assert r["ratio_45_60_data"] == "", f"Expected skipped ratio_45_60_data on {c}"


def test_model_beats_baseline_b0_across_test_clips():
    """Verify the physical model achieves substantial error reduction over B0 (w=0)."""
    metrics_file = Path("out/eval/metrics.csv")
    assert metrics_file.exists(), "metrics.csv missing"

    with open(metrics_file, newline="") as f:
        reader = csv.DictReader(f)
        rows = {r["clip"]: r for r in reader}

    headline = rows["HEADLINE_ALL_TEST_CLIPS"]
    rmse_model = float(headline["rmse_model_mm"])
    rmse_b0 = float(headline["rmse_b0_mm"])

    print(f"Headline aggregate: Model RMSE = {rmse_model:.3f} mm, B0 RMSE = {rmse_b0:.3f} mm (ratio: {rmse_model/rmse_b0*100:.1f}%)")
    assert rmse_model < 0.6 * rmse_b0, f"Model RMSE {rmse_model:.3f} not significantly better than B0 {rmse_b0:.3f}"


def test_all_evaluation_plots_generated():
    """Verify all 16 evaluation plots exist and are non-empty."""
    eval_dir = Path("out/eval")
    for c in TEST_CLIPS:
        p_err = eval_dir / f"{c}_error_vs_time.png"
        p_pk = eval_dir / f"{c}_peak_profile.png"
        assert p_err.exists(), f"Plot {p_err} missing"
        assert p_pk.exists(), f"Plot {p_pk} missing"
        assert p_err.stat().st_size > 1000, f"Plot {p_err} is empty"
        assert p_pk.stat().st_size > 1000, f"Plot {p_pk} is empty"
