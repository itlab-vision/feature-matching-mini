import os
import re
import subprocess
import sys
from pathlib import Path

import cv2 as cv
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def load_img():
    def _load(name):
        path = Path(__file__).parent.parent / "test_data" / name
        img = cv.imread(str(path))
        if img is None:
            pytest.fail(f"Failed to load image. On path: {path.absolute()}")
        return img

    return _load


@pytest.fixture
def image_pair(tmp_path, load_img):
    img1_raw = load_img("box.png")
    img2_raw = load_img("box_in_scene.png")

    target_size = (64, 64)
    img1_resized = cv.resize(img1_raw, target_size, interpolation=cv.INTER_AREA)
    img2_resized = cv.resize(img2_raw, target_size, interpolation=cv.INTER_AREA)

    img1_path = tmp_path / "box_small.png"
    img2_path = tmp_path / "box_in_scene_small.png"
    assert cv.imwrite(str(img1_path), img1_resized)
    assert cv.imwrite(str(img2_path), img2_resized)

    return img1_path, img2_path


def test_pipeline_performance_cli(image_pair):
    img1, img2 = image_pair
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "samples.performance_pipeline_benchmark",
            "-det",
            "sift",
            "-detb",
            "opencv",
            "-des",
            "sift",
            "-desb",
            "opencv",
            "-mat",
            "bf",
            "-matb",
            "opencv",
            "-i1",
            str(img1),
            "-i2",
            str(img2),
            "-n",
            "2",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=240,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    output = result.stdout + result.stderr

    for label in ("Min time pipeline test", "Mean time pipeline test"):
        match = re.search(rf"{re.escape(label)}:\s*([\d\.eE\+\-]+)", output)
        assert match is not None, f"Missing '{label}' in output:\n{output}"
        assert float(match.group(1)) >= 0.0, match.group(0)


def test_staged_performance_cli(image_pair):
    img1, img2 = image_pair
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "samples.performance_staged_benchmark",
            "-det",
            "sift",
            "-detb",
            "opencv",
            "-des",
            "sift",
            "-desb",
            "opencv",
            "-mat",
            "bf",
            "-matb",
            "opencv",
            "-i1",
            str(img1),
            "-i2",
            str(img2),
            "-n",
            "2",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=240,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    output = result.stdout + result.stderr

    for label in (
        "Min time detection",
        "Mean time detection",
        "Min time descriptor",
        "Mean time descriptor",
        "Min time match",
        "Mean time match",
    ):
        match = re.search(rf"{re.escape(label)}:\s*([\d\.eE\+\-]+)", output)
        assert match is not None, f"Missing '{label}' in output:\n{output}"
        assert float(match.group(1)) >= 0.0, match.group(0)


@pytest.mark.skipif(
    os.environ.get("RUN_PERFORMANCE_ALL_COMBINATIONS") != "1",
    reason="Full performance matrix requires all model weights and runtimes",
)
def test_performance_table_all_combinations(tmp_path, image_pair):
    from samples.performance_benchmark_table import get_all_combinations

    img1, img2 = image_pair
    output_csv = tmp_path / "perf_results.csv"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "samples.performance_benchmark_table",
            "-i1",
            str(img1),
            "-i2",
            str(img2),
            "-o",
            str(output_csv),
            "-d",
            "cpu",
            "-n",
            "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert output_csv.exists()

    df = pd.read_csv(output_csv)
    expected_combos = get_all_combinations()
    columns = [
        "detector",
        "detector_backend",
        "descriptor",
        "descriptor_backend",
        "matcher",
        "matcher_backend",
    ]

    assert len(df) == len(expected_combos)
    actual_tuples = {tuple(row[col] for col in columns) for _, row in df.iterrows()}
    expected_tuples = {tuple(combo[col] for col in columns) for combo in expected_combos}
    assert actual_tuples == expected_tuples
    assert (df["iterations"] == 1).all()
