import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def two_pairs(tmp_path):
    reference = cv.imread(str(REPO_ROOT / "test_data" / "box.png"))
    assert reference is not None
    height, width = reference.shape[:2]

    dataset = tmp_path / "hpatches"
    scene = dataset / "v_smoke"
    scene.mkdir(parents=True)
    assert cv.imwrite(str(scene / "1.ppm"), reference)

    for index in (2, 3):
        homography = np.eye(3)
        homography[:2, 2] = (3 * index, -2 * index)
        target = cv.warpPerspective(reference, homography, (width, height))
        assert cv.imwrite(str(scene / f"{index}.ppm"), target)
        np.savetxt(scene / f"H_1_{index}", homography)
    return dataset


def test_hpatches_quality_evaluation_cli(tmp_path, two_pairs):
    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "samples" / "hpatches_benchmark.py"),
            "--path", str(two_pairs),
            "--detector", "sift", "--descriptor", "sift", "--matcher", "bf",
            "--tasks", "matchingap", "matchingscore", "homographyauc",
            "--eval-thresholds", "3",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    output = result.stdout + result.stderr
    for metric in ("MAP", "MS", "Prec", "AUC"):
        score = re.search(rf"@ 3\.0px, [^\n]*\b{metric}: (\d+\.\d+)", output)
        assert score is not None, f"Missing {metric} score at 3 px:\n{output}"
        assert 0 < float(score.group(1)) <= 1, score.group(0)


@pytest.mark.skipif(
    os.environ.get("RUN_HPATCHES_ALL_COMBINATIONS") != "1",
    reason="Full matrix needs the model weights and ExecuTorch runtime",
)
def test_hpatches_all_combinations_on_two_pairs(tmp_path, two_pairs):
    from samples.hpatches_table_benchmark import get_all_combinations

    output_path = tmp_path / "results.csv"
    result = subprocess.run(
        [
            sys.executable, "-m", "samples.hpatches_table_benchmark",
            "--path", str(two_pairs), "--output", str(output_path),
            "--num-scenes", "1", "--no-skip", "--strict", "--eval-thresholds", "3",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    errors = [line for line in result.stderr.splitlines() if "Failed combination" in line]
    assert result.returncode == 0, "\n".join(errors) if errors else result.stdout + result.stderr
    with output_path.open(newline="") as report:
        rows = list(csv.DictReader(report))
    columns = ("detector", "detector_backend", "descriptor", "descriptor_backend", "matcher", "matcher_backend")
    expected = get_all_combinations()
    assert len(rows) == len(expected)
    assert {tuple(row[column] for column in columns) for row in rows} == set(expected)
    assert all(row["num_pairs"] == "2" for row in rows)
