"""
Unit tests for the pure/testable logic in main.py (clustering, grid
reconstruction, temporal smoothing). These don't need a real video or the
EasyOCR model, so they run fast and can be checked in CI.

Run with:  python -m pytest test_main.py -v
       or: python test_main.py   (falls back to plain asserts, no pytest needed)
"""

import sys
import types
from collections import deque

# Stub out easyocr so importing main.py doesn't require the (heavy) real
# dependency just to test the pure helper functions.
if "easyocr" not in sys.modules:
    fake = types.ModuleType("easyocr")
    fake.Reader = object
    sys.modules["easyocr"] = fake

import main  # noqa: E402


def make_detection(x, y, text, w=30, h=20, prob=0.9):
    bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    return (bbox, text, prob)


def test_cluster_1d_groups_close_values():
    ids, n = main.cluster_1d([10, 12, 11, 100, 102], threshold=5)
    assert n == 2
    assert ids[0] == ids[1] == ids[2]
    assert ids[3] == ids[4]
    assert ids[0] != ids[3]


def test_cluster_1d_empty():
    ids, n = main.cluster_1d([], threshold=5)
    assert ids == [] and n == 0


def test_group_into_grid_basic_two_rows_two_cols():
    detections = [
        make_detection(10, 10, "TARUN"),
        make_detection(200, 12, "1"),
        make_detection(10, 60, "TTC"),
        make_detection(200, 62, "74"),
    ]
    grid = main.group_into_grid(detections, y_threshold=15, x_threshold=25)
    assert len(grid) == 2
    assert grid[0][0] == "TARUN" and grid[0][1] == "1"
    assert grid[1][0] == "TTC" and grid[1][1] == "74"


def test_group_into_grid_empty_input():
    assert main.group_into_grid([], 15, 25) == []


def test_smooth_grid_stream_majority_vote_fixes_misread():
    # Same physical cell read as "8" three times and misread as "B" once -
    # the smoothed grid should keep the majority value "8".
    history = deque(maxlen=5)
    history.append([["TARUN", "8"]])
    history.append([["TARUN", "8"]])
    history.append([["TARUN", "B"]])   # one-off OCR misread
    history.append([["TARUN", "8"]])
    smoothed = main.smooth_grid_stream(history)
    assert smoothed[0][1] == "8"


def test_smooth_grid_stream_tracks_real_change_once_window_slides():
    history = deque(maxlen=3)
    for _ in range(3):
        history.append([["TTL", "31"]])
    # score updates to 45 and stays there long enough to fill the window
    for _ in range(3):
        history.append([["TTL", "45"]])
    smoothed = main.smooth_grid_stream(history)
    assert smoothed[0][1] == "45"


def test_smooth_grid_stream_empty_history():
    assert main.smooth_grid_stream(deque()) == []


def test_format_grid_handles_empty():
    assert main.format_grid([]) == "(empty)"


def test_format_grid_produces_aligned_text():
    out = main.format_grid([["TARUN", "1"], ["TTC", "74"]])
    assert "TARUN" in out and "74" in out


def run_all():
    tests = [obj for name, obj in globals().items() if name.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"  ok  - {t.__name__}")
    print(f"\n{passed}/{len(tests)} tests passed")


if __name__ == "__main__":
    run_all()
