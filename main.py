"""
Bowling scoreboard data extraction from video (OCR based).
FOG - Computer Vision Engineer assignment, Round 1.
Author: Jatindra Patel
"""

import argparse
import csv
import json
import logging
import os
import time
from collections import Counter, deque
from dataclasses import dataclass, asdict
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
import easyocr


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("scoreboard_extractor")


@dataclass
class FrameResult:
    frame: int
    timestamp_sec: float
    detected_text: List[str]
    grid: List[List[str]]
    smoothed_grid: List[List[str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract bowling scoreboard data from a video.")
    parser.add_argument("video", nargs="?", default="bowling_scoreboard.mp4",
                         help="input video path (default: bowling_scoreboard.mp4)")
    parser.add_argument("--every", type=int, default=30, help="process every Nth frame")
    parser.add_argument("--roi", type=float, default=0.40, help="top fraction of frame = scoreboard ROI")
    parser.add_argument("--min-conf", type=float, default=0.30, help="min OCR confidence to keep")
    parser.add_argument("--upscale", type=float, default=1.5, help="upscale ROI before OCR")
    parser.add_argument("--no-clahe", action="store_true", help="disable CLAHE contrast boost")
    parser.add_argument("--row-gap", type=float, default=15, help="row clustering distance (px)")
    parser.add_argument("--col-gap", type=float, default=25, help="column clustering distance (px)")
    parser.add_argument("--window", type=int, default=5, help="frames to smooth over")
    parser.add_argument("--out-json", default="output_scoreboard.json")
    parser.add_argument("--out-csv", default="output_scoreboard.csv")
    parser.add_argument("--out-final", default="final_scorecard.json")
    parser.add_argument("--frames-dir", default="extracted_frames")
    parser.add_argument("--annotated-dir", default="annotated_frames")
    parser.add_argument("--keep-duplicates", action="store_true",
                         help="don't skip frames that read the same as the last kept frame")
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


def cluster_1d(values: List[float], threshold: float) -> Tuple[List[int], int]:
    """Groups nearby positions together (used for both rows and columns).
    A value joins the closest existing cluster if it's within `threshold`,
    else it starts a new one. Returns cluster id per value + total clusters,
    ordered top-to-bottom / left-to-right."""
    if not values:
        return [], 0

    order = sorted(range(len(values)), key=lambda i: values[i])
    clusters = []  # {"mean": x, "members": [i, ...]}
    for i in order:
        v = values[i]
        placed = False
        for c in clusters:
            if abs(c["mean"] - v) <= threshold:
                c["members"].append(i)
                c["mean"] = sum(values[m] for m in c["members"]) / len(c["members"])
                placed = True
                break
        if not placed:
            clusters.append({"mean": v, "members": [i]})

    clusters.sort(key=lambda c: c["mean"])
    cluster_id = [0] * len(values)
    for cid, c in enumerate(clusters):
        for m in c["members"]:
            cluster_id[m] = cid
    return cluster_id, len(clusters)


def group_into_grid(ocr_results, y_threshold: float = 15, x_threshold: float = 25) -> List[List[str]]:
    """EasyOCR gives a flat list of (bbox, text, conf) with no order. Cluster
    by y first (rows), then by x (columns), so we get an actual grid back
    instead of a jumbled list - columns line up the same across every row."""
    if not ocr_results:
        return []

    items = []
    for bbox, text, _prob in ocr_results:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        items.append({"text": text, "x": sum(xs) / len(xs), "y": sum(ys) / len(ys)})

    row_ids, n_rows = cluster_1d([i["y"] for i in items], y_threshold)
    col_ids, n_cols = cluster_1d([i["x"] for i in items], x_threshold)

    grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    for item, r, c in zip(items, row_ids, col_ids):
        if grid[r][c] and len(grid[r][c]) >= len(item["text"]):
            continue
        grid[r][c] = item["text"]

    return grid


def smooth_grid_stream(grid_history: Deque[List[List[str]]]) -> List[List[str]]:
    """Majority vote per cell over the last few frames' grids. Fixes a random
    one-frame misread (digit read wrong once) without freezing values that
    are actually still changing - a vote over the whole video would be wrong
    since the running total does change as the game goes on."""
    grids = [g for g in grid_history if g]
    if not grids:
        return []

    max_rows = max(len(g) for g in grids)
    max_cols = max((len(row) for g in grids for row in g), default=0)

    smoothed = []
    for r in range(max_rows):
        row_out = []
        for c in range(max_cols):
            votes = Counter()
            for g in grids:
                if r < len(g) and c < len(g[r]) and g[r][c]:
                    votes[g[r][c]] += 1
            row_out.append(votes.most_common(1)[0][0] if votes else "")
        smoothed.append(row_out)
    return smoothed


def preprocess_roi(roi_bgr: np.ndarray, upscale: float = 1.5, apply_clahe: bool = True) -> np.ndarray:
    """grayscale -> CLAHE (contrast boost, helps where the pins graphic
    overlaps the score row) -> upscale so OCR has more pixels to work with."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    if upscale and upscale != 1.0:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    return gray


def draw_annotations(processed_gray: np.ndarray, ocr_results) -> np.ndarray:
    """draws OCR boxes on the actual image that was fed to OCR (post
    preprocessing) so we can see what it saw, not just trust the output."""
    annotated = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2BGR)
    for bbox, text, prob in ocr_results:
        pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
        color = (0, 255, 0) if prob > 0.6 else (0, 200, 255)
        cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)
        x, y = int(bbox[0][0]), int(bbox[0][1])
        cv2.putText(annotated, text, (x, max(y - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return annotated


def format_grid(grid: List[List[str]]) -> str:
    """quick padded ascii table for printing the final grid to terminal."""
    if not grid:
        return "(empty)"
    n_cols = max(len(row) for row in grid)
    col_widths = [0] * n_cols
    for row in grid:
        for c, cell in enumerate(row):
            col_widths[c] = max(col_widths[c], len(cell))
    lines = []
    for row in grid:
        padded = [cell.ljust(col_widths[c]) for c, cell in enumerate(row)]
        padded += [" " * col_widths[c] for c in range(len(row), n_cols)]
        lines.append(" | ".join(padded))
    return "\n".join(lines)


def process_scoreboard_video(args: argparse.Namespace) -> None:
    if not os.path.isfile(args.video):
        log.error(f"Video file not found: {args.video}")
        return

    log.info("Loading OCR engine (EasyOCR)...")
    reader = easyocr.Reader(["en"], gpu=args.gpu)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        log.error(f"Could not open video: {args.video}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    os.makedirs(args.frames_dir, exist_ok=True)
    os.makedirs(args.annotated_dir, exist_ok=True)

    results: List[FrameResult] = []
    grid_history: Deque[List[List[str]]] = deque(maxlen=max(args.window, 1))
    last_signature: Optional[Tuple[str, ...]] = None
    frame_count = 0
    kept_count = 0
    skipped_duplicates = 0

    start_time = time.time()
    log.info(f"Processing '{args.video}' ({total_frames} frames @ {fps:.1f} fps, "
              f"every {args.every}th frame)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # skip most frames, scoreboard doesn't change every frame anyway
        if frame_count % args.every != 0:
            continue

        h, w = frame.shape[:2]
        roi = frame[0:int(h * args.roi), 0:w]  # scoreboard sits in the top part of the frame
        processed = preprocess_roi(roi, upscale=args.upscale, apply_clahe=not args.no_clahe)

        ocr_results = reader.readtext(processed)
        kept = [r for r in ocr_results if r[2] > args.min_conf]
        detected_text = [text for _bbox, text, _prob in kept]
        signature = tuple(detected_text)

        # same reading as last time -> nothing new, skip saving it again
        is_unchanged = signature == last_signature and signature != ()
        if is_unchanged and not args.keep_duplicates:
            skipped_duplicates += 1
            continue

        last_signature = signature
        timestamp = round(frame_count / fps, 2)

        grid = group_into_grid(
            kept,
            y_threshold=args.row_gap * args.upscale,
            x_threshold=args.col_gap * args.upscale,
        )
        grid_history.append(grid)
        smoothed = smooth_grid_stream(grid_history)

        results.append(FrameResult(
            frame=frame_count, timestamp_sec=timestamp,
            detected_text=detected_text, grid=grid, smoothed_grid=smoothed,
        ))
        kept_count += 1

        cv2.imwrite(os.path.join(args.frames_dir, f"frame_{frame_count}.jpg"), roi)
        annotated = draw_annotations(processed, kept)
        cv2.imwrite(os.path.join(args.annotated_dir, f"frame_{frame_count}_annotated.jpg"), annotated)

        log.info(f"Frame {frame_count} ({timestamp}s): {detected_text}")

    cap.release()

    with open(args.out_json, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=4)

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "timestamp_sec", "detected_text"])
        for r in results:
            writer.writerow([r.frame, r.timestamp_sec, " | ".join(r.detected_text)])

    # last frame's smoothed grid = pipeline's best final guess at the full scorecard
    final_grid = results[-1].smoothed_grid if results else []
    with open(args.out_final, "w") as f:
        json.dump({"final_scorecard": final_grid}, f, indent=4)

    elapsed = time.time() - start_time
    log.info("Done.")
    log.info(f"Frames kept: {kept_count}  |  skipped as duplicate: {skipped_duplicates}")
    log.info(f"Per-frame JSON   -> {args.out_json}")
    log.info(f"CSV              -> {args.out_csv}")
    log.info(f"Final scorecard  -> {args.out_final}")
    log.info(f"Cropped frames   -> {args.frames_dir}/")
    log.info(f"Annotated frames -> {args.annotated_dir}/")
    log.info(f"Time taken       -> {elapsed:.1f}s")
    if final_grid:
        log.info("Final scorecard:\n" + format_grid(final_grid))


def main():
    args = parse_args()
    process_scoreboard_video(args)


if __name__ == "__main__":
    main()
