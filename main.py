"""
Bowling Scoreboard Data Extraction
-----------------------------------
Extracts scoreboard data (player name, frame-wise pin counts, totals) from a
bowling alley video using frame sampling + ROI cropping + OCR (EasyOCR).

Assignment : FOG - Computer Vision Engineer, Round 1
Author     : Jatindra Patel
"""

import argparse
import csv
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import cv2
import numpy as np
import easyocr


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scoreboard_extractor")


@dataclass
class FrameResult:
    frame: int
    timestamp_sec: float
    detected_text: List[str]
    rows: List[List[str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract bowling scoreboard data from a video using OCR."
    )
    parser.add_argument(
        "video", nargs="?", default="bowling_scoreboard.mp4",
        help="Path to the input video (default: bowling_scoreboard.mp4)",
    )
    parser.add_argument(
        "--every", type=int, default=30,
        help="Process every Nth frame (default: 30, ~1 frame/sec at 30fps)",
    )
    parser.add_argument(
        "--roi", type=float, default=0.40,
        help="Fraction of frame height (from top) treated as the scoreboard ROI (default: 0.40)",
    )
    parser.add_argument(
        "--min-conf", type=float, default=0.30,
        help="Minimum OCR confidence required to keep a detection (default: 0.30)",
    )
    parser.add_argument("--out-json", default="output_scoreboard.json",
                         help="Path to write the structured JSON output")
    parser.add_argument("--out-csv", default="output_scoreboard.csv",
                         help="Path to write a flattened CSV version of the output")
    parser.add_argument("--frames-dir", default="extracted_frames",
                         help="Directory to save cropped scoreboard (ROI) frames")
    parser.add_argument("--annotated-dir", default="annotated_frames",
                         help="Directory to save ROI frames with OCR bounding boxes drawn on them")
    parser.add_argument("--keep-duplicates", action="store_true",
                         help="Save every sampled frame even if the scoreboard text hasn't changed "
                              "since the last saved frame (by default, unchanged frames are skipped)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for OCR if available")
    return parser.parse_args()


def group_into_rows(ocr_results, y_threshold: int = 15) -> List[List[str]]:
    """
    EasyOCR returns each detection as (bbox, text, confidence), with bbox being
    4 (x, y) corner points. A scoreboard is a grid, so detections that share
    roughly the same vertical position belong to the same row (a player row,
    the header row, etc). This clusters raw detections into rows by y-position
    and sorts each row left-to-right by x-position, so the output mirrors the
    actual scoreboard layout instead of being one flat, unordered list of
    strings.
    """
    items = []
    for bbox, text, _prob in ocr_results:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        items.append({"text": text, "x": sum(xs) / len(xs), "y": sum(ys) / len(ys)})

    items.sort(key=lambda i: i["y"])

    rows = []
    for item in items:
        placed = False
        for row in rows:
            if abs(row["y"] - item["y"]) <= y_threshold:
                row["items"].append(item)
                row["y"] = sum(i["y"] for i in row["items"]) / len(row["items"])
                placed = True
                break
        if not placed:
            rows.append({"y": item["y"], "items": [item]})

    return [[i["text"] for i in sorted(row["items"], key=lambda i: i["x"])] for row in rows]


def draw_annotations(roi, ocr_results):
    """Draws the OCR bounding boxes + recognized text on a copy of the ROI, so
    the detection step is visually verifiable instead of a black box."""
    annotated = roi.copy()
    for bbox, text, prob in ocr_results:
        pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
        color = (0, 255, 0) if prob > 0.6 else (0, 200, 255)
        cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)
        x, y = int(bbox[0][0]), int(bbox[0][1])
        cv2.putText(annotated, text, (x, max(y - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return annotated


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
    last_signature: Optional[Tuple[str, ...]] = None
    frame_count = 0
    kept_count = 0
    skipped_duplicates = 0

    start_time = time.time()
    log.info(f"Processing '{args.video}' ({total_frames} frames @ {fps:.1f} fps, "
              f"sampling every {args.every}th frame)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % args.every != 0:
            continue

        h, w = frame.shape[:2]
        roi = frame[0:int(h * args.roi), 0:w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        ocr_results = reader.readtext(gray_roi)
        kept = [r for r in ocr_results if r[2] > args.min_conf]
        detected_text = [text for _bbox, text, _prob in kept]
        signature = tuple(detected_text)

        is_unchanged = signature == last_signature and signature != ()
        if is_unchanged and not args.keep_duplicates:
            skipped_duplicates += 1
            continue

        last_signature = signature
        timestamp = round(frame_count / fps, 2)
        rows = group_into_rows(kept)

        results.append(FrameResult(
            frame=frame_count, timestamp_sec=timestamp,
            detected_text=detected_text, rows=rows,
        ))
        kept_count += 1

        cv2.imwrite(os.path.join(args.frames_dir, f"frame_{frame_count}.jpg"), roi)
        annotated = draw_annotations(roi, kept)
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

    elapsed = time.time() - start_time
    log.info("Done.")
    log.info(f"Frames kept (unique): {kept_count}  |  skipped as duplicates: {skipped_duplicates}")
    log.info(f"JSON              -> {args.out_json}")
    log.info(f"CSV               -> {args.out_csv}")
    log.info(f"Cropped frames    -> {args.frames_dir}/")
    log.info(f"Annotated frames  -> {args.annotated_dir}/")
    log.info(f"Total time        -> {elapsed:.1f}s")


def main():
    args = parse_args()
    process_scoreboard_video(args)


if __name__ == "__main__":
    main()
