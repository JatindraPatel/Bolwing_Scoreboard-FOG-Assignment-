# 🎳 Bowling Scoreboard Data Extraction — Video OCR + Temporal Grid Reconstruction

Built for the **Computer Vision Engineer Assignment (Round 1) at FOG**.

Given a video of a live bowling alley scoreboard, this pipeline extracts the scoreboard
data (player name, per-frame pin counts, running totals) and reconstructs it back into a
proper table — not just a flat dump of whatever text an OCR engine happened to read.

---

## Why this isn't "just OCR on some frames"

A minimal solution here is three lines: sample a frame, crop the top of the image, run
OCR, dump whatever text comes back into a JSON list. That works, but it throws away two
things that actually matter for a scoreboard specifically:

1. **Layout.** A scoreboard is a grid — rows per player, columns per frame. A flat list
   of strings loses that structure entirely; you can't tell which number belongs to
   which column without re-reading the image yourself.
2. **Reliability over time.** OCR on a single video frame is noisy — a digit gets
   misread once in a while, especially where a graphic (the pins animation here)
   partially overlaps the score row. A single frame's reading isn't something you'd
   want to treat as ground truth.

So this pipeline does two extra things beyond baseline OCR:

- **Grid reconstruction** — clusters each frame's raw OCR detections by position
  (row = y-position, column = x-position) to rebuild an actual 2D table per frame.
- **Temporal smoothing** — keeps a rolling window of the last few frames' grids and
  majority-votes each cell, so a one-off misread on a single frame gets corrected by
  the surrounding frames, while genuine score changes still get picked up as the
  window moves forward.

---

## Pipeline

```
video
  │
  ├─ sample every Nth frame           (--every, default 30)
  ├─ crop top X% of frame (ROI)       (--roi, default 0.40)
  ├─ grayscale + CLAHE contrast boost (--no-clahe to disable)
  ├─ upscale before OCR               (--upscale, default 1.5x)
  ├─ EasyOCR → (bbox, text, confidence) per detection
  ├─ drop detections below --min-conf
  ├─ skip frame if text is identical to the last kept frame (dedup)
  ├─ cluster detections → 2D grid (rows via y, columns via x)
  ├─ push grid into rolling window → majority-vote → smoothed grid
  └─ save: raw crop, annotated crop, per-frame JSON, CSV, final scorecard
```

**On the ROI crop + fixed top-40% assumption:** the scoreboard overlay sits in a
consistent screen position for the whole video, so a static crop is enough here and
keeps OCR fast (no need to run it on the full frame, or on frames that don't matter).
It's exposed as `--roi` rather than hardcoded so it can be adjusted for a different
camera angle/broadcast layout without touching the code.

**On CLAHE + upscaling:** scoreboard text in a screen-recorded video is small relative
to the frame and the background brightness isn't uniform (the pins graphic behind part
of the score row, for instance). CLAHE evens out local contrast and upscaling gives OCR
more pixels to work with — both are standard, cheap wins for small-text OCR before
reaching for anything heavier.

**On the dedup + smoothing combination:** these solve different problems and are easy
to mix up, so worth being explicit — dedup (skip-if-identical) exists to avoid writing
the same reading over and over while the scoreboard is static; smoothing (majority vote
over a window) exists to fix single-frame misreads *within* a run of readings that
aren't identical because of OCR noise. Smoothing over the *entire* video instead of a
rolling window would be wrong here, since real values (running totals) do change as the
game progresses — a global vote would freeze them at whatever value appeared most often
across the whole clip.

---

## Tech Stack

| Component        | Tool/Library |
|-------------------|--------------|
| Language          | Python 3.10+ |
| Video / Image ops | OpenCV (`opencv-python`) |
| OCR Engine        | EasyOCR |
| Array/Geometry    | NumPy |
| Testing           | Plain-assert tests (`test_main.py`), pytest-compatible |
| Output Formats    | JSON, CSV |

---

## Project Structure

```
.
├── main.py                     # Pipeline: capture → preprocess → OCR → grid → smoothing
├── test_main.py                # Unit tests for the clustering / grid / smoothing logic
├── requirements.txt
├── bowling_scoreboard.mp4      # Input video
├── extracted_frames/           # Raw cropped ROI per kept frame
├── annotated_frames/           # Same frames with OCR boxes drawn on the preprocessed image
├── output_scoreboard.json      # Per-frame: detected_text, grid, smoothed_grid
├── output_scoreboard.csv       # Flattened per-frame text, for a quick spreadsheet look
├── final_scorecard.json        # The last frame's temporally-smoothed grid (best-guess final table)
└── README.md
```

---

## Installation

```bash
git clone https://github.com/JatindraPatel/Bolwing_Scoreboard-FOG-Assignment-.git
cd Bolwing_Scoreboard-FOG-Assignment-
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py
```

or with a specific video and tuned parameters:

```bash
python main.py path/to/video.mp4 --every 20 --roi 0.35 --min-conf 0.4 --window 7
```

| Flag | Default | What it does |
|---|---|---|
| `video` (positional) | `bowling_scoreboard.mp4` | Input video path |
| `--every` | `30` | Process every Nth frame |
| `--roi` | `0.40` | Fraction of frame height (top) treated as the scoreboard ROI |
| `--min-conf` | `0.30` | Minimum OCR confidence to keep a detection |
| `--upscale` | `1.5` | Upscale factor applied to the ROI before OCR |
| `--no-clahe` | off | Disable CLAHE contrast enhancement |
| `--row-gap` | `15` | Base y-distance (px) for clustering detections into rows |
| `--col-gap` | `25` | Base x-distance (px) for clustering detections into columns |
| `--window` | `5` | Rolling window size (in kept frames) for majority-vote smoothing |
| `--keep-duplicates` | off | Don't skip frames whose text matches the previous kept frame |
| `--out-json` / `--out-csv` / `--out-final` | see above | Output paths |
| `--frames-dir` / `--annotated-dir` | see above | Output image folders |
| `--gpu` | off | Use GPU for OCR if available |

---

## Sample Output

**`output_scoreboard.json`** (one entry per kept frame):
```json
{
    "frame": 390,
    "timestamp_sec": 13.0,
    "detected_text": ["TARUN", "1", "2", "3", "4", "5", "6", "TTL", "J", "X", "5", "31"],
    "grid": [
        ["TARUN", "1", "2", "3", "4", "5", "6", "TTL"],
        ["J", "X", "5", "", "7", "4", "", "31"]
    ],
    "smoothed_grid": [
        ["TARUN", "1", "2", "3", "4", "5", "6", "TTL"],
        ["J", "X", "5", "-", "7", "4", "-", "31"]
    ]
}
```

**`final_scorecard.json`** — the temporally-smoothed grid from the last processed frame,
i.e. the pipeline's single best-guess reconstruction of the full scoreboard.

---

## Testing

The clustering / grid-reconstruction / smoothing logic is pure Python (no OCR or video
needed to test it), so it's covered by unit tests using synthetic detections:

```bash
python test_main.py
# or, if pytest is installed:
python -m pytest test_main.py -v
```

Covers: 1D clustering correctness, grid reconstruction from raw detections, majority-vote
smoothing fixing a single-frame misread, smoothing correctly tracking a genuine value
change once the window slides past the old value, and edge cases (empty input).

---

## Assumptions & Limitations

- Assumes the scoreboard sits in a fixed screen region for the whole video (true for
  this footage); `--roi` makes this adjustable rather than hardcoded, but a moving or
  panning camera would need actual scoreboard detection, not a static crop.
- Row/column clustering thresholds (`--row-gap`, `--col-gap`) are tuned for this
  footage's resolution and text size — a very differently-sized scoreboard may need
  different values.
- EasyOCR still occasionally misreads visually similar characters (`0`/`O`, `1`/`I`);
  smoothing reduces the impact of this but doesn't eliminate it, and the annotated
  frames make it easy to see exactly where a misread happened.

## Future Improvements

- Replace the fixed ROI crop with an actual scoreboard detector (classical template
  matching, or a small trained detector) so the pipeline works on footage where the
  scoreboard isn't in a fixed position.
- Map the reconstructed grid into a proper per-player bowling schema (10 frames + total,
  with `X`/`/` scoring conventions) instead of a generic text table.
- Surface each smoothed cell's vote agreement ratio (e.g. "4/5 frames agreed") as a
  simple confidence score alongside the final value.

---

## Author

**Jatindra Patel**
Data Analyst Intern | Aspiring BI/Data Analyst
[GitHub](https://github.com/JatindraPatel) • [LinkedIn](https://linkedin.com/in/jatindrapatel/) • [Portfolio](https://jatindraportfolio.vercel.app/)

*Submitted for the Computer Vision Engineer Assignment (Round 1) at FOG.*
