# 🎳 Bowling Scoreboard Data Extraction using Computer Vision (OCR)

This project was built for the **Computer Vision Engineer Assignment (Round 1) at FOG**.
It extracts scoreboard data (player names, frame scores, pins knocked down, totals, etc.)
from a bowling alley video using frame sampling, ROI cropping, OCR, and a bit of
post-processing that turns raw OCR text back into the scoreboard's row/column structure.

---

## 📌 Problem Statement

Given a video of a bowling scoreboard display, build a Computer Vision pipeline that:
- Processes the video frame-by-frame
- Detects and extracts the scoreboard region
- Reads the text/numbers on the scoreboard accurately
- Outputs the extracted data in a structured, usable format

---

## 🧠 Approach

Bowling scoreboard displays are static overlays positioned at the **top of the screen**,
so the pipeline avoids running OCR on every single frame and instead:

1. **Frame Sampling** – Reads the video and processes every **Nth frame** (default: 30,
   ~1 frame/sec at 30 FPS) — configurable via CLI, since scoreboard values don't change
   every frame.
2. **ROI Cropping** – Crops the **top 40% of the frame** (configurable), where the
   scoreboard is consistently located, instead of running OCR on the full frame. This
   removes irrelevant background (lanes, players, pins) and speeds up OCR.
3. **Preprocessing** – Converts the cropped ROI to grayscale to improve OCR readability.
4. **OCR Extraction** – Uses **EasyOCR** to detect and read all text within the cropped
   scoreboard region.
5. **Confidence Filtering** – Discards OCR detections below a configurable confidence
   threshold (default `0.3`).
6. **Row Reconstruction** – This is the part most basic OCR scripts skip: EasyOCR returns
   a flat, unordered list of text strings, which loses the scoreboard's grid structure.
   This project clusters each detection by its **vertical (y) position** to regroup
   the raw text back into rows (one row per player/header line), and sorts each row
   left-to-right, so the output actually mirrors the scoreboard layout instead of being
   a jumbled list.
7. **Duplicate Skipping** – Since the scoreboard often doesn't change for several seconds
   at a time, consecutive sampled frames with *identical* detected text are skipped by
   default (`--keep-duplicates` to disable this), so the output isn't full of repeated,
   redundant entries.
8. **Visual Verification** – Every kept frame's OCR detections are drawn as bounding
   boxes + labels on a copy of the cropped ROI and saved to `annotated_frames/`, so the
   detection step can be visually verified frame-by-frame instead of trusting a black box.
9. **Structured Output** – Results are saved to both **JSON** (`output_scoreboard.json`,
   with the reconstructed rows included) and a flattened **CSV** (`output_scoreboard.csv`)
   for quick inspection in a spreadsheet.

---

## 🛠️ Tech Stack

| Component        | Tool/Library |
|-------------------|--------------|
| Language          | Python 3.10+ |
| Video Processing  | OpenCV (`opencv-python`) |
| OCR Engine        | EasyOCR |
| Array/Geometry    | NumPy |
| Output Formats    | JSON, CSV |

---

## 📂 Project Structure

```
.
├── main.py                     # Main script — video processing + OCR + row reconstruction
├── bowling_scoreboard.mp4      # Input video (sample scoreboard footage)
├── extracted_frames/           # Auto-generated cropped scoreboard frames
├── annotated_frames/           # Auto-generated frames with OCR boxes drawn on them
├── output_scoreboard.json      # Auto-generated structured OCR output (with rows)
├── output_scoreboard.csv       # Auto-generated flattened CSV output
└── README.md
```

---

## ⚙️ Installation

1. Clone the repository
   ```bash
   git clone https://github.com/JatindraPatel/Bolwing_Scoreboard-FOG-Assignment-.git
   cd Bolwing_Scoreboard-FOG-Assignment-
   ```

2. Create a virtual environment (recommended)
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. Install dependencies
   ```bash
   pip install opencv-python easyocr numpy
   ```

---

## ▶️ Usage

Run with defaults (looks for `bowling_scoreboard.mp4` in the current folder):
```bash
python main.py
```

Or point it at a specific video and tweak the pipeline via CLI flags:
```bash
python main.py path/to/video.mp4 --every 20 --roi 0.35 --min-conf 0.4
```

| Flag | Default | Description |
|------|---------|--------------|
| `video` (positional) | `bowling_scoreboard.mp4` | Path to the input video |
| `--every` | `30` | Process every Nth frame |
| `--roi` | `0.40` | Fraction of frame height (from top) treated as scoreboard ROI |
| `--min-conf` | `0.30` | Minimum OCR confidence to keep a detection |
| `--out-json` | `output_scoreboard.json` | JSON output path |
| `--out-csv` | `output_scoreboard.csv` | CSV output path |
| `--frames-dir` | `extracted_frames` | Where cropped ROI frames are saved |
| `--annotated-dir` | `annotated_frames` | Where annotated (boxed) ROI frames are saved |
| `--keep-duplicates` | off | Save every sampled frame, even unchanged ones |
| `--gpu` | off | Use GPU for OCR if available |

On completion you get:
- `extracted_frames/` → plain cropped scoreboard images
- `annotated_frames/` → the same frames with OCR bounding boxes + text drawn on them
- `output_scoreboard.json` → structured OCR output, including reconstructed rows
- `output_scoreboard.csv` → the same data, flattened for quick viewing in Excel/Sheets

---

## 📄 Sample Output (`output_scoreboard.json`)

```json
{
    "frame": 120,
    "timestamp_sec": 4.0,
    "detected_text": ["TARUN", "1", "12", "3 | 4 | 5", "TTC", "74", "0", "15", "20"],
    "rows": [
        ["TARUN", "1", "12", "3 | 4 | 5"],
        ["TTC", "74", "0", "15", "20"]
    ]
}
```

Each object represents one *unique* sampled frame (duplicates skipped) and contains:
- `frame` / `timestamp_sec` – which frame this is and when it occurs in the video
- `detected_text` – flat list of everything OCR read off the scoreboard (kept for
  backward compatibility / quick scanning)
- `rows` – the same text, reconstructed into the scoreboard's actual row layout based
  on each detection's vertical position

---

## 🎥 Demo Video

The demo video (linked in the submission) shows:
- The input bowling scoreboard video
- The script running end-to-end in the terminal
- The scoreboard region being detected/cropped
- The final extracted JSON/CSV output

---

## ⚠️ Assumptions & Limitations

- Assumes the scoreboard is consistently positioned in the **top 40%** of the frame by
  default; this is a CLI flag (`--roi`) so it can be adjusted per video/camera angle.
- Row grouping uses a fixed y-distance threshold (15px) to cluster detections into
  rows — works well for this footage's resolution but may need tuning for very
  different video resolutions.
- EasyOCR occasionally misreads visually similar characters (e.g. `0`/`O`, `1`/`I`)
  under low-contrast or motion-blur conditions; the annotated frames make it easy to
  spot exactly which detections these were.

---

## 🚀 Future Improvements

- Auto-detect the scoreboard region dynamically (template matching / lightweight
  detector) instead of a fixed top-percentage crop.
- Map reconstructed rows into a proper per-player schema (name, 10 frame scores,
  running total) using bowling scoring conventions (`X` = strike, `/` = spare).
- Track OCR confidence trends over time to auto-flag frames likely to have
  misreads, instead of relying on a single static threshold.

---

## 👤 Author

**Jatindra Patel**
Data Analyst Intern | Aspiring BI/Data Analyst
[GitHub](https://github.com/JatindraPatel) • [LinkedIn](https://linkedin.com/in/jatindrapatel/) • [Portfolio](https://jatindraportfolio.vercel.app/)

*Submitted for the Computer Vision Engineer Assignment (Round 1) at FOG.*
