# 🎳 Bowling Scoreboard Data Extraction using Computer Vision (OCR)

This project was built for the **Computer Vision Engineer Assignment (Round 1) at FOG**.
It extracts scoreboard data (player names, frame scores, pins knocked down, totals, etc.)
from a bowling alley video using frame sampling, Region-of-Interest (ROI) cropping, and
OCR (Optical Character Recognition).

---

## 📌 Problem Statement

Given a video of a bowling scoreboard display, build a Computer Vision pipeline that:
- Processes the video frame-by-frame
- Detects and extracts the scoreboard region
- Reads the text/numbers on the scoreboard accurately
- Outputs the extracted data in a structured, usable format

---

## 🧠 Approach

Since bowling scoreboard displays are usually static overlays positioned at the **top of
the screen**, the pipeline avoids processing every single frame (which would be slow and
redundant) and instead:

1. **Frame Sampling** – Reads the video and processes every **30th frame** (~1 frame/sec
   at 30 FPS), which is more than enough since scoreboard values don't change every frame.
2. **ROI Cropping** – Instead of running OCR on the full frame, the script crops the
   **top 40% of the frame**, where the scoreboard is consistently located. This reduces
   noise, speeds up OCR, and improves accuracy by removing irrelevant background (lanes,
   players, pins, etc.).
3. **Preprocessing** – Converts the cropped ROI to grayscale to improve OCR readability
   and reduce the effect of lighting/color variation on the display.
4. **OCR Extraction** – Uses **EasyOCR** to detect and read all text within the cropped
   scoreboard region (player names, frame-by-frame pin counts, totals, etc.).
5. **Confidence Filtering** – Discards OCR detections with a confidence score below
   `0.3` to filter out noisy/false text detections.
6. **Structured Output** – Saves the results (frame number, timestamp, detected text)
   into a JSON file (`output_scoreboard.json`), and also saves each processed/cropped
   frame as an image in `extracted_frames/` for verification and debugging.

---

## 🛠️ Tech Stack

| Component        | Tool/Library |
|-------------------|--------------|
| Language          | Python 3.10+ |
| Video Processing  | OpenCV (`opencv-python`) |
| OCR Engine        | EasyOCR |
| Output Format     | JSON |

---

## 📂 Project Structure

```
.
├── main.py                     # Main script — video processing + OCR pipeline
├── bowling_scoreboard.mp4      # Input video (sample scoreboard footage)
├── extracted_frames/           # Auto-generated cropped scoreboard frames
├── output_scoreboard.json      # Auto-generated structured OCR output
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
   pip install opencv-python easyocr
   ```

---

## ▶️ Usage

1. Place your input video in the project root and update the filename if needed
   (default: `bowling_scoreboard.mp4`) in the last line of `main.py`:
   ```python
   process_scoreboard_video("bowling_scoreboard.mp4")
   ```

2. Run the script:
   ```bash
   python main.py
   ```

3. On completion, you will get:
   - `extracted_frames/` → cropped scoreboard images for every sampled frame
   - `output_scoreboard.json` → structured OCR results

---

## 📄 Sample Output (`output_scoreboard.json`)

```json
{
    "frame": 120,
    "timestamp_sec": 4.0,
    "detected_text": [
        "TARUN",
        "1",
        "12",
        "3 | 4 | 5",
        "TTC",
        "74",
        "0",
        "15",
        "20"
    ]
}
```

Each object represents one sampled frame and contains:
- `frame` – frame number in the video
- `timestamp_sec` – corresponding timestamp in seconds
- `detected_text` – list of all text/numbers detected on the scoreboard for that frame
  (player names, per-frame pin counts, running totals, etc.)

---

## 🎥 Demo Video

The demo video (submitted alongside this repository) shows:
- The input bowling scoreboard video
- The script running end-to-end in the terminal
- The scoreboard region being detected and cropped
- The final extracted JSON output

---

## ⚠️ Assumptions & Limitations

- Assumes the scoreboard is consistently positioned in the **top 40%** of the video frame.
  For different camera angles/layouts, the ROI crop coordinates would need adjustment.
- Sampling every 30th frame assumes a 30 FPS source video; this can be tuned for other
  frame rates.
- EasyOCR occasionally misreads visually similar characters (e.g., `0`/`O`, `1`/`I`)
  under low-contrast or motion-blur conditions.
- No frame-to-frame deduplication is currently applied — visually identical/static
  scoreboard frames may produce repeated OCR results.

---

## 🚀 Future Improvements

- Auto-detect the scoreboard region dynamically (e.g., via template matching or a
  lightweight object detector) instead of a fixed crop percentage.
- Deduplicate consecutive frames with identical scoreboard values to reduce redundant
  output entries.
- Map raw OCR text into a structured schema (per-player, per-frame pin counts + running
  total) instead of a flat list of detected strings.
- Add post-processing/spell-correction rules specific to bowling scoring conventions
  (e.g., `X` for strike, `/` for spare) to improve accuracy.

---

## 👤 Author

**Jatindra Patel**
Data Analyst Intern | Aspiring BI/Data Analyst
[GitHub](https://github.com/JatindraPatel) • [LinkedIn](https://linkedin.com/in/jatindrapatel/) • [Portfolio](https://jatindraportfolio.vercel.app/)

*Submitted for the Computer Vision Engineer Assignment (Round 1) at FOG.*
