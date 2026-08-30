import cv2
import easyocr
import json
import os

def process_scoreboard_video(video_path, output_json_path="output_scoreboard.json"):
    print("OCR Engine loading ho raha hai...")
    reader = easyocr.Reader(['en'], gpu=False)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Video file nahi mili: {video_path}")
        return

    frame_count = 0
    extracted_data = []

    os.makedirs("extracted_frames", exist_ok=True)

    print("Video process ho rahi hai...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Har 30th frame process hoga (har 1 second mein)
        if frame_count % 30 == 0:
            h, w, _ = frame.shape
            
            # Scoreboard ko crop karna (Top 40% area)
            roi = frame[0:int(h * 0.4), 0:w]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # OCR se text read karna
            ocr_results = reader.readtext(gray_roi)

            frame_ocr_texts = []
            for bbox, text, prob in ocr_results:
                if prob > 0.3:
                    frame_ocr_texts.append(text)

            extracted_data.append({
                "frame": frame_count,
                "timestamp_sec": round(frame_count / cap.get(cv2.CAP_PROP_FPS), 2),
                "detected_text": frame_ocr_texts
            })

            cv2.imwrite(f"extracted_frames/frame_{frame_count}.jpg", roi)
            print(f"Frame {frame_count}: {frame_ocr_texts}")

    cap.release()

    # Result ko JSON file mein save karna
    with open(output_json_path, "w") as f:
        json.dump(extracted_data, f, indent=4)

    print(f"\nKaam complete ho gaya! Output '{output_json_path}' mein save ho gaya hai.")

if __name__ == "__main__":
    process_scoreboard_video("bowling_scoreboard.mp4")