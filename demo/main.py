import cv2
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

# =========================
# 1. Model path
# =========================
# model_path = r"F:\voilence\violence-detection\best (12).pt"   working
# model_path = r"F:\voilence\violence-detection\Yolo_nano_weights.pt"
model_path = r"F:\voilence\violence-detection\Yolo_small_weights.pt"

# =========================
# 2. Input video
# =========================
# video_path = r"F:\voilence\violence-detection\fight.mp4"
video_path = r"F:\voilence\violence-detection\data\gemini_generated_video_d617616c.mp4"

# =========================
# 3. Output path
#    Same folder as input video
# =========================
count=0
video_path = Path(video_path)
output_path = video_path.parent / f"fight_detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
# Load model
model = YOLO(model_path)


def detect_class_in_video(video_path, output_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    # Video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    # MP4 writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (frame_width, frame_height)
    )

    frame_count = 0

    print("Processing video...")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # YOLO inference
        results = model(frame, verbose=False)

        # Process detections
        for detection in results[0].boxes:

            class_id = int(detection.cls[0])
            confidence = float(detection.conf[0])

            # Class 1 = Violence
            if class_id == 1:

                x1, y1, x2, y2 = map(
                    int,
                    detection.xyxy[0].tolist()
                )

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                # Label
                label = f"Violence {confidence:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

        # Save frame
        out.write(frame)

        # Optional progress
        if frame_count % 30 == 0:
            print(f"Processed frames: {frame_count}")

    # Release resources
    cap.release()
    out.release()

    print("\nProcessing completed!")
    print(f"Output saved at:\n{output_path}")


# =========================
# 4. Run detection
# =========================
detect_class_in_video(video_path, output_path)