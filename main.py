"""
Pedestrian Crossing Analyzer
-----------------------------
This script reads a video ('traffic.mp4'), detects people using YOLOv8,
checks whether each person is inside a defined crosswalk polygon,
and saves an annotated output video ('result.mp4').

Requirements:
    pip install opencv-python ultralytics numpy
"""

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

INPUT_VIDEO  = "cross-walk.mp4"     # Path to the input video file
OUTPUT_VIDEO = "result.mp4"      # Path to the output (annotated) video file
MODEL_PATH   = "yolov8n.pt"      # Lightweight YOLOv8 model (nano version)
PERSON_CLASS = 0                 # COCO class ID for 'person'

# ---------------------------------------------------------------------------
# 2. DEFINE THE CROSSWALK POLYGON
# ---------------------------------------------------------------------------
# The crosswalk is defined as a polygon (list of 2D points).
# These coordinates are in PIXELS relative to the video frame.
# IMPORTANT: Adjust these four points to match the crosswalk in YOUR video.
# The polygon is drawn counter-clockwise / clockwise (order matters for the shape).
#
# Here's an example for a 1280x720 video. Replace with your own points:
crosswalk_polygon = np.array([
    [0, 370],    # Top-left (start of zebra stripes, left edge)
    [1420, 400],   # Top-right (start of zebra stripes, right edge)
    [1490, 720],   # Bottom-right (bottom-right of zebra stripes)
    [0, 720],     # Bottom-left (bottom-left of zebra stripes)
], dtype=np.int32)


# ---------------------------------------------------------------------------
# 3. LOAD THE YOLOv8 MODEL
# ---------------------------------------------------------------------------
# The model will automatically download 'yolov8n.pt' the first time you run it.
print("[INFO] Loading YOLOv8 model...")
model = YOLO(MODEL_PATH)


# ---------------------------------------------------------------------------
# 4. OPEN THE INPUT VIDEO
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise IOError(f"Could not open video file: {INPUT_VIDEO}")

# Read video properties so we can create an output video with the SAME settings
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps          = cap.get(cv2.CAP_PROP_FPS)

print(f"[INFO] Input video: {frame_width}x{frame_height} @ {fps:.2f} FPS")


# ---------------------------------------------------------------------------
# 5. SET UP THE VIDEO WRITER (for the output file)
# ---------------------------------------------------------------------------
# 'mp4v' is a widely supported codec for MP4 files.
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))


# ---------------------------------------------------------------------------
# 6. HELPER FUNCTION: does a point fall inside the crosswalk polygon?
# ---------------------------------------------------------------------------
def is_inside_polygon(point, polygon):
    """
    Uses OpenCV's pointPolygonTest to check whether a point lies inside
    a given polygon. Returns True if inside (or on the edge), False otherwise.
    """
    # cv2.pointPolygonTest returns:
    #   positive value  -> inside
    #   zero            -> on the edge
    #   negative value  -> outside
    result = cv2.pointPolygonTest(polygon, point, measureDist=False)
    return result >= 0


# ---------------------------------------------------------------------------
# 7. PROCESS THE VIDEO FRAME BY FRAME
# ---------------------------------------------------------------------------
frame_count = 0

while True:
    # Read the next frame
    ret, frame = cap.read()

    # If ret is False, we've reached the end of the video
    if not ret:
        break

    frame_count += 1

    # -----------------------------------------------------------------------
    # 7a. Run YOLOv8 inference on this frame
    # -----------------------------------------------------------------------
    # verbose=False silences the per-frame console output from Ultralytics.
    results = model(frame, verbose=False)[0]

    # -----------------------------------------------------------------------
    # 7b. Draw the crosswalk polygon (BLUE) on the frame
    # -----------------------------------------------------------------------
    # cv2.polylines draws the polygon outline.
    # We use a copy of the polygon reshaped as (N, 1, 2) which OpenCV expects.
    cv2.polylines(
        frame,
        [crosswalk_polygon.reshape((-1, 1, 2))],
        isClosed=True,
        color=(255, 0, 0),   # BGR color: Blue
        thickness=3,
    )

    # Optional: also fill the polygon with a semi-transparent blue overlay
    # to make the crosswalk zone visually obvious.
    overlay = frame.copy()
    cv2.fillPoly(overlay, [crosswalk_polygon], color=(255, 0, 0))
    frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)

    # -----------------------------------------------------------------------
    # 7c. Loop through each detected object
    # -----------------------------------------------------------------------
    for box in results.boxes:
        # Get the class ID for this detection
        class_id = int(box.cls[0])

        # Skip anything that is not a 'person'
        if class_id != PERSON_CLASS:
            continue

        # Get the bounding box coordinates (x1, y1, x2, y2) as integers
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # -------------------------------------------------------------------
        # 7d. Compute the "feet" point = bottom-center of the bounding box
        # -------------------------------------------------------------------
        feet_x = (x1 + x2) // 2      # horizontal center
        feet_y = y2                  # bottom edge

        # -------------------------------------------------------------------
        # 7e. Check if the feet are inside the crosswalk polygon
        # -------------------------------------------------------------------
        inside = is_inside_polygon((feet_x, feet_y), crosswalk_polygon)

        # -------------------------------------------------------------------
        # 7f. Choose colors / labels based on the check
        # -------------------------------------------------------------------
        if inside:
            color = (0, 255, 0)          # Green (BGR)
            label = "Using Crossing"
        else:
            color = (0, 0, 255)          # Red (BGR)
            label = "Not Using Crossing"

        # -------------------------------------------------------------------
        # 7g. Draw the bounding box around the person
        # -------------------------------------------------------------------
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

        # -------------------------------------------------------------------
        # 7h. Draw the label with a filled background for readability
        # -------------------------------------------------------------------
        # Compute text size so we can size the background rectangle
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        # Background rectangle just above the bounding box
        cv2.rectangle(
            frame,
            (x1, y1 - text_h - 10),
            (x1 + text_w + 10, y1),
            color,
            thickness=-1,   # -1 means filled
        )

        # Put the text on top of the filled rectangle
        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # white text
            2,
            cv2.LINE_AA,
        )

        # -------------------------------------------------------------------
        # 7i. Draw a small circle at the "feet" point for clarity
        # -------------------------------------------------------------------
        cv2.circle(frame, (feet_x, feet_y), 5, color, -1)

    # -----------------------------------------------------------------------
    # 7j. Write the annotated frame to the output video
    # -----------------------------------------------------------------------
    out.write(frame)

    # Print progress every 30 frames
    if frame_count % 30 == 0:
        print(f"[INFO] Processed {frame_count} frames...")


# ---------------------------------------------------------------------------
# 8. CLEAN UP: release everything safely
# ---------------------------------------------------------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"[DONE] Processed {frame_count} frames.")
print(f"[DONE] Output saved to: {OUTPUT_VIDEO}")
