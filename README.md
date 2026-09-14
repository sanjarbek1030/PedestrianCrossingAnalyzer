# 🚶‍♂️ Pedestrian Crossing Analyzer

A computer vision project that analyzes traffic videos to determine whether pedestrians are **using a crosswalk** or **jaywalking** — using YOLOv8 for person detection and OpenCV for polygon-based zone analysis.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 What It Does

Given a traffic video, this tool:

1. Detects **every person** in each frame using **YOLOv8n** (lightweight nano model).
2. Defines a **crosswalk polygon** in pixel space.
3. Checks whether each person's **feet (bottom-center of bounding box)** fall inside the crosswalk.
4. Draws color-coded annotations:
   - 🟩 **Green box** → `"Using Crossing"`
   - 🟥 **Red box** → `"Not Using Crossing"`
   - 🟦 **Blue polygon** → the crosswalk zone
5. Saves an annotated output video with the **same resolution and framerate** as the input.

---

## 🖼️ Demo

| Input Frame | Annotated Output |
|---|---|
| Raw traffic video frame | People color-coded by crosswalk usage |

*(Add your own GIF or screenshots here after running the script.)*

---

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/your-username/pedestrian-crossing-analyzer.git
cd pedestrian-crossing-analyzer

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

# Install dependencies
pip install opencv-python ultralytics numpy
```

---

## 🚀 Usage

1. Place your input video in the project root and name it **`traffic.mp4`**.
2. Adjust the `crosswalk_polygon` coordinates in the script to match your video.
3. Run:

```bash
python pedestrian_crossing_analyzer.py
```

4. The annotated video will be saved as **`result.mp4`**.

---

## ⚙️ Configuration

Edit these values at the top of the script:

```python
INPUT_VIDEO  = "cross-walk.mp4"     # Input video path
OUTPUT_VIDEO = "result.mp4"      # Output video path
MODEL_PATH   = "yolov8n.pt"      # YOLOv8 model variant
PERSON_CLASS = 0                 # COCO class ID for 'person'
```

### Defining the Crosswalk Polygon

The crosswalk is defined as a **4-point polygon** in pixel coordinates:

```python
crosswalk_polygon = np.array([
    [0, 370],    # Top-left (start of zebra stripes, left edge)
    [1420, 400],   # Top-right (start of zebra stripes, right edge)
    [1490, 720],   # Bottom-right (bottom-right of zebra stripes)
    [0, 720],     # Bottom-left (bottom-left of zebra stripes)
], dtype=np.int32)
```

**Tip:** Use the built-in mouse-click debugger to find exact coordinates:

```python
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: ({x}, {y})")

cv2.namedWindow("debug")
cv2.setMouseCallback("debug", mouse_callback)
```

---

## 🧠 How It Works

| Step | Technique |
|------|-----------|
| **Person detection** | Ultralytics YOLOv8n (`class_id == 0`) |
| **Feet localization** | Bottom-center of bounding box |
| **Zone check** | `cv2.pointPolygonTest` against crosswalk polygon |
| **Visualization** | Colored rectangles, labels, and polygon overlay |
| **Video I/O** | `cv2.VideoCapture` / `cv2.VideoWriter` (mp4v codec) |

**No tracking libraries** — pure OpenCV + NumPy + Ultralytics YOLO.

---

## 📁 Project Structure

```
pedestrian-crossing-analyzer/
├── pedestrian_crossing_analyzer.py   # Main script
├── traffic.mp4                        # Input video (not included)
├── result.mp4                         # Output video (generated)
├── requirements.txt
└── README.md
```

---

## 📦 Requirements

```
opencv-python>=4.8.0
ultralytics>=8.0.0
numpy>=1.24.0
```

---

## 🔧 Tuning Tips

| Goal | Change |
|------|--------|
| Higher accuracy | Use `yolov8s.pt` or `yolov8m.pt` |
| GPU acceleration | Install CUDA-enabled PyTorch |
| Lower false positives | Add `conf=0.5` to `model(frame, conf=0.5)` |
| Multiple crosswalks | Store multiple polygons in a list |

---

## 🗺️ Roadmap

- [ ] Multi-crosswalk support
- [ ] Pedestrian tracking across frames (ByteTrack)
- [ ] Real-time RTSP / webcam input
- [ ] CSV/JSON report of crossing statistics
- [ ] Zone-intrusion alerts

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📜 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [COCO Dataset](https://cocodataset.org/) (for class IDs)

---

## ⭐ Show Your Support

If this project helped you, please give it a ⭐ on GitHub!
