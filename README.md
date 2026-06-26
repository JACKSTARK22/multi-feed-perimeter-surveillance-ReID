# Live Person Re-Identification (Re-ID) with YOLOv8 & OSNet

A real-time person re-identification and tracking system that combines the detection capabilities of **YOLOv8** with the robust feature embedding extraction of **OSNet (Deep-Person-ReID)** optimized via an **ArcFace loss** layer. 

The framework operates in a two-stage pipeline: First, it extracts and indexes feature templates of known individuals from a reference directory. Second, it processes real-time media streams, performing persistent tracking and identity matching against the indexed gallery with temporal embedding smoothing.

---

## 🛠️ Key Features

- **Advanced Feature Extraction**: Employs an `OSNet_x0_75` backbone enhanced with an ArcFace loss head for maximizing intra-class similarity and inter-class variance.
- **Dual-Stage Pipeline**: 
  1. **Gallery Builder**: Processes standard profile/reference image subsets of known individuals, creating a compressed, unified template pickle (`.pkl`) matrix using flip-augmentation averaging.
  2. **Live Tracking & Re-ID**: Merges high-efficiency YOLOv8 bounding boxes with a greedy IoU sequence tracker to significantly reduce inference overhead by skipping Re-ID evaluation across static frames.
- **Temporal Embedding Smoothing**: Computes a moving average over a configurable rolling queue window (`EMBEDDING_HISTORY`) to stabilize identity matching under dynamic illumination or orientation shifts.
- **Top-5 Prioritization**: Automatically prioritizes and tracks the top 5 highest-probability identity candidates in the field of view, visualizing rank-1 candidates with high-contrast target overlays.
- **Analytical Threshold Tuning**: Includes a rigorous evaluation validation script that calculates precision, recall, and $F_1$ scores across multi-threshold steps to map the optimal decision boundary for custom environments.

---

## 📂 Project Structure

```text
├── reid_live.py          # Core application script containing model, tracker, and CLI modes
├── main.py               # Simple programmatic wrapper script for configuration automation
├── known_persons/        # Reference image gallery directory structured by target identity
│   ├── john_doe/
│   │   ├── face1.jpg
│   │   └── profile2.png
│   └── jane_smith/
│       └── frame1.jpg
└── gallery.pkl           # Generated feature database containing consolidated 512-D unit vectors
```

---

## ⚡ Prerequisites & Installation

### Core Dependencies
Ensure you have a Python environment setup (v3.8+ recommended) with active CUDA support for optimal frame rates.

Install the required distributions directly via pip:
```bash
pip install ultralytics opencv-python torch torchvision torchreid
```
*(Note: `reid_live.py` implements an automated dependency checker that will attempt compilation or installation via subprocess hooks upon invocation if any elements are absent).*

---

## 🚀 Execution & Usage

The application provides two interface strategies: **CLI Arguments** directly via `reid_live.py`, or **Programmatic Configuration** via modifying `main.py`.

### Option A: Using the Command Line Interface (`reid_live.py`)

#### Step 1: Encode and Enroll Known Persons
Generate the centralized gallery vector matrix by pointing the builder at a structured directory:
```bash
python reid_live.py --mode build_gallery     --gallery_dir ./known_persons     --checkpoint ./osnet_x0_75_BEST.pth     --output gallery.pkl
```

#### Step 2: Run Live Inference / Tracking
Launch real-time parsing against a hardware camera, network stream, or compressed video source file:
```bash
python reid_live.py --mode live     --source 0     --checkpoint ./osnet_x0_75_BEST.pth     --gallery gallery.pkl     --threshold 0.55
```
*Parameters:*
- `--source`: Set to `"0"` for system webcam, an RTSP connection string (`rtsp://...`), or path to a local media file (`video.mp4`).
- `--threshold`: Defines the minimum cosine similarity match target. Scores below this are flagged as `Unknown`.

#### Step 3: Evaluate Optimal Threshold Boundaries
Run validation optimization routines over your benchmark evaluation split:
```bash
python reid_live.py --mode evaluate_threshold     --val_dir ./validation_set     --gallery gallery.pkl     --checkpoint ./osnet_x0_75_BEST.pth
```

---

### Option B: Using the Programmatic Wrapper (`main.py`)

For rapid IDE prototyping or automated deployments without shell parameters, configure your parameters directly within the `Config` structure of `main.py` and invoke:

```python
class Config:
    def __init__(self):
        self.checkpoint = "osnet_x0_75_BEST.pth" 
        self.gallery_dir = "known_persons" 
        self.gallery = "gallery.pkl"
        self.output = "gallery.pkl"
        self.source = "0"  # 0 for webcam, or path string
        self.threshold = 0.60
```
Run the automated routine directly:
```bash
python main.py
```

---

## 🎮 Runtime Interface Controls

When executing inside a live window overlay environment (`--mode live`):
- `q`: Gracefully disconnect inputs and close window sessions.
- `s`: Capture a crisp local JPEG snapshot timestamped screenshot of the raw frame tracking pipeline buffers (`screenshot_<timestamp>.jpg`).

---

## ⚙️ Performance Tuning & Architecture Configurations

Fine-tune internal hyperparameters directly near the top of `reid_live.py` to balance computational throughput and structural correctness based on hardware profiles:

| Hyperparameter | Default | Purpose |
| :--- | :--- | :--- |
| `IMG_H`, `IMG_W` | `256`, `128` | Dimensions to standardise cropped bounding boxes for the OSNet architecture input tensor. |
| `REID_EVERY_N_FRAMES` | `60` | Frame interval to execute deep feature reassignment per tracked target. Higher values maximize FPS by relying on the lightweight IoU tracker between steps. |
| `EMBEDDING_HISTORY` | `5` | Length of the rolling historical queue used to average features across a timeline trajectory to minimize occlusion fluctuations. |
| `YOLO_MODEL` | `"yolov8m.pt"` | Pretrained detection model tier. Swap to `"yolov8n.pt"` for low-latency embedded units, or `"yolov8x.pt"` for complex crowd spaces. |
