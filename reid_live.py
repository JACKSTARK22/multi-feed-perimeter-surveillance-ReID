"""
reid_live.py — Live Person Re-Identification
─────────────────────────────────────────────
Uses your trained OSNet model + YOLOv8 detector on any camera or video.

MODES:
  1. build_gallery   — enrol known persons from a folder of reference images
  2. live            — run ReID on a webcam / RTSP stream / video file
  3. evaluate_threshold — find the best similarity threshold on your val set

USAGE:
  # Step 1: install deps (once)
  pip install ultralytics opencv-python torchreid

  # Step 2: enrol known persons
  python reid_live.py --mode build_gallery \
      --gallery_dir /path/to/known_persons \
      --checkpoint  /kaggle/working/osnet_x0_75_BEST.pth

  # Step 3: go live
  python reid_live.py --mode live \
      --source 0 \                         # 0=webcam, or rtsp://..., or video.mp4
      --checkpoint /kaggle/working/osnet_x0_75_BEST.pth \
      --gallery    gallery.pkl

GALLERY FOLDER STRUCTURE:
  known_persons/
    john_doe/
      img1.jpg
      img2.jpg
    jane_smith/
      img1.jpg
"""

import os, sys, time, argparse, pickle, warnings
import subprocess

for pkg in ["ultralytics", "opencv-python", "torchreid"]:
    imp = "cv2" if pkg == "opencv-python" else pkg
    try:
        __import__(imp)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

import cv2
import torchreid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from collections import defaultdict, deque
from ultralytics import YOLO

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

IMG_H = 256
IMG_W = 128

# Similarity threshold: scores above this = known person, below = Unknown
# Start with 0.5 and tune using --mode evaluate_threshold
DEFAULT_THRESHOLD = 0.45

# How many frames to skip between re-identification of a tracked person
# 1 = every frame (most accurate, slowest), 5 = every 5th frame (faster)
REID_EVERY_N_FRAMES = 60

# Number of recent embeddings to average per tracked person (temporal smoothing)
# Higher = more stable IDs but slower to update if person changes appearance
EMBEDDING_HISTORY = 5

# Colours for bounding boxes (BGR)
COLOURS = [
    (0,200,255), (0,255,100), (255,100,0), (200,0,255),
    (0,255,200), (255,200,0), (100,0,255), (255,0,100),
]

# YOLO model — 'yolov8n.pt' is fastest, 'yolov8m.pt' is more accurate
YOLO_MODEL = "yolov8m.pt"

# Person class ID in COCO (YOLO default)
PERSON_CLASS = 0


# ═══════════════════════════════════════════════════════════════════════
# MODEL  (copied from osnet_reid_enhanced.py — must match exactly)
# ═══════════════════════════════════════════════════════════════════════

import math

class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features,
                 scale=30.0, margin=0.30, easy_margin=False):
        super().__init__()
        self.scale  = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(margin);  self.sin_m = math.sin(margin)
        self.th    = math.cos(math.pi - margin)
        self.mm    = math.sin(math.pi - margin) * margin
        self.easy_margin = easy_margin

    def forward(self, embeddings, labels):
        embeddings  = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight,  p=2, dim=1)
        cosine = F.linear(embeddings, weight_norm).clamp(-1., 1.)
        sine   = torch.sqrt(1. - cosine.pow(2))
        phi    = cosine * self.cos_m - sine * self.sin_m
        phi    = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1,1).long(), 1)
        return (one_hot * phi + (1. - one_hot) * cosine) * self.scale


class OSNetWithArcFace(nn.Module):
    def __init__(self, num_classes, arc_scale=30., arc_margin=0.30):
        super().__init__()
        self.backbone = torchreid.models.build_model(
            name="osnet_x0_75", num_classes=num_classes,
            pretrained=False, loss="softmax")
        self.backbone.classifier = nn.Identity()
        self.feat_dim   = 512
        self.bottleneck = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.arcface    = ArcFaceLoss(self.feat_dim, num_classes,
                                      scale=arc_scale, margin=arc_margin)

    def forward(self, x, labels=None):
        feats    = self.backbone(x)
        bn_feats = self.bottleneck(feats)
        if self.training and labels is not None:
            return feats, self.arcface(bn_feats, labels)
        return F.normalize(bn_feats, p=2, dim=1)

    @torch.no_grad()
    def extract(self, x):
        feats    = self.backbone(x)
        bn_feats = self.bottleneck(feats)
        return F.normalize(bn_feats, p=2, dim=1)

    @torch.no_grad()
    def extract_with_flip(self, x):
        f1 = self.extract(x)
        f2 = self.extract(torch.flip(x, dims=[3]))
        return F.normalize((f1 + f2) / 2.0, p=2, dim=1)


# ═══════════════════════════════════════════════════════════════════════
# LOAD MODEL FROM CHECKPOINT
# ═══════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path, device):
    print(f"\n[Model] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    num_classes = ckpt["num_classes"]
    arc_scale   = ckpt.get("arc_scale",  30.0)
    arc_margin  = ckpt.get("arc_margin", 0.30)

    model = OSNetWithArcFace(num_classes, arc_scale, arc_margin).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    r1_  = ckpt.get("val_rank1", None)
    map_ = ckpt.get("val_mAP",   None)
    print(f"[Model] num_classes={num_classes}  "
          f"val_Rank1={f'{r1_:.2%}' if r1_ is not None else '?'}  "
          f"val_mAP={f'{map_:.2%}' if map_ is not None else '?'}  "
          f"epoch={ckpt.get('epoch', '?')}")
    return model


# ═══════════════════════════════════════════════════════════════════════
# IMAGE PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════

eval_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def preprocess_crop(bgr_crop):
    """Convert a BGR numpy crop (from OpenCV) to a model-ready tensor."""
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return eval_transform(pil)


# ═══════════════════════════════════════════════════════════════════════
# GALLERY  —  stores reference embeddings for known persons
# ═══════════════════════════════════════════════════════════════════════

class Gallery:
    """
    Holds one or more reference embeddings per named person.
    Matching uses the average embedding across all reference images.
    """
    def __init__(self):
        self.embeddings = {}   # name → averaged 512-d unit vector

    def add(self, name, embedding):
        """Add or update a person's embedding (averages multiple references)."""
        if name in self.embeddings:
            # Running average — keeps the gallery compact
            self.embeddings[name] = F.normalize(
                (self.embeddings[name] + embedding) / 2.0,
                p=2, dim=0)
        else:
            self.embeddings[name] = embedding

    def match(self, query_emb, threshold):
        """
        Returns (name, score) of the best match.
        Returns ("Unknown", score) if best score < threshold.
        """
        if not self.embeddings:
            return "Unknown", 0.0

        best_name, best_score = "Unknown", 0.0
        for name, ref_emb in self.embeddings.items():
            score = float(torch.dot(query_emb.cpu(), ref_emb.cpu()))
            if score > best_score:
                best_score, best_name = score, name

        if best_score < threshold:
            return "Unknown", best_score
        return best_name, best_score

    def save(self, path):
        data = {name: emb.cpu().numpy()
                for name, emb in self.embeddings.items()}
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[Gallery] Saved {len(data)} persons → {path}")

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        g = cls()
        for name, arr in data.items():
            g.embeddings[name] = torch.tensor(arr)
        print(f"[Gallery] Loaded {len(g.embeddings)} persons from {path}")
        return g

    def __len__(self):
        return len(self.embeddings)


# ═══════════════════════════════════════════════════════════════════════
# TRACKER  —  lightweight IoU-based tracker to assign consistent IDs
#             so we don't re-run ReID on every single frame
# ═══════════════════════════════════════════════════════════════════════

def iou(boxA, boxB):
    """Intersection over Union of two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


class SimpleTracker:
    """
    Assigns a persistent track_id to each detected person across frames
    using greedy IoU matching. Keeps a rolling history of embeddings per
    track for temporal smoothing.
    """
    def __init__(self, iou_thresh=0.35, max_lost=30):
        self.iou_thresh   = iou_thresh
        self.max_lost     = max_lost      # frames before a track is dropped
        self.tracks       = {}            # track_id → dict
        self.next_id      = 0
        self.frame_count  = 0

    def update(self, detections):
        """
        detections: list of [x1,y1,x2,y2] boxes
        Returns: list of track_ids aligned with detections
        """
        self.frame_count += 1
        active = {tid: t for tid, t in self.tracks.items()
                  if t["lost"] < self.max_lost}

        if not active:
            track_ids = []
            for box in detections:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {
                    "box": box, "lost": 0,
                    "frame_first_seen": self.frame_count,
                    "emb_history": deque(maxlen=EMBEDDING_HISTORY),
                    "identity": None, "identity_score": 0.0,
                    "reid_frame": -1,
                }
                track_ids.append(tid)
            return track_ids

        # Greedy IoU matching
        active_ids   = list(active.keys())
        active_boxes = [active[tid]["box"] for tid in active_ids]
        matched_det  = set(); matched_trk = set()
        assignments  = {}  # det_idx → track_id

        if len(detections) == 0:
            for tid in active_ids:
                self.tracks[tid]["lost"] += 1
            return []

        iou_matrix = np.array([
            [iou(det, trk) for trk in active_boxes]
            for det in detections
        ])

        while True:
            if iou_matrix.size == 0: break
            idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            di, ti = idx
            if iou_matrix[di, ti] < self.iou_thresh: break
            det_idx = di; trk_idx = ti
            assignments[det_idx] = active_ids[trk_idx]
            matched_det.add(det_idx); matched_trk.add(trk_idx)
            iou_matrix[di, :] = -1
            iou_matrix[:, ti] = -1

        # Update matched tracks
        for di, tid in assignments.items():
            self.tracks[tid]["box"]  = detections[di]
            self.tracks[tid]["lost"] = 0

        # Mark unmatched tracks as lost
        for ti, tid in enumerate(active_ids):
            if ti not in matched_trk:
                self.tracks[tid]["lost"] += 1

        # Create new tracks for unmatched detections
        track_ids = []
        for di, box in enumerate(detections):
            if di in assignments:
                track_ids.append(assignments[di])
            else:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {
                    "box": box, "lost": 0,
                    "frame_first_seen": self.frame_count,
                    "emb_history": deque(maxlen=EMBEDDING_HISTORY),
                    "identity": None, "identity_score": 0.0,
                    "reid_frame": -1,
                }
                track_ids.append(tid)

        return track_ids


# ═══════════════════════════════════════════════════════════════════════
# DRAWING UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def draw_box(frame, box, label, score, colour):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    text     = f"{label}  {score:.2f}"
    font     = cv2.FONT_HERSHEY_SIMPLEX
    scale    = 0.55
    thick    = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)

    # Background pill behind label
    cv2.rectangle(frame,
                  (x1, y1 - th - baseline - 6),
                  (x1 + tw + 8, y1),
                  colour, -1)
    cv2.putText(frame, text,
                (x1 + 4, y1 - baseline - 2),
                font, scale, (0, 0, 0), thick, cv2.LINE_AA)


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2, cv2.LINE_AA)


def draw_gallery_legend(frame, gallery):
    """Small legend in top-right corner showing enrolled persons."""
    names = list(gallery.embeddings.keys())
    x = frame.shape[1] - 160
    cv2.rectangle(frame, (x-4, 4), (frame.shape[1]-4, 20 + 18*len(names)),
                  (30, 30, 30), -1)
    cv2.putText(frame, "Gallery:", (x, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
    for i, name in enumerate(names):
        col = COLOURS[i % len(COLOURS)]
        cv2.putText(frame, f"  {name}", (x, 34 + 18*i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)


# ═══════════════════════════════════════════════════════════════════════
# MODE 1: BUILD GALLERY
# ═══════════════════════════════════════════════════════════════════════

def build_gallery(args):
    """
    Enrols known persons from a folder structure:
        gallery_dir/
            person_name/
                img1.jpg  img2.jpg ...
    Saves gallery.pkl to --output (default: gallery.pkl)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.checkpoint, device)
    gallery = Gallery()

    gallery_dir = args.gallery_dir
    if not os.path.isdir(gallery_dir):
        raise RuntimeError(f"Gallery dir not found: {gallery_dir}")

    VALID = {".jpg", ".jpeg", ".png", ".bmp"}
    total_imgs = 0

    print(f"\n[Gallery] Scanning {gallery_dir} ...")
    for person_name in sorted(os.listdir(gallery_dir)):
        person_dir = os.path.join(gallery_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        imgs = [f for f in os.listdir(person_dir)
                if os.path.splitext(f)[1].lower() in VALID]
        if not imgs:
            continue

        embeddings = []
        for fname in imgs:
            path = os.path.join(person_dir, fname)
            try:
                bgr  = cv2.imread(path)
                if bgr is None: continue
                t    = preprocess_crop(bgr).unsqueeze(0).to(device)
                emb  = model.extract_with_flip(t).squeeze(0)
                embeddings.append(emb)
                total_imgs += 1
            except Exception as e:
                print(f"  [Warn] Skipped {path}: {e}")

        if embeddings:
            avg_emb = F.normalize(torch.stack(embeddings).mean(0), p=2, dim=0)
            gallery.add(person_name, avg_emb)
            print(f"  Enrolled: {person_name:30s} ({len(embeddings)} images)")

    out = getattr(args, "output", "gallery.pkl")
    gallery.save(out)
    print(f"\n[Gallery] Done — {len(gallery)} persons, {total_imgs} images total")


# ═══════════════════════════════════════════════════════════════════════
# MODE 2: LIVE INFERENCE
# ═══════════════════════════════════════════════════════════════════════

def run_live(args):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = load_model(args.checkpoint, device)
    gallery   = Gallery.load(args.gallery)
    detector  = YOLO(YOLO_MODEL)
    tracker   = SimpleTracker()
    threshold = getattr(args, "threshold", DEFAULT_THRESHOLD)

    # Source: 0 = webcam, integer = camera index, string = file/RTSP
    source = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)         # Disable auto-focus
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure mode
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)           # Disable auto white balance
    fps_history = deque(maxlen=30)
    name_colour = {}   # person name → colour for consistent box colour

    print(f"\n[Live] Starting   source={source}   threshold={threshold}")
    print(f"[Live] Gallery has {len(gallery)} enrolled persons")
    print(f"[Live] Press 'q' to quit, 's' to save screenshot\n")

    frame_idx = 0

    while True:
        t_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            print("[Live] Stream ended.")
            break

        frame_idx += 1

        # ── 1. Detect persons ────────────────────────────────────────
        results = detector(frame, classes=[PERSON_CLASS],
                           verbose=False, conf=0.35)
        boxes   = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box
                # Ignore very small detections (likely noise)
                if (x2 - x1) < 20 or (y2 - y1) < 40:
                    continue
                boxes.append([x1, y1, x2, y2])

        # ── 2. Track ─────────────────────────────────────────────────
        track_ids = tracker.update(boxes)

        # ── 3. Re-ID (every N frames per track) ──────────────────────
        # Create a temporary list to hold everyone in the current frame
        people_in_frame = []

        for i, (box, tid) in enumerate(zip(boxes, track_ids)):
            track = tracker.tracks[tid]
            since_last_reid = frame_idx - track["reid_frame"]

            if since_last_reid >= REID_EVERY_N_FRAMES:
                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                crop = frame[y1:y2, x1:x2]

                if crop.size > 0:
                    t_in  = preprocess_crop(crop).unsqueeze(0).to(device)
                    emb   = model.extract_with_flip(t_in).squeeze(0)

                    track["emb_history"].append(emb)
                    track["reid_frame"] = frame_idx

                    avg_emb = F.normalize(
                        torch.stack(list(track["emb_history"])).mean(0),
                        p=2, dim=0)

                    # Standard matching (get highest score against gallery)
                    identity, score = gallery.match(avg_emb, threshold)
                    track["identity"]       = identity
                    track["identity_score"] = score

            # Add this person's data to our list for sorting
            people_in_frame.append({
                "box": box,
                "identity": track["identity"] or "Unknown",
                "score": track["identity_score"]
            })

        # ── 4. Sort and Draw ONLY Top 5 ──────────────────────────────
        # Sort everyone in the frame by their ReID score (highest first)
        people_in_frame.sort(key=lambda x: x["score"], reverse=True)

        # Slice the list to keep only the top 5
        top_5_people = people_in_frame[:5]

        # Loop through the top 5 and draw them
        for rank, person in enumerate(top_5_people):
            # Rank 0 is the #1 highest probability. Ranks 1-4 are the next four.
            if rank == 0:
                colour = (0, 255, 0)   # Green for Top 1
            else:
                colour = (0, 255, 255) # Yellow for Top 2-5

            # If you ONLY want to draw people who passed the threshold, 
            # you can optionally add: if person["identity"] == "Unknown": continue
            
            draw_box(frame, person["box"], person["identity"], person["score"], colour)

        # ── 5. HUD ───────────────────────────────────────────────────
        # ── 5. HUD ───────────────────────────────────────────────────
        fps_history.append(1.0 / max(time.perf_counter() - t_start, 1e-6))
        draw_fps(frame, sum(fps_history) / len(fps_history))
        draw_gallery_legend(frame, gallery)

        n_tracked = len([t for t in tracker.tracks.values()
                         if t["lost"] == 0])
        cv2.putText(frame, f"Tracked: {n_tracked}",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("ReID Live", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            fname = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[Live] Screenshot saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("[Live] Done.")


# ═══════════════════════════════════════════════════════════════════════
# MODE 3: EVALUATE THRESHOLD
# ═══════════════════════════════════════════════════════════════════════

def evaluate_threshold(args):
    """
    Sweeps similarity thresholds on your val set and prints a table
    of precision, recall, and F1 at each threshold.
    Use this to pick the best threshold for your dataset.
    """
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = load_model(args.checkpoint, device)
    gallery = Gallery.load(args.gallery)

    val_dir = args.val_dir
    VALID   = {".jpg", ".jpeg", ".png"}

    print(f"\n[Threshold Eval] Scanning {val_dir} ...")
    scores_pos = []  # cosine scores for genuine pairs
    scores_neg = []  # cosine scores for impostor pairs

    enrolled_names = set(gallery.embeddings.keys())

    for person_name in sorted(os.listdir(val_dir)):
        person_dir = os.path.join(val_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        imgs = [f for f in os.listdir(person_dir)
                if os.path.splitext(f)[1].lower() in VALID]

        for fname in imgs:
            path = os.path.join(person_dir, fname)
            bgr  = cv2.imread(path)
            if bgr is None: continue

            t   = preprocess_crop(bgr).unsqueeze(0).to(device)
            emb = model.extract_with_flip(t).squeeze(0)

            for name, ref_emb in gallery.embeddings.items():
                score = float(torch.dot(emb.cpu(), ref_emb.cpu()))
                if name == person_name:
                    scores_pos.append(score)
                else:
                    scores_neg.append(score)

    if not scores_pos:
        print("[Threshold Eval] No matching persons found between val set and gallery.")
        return

    print(f"\n  Genuine pairs  : {len(scores_pos)}")
    print(f"  Impostor pairs : {len(scores_neg)}")
    print(f"\n  {'Threshold':>10}  {'Precision':>10}  {'Recall':>10}  {'F1':>8}")
    print(f"  {'-'*46}")

    best_f1, best_thresh = 0., 0.
    for t in np.arange(0.3, 0.85, 0.05):
        tp = sum(1 for s in scores_pos if s >= t)
        fp = sum(1 for s in scores_neg if s >= t)
        fn = sum(1 for s in scores_pos if s <  t)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.
        f1        = 2 * precision * recall / (precision + recall) \
                    if (precision + recall) > 0 else 0.
        marker = "  ←" if f1 > best_f1 else ""
        print(f"  {t:>10.2f}  {precision:>10.3f}  {recall:>10.3f}  {f1:>8.3f}{marker}")
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    print(f"\n  Best threshold : {best_thresh:.2f}  (F1 = {best_f1:.3f})")
    print(f"  Use: --threshold {best_thresh:.2f}\n")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Live ReID Inference")
    p.add_argument("--mode",
                   choices=["build_gallery", "live", "evaluate_threshold"],
                   default="live")
    p.add_argument("--checkpoint", required=True,
                   help="Path to osnet_x0_75_BEST.pth")
    p.add_argument("--gallery",    default="gallery.pkl",
                   help="Path to gallery.pkl (live / evaluate_threshold)")
    p.add_argument("--gallery_dir",
                   help="Folder of known persons for build_gallery mode")
    p.add_argument("--source",     default="0",
                   help="Camera index / RTSP URL / video path (live mode)")
    p.add_argument("--threshold",  type=float, default=DEFAULT_THRESHOLD,
                   help=f"Similarity threshold (default {DEFAULT_THRESHOLD})")
    p.add_argument("--output",     default="gallery.pkl",
                   help="Output path for gallery.pkl (build_gallery mode)")
    p.add_argument("--val_dir",
                   help="Val folder for evaluate_threshold mode")
    return p.parse_args()


def main():
    args = parse_args()
    if   args.mode == "build_gallery":
        build_gallery(args)
    elif args.mode == "live":
        run_live(args)
    elif args.mode == "evaluate_threshold":
        evaluate_threshold(args)


if __name__ == "__main__":
    main()
