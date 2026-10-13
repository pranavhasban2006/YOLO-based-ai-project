"""Facial emotion inference for an image or a video.

Usage (project venv):
    python -I project/code/infer.py <image_or_video_path> [--model PATH] [--device cpu|cuda]

Prints one JSON object with the 7 class probabilities to stdout.

Pipeline (matches the training pipeline in project/code/build_notebook.py and phase2_faces.py):
  1. Image: one frame. Video: 5 frames at 20%..80% of the clip (evenly spaced).
  2. Face: MediaPipe BlazeFace (largest box); OpenCV Haar fallback. The box is cropped
     to a square with a 20% margin and resized to 224x224 RGB.
  3. Model input: grayscale, resized to 48x48, normalised with mean 0.5 and std 0.5.
  4. Probabilities are averaged over the frames where a face was found.
     If no face is found in any frame, the whole frame is used instead and the JSON
     says face_found_frames = 0.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mpi
import numpy as np
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image
from torch import nn

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
DEFAULT_MODEL = PROJECT_DIR / "models" / "face_emotion_best.pt"
CLASSES_FILE = PROJECT_DIR / "models" / "classes.json"
MEDIAPIPE_MODEL = REPO_ROOT / "models" / "blaze_face_short_range.tflite"

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CROP_SIZE = 224
MARGIN = 0.2
N_FRAMES = 5
SAMPLE_LO, SAMPLE_HI = 0.2, 0.8
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm"}


class SmallCNN(nn.Module):
    """4 blocks of (Conv3x3-BN-ReLU x2, MaxPool), 64/128/256/512 channels, GAP, dropout 0.4, linear to 7."""

    def __init__(self, n_classes: int = 7):
        super().__init__()
        layers, c_in = [], 1
        for c in [64, 128, 256, 512]:
            layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                       nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(512, n_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


class FaceFinder:
    """Largest face in an RGB frame: MediaPipe first, OpenCV Haar when MediaPipe finds nothing."""

    def __init__(self):
        self.detector = None
        if MEDIAPIPE_MODEL.is_file():
            options = vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(MEDIAPIPE_MODEL)),
                min_detection_confidence=0.5,
            )
            self.detector = vision.FaceDetector.create_from_options(options)
        self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def find(self, rgb: np.ndarray):
        if self.detector is not None:
            img = mpi.Image(image_format=mpi.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            boxes = [(d.bounding_box.origin_x, d.bounding_box.origin_y, d.bounding_box.width, d.bounding_box.height)
                     for d in self.detector.detect(img).detections]
            if boxes:
                return max(boxes, key=lambda b: b[2] * b[3])
        if not self.haar.empty():
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            faces = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces):
                return tuple(int(v) for v in max(faces, key=lambda b: b[2] * b[3]))
        return None


def crop_face(rgb: np.ndarray, box) -> np.ndarray:
    """Square crop centred on the box, each side extended by MARGIN * largest side. Zero-padded at edges."""
    x, y, w, h = box
    side = int(round(max(w, h) * (1 + 2 * MARGIN)))
    x0 = int(round(x + w / 2 - side / 2))
    y0 = int(round(y + h / 2 - side / 2))
    H, W = rgb.shape[:2]
    pad = max(0, -x0, -y0, x0 + side - W, y0 + side - H)
    if pad:
        rgb = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    crop = rgb[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]
    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)


def to_model_tensor(rgb_224: np.ndarray) -> torch.Tensor:
    """Grayscale 48x48 in [-1, 1], shape (1, 1, 48, 48). Same steps as the eval transform in training."""
    gray = Image.fromarray(rgb_224).convert("L").resize((48, 48), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(gray, dtype=np.float32) / 255.0)
    return ((x - 0.5) / 0.5).unsqueeze(0).unsqueeze(0)


def load_model(path: Path, device: torch.device) -> SmallCNN:
    model = SmallCNN(n_classes=len(CLASSES))
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def frame_indices(n_frames: int) -> list[int]:
    fracs = np.linspace(SAMPLE_LO, SAMPLE_HI, N_FRAMES)
    return [int(round(f * (n_frames - 1))) for f in fracs]


def read_video_frames(path: Path) -> tuple[list[np.ndarray], int]:
    """Return the sampled RGB frames and the number of frames in the clip."""
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:  # some containers report no count, so count with a grab() pass
        n = 0
        while cap.grab():
            n += 1
        cap.release()
        cap = cv2.VideoCapture(str(path))
    if n <= 0:
        raise SystemExit(f"could not read any frames from {path}")
    frames = []
    for idx in frame_indices(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if ok:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise SystemExit(f"could not decode any sampled frame from {path}")
    return frames, n


def predict_frames(frames: list[np.ndarray], model: SmallCNN, device: torch.device, finder: FaceFinder):
    """Average the softmax probabilities over the frames where a face was found.
    If no frame has a face, average over the whole frames."""
    with_face, whole = [], []
    for rgb in frames:
        box = finder.find(rgb)
        if box is None:
            whole.append(rgb)
        else:
            with_face.append(crop_face(rgb, box))
    crops = with_face if with_face else [np.asarray(Image.fromarray(f).resize((CROP_SIZE, CROP_SIZE))) for f in whole]
    if not crops:
        raise SystemExit("no frames to predict on")
    x = torch.cat([to_model_tensor(c) for c in crops]).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).mean(dim=0).cpu().numpy()
    return probs, len(with_face)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="image or video file")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    path = args.path.resolve()
    if not path.is_file():
        raise SystemExit(f"not a file: {path}")
    ext = path.suffix.lower()
    if ext not in IMAGE_EXTS | VIDEO_EXTS:
        raise SystemExit(f"unsupported file type: {ext}")

    classes = json.loads(CLASSES_FILE.read_text(encoding="utf-8")) if CLASSES_FILE.is_file() else CLASSES
    assert classes == CLASSES, "classes.json does not match the model's class order"
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)
    finder = FaceFinder()

    if ext in VIDEO_EXTS:
        frames, n_total = read_video_frames(path)
        kind = "video"
    else:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"could not read image: {path}")
        frames, n_total = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)], 1
        kind = "image"

    probs, n_face = predict_frames(frames, model, device, finder)
    result = {
        "input": str(path),
        "kind": kind,
        "frames_sampled": len(frames),
        "frames_in_clip": n_total,
        "face_found_frames": n_face,
        "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
        "top_label": CLASSES[int(np.argmax(probs))],
    }
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
