"""Runtime configuration. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT_DIR / "models"))

# Class index order used by both checkpoints. Alphabetical; do not change.
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Which face checkpoint is active: "resnet18" (default) or "cnn".
# If the requested file is missing, the other one is used as a fallback.
ACTIVE_FACE_MODEL = os.getenv("ACTIVE_FACE_MODEL", "resnet18")
FACE_CHECKPOINTS = {
    "resnet18": MODELS_DIR / "resnet18_fer.pt",
    "cnn": MODELS_DIR / "cnn_baseline.pt",
}
FACE_DETECTOR_PATH = MODELS_DIR / "blaze_face_short_range.tflite"
FACE_MARGIN = 0.20
FACE_MIN_CONFIDENCE = float(os.getenv("FACE_MIN_CONFIDENCE", "0.5"))

DB_PATH = Path(os.getenv("DB_PATH", ROOT_DIR / "backend" / "data" / "sessions.db"))

MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_MB", "10")) * 1024 * 1024
MAX_VIDEO_BYTES = int(os.getenv("MAX_VIDEO_MB", "100")) * 1024 * 1024
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_MB", "25")) * 1024 * 1024
MAX_WS_FRAME_BYTES = 2 * 1024 * 1024

IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_TYPES = {"video/mp4", "video/webm", "video/quicktime", "video/x-msvideo", "video/avi"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".avi"}

VIDEO_SAMPLE_FPS = 5.0
VIDEO_SMOOTH_WINDOW = 5
VIDEO_MAX_SAMPLES = 1500  # 5 minutes at 5 fps; protects the CPU from very long clips

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")
