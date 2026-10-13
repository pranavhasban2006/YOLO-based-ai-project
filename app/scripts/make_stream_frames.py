"""Extract N consecutive frames from a RAVDESS clip as 320-px-wide JPEGs (quality 0.7, like the browser).

Usage (project venv):  python -I make_stream_frames.py <video.mp4> <out_dir> [--n 100]
"""
import argparse
from pathlib import Path

import cv2

ap = argparse.ArgumentParser()
ap.add_argument("video", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--n", type=int, default=100)
args = ap.parse_args()

args.out.mkdir(parents=True, exist_ok=True)
for old in args.out.glob("*.jpg"):
    old.unlink()
cap = cv2.VideoCapture(str(args.video))
src_fps = cap.get(cv2.CAP_PROP_FPS)
written = 0
while written < args.n:
    ok, bgr = cap.read()
    if not ok:
        break
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (320, round(320 * h / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(args.out / f"{written:04d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    written += 1
cap.release()
print(f"wrote {written} frames from {args.video.name} (source {src_fps:.1f} fps) to {args.out}")
