"""Phase 2: sample video frames, crop faces, standardise images to 224x224 RGB JPEG.

Rules (see project/DECISIONS.md, Phase 2):
  ravdess, crema_d  5 frames per clip at 20%..80% of clip length; MediaPipe face detection
                    (Haar fallback); largest face; square crop with 20% margin; 224x224 JPEG q95
  jaffe             one still per image; same detection and crop as video frames (frame_id 0)
  fer2013           no detection; the original 48x48 grayscale path is stored; 3-channel and
                    224 resize happen at load time in the training loader

Writes project/data/faces/<dataset>/<subject>/<stem>_f<k>.jpg, project/data/faces_index.parquet,
reports/contact_<dataset>.png and logs/phase_2_report.md. Resumable: an existing valid output
image is reused and not re-decoded.

face_path is stored relative to the repo root (forward slashes), like the index path is relative to datasets/.
"""
import os

# One thread per worker. Without this, each spawned process runs OpenCV and TFLite at full
# thread count and the pool slows to a crawl (observed: ~70 s per clip in the pool vs ~0.5 s alone).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import multiprocessing as mp
import random
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mpi
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw
from tqdm import tqdm

from common import CLASSES, DATASETS_DIR, REPO_ROOT, SEED, project_path, read_state, write_state

sys.stdout.reconfigure(encoding="utf-8")

OUT_SIZE = 224
MARGIN = 0.2
N_FRAMES = 5
SAMPLE_LO, SAMPLE_HI = 0.2, 0.8
JPEG_QUALITY = 95
VIDEO_DATASETS = {"ravdess", "crema_d"}
IMAGE_DETECT_DATASETS = {"jaffe"}
PASSTHROUGH_DATASETS = {"fer2013"}
WORKERS = min(4, os.cpu_count() or 1)
MEDIAPIPE_MODEL = REPO_ROOT / "models" / "blaze_face_short_range.tflite"
FACES_DIR = project_path("data", "faces")
INDEX_IN = project_path("data", "index.csv")
INDEX_OUT = project_path("data", "faces_index.parquet")
REPORTS = project_path("reports")
LOG_OUT = project_path("logs", "phase_2_report.md")

# Per-process state, created once by the pool initializer (MediaPipe objects are not picklable).
_detector = None
_haar = None


def _init_worker() -> None:
    global _detector, _haar
    cv2.setNumThreads(1)
    options = vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MEDIAPIPE_MODEL)),
        min_detection_confidence=0.5,
    )
    _detector = vision.FaceDetector.create_from_options(options)
    _haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def _detect_largest(rgb: np.ndarray):
    """Return (x, y, w, h, method) of the largest face, or None. method is 'mediapipe' or 'haar'."""
    img = mpi.Image(image_format=mpi.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    res = _detector.detect(img)
    boxes = [(d.bounding_box.origin_x, d.bounding_box.origin_y, d.bounding_box.width, d.bounding_box.height)
             for d in res.detections]
    if boxes:
        x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
        return int(x), int(y), int(w), int(h), "mediapipe"

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = _haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces):
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        return int(x), int(y), int(w), int(h), "haar"
    return None


def _crop_face(rgb: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Square crop centred on the box; each side extends MARGIN * largest side beyond the box.
    Out-of-frame area is zero-padded. Resized to OUT_SIZE x OUT_SIZE."""
    side = int(round(max(w, h) * (1 + 2 * MARGIN)))
    cx, cy = x + w / 2, y + h / 2
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    H, W = rgb.shape[:2]
    pad = max(0, -x0, -y0, x0 + side - W, y0 + side - H)
    if pad:
        rgb = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    crop = rgb[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]
    return cv2.resize(crop, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)


def _save_jpeg(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.jpg")
    Image.fromarray(arr).save(tmp, format="JPEG", quality=JPEG_QUALITY)
    os.replace(tmp, path)


def _valid_image(path: Path) -> bool:
    """True when the file opens as an image and is OUT_SIZE x OUT_SIZE (a half-written file fails)."""
    if not path.is_file():
        return False
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            return im.size == (OUT_SIZE, OUT_SIZE)
    except Exception:
        return False


def _frame_targets(n: int) -> list[int]:
    """Five evenly spaced frame indices between 20% and 80% of the clip."""
    fracs = np.linspace(SAMPLE_LO, SAMPLE_HI, N_FRAMES)
    return [int(round(f * (n - 1))) for f in fracs]


def _frame_count(path: str) -> int:
    """Frame count from metadata. Some FLV files report none, so count with a cheap grab() pass."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        n = 0
        while cap.grab():
            n += 1
    cap.release()
    return n


def _read_frames(path: str, targets: list[int]) -> dict:
    """Decode the wanted frame indices in one sequential pass. Returns {frame_index: BGR frame}."""
    cap = cv2.VideoCapture(path)
    frames = {}
    want = sorted(set(targets))
    ti, i = 0, 0
    while ti < len(want):
        if not cap.grab():
            break
        if i == want[ti]:
            ok, frame = cap.retrieve()
            if ok:
                frames[want[ti]] = frame
            ti += 1
        i += 1
    cap.release()
    return frames


def _process_task(task: dict) -> dict:
    """Video clip or jaffe still. Returns saved (frame_id, path) pairs and counters; never raises."""
    out = {"key": task["key"], "saved": [], "attempted": 0, "no_face": 0, "haar": 0, "error": None}
    try:
        stem = Path(task["path"]).stem
        src = str(DATASETS_DIR / task["path"])
        subject_dir = FACES_DIR / task["dataset"] / task["subject"]

        if task["kind"] == "image":
            out["attempted"] = 1
            dst = subject_dir / f"{stem}_f0.jpg"
            if _valid_image(dst):
                out["saved"].append((0, dst))
                return out
            rgb = np.array(Image.open(src).convert("RGB"))
            hit = _detect_largest(rgb)
            if hit is None:
                out["no_face"] = 1
                return out
            x, y, w, h, method = hit
            out["haar"] += method == "haar"
            _save_jpeg(_crop_face(rgb, x, y, w, h), dst)
            out["saved"].append((0, dst))
            return out

        # Video: decide which of the 5 frames still need work, so finished clips are not decoded.
        out["attempted"] = N_FRAMES
        n = _frame_count(src)
        if n <= 0:
            out["no_face"] = N_FRAMES
            return out
        targets = _frame_targets(n)
        todo = []
        for k, t in enumerate(targets, start=1):
            dst = subject_dir / f"{stem}_f{k}.jpg"
            if _valid_image(dst):
                out["saved"].append((t, dst))
            else:
                todo.append((t, dst))
        if not todo:
            return out

        frames = _read_frames(src, [t for t, _ in todo])
        for t, dst in todo:
            bgr = frames.get(t)
            if bgr is None:
                out["no_face"] += 1
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            hit = _detect_largest(rgb)
            if hit is None:
                out["no_face"] += 1
                continue
            x, y, w, h, method = hit
            out["haar"] += method == "haar"
            _save_jpeg(_crop_face(rgb, x, y, w, h), dst)
            out["saved"].append((t, dst))
    except Exception as e:  # keep the pool alive; the error is reported and the unfinished frames are counted as skipped
        out["error"] = f"{type(e).__name__}: {e}"
        out["no_face"] = out["attempted"] - len(out["saved"])
    return out


def _rel(p: Path) -> str:
    return Path(os.path.relpath(p, REPO_ROOT)).as_posix()


def build_tasks(index: pd.DataFrame) -> list[dict]:
    tasks = []
    for i, row in index.iterrows():
        ds = row["dataset"]
        if ds in VIDEO_DATASETS:
            kind = "video"
        elif ds in IMAGE_DETECT_DATASETS:
            kind = "image"
        else:
            continue
        tasks.append({"key": int(i), "kind": kind, "dataset": ds, "path": row["path"],
                      "subject": "" if pd.isna(row["subject"]) else str(row["subject"])})
    return tasks


def run_extraction(index: pd.DataFrame) -> tuple[list[dict], dict]:
    """Returns (output rows, per-dataset counters)."""
    tasks = build_tasks(index)
    rows_out: list[dict] = []
    stats: dict[str, Counter] = {}
    index_cols = list(index.columns)

    def base_row(key: int) -> dict:
        return {c: index.at[key, c] for c in index_cols}

    # fer2013: no detection. Keep the original path, one row per image.
    fer = index[index["dataset"].isin(PASSTHROUGH_DATASETS)]
    c = stats.setdefault("fer2013", Counter())
    for i, row in tqdm(fer.iterrows(), total=len(fer), desc="fer2013 passthrough"):
        c["attempted"] += 1
        if not (DATASETS_DIR / row["path"]).is_file():
            c["no_face"] += 1  # original file missing; reported as skipped
            continue
        r = base_row(i)
        r["face_path"] = f"datasets/{row['path']}"
        r["frame_id"] = 0
        rows_out.append(r)
        c["saved"] += 1

    ctx = mp.get_context("spawn")  # spawn-safe on Windows; workers import this module fresh
    with ctx.Pool(WORKERS, initializer=_init_worker) as pool:
        for res in tqdm(pool.imap_unordered(_process_task, tasks, chunksize=8),
                        total=len(tasks), desc="faces (video+jaffe)"):
            key = res["key"]
            c = stats.setdefault(index.at[key, "dataset"], Counter())
            c["attempted"] += res["attempted"]
            c["no_face"] += res["no_face"]
            c["haar"] += res["haar"]
            c["saved"] += len(res["saved"])
            if res["error"]:
                c["errors"] += 1
                tqdm.write(f"ERROR {index.at[key, 'path']}: {res['error']}")
            for frame_id, dst in res["saved"]:
                r = base_row(key)
                r["face_path"] = _rel(dst)
                r["frame_id"] = int(frame_id)
                rows_out.append(r)
    return rows_out, stats


def contact_sheet(df: pd.DataFrame, dataset: str, rng: random.Random) -> Path:
    sub = df[df["dataset"] == dataset]
    pick = sub.sample(n=min(40, len(sub)), random_state=rng.randrange(1 << 31))
    cols, cell, cap_h = 8, 160, 18
    rows = (len(pick) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + cap_h)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for n, (_, r) in enumerate(pick.iterrows()):
        im = Image.open(REPO_ROOT / r["face_path"]).convert("RGB").resize((cell, cell), Image.BILINEAR)
        x, y = (n % cols) * cell, (n // cols) * (cell + cap_h)
        sheet.paste(im, (x, y))
        draw.text((x + 4, y + cell + 2), str(r["label"]), fill=(230, 230, 230))
    out = REPORTS / f"contact_{dataset}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def write_report(faces: pd.DataFrame, stats: dict, sheets: list[Path], elapsed: float) -> None:
    lines = ["# Phase 2 report", "", f"Generated: {pd.Timestamp.now().isoformat(timespec='seconds')}", "",
             f"Output index: `project/data/faces_index.parquet` ({len(faces)} rows)", "",
             "## Extraction per dataset", "",
             "| dataset | attempted | saved | skipped (no face) | haar fallback used | errors |",
             "|---|---:|---:|---:|---:|---:|"]
    for ds in sorted(stats):
        c = stats[ds]
        lines.append(f"| {ds} | {c['attempted']} | {c['saved']} | {c['no_face']} | {c['haar']} | {c['errors']} |")
    lines += ["", "fer2013 is passthrough (no detection): 'attempted' counts images, 'saved' counts index rows.", ""]
    lines += ["## Counts per dataset and class (rows in faces_index)", ""]
    ct = faces.groupby(["dataset", "label"]).size().unstack(fill_value=0)
    ct = ct.reindex(columns=[c for c in CLASSES if c in ct.columns] + [c for c in ct.columns if c not in CLASSES])
    lines += ["```", ct.to_string(), "```"]
    lines += ["", "## Contact sheets", ""] + [f"- `{_rel(s)}`" for s in sheets]
    lines += ["", f"Elapsed: {elapsed / 60:.1f} min", ""]
    LOG_OUT.parent.mkdir(parents=True, exist_ok=True)
    LOG_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    t0 = time.time()
    # keep_default_na=False: JAFFE subject "NA" must stay a string, not become a missing value.
    index = pd.read_csv(INDEX_IN, dtype={"subject": str, "label_ferplus": str}, keep_default_na=False)
    print(f"index rows: {len(index)}  workers: {WORKERS}")
    rows, stats = run_extraction(index)
    faces = pd.DataFrame(rows)
    faces["frame_id"] = faces["frame_id"].astype("int64")
    faces.to_parquet(INDEX_OUT, index=False)
    print(f"faces_index: {len(faces)} rows -> {INDEX_OUT.name}")

    for ds in sorted(stats):
        c = stats[ds]
        print(f"{ds}: attempted={c['attempted']} saved={c['saved']} skipped_no_face={c['no_face']} "
              f"haar={c['haar']} errors={c['errors']}")

    rng = random.Random(SEED)
    sheets = [contact_sheet(faces, ds, rng) for ds in sorted(faces["dataset"].unique())]
    write_report(faces, stats, sheets, time.time() - t0)

    state = read_state()
    state["phase_done"] = 2
    write_state(state)
    print(f"phase_done = 2  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
