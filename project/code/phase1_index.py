"""Phase 1 steps 2-5: parse labels, build the master index, stats, charts and samples.

Label rules (from face_model_phases.md, Phase 1; verified against the files on disk):
  fer2013  folder name is the label; split_hint = Training / PublicTest / PrivateTest
  ferplus  label_ferplus column, from phase1_ferplus.py (FER+ majority vote, pixel join)
  jaffe    filename code AN DI FE HA NE SA SU; subject = first two letters (KA, KL, ...)
  ravdess  speech Video files only (modality 01 or 02, vocal channel 01); calm merged into neutral
  crema_d  VideoFlash/*.flv; emotion ANG DIS FEA HAP NEU SAD; no surprise class

Writes project/data/index.csv (parquet unavailable here, see DECISIONS.md), reports and samples.
"""
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from common import CLASSES, CLASS_TO_IDX, DATASETS_DIR, SEED, project_path, read_state, write_state  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

JAFFE_CODE = {"AN": "angry", "DI": "disgust", "FE": "fear", "HA": "happy", "NE": "neutral", "SA": "sad", "SU": "surprise"}
RAVDESS_CODE = {"01": "neutral", "02": "neutral", "03": "happy", "04": "sad", "05": "angry", "06": "fear", "07": "disgust", "08": "surprise"}
CREMA_CODE = {"ANG": "angry", "DIS": "disgust", "FEA": "fear", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}

INDEX_COLUMNS = ["dataset", "path", "label", "label_id", "subject", "split_hint", "kind", "label_ferplus"]


def rel(p: Path) -> str:
    return p.relative_to(DATASETS_DIR).as_posix()


def row(dataset, path, label, subject, split_hint, kind, label_ferplus=""):
    return {
        "dataset": dataset,
        "path": path,
        "label": label,
        "label_id": CLASS_TO_IDX[label] if label in CLASS_TO_IDX else -1,
        "subject": subject,
        "split_hint": split_hint,
        "kind": kind,
        "label_ferplus": label_ferplus,
    }


def build_fer2013(lines: list[str]) -> pd.DataFrame:
    match = pd.read_csv(project_path("data", "fer2013_match.csv"), keep_default_na=False)
    rows = []
    for _, r in match.iterrows():
        rows.append(row("fer2013", r["path"], r["folder_label"], "", r["usage_from_name"], "image", r["label_ferplus"]))
    lines.append(f"fer2013: {len(rows)} images from fer2013_match.csv")
    return pd.DataFrame(rows)


def build_jaffe(lines: list[str]) -> pd.DataFrame:
    root = DATASETS_DIR / "jaffe" / "extracted" / "jaffe"
    rows, bad = [], []
    for p in sorted(root.glob("*.tiff")):
        parts = p.name.split(".")  # KA . AN1 . 39 . tiff
        code = parts[1][:2] if len(parts) >= 3 else ""
        if code not in JAFFE_CODE or len(parts[0]) != 2:
            bad.append(p.name)
            continue
        rows.append(row("jaffe", rel(p), JAFFE_CODE[code], parts[0], "none", "image"))
    lines.append(f"jaffe: {len(rows)} images parsed; unparsed names: {bad or 'none'}")
    return pd.DataFrame(rows)


def build_ravdess(lines: list[str]) -> pd.DataFrame:
    root = DATASETS_DIR / "RAVDESS" / "Video_Speech"
    rows, skipped = [], Counter()
    for p in sorted(root.rglob("*.mp4")):
        f = p.stem.split("-")
        if len(f) != 7:
            skipped["bad name"] += 1
            continue
        modality, vocal, emotion, _intensity, _statement, _repetition, actor = f
        if modality not in ("01", "02"):  # 01 audio-video, 02 video-only; 03 audio-only excluded
            skipped[f"modality {modality}"] += 1
            continue
        if vocal != "01":  # 01 speech, 02 song
            skipped[f"vocal {vocal}"] += 1
            continue
        rows.append(row("ravdess", rel(p), RAVDESS_CODE[emotion], actor, "none", "video"))
    lines.append(f"ravdess: {len(rows)} video files parsed; skipped: {dict(skipped) or 'none'}")
    return pd.DataFrame(rows)


def build_crema(lines: list[str]) -> pd.DataFrame:
    root = DATASETS_DIR / "crema_d" / "VideoFlash"
    rows, bad = [], []
    for p in sorted(root.glob("*.flv")):
        f = p.stem.split("_")  # ActorID _ Sentence _ Emotion _ Level
        if len(f) != 4 or f[2] not in CREMA_CODE:
            bad.append(p.name)
            continue
        rows.append(row("crema_d", rel(p), CREMA_CODE[f[2]], f[0], "none", "video"))
    lines.append(f"crema_d: {len(rows)} video files parsed; unparsed names: {bad or 'none'}")
    return pd.DataFrame(rows)


def check_readable(index: pd.DataFrame) -> pd.DataFrame:
    """Flag (never remove) files that cannot be opened."""
    bad = []
    for _, r in index.iterrows():
        p = DATASETS_DIR / r["path"]
        try:
            if r["kind"] == "image":
                with Image.open(p) as im:
                    im.load()
            else:
                cap = cv2.VideoCapture(str(p))
                ok = cap.isOpened() and cap.read()[0]
                cap.release()
                if not ok:
                    raise RuntimeError("cv2 could not read a frame")
        except Exception as exc:
            bad.append({"dataset": r["dataset"], "path": r["path"], "error": str(exc)[:200]})
    return pd.DataFrame(bad, columns=["dataset", "path", "error"])


def stats_rows(index: pd.DataFrame) -> pd.DataFrame:
    out = []
    groups = [(ds, sub) for ds, sub in index.groupby("dataset")]
    for ds, sub in groups:
        for split, s2 in sub.groupby("split_hint"):
            counts = s2["label"].value_counts().reindex(CLASSES, fill_value=0)
            total = counts.sum()
            for label in CLASSES:
                out.append({"dataset": ds, "split_hint": split, "label": label, "count": int(counts[label]),
                            "share": round(counts[label] / total, 4) if total else 0.0})
        # FER+ relabelled view of fer2013 (same images, different label column).
        if ds == "fer2013":
            fp = sub[sub["label_ferplus"] != ""]
            counts = fp["label_ferplus"].value_counts().reindex(CLASSES, fill_value=0)
            total = counts.sum()
            for label in CLASSES:
                out.append({"dataset": "fer2013_ferplus", "split_hint": "all", "label": label,
                            "count": int(counts[label]), "share": round(counts[label] / total, 4)})
    return pd.DataFrame(out)


def chart(stats: pd.DataFrame, dataset: str, split: str, title: str, path: Path) -> str:
    s = stats[(stats["dataset"] == dataset) & (stats["split_hint"] == split)]
    counts = s.set_index("label")["count"].reindex(CLASSES)
    colors = ["#c0392b" if c == 0 else "#2e6da4" for c in counts.values]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(CLASSES, counts.values, color=colors)
    for b, c in zip(bars, counts.values):
        ax.annotate(f"{int(c):,}", (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom", fontsize=8)
    nonzero = counts[counts > 0]
    ratio = nonzero.max() / nonzero.min() if len(nonzero) else float("nan")
    missing = [c for c in CLASSES if counts[c] == 0]
    ax.set_title(f"{title}\nimbalance max/min = {ratio:.1f}" + (f"; missing: {', '.join(missing)}" if missing else ""), fontsize=10)
    ax.set_ylabel("files")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return f"imbalance {ratio:.2f}; missing={missing or 'none'}"


def write_samples(index: pd.DataFrame, lines: list[str]) -> None:
    rng = random.Random(SEED)
    out_root = project_path("reports", "samples")
    for ds in ("fer2013", "jaffe"):
        sub = index[index["dataset"] == ds]
        ds_dir = out_root / ds
        tiles = []
        for label in CLASSES:
            pool = sub[sub["label"] == label]["path"].tolist()
            picks = rng.sample(pool, min(8, len(pool)))
            label_dir = ds_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
            row_tiles = []
            for pth in picks:
                with Image.open(DATASETS_DIR / pth) as im:
                    im = im.convert("L")
                    scale = 3 if ds == "fer2013" else 1
                    im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
                    stem = Path(pth).stem
                    im.save(label_dir / f"{stem}.png")
                    row_tiles.append(np.asarray(im.resize((96, 96)), dtype=np.uint8))
            while len(row_tiles) < 8:
                row_tiles.append(np.zeros((96, 96), dtype=np.uint8))
            tiles.append(np.hstack(row_tiles))
        sheet = np.vstack(tiles)
        Image.fromarray(sheet).save(out_root / f"contact_{ds}.png")
        lines.append(f"samples: {ds} 8 per class -> {ds_dir.relative_to(project_path())}; contact sheet contact_{ds}.png")


def main() -> None:
    lines: list[str] = ["# Phase 1 index build", ""]
    parts = [build_fer2013(lines), build_jaffe(lines), build_ravdess(lines), build_crema(lines)]
    index = pd.concat(parts, ignore_index=True)
    index = index[INDEX_COLUMNS]

    index_path = project_path("data", "index.csv")
    index.to_csv(index_path, index=False)
    lines.append(f"index rows: {len(index)} -> {index_path.name} (csv; pyarrow not installed)")
    lines.append(index.groupby(["dataset", "kind"]).size().to_string())
    lines.append("")

    unreadable = check_readable(index)
    unreadable_path = project_path("reports", "unreadable_files.csv")
    unreadable.to_csv(unreadable_path, index=False)
    lines.append(f"unreadable files (flagged, not removed): {len(unreadable)} -> {unreadable_path.name}")
    if len(unreadable):
        lines.append(unreadable.groupby("dataset").size().to_string())
    lines.append("")

    stats = stats_rows(index)
    stats_path = project_path("reports", "dataset_stats.csv")
    stats.to_csv(stats_path, index=False)
    lines.append(f"stats -> {stats_path.name}")

    charts = {}
    charts["fer2013"] = chart(stats, "fer2013", "Training", "FER-2013 train (folder labels)", project_path("reports", "class_counts_fer2013_train.png"))
    charts["fer2013 test"] = chart(stats, "fer2013", "PublicTest", "FER-2013 PublicTest (folder labels)", project_path("reports", "class_counts_fer2013_publictest.png"))
    charts["fer2013 private"] = chart(stats, "fer2013", "PrivateTest", "FER-2013 PrivateTest (folder labels)", project_path("reports", "class_counts_fer2013_privatetest.png"))
    charts["fer2013 ferplus"] = chart(stats, "fer2013_ferplus", "all", "FER-2013 images, FER+ majority labels", project_path("reports", "class_counts_fer2013_ferplus.png"))
    charts["jaffe"] = chart(stats, "jaffe", "none", "JAFFE", project_path("reports", "class_counts_jaffe.png"))
    charts["ravdess"] = chart(stats, "ravdess", "none", "RAVDESS speech video", project_path("reports", "class_counts_ravdess.png"))
    charts["crema_d"] = chart(stats, "crema_d", "none", "CREMA-D video", project_path("reports", "class_counts_crema_d.png"))
    for k, v in charts.items():
        lines.append(f"chart {k}: {v}")
    lines.append("")

    write_samples(index, lines)

    state = read_state()
    state["phase_done"] = 1
    write_state(state)
    lines.append("state.json: phase_done = 1")

    log = project_path("logs", "phase1_index_build.txt")
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
