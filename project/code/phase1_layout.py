"""Phase 1 step 1: discover the real on-disk layout of each dataset and write project/logs/layout_<dataset>.txt.

Read-only. Lists folders, extension counts and file-name patterns; nothing is inferred
from the manifest alone.
"""
import re
import sys
from collections import Counter
from pathlib import Path

from common import DATASETS_DIR, project_path

sys.stdout.reconfigure(encoding="utf-8")

# Roots actually used by the pipeline (see DECISIONS.md, Phase 0 and Phase 1).
ROOTS = {
    "fer2013": DATASETS_DIR / "FER-2013" / "extracted",
    "ferplus": DATASETS_DIR / "ferplus",
    "jaffe": DATASETS_DIR / "jaffe" / "extracted",
    "ravdess": DATASETS_DIR / "RAVDESS",
    "crema_d": DATASETS_DIR / "crema_d",
}

# Things that exist next to the used roots and are deliberately not indexed.
NOTES = {
    "fer2013": [
        "Used: datasets/FER-2013/extracted (complete copy, 28,709 train + 7,178 test).",
        "Ignored: datasets/fer2013/ (partial, train 3,028 of 28,709). Not indexed.",
        "Test folder merges PublicTest_ and PrivateTest_ files; the prefix gives the split.",
        "Image names are Training_<id>.jpg, PublicTest_<id>.jpg, PrivateTest_<id>.jpg. The <id> is not a row index.",
        "Pixel source for the FER+ join: datasets image video/challenges-...zip -> fer2013.tar.gz -> fer2013.csv",
    ],
    "ferplus": [
        "Used: fer2013new.csv (35,887 rows, same row order as the original fer2013.csv).",
        "data/FER2013{Train,Valid,Test}/label.csv are the FER+ label files from the repo; they hold no pixels.",
        "No images in this folder. FER+ is a relabelling of the FER-2013 images, not a separate image set.",
    ],
    "jaffe": [
        "Used: datasets/jaffe/extracted/jaffe/*.tiff (213 images). README files are not indexed.",
        "Duplicate zip jaffe/jaffe.zip is not indexed.",
    ],
    "ravdess": [
        "Used: datasets/RAVDESS/Video_Speech (2,880 mp4, 1,440 modality 01 + 1,440 modality 02). Speech only.",
        "Excluded: RAVDESS/Video_Song (song, not speech; Phase 0 decision). Its .download.lock is ignored.",
        "Excluded: RAVDESS/Audio_Speech (audio only, no video). Phase 1 keeps video rows only.",
        "Ignored duplicate: datasets/ravdess/ (same counts as RAVDESS/; lowercase copy, not indexed).",
    ],
    "crema_d": [
        "Used: datasets/crema_d/VideoFlash/*.flv (7,442 clips). Video only.",
        "AudioWAV and AudioMP3 are audio only and are not indexed.",
        "Git metadata under crema_d/.git is ignored.",
    ],
}

VIDEO_OR_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".mp4", ".flv", ".avi", ".wav", ".mp3", ".csv"}


def generalise(name: str) -> str:
    """Turn a file name into a pattern: digits become N, letter runs stay."""
    stem = re.sub(r"\d+", "N", name)
    return stem


def describe(dataset: str, root: Path) -> str:
    lines = [f"# Layout: {dataset}", f"root: {root}", ""]
    if not root.exists():
        lines.append("MISSING: root does not exist")
        return "\n".join(lines)

    all_files = [p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]
    ext_counts = Counter(p.suffix.lower() for p in all_files)
    lines.append(f"total files (excluding .git): {len(all_files)}")
    lines.append("extensions: " + ", ".join(f"{e or '(none)'}={n}" for e, n in ext_counts.most_common()))
    lines.append("")

    lines.append("top-level entries:")
    for entry in sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        if entry.name == ".git":
            continue
        if entry.is_dir():
            n = sum(1 for p in entry.rglob("*") if p.is_file())
            lines.append(f"  [dir ] {entry.name}/  ({n} files below)")
        else:
            lines.append(f"  [file] {entry.name}  ({entry.stat().st_size} bytes)")
    lines.append("")

    # Immediate subfolders one level down, with counts and two example names.
    lines.append("subfolders (depth 2) with file counts and example names:")
    for sub in sorted(p for p in root.iterdir() if p.is_dir() and p.name != ".git"):
        for child in sorted(p for p in sub.iterdir() if p.is_dir() and p.name != ".git")[:40]:
            files = [p for p in child.rglob("*") if p.is_file()]
            if not files:
                continue
            examples = ", ".join(sorted(p.name for p in files)[:2])
            lines.append(f"  {sub.name}/{child.name}/  files={len(files)}  e.g. {examples}")
    lines.append("")

    # File-name patterns (only for media / data files), counted after digit generalisation.
    media = [p for p in all_files if p.suffix.lower() in VIDEO_OR_IMAGE_EXT]
    if media:
        patterns = Counter(generalise(p.name) for p in media)
        lines.append(f"file-name patterns over {len(media)} media/data files (top 12, digits -> N):")
        for pat, n in patterns.most_common(12):
            lines.append(f"  {n:6d}  {pat}")
    lines.append("")

    lines.append("notes:")
    for note in NOTES.get(dataset, []):
        lines.append(f"  - {note}")
    return "\n".join(lines) + "\n"


def main() -> None:
    out_dir = project_path("logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset, root in ROOTS.items():
        text = describe(dataset, root)
        target = out_dir / f"layout_{dataset}.txt"
        target.write_text(text, encoding="utf-8")
        print(f"wrote {target.name}  ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
