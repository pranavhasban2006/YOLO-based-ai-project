"""Phase 1 step 2 (FER+ part): relabel FER-2013 with FER+ majority votes.

FER+ ships votes per row of the original fer2013.csv, not per image file. The image files
in datasets/FER-2013/extracted have Kaggle ids in their names, not row numbers, so the
join goes through pixels:

  1. Read the original fer2013.csv (from the challenge zip's fer2013.tar.gz). Its rows are
     in the same order as FER+ fer2013new.csv; this is verified below, not assumed.
  2. Decode every FER-2013 image to 48x48 greyscale.
  3. For each split (Training / PublicTest / PrivateTest), match each image to the nearest
     original row by L2 distance on pixels, restricted to rows of the same Usage.
  4. Check every match: the original emotion code must agree with the folder label.
     A match that disagrees, or is ambiguous, is flagged, not dropped.

Outputs
  project/data/fer2013_match.csv     one row per FER-2013 image: path, orig_row, distances, flags
  project/reports/ferplus_relabel.csv one row per original row: votes, majority, label_ferplus, drop reason
  project/logs/ferplus_join.txt      checks and counts
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from common import DATASETS_DIR, PROJECT_DIR, project_path

sys.stdout.reconfigure(encoding="utf-8")

ORIG_CSV = PROJECT_DIR / "data" / "_src_fer2013_original" / "fer2013" / "fer2013.csv"
FERPLUS_CSV = DATASETS_DIR / "ferplus" / "fer2013new.csv"
FER_ROOT = DATASETS_DIR / "FER-2013" / "extracted"

# Original fer2013.csv emotion codes (Goodfellow et al., 2013).
ORIG_EMOTION = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}

# FER+ vote columns, in the order used by fer2013new.csv.
VOTE_COLS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt", "unknown", "NF"]
# Vote column -> target class. Only the 7 target classes map; everything else is dropped.
VOTE_TO_CLASS = {
    "neutral": "neutral",
    "happiness": "happy",
    "surprise": "surprise",
    "sadness": "sad",
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
}
DROP_IF_WINNER = {"unknown", "NF", "contempt"}

# Usage value in fer2013.csv and the file-name prefix on disk.
USAGE_FROM_PREFIX = {"Training": "Training", "PublicTest": "PublicTest", "PrivateTest": "PrivateTest"}


def ferplus_label(row: pd.Series) -> tuple[str | None, str]:
    """Return (label or None, reason). Majority over all 10 vote columns.

    Dropped when the top two counts tie, or the winner is unknown / NF / contempt.
    """
    counts = row[VOTE_COLS].astype(int).to_numpy()
    order = np.argsort(-counts, kind="stable")
    top, second = counts[order[0]], counts[order[1]]
    if top == 0:
        return None, "no votes"
    if top == second:
        return None, "tie for top vote"
    winner = VOTE_COLS[order[0]]
    if winner in DROP_IF_WINNER:
        return None, f"majority is {winner}"
    return VOTE_TO_CLASS[winner], "ok"


def load_original() -> pd.DataFrame:
    if not ORIG_CSV.exists():
        raise FileNotFoundError(f"original fer2013.csv not found at {ORIG_CSV}")
    df = pd.read_csv(ORIG_CSV)
    df["orig_row"] = np.arange(len(df))
    return df


def load_pixels(df: pd.DataFrame) -> np.ndarray:
    return np.array([p.split() for p in df["pixels"]], dtype=np.float32)


def check_alignment(orig: pd.DataFrame, fp: pd.DataFrame, lines: list[str]) -> None:
    lines.append(f"original fer2013.csv rows: {len(orig)}; fer2013new.csv rows: {len(fp)}")
    assert len(orig) == len(fp), "row count differs between original and FER+"
    same_usage = (orig["Usage"].to_numpy() == fp["Usage"].to_numpy()).all()
    lines.append(f"Usage column identical row by row: {same_usage}")
    assert same_usage, "row order differs; FER+ join would be wrong"


def decode_images(paths: list[Path]) -> tuple[np.ndarray, list[str], set[Path]]:
    vecs = np.zeros((len(paths), 48 * 48), dtype=np.float32)
    errors = []
    failed = set()
    for i, p in enumerate(paths):
        try:
            with Image.open(p) as im:
                im = im.convert("L")
                if im.size != (48, 48):
                    im = im.resize((48, 48))
                vecs[i] = np.asarray(im, dtype=np.float32).ravel()
        except Exception as exc:  # flag, never drop
            errors.append(f"{p}: {exc}")
            failed.add(p)
    return vecs, errors, failed


def match_split(img_vecs: np.ndarray, orig_vecs: np.ndarray, chunk: int = 2048):
    """Nearest original row for each image (L2). Returns best idx, best dist, second dist."""
    orig_sq = (orig_vecs ** 2).sum(axis=1)
    best_idx = np.zeros(len(img_vecs), dtype=np.int64)
    best_d = np.zeros(len(img_vecs), dtype=np.float32)
    second_d = np.zeros(len(img_vecs), dtype=np.float32)
    for start in range(0, len(img_vecs), chunk):
        block = img_vecs[start:start + chunk]
        d2 = (block ** 2).sum(axis=1)[:, None] + orig_sq[None, :] - 2.0 * block @ orig_vecs.T
        d2 = np.maximum(d2, 0.0)
        part = np.argpartition(d2, 1, axis=1)[:, :2]
        vals = np.take_along_axis(d2, part, axis=1)
        order = np.argsort(vals, axis=1)
        part = np.take_along_axis(part, order, axis=1)
        vals = np.take_along_axis(vals, order, axis=1)
        best_idx[start:start + chunk] = part[:, 0]
        best_d[start:start + chunk] = np.sqrt(vals[:, 0]) / 48.0
        second_d[start:start + chunk] = np.sqrt(vals[:, 1]) / 48.0
    return best_idx, best_d, second_d


def main() -> None:
    lines: list[str] = ["# FER+ join checks", ""]
    orig = load_original()
    fp = pd.read_csv(FERPLUS_CSV)
    check_alignment(orig, fp, lines)

    # FER+ majority per original row.
    labels, reasons = zip(*[ferplus_label(r) for _, r in fp.iterrows()])
    relabel = pd.DataFrame({
        "orig_row": np.arange(len(fp)),
        "Usage": fp["Usage"],
        "image_name_ferplus": fp["Image name"],
        "orig_emotion": orig["emotion"].map(ORIG_EMOTION).to_numpy(),
        "label_ferplus": labels,
        "drop_reason": reasons,
    })
    for col in VOTE_COLS:
        relabel[f"vote_{col}"] = fp[col].to_numpy()
    relabel["label_ferplus"] = relabel["label_ferplus"].where(relabel["label_ferplus"].notna(), "")
    relabel_path = project_path("reports", "ferplus_relabel.csv")
    relabel_path.parent.mkdir(parents=True, exist_ok=True)
    relabel.to_csv(relabel_path, index=False)
    lines.append(f"FER+ rows kept (label_ferplus set): {(relabel['label_ferplus'] != '').sum()} of {len(relabel)}")
    lines.append("drop reasons: " + ", ".join(f"{k}={v}" for k, v in relabel["drop_reason"].value_counts().items()))
    lines.append("")

    # FER-2013 images, grouped by split prefix.
    paths = sorted(FER_ROOT.rglob("*.jpg"))
    rows = []
    decode_errors: list[str] = []
    failed_paths: set[Path] = set()
    for prefix, usage in USAGE_FROM_PREFIX.items():
        group = [p for p in paths if p.name.startswith(prefix + "_")]
        vecs, errs, failed = decode_images(group)
        decode_errors.extend(errs)
        failed_paths.update(failed)
        orig_idx = np.where(orig["Usage"].to_numpy() == usage)[0]
        orig_vecs = load_pixels(orig.iloc[orig_idx])
        best, d1, d2 = match_split(vecs, orig_vecs)
        for p, b, dist1, dist2 in zip(group, best, d1, d2):
            row_id = int(orig_idx[b])
            rows.append({
                "path": p.relative_to(DATASETS_DIR).as_posix(),
                "folder_label": p.parent.name,
                "usage_from_name": usage,
                "orig_row": row_id,
                "orig_emotion": ORIG_EMOTION[int(orig.loc[row_id, "emotion"])],
                "match_dist": round(float(dist1), 4),
                "second_dist": round(float(dist2), 4),
                "decode_error": p in failed_paths,
            })
        lines.append(f"{prefix}: images={len(group)} original rows={len(orig_idx)}")

    match = pd.DataFrame(rows)
    match["folder_matches_orig"] = match["folder_label"] == match["orig_emotion"]

    # Exact pixel duplicates in the original CSV: the image cannot be tied to one row.
    # Such an image gets a FER+ label only if every duplicate row agrees on it.
    same_pixels_rows = orig.groupby("pixels")["orig_row"].apply(list).to_dict()
    fp_label = relabel["label_ferplus"].to_numpy()
    fp_reason = relabel["drop_reason"].to_numpy()
    dup_count = [len(same_pixels_rows[orig.at[r, "pixels"]]) for r in match["orig_row"]]
    match["dup_pixels"] = np.array(dup_count) > 1
    labels_out, reasons_out = [], []
    for r in match["orig_row"]:
        group = same_pixels_rows[orig.at[r, "pixels"]]
        labels = {fp_label[g] for g in group}
        if len(labels) == 1 and "" not in labels:
            labels_out.append(labels.pop())
            reasons_out.append("ok")
        elif len(labels) == 1:
            labels_out.append("")
            reasons_out.append(fp_reason[r])
        else:
            labels_out.append("")
            reasons_out.append("duplicate pixels with conflicting FER+ labels")
    match["label_ferplus"] = labels_out
    match["ferplus_drop_reason"] = reasons_out
    match["ferplus_vs_folder_agree"] = match["label_ferplus"] == match["folder_label"]

    lines.append("")
    lines.append(f"images matched: {len(match)}")
    lines.append(f"folder label == original emotion: {int(match['folder_matches_orig'].sum())} of {len(match)}")
    lines.append(f"distinct original rows matched: {match['orig_row'].nunique()} of {len(orig)}")
    lines.append(f"images whose pixels appear on more than one original row: {int(match['dup_pixels'].sum())}")
    lines.append(f"match_dist quantiles (0,50,90,99,100): {np.quantile(match['match_dist'], [0, .5, .9, .99, 1]).round(4).tolist()}")
    lines.append(f"second_dist median (non-duplicate images): {match.loc[~match['dup_pixels'], 'second_dist'].median():.2f}")
    lines.append(f"decode errors: {len(decode_errors)}")
    for err in decode_errors[:20]:
        lines.append(f"  {err}")

    mismatch = match[~match["folder_matches_orig"]]
    lines.append(f"mismatched folder/original labels by folder: {mismatch['folder_label'].value_counts().to_dict()}")
    lines.append(f"mismatched rows that are pixel duplicates: {int(mismatch['dup_pixels'].sum())} of {len(mismatch)}")

    match_path = project_path("data", "fer2013_match.csv")
    match.to_csv(match_path, index=False)
    lines.append("")
    lines.append(f"FER-2013 images with label_ferplus set: {(match['label_ferplus'] != '').sum()} of {len(match)}")
    lines.append(f"of those, FER+ label == folder label: {int(match.loc[match['label_ferplus'] != '', 'ferplus_vs_folder_agree'].sum())}")

    log_path = project_path("logs", "ferplus_join.txt")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {match_path.name}, {relabel_path.name}, {log_path.name}")


if __name__ == "__main__":
    main()
