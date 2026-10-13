"""Phase 3: subject-disjoint splits, experiment definitions and class weights.

Rules (see project/DECISIONS.md, Phase 3):
  ravdess, crema_d, jaffe   split BY SUBJECT, 70/15/15 of subjects, seed 42, greedy class balance
  fer2013                   official split: Training -> train (10% carved off as val, stratified);
                            PublicTest + PrivateTest -> test. Labels come from label_ferplus.

Writes project/data/splits.csv, project/data/experiments.json, project/data/class_weights.json,
project/logs/phase_3_report.md and sets state.json phase_done = 3.
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import CLASSES, CLASS_TO_IDX, SEED, project_path, read_state, write_state

sys.stdout.reconfigure(encoding="utf-8")

FACES_IN = project_path("data", "faces_index.parquet")
SPLITS_OUT = project_path("data", "splits.csv")
EXPERIMENTS_OUT = project_path("data", "experiments.json")
WEIGHTS_OUT = project_path("data", "class_weights.json")
REPORT_OUT = project_path("logs", "phase_3_report.md")
FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}
SUBJECT_DATASETS = ["ravdess", "crema_d", "jaffe"]
FER_VAL_FRAC = 0.10


def assign_subjects(sub_rows: pd.DataFrame, seed: int) -> dict:
    """Map subject -> split. Three subjects go first, one per split, so every split is non-empty;
    the rest are placed greedily to match the target class mix of each split."""
    rng = np.random.default_rng(seed)
    subjects = sorted(sub_rows["subject"].unique())
    rng.shuffle(subjects)
    counts = {s: sub_rows[sub_rows["subject"] == s]["label"].value_counts().reindex(CLASSES, fill_value=0)
              for s in subjects}
    total = sub_rows["label"].value_counts().reindex(CLASSES, fill_value=0)
    target = {sp: total * f for sp, f in FRACS.items()}
    cur = {sp: pd.Series(0, index=CLASSES, dtype=float) for sp in FRACS}
    assignment = {}
    order = sorted(subjects, key=lambda s: -counts[s].sum())  # big subjects first, shuffled ties
    first = dict(zip(FRACS, subjects[:3]))
    for sp, s in first.items():
        assignment[s] = sp
        cur[sp] += counts[s]
    for s in order:
        if s in assignment:
            continue
        # Fill the split that is furthest below its share of rows; class balance breaks ties.
        # (Distance-to-target alone would always pick the smallest split and starve train.)
        best = min(FRACS, key=lambda sp: (
            float(cur[sp].sum() / target[sp].sum()),
            float(((cur[sp] + counts[s]) / target[sp].replace(0, 1) - 1).abs().mean()),
        ))
        assignment[s] = best
        cur[best] += counts[s]
    return assignment


def build_video_and_jaffe(faces: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for ds in SUBJECT_DATASETS:
        sub = faces[faces["dataset"] == ds].copy()
        mapping = assign_subjects(sub, SEED)
        sub["split"] = sub["subject"].map(mapping)
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def build_fer(faces: pd.DataFrame) -> pd.DataFrame:
    fer = faces[faces["dataset"] == "fer2013"].copy()
    fer["label"] = fer["label_ferplus"]
    fer = fer[fer["label"].notna() & (fer["label"] != "")].copy()
    fer["split"] = np.where(fer["split_hint"].isin(["PublicTest", "PrivateTest"]), "test", "train")
    train_idx = fer.index[fer["split"] == "train"]
    tr, va = train_test_split(train_idx, test_size=FER_VAL_FRAC, random_state=SEED,
                              stratify=fer.loc[train_idx, "label"])
    fer.loc[va, "split"] = "val"
    return fer


def main() -> None:
    faces = pd.read_parquet(FACES_IN)
    # Only the 7 target classes are kept. Any other label is a bug from earlier phases.
    bad = set(faces["label"].dropna()) - set(CLASSES)
    assert not bad, f"unexpected folder labels: {bad}"

    fer = build_fer(faces)
    rest = build_video_and_jaffe(faces)
    fer_dropped = int(((faces["dataset"] == "fer2013") & (faces["label_ferplus"].isna() | (faces["label_ferplus"] == ""))).sum())

    splits = pd.concat([rest, fer], ignore_index=True)
    splits["label_id"] = splits["label"].map(CLASS_TO_IDX).astype("int64")
    splits["subject"] = splits["subject"].fillna("")

    # Leakage checks. Fail loudly.
    for ds in SUBJECT_DATASETS:
        sub = splits[splits["dataset"] == ds]
        per = sub.groupby("subject")["split"].nunique()
        assert (per == 1).all(), f"{ds}: a subject appears in more than one split"
    assert not splits["face_path"].duplicated().any(), "a face_path appears twice"
    assert set(splits["split"]) == {"train", "val", "test"}

    keep = ["face_path", "dataset", "split", "label", "label_id", "subject", "frame_id", "path", "kind"]
    splits[keep].to_csv(SPLITS_OUT, index=False)

    # Class weights for the combined training set (inverse frequency, mean weight 1).
    tr = splits[splits["split"] == "train"]
    counts = tr["label_id"].value_counts().reindex(range(len(CLASSES)), fill_value=0).to_numpy()
    weights = len(tr) / (len(CLASSES) * np.maximum(counts, 1))
    weights = weights / weights.mean()
    WEIGHTS_OUT.write_text(json.dumps({
        "classes": CLASSES,
        "train_counts": {c: int(n) for c, n in zip(CLASSES, counts)},
        "weights": {c: round(float(w), 4) for c, w in zip(CLASSES, weights)},
        "note": "combined train split of all datasets; inverse frequency, normalised to mean 1",
    }, indent=2), encoding="utf-8")

    experiments = {
        "classes": CLASSES,
        "E1_within_dataset": {
            "description": "train and test on the same dataset",
            "runs": {ds: {"train": [ds], "test": ds} for ds in ["fer2013", "jaffe", "ravdess", "crema_d"]},
        },
        "E2_combined": {
            "description": "train on train splits of all datasets; report each dataset's test split",
            "train": ["fer2013", "jaffe", "ravdess", "crema_d"],
            "test": ["fer2013", "jaffe", "ravdess", "crema_d"],
        },
        "E3_leave_one_dataset_out": {
            "description": "train on all datasets except D, test on all of D (main real-dataset result)",
            "runs": {ds: {"train": [d for d in ["fer2013", "jaffe", "ravdess", "crema_d"] if d != ds],
                          "test": ds}
                     for ds in ["fer2013", "jaffe", "ravdess", "crema_d"]},
        },
        "shared_classes_note": "crema_d has no surprise class; also report metrics on the 6 shared classes",
        "shared_classes": [c for c in CLASSES if c != "surprise"],
    }
    EXPERIMENTS_OUT.write_text(json.dumps(experiments, indent=2), encoding="utf-8")

    # Report.
    table = splits.groupby(["dataset", "split"]).size().unstack(fill_value=0)
    subj = splits[splits["subject"] != ""].groupby(["dataset", "split"])["subject"].nunique().unstack(fill_value=0)
    lines = ["# Phase 3 report", "",
             f"Rows written: {len(splits)} -> project/data/splits.csv", "",
             "## Rows per dataset and split", "", "```", table.to_string(), "```", "",
             "## Subjects per dataset and split (video and jaffe)", "", "```", subj.to_string(), "```", "",
             f"fer2013 rows dropped because label_ferplus is empty: {fer_dropped}", "",
             "## Class counts in the combined train split", "", "```",
             pd.Series(counts, index=CLASSES).to_string(), "```", "",
             "## Class weights (saved to project/data/class_weights.json)", "",
             "```", pd.Series(weights, index=CLASSES).round(3).to_string(), "```", "",
             "Checks passed: no subject in two splits within a dataset; no face_path twice.", ""]
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")

    state = read_state()
    state["phase_done"] = 3
    write_state(state)
    print(table.to_string())
    print(subj.to_string())
    print(f"fer2013 dropped (no FER+ label): {fer_dropped}")
    print("phase_done = 3")


if __name__ == "__main__":
    main()
