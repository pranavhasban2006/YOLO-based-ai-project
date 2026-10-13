"""Phase 4a: package face crops, fer2013 images, splits and experiment files for Colab.

Nothing under ./datasets/ or the originals under ./project/data/ is moved or changed.
Writes project/colab_upload/{faces.zip, splits.csv, faces_index.csv, experiments.json,
class_weights.json, classes.json} and project/logs/phase_4_package.json (used by the report).
"""
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from common import CLASSES, DATASETS_DIR, REPO_ROOT, project_path

sys.stdout.reconfigure(encoding="utf-8")

UPLOAD = project_path("colab_upload")
FACES_ROOT = project_path("data", "faces")
SPLITS = project_path("data", "splits.csv")
FACES_IDX = project_path("data", "faces_index.parquet")
ZIP_PATH = UPLOAD / "faces.zip"
REPORT_JSON = project_path("logs", "phase_4_package.json")


def zip_path_for(face_path: str, dataset: str, split: str, label: str) -> str:
    """Path inside faces.zip. Crops keep faces/<dataset>/<subject>/...; fer2013 originals go to
    fer2013/<split>/<label>/<file> (label = the FER+ label the row is used with)."""
    if dataset == "fer2013":
        return f"fer2013/{split}/{label}/{Path(face_path).name}"
    rel = Path(face_path).relative_to("project/data/faces")
    return f"faces/{rel.as_posix()}"


def main() -> None:
    UPLOAD.mkdir(parents=True, exist_ok=True)
    splits = pd.read_csv(SPLITS, dtype={"subject": str}, keep_default_na=False)
    faces = pd.read_parquet(FACES_IDX)

    # Rewrite each row's path to the zip root.
    # fer2013 source paths are "datasets/..." (repo relative), crops are "project/data/faces/...".
    splits["zip_path"] = [
        zip_path_for(fp, ds, sp, lb)
        for fp, ds, sp, lb in zip(splits["face_path"], splits["dataset"], splits["split"], splits["label"])
    ]
    src_of = dict(zip(splits["zip_path"], splits["face_path"]))

    # Verify every source exists before zipping.
    missing = []
    for zp, fp in src_of.items():
        src = REPO_ROOT / fp
        if not src.is_file():
            missing.append(fp)
    dropped = len(missing)
    if missing:
        keep = [fp not in set(missing) for fp in splits["face_path"]]
        splits = splits[keep].reset_index(drop=True)
        src_of = {zp: fp for zp, fp in src_of.items() if fp not in set(missing)}
        print(f"WARNING: {dropped} source files missing; rows dropped and listed in the report")

    # Build the zip (JPEG and PNG are already compressed, so store without deflate).
    total_bytes = 0
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for n, (zp, fp) in enumerate(src_of.items(), start=1):
            zf.write(REPO_ROOT / fp, arcname=zp)
            total_bytes += (REPO_ROOT / fp).stat().st_size
            if n % 10000 == 0:
                print(f"zipped {n}/{len(src_of)}")
    print(f"faces.zip: {len(src_of)} files, {ZIP_PATH.stat().st_size / 1e9:.2f} GB")

    # Write the CSVs with zip-relative paths. "path" is the source path on this PC, so it is dropped.
    splits_out = splits.copy()
    splits_out["face_path"] = splits_out["zip_path"]
    splits_out = splits_out.drop(columns=["zip_path", "path"])
    splits_out.to_csv(UPLOAD / "splits.csv", index=False)

    # faces_index.csv: the same rows (kept rows only) with the zip path and split. Dropped rows
    # were removed from splits, so they are removed here too.
    # faces_index.csv: the rows that are in splits, with zip paths and the split they landed in.
    src_to_zip = dict(zip(splits["face_path"], splits["zip_path"]))
    src_to_split = dict(zip(splits["face_path"], splits["split"]))
    fidx = faces[faces["face_path"].isin(src_to_zip)].copy()
    fidx["split"] = fidx["face_path"].map(src_to_split)
    fidx["face_path"] = fidx["face_path"].map(src_to_zip)
    fidx = fidx.drop(columns=["path"])
    fidx.to_csv(UPLOAD / "faces_index.csv", index=False)
    fidx_out = fidx

    shutil.copy(project_path("data", "experiments.json"), UPLOAD / "experiments.json")
    shutil.copy(project_path("data", "class_weights.json"), UPLOAD / "class_weights.json")
    (UPLOAD / "classes.json").write_text(json.dumps(CLASSES), encoding="utf-8")

    # Verify by extracting into a temp folder outside the project, then deleting it.
    tmp = Path(tempfile.mkdtemp(prefix="yolo_verify_"))
    try:
        with zipfile.ZipFile(ZIP_PATH) as zf:
            bad = zf.testzip()
            assert bad is None, f"corrupt member {bad}"
            zf.extractall(tmp)
        check_missing = []
        for name in ["splits.csv", "faces_index.csv"]:
            df = pd.read_csv(UPLOAD / name, keep_default_na=False)
            for p in df["face_path"]:
                if not (tmp / p).is_file():
                    check_missing.append(p)
        print(f"verify: missing after extraction = {len(check_missing)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert not tmp.exists()

    total = sum(f.stat().st_size for f in UPLOAD.rglob("*") if f.is_file())
    report = {
        "zip_files": len(src_of),
        "zip_gb": round(ZIP_PATH.stat().st_size / 1e9, 3),
        "upload_gb_total": round(total / 1e9, 3),
        "dropped_missing_sources": dropped,
        "verify_missing_after_extract": len(check_missing),
        "splits_rows": int(len(splits_out)),
        "faces_index_rows": int(len(fidx_out)),
        "fer2013_images_in_zip": int((splits_out["dataset"] == "fer2013").sum()),
        "fer2013_rows_dropped_no_ferplus_label": int(
            ((faces["dataset"] == "fer2013") & (faces["label_ferplus"].isna() | (faces["label_ferplus"] == ""))).sum()),
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
