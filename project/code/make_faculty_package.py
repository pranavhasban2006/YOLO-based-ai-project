r"""Collects the CSV and image files for the faculty into one folder: Faculty_Files/ at the repo root.

  1_CSV_files/
    a_dataset_csv/      each dataset as a CSV (FER-2013 in its original 'emotion,pixels,Usage' form, text
                        datasets, speech file list, WESAD 32 statistical features per window)
    b_feature_csv/      the deep-feature CSVs (f0..fN, label) used by the SMOTE / 80:20 / 10-fold protocol,
                        best model of each modality
    c_results_csv/      protocol summary + held-out test metrics of every model
  2_Image_files/
    a_dataset_samples/  real sample images, 2 per class (FER-2013, CK+, RAVDESS video frames) as PNG files
    b_figures/          every chart and diagram from the Review-2 and Final-Results documents

Usage (repo root): project\.venv-train\Scripts\python project\code\make_faculty_package.py
"""
import collections
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import make_review2 as mr  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "project" / "data"
REP = ROOT / "project" / "reports"
OUT = ROOT / "Faculty_Files"
CSV_DS, CSV_FEAT, CSV_RES = OUT / "1_CSV_files" / "a_dataset_csv", OUT / "1_CSV_files" / "b_feature_csv", \
    OUT / "1_CSV_files" / "c_results_csv"
IMG_DS, IMG_FIG = OUT / "2_Image_files" / "a_dataset_samples", OUT / "2_Image_files" / "b_figures"
APP7 = mr.APP7
BEST_FEATURES = ["face_densenet_se", "audio_hubert_transformer", "text_roberta_bilstm_attn_goemotions",
                 "text_bert_cnn_bilstm_emotion", "wesad_cnn_gru"]


def dataset_csvs():
    # FER-2013 in its original Kaggle CSV layout: emotion (0-6), pixels (2304 space-separated values), Usage
    rows = []
    for split, usage in (("train", "Training"), ("test", "PublicTest")):
        for k, c in enumerate(APP7):
            for f in sorted((DATA / "fer2013" / split / c).glob("*"))[:2]:
                px = np.asarray(Image.open(f).convert("L").resize((48, 48))).reshape(-1)
                rows.append((k, " ".join(map(str, px)), usage, c, f.name))
    pd.DataFrame(rows, columns=["emotion", "pixels", "Usage", "emotion_name", "file"]).to_csv(
        CSV_DS / "fer2013_sample_2_per_class.csv", index=False)
    counts = [(s, c, len(list((DATA / "fer2013" / s / c).glob("*")))) for s in ("train", "test") for c in APP7]
    pd.DataFrame(counts, columns=["split", "emotion", "images"]).to_csv(CSV_DS / "fer2013_class_counts.csv",
                                                                        index=False)
    # Text datasets, complete
    go, em, (tw, tw_names) = mr.text_sets()
    for s, (t, y) in go.items():
        pd.DataFrame({"text": t, "label": y, "emotion": [APP7[i] for i in y]}).to_csv(
            CSV_DS / f"goemotions_ekman7_{s}.csv", index=False)
    em6 = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    for s, d in em.items():
        pd.DataFrame({"text": d["text"], "label": d["label"], "emotion": [em6[i] for i in d["label"]]}).to_csv(
            CSV_DS / f"dair_ai_emotion_{s}.csv", index=False)
    for s, (t, y) in tw.items():
        pd.DataFrame({"text": t, "label": y, "emotion": [tw_names[i] for i in y]}).to_csv(
            CSV_DS / f"tweeteval_emotion_{s}.csv", index=False)
    # Speech: one row per clip
    import soundfile as sf
    rows = []
    for src, lab, p in mr.audio_items():
        info = sf.info(p)
        rows.append((src, lab, p.name, round(info.duration, 3), info.samplerate))
    pd.DataFrame(rows, columns=["dataset", "emotion", "file", "duration_s", "sample_rate"]).to_csv(
        CSV_DS / "speech_ravdess_tess_savee_files.csv", index=False)
    # WESAD: 32 statistical features per 60 s window (mean, std, min, max of 8 chest channels)
    d = np.load(DATA / "wesad_windows.npz")
    x, y, subj = d["x"], d["y"], d["subj"]
    names = ["ECG", "EDA", "EMG", "Resp", "Temp", "ACC_x", "ACC_y", "ACC_z"]
    feats = {}
    for ci, n in enumerate(names):
        ch = x[:, ci].astype(np.float32)
        feats.update({f"{n}_mean": ch.mean(1), f"{n}_std": ch.std(1), f"{n}_min": ch.min(1), f"{n}_max": ch.max(1)})
    df = pd.DataFrame(feats).round(5)
    df.insert(0, "subject", subj)
    df["label"] = y
    df["state"] = [["baseline", "stress", "amusement"][i] for i in y]
    df.to_csv(CSV_DS / "wesad_window_features_32.csv", index=False)


def feature_csvs():
    for stem in BEST_FEATURES:
        src = ROOT / "project" / "features" / f"{stem}.csv"
        shutil.copy2(src, CSV_FEAT / f"{stem}.csv")
        # preview: first 2 rows per class, first 10 feature columns
        df = pd.read_csv(src)
        cols = [c for c in df.columns if c.startswith("f")][:10] + ["label"]
        df.groupby("label").head(2)[cols].sort_values("label").to_csv(CSV_FEAT / f"PREVIEW_{stem}.csv", index=False)


def results_csvs():
    shutil.copy2(REP / "protocol" / "summary.csv", CSV_RES / "protocol_summary_all_models.csv")
    rows, cm_rows = [], []
    for f in sorted(REP.glob("*_metrics.json")):
        m = json.loads(f.read_text())
        if "per_class" not in m:
            continue
        pc = list(m["per_class"].values())
        t = m["inference_time"]
        name = f.stem.replace("_metrics", "")
        rows.append({"model": name, "dataset": m.get("dataset"), "test_size": m.get("test_size"),
                     "accuracy": m["accuracy"], "macro_precision": round(np.mean([v["precision"] for v in pc]), 4),
                     "macro_recall": round(np.mean([v["recall"] for v in pc]), 4), "macro_f1": m["macro_f1"],
                     "cpu_ms": next(v for k, v in t.items() if k.startswith("cpu")),
                     "gpu_ms": next((v for k, v in t.items() if k.startswith("cuda")), None)})
        labels = m["confusion_matrix"]["labels"]
        for lab, r in zip(labels, m["confusion_matrix"]["rows_true_cols_pred"]):
            cm_rows.append({"model": name, "true_class": lab, **{f"pred_{l}": v for l, v in zip(labels, r)}})
    pd.DataFrame(rows).to_csv(CSV_RES / "test_results_all_models.csv", index=False)
    pd.DataFrame(cm_rows).to_csv(CSV_RES / "confusion_matrices_all_models.csv", index=False)


def images():
    for c in APP7:
        dst = IMG_DS / "FER-2013" / c
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted((DATA / "fer2013" / "train" / c).glob("*"))[:2]:
            Image.open(f).save(dst / f"{Path(f).stem}.png")
    z, ck = mr.ck_files()
    for c, files in ck.items():
        dst = IMG_DS / "CK+" / c
        dst.mkdir(parents=True, exist_ok=True)
        for n in (files[0], files[len(files) // 2]):
            Image.open(io.BytesIO(z.read(n))).save(dst / Path(n).name)
    import tempfile
    import cv2
    tmp = Path(tempfile.mkdtemp())
    by = collections.defaultdict(list)
    for zp, n in mr.ravdess_video():
        if Path(n).stem.startswith("02"):
            by[mr.RAV_EMO[Path(n).stem.split("-")[2]]].append((zp, n))
    for e, cand in by.items():
        dst = IMG_DS / "RAVDESS_video_frames" / e
        dst.mkdir(parents=True, exist_ok=True)
        for zp, n in (cand[0], cand[len(cand) // 2]):
            f = tmp / Path(n).name
            f.write_bytes(zipfile.ZipFile(zp).read(n))
            cap = cv2.VideoCapture(str(f))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.6))
            ok, fr = cap.read()
            cap.release()
            if ok:
                cv2.imwrite(str(dst / f"{Path(n).stem}.png"), fr)
    for sub, src in (("review2_dataset_and_methodology", REP / "review2" / "fig"),
                     ("final_results", REP / "final" / "fig")):
        shutil.copytree(src, IMG_FIG / sub, dirs_exist_ok=True)


README = """FACULTY FILES - Multimodal Emotion Recognition using Hybrid Deep Learning Models
Team: Prakul Jain (24BIT0548), Pranav Hasban (24BIT0537), Shreya Mehrotra (24BIT0547)

1_CSV_files
  a_dataset_csv   The datasets in CSV form.
                  fer2013_sample_2_per_class.csv  original FER-2013 layout: emotion, pixels (48x48 = 2304 values), Usage
                  fer2013_class_counts.csv        images per class (train / test)
                  goemotions_ekman7_*.csv         GoEmotions mapped to 7 classes (text, label, emotion)
                  dair_ai_emotion_*.csv           dair-ai Emotion, 6 classes
                  tweeteval_emotion_*.csv         TweetEval emotion, 4 classes
                  speech_ravdess_tess_savee_files.csv   every speech clip: dataset, emotion, file, duration
                  wesad_window_features_32.csv    WESAD: 32 statistical features (mean/std/min/max of 8 chest
                                                  channels) per 60 s window + subject + label
  b_feature_csv   The CSV files used by the SMOTE / 80:20 / 70:30 / 10-fold protocol: one row per held-out sample,
                  columns f0..fN = deep features learned by the hybrid model, label = class. Best model of each
                  modality; PREVIEW_*.csv shows 2 rows per class and the first 10 feature columns.
                  (CSV files of all 27 models: project/features/)
  c_results_csv   protocol_summary_all_models.csv   accuracy / macro-F1 for 80:20, 70:30, 10-fold, with and without SMOTE
                  test_results_all_models.csv       accuracy, precision, recall, F1, CPU/GPU time of every model
                  confusion_matrices_all_models.csv confusion matrix of every model

2_Image_files
  a_dataset_samples  real sample images, 2 per class: FER-2013, CK+, RAVDESS video frames
  b_figures          all charts and diagrams: dataset samples, class balance before/after SMOTE, architecture
                     diagrams, flowchart, comparison charts, confusion matrices, inference time
"""


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for p in (CSV_DS, CSV_FEAT, CSV_RES, IMG_DS, IMG_FIG):
        p.mkdir(parents=True, exist_ok=True)
    dataset_csvs()
    feature_csvs()
    results_csvs()
    images()
    (OUT / "README.txt").write_text(README, encoding="utf-8")
    files = [f for f in OUT.rglob("*") if f.is_file()]
    print(f"saved {OUT}: {len(files)} files, {sum(f.stat().st_size for f in files) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
