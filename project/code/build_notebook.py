"""Phase 4b: write project/colab_upload/train_colab.ipynb (nbformat 4).

The code cells are plain Python. The local test (test_notebook.py) runs the same cells with
YOLO_TEST=1 and a tiny configuration, so the notebook that goes to Colab is the code that was tested.
"""
import json
import sys
from pathlib import Path

from common import project_path

sys.stdout.reconfigure(encoding="utf-8")
OUT = project_path("colab_upload", "train_colab.ipynb")

CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


md("""
# Facial emotion model: CNN and ResNet-18 on FER-2013, JAFFE, RAVDESS and CREMA-D

Runs top to bottom with **Runtime > Run all**. Needs a GPU (T4). Resumes if the session
disconnects: run all again and finished runs are skipped, unfinished runs continue from their last checkpoint.

Data: `MyDrive/YOLO/colab_upload/` (faces.zip, splits.csv, experiments.json, ...).
Results: `MyDrive/YOLO/colab_results/`.

Order: E2 (combined training, each dataset's test split reported) for both models, then E3
(leave-one-dataset-out) with the model that has the higher mean macro-F1 in E2.
""")

code(r'''
# ---- 1. Configuration -----------------------------------------------------------------------
import os, time, math, json, shutil, zipfile, random
from pathlib import Path
import numpy as np
import pandas as pd

T0 = time.time()
SEED = 42
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
DATASETS = ["fer2013", "jaffe", "ravdess", "crema_d"]
MAX_HOURS = float(os.environ.get("MAX_HOURS", "6"))  # time budget for E3 (leave-one-dataset-out)

# The local test switches TEST_MODE on through an environment variable. Colab never sets it.
TEST_MODE = os.environ.get("YOLO_TEST") == "1"
if TEST_MODE:
    TEST_ROOT = Path(os.environ["YOLO_TEST_ROOT"])
    UPLOAD_DIR = Path(os.environ["YOLO_UPLOAD_DIR"])
    DATA_DIR = TEST_ROOT / "data"
    RESULTS_DIR = TEST_ROOT / "results"
    EP_CNN, EP_RES, HEAD_EPOCHS, E3_EPOCHS = 1, 2, 1, 1
    BATCH = 16
    NUM_WORKERS = 0
    RUN_E3 = os.environ.get("YOLO_TEST_E3") == "1"
    TEST_SUBSET_PER_GROUP = 12
else:
    UPLOAD_DIR = Path("/content/drive/MyDrive/YOLO/colab_upload")
    DATA_DIR = Path("/content/data")
    RESULTS_DIR = Path("/content/drive/MyDrive/YOLO/colab_results")
    EP_CNN, EP_RES, HEAD_EPOCHS, E3_EPOCHS = 40, 25, 3, 20
    BATCH = 64
    NUM_WORKERS = 2
    RUN_E3 = True
    TEST_SUBSET_PER_GROUP = None
PATIENCE = 6
CAP_FRAC = 0.5  # a dataset contributes at most this share of one epoch

print("TEST_MODE:", TEST_MODE, "| MAX_HOURS:", MAX_HOURS)
''')

code(r'''
# ---- 2. GPU check and Google Drive ----------------------------------------------------------
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and not TEST_MODE else "cpu")
if not TEST_MODE:
    if DEVICE.type != "cuda":
        raise RuntimeError("No GPU attached. Choose Runtime > Change runtime type > T4 GPU, then Run all again.")
    print("GPU:", torch.cuda.get_device_name(0))
    from google.colab import drive
    drive.mount("/content/drive")
else:
    print("local test on", DEVICE)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
USE_AMP = DEVICE.type == "cuda"
torch.backends.cudnn.benchmark = USE_AMP
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
''')

code(r'''
# ---- 3. Unzip faces, load splits ------------------------------------------------------------
SPLITS_CSV = UPLOAD_DIR / "splits.csv"
ZIP_FILE = UPLOAD_DIR / "faces.zip"
splits = pd.read_csv(SPLITS_CSV, dtype={"subject": str}, keep_default_na=False)
assert set(splits["label"]) <= set(CLASSES), "unexpected label in splits.csv"

if TEST_MODE:
    # Test: pull only a small subset out of the zip, so the test stays fast.
    splits = splits.groupby(["dataset", "split"], group_keys=False).head(TEST_SUBSET_PER_GROUP).reset_index(drop=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_FILE) as zf:
        for p in splits["face_path"]:
            if not (DATA_DIR / p).is_file():
                zf.extract(p, DATA_DIR)
else:
    if not (DATA_DIR / "faces").exists() or not (DATA_DIR / "fer2013").exists():
        print("unzipping faces.zip into", DATA_DIR, "(about 1 to 2 minutes)")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ZIP_FILE) as zf:
            zf.extractall(DATA_DIR)
    else:
        print("data already in", DATA_DIR, "- skipping unzip")

missing = [p for p in splits["face_path"] if not (DATA_DIR / p).is_file()]
assert not missing, f"{len(missing)} files missing, first: {missing[:3]}"
print(splits.groupby(["dataset", "split"]).size().unstack(fill_value=0))
''')

code(r'''
# ---- 4. Datasets, transforms and the per-epoch sampler --------------------------------------
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms as T

def make_transform(arch, train):
    if arch == "cnn":
        base = [T.Grayscale(1), T.Resize((48, 48))]
        norm = [T.Normalize([0.5], [0.5])]
    else:
        base = [T.Resize((224, 224))]
        norm = [T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    if not train:
        return T.Compose(base + [T.ToTensor()] + norm)
    aug = [T.RandomHorizontalFlip(), T.RandomRotation(15), T.ColorJitter(brightness=0.3, contrast=0.3)]
    return T.Compose(base + aug + [T.ToTensor()] + norm + [T.RandomErasing(p=0.25, scale=(0.02, 0.15))])

class FaceDataset(Dataset):
    def __init__(self, df, arch, train):
        self.paths = [str(DATA_DIR / p) for p in df["face_path"]]
        self.labels = df["label_id"].to_numpy()
        self.tf = make_transform(arch, train)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), int(self.labels[i])

def epoch_indices(df_train, rng):
    """Each dataset gives at most CAP_FRAC of the training set's size per epoch; a fresh random subset each epoch."""
    cap = int(CAP_FRAC * len(df_train))
    picked = []
    for _, grp in df_train.groupby("dataset"):
        ids = grp.index.to_numpy()
        picked.append(rng.choice(ids, size=min(len(ids), cap), replace=False))
    idx = np.concatenate(picked)
    rng.shuffle(idx)
    return idx

def class_weights_from(df_train):
    counts = np.bincount(df_train["label_id"], minlength=len(CLASSES)).astype(float)
    w = len(df_train) / (len(CLASSES) * np.maximum(counts, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)
''')

code(r'''
# ---- 5. Models ------------------------------------------------------------------------------
import torch.nn as nn
import torchvision

class SmallCNN(nn.Module):
    """4 blocks of (Conv3x3-BN-ReLU x2, MaxPool), 64/128/256/512 channels, GAP, dropout 0.4, linear to 7."""
    def __init__(self, n_classes=7):
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

def build_model(arch):
    if arch == "cnn":
        return SmallCNN()
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    return m

def set_trainable(model, arch, epoch):
    """ResNet: head only for the first HEAD_EPOCHS epochs, then all layers."""
    head_only = arch == "resnet18" and epoch < HEAD_EPOCHS
    for name, p in model.named_parameters():
        p.requires_grad = (not head_only) or name.startswith("fc.")

def lr_at(arch, epoch, total):
    if arch == "cnn":
        base, warm, start, n = 1e-3, 2, 0, total
    else:
        if epoch < HEAD_EPOCHS:
            return 1e-3
        base, warm, start, n = 1e-4, 0, HEAD_EPOCHS, total - HEAD_EPOCHS
    e = epoch - start
    if e < warm:
        return base * (e + 1) / warm
    prog = (e - warm) / max(1, n - warm)
    return base * 0.5 * (1 + math.cos(math.pi * prog))
''')

code(r'''
# ---- 6. Metrics -----------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def score(y, p):
    """Macro-F1 is averaged over the classes that occur in y. crema_d has no surprise, so it is scored on 6."""
    prec, rec, f1, sup = precision_recall_fscore_support(y, p, labels=list(range(len(CLASSES))), zero_division=0)
    present = [c for c in range(len(CLASSES)) if (y == c).any()]
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(np.mean([f1[c] for c in present])),
        "classes_scored": [CLASSES[c] for c in present],
        "per_class": {CLASSES[c]: {"precision": float(prec[c]), "recall": float(rec[c]),
                                   "f1": float(f1[c]), "support": int(sup[c])} for c in range(len(CLASSES))},
    }

def save_confusion(y, p, path, title):
    cm = confusion_matrix(y, p, labels=list(range(len(CLASSES))))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
''')

code(r'''
# ---- 7. Training with checkpoints, resume, early stopping and OOM fallback ------------------
EPOCH_SECONDS = {}  # arch -> list of epoch durations, used to estimate E3 run time
BATCH_CURRENT = BATCH

@torch.no_grad()
def predict(model, df, arch):
    model.eval()
    loader = DataLoader(FaceDataset(df, arch, False), batch_size=BATCH_CURRENT, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda")
    ys, ps = [], []
    for x, y in loader:
        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            out = model(x.to(DEVICE, non_blocking=True))
        ps.append(out.float().argmax(1).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)

def save_atomic(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)

def run_experiment(run_name, arch, train_df, val_df, tests, n_epochs):
    """tests: {name: DataFrame}. Returns the metrics dict. Skips a finished run."""
    global BATCH_CURRENT
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "done.json").exists():
        print(f"[{run_name}] already done, skipping")
        return json.loads((run_dir / "metrics.json").read_text())

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"[{run_name}] train={len(train_df)} val={len(val_df)} epochs={n_epochs} batch={BATCH_CURRENT}")
    try:
        model = build_model(arch).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_AMP)
        criterion = nn.CrossEntropyLoss(weight=class_weights_from(train_df).to(DEVICE), label_smoothing=0.1)
        state = {"epoch": 0, "best_f1": -1.0, "best_epoch": -1, "bad": 0, "history": []}
        last = run_dir / "last.pt"
        if last.exists():
            ck = torch.load(last, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
            state = ck["state"]
            print(f"[{run_name}] resumed from epoch {state['epoch']}")

        ds_train = FaceDataset(train_df, arch, True)
        ds_val = FaceDataset(val_df, arch, False)
        for epoch in range(state["epoch"], n_epochs):
            t_ep = time.time()
            lr = lr_at(arch, epoch, n_epochs)
            for g in opt.param_groups:
                g["lr"] = lr
            set_trainable(model, arch, epoch)
            rng = np.random.default_rng(SEED * 1000 + epoch)
            idx = epoch_indices(train_df, rng)
            loader = DataLoader(Subset(ds_train, idx), batch_size=BATCH_CURRENT, shuffle=True,
                                num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda",
                                generator=torch.Generator().manual_seed(SEED + epoch), drop_last=False)
            model.train()
            loss_sum, n_seen, n_ok = 0.0, 0, 0
            for x, y in loader:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                    out = model(x)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                loss_sum += loss.item() * len(y); n_seen += len(y)
                n_ok += int((out.argmax(1) == y).sum().item())

            yv, pv = predict(model, val_df, arch)
            val = score(yv, pv)
            improved = val["macro_f1"] > state["best_f1"]
            if improved:
                state["best_f1"], state["best_epoch"], state["bad"] = val["macro_f1"], epoch, 0
                torch.save(model.state_dict(), run_dir / "best.pt")
            else:
                state["bad"] += 1
            sec = time.time() - t_ep
            EPOCH_SECONDS.setdefault(arch, []).append(sec)
            state["history"].append({"epoch": epoch, "lr": lr, "train_loss": loss_sum / max(n_seen, 1),
                                     "train_acc": n_ok / max(n_seen, 1), "val_acc": val["accuracy"],
                                     "val_macro_f1": val["macro_f1"], "seconds": round(sec, 1)})
            state["epoch"] = epoch + 1
            save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                         "state": state}, last)
            print(f"[{run_name}] epoch {epoch + 1}/{n_epochs} lr={lr:.2e} train_loss={loss_sum / max(n_seen, 1):.3f} "
                  f"val_f1={val['macro_f1']:.4f} best={state['best_f1']:.4f}@{state['best_epoch'] + 1} {sec:.0f}s")
            if state["bad"] >= PATIENCE:
                print(f"[{run_name}] early stop at epoch {epoch + 1} (no val macro-F1 gain for {PATIENCE} epochs)")
                break

        pd.DataFrame(state["history"]).to_csv(run_dir / "history.csv", index=False)
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=DEVICE, weights_only=True))
        out = {"run": run_name, "arch": arch, "train_datasets": sorted(train_df["dataset"].unique().tolist()),
               "epochs_run": state["epoch"], "best_epoch": state["best_epoch"] + 1,
               "best_val_macro_f1": state["best_f1"], "tests": {}}
        for name, tdf in tests.items():
            y, p = predict(model, tdf, arch)
            out["tests"][name] = score(y, p)
            save_confusion(y, p, run_dir / f"confusion_{name}.png", f"{run_name}: test {name}")
        (run_dir / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        (run_dir / "done.json").write_text(json.dumps({"finished": time.strftime("%Y-%m-%d %H:%M:%S")}), encoding="utf-8")
        print(f"[{run_name}] done")
        return out
    except torch.cuda.OutOfMemoryError:
        if BATCH_CURRENT <= 8:
            raise
        BATCH_CURRENT //= 2
        torch.cuda.empty_cache()
        print(f"[{run_name}] out of memory, retrying with batch {BATCH_CURRENT} (resumes from last.pt)")
        return run_experiment(run_name, arch, train_df, val_df, tests, n_epochs)

def avg_epoch_seconds(arch):
    v = EPOCH_SECONDS.get(arch, [])
    return float(np.mean(v)) if v else None
''')

code(r'''
# ---- 8. E2: combined training, each dataset's test split reported separately ----------------
train_all = splits[splits["split"] == "train"]
val_all = splits[splits["split"] == "val"]
test_by_ds = {ds: splits[(splits["split"] == "test") & (splits["dataset"] == ds)] for ds in DATASETS}

E2 = {}
for arch, n_ep in [("cnn", EP_CNN), ("resnet18", EP_RES)]:
    E2[arch] = run_experiment(f"E2_{arch}", arch, train_all, val_all, test_by_ds, n_ep)

rows = []
for arch in E2:
    for ds in DATASETS:
        t = E2[arch]["tests"][ds]
        rows.append({"model": arch, "dataset": ds, "accuracy": t["accuracy"], "macro_f1": t["macro_f1"], "n_test": t["n"]})
cmp = pd.DataFrame(rows)
mean_rows = cmp.groupby("model")[["accuracy", "macro_f1"]].mean().reset_index()
mean_rows["dataset"] = "mean"; mean_rows["n_test"] = np.nan
cmp = pd.concat([cmp, mean_rows[cmp.columns]], ignore_index=True)
cmp.to_csv(RESULTS_DIR / "compare_E2.csv", index=False)
print(cmp.to_string(index=False))

mean_f1 = {a: float(mean_rows.loc[mean_rows["model"] == a, "macro_f1"].iloc[0]) for a in E2}
WINNER = max(mean_f1, key=mean_f1.get)
(RESULTS_DIR / "choice.json").write_text(json.dumps({"winner": WINNER, "mean_macro_f1": mean_f1}, indent=2))
print("E3 uses:", WINNER, "(higher mean macro-F1 across the four E2 test splits)")
''')

code(r'''
# ---- 9. E3: leave-one-dataset-out with the winning model ------------------------------------
E3_SKIPPED, E3_DONE = [], []
if not RUN_E3:
    print("E3 not run in this configuration")
else:
    e2_ep = avg_epoch_seconds(WINNER) or 0.0
    for held in DATASETS:
        run_name = f"E3_{WINNER}_holdout_{held}"
        train_df = splits[(splits["split"] == "train") & (splits["dataset"] != held)]
        val_df = splits[(splits["split"] == "val") & (splits["dataset"] != held)]
        test_df = splits[splits["dataset"] == held]  # all rows of the held-out dataset
        done = (RESULTS_DIR / run_name / "done.json").exists()
        if not done and e2_ep:
            est = e2_ep * E3_EPOCHS * len(train_df) / max(1, len(train_all))
            elapsed = time.time() - T0
            if elapsed + est > MAX_HOURS * 3600:
                E3_SKIPPED.append(held)
                print(f"skip {run_name}: elapsed {elapsed / 3600:.2f} h + estimated {est / 3600:.2f} h > MAX_HOURS={MAX_HOURS}")
                continue
        run_experiment(run_name, WINNER, train_df, val_df, {held: test_df}, E3_EPOCHS)
        E3_DONE.append(held)

    e3_rows = []
    for held in DATASETS:
        run_name = f"E3_{WINNER}_holdout_{held}"
        m_path = RESULTS_DIR / run_name / "metrics.json"
        in_ds = E2[WINNER]["tests"][held]["macro_f1"]
        if m_path.exists():
            m = json.loads(m_path.read_text())["tests"][held]
            e3_rows.append({"held_out": held, "status": "done", "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                            "in_dataset_E2_macro_f1": in_ds})
        else:
            e3_rows.append({"held_out": held, "status": "skipped", "accuracy": np.nan, "macro_f1": np.nan,
                            "in_dataset_E2_macro_f1": in_ds})
    e3 = pd.DataFrame(e3_rows)
    e3.to_csv(RESULTS_DIR / "E3_summary.csv", index=False)
    print(e3.to_string(index=False))
    if E3_SKIPPED:
        print("E3 skipped (time budget):", E3_SKIPPED)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(e3))
    ax.bar(x - 0.2, e3["in_dataset_E2_macro_f1"], 0.4, label="in-dataset (E2)")
    ax.bar(x + 0.2, e3["macro_f1"].fillna(0), 0.4, label="held-out (E3)")
    ax.set_xticks(x, e3["held_out"]); ax.set_ylim(0, 1); ax.set_ylabel("macro-F1")
    ax.set_title(f"Leave-one-dataset-out, {WINNER}"); ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "E3_summary.png", dpi=120); plt.close(fig)
''')

code(r'''
# ---- 10. Final outputs ----------------------------------------------------------------------
shutil.copy(UPLOAD_DIR / "classes.json", RESULTS_DIR / "classes.json")
shutil.copy(RESULTS_DIR / f"E2_{WINNER}" / "best.pt", RESULTS_DIR / "best_model.pt")
print("ALL DONE")
print("Output folder:", RESULTS_DIR)
for p in sorted(RESULTS_DIR.rglob("*")):
    if p.is_file() and p.suffix not in (".pt", ".tmp"):
        print("  ", p.relative_to(RESULTS_DIR))
print(f"total time {(time.time() - T0) / 3600:.2f} h")
''')


def to_lines(text):
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def build():
    cells = []
    for kind, text in CELLS:
        base = {"cell_type": kind, "metadata": {}, "source": to_lines(text)}
        if kind == "code":
            base.update({"execution_count": None, "outputs": []})
        cells.append(base)
    nb = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "language_info": {"name": "python"}},
        "cells": cells,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT, "cells:", len(cells))


if __name__ == "__main__":
    build()
