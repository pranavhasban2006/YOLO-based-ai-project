r"""Faculty evaluation protocol on a feature CSV (f0..fN, label), exactly as in the flowchart:

    load CSV -> preprocess (drop bad rows, z-score fitted on training part only) -> split
      A) hold-out 80:20 and 70:30, each without SMOTE and with SMOTE on the training data only
      B) 10-fold stratified cross-validation, without SMOTE and with SMOTE on the training folds only
    -> train classifier (MLP) with overfitting control: early stopping on a validation slice, L1 + L2
       regularisation (both together = elastic net) -> over/underfitting check -> test -> best-model selection.

Splits use a fixed seed, and every CSV of a modality has the same rows in the same order, so every model is
compared on identical splits and folds.

Usage (repo root):
    project\.venv-train\Scripts\python project\code\protocol.py project\features\face_resnet18.csv
    project\.venv-train\Scripts\python project\code\protocol.py --summarize      # best model per modality
Writes project/reports/protocol/<csv name>.json and project/reports/protocol/summary.csv
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "project" / "reports" / "protocol"
SEED = 0
L1, L2 = 1e-5, 1e-3            # L1 penalty on weights (added to loss) and L2 (AdamW weight decay)
PATIENCE, MAX_EPOCHS = 10, 200


def train_mlp(xtr, ytr, xval, yval, k):
    torch.manual_seed(SEED)
    model = nn.Sequential(nn.Linear(xtr.shape[1], 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, k))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=L2)
    ce = nn.CrossEntropyLoss()
    xt, yt = torch.tensor(xtr, dtype=torch.float32), torch.tensor(ytr)
    xv = torch.tensor(xval, dtype=torch.float32)
    best, best_state, wait, epochs = -1.0, None, 0, 0
    for epochs in range(1, MAX_EPOCHS + 1):
        model.train()
        for idx in torch.randperm(len(yt)).split(256):
            loss = ce(model(xt[idx]), yt[idx]) + L1 * sum(p.abs().sum() for n, p in model.named_parameters()
                                                         if n.endswith("weight"))
            opt.zero_grad()
            loss.backward()
            opt.step()
        f1 = f1_score(yval, predict(model, xv), average="macro")
        if f1 > best:
            best, best_state, wait = f1, {n: p.clone() for n, p in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:                   # early stopping
                break
    model.load_state_dict(best_state)
    return model, epochs


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float32)).argmax(1).numpy()


def run(xtr, ytr, xte, yte, k, smote):
    """One train/test experiment: validation slice for early stopping, scaler + SMOTE fitted on training only."""
    xfit, xval, yfit, yval = train_test_split(xtr, ytr, test_size=0.1, stratify=ytr, random_state=SEED)
    sc = StandardScaler().fit(xfit)
    xfit, xval, xte = sc.transform(xfit), sc.transform(xval), sc.transform(xte)
    xfit_orig, yfit_orig = xfit, yfit
    if smote:
        xfit, yfit = SMOTE(random_state=SEED).fit_resample(xfit, yfit)
    model, epochs = train_mlp(xfit, yfit, xval, yval, k)
    t0 = time.perf_counter()
    pte = predict(model, xte)
    ms = (time.perf_counter() - t0) / len(xte) * 1000
    tr_acc, val_acc = accuracy_score(yfit_orig, predict(model, xfit_orig)), accuracy_score(yval, predict(model, xval))
    gap = tr_acc - val_acc
    fit = "overfitting" if gap > 0.10 else "underfitting" if tr_acc < 0.5 else "good fit"
    return yte, pte, {"train_acc": tr_acc, "val_acc": val_acc, "train_val_gap": gap, "fit": fit,
                      "epochs_until_early_stop": epochs, "train_size_after_smote": int(len(yfit)),
                      "head_ms_per_sample": ms}


def scores(y, p, k):
    pr, rc, f, s = precision_recall_fscore_support(y, p, labels=range(k), zero_division=0)
    return {"accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro"),
            "macro_precision": pr.mean(), "macro_recall": rc.mean(),
            "per_class": {"precision": pr.tolist(), "recall": rc.tolist(), "f1": f.tolist(), "support": s.tolist()},
            "confusion_matrix": confusion_matrix(y, p, labels=range(k)).tolist()}


def evaluate_csv(path):
    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()                     # preprocessing: drop bad rows
    g = df.pop("group").to_numpy() if "group" in df else None   # optional grouping (e.g. overlapping windows)
    x, y = df.drop(columns="label").to_numpy(np.float32), df["label"].to_numpy()
    k = int(y.max()) + 1
    res = {"csv": Path(path).name, "rows": len(y), "grouped_splits": g is not None, "features": x.shape[1], "class_counts": np.bincount(y).tolist()}
    for test_size in (0.2, 0.3):
        if g is None:
            xtr, xte, ytr, yte = train_test_split(x, y, test_size=test_size, stratify=y, random_state=SEED)
        else:
            tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=SEED).split(x, y, g))
            xtr, xte, ytr, yte = x[tr], x[te], y[tr], y[te]
        for smote in (False, True):
            name = f"holdout_{round((1 - test_size) * 100)}_{round(test_size * 100)}_{'smote' if smote else 'no_smote'}"
            yt, pt, info = run(xtr, ytr, xte, yte, k, smote)
            res[name] = {**scores(yt, pt, k), **info}
            print(f"{name}: acc {res[name]['accuracy']:.4f} macroF1 {res[name]['macro_f1']:.4f} ({info['fit']})",
                  flush=True)
    for smote in (False, True):
        name = f"cv10_{'smote' if smote else 'no_smote'}"
        folds, ys, ps = [], [], []
        cv = (StratifiedKFold(10, shuffle=True, random_state=SEED) if g is None
              else StratifiedGroupKFold(10, shuffle=True, random_state=SEED))
        for tr, te in cv.split(x, y, g):
            yt, pt, info = run(x[tr], y[tr], x[te], y[te], k, smote)
            folds.append({**{m: v for m, v in scores(yt, pt, k).items() if m != "per_class"}, **info})
            ys.append(yt)
            ps.append(pt)
        accs, f1s = [f["accuracy"] for f in folds], [f["macro_f1"] for f in folds]
        res[name] = {"accuracy_mean": np.mean(accs), "accuracy_std": np.std(accs),
                     "macro_f1_mean": np.mean(f1s), "macro_f1_std": np.std(f1s),
                     "pooled": scores(np.concatenate(ys), np.concatenate(ps), k), "folds": folds}
        print(f"{name}: acc {np.mean(accs):.4f}±{np.std(accs):.4f} macroF1 {np.mean(f1s):.4f}±{np.std(f1s):.4f}",
              flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{Path(path).stem}.json").write_text(json.dumps(res, indent=2, default=float))


def summarize():
    rows = []
    for f in sorted(OUT.glob("*.json")):
        r = json.loads(f.read_text())
        modality, model = f.stem.split("_", 1)
        if modality == "text":                       # each text dataset is its own comparison
            model, dataset = model.rsplit("_", 1)
            modality = f"text_{dataset}"
        row = {"modality": modality, "model": model}
        for name in ("holdout_80_20_no_smote", "holdout_80_20_smote", "holdout_70_30_no_smote", "holdout_70_30_smote"):
            row[f"{name}_acc"], row[f"{name}_f1"] = round(r[name]["accuracy"], 4), round(r[name]["macro_f1"], 4)
        for name in ("cv10_no_smote", "cv10_smote"):
            row[f"{name}_acc"] = f"{r[name]['accuracy_mean']:.4f}±{r[name]['accuracy_std']:.4f}"
            row[f"{name}_f1"] = f"{r[name]['macro_f1_mean']:.4f}±{r[name]['macro_f1_std']:.4f}"
            row[f"{name}_f1_mean"] = r[name]["macro_f1_mean"]
        rows.append(row)
    df = pd.DataFrame(rows)
    # Best model per modality: highest mean macro-F1 under 10-fold CV with SMOTE (the most robust estimate).
    df["best_in_modality"] = df["cv10_smote_f1_mean"] == df.groupby("modality")["cv10_smote_f1_mean"].transform("max")
    df.drop(columns=[c for c in df if c.endswith("_f1_mean")]).to_csv(OUT / "summary.csv", index=False)
    print(df[["modality", "model", "cv10_smote_f1", "best_in_modality"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="*")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    for c in a.csv:
        evaluate_csv(c)
    if a.summarize:
        summarize()
