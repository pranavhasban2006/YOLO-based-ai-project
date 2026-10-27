r"""Faculty protocol for WESAD with subject-independent splits, retraining the actual hybrid model every time.

Why separate from protocol.py: WESAD windows of one person are strongly correlated, so any split that puts the
same subject on both sides measures re-identification of a known person (~0.95-1.0 F1), not emotion recognition
for a new person. Here every split is by SUBJECT:
    hold-out 80:20 and 70:30  -> GroupShuffleSplit over the 15 subjects (12/3 and 10/5 subjects)
    10-fold CV                -> GroupKFold over the 15 subjects (each subject is tested exactly once)
each with and without SMOTE (on the flattened training windows only). Early stopping uses one held-out training
subject; L2 via AdamW weight decay plus an L1 penalty. Output is protocol.py-compatible JSON, so
`protocol.py --summarize` includes it.

Usage (repo root): project\.venv-train\Scripts\python project\code\wesad_protocol.py [--models cnn_lstm ...]
Writes project/reports/protocol/wesad_<model>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
import train_wesad as tw  # noqa: E402
from protocol import scores  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "project" / "reports" / "protocol"
SEED, MAX_EPOCHS, PATIENCE, L1 = 0, 15, 4, 1e-6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model, x):
    model.eval()
    out = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(x), batch_size=256):
            out.append(model(xb.to(DEVICE)).argmax(1).cpu())
    return torch.cat(out).numpy()


def fit(name, xtr, ytr, xval, yval):
    torch.manual_seed(SEED)
    model = tw.BUILDERS[name]().to(DEVICE)
    counts = np.bincount(ytr, minlength=3)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)), dtype=torch.float32,
                                                 device=DEVICE))
    tl = DataLoader(TensorDataset(xtr, torch.from_numpy(ytr)), batch_size=64, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)                    # L2
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=MAX_EPOCHS * len(tl))
    best, state, wait, ep = -1.0, None, 0, 0
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            xb = xb + 0.05 * torch.randn_like(xb)
            loss = ce(model(xb), yb) + L1 * sum(p.abs().sum() for n, p in model.named_parameters()
                                                if n.endswith("weight"))
            if name == "ae_mlp":
                loss = loss + nn.functional.mse_loss(model.recon, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        f1 = f1_score(yval, predict(model, xval), average="macro")
        if f1 > best:
            best, state, wait = f1, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:                                                              # early stopping
                break
    model.load_state_dict(state)
    return model, ep


def run(name, x, y, subj, tr, te, smote):
    """Train on subjects tr (one of them held out for early stopping), test on subjects te."""
    tr_subjects = np.unique(subj[tr])
    val_subject = np.random.default_rng(SEED).choice(tr_subjects)
    fit_idx, val_idx = tr[subj[tr] != val_subject], tr[subj[tr] == val_subject]
    xfit, yfit = x[fit_idx], y[fit_idx]
    if smote:
        flat, yfit = SMOTE(random_state=SEED).fit_resample(xfit.reshape(len(xfit), -1), yfit)
        xfit = flat.reshape(-1, *x.shape[1:])
    model, epochs = fit(name, torch.from_numpy(xfit), yfit, torch.from_numpy(x[val_idx]), y[val_idx])
    t0 = time.perf_counter()
    pte = predict(model, torch.from_numpy(x[te]))
    ms = (time.perf_counter() - t0) / len(te) * 1000
    tr_acc = float((predict(model, torch.from_numpy(x[fit_idx])) == y[fit_idx]).mean())
    val_acc = float((predict(model, torch.from_numpy(x[val_idx])) == y[val_idx]).mean())
    gap = tr_acc - val_acc
    fit_state = "overfitting" if gap > 0.10 else "underfitting" if tr_acc < 0.5 else "good fit"
    return y[te], pte, {"train_acc": tr_acc, "val_acc": val_acc, "train_val_gap": gap, "fit": fit_state,
                        "epochs_until_early_stop": epochs, "train_size_after_smote": int(len(yfit)),
                        "test_subjects": sorted(np.unique(subj[te]).tolist()), "ms_per_window": ms}


def evaluate_model(name, x, y, subj, budget_s):
    """Resumable: progress is saved after every run/fold, so a time-limited call continues where it stopped."""
    path = OUT / f"wesad_{name}.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    if res.get("grouped_splits") != "by subject":                       # older, leaky window-level results
        res = {"model": name, "rows": len(y), "subjects": len(np.unique(subj)), "grouped_splits": "by subject",
               "class_counts": np.bincount(y).tolist()}
    save = lambda: (OUT.mkdir(parents=True, exist_ok=True), path.write_text(json.dumps(res, indent=2, default=float)))
    t_start = time.time()
    out_of_time = lambda: time.time() - t_start > budget_s
    for test_size in (0.2, 0.3):
        tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=SEED).split(x, y, subj))
        for smote in (False, True):
            key = f"holdout_{round((1 - test_size) * 100)}_{round(test_size * 100)}_{'smote' if smote else 'no_smote'}"
            if key in res:
                continue
            if out_of_time():
                return False
            yt, pt, info = run(name, x, y, subj, tr, te, smote)
            res[key] = {**scores(yt, pt, 3), **info}
            save()
            print(f"{name} {key}: acc {res[key]['accuracy']:.4f} macroF1 {res[key]['macro_f1']:.4f} "
                  f"({info['fit']}, test {info['test_subjects']})", flush=True)
    for smote in (False, True):
        key = f"cv10_{'smote' if smote else 'no_smote'}"
        if key in res:
            continue
        part = res.setdefault(f"{key}_partial", [])
        for i, (tr, te) in enumerate(GroupKFold(10).split(x, y, subj)):
            if i < len(part):
                continue
            if out_of_time():
                return False
            yt, pt, info = run(name, x, y, subj, tr, te, smote)
            part.append({**{k: v for k, v in scores(yt, pt, 3).items() if k != "per_class"}, **info,
                         "y": yt.tolist(), "p": pt.tolist()})
            save()
            print(f"{name} {key} fold {i + 1}/10: macroF1 {part[-1]['macro_f1']:.4f} test {info['test_subjects']}",
                  flush=True)
        accs, f1s = [f["accuracy"] for f in part], [f["macro_f1"] for f in part]
        ys = np.concatenate([f.pop("y") for f in part])
        ps = np.concatenate([f.pop("p") for f in part])
        res[key] = {"accuracy_mean": np.mean(accs), "accuracy_std": np.std(accs), "macro_f1_mean": np.mean(f1s),
                    "macro_f1_std": np.std(f1s), "pooled": scores(ys, ps, 3), "folds": part}
        del res[f"{key}_partial"]
        save()
        print(f"{name} {key}: acc {np.mean(accs):.4f}±{np.std(accs):.4f} macroF1 {np.mean(f1s):.4f}±{np.std(f1s):.4f}",
              flush=True)
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(tw.BUILDERS))
    ap.add_argument("--budget", type=float, default=1e9, help="seconds; stop cleanly (resumable) after this")
    a = ap.parse_args()
    d = np.load(tw.CACHE)
    x, y, subj = d["x"].astype(np.float32), d["y"], d["subj"]
    t0 = time.time()
    for m in a.models:
        if not evaluate_model(m, x, y, subj, a.budget - (time.time() - t0)):
            print("PAUSED (time budget) - run again to resume", flush=True)
            break
    else:
        print("ALL DONE", flush=True)
