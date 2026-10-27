r"""Physiological emotion/affect models on WESAD (chest RespiBAN), faculty hybrid list.

Task: 3-class affect state from 60 s windows: baseline / stress / amusement (WESAD labels 1/2/3).
Split is subject-independent and fixed (same for every model): no subject appears in two splits.

Usage (repo root, CUDA training venv):
    project\.venv-train\Scripts\python -I project\code\train_wesad.py --prepare     # once: pkl -> npz cache
    project\.venv-train\Scripts\python -I project\code\train_wesad.py --model cnn_lstm
Models: cnn_lstm, cnn_bilstm, cnn_gru, ae_mlp, transformer_bilstm
Writes models/wesad_<model>.pt and project/reports/wesad_<model>_metrics.json.

Note: WESAD ships as Python pickles. Only load the official release; unpickling runs code from the file.
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "project" / "data" / "wesad" / "WESAD"
CACHE = ROOT / "project" / "data" / "wesad_windows.npz"
CLASSES = ["baseline", "stress", "amusement"]
CHANNELS = ["ECG", "EDA", "EMG", "Resp", "Temp", "ACC_x", "ACC_y", "ACC_z"]
FS_RAW, FS = 700, 35                      # RespiBAN 700 Hz, mean-pooled to 35 Hz
WIN, STRIDE = 60 * FS, 5 * FS             # 60 s windows, 5 s hop
SPLIT = {"test": ["S2", "S9", "S16"], "val": ["S5", "S13"]}   # remaining 10 subjects train


def prepare():
    xs, ys, subj = [], [], []
    for d in sorted(RAW.glob("S*"), key=lambda p: int(p.name[1:])):
        with open(d / f"{d.name}.pkl", "rb") as f:
            raw = pickle.load(f, encoding="latin1")
        c = raw["signal"]["chest"]
        sig = np.concatenate([c["ECG"], c["EDA"], c["EMG"], c["Resp"], c["Temp"], c["ACC"]], 1).astype(np.float32)
        lab = raw["label"]
        f = FS_RAW // FS
        n = len(sig) // f
        sig = sig[:n * f].reshape(n, f, -1).mean(1)
        lab = lab[:n * f].reshape(n, f)[:, 0]
        sig = (sig - sig.mean(0)) / (sig.std(0) + 1e-6)        # per-subject z-score, removes person offsets
        for k, cls in enumerate((1, 2, 3)):
            idx = np.flatnonzero(lab == cls)
            # contiguous segments of this condition, windowed inside each segment only
            for seg in np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1):
                for s in range(seg[0], seg[-1] + 2 - WIN, STRIDE):
                    xs.append(sig[s:s + WIN].T)
                    ys.append(k)
                    subj.append(d.name)
        print(f"{d.name}: {len(sig)} samples @ {FS} Hz, windows so far {len(xs)}", flush=True)
    np.savez_compressed(CACHE, x=np.stack(xs).astype(np.float16), y=np.array(ys), subj=np.array(subj))
    print(f"saved {CACHE}")


# ---- Faculty hybrid models; input B x 8 x 2100.

def conv_stem(c_in=8, c=64):
    """1D-CNN front end: 3 conv blocks, x8 downsampling -> B x 128 x T/8."""
    def b(i, o):
        return nn.Sequential(nn.Conv1d(i, o, 7, padding=3), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2))
    return nn.Sequential(b(c_in, c), b(c, c * 2), b(c * 2, c * 2))


class CNNRNN(nn.Module):
    def __init__(self, rnn="lstm", bidir=False, n=3, h=128):
        super().__init__()
        self.cnn = conv_stem()
        self.rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn](128, h, batch_first=True, bidirectional=bidir)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(h * (2 if bidir else 1), n))

    def forward(self, x):
        s, _ = self.rnn(self.cnn(x).transpose(1, 2))
        return self.head(s.mean(1))


class AutoencoderMLP(nn.Module):
    """Conv autoencoder learns a compressed code; MLP classifies the code. Trained jointly (recon + CE)."""
    def __init__(self, n=3, z=64):
        super().__init__()
        self.enc = nn.Sequential(conv_stem(), nn.Conv1d(128, z, 1))
        self.dec = nn.Sequential(nn.Upsample(scale_factor=8), nn.Conv1d(z, 64, 7, padding=3), nn.ReLU(),
                                 nn.Conv1d(64, 8, 7, padding=3))
        self.mlp = nn.Sequential(nn.Linear(z * 2, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, n))
        self.recon = None

    def forward(self, x):
        code = self.enc(x)
        r = self.dec(code)                                       # 2100 / 8 rounds down: pad back to length
        self.recon = nn.functional.pad(r, (0, x.shape[-1] - r.shape[-1]))[..., :x.shape[-1]]
        return self.mlp(torch.cat([code.mean(-1), code.amax(-1)], 1))


class TransformerBiLSTM(nn.Module):
    def __init__(self, n=3, d=128):
        super().__init__()
        self.cnn = conv_stem()                                   # tokens: 262 steps of 128 features
        self.pos = nn.Parameter(torch.randn(1, WIN // 8, d) * 0.02)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, d * 4, 0.1, batch_first=True,
                                                                   norm_first=True), 2)
        self.lstm = nn.LSTM(d, d, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * d, n))

    def forward(self, x):
        t = self.cnn(x).transpose(1, 2)
        s, _ = self.lstm(self.tf(t + self.pos[:, :t.shape[1]]))
        return self.head(s.mean(1))


BUILDERS = {
    "cnn_lstm": lambda: CNNRNN("lstm"),
    "cnn_bilstm": lambda: CNNRNN("lstm", bidir=True),
    "cnn_gru": lambda: CNNRNN("gru"),
    "ae_mlp": AutoencoderMLP,
    "transformer_bilstm": TransformerBiLSTM,
}


def loaders(batch):
    d = np.load(CACHE)
    x, y, subj = torch.from_numpy(d["x"].astype(np.float32)), torch.from_numpy(d["y"]), d["subj"]
    held = set(SPLIT["test"]) | set(SPLIT["val"])
    m = {k: np.isin(subj, v) for k, v in SPLIT.items()}
    m["train"] = ~np.isin(subj, list(held))
    mk = lambda k, sh: DataLoader(TensorDataset(x[m[k]], y[m[k]]), batch_size=batch, shuffle=sh)
    counts = np.bincount(y[m["train"]].numpy(), minlength=3)
    return mk("train", True), mk("val", False), mk("test", False), counts


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(model(x.to(device)).argmax(1).cpu())
            ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def train(name, epochs, batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    tl, vl, testl, counts = loaders(batch)
    model = BUILDERS[name]().to(device)
    weights = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32, device=device)
    ce = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * len(tl))
    out = ROOT / "models" / f"wesad_{name}.pt"
    best = -1.0
    print(f"{name}: {len(tl.dataset)} train / {len(vl.dataset)} val / {len(testl.dataset)} test windows, "
          f"class counts {counts.tolist()} on {device}", flush=True)
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            x = x + 0.05 * torch.randn_like(x)                   # light jitter augmentation
            loss = ce(model(x), y)
            if name == "ae_mlp":
                loss = loss + nn.functional.mse_loss(model.recon, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        yv, pv = evaluate(model, vl, device)
        f1 = f1_score(yv, pv, average="macro")
        if f1 > best:
            best = f1
            torch.save(model.state_dict(), out)
        print(f"epoch {ep}/{epochs} val_acc {accuracy_score(yv, pv):.4f} val_macroF1 {f1:.4f} "
              f"({time.time() - t0:.0f}s){' *' if f1 == best else ''}", flush=True)

    model.load_state_dict(torch.load(out, weights_only=True))
    yt, pt = evaluate(model, testl, device)
    p, r, f, s = precision_recall_fscore_support(yt, pt, labels=range(3), zero_division=0)
    timing = {}
    x1 = testl.dataset[0][0].unsqueeze(0)
    for dev in ["cpu"] + (["cuda"] if device.type == "cuda" else []):
        m = model.to(dev).eval()
        xd = x1.to(dev)
        with torch.no_grad():
            for _ in range(5):
                m(xd)
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(50):
                m(xd)
            if dev == "cuda":
                torch.cuda.synchronize()
        timing[f"{dev}_ms_per_window"] = round((time.perf_counter() - t0) / 50 * 1000, 2)
    metrics = {
        "model": name, "dataset": "WESAD chest, 60 s windows, subject-independent split", "split": SPLIT,
        "epochs": epochs, "test_size": len(yt),
        "accuracy": round(accuracy_score(yt, pt), 4), "macro_f1": round(f1_score(yt, pt, average="macro"), 4),
        "per_class": {c: {"precision": round(p[i], 4), "recall": round(r[i], 4), "f1": round(f[i], 4),
                          "support": int(s[i])} for i, c in enumerate(CLASSES)},
        "confusion_matrix": {"labels": CLASSES,
                             "rows_true_cols_pred": confusion_matrix(yt, pt, labels=range(3)).tolist()},
        "inference_time": timing,
    }
    rep = ROOT / "project" / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / f"wesad_{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("accuracy", "macro_f1", "inference_time")}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--model", choices=BUILDERS)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    if a.prepare:
        prepare()
    if a.model:
        train(a.model, a.epochs, a.batch)
