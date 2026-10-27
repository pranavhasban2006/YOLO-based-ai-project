r"""Text-emotion models, faculty hybrid list: transformer encoder + BiLSTM / CNN / attention heads, fine-tuned end to end.

Datasets (official splits, identical for every model):
    goemotions  GoEmotions mapped to the app's 7 classes with the standard Ekman grouping; only comments whose
                labels all fall in one group are kept (multi-group comments are ambiguous for single-label training).
    emotion     dair-ai Emotion, 6 classes (sadness, joy, love, anger, fear, surprise).

Usage (repo root, CUDA training venv):
    project\.venv-train\Scripts\python project\code\train_text.py --model distilbert_cnn --dataset goemotions
Models: roberta_bilstm_attn, bert_bilstm_attn, deberta_attn, distilbert_cnn, bert_cnn_bilstm
Writes models/text_<model>_<dataset>.pt and project/reports/text_<model>_<dataset>_metrics.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "project" / "data" / "text"
MAX_LEN = 64

APP_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
GO_NAMES = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
            "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
            "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
            "remorse", "sadness", "surprise", "neutral"]
EKMAN = {"angry": ["anger", "annoyance", "disapproval"], "disgust": ["disgust"], "fear": ["fear", "nervousness"],
         "happy": ["joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism", "relief", "pride",
                   "admiration", "desire", "caring"],
         "sad": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
         "surprise": ["surprise", "realization", "confusion", "curiosity"], "neutral": ["neutral"]}
GO_TO_APP = {GO_NAMES.index(g): APP_CLASSES.index(c) for c, gs in EKMAN.items() for g in gs}
DATASETS = {"goemotions": APP_CLASSES, "emotion": ["sadness", "joy", "love", "anger", "fear", "surprise"]}
ENCODERS = {"roberta": "roberta-base", "bert": "bert-base-uncased", "deberta": "microsoft/deberta-v3-base",
            "distilbert": "distilbert-base-uncased"}


def load_split(dataset, split):
    t = pq.read_table(DATA / dataset / f"{split}.parquet").to_pydict()
    if dataset == "emotion":
        return t["text"], t["label"]
    texts, ys = [], []
    for text, labels in zip(t["text"], t["labels"]):
        groups = {GO_TO_APP[l] for l in labels}
        if len(groups) == 1:
            texts.append(text)
            ys.append(groups.pop())
    return texts, ys


# ---- Faculty hybrid heads on top of a pretrained encoder's token states (B,T,H).

class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, s, mask):
        a = self.score(s).squeeze(-1).masked_fill(mask == 0, -1e4)
        return (s * torch.softmax(a, 1).unsqueeze(-1)).sum(1)


class BiLSTMAttnHead(nn.Module):
    def __init__(self, h, n, k=256):
        super().__init__()
        self.lstm = nn.LSTM(h, k, batch_first=True, bidirectional=True)
        self.pool, self.out = AttnPool(2 * k), nn.Linear(2 * k, n)

    def forward(self, s, mask):
        return self.out(self.pool(self.lstm(s)[0], mask))


class AttnHead(nn.Module):
    def __init__(self, h, n):
        super().__init__()
        self.pool, self.out = AttnPool(h), nn.Linear(h, n)

    def forward(self, s, mask):
        return self.out(self.pool(s, mask))


class CNNHead(nn.Module):
    """Parallel 1D convolutions (kernel 2/3/4: local phrase patterns) + max-over-time pooling."""
    def __init__(self, h, n, k=128):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv1d(h, k, w, padding=w // 2) for w in (2, 3, 4))
        self.out = nn.Linear(3 * k, n)

    def forward(self, s, mask):
        x = (s * mask.unsqueeze(-1)).transpose(1, 2)
        return self.out(torch.cat([torch.relu(c(x)).amax(-1) for c in self.convs], 1))


class CNNBiLSTMHead(nn.Module):
    def __init__(self, h, n, k=128):
        super().__init__()
        self.conv = nn.Conv1d(h, k * 2, 3, padding=1)
        self.lstm = nn.LSTM(k * 2, k, batch_first=True, bidirectional=True)
        self.pool, self.out = AttnPool(2 * k), nn.Linear(2 * k, n)

    def forward(self, s, mask):
        c = torch.relu(self.conv((s * mask.unsqueeze(-1)).transpose(1, 2))).transpose(1, 2)
        return self.out(self.pool(self.lstm(c)[0], mask))


MODELS = {  # name -> (encoder, head)
    "roberta_bilstm_attn": ("roberta", BiLSTMAttnHead),
    "bert_bilstm_attn": ("bert", BiLSTMAttnHead),
    "deberta_attn": ("deberta", AttnHead),
    "distilbert_cnn": ("distilbert", CNNHead),
    "bert_cnn_bilstm": ("bert", CNNBiLSTMHead),
}


class TextModel(nn.Module):
    def __init__(self, name, n):
        super().__init__()
        enc, head = MODELS[name]
        # float32 master weights: AMP scales fp32 grads (deberta-v3 otherwise loads in fp16 and AMP refuses)
        self.encoder = AutoModel.from_pretrained(ENCODERS[enc], torch_dtype=torch.float32)
        self.drop = nn.Dropout(0.2)
        self.head = head(self.encoder.config.hidden_size, n)

    def forward(self, input_ids, attention_mask):
        s = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(self.drop(s), attention_mask)


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
        for b in loader:
            ps.append(model(b["input_ids"].to(device), b["attention_mask"].to(device)).argmax(1).cpu())
            ys.append(b["y"])
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--dataset", choices=DATASETS, default="goemotions")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval-only", action="store_true", help="skip training; test the saved checkpoint")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    classes = DATASETS[a.dataset]
    tok = AutoTokenizer.from_pretrained(ENCODERS[MODELS[a.model][0]])

    def loader(split, shuffle):
        texts, ys = load_split(a.dataset, split)
        enc = tok(texts, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
        rows = [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
                 "y": torch.tensor(ys[i])} for i in range(len(ys))]
        return DataLoader(rows, batch_size=a.batch, shuffle=shuffle), np.array(ys)

    tl, ytr = loader("train", True)
    vl, _ = loader("validation", False)
    testl, _ = loader("test", False)
    counts = np.bincount(ytr, minlength=len(classes))
    weights = torch.tensor(np.sqrt(counts.sum() / (len(classes) * np.maximum(counts, 1))),
                           dtype=torch.float32, device=device)   # sqrt-damped: GoEmotions-Ekman is very skewed

    model = TextModel(a.model, len(classes)).to(device)
    enc_params = list(model.encoder.parameters())
    head_params = [p for n_, p in model.named_parameters() if not n_.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": 2e-5}, {"params": head_params, "lr": 1e-3}],
                            weight_decay=1e-2)
    total = a.epochs * len(tl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / (0.1 * total)) * max(0.0, (total - s) / (0.9 * total)))
    ce = nn.CrossEntropyLoss(weight=weights)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    out = ROOT / "models" / f"text_{a.model}_{a.dataset}.pt"
    best = -1.0
    print(f"{a.model} on {a.dataset}: {len(tl.dataset)} train / {len(vl.dataset)} val / {len(testl.dataset)} test "
          f"on {device}, class counts {counts.tolist()}", flush=True)
    for ep in range(1, 0 if a.eval_only else a.epochs + 1):
        model.train()
        t0 = time.time()
        for b in tl:
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = ce(model(b["input_ids"].to(device), b["attention_mask"].to(device)), b["y"].to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        yv, pv = evaluate(model, vl, device)
        f1 = f1_score(yv, pv, average="macro")
        if f1 > best:
            best = f1
            torch.save(model.state_dict(), out)
        print(f"epoch {ep}/{a.epochs} val_acc {accuracy_score(yv, pv):.4f} val_macroF1 {f1:.4f} "
              f"({time.time() - t0:.0f}s){' *' if f1 == best else ''}", flush=True)

    model.load_state_dict(torch.load(out, weights_only=True))
    yt, pt = evaluate(model, testl, device)
    k = len(classes)
    p, r, f, s = precision_recall_fscore_support(yt, pt, labels=range(k), zero_division=0)
    timing = {}
    b1 = next(iter(DataLoader(testl.dataset, batch_size=1)))
    for dev in ["cpu"] + (["cuda"] if device.type == "cuda" else []):
        m = model.to(dev).eval()
        ids, am = b1["input_ids"].to(dev), b1["attention_mask"].to(dev)
        with torch.no_grad():
            for _ in range(3):
                m(ids, am)
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                m(ids, am)
            if dev == "cuda":
                torch.cuda.synchronize()
        timing[f"{dev}_ms_per_text"] = round((time.perf_counter() - t0) / 20 * 1000, 2)
    metrics = {
        "model": a.model, "encoder": ENCODERS[MODELS[a.model][0]], "dataset": a.dataset, "epochs": a.epochs,
        "test_size": len(yt),
        "accuracy": round(accuracy_score(yt, pt), 4), "macro_f1": round(f1_score(yt, pt, average="macro"), 4),
        "per_class": {c: {"precision": round(p[i], 4), "recall": round(r[i], 4), "f1": round(f[i], 4),
                          "support": int(s[i])} for i, c in enumerate(classes)},
        "confusion_matrix": {"labels": classes, "rows_true_cols_pred": confusion_matrix(yt, pt, labels=range(k)).tolist()},
        "inference_time": timing,
    }
    rep = ROOT / "project" / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / f"text_{a.model}_{a.dataset}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k_: metrics[k_] for k_ in ("accuracy", "macro_f1", "inference_time")}), flush=True)


if __name__ == "__main__":
    main()
