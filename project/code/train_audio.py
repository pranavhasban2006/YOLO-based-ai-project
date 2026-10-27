r"""Speech-emotion models on RAVDESS + TESS + SAVEE (+ CREMA-D when real audio is present), faculty hybrid list.

Labels are mapped to the app's 7 classes (angry, disgust, fear, happy, neutral, sad, surprise);
RAVDESS "calm" has no counterpart and is dropped. Clips are 16 kHz mono, centre-cropped / padded to 4 s.
Split: stratified by (dataset, emotion), 70/15/15, seed 0, fixed for every model.

Usage (repo root, CUDA training venv):
    project\.venv-train\Scripts\python project\code\train_audio.py --prepare
    project\.venv-train\Scripts\python project\code\train_audio.py --model cnn_bilstm_attn
Models: wav2vec2_bilstm_attn, cnn_bilstm_attn, hubert_transformer, ast_mlp, cnn_gru
The pretrained encoders (wav2vec 2.0, HuBERT, AST) are frozen and their features cached once: 4 GB of VRAM
is not enough to fine-tune them, and frozen features keep the comparison identical in data and split.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "project" / "data" / "audio"
CACHE = DATA / "audio_16k.npz"
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
SR, LEN = 16000, 4 * 16000

RAVDESS = {"01": "neutral", "03": "happy", "04": "sad", "05": "angry", "06": "fear", "07": "disgust", "08": "surprise"}
SAVEE = {"a": "angry", "d": "disgust", "f": "fear", "h": "happy", "n": "neutral", "sa": "sad", "su": "surprise"}
TESS = {"angry": "angry", "disgust": "disgust", "fear": "fear", "happy": "happy", "neutral": "neutral",
        "sad": "sad", "ps": "surprise", "pleasant_surprise": "surprise", "pleasant_surprised": "surprise"}
CREMA = {"ANG": "angry", "DIS": "disgust", "FEA": "fear", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
ENCODERS = {"wav2vec2": "facebook/wav2vec2-base", "hubert": "facebook/hubert-base-ls960",
            "ast": "MIT/ast-finetuned-audioset-10-10-0.4593"}


def label_of(src, p):
    n = p.stem
    if src == "ravdess":
        return RAVDESS.get(n.split("-")[2])
    if src == "savee":
        return SAVEE.get(re.match(r"[A-Z]+_([a-z]+)\d+", n).group(1))
    if src == "tess":
        return TESS.get(n.split("_", 2)[-1].lower())
    if src == "crema":
        return CREMA.get(n.split("_")[2])


def load(p):
    w, sr = sf.read(p, dtype="float32", always_2d=True)
    w = torch.from_numpy(w.mean(1))
    if sr != SR:
        w = torchaudio.functional.resample(w, sr, SR)
    if len(w) >= LEN:
        s = (len(w) - LEN) // 2
        return w[s:s + LEN]
    return nn.functional.pad(w, (0, LEN - len(w)))


def prepare():
    # Collect labels first, then decode into one preallocated int16 array (a list of clips + np.stack doubled peak RAM).
    items = []
    for src in ("ravdess", "tess", "savee", "crema"):
        files = sorted(p for p in (DATA / src).rglob("*") if p.suffix.lower() in (".wav", ".mp3"))
        files = [p for p in files if p.stat().st_size > 1000]          # skip Git-LFS pointer stubs
        found = [(p, lab, src) for p in files if (lab := label_of(src, p)) is not None]
        items += found
        print(f"{src}: {len(found)} clips", flush=True)
    wav = np.empty((len(items), LEN), dtype=np.int16)
    for i, (p, _, _) in enumerate(items):
        wav[i] = (load(p).clamp(-1, 1) * 32767).to(torch.int16).numpy()
    ys = [CLASSES.index(lab) for _, lab, _ in items]
    np.savez(CACHE, wav=wav, y=np.array(ys), src=np.array([s_ for _, _, s_ in items]))
    print(f"saved {CACHE} ({len(ys)} clips), class counts {np.bincount(ys, minlength=7).tolist()}")


def splits(y, src):
    idx = np.arange(len(y))
    strat = np.char.add(src.astype(str), y.astype(str))
    tr, rest = train_test_split(idx, test_size=0.3, stratify=strat, random_state=0)
    va, te = train_test_split(rest, test_size=0.5, stratify=strat[rest], random_state=0)
    return tr, va, te


def encoder_features(enc, wav, device):
    """Frozen pretrained encoder -> cached features. wav2vec2/HuBERT: frame sequence (T x 768); AST: pooled 768."""
    out = DATA / f"audio_feat_{enc}.npy"
    if out.exists():
        return np.load(out, mmap_mode="r")
    from transformers import AutoFeatureExtractor, AutoModel, ASTModel
    name = ENCODERS[enc]
    fe = AutoFeatureExtractor.from_pretrained(name)
    model = (ASTModel if enc == "ast" else AutoModel).from_pretrained(name).to(device).eval().half()
    feats = []
    with torch.no_grad():
        for i in range(0, len(wav), 16):
            batch = [w.astype(np.float32) / 32767 for w in wav[i:i + 16]]
            x = fe(batch, sampling_rate=SR, return_tensors="pt")
            x = {k: v.to(device).half() if v.is_floating_point() else v.to(device) for k, v in x.items()}
            h = model(**x).last_hidden_state
            feats.append((h[:, :2].mean(1) if enc == "ast" else h).float().cpu().numpy().astype(np.float16))
            if i % 800 == 0:
                print(f"  {enc} features {i}/{len(wav)}", flush=True)
    feats = np.concatenate(feats)
    np.save(out, feats)
    return feats


# ---- Faculty hybrid models.

class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, s):                                            # B,T,d -> B,d
        return (s * torch.softmax(self.score(s), 1)).sum(1)


class LogMel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=1024, hop_length=320, n_mels=64)

    def forward(self, w):                                            # B,LEN -> B,1,64,201
        m = torch.log(self.mel(w) + 1e-6)
        return ((m - m.mean((1, 2), keepdim=True)) / (m.std((1, 2), keepdim=True) + 1e-6)).unsqueeze(1)


def mel_cnn():
    """2D CNN over log-mel; frequency pooled away, time kept -> B,T',256."""
    def b(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
    return nn.Sequential(b(1, 32), b(32, 64), b(64, 128), b(128, 256))


class MelCNNRNN(nn.Module):
    def __init__(self, rnn, n=7, h=128):
        super().__init__()
        self.mel, self.cnn = LogMel(), mel_cnn()
        bi = rnn == "bilstm"
        self.rnn = (nn.LSTM if bi else nn.GRU)(256, h, batch_first=True, bidirectional=bi)
        d = h * (2 if bi else 1)
        self.pool = AttnPool(d) if bi else None
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(d, n))

    def forward(self, w):
        f = self.cnn(self.mel(w)).mean(2).transpose(1, 2)              # B,T',256
        s, hn = self.rnn(f)
        return self.head(self.pool(s) if self.pool else hn[-1])


class SeqBiLSTMAttn(nn.Module):
    """wav2vec 2.0 frame features -> BiLSTM -> attention pooling."""
    def __init__(self, n=7, d=768, h=128):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.1))
        self.lstm = nn.LSTM(d, h, batch_first=True, bidirectional=True)
        self.pool, self.head = AttnPool(2 * h), nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * h, n))

    def forward(self, x):
        s, _ = self.lstm(self.proj(x))
        return self.head(self.pool(s))


class SeqTransformer(nn.Module):
    """HuBERT frame features -> 2-layer Transformer encoder -> mean pool."""
    def __init__(self, n=7, d=768, k=256):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, k))
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(k, 4, k * 4, 0.1, batch_first=True,
                                                                   norm_first=True), 2)
        self.head = nn.Sequential(nn.LayerNorm(k), nn.Dropout(0.3), nn.Linear(k, n))

    def forward(self, x):
        return self.head(self.tf(self.proj(x)).mean(1))


def mlp(n=7, d=768):
    return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, n))


# model name -> (input: "wav" or encoder name, builder, epochs, lr)
MODELS = {
    "cnn_bilstm_attn": ("wav", lambda: MelCNNRNN("bilstm"), 40, 1e-3),
    "cnn_gru": ("wav", lambda: MelCNNRNN("gru"), 40, 1e-3),
    "wav2vec2_bilstm_attn": ("wav2vec2", SeqBiLSTMAttn, 30, 5e-4),
    "hubert_transformer": ("hubert", SeqTransformer, 30, 3e-4),
    "ast_mlp": ("ast", mlp, 60, 1e-3),
}


def evaluate(model, loader, device, prep):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(model(prep(x.to(device))).argmax(1).cpu())
            ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def train(name, batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    inp, builder, epochs, lr = MODELS[name]
    d = np.load(CACHE)
    y, src = d["y"], d["src"]
    tr, va, te = splits(y, src)
    if inp == "wav":
        x = torch.from_numpy(d["wav"])
        prep = lambda t: t.float() / 32767
    else:
        x = torch.from_numpy(np.asarray(encoder_features(inp, d["wav"], device)))
        prep = lambda t: t.float()
    yt_all = torch.from_numpy(y)
    mk = lambda i, sh: DataLoader(TensorDataset(x[i], yt_all[i]), batch_size=batch, shuffle=sh)
    tl, vl, testl = mk(tr, True), mk(va, False), mk(te, False)
    counts = np.bincount(y[tr], minlength=7)
    weights = torch.tensor(counts.sum() / (7 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    model = builder().to(device)
    ce = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * len(tl))
    out = ROOT / "models" / f"audio_{name}.pt"
    best = -1.0
    print(f"{name}: {len(tr)} train / {len(va)} val / {len(te)} test clips on {device}", flush=True)
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        for xb, yb in tl:
            xb, yb = prep(xb.to(device)), yb.to(device)
            if inp == "wav":                                         # random gain + time shift
                xb = torch.roll(xb * (0.7 + 0.6 * torch.rand(len(xb), 1, device=device)),
                                int(np.random.randint(-SR // 4, SR // 4)), 1)
            loss = ce(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        yv, pv = evaluate(model, vl, device, prep)
        f1 = f1_score(yv, pv, average="macro")
        if f1 > best:
            best = f1
            torch.save(model.state_dict(), out)
        print(f"epoch {ep}/{epochs} val_acc {accuracy_score(yv, pv):.4f} val_macroF1 {f1:.4f} "
              f"({time.time() - t0:.0f}s){' *' if f1 == best else ''}", flush=True)

    model.load_state_dict(torch.load(out, weights_only=True))
    yt, pt = evaluate(model, testl, device, prep)
    p, r, f, s = precision_recall_fscore_support(yt, pt, labels=range(7), zero_division=0)
    # Inference time covers the trainable head only; for encoder models the frozen encoder adds its own cost.
    timing = {}
    x1 = prep(testl.dataset[0][0].unsqueeze(0))
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
        timing[f"{dev}_ms_per_clip"] = round((time.perf_counter() - t0) / 50 * 1000, 2)
    metrics = {
        "model": name, "input": inp if inp == "wav" else f"frozen {ENCODERS[inp]} features",
        "dataset": "RAVDESS+TESS+SAVEE" + ("+CREMA-D" if "crema" in set(src) else ""),
        "sources_in_test": {k: int(v) for k, v in zip(*np.unique(src[te], return_counts=True))},
        "epochs": epochs, "test_size": len(te),
        "accuracy": round(accuracy_score(yt, pt), 4), "macro_f1": round(f1_score(yt, pt, average="macro"), 4),
        "per_class": {c: {"precision": round(p[i], 4), "recall": round(r[i], 4), "f1": round(f[i], 4),
                          "support": int(s[i])} for i, c in enumerate(CLASSES)},
        "confusion_matrix": {"labels": CLASSES,
                             "rows_true_cols_pred": confusion_matrix(yt, pt, labels=range(7)).tolist()},
        "inference_time": timing,
    }
    rep = ROOT / "project" / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / f"audio_{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("accuracy", "macro_f1", "inference_time")}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    if a.prepare:
        prepare()
    if a.model:
        train(a.model, a.batch)
