r"""Two-stage cascade for every faculty hybrid "A1 + A2": A1 is trained and tested ALONE, then A1's output is the
input of A2, and we count how many of A1's mistakes A2 corrects.

Stage 1  A1 alone = A1's network + a linear classifier, trained and tested by itself -> "A1 accuracy".
Stage 2  A1 is frozen. For every sample its output (feature map / frame or token sequence, plus its class scores)
         is stored, and A2 is trained on it: final scores = A1 scores + A2(A1 features). A2's last layer starts at
         zero, so before stage 2 the cascade predicts exactly what A1 predicts; A2 learns only the corrections.
Report   per test sample: A1 right/wrong vs A1->A2 right/wrong -> fixed (wrong->right), broken (right->wrong), net.

Splits are those of the original training scripts: face FER-2013 90/10 train/val + official test set,
speech 70/15/15 stratified, text official train/validation/test. Early stopping on validation macro-F1, AdamW (L2).

Usage (repo root); every step is resumable (finished work is skipped):
    python project/code/cascade.py audio
    python project/code/cascade.py face  [--only densenet_se]
    python project/code/cascade.py text --dataset goemotions
    python project/code/cascade.py summary
Writes models/cascade/*.pt, project/reports/cascade/<hybrid>.json (+ _preds.npz) and cascade_summary.{csv,md}.
CASCADE_OUT=<dir> puts models + reports there instead (Colab: a Google Drive folder, survives disconnects).
"""
import argparse
import contextlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ["CASCADE_OUT"]) if os.environ.get("CASCADE_OUT") else None
CK = OUT / "models" if OUT else ROOT / "models" / "cascade"
REP = OUT / "reports" if OUT else ROOT / "project" / "reports" / "cascade"
CACHE = ROOT / "project" / "data" / "cascade_cache"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = True          # mixed precision; switched off for audio (STFT in fp16 is not supported)


def autocast():
    return torch.autocast(DEV.type) if AMP and DEV.type == "cuda" else contextlib.nullcontext()


def to_t(a):
    t = torch.from_numpy(np.asarray(a))
    t = t.float() / 32767 if t.dtype == torch.int16 else t.float() if t.is_floating_point() else t.long()
    return t.to(DEV, non_blocking=True)


def batches(arrs, y, idx, bs, shuffle=False):
    """Mini-batches of rows `idx` of numpy arrays / memmaps (rows sorted inside a batch for fast memmap reads)."""
    def gen():
        order = np.random.permutation(idx) if shuffle else idx
        for i in range(0, len(order), bs):
            b = np.sort(order[i:i + bs])
            yield [to_t(a[b]) for a in arrs], torch.from_numpy(y[b]).long().to(DEV)
    return gen


def class_weights(y, k, damp=False):
    c = np.bincount(y, minlength=k)
    w = c.sum() / (k * np.maximum(c, 1))
    return torch.tensor(np.sqrt(w) if damp else w, dtype=torch.float32, device=DEV)


def predict(model, batch_gen):
    model.eval()
    ys, ps = [], []
    with torch.no_grad(), autocast():
        for xs, y in batch_gen():
            ps.append(model(*xs).float().argmax(1).cpu())
            ys.append(y.cpu())
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def train_loop(model, train_gen, val_gen, epochs, opt, sched, ce, out, patience, log):
    """Best-on-validation-macro-F1 checkpointing with early stopping; leaves the best weights loaded."""
    scaler = torch.amp.GradScaler(enabled=AMP and DEV.type == "cuda")
    best, bad = -1.0, 0
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        for xs, y in train_gen():
            with autocast():
                loss = ce(model(*xs).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if sched:
                sched.step()
        yv, pv = predict(model, val_gen)
        f1 = f1_score(yv, pv, average="macro")
        improved = f1 > best
        if improved:
            best, bad = f1, 0
            torch.save(model.state_dict(), out)
        else:
            bad += 1
        print(f"{log} epoch {ep}/{epochs} val_acc {accuracy_score(yv, pv):.4f} val_macroF1 {f1:.4f} "
              f"({time.time() - t0:.0f}s){' *' if improved else ''}", flush=True)
        if bad >= patience:
            print(f"{log} early stopping (no improvement for {patience} epochs)", flush=True)
            break
    model.load_state_dict(torch.load(out, weights_only=True))


class A1(nn.Module):
    """A1 alone: A1's network -> pooling -> linear classifier."""
    def __init__(self, backbone, d, n, pool, drop=0.3):
        super().__init__()
        self.backbone, self.pool = backbone, pool
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(d, n))

    def forward(self, *xs):
        return self.head(self.pool(self.backbone(*xs), *xs))


class Cascade(nn.Module):
    """Stage 2: inputs are A1's stored outputs (features [, mask], scores); final scores = A1 scores + A2(features)."""
    def __init__(self, a2):
        super().__init__()
        self.a2 = a2
        last = [m for m in a2.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, *xs):
        return xs[-1] + self.a2(*xs[:-1])


def stage1_done(out):
    return out.with_suffix(".done").exists()


def all_done(tags):
    return all((REP / f"{t}.json").exists() for t in tags)


def cache_a1(a1, batch_iter, n, stem):
    """Runs the trained, frozen A1 once over every sample and keeps its output: features (A2's input) + scores."""
    fpath, spath = CACHE / f"{stem}_features.npy", CACHE / f"{stem}_scores.npy"
    if not spath.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        a1.eval()
        feats, scores, i = None, [], 0
        with torch.no_grad(), autocast():
            for xs in batch_iter:
                f = a1.backbone(*xs)
                if feats is None:
                    feats = np.lib.format.open_memmap(fpath, "w+", np.float16, (n, *f.shape[1:]))
                feats[i:i + len(f)] = f.half().cpu().numpy()
                scores.append(a1.head(a1.pool(f, *xs)).float().cpu().numpy())
                i += len(f)
        feats.flush()
        del feats
        np.save(spath, np.concatenate(scores))
        print(f"cached A1 output {stem}: {n} samples", flush=True)
    return np.load(fpath, mmap_mode="r"), np.load(spath)


def drop_cache(stem):
    """The feature cache is only needed for stage 2 (face/text caches are 2-15 GB each)."""
    for p in CACHE.glob(f"{stem}_*.npy"):
        p.unlink()


def stage2(tag, meta, a2, feats, extras, scores, y, tr, va, te, classes, epochs, lr, bs=64, patience=5):
    REP.mkdir(parents=True, exist_ok=True)
    if (REP / f"{tag}.json").exists():
        print(f"{tag}: stage 2 already done", flush=True)
        return
    k = len(classes)
    torch.manual_seed(0)
    model = Cascade(a2).to(DEV)
    arrs = [feats, *extras, scores]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    ce = nn.CrossEntropyLoss(weight=class_weights(y[tr], k, damp=meta["modality"] == "text"))
    CK.mkdir(parents=True, exist_ok=True)
    train_loop(model, batches(arrs, y, tr, bs, True), batches(arrs, y, va, 256), epochs, opt, None, ce,
               CK / f"{tag}_stage2.pt", patience, f"{tag} [A2 {meta['A2']}]")
    yt, p2 = predict(model, batches(arrs, y, te, 256))
    p1 = scores[te].argmax(1)
    r1, r2 = p1 == yt, p2 == yt
    acc = lambda p: round(accuracy_score(yt, p), 4)
    f1 = lambda p: round(f1_score(yt, p, average="macro", labels=range(k)), 4)
    rep = {**meta, "test_size": int(len(yt)), "classes": classes,
           "stage1_A1_alone": {"accuracy": acc(p1), "macro_f1": f1(p1), "correct": int(r1.sum()),
                               "wrong": int((~r1).sum()),
                               "train_accuracy": round(float((scores[tr].argmax(1) == y[tr]).mean()), 4)},
           "stage2_A1_then_A2": {"accuracy": acc(p2), "macro_f1": f1(p2), "correct": int(r2.sum()),
                                 "wrong": int((~r2).sum())},
           "A1_wrong_fixed_by_A2": int((~r1 & r2).sum()), "A1_right_broken_by_A2": int((r1 & ~r2).sum()),
           "wrong_in_both": int((~r1 & ~r2).sum()), "net_gain_samples": int(r2.sum() - r1.sum()),
           "accuracy_gain_points": round(100 * (acc(p2) - acc(p1)), 2),
           "confusion_A1": confusion_matrix(yt, p1, labels=range(k)).tolist(),
           "confusion_A1_then_A2": confusion_matrix(yt, p2, labels=range(k)).tolist()}
    (REP / f"{tag}.json").write_text(json.dumps(rep, indent=2))
    np.savez(REP / f"{tag}_preds.npz", test_index=te, y=yt, pred_A1=p1, pred_A1_then_A2=p2)
    print(f"{tag}: A1 alone {rep['stage1_A1_alone']['accuracy']} -> A1+A2 {rep['stage2_A1_then_A2']['accuracy']} | "
          f"fixed {rep['A1_wrong_fixed_by_A2']}, broken {rep['A1_right_broken_by_A2']}, "
          f"net {rep['net_gain_samples']:+d} of {len(yt)}", flush=True)


# ------------------------------------------------------------------------------------------------ speech

def audio(only):
    global AMP
    AMP = False
    import train_audio as ta

    class MelCNN(nn.Module):
        """The CNN of CNN+BiLSTM+Attn / CNN+GRU: log-mel -> 2D CNN -> frame sequence B,T',256."""
        def __init__(self):
            super().__init__()
            self.mel, self.cnn = ta.LogMel(), ta.mel_cnn()

        def forward(self, w):
            return self.cnn(self.mel(w)).mean(2).transpose(1, 2)

    class RNNHead(nn.Module):
        def __init__(self, rnn, n=7, d=256, h=128):
            super().__init__()
            bi = rnn == "bilstm"
            self.rnn = (nn.LSTM if bi else nn.GRU)(d, h, batch_first=True, bidirectional=bi)
            dd = h * (2 if bi else 1)
            self.pool = ta.AttnPool(dd) if bi else None
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(dd, n))

        def forward(self, f):
            s, hn = self.rnn(f)
            return self.head(self.pool(s) if self.pool else hn[-1])

    d = np.load(ta.CACHE)
    y, src, wav = d["y"], d["src"], d["wav"]
    tr, va, te = (np.sort(s) for s in ta.splits(y, src))
    seq_mean = lambda f, x: f.mean(1)
    # A1 -> (input, A1 network, feature dim, pooling, epochs, {hybrid: (A2 name, A2 builder)})
    A1S = {
        "CNN": (wav, MelCNN, 256, seq_mean, 40,
                {"cnn_bilstm_attn": ("BiLSTM + Attention", lambda: RNNHead("bilstm")),
                 "cnn_gru": ("GRU", lambda: RNNHead("gru"))}),
        "wav2vec 2.0": ("wav2vec2", nn.Identity, 768, seq_mean, 30,
                        {"wav2vec2_bilstm_attn": ("BiLSTM + Attention", ta.SeqBiLSTMAttn)}),
        "HuBERT": ("hubert", nn.Identity, 768, seq_mean, 30,
                   {"hubert_transformer": ("Transformer encoder", ta.SeqTransformer)}),
        "AST": ("ast", nn.Identity, 768, lambda f, x: f, 60, {"ast_mlp": ("MLP classifier", ta.mlp)}),
    }
    for a1_name, (inp, backbone, dim, pool, epochs, hybrids) in A1S.items():
        hybrids = {h: v for h, v in hybrids.items() if not only or only == h}
        tags = [f"audio_{h}" for h in hybrids]
        if not hybrids or all_done(tags):
            continue
        x = np.asarray(ta.encoder_features(inp, wav, DEV)) if isinstance(inp, str) else inp
        stem = f"audio_{a1_name.split()[0].lower()}"
        torch.manual_seed(0)
        a1 = A1(backbone(), dim, 7, pool).to(DEV)
        out = CK / f"{stem}_A1.pt"
        CK.mkdir(parents=True, exist_ok=True)
        if not stage1_done(out):
            train = batches([x], y, tr, 32, True)
            if inp is wav:                                        # random gain + time shift, as in train_audio.py
                plain = train

                def train():
                    for (w,), yb in plain():
                        g = 0.7 + 0.6 * torch.rand(len(w), 1, device=DEV)
                        yield [torch.roll(w * g, int(np.random.randint(-ta.SR // 4, ta.SR // 4)), 1)], yb
            opt = torch.optim.AdamW(a1.parameters(), lr=1e-3, weight_decay=1e-2)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * -(-len(tr) // 32))
            ce = nn.CrossEntropyLoss(weight=class_weights(y[tr], 7), label_smoothing=0.1)
            train_loop(a1, train, batches([x], y, va, 128), epochs, opt, sched, ce, out, 10,
                       f"{stem} [A1 {a1_name} alone]")
            out.with_suffix(".done").touch()
        a1.load_state_dict(torch.load(out, weights_only=True))
        feats, scores = cache_a1(a1, ([to_t(x[i:i + 64])] for i in range(0, len(y), 64)), len(y), stem)
        for hyb, (a2_name, a2) in hybrids.items():
            meta = {"modality": "speech", "dataset": "RAVDESS+TESS+SAVEE", "hybrid": hyb, "A1": a1_name,
                    "A2": a2_name, "split": "70/15/15 stratified (train/val/test)"}
            stage2(f"audio_{hyb}", meta, a2(), feats, [], scores, y, tr, va, te, ta.CLASSES, 30, 1e-3, patience=8)
        del feats
        drop_cache(stem)


# ------------------------------------------------------------------------------------------------ face

def face(only):
    from torch.utils.data import ConcatDataset, DataLoader, Subset
    from torchvision import datasets, models as W
    import train_face as tf

    def resnet50_trunk():
        r = W.resnet50(weights=W.ResNet50_Weights.IMAGENET1K_V2)
        return nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3)

    def resnet18_trunk():
        return nn.Sequential(*list(W.resnet18(weights=W.ResNet18_Weights.IMAGENET1K_V1).children())[:-2])

    class PoolHead(nn.Module):
        """CBAM / SE attention on A1's feature map -> global average pool -> linear."""
        def __init__(self, attn, c, n=7):
            super().__init__()
            self.attn, self.out = attn, tf.head(c, n)

        def forward(self, f):
            return self.out(self.attn(f).mean((2, 3)))

    class SpatialHead(nn.Module):
        def __init__(self, c, n=7):
            super().__init__()
            self.attn, self.out = tf.SpatialAttention(c), tf.head(c, n)

        def forward(self, f):
            return self.out(self.attn(f))

    class ViTHead(nn.Module):
        """ResNet-50's 14x14x1024 map -> 196 tokens -> 4-layer Vision Transformer encoder (as in train_face.py)."""
        def __init__(self, c=1024, n=7, d=256):
            super().__init__()
            self.proj = nn.Conv2d(c, d, 1)
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            self.pos = nn.Parameter(torch.randn(1, 197, d) * 0.02)
            self.vit = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, d * 4, 0.1, batch_first=True,
                                                                        norm_first=True), 4)
            self.norm, self.out = nn.LayerNorm(d), tf.head(d, n)

        def forward(self, f):
            t = self.proj(f).flatten(2).transpose(1, 2)
            t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1) + self.pos
            return self.out(self.norm(self.vit(t)[:, 0]))

    class BiLSTMHead(nn.Module):
        """CNN's 7x7 map read as a 49-step sequence -> BiLSTM -> attention pooling (as in train_face.py)."""
        def __init__(self, c=512, n=7, h=256):
            super().__init__()
            self.lstm = nn.LSTM(c, h, batch_first=True, bidirectional=True)
            self.score, self.out = nn.Linear(2 * h, 1), tf.head(2 * h, n)

        def forward(self, f):
            s, _ = self.lstm(f.flatten(2).transpose(1, 2))
            return self.out((s * torch.softmax(self.score(s), 1)).sum(1))

    # hybrid -> (A1 name, A1 network, channels, A2 name, A2 builder, lr, batch)
    FACE = {
        "mobilenet_attn": ("MobileNetV3", lambda: W.mobilenet_v3_large(weights=W.MobileNet_V3_Large_Weights.IMAGENET1K_V2).features,
                           960, "Spatial attention", lambda: SpatialHead(960), 3e-4, 64),
        "cnn_bilstm": ("CNN (ResNet-18)", resnet18_trunk, 512, "BiLSTM + Attention", BiLSTMHead, 2e-4, 64),
        "densenet_se": ("DenseNet121", lambda: nn.Sequential(W.densenet121(weights=W.DenseNet121_Weights.IMAGENET1K_V1).features, nn.ReLU()),
                        1024, "SE attention", lambda: PoolHead(tf.SE(1024), 1024), 2e-4, 32),
        "effnet_cbam": ("EfficientNetB3", lambda: W.efficientnet_b3(weights=W.EfficientNet_B3_Weights.IMAGENET1K_V1).features,
                        1536, "CBAM attention", lambda: PoolHead(tf.CBAM(1536), 1536), 2e-4, 32),
        "resnet50_vit": ("ResNet50", resnet50_trunk, 1024, "Vision Transformer", ViTHead, 2e-4, 32),
    }
    train_aug = datasets.ImageFolder(tf.DATA / "train", tf.transforms_for("hybrid", True))
    train_plain = datasets.ImageFolder(tf.DATA / "train", tf.transforms_for("hybrid", False))
    test = datasets.ImageFolder(tf.DATA / "test", tf.transforms_for("hybrid", False))
    assert train_aug.classes == tf.CLASSES
    idx = np.random.default_rng(0).permutation(len(train_aug))              # the 90/10 split of train_face.py
    n_val = len(idx) // 10
    y = np.array(train_plain.targets + test.targets)
    tr, va, te = np.sort(idx[n_val:]), np.sort(idx[:n_val]), len(train_plain) + np.arange(len(test))

    for hyb, (a1_name, backbone, c, a2_name, a2, lr, bs) in FACE.items():
        if (only and only != hyb) or all_done([f"face_{hyb}"]):
            continue
        stem = f"face_{hyb.split('_')[0]}"
        out = CK / f"{stem}_A1.pt"
        CK.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(0)
        a1 = A1(backbone(), c, 7, lambda f, x: f.mean((2, 3))).to(DEV)
        workers = 0 if hyb == "resnet50_vit" and DEV.type == "cuda" and torch.cuda.get_device_properties(0).total_memory < 8e9 else 2
        kw = dict(batch_size=bs, num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
        if not stage1_done(out):
            tl = DataLoader(Subset(train_aug, idx[n_val:]), shuffle=True, **kw)
            vl = DataLoader(Subset(train_plain, idx[:n_val]), **kw)
            loader_gen = lambda dl: (lambda: (([xb.to(DEV, non_blocking=True)], yb.to(DEV)) for xb, yb in dl))
            opt = torch.optim.AdamW(a1.parameters(), lr=lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=15 * len(tl))
            ce = nn.CrossEntropyLoss(weight=class_weights(y[tr], 7), label_smoothing=0.1)
            train_loop(a1, loader_gen(tl), loader_gen(vl), 15, opt, sched, ce, out, 4, f"{stem} [A1 {a1_name} alone]")
            out.with_suffix(".done").touch()
            del tl, vl
        a1.load_state_dict(torch.load(out, weights_only=True))
        rows = DataLoader(ConcatDataset([train_plain, test]), batch_size=64, num_workers=workers)
        feats, scores = cache_a1(a1, ([xb.to(DEV)] for xb, _ in rows), len(y), stem)
        del a1
        torch.cuda.empty_cache()
        meta = {"modality": "face", "dataset": "FER-2013", "hybrid": hyb, "A1": a1_name, "A2": a2_name,
                "split": "90/10 train/validation of the FER-2013 train set; official FER-2013 test set"}
        stage2(f"face_{hyb}", meta, a2(), feats, [], scores, y, tr, va, te, tf.CLASSES, 15,
               3e-4 if hyb == "resnet50_vit" else 1e-3)
        del feats
        drop_cache(stem)


# ------------------------------------------------------------------------------------------------ text

def text(dataset, only):
    from transformers import AutoModel, AutoTokenizer
    import train_text as tt

    class Encoder(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.m = AutoModel.from_pretrained(name, torch_dtype=torch.float32)

        def forward(self, ids, mask):
            return self.m(input_ids=ids, attention_mask=mask).last_hidden_state

    class TextA2(nn.Module):
        def __init__(self, head, h, n):
            super().__init__()
            self.drop, self.head = nn.Dropout(0.2), head(h, n)

        def forward(self, s, mask):
            return self.head(self.drop(s), mask)

    masked_mean = lambda f, ids, m: (f * m[..., None]).sum(1) / m.sum(1, keepdim=True)
    A2_NAMES = {tt.BiLSTMAttnHead: "BiLSTM + Attention", tt.AttnHead: "Attention classifier",
                tt.CNNHead: "CNN", tt.CNNBiLSTMHead: "CNN + BiLSTM"}
    A1_NAMES = {"roberta": "RoBERTa", "bert": "BERT", "deberta": "DeBERTa", "distilbert": "DistilBERT"}
    classes = tt.DATASETS[dataset]
    k = len(classes)
    for enc in ("distilbert", "bert", "roberta", "deberta"):
        hybrids = {h: head for h, (e, head) in tt.MODELS.items() if e == enc and (not only or only == h)}
        tags = [f"text_{h}_{dataset}" for h in hybrids]
        if not hybrids or all_done(tags):
            continue
        tok = AutoTokenizer.from_pretrained(tt.ENCODERS[enc])
        parts = [tt.load_split(dataset, s) for s in ("train", "validation", "test")]
        e = tok(sum((list(p[0]) for p in parts), []), truncation=True, max_length=tt.MAX_LEN,
                padding="max_length", return_tensors="np")
        ids, mask = e["input_ids"].astype(np.int64), e["attention_mask"].astype(np.int8)
        y = np.array(sum((list(p[1]) for p in parts), []))
        n1, n2 = len(parts[0][1]), len(parts[1][1])
        tr, va, te = np.arange(n1), np.arange(n1, n1 + n2), np.arange(n1 + n2, len(y))
        stem = f"text_{enc}_{dataset}"
        out = CK / f"{stem}_A1.pt"
        CK.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(0)
        enc_net = Encoder(tt.ENCODERS[enc])
        h = enc_net.m.config.hidden_size
        a1 = A1(enc_net, h, k, masked_mean, drop=0.2).to(DEV)
        if not stage1_done(out):
            opt = torch.optim.AdamW([{"params": a1.backbone.parameters(), "lr": 2e-5},
                                     {"params": a1.head.parameters(), "lr": 1e-3}], weight_decay=1e-2)
            total = 4 * -(-len(tr) // 16)
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda s: min(1.0, s / (0.1 * total)) * max(0.0, (total - s) / (0.9 * total)))
            ce = nn.CrossEntropyLoss(weight=class_weights(y[tr], k, damp=True))
            train_loop(a1, batches([ids, mask], y, tr, 16, True), batches([ids, mask], y, va, 64), 4, opt, sched,
                       ce, out, 2, f"{stem} [A1 {A1_NAMES[enc]} alone]")
            out.with_suffix(".done").touch()
        a1.load_state_dict(torch.load(out, weights_only=True))
        feats, scores = cache_a1(a1, ([to_t(ids[i:i + 64]), to_t(mask[i:i + 64])] for i in range(0, len(y), 64)),
                                 len(y), stem)
        del a1, enc_net
        torch.cuda.empty_cache()
        for hyb, head in hybrids.items():
            meta = {"modality": "text", "dataset": dataset, "hybrid": hyb, "A1": A1_NAMES[enc],
                    "A2": A2_NAMES[head], "split": "official train/validation/test"}
            stage2(f"text_{hyb}_{dataset}", meta, TextA2(head, h, k), feats, [mask], scores, y, tr, va, te,
                   classes, 10, 1e-3, patience=3)
        del feats
        drop_cache(stem)


# ------------------------------------------------------------------------------------------------ summary

def wrong_only(r, preds):
    """Faculty protocol: only the samples A1 got WRONG go to A2; A1's right answers are kept.
    Final = A1's correct + the ones A2 fixes (routing uses the true labels, so nothing can be broken)."""
    y, p1, p2 = preds["y"], preds["pred_A1"], preds["pred_A1_then_A2"]
    final = np.where(p1 == y, p1, p2)
    k = len(r["classes"])
    return {"accuracy": round(float((final == y).mean()), 4),
            "macro_f1": round(f1_score(y, final, average="macro", labels=range(k)), 4),
            "fed_to_A2": int((p1 != y).sum()), "fixed_by_A2": int(((p1 != y) & (p2 == y)).sum()),
            "confusion": confusion_matrix(y, final, labels=range(k)).tolist()}


def summary():
    rows = []
    for f in sorted(REP.glob("*.json")):
        r = json.loads(f.read_text())
        preds = f.with_name(f"{f.stem}_preds.npz")
        if "wrong_only_to_A2" not in r and not preds.exists():
            print(f"skipping {f.stem}: {preds.name} is missing")
            continue
        if "wrong_only_to_A2" not in r:
            r["wrong_only_to_A2"] = wrong_only(r, np.load(preds))
            f.write_text(json.dumps(r, indent=2))
        s1, s2, w = r["stage1_A1_alone"], r["stage2_A1_then_A2"], r["wrong_only_to_A2"]
        rows.append({"modality": r["modality"], "dataset": r["dataset"], "hybrid": f"{r['A1']} + {r['A2']}",
                     "test samples": r["test_size"], "A1 alone acc": s1["accuracy"], "A1 alone F1": s1["macro_f1"],
                     "A1 wrong (fed to A2)": w["fed_to_A2"], "fixed by A2": w["fixed_by_A2"],
                     "A1 + A2 (wrong only) acc": w["accuracy"], "A1 + A2 (wrong only) F1": w["macro_f1"],
                     "gain (points)": round(100 * (w["accuracy"] - s1["accuracy"]), 2),
                     "A2 on all samples acc": s2["accuracy"], "broken when A2 sees all": r["A1_right_broken_by_A2"]})
    if not rows:
        print("no cascade results yet")
        return
    cols = list(rows[0])
    (REP / "cascade_summary.csv").write_text(
        "\n".join([",".join(cols)] + [",".join(str(r[c]) for c in cols) for r in rows]) + "\n")
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (REP / "cascade_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("modality", choices=["audio", "face", "text", "summary"])
    ap.add_argument("--only", help="run a single hybrid, e.g. densenet_se")
    ap.add_argument("--dataset", choices=["goemotions", "emotion"], default="goemotions")
    a = ap.parse_args()
    np.random.seed(0)
    if a.modality == "audio":
        audio(a.only)
    elif a.modality == "face":
        face(a.only)
    elif a.modality == "text":
        text(a.dataset, a.only)
    summary()
