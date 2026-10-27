r"""Train a facial-emotion model on FER-2013 locally (GPU) and report the faculty's evaluation metrics.

Usage (from the repo root, with the CUDA training venv):
    project\.venv-train\Scripts\python project\code\train_face.py --model cnn
    project\.venv-train\Scripts\python project\code\train_face.py --model resnet18

Writes models/<checkpoint>.pt (state_dict, loadable by backend/app/modalities/face.py)
and project/reports/face_<model>_metrics.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parents[2]


# ---- Baselines. Copied from backend/app/modalities/face.py (importing it would pull in fastapi/mediapipe);
# they must stay identical there, or the backend's strict load_state_dict will reject the checkpoint.

def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
        nn.MaxPool2d(2))


class EmotionCNN(nn.Module):
    def __init__(self, n=7):
        super().__init__()
        self.features = nn.Sequential(block(1, 64), block(64, 128),
                                      block(128, 256), block(256, 512))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(512, n))

    def forward(self, x):
        return self.head(self.features(x))


def build_resnet18(n=7) -> nn.Module:
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(512, n)
    return m


CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
DATA = ROOT / "project" / "data" / "fer2013"
CHECKPOINTS = {"cnn": "cnn_baseline.pt", "resnet18": "resnet18_fer.pt",
               "effnet_cbam": "face_effnet_cbam.pt", "resnet50_vit": "face_resnet50_vit.pt",
               "densenet_se": "face_densenet_se.pt", "mobilenet_attn": "face_mobilenet_attn.pt",
               "cnn_bilstm": "face_cnn_bilstm.pt"}
HYBRIDS = {"effnet_cbam", "resnet50_vit", "densenet_se", "mobilenet_attn", "cnn_bilstm"}


# ---- Faculty hybrid models. All take 3x224x224 ImageNet-normalised input and use pretrained backbones.

class CBAM(nn.Module):
    """Channel attention (shared MLP over avg+max pool) then spatial attention (7x7 conv)."""
    def __init__(self, c, r=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(c, c // r, 1), nn.ReLU(), nn.Conv2d(c // r, c, 1))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        x = x * torch.sigmoid(self.mlp(x.mean((2, 3), keepdim=True)) + self.mlp(x.amax((2, 3), keepdim=True)))
        return x * torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))


class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(), nn.Linear(c // r, c), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x.mean((2, 3)))[:, :, None, None]


class SpatialAttention(nn.Module):
    """Learned per-location weights, softmax over the feature map, attention-weighted pooling."""
    def __init__(self, c):
        super().__init__()
        self.score = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        w = torch.softmax(self.score(x).flatten(2), -1)          # B,1,HW
        return (x.flatten(2) * w).sum(-1)                         # B,C


def head(c, n=7):
    return nn.Sequential(nn.Dropout(0.3), nn.Linear(c, n))


class EffNetCBAM(nn.Module):
    def __init__(self, n=7):
        super().__init__()
        self.features = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1).features
        self.cbam, self.head = CBAM(1536), head(1536, n)

    def forward(self, x):
        return self.head(self.cbam(self.features(x)).mean((2, 3)))


class ResNet50ViT(nn.Module):
    """ResNet-50 up to layer3 (local features, 14x14) -> 4-layer Transformer encoder over the 196 tokens."""
    def __init__(self, n=7, d=256):
        super().__init__()
        r = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.cnn = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3)
        self.proj = nn.Conv2d(1024, d, 1)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, 197, d) * 0.02)
        self.vit = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, d * 4, 0.1, batch_first=True,
                                                                    norm_first=True), 4)
        self.norm, self.head = nn.LayerNorm(d), head(d, n)

    def forward(self, x):
        t = self.proj(self.cnn(x)).flatten(2).transpose(1, 2)    # B,196,d
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1) + self.pos
        return self.head(self.norm(self.vit(t)[:, 0]))


class DenseNetSE(nn.Module):
    def __init__(self, n=7):
        super().__init__()
        self.features = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1).features
        self.se, self.head = SE(1024), head(1024, n)

    def forward(self, x):
        return self.head(self.se(torch.relu(self.features(x))).mean((2, 3)))


class MobileNetAttn(nn.Module):
    def __init__(self, n=7):
        super().__init__()
        self.features = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2).features
        self.attn, self.head = SpatialAttention(960), head(960, n)

    def forward(self, x):
        return self.head(self.attn(self.features(x)))


class CNNBiLSTM(nn.Module):
    """ResNet-18 trunk -> 7x7 map read as a 49-step sequence (row-major) -> BiLSTM -> attention pooling.
    FER-2013 has single images, so the sequence is spatial here; on video the same head takes per-frame features."""
    def __init__(self, n=7, h=256):
        super().__init__()
        r = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(r.children())[:-2])
        self.lstm = nn.LSTM(512, h, batch_first=True, bidirectional=True)
        self.score, self.head = nn.Linear(2 * h, 1), head(2 * h, n)

    def forward(self, x):
        s, _ = self.lstm(self.cnn(x).flatten(2).transpose(1, 2))  # B,49,2h
        return self.head((s * torch.softmax(self.score(s), 1)).sum(1))


HYBRID_BUILDERS = {"effnet_cbam": EffNetCBAM, "resnet50_vit": ResNet50ViT, "densenet_se": DenseNetSE,
                   "mobilenet_attn": MobileNetAttn, "cnn_bilstm": CNNBiLSTM}


def transforms_for(name, train):
    if name == "cnn":
        size, ch, mean, std = 48, 1, [0.5], [0.5]
    else:
        size, ch, mean, std = 224, 3, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    aug = [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
           transforms.RandomResizedCrop(size, scale=(0.85, 1.0))] if train else [transforms.Resize((size, size))]
    return transforms.Compose([transforms.Grayscale(ch), *aug, transforms.ToTensor(),
                               transforms.Normalize(mean, std)])


def build(name):
    if name == "cnn":
        return EmotionCNN()
    if name in HYBRIDS:
        return HYBRID_BUILDERS[name]()
    m = build_resnet18()
    pre = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    pre.fc = m.fc
    return pre


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
        for x, y in loader:
            ps.append(model(x.to(device)).argmax(1).cpu())
            ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=CHECKPOINTS, required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--eval-only", action="store_true", help="skip training; test the saved checkpoint")
    args = ap.parse_args()
    epochs = args.epochs or (40 if args.model == "cnn" else 15)
    # 4 GB VRAM (RTX 3050 laptop): the heavier backbones need smaller batches.
    batch = args.batch or {"cnn": 128, "resnet18": 64, "mobilenet_attn": 64, "cnn_bilstm": 64}.get(args.model, 32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    train_full = datasets.ImageFolder(DATA / "train", transforms_for(args.model, True))
    val_full = datasets.ImageFolder(DATA / "train", transforms_for(args.model, False))
    test = datasets.ImageFolder(DATA / "test", transforms_for(args.model, False))
    assert train_full.classes == CLASSES, train_full.classes
    idx = np.random.default_rng(0).permutation(len(train_full))
    n_val = len(idx) // 10
    train, val = Subset(train_full, idx[n_val:]), Subset(val_full, idx[:n_val])
    # ResNet50+ViT got the run killed for low RAM with worker processes; load in-process for it.
    workers = 0 if args.model == "resnet50_vit" else 2
    kw = dict(batch_size=batch, num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    tl = DataLoader(train, shuffle=True, **kw)
    vl, testl = DataLoader(val, **kw), DataLoader(test, **kw)

    # Class weights counter FER-2013's imbalance (disgust is ~1.5% of the data).
    counts = np.bincount([train_full.targets[i] for i in idx[n_val:]], minlength=7)
    weights = torch.tensor(counts.sum() / (7 * counts), dtype=torch.float32, device=device)

    model = build(args.model).to(device)
    lr = 1e-3 if args.model == "cnn" else 3e-4 if args.model in ("resnet18", "mobilenet_attn") else 2e-4
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * len(tl))
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    out = ROOT / "models" / CHECKPOINTS[args.model]
    best = -1.0
    print(f"{args.model}: {len(train)} train / {len(val)} val / {len(test)} test on {device}", flush=True)

    for ep in range(1, 0 if args.eval_only else epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(y)
        yv, pv = evaluate(model, vl, device)
        f1 = f1_score(yv, pv, average="macro")
        if f1 > best:
            best = f1
            torch.save(model.state_dict(), out)
        print(f"epoch {ep}/{epochs} loss {total / len(train):.4f} val_acc {accuracy_score(yv, pv):.4f} "
              f"val_macroF1 {f1:.4f} ({time.time() - t0:.0f}s){' *' if f1 == best else ''}", flush=True)

    model.load_state_dict(torch.load(out, weights_only=True))
    yt, pt = evaluate(model, testl, device)
    p, r, f, s = precision_recall_fscore_support(yt, pt, zero_division=0)

    # Inference time: single face crop, batch 1, on CPU (what the backend serves on) and GPU.
    timing = {}
    for dev in ["cpu"] + (["cuda"] if device.type == "cuda" else []):
        m = model.to(dev).eval()
        x = test[0][0].unsqueeze(0).to(dev)
        with torch.no_grad():
            for _ in range(5):
                m(x)
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(50):
                m(x)
            if dev == "cuda":
                torch.cuda.synchronize()
        timing[f"{dev}_ms_per_image"] = round((time.perf_counter() - t0) / 50 * 1000, 2)

    metrics = {
        "model": args.model, "dataset": "FER-2013", "epochs": epochs, "test_size": len(test),
        "accuracy": round(accuracy_score(yt, pt), 4),
        "macro_f1": round(f1_score(yt, pt, average="macro"), 4),
        "per_class": {c: {"precision": round(p[i], 4), "recall": round(r[i], 4), "f1": round(f[i], 4),
                          "support": int(s[i])} for i, c in enumerate(CLASSES)},
        "confusion_matrix": {"labels": CLASSES, "rows_true_cols_pred": confusion_matrix(yt, pt).tolist()},
        "inference_time": timing,
    }
    rep = ROOT / "project" / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / f"face_{args.model}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("accuracy", "macro_f1", "inference_time")}), flush=True)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
