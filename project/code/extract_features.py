r"""Step 1 of the faculty protocol: turn each trained model into a CSV of feature vectors.

Each row is one held-out sample: f0..fN = the model's penultimate-layer embedding (the input of its final
classifier layer), label = class index. Only samples the backbone never trained on are exported (the official
test split), so the protocol's own splits/folds are not contaminated by backbone training data.

Usage (repo root):
    project\.venv-train\Scripts\python project\code\extract_features.py --modality face --model densenet_se
    ... --modality text --model bert_bilstm_attn@goemotions    (text models are <model>@<dataset>)
Writes project/features/<modality>_<model>.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "project" / "features"
sys.path.insert(0, str(Path(__file__).parent))


def embed(model, n_classes, batches, device):
    """Capture the input of the last Linear layer that outputs n_classes logits."""
    fc = [m for m in model.modules() if isinstance(m, nn.Linear) and m.out_features == n_classes][-1]
    feats = []
    hook = fc.register_forward_pre_hook(lambda _m, inp: feats.append(inp[0].detach().float().cpu()))
    model.eval()
    with torch.no_grad():
        for x in batches:
            model(*[t.to(device) for t in x])
    hook.remove()
    return torch.cat(feats).numpy()


def face(name, device):
    import train_face as tf
    from torchvision import datasets
    model = tf.build(name)
    model.load_state_dict(torch.load(ROOT / "models" / tf.CHECKPOINTS[name], map_location="cpu", weights_only=True))
    model.to(device)
    test = datasets.ImageFolder(tf.DATA / "test", tf.transforms_for(name, False))
    dl = DataLoader(test, batch_size=64, num_workers=2)
    x = embed(model, 7, ((xb,) for xb, _ in dl), device)
    return x, np.array(test.targets), tf.CLASSES


def wesad(name, device):
    """Held-out test subjects only (S2, S9, S16): windows from people the model never saw."""
    import train_wesad as tw
    model = tw.BUILDERS[name]()
    model.load_state_dict(torch.load(ROOT / "models" / f"wesad_{name}.pt", map_location="cpu", weights_only=True))
    model.to(device)
    _, _, testl, _ = tw.loaders(64)
    ys = torch.cat([y for _, y in testl]).numpy()
    subj = np.load(tw.CACHE)["subj"]
    subj = subj[np.isin(subj, tw.SPLIT["test"])]
    # Windows overlap (60 s long, 5 s hop), so neighbours are near-duplicates. Group runs of 24 consecutive
    # windows (same subject + condition) so protocol.py never puts neighbours on both sides of a split.
    run = np.cumsum(np.r_[True, (subj[1:] != subj[:-1]) | (ys[1:] != ys[:-1])])
    pos = np.arange(len(ys)) - np.r_[0, np.flatnonzero(np.diff(run)) + 1][run - 1]
    groups = run * 10000 + pos // 24
    return embed(model, 3, ((x,) for x, _ in testl), device), ys, tw.CLASSES, groups


def audio(name, device):
    """Held-out 15% test clips (same stratified split as train_audio.py)."""
    import train_audio as ta
    inp, builder, _, _ = ta.MODELS[name]
    model = builder()
    model.load_state_dict(torch.load(ROOT / "models" / f"audio_{name}.pt", map_location="cpu", weights_only=True))
    model.to(device)
    d = np.load(ta.CACHE)
    y = d["y"]
    _, _, te = ta.splits(y, d["src"])
    if inp == "wav":
        x = torch.from_numpy(d["wav"][te]).float() / 32767
    else:
        x = torch.from_numpy(np.asarray(ta.encoder_features(inp, d["wav"], device))[te]).float()
    return embed(model, 7, ((xb,) for xb in x.split(32)), device), y[te], ta.CLASSES


def text(spec, device):
    """spec = <model>@<dataset>; official test split, never used in training."""
    import train_text as tt
    from transformers import AutoTokenizer
    name, dataset = spec.split("@")
    classes = tt.DATASETS[dataset]
    model = tt.TextModel(name, len(classes))
    model.load_state_dict(torch.load(ROOT / "models" / f"text_{name}_{dataset}.pt", map_location="cpu",
                                     weights_only=True))
    model.to(device)
    texts, ys = tt.load_split(dataset, "test")
    enc = AutoTokenizer.from_pretrained(tt.ENCODERS[tt.MODELS[name][0]])(
        texts, truncation=True, max_length=tt.MAX_LEN, padding="max_length", return_tensors="pt")
    batches = zip(enc["input_ids"].split(64), enc["attention_mask"].split(64))
    return embed(model, len(classes), batches, device), np.array(ys), classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["face", "wesad", "audio", "text"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")   # CPU by default: the GPU is busy with the training queue
    a = ap.parse_args()
    x, y, classes, *groups = {"face": face, "wesad": wesad, "audio": audio, "text": text}[a.modality](
        a.model, torch.device(a.device))
    df = pd.DataFrame(x.astype(np.float32), columns=[f"f{i}" for i in range(x.shape[1])])
    df["label"] = y
    if groups:                                     # optional: rows that must stay on the same side of every split
        df["group"] = groups[0]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{a.modality}_{a.model.replace('@', '_')}.csv"
    df.to_csv(path, index=False)
    print(f"{path}: {len(df)} rows x {x.shape[1]} features, classes {classes}, counts {np.bincount(y).tolist()}")


if __name__ == "__main__":
    main()
