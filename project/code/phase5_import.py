"""Phase 5: check and import the Colab results.

Usage: python -I phase5_import.py <project_dir>

Writes project/logs/phase_5_import.json with everything it found. Does not touch state.json;
the caller decides that from the summary.
"""
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

ROOT = Path(sys.argv[1]).resolve()
SRC = ROOT / "colab_results"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
LOG = ROOT / "logs" / "phase_5_import.json"
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
SUFFIX = "__colab_import"


class SmallCNN(nn.Module):
    """Copied from project/code/build_notebook.py (SmallCNN), so the strict load is against the real layout."""
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


def build(arch):
    if arch == "cnn":
        return SmallCNN(), (1, 1, 48, 48)
    m = torchvision.models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    return m, (1, 3, 224, 224)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def arch_from_name(name):
    if "resnet18" in name:
        return "resnet18"
    if "cnn" in name:
        return "cnn"
    return None


def check_checkpoint(path, arch):
    """Load on CPU, strict-load into the architecture, run one forward pass. Returns (ok, detail)."""
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        return False, f"torch.load failed: {type(e).__name__}: {e}"
    sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    if not isinstance(sd, dict) or not all(torch.is_tensor(v) for v in sd.values()):
        return False, "not a state_dict"
    model, shape = build(arch)
    try:
        model.load_state_dict(sd, strict=True)
    except Exception as e:
        return False, f"strict load into {arch} failed: {type(e).__name__}: {str(e)[:300]}"
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(shape))
    if tuple(out.shape) != (1, len(CLASSES)):
        return False, f"unexpected output shape {tuple(out.shape)}"
    return True, f"ok, {sum(v.numel() for v in sd.values())} values, output {tuple(out.shape)}"


def parse_json(path):
    try:
        return True, json.loads(path.read_text(encoding="utf-8")), ""
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


def parse_csv(path):
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows or not rows[0]:
            return False, None, "no rows or no header"
        return True, rows, ""
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


def copy_safe(src, dst):
    """Copy src to dst. Identical file: skip. Different file already there: copy under a suffixed name."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
        return {"src": str(src.relative_to(ROOT)), "dst": str(dst.relative_to(ROOT)), "action": "copied"}
    if sha(src) == sha(dst):
        return {"src": str(src.relative_to(ROOT)), "dst": str(dst.relative_to(ROOT)), "action": "identical, skipped"}
    alt = dst.with_name(f"{dst.stem}{SUFFIX}{dst.suffix}")
    if alt.exists() and sha(src) == sha(alt):
        return {"src": str(src.relative_to(ROOT)), "dst": str(alt.relative_to(ROOT)), "action": "identical copy already present, skipped"}
    if alt.exists():
        alt = alt.with_name(f"{alt.stem}_{sha(src)[:8]}{alt.suffix}")
    shutil.copy2(src, alt)
    return {"src": str(src.relative_to(ROOT)), "dst": str(alt.relative_to(ROOT)),
            "action": f"CONFLICT: existing {dst.name} differs, saved under new name"}


def main():
    summary = {"checkpoints": {}, "files": {}, "copies": [], "blocking": [], "unparseable": [], "stop_reason": None}

    if not SRC.is_dir():
        summary["stop_reason"] = "project/colab_results/ does not exist"
        return summary

    # Step 1: checkpoints
    run_dirs = sorted(p for p in SRC.iterdir() if p.is_dir() and (p / "best.pt").exists())
    if not run_dirs:
        summary["stop_reason"] = "no run folder with best.pt"
        return summary
    for run in run_dirs:
        arch = arch_from_name(run.name)
        entry = {"arch_from_folder": arch}
        if arch is None:
            entry["best"] = (False, "cannot tell architecture from folder name")
            summary["blocking"].append(f"{run.name}: architecture unknown")
        else:
            entry["best"] = check_checkpoint(run / "best.pt", arch)
            if (run / "last.pt").exists():
                entry["last"] = check_checkpoint(run / "last.pt", arch)
            if not entry["best"][0]:
                summary["blocking"].append(f"{run.name}/best.pt: {entry['best'][1]}")
        summary["checkpoints"][run.name] = entry

    top_best = SRC / "best_model.pt"
    if top_best.exists():
        winner_arch = json.loads((SRC / "choice.json").read_text())["winner"] if (SRC / "choice.json").exists() else None
        entry = {"arch_from_choice": winner_arch, "best": check_checkpoint(top_best, winner_arch or "cnn")}
        e2 = SRC / f"E2_{winner_arch}" / "best.pt" if winner_arch else None
        entry["same_bytes_as_E2_winner"] = bool(e2 and e2.exists() and sha(e2) == sha(top_best))
        summary["checkpoints"]["best_model.pt"] = entry
        if not entry["best"][0]:
            summary["blocking"].append(f"best_model.pt: {entry['best'][1]}")

    # Step 2: classes
    cls_path = SRC / "classes.json"
    ok, cls, err = parse_json(cls_path) if cls_path.exists() else (False, None, "missing")
    summary["classes"] = {"found": cls, "expected": CLASSES, "match": ok and cls == CLASSES}
    if not summary["classes"]["match"]:
        summary["stop_reason"] = f"classes.json differs from expected: {cls if ok else err}"
        return summary

    # Step 3: parse metrics and CSVs
    for run in run_dirs:
        for name in ["metrics.json", "done.json"]:
            p = run / name
            if not p.exists():
                summary["unparseable"].append(f"{run.name}/{name}: missing")
                continue
            ok, _, err = parse_json(p)
            if not ok:
                summary["unparseable"].append(f"{run.name}/{name}: {err}")
        p = run / "history.csv"
        if p.exists():
            ok, _, err = parse_csv(p)
            if not ok:
                summary["unparseable"].append(f"{run.name}/history.csv: {err}")
        else:
            summary["unparseable"].append(f"{run.name}/history.csv: missing")
        m = run / "metrics.json"
        if m.exists() and parse_json(m)[0]:
            mj = parse_json(m)[1]
            summary["files"][f"{run.name}/metrics"] = {"arch_in_file": mj.get("arch"),
                                                       "arch_match": mj.get("arch") == arch_from_name(run.name),
                                                       "tests": sorted(mj.get("tests", {}).keys())}
    for name in ["compare_E2.csv", "E3_summary.csv"]:
        ok, rows, err = parse_csv(SRC / name) if (SRC / name).exists() else (False, None, "missing")
        if not ok:
            summary["unparseable"].append(f"{name}: {err}")
        else:
            summary["files"][name] = {"rows": len(rows), "columns": list(rows[0].keys())}
    ok, _, err = parse_json(SRC / "choice.json") if (SRC / "choice.json").exists() else (False, None, "missing")
    if not ok:
        summary["unparseable"].append(f"choice.json: {err}")

    # Step 4: copy. Nothing is copied if a checkpoint failed to load.
    if summary["blocking"]:
        summary["stop_reason"] = "checkpoint check failed, nothing copied"
        return summary
    for run in run_dirs:
        for name in ["metrics.json", "history.csv", "done.json"] + [p.name for p in run.glob("confusion_*.png")]:
            if (run / name).exists():
                summary["copies"].append(copy_safe(run / name, RESULTS / run.name / name))
        for ck in ["best.pt", "last.pt"]:
            if (run / ck).exists():
                summary["copies"].append(copy_safe(run / ck, MODELS / run.name / ck))
    for name in ["compare_E2.csv", "E3_summary.csv", "E3_summary.png", "choice.json", "classes.json"]:
        if (SRC / name).exists():
            summary["copies"].append(copy_safe(SRC / name, RESULTS / name))
    if top_best.exists():
        summary["copies"].append(copy_safe(top_best, MODELS / "best_model.pt"))
    return summary


if __name__ == "__main__":
    result = main()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["stop_reason", "blocking", "unparseable"]}, indent=2, default=str))
    print("copies:", len(result["copies"]))
