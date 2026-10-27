r"""Exports the A1 -> A2 cascade results (Cascade_results/reports/*.json, written by cascade.py) into one compact file
the dashboard's "A1 -> A2 Results" tab reads: frontend/src/data/cascadeResults.json.

Usage (repo root): project\.venv-train\Scripts\python project\code\export_cascade_for_app.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "Cascade_results" / "reports"
OUT = ROOT / "frontend" / "src" / "data" / "cascadeResults.json"
ORDER = {"speech": 0, "face": 1, "text": 2}

rows = []
for f in REP.glob("*.json"):
    r = json.loads(f.read_text())
    s1, s2, w = r["stage1_A1_alone"], r["stage2_A1_then_A2"], r.get("wrong_only_to_A2")
    if w is None:
        raise SystemExit(f"{f.name} has no wrong-only result: run `cascade.py summary` first")
    rows.append({
        "id": f.stem, "modality": r["modality"], "dataset": r["dataset"], "A1": r["A1"], "A2": r["A2"],
        "split": r["split"], "testSize": r["test_size"], "classes": r["classes"],
        "a1": {"acc": s1["accuracy"], "f1": s1["macro_f1"], "correct": s1["correct"], "wrong": s1["wrong"]},
        "wrongOnly": {"acc": w["accuracy"], "f1": w["macro_f1"], "fed": w["fed_to_A2"], "fixed": w["fixed_by_A2"]},
        "allSamples": {"acc": s2["accuracy"], "f1": s2["macro_f1"], "fixed": r["A1_wrong_fixed_by_A2"],
                       "broken": r["A1_right_broken_by_A2"]},
        "confusionA1": r["confusion_A1"],
        "confusionFinal": w.get("confusion"),
    })
rows.sort(key=lambda x: (ORDER[x["modality"]], x["dataset"], x["A1"], x["A2"]))
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(rows, separators=(",", ":")))
print(f"{len(rows)} hybrids -> {OUT} ({OUT.stat().st_size // 1024} KB)")
