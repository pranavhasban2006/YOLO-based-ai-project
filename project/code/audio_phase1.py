"""Audio phase 1: one index of every usable speech clip (label, speaker, duration, sample rate).

Run with the project venv:  python -I project/code/audio_phase1.py
Precondition: project/audio_state.json phase_done >= 0.
"""
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_common import (CLASSES, DATA, DATASET_SPECS, DATASETS, INDEX, LOGS, PARSERS,  # noqa: E402
                          STATE, find_dataset_dir, list_wavs, wav_root)

SILENCE_PEAK = 1e-4


def layout_text(name, wavs, excluded):
    root = wav_root(name)
    lines = [f"Dataset: {name}", f"Folder: {os.path.relpath(root, DATASETS).replace(chr(92), '/')}", f"Wav files: {len(wavs)}",
             "Example file names (first 5 sorted):"]
    lines += [f"  {w.name}" for w in wavs[:5]]
    subdirs = sorted({str(w.parent.relative_to(root)) for w in wavs}) or ["."]
    lines.append(f"Distinct folders holding wavs: {len(subdirs)} (first 3: {subdirs[:3]})")
    lines.append("Parse rule: " + {
        "ravdess": "modality-channel-emotion-intensity-statement-repetition-actor; channel 01 only; emotion 01..08 -> neutral(01,02), happy, sad, angry, fear, disgust, surprise; speaker = actor (field 7)",
        "crema_d": "ActorID_Sentence_Emotion_Level; speaker = ActorID; ANG DIS FEA HAP NEU SAD; no surprise",
        "tess": "OAF/YAF_<word>_<emotion>; speaker = prefix; emotion = last token, ps -> surprise",
        "emodb": "2-digit speaker, 3-char text code, emotion letter, version; W A E F T N map; L (boredom) excluded",
        "savee": "<speaker>_<code><nn>; codes a d f h n sa su; speaker = DC JE JK KL",
        "esd": "<speaker>_<nnnnnn>; utterance blocks of 350: neutral, angry, happy, sad, surprise; speaker = 4-digit id; no fear/disgust",
        "meld": "<split>_dia<D>_utt<U> extracted from mp4; label and speaker from the csv (joy->happy, anger->angry, sadness->sad); official train/dev/test split in the file name",
    }[name])
    lines.append(f"Excluded during parse or read: {excluded}")
    return "\n".join(lines) + "\n"


def main():
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    if state.get("phase_done", -1) < 0:
        raise SystemExit("Precondition failed: audio_state.json phase_done must be >= 0 (run audio_phase0.py first).")
    avail = [n for n, s in state["datasets_available"].items() if s == "COMPLETE"]
    LOGS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    rows, excluded_log = [], []
    for name in avail:
        parser = PARSERS[name]
        wavs = list_wavs(name)
        excl_counts = Counter()
        for w in wavs:
            rel = os.path.relpath(w, DATASETS).replace("\\", "/")
            speaker, label_or_reason = parser(w.stem)
            if speaker is None:
                excl_counts[label_or_reason] += 1
                excluded_log.append((rel, label_or_reason))
                continue
            label = label_or_reason
            if label not in CLASSES:
                excl_counts["not_in_7_classes"] += 1
                excluded_log.append((rel, "not_in_7_classes"))
                continue
            try:
                info = sf.info(str(w))
                data, _ = sf.read(str(w), dtype="float32", always_2d=False)
            except Exception as exc:  # unreadable file
                excl_counts["unreadable"] += 1
                excluded_log.append((rel, f"unreadable: {type(exc).__name__}"))
                continue
            peak = float(np.max(np.abs(data))) if data.size else 0.0
            if peak < SILENCE_PEAK:
                excl_counts["silent"] += 1
                excluded_log.append((rel, f"silent peak {peak:.2e}"))
                continue
            rows.append({
                "path": rel,
                "dataset": name,
                "speaker": speaker,
                "label": label,
                "duration_sec": round(info.frames / info.samplerate, 3),
                "sample_rate": info.samplerate,
            })
        layout = layout_text(name, wavs, dict(excl_counts))
        (LOGS / f"audio_layout_{name}.txt").write_text(layout, encoding="utf-8")
        print(layout)

    with INDEX.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["path", "dataset", "speaker", "label", "duration_sec", "sample_rate"])
        wr.writeheader()
        wr.writerows(rows)

    # dataset x label table
    counts = defaultdict(Counter)
    speakers = defaultdict(set)
    for r in rows:
        counts[r["dataset"]][r["label"]] += 1
        speakers[r["dataset"]].add(r["speaker"])
    header = "| dataset | speakers | " + " | ".join(CLASSES) + " | total |"
    sep = "|---|---|" + "---|" * (len(CLASSES) + 1)
    table = [header, sep]
    absent = []
    for name in avail:
        cells = [str(counts[name][c]) for c in CLASSES]
        table.append(f"| {name} | {len(speakers[name])} | " + " | ".join(cells) + f" | {sum(counts[name].values())} |")
        miss = [c for c in CLASSES if counts[name][c] == 0]
        if miss:
            absent.append(f"- {name}: no clips for {', '.join(miss)}")
    tot = Counter()
    for name in avail:
        tot.update(counts[name])
    table.append("| **all** | " + str(sum(len(speakers[n]) for n in avail)) + " | " +
                 " | ".join(str(tot[c]) for c in CLASSES) + f" | {sum(tot.values())} |")

    ex = Counter(reason.split(":")[0] for _, reason in excluded_log)
    report = [f"# Audio phase 1 report ({datetime.now():%Y-%m-%d %H:%M})", "",
              f"Index: `project/data/audio_index.csv`, {len(rows)} clips.", "",
              "## Clips per dataset and label", "", *table, "",
              "## Classes absent per dataset", "", *(absent or ["- none"]), "",
              "Known by design: CREMA-D has no surprise; EMO-DB has no surprise (and boredom is excluded, see DECISIONS.md).",
              "", "## Excluded clips (not in the index)", "",
              f"Total excluded: {len(excluded_log)}"]
    report += [f"- {k}: {v}" for k, v in ex.most_common()]
    (LOGS / "audio_phase_1_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))

    state["phase_done"] = 1
    state["index_clips"] = len(rows)
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print("phase_done = 1")


if __name__ == "__main__":
    main()
