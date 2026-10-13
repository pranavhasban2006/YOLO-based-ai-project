"""Audio phase 2: 16 kHz mono float32 clips, leakage-free splits, Colab zip.

Run with the project venv:  python -I project/code/audio_phase2.py
Precondition: project/audio_state.json phase_done >= 1.
Writes only under project/data/, project/logs/ and project/colab_upload_audio/.
"""
import csv
import json
import random
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_common import (CLASSES, DATA, DATASETS, LOGS, MAX_SECONDS, SEED, SR, STATE,  # noqa: E402
                          meld_split, tess_word)

OUT = DATA / "audio_16k_pcm"                                   # v2: 16-bit PCM, 7 datasets, capped at 6000 per dataset
UPLOAD = Path(__file__).resolve().parents[1] / "colab_upload_audio_v2"
NOTEBOOK_SRC = Path(__file__).resolve().parents[1] / "colab_upload_audio" / "audio_colab_train.ipynb"
CAP = 6000
INDEX_IN = DATA / "audio_index.csv"


def split_speakers(speakers, rng, n_val=None, n_test=None):
    """Split a list of speakers. Default ~70/15/15 (at least one speaker in each split)."""
    speakers = sorted(speakers)
    rng.shuffle(speakers)
    n = len(speakers)
    if n_val is None:
        n_val = max(1, round(0.15 * n))
    if n_test is None:
        n_test = max(1, round(0.15 * n))
    test = speakers[:n_test]
    val = speakers[n_test:n_test + n_val]
    train = speakers[n_test + n_val:]
    return {"train": set(train), "val": set(val), "test": set(test)}


def convert(src: Path, dst: Path):
    """Return (status, duration_sec). status: ok | skipped_existing | empty_after_trim."""
    if dst.exists() and dst.stat().st_size > 0:
        info = sf.info(str(dst))
        return "skipped_existing", round(info.frames / info.samplerate, 3)
    y, _ = librosa.load(str(src), sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    if y.size == 0:
        return "empty_after_trim", 0.0
    max_len = int(MAX_SECONDS * SR)
    if y.size > max_len:  # centre crop, no padding here (the notebook pads)
        start = (y.size - max_len) // 2
        y = y[start:start + max_len]
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), np.clip(y, -1.0, 1.0), SR, subtype="PCM_16")
    return "ok", round(y.size / SR, 3)


def convert_safe(job):
    src, dst = job
    try:
        status, dur = convert(src, dst)
    except Exception as exc:
        status, dur = f"error:{type(exc).__name__}", 0.0
    return dst, status, dur


def main():
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("phase_done", -1) < 1:
        raise SystemExit("Precondition failed: audio_state.json phase_done must be >= 1.")
    with INDEX_IN.open(encoding="utf-8") as f:
        clips = list(csv.DictReader(f))
    print(f"{len(clips)} clips in the index")

    # 0. cap each dataset at CAP clips: stratified by label, rarer classes kept whole first (water-filling), seed 42
    cap_rng = random.Random(SEED)
    cap_notes = []
    kept = []
    for name in sorted({c["dataset"] for c in clips}):
        ds = [c for c in clips if c["dataset"] == name]
        if len(ds) <= CAP:
            kept += ds
            continue
        by_label = defaultdict(list)
        for c in ds:
            by_label[c["label"]].append(c)
        quota, left = {}, CAP
        order = sorted(by_label, key=lambda k: len(by_label[k]))
        for i, k in enumerate(order):
            quota[k] = min(len(by_label[k]), left // (len(order) - i))
            left -= quota[k]
        for k in order:
            kept += cap_rng.sample(sorted(by_label[k], key=lambda c: c["path"]), quota[k])
        cap_notes.append(f"- {name}: capped {len(ds)} -> {sum(quota.values())} clips (stratified by label, rare classes first, seed 42): "
                         + ", ".join(f"{k} {quota[k]}/{len(by_label[k])}" for k in order))
    clips = kept
    print(f"{len(clips)} clips after the {CAP}-per-dataset cap")

    # 1. convert
    status_count = Counter()
    converted = []
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(DATASETS / c["path"], OUT / c["dataset"] / (Path(c["path"]).stem + ".wav")) for c in clips]
    with ProcessPoolExecutor(6) as ex:
        for i, (c, (dst, status, dur)) in enumerate(zip(clips, ex.map(convert_safe, jobs, chunksize=32)), 1):
            status_count[status] += 1
            if status in ("ok", "skipped_existing"):
                converted.append({**c, "dst": dst, "duration_16k": dur})
            if i % 1000 == 0:
                print(f"  converted {i}/{len(clips)} {dict(status_count)}", flush=True)
    print("conversion:", dict(status_count))

    # 2. speaker-level splits per dataset
    rng = random.Random(SEED)
    by_ds = defaultdict(list)
    for c in converted:
        by_ds[c["dataset"]].append(c)
    split_of_speaker = {}
    decisions = []
    for name in sorted(by_ds):
        if name == "tess":
            # Two real speakers (OAF, YAF) plus one stray file (OA). Split by word, not speaker (see DECISIONS.md).
            words = sorted({tess_word(Path(c["path"]).stem) for c in by_ds[name]})
            rng.shuffle(words)
            n = len(words)
            n_val = max(1, round(0.15 * n))
            n_test = max(1, round(0.15 * n))
            word_split = {}
            for w in words[:n_test]:
                word_split[w] = "test"
            for w in words[n_test:n_test + n_val]:
                word_split[w] = "val"
            for w in words[n_test + n_val:]:
                word_split[w] = "train"
            for c in by_ds[name]:
                c["split"] = word_split[tess_word(Path(c["path"]).stem)]
            decisions.append(f"- tess: split by word ({n} words: train/val/test by 70/15/15, seed 42). "
                             "Both speakers appear in every split, so this is NOT speaker-independent.")
        else:
            speakers = {c["speaker"] for c in by_ds[name]}
            if name == "meld":
                # official train/dev/test split (dev = val); speakers (cast members) overlap across splits
                for c in by_ds[name]:
                    c["split"] = {"train": "train", "dev": "val", "test": "test"}[meld_split(Path(c["path"]).stem)]
                decisions.append("- meld: official train/dev/test split (dev = val). The same cast members appear in all splits, "
                                 "so this is NOT speaker-independent.")
                continue
            if name == "savee":
                split = split_speakers(speakers, rng, n_val=1, n_test=1)  # 2 / 1 / 1 of 4 speakers
                decisions.append(f"- savee: speakers {len(speakers)} split 2/1/1 by speaker (seed 42): "
                                 f"train {sorted(split['train'])}, val {sorted(split['val'])}, test {sorted(split['test'])}.")
            elif name == "esd":
                split = split_speakers(speakers, rng, n_val=1, n_test=2)  # 7 / 1 / 2 of 10 speakers
                decisions.append(f"- esd: speakers {len(speakers)} split 7/1/2 by speaker (seed 42): "
                                 f"train {sorted(split['train'])}, val {sorted(split['val'])}, test {sorted(split['test'])}.")
            elif name == "emodb":
                split = split_speakers(speakers, rng, n_val=1, n_test=2)  # 7 / 1 / 2 of 10 speakers
                decisions.append(f"- emodb: speakers {len(speakers)} split 7/1/2 by speaker (seed 42): "
                                 f"train {sorted(split['train'])}, val {sorted(split['val'])}, test {sorted(split['test'])}.")
            else:
                split = split_speakers(speakers, rng)
            for s, sp in ((s, k) for k in split for s in split[k]):
                split_of_speaker[(name, s)] = sp
            for c in by_ds[name]:
                c["split"] = split_of_speaker[(name, c["speaker"])]
            decisions.append(f"- {name}: {len(speakers)} speakers, train {len(split['train'])} / val {len(split['val'])} / "
                             f"test {len(split['test'])} speakers (seed 42, by speaker, no overlap).")

    # 3. checks
    for name in by_ds:
        splits_of = defaultdict(set)
        for c in by_ds[name]:
            splits_of[c["split"]].add(c["speaker"] if name != "tess" else tess_word(Path(c["path"]).stem))
        for sp in ("train", "val", "test"):
            assert splits_of[sp], f"{name}: split {sp} is empty"
        if name not in ("tess", "meld"):
            for a in ("train", "val", "test"):
                for b in ("train", "val", "test"):
                    if a < b:
                        assert not (splits_of[a] & splits_of[b]), f"{name}: speaker overlap {a}/{b}"
        elif name == "tess":
            for a in ("train", "val", "test"):
                for b in ("train", "val", "test"):
                    if a < b:
                        assert not (splits_of[a] & splits_of[b]), f"tess: word overlap {a}/{b}"

    # 4. write the 16 kHz index
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in sorted(converted, key=lambda r: (r["dataset"], r["path"])):
        rows.append({"path": f"{c['dataset']}/{Path(c['path']).stem}.wav", "dataset": c["dataset"],
                     "speaker": c["speaker"], "label": c["label"], "split": c["split"]})
    with (OUT / "index.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["path", "dataset", "speaker", "label", "split"])
        wr.writeheader()
        wr.writerows(rows)

    # report: dataset x split x label
    table = defaultdict(Counter)
    for r in rows:
        table[(r["dataset"], r["split"])][r["label"]] += 1
    lines = [f"# Audio phase 2 report ({datetime.now():%Y-%m-%d %H:%M})", "",
             f"Clips written to `project/data/audio_16k/`: {len(rows)}", "",
             "Conversion: " + ", ".join(f"{k} {v}" for k, v in status_count.items()), "",
             "## Size cap", ""] + (cap_notes or ["- none"]) + ["", "## Splits by speaker", ""] + [f"- {d}" for d in decisions] + [
             "", "## Dataset x split x label", "",
             "| dataset | split | " + " | ".join(CLASSES) + " | total |",
             "|---|---|" + "---|" * (len(CLASSES) + 1)]
    for (ds, sp) in sorted(table):
        cells = [str(table[(ds, sp)][c]) for c in CLASSES]
        lines.append(f"| {ds} | {sp} | " + " | ".join(cells) + f" | {sum(table[(ds, sp)].values())} |")
    lines += ["", "Overlap check: no speaker appears in two splits (asserted) for ravdess, crema_d, emodb, savee and esd. "
              "TESS: no word appears in two splits (asserted); speakers overlap by design. MELD: official split, cast members overlap."]
    (LOGS / "audio_phase_2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # 5. Colab zip (stored, no compression)
    UPLOAD.mkdir(parents=True, exist_ok=True)
    zip_path = UPLOAD / "audio_16k.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        z.write(OUT / "index.csv", "index.csv")
        for r in rows:
            z.write(OUT / r["path"], r["path"])
    if NOTEBOOK_SRC.exists():
        import shutil
        shutil.copy2(NOTEBOOK_SRC, UPLOAD / NOTEBOOK_SRC.name)
        nb = "copied"
    else:
        nb = f"not copied: {NOTEBOOK_SRC.relative_to(DATASETS.parent)} does not exist and no copy was supplied"
    size_mb = zip_path.stat().st_size / 2**20
    print(f"zip: {zip_path} {size_mb:.1f} MB ({size_mb / 1024:.2f} GB)")
    print("notebook:", nb)
    with (LOGS / "audio_phase_2_report.md").open("a", encoding="utf-8") as f:
        f.write(f"\n## Colab package\n\n- {zip_path.relative_to(DATA.parent.parent)}: {size_mb:.1f} MB\n- Notebook: {nb}\n")

    state["phase_done"] = 2
    state["audio_16k_clips"] = len(rows)
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print("phase_done = 2")


if __name__ == "__main__":
    main()
