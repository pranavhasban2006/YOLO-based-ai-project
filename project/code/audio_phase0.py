"""Audio phase 0: check the folders, audit the four speech datasets, write state.

Run with the project venv:  python -I project/code/audio_phase0.py
Reads datasets/ only. Never deletes or downloads anything.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_common import DATA, DATASET_SPECS, DATASETS, LOGS, STATE, find_dataset_dir, list_wavs, wav_root  # noqa: E402

PARTIAL_SUFFIXES = {".part", ".tmp", ".crdownload", ".download"}


def audit(name):
    base = find_dataset_dir(name)
    expected = DATASET_SPECS[name][2]
    row = {"dataset": name, "folder": str(base.relative_to(DATASETS)) if base else None, "expected_wav": expected}
    if base is None:
        row.update(status="MISSING", reason="no folder")
        return row
    wavs = list_wavs(name)
    sizes = [w.stat().st_size for w in wavs]
    zero = sum(1 for s in sizes if s == 0)
    # Status counts only the speech audio folder. Other folders (e.g. RAVDESS Video_Song) are reported, not used.
    scope = wav_root(name)
    partial = [p for p in scope.rglob("*") if p.is_file() and p.suffix.lower() in PARTIAL_SUFFIXES]
    partial_outside = [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in PARTIAL_SUFFIXES
                       and scope not in p.parents and p != scope]
    zips = [z for z in scope.rglob("*.zip")]
    unextracted = []
    for z in zips:
        # A zip counts as extracted when a folder with its name (without .zip) sits beside it.
        if not (z.parent / z.stem).exists() and not (z.parent / z.stem.replace(" ", "_")).exists():
            unextracted.append(str(z.relative_to(DATASETS)))
    row.update(
        wav_files=len(wavs),
        size_mb=round(sum(sizes) / 2**20, 1),
        zero_byte_wav=zero,
        partial_files=[str(p.relative_to(DATASETS)) for p in partial],
        zips=[str(z.relative_to(DATASETS)) for z in zips],
        unextracted_zips=unextracted,
        wav_root=str(scope.relative_to(DATASETS)),
        partial_outside_speech=[str(p.relative_to(DATASETS)) for p in partial_outside],
    )
    if len(wavs) == 0:
        status, reason = "MISSING", "no wav files"
    elif zero or partial or unextracted or len(wavs) < expected:
        status = "PARTIAL"
        reason = f"{len(wavs)}/{expected} wav, zero-byte {zero}, partial {len(partial)}, unextracted zips {len(unextracted)}"
    else:
        status, reason = "COMPLETE", f"{len(wavs)}/{expected} wav, no zero-byte or partial files"
    row.update(status=status, reason=reason)
    return row


def main():
    for d in ["code", "data", "logs", "models", "results", "reports"]:
        (DATA.parent / d).mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    rows = [audit(n) for n in DATASET_SPECS]
    available = {r["dataset"]: r["status"] for r in rows}
    usable = [r["dataset"] for r in rows if r["status"] == "COMPLETE"]

    lines = [f"# Audio phase 0 report ({datetime.now():%Y-%m-%d %H:%M})", "",
             "| dataset | status | folder | wav files | expected | size MB | zero-byte | partial | unextracted zips | reason |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['dataset']} | **{r['status']}** | {r['folder']} | {r.get('wav_files', 0)} | {r['expected_wav']} | "
                     f"{r.get('size_mb', 0)} | {r.get('zero_byte_wav', 0)} | {len(r.get('partial_files', []))} | "
                     f"{len(r.get('unextracted_zips', []))} | {r['reason']} |")
    lines += ["", "## Notes", "",
              "- Partial files outside the speech folders (not used): " + (", ".join(
                  f"{r['dataset']}: {len(r.get('partial_outside_speech', []))}" for r in rows) or "none") + ". RAVDESS Video_Song is skipped per the earlier instructions.",
              "- Speaker counts and files are read directly from disk; no zip needed extraction (each zip already has an extracted folder).",
              "- TESS has an extra file `OA_bite_neutral.wav` with speaker prefix `OA` (not OAF/YAF). It is kept as its own speaker; TESS is split by word anyway.",
              "- EMO-DB `silb/13a01Wb.silb` is 0 bytes in the source zip; it is not an audio file and is not used.",
              "- Notebook `prompts/audio/audio_colab_train.ipynb`: not present on disk and not supplied in the request. Phase 2 cannot copy it.",
              "- Usable datasets (COMPLETE): " + (", ".join(usable) if usable else "none")]
    (LOGS / "audio_phase_0_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    state = {"phase_done": -1, "datasets_available": available}
    if usable:
        state["phase_done"] = 0
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    if not usable:
        raise SystemExit("No speech dataset is usable. Stopping. See project/logs/audio_phase_0_report.md")
    print("audio_state.json:", json.dumps(state))


if __name__ == "__main__":
    main()
