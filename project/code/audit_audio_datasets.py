# Phase 0 audit: walk datasets/ once and report per-dataset counts, sizes, partial and zero-byte files, zips.
import json, os, sys

ROOT = sys.argv[1]
DATASETS = {
    "ravdess": ["RAVDESS", "ravdess", "ravdess_speech"],
    "crema_d": ["crema_d", "CREMA-D", "crema-d"],
    "tess": ["tess", "TESS"],
    "savee": ["savee", "SAVEE"],
    "meld": ["meld", "MELD", "MELD.Raw"],
    "esd": ["esd", "ESD"],
    "emodb": ["emodb", "EMO-DB"],
}
PARTIAL = (".part", ".tmp", ".crdownload", ".download")
AUDIO = (".wav", ".mp4")

out = {}
for name, folders in DATASETS.items():
    # Windows paths are case-insensitive, so take only the first matching folder.
    found = [f for f in folders if os.path.isdir(os.path.join(ROOT, f))][:1]
    if not found:
        out[name] = {"status": "MISSING"}
        continue
    n = size = partial = zero = audio = 0
    zips = []
    for f in found:
        for dp, dn, fn in os.walk(os.path.join(ROOT, f)):
            for x in fn:
                p = os.path.join(dp, x)
                try:
                    s = os.path.getsize(p)
                except OSError:
                    continue
                n += 1; size += s
                low = x.lower()
                if low.endswith(PARTIAL): partial += 1
                if s == 0: zero += 1
                if low.endswith(AUDIO): audio += 1
                if low.endswith(".zip"):
                    stem = os.path.splitext(p)[0]
                    zips.append({"zip": os.path.relpath(p, ROOT).replace("\\", "/"),
                                 "bytes": s, "extract_dir_exists": os.path.isdir(stem)})
    out[name] = {"folders": found, "files": n, "audio_files": audio, "bytes": size,
                 "partial": partial, "zero_byte": zero, "zips": zips}

top = sorted(os.listdir(ROOT))
out["_top_level"] = top
print(json.dumps(out, indent=1))
