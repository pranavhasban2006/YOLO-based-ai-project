"""Extract 16 kHz mono wav from every MELD mp4 listed in the csv files into project/data/meld_wav/.
Names: <split>_dia<D>_utt<U>.wav. Resumable (skips existing non-empty files). Run: python -I audio_meld_extract.py
"""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_common import MELD_DIRS, MELD_WAV, find_dataset_dir, meld_table  # noqa: E402

FFMPEG = r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe"


def one(job):
    src, dst = job
    if dst.exists() and dst.stat().st_size > 0:
        return "skip"
    if not src.exists():
        return "missing"
    r = subprocess.run([FFMPEG, "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(dst)],
                       capture_output=True)
    if r.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        dst.unlink(missing_ok=True)
        return "failed"
    return "ok"


def main():
    base = find_dataset_dir("meld") / "MELD.Raw"
    MELD_WAV.mkdir(parents=True, exist_ok=True)
    jobs = [(base / MELD_DIRS[s] / f"dia{d}_utt{u}.mp4", MELD_WAV / f"{s}_dia{d}_utt{u}.wav") for (s, d, u) in meld_table()]
    print(len(jobs), "jobs", flush=True)
    counts = {}
    with ThreadPoolExecutor(4) as ex:
        for i, st in enumerate(ex.map(one, jobs), 1):
            counts[st] = counts.get(st, 0) + 1
            if i % 500 == 0:
                print(i, counts, flush=True)
    print("done", counts, flush=True)


if __name__ == "__main__":
    main()
