"""Phase 0 dataset audit. Read-only: never downloads, moves or deletes under datasets/.

Writes project/logs/audit_datasets.json and project/logs/audit_datasets.md.
Run with: project\\.venv\\Scripts\\python.exe -I project\\code\\audit_datasets.py
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"
LOGS = ROOT / "project" / "logs"

PARTIAL_EXT = {".part", ".tmp", ".crdownload", ".download"}
LFS_HEADER = b"version https://git-lfs.github.com/spec"

# Each entry: directories that make up the dataset, the subset that decides
# COMPLETE/PARTIAL, and expected file counts (None = not checked).
SPECS = {
    "fer2013": {
        "paths": ["FER-2013"],          # canonical copy per manifest.json
        "extra_paths": ["fer2013"],     # old partial copy, reported only
        "expected": {"extracted/train": 28709, "extracted/test": 7178},
        "zip_glob": ["FER-2013/*.zip"],
        "use_subset": ["FER-2013/extracted"],
    },
    "ferplus": {
        "paths": ["ferplus"],
        "expected": {"fer2013new.csv": None},
        "zip_glob": [],
        "use_subset": ["ferplus"],
    },
    "jaffe": {
        "paths": ["jaffe"],
        "expected": {"extracted": None},
        "zip_glob": ["jaffe/*.zip"],
        "use_subset": ["jaffe/extracted"],
    },
    "ravdess": {
        "paths": ["RAVDESS"],
        "expected": {"Audio_Speech": None, "Video_Speech": None},
        "zip_glob": ["RAVDESS/**/*.zip"],
        # Song videos are not used by the project (see DECISIONS.md), so they
        # are reported but do not affect COMPLETE/PARTIAL.
        "use_subset": ["RAVDESS/Audio_Speech", "RAVDESS/Video_Speech"],
    },
    "crema_d": {
        "paths": ["crema_d"],
        "expected": {"AudioWAV": 7442, "AudioMP3": 7442},
        "zip_glob": [],
        "use_subset": ["crema_d/AudioWAV", "crema_d/AudioMP3"],
    },
}


def walk_files(base: Path):
    if not base.exists():
        return []
    return [p for p in base.rglob("*") if p.is_file() and ".git" not in p.parts]


def count_dir(base: Path):
    files = walk_files(base)
    return {
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "partial": [str(p.relative_to(DATASETS)) for p in files if p.suffix.lower() in PARTIAL_EXT],
        "zero_byte": [str(p.relative_to(DATASETS)) for p in files if p.stat().st_size == 0],
        "lfs_pointers": [
            str(p.relative_to(DATASETS))
            for p in files
            if 0 < p.stat().st_size < 1024 and p.read_bytes().startswith(LFS_HEADER)
        ],
    }


def unzip_test(zip_path: Path):
    proc = subprocess.run(
        ["unzip", "-t", str(zip_path)], capture_output=True, text=True, errors="replace"
    )
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-1:] or [""]
    return {"ok": proc.returncode == 0, "summary": tail[0]}


def expected_check(rel_base: Path, expected: dict):
    results = {}
    for name, want in expected.items():
        target = DATASETS / rel_base / name if name else DATASETS / rel_base
        if not target.exists():
            results[name] = {"found": False, "expected": want}
            continue
        n = len(walk_files(target)) if target.is_dir() else None
        results[name] = {"found": True, "files": n, "expected": want}
    return results


def audit_one(key: str, spec: dict):
    out = {"dataset": key, "exists": False, "status": "MISSING", "reasons": []}
    present = [p for p in spec["paths"] if (DATASETS / p).exists()]
    if not present:
        out["reasons"].append("no directory found under datasets/")
        return out
    out["exists"] = True
    out["folders"] = present

    all_stats = {}
    for p in spec["paths"] + spec.get("extra_paths", []):
        if (DATASETS / p).exists():
            all_stats[p] = count_dir(DATASETS / p)
    out["whole_tree"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("partial", "zero_byte", "lfs_pointers")} | {
        "partial_files": len(v["partial"]),
        "zero_byte_files": len(v["zero_byte"]),
        "lfs_pointer_files": len(v["lfs_pointers"]),
    } for k, v in all_stats.items()}
    out["partial_files"] = [f for v in all_stats.values() for f in v["partial"]]
    out["zero_byte_files"] = [f for v in all_stats.values() for f in v["zero_byte"]]
    out["lfs_pointer_files"] = [f for v in all_stats.values() for f in v["lfs_pointers"]]

    zips = []
    for pattern in spec["zip_glob"]:
        for z in DATASETS.glob(pattern):
            zips.append({"path": str(z.relative_to(DATASETS)), "bytes": z.stat().st_size, **unzip_test(z)})
    out["zips"] = zips

    # Extraction status: a zip is "extracted" if a sibling folder with the same stem exists
    # or, for fer2013/jaffe, the extracted/ folder is non-empty.
    out["zips_extracted"] = {}
    for z in zips:
        stem = Path(z["path"]).stem
        zpath = DATASETS / z["path"]
        out["zips_extracted"][z["path"]] = (zpath.parent / stem).exists() or (zpath.parent / "extracted").exists()

    use_partial, use_zero, use_lfs, use_files = [], [], [], 0
    for sub in spec["use_subset"]:
        s = count_dir(DATASETS / sub)
        use_files += s["files"]
        use_partial += s["partial"]
        use_zero += s["zero_byte"]
        use_lfs += s["lfs_pointers"]
    out["use_subset"] = spec["use_subset"]
    out["use_subset_files"] = use_files
    out["expected"] = expected_check(Path(spec["paths"][0]), spec["expected"])

    reasons = []
    if use_partial:
        reasons.append(f"{len(use_partial)} partial/temp file(s) in used subset")
    if use_zero:
        reasons.append(f"{len(use_zero)} zero-byte file(s) in used subset")
    if use_lfs:
        reasons.append(f"{len(use_lfs)} git-lfs pointer stub(s) in used subset")
    if any(not z["ok"] for z in zips):
        reasons.append("unzip -t failed for: " + ", ".join(z["path"] for z in zips if not z["ok"]))
    for name, info in out["expected"].items():
        if not info["found"]:
            reasons.append(f"expected '{name}' not found")
        elif info.get("expected") is not None and info.get("files") is not None and info["files"] != info["expected"]:
            reasons.append(f"'{name}' has {info['files']} files, expected {info['expected']}")
    if use_files == 0:
        reasons.append("used subset is empty")

    if use_files == 0:
        out["status"] = "MISSING"
    elif reasons:
        out["status"] = "PARTIAL"
    else:
        out["status"] = "COMPLETE"
    out["reasons"] = reasons
    return out


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    results = {k: audit_one(k, v) for k, v in SPECS.items()}
    for r in results.values():
        print(f"{r['dataset']:10s} {r['status']:9s} files={r.get('use_subset_files', 0)} {('; '.join(r['reasons']) or 'ok')}")

    (LOGS / "audit_datasets.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    lines = [f"# Dataset audit ({datetime.now().isoformat(timespec='seconds')})", ""]
    lines += ["| dataset | status | used files | zips (unzip -t) | reasons |", "|---|---|---|---|---|"]
    for r in results.values():
        zs = ", ".join(f"{z['path']}: {'ok' if z['ok'] else 'FAIL'}" for z in r.get("zips", [])) or "none"
        lines.append(f"| {r['dataset']} | {r['status']} | {r.get('use_subset_files', 0)} | {zs} | {'; '.join(r['reasons']) or '-'} |")
    lines.append("")
    for r in results.values():
        lines.append(f"## {r['dataset']} — {r['status']}")
        if not r["exists"]:
            lines += ["Not found.", ""]
            continue
        lines.append(f"- folders: {', '.join(r['folders'])}")
        for k, v in r.get("whole_tree", {}).items():
            lines.append(f"- {k}: {v['files']} files, {v['bytes'] / 1e9:.3f} GB, partial={v['partial_files']}, zero-byte={v['zero_byte_files']}, lfs-pointers={v['lfs_pointer_files']}")
        for name, info in r.get("expected", {}).items():
            lines.append(f"- expected `{name}`: found={info['found']} files={info.get('files')} expected={info.get('expected')}")
        for z in r.get("zips", []):
            lines.append(f"- zip `{z['path']}` {z['bytes'] / 1e6:.1f} MB unzip -t: {z['summary']} extracted={r['zips_extracted'].get(z['path'])}")
        if r["partial_files"]:
            lines.append("- partial files: " + ", ".join(r["partial_files"]))
        if r["zero_byte_files"]:
            lines.append("- zero-byte files: " + ", ".join(r["zero_byte_files"][:20]))
        if r["reasons"]:
            lines.append("- reasons: " + "; ".join(r["reasons"]))
        lines.append("")
    (LOGS / "audit_datasets.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", LOGS / "audit_datasets.md")


if __name__ == "__main__":
    sys.exit(main())
