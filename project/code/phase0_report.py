"""Phase 0 environment report: GPU, torch, disk, dataset status and mediapipe.

Writes project/logs/phase_0_report.md. Read-only apart from that file.
Run with: project\\.venv\\Scripts\\python.exe project\\code\\phase0_report.py
"""
import json
import shutil
import sys
from datetime import datetime
from importlib import metadata

import torch

from common import CLASSES, DATASETS_DIR, PROJECT_DIR, REPO_ROOT, ensure_dirs, get_device, project_path, read_state, write_state

AUDIT_JSON = project_path("logs", "audit_datasets.json")
REPORT = project_path("logs", "phase_0_report.md")


def pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def gpu_lines() -> list[str]:
    lines = [f"- CUDA available: {torch.cuda.is_available()}", f"- torch CUDA build: {torch.version.cuda}"]
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        lines.append(f"- GPU {i}: {props.name}, {total / 2**30:.2f} GiB total, {free / 2**30:.2f} GiB free")
    if torch.cuda.device_count() == 0:
        lines.append("- No CUDA device visible to torch; training would run on CPU.")
    return lines


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ensure_dirs()
    disk = shutil.disk_usage(REPO_ROOT)
    device = get_device()
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))

    lines = [
        "# Phase 0 report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Environment",
        f"- Python: {sys.version.split()[0]} ({sys.executable})",
        f"- torch: {torch.__version__}",
        f"- torchvision: {pkg_version('torchvision')}",
        f"- Selected device: {device}",
        *gpu_lines(),
        f"- Free disk on {REPO_ROOT.anchor}: {disk.free / 2**30:.1f} GiB of {disk.total / 2**30:.1f} GiB",
        "",
        "## Packages",
    ]
    for name in ["numpy", "pandas", "scikit-learn", "opencv-contrib-python", "pillow", "tqdm",
                 "matplotlib", "seaborn", "mediapipe"]:
        lines.append(f"- {name}: {pkg_version(name)}")
    lines += [
        "",
        "Face detection: mediapipe `tasks` FaceDetector (uses models/blaze_face_short_range.tflite)."
        " OpenCV Haar cascades are bundled as a fallback.",
        "",
        "## Dataset audit",
        f"Source: project/logs/audit_datasets.md (written {datetime.fromtimestamp(AUDIT_JSON.stat().st_mtime).isoformat(timespec='seconds')}).",
        "",
        "| dataset | status | used files | path |",
        "|---|---|---|---|",
    ]
    for name, info in audit.items():
        lines.append(f"| {name} | {info['status']} | {info['use_subset_files']} | {', '.join(info['folders'])} |")
    lines += ["", f"Classes: {', '.join(CLASSES)}", f"Datasets root: {DATASETS_DIR}", f"Project root: {PROJECT_DIR}", ""]

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    state = read_state()
    state["datasets_available"] = {
        name: "available" if info["status"] == "COMPLETE" else "skipped" for name, info in audit.items()
    }
    write_state(state)
    print("\n".join(lines[:20]))
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
