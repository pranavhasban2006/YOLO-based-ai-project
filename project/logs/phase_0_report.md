# Phase 0 report

Generated: 2026-10-10T00:53:02

## Environment
- Python: 3.13.5 (D:\Projects\YOLO\project\.venv\Scripts\python.exe)
- torch: 2.11.0+cu128
- torchvision: 0.26.0+cu128
- Selected device: cuda:0
- CUDA available: True
- torch CUDA build: 12.8
- GPU 0: NVIDIA GeForce GTX 1650 Ti, 4.00 GiB total, 3.21 GiB free
- Free disk on D:\: 437.2 GiB of 931.5 GiB

## Packages
- numpy: 2.5.3
- pandas: 3.0.6
- scikit-learn: 1.9.1
- opencv-contrib-python: 5.0.0.93
- pillow: 12.3.0
- tqdm: 4.70.1
- matplotlib: 3.11.2
- seaborn: 0.13.2
- mediapipe: 1.1.0

Face detection: mediapipe `tasks` FaceDetector (uses models/blaze_face_short_range.tflite). OpenCV Haar cascades are bundled as a fallback.

## Dataset audit
Source: project/logs/audit_datasets.md (written 2026-10-09T23:27:13).

| dataset | status | used files | path |
|---|---|---|---|
| fer2013 | COMPLETE | 35887 | FER-2013 |
| ferplus | COMPLETE | 14 | ferplus |
| jaffe | COMPLETE | 214 | jaffe |
| ravdess | COMPLETE | 4345 | RAVDESS |
| crema_d | COMPLETE | 14884 | crema_d |

Classes: angry, disgust, fear, happy, neutral, sad, surprise
Datasets root: D:\Projects\YOLO\datasets
Project root: D:\Projects\YOLO\project
## Post-audit check (2026-10-10, os.walk over datasets/)
- 102,411 files scanned. None newer than the audit (23:27 on 2026-10-09). No .part/.tmp/.crdownload/.download files.
- 132 zero-byte files, all under `.git` folders (the audit skips `.git`, so it did not count them):
  - 130 in `datasets/crema_d/.git/lfs/tmp/` (git-lfs temp leftovers from an interrupted pull)
  - 1 `datasets/crema_d/.git/index.lock` (stale lock file)
  - 1 `datasets/emodb/silb/13a01Wb.silb` (0 bytes in the source zip, per manifest; emodb is out of scope)
- No zero-byte file in any data folder used by this project. CREMA-D data folders are complete.
- Not deleted (rule: nothing under datasets/ is removed by this project). Safe to clear by hand if git is used in crema_d.
