# Phase 5 report

Status: **done**. phase_done set to 5 (was 4).

Precondition check: phase_done was 4. `project/colab_results/` exists, with `best.pt` and `metrics.json` in every run folder.

## 1. Checkpoints (loaded on CPU, `weights_only=True`, strict load, one forward pass)

| Run folder | Arch (from folder) | best.pt | last.pt | Output |
|---|---|---|---|---|
| E2_cnn | cnn | ok | ok | (1, 7) |
| E2_resnet18 | resnet18 | ok | ok | (1, 7) |
| E3_cnn_holdout_crema_d | cnn | ok | ok | (1, 7) |
| E3_cnn_holdout_fer2013 | cnn | ok | ok | (1, 7) |
| E3_cnn_holdout_jaffe | cnn | ok | ok | (1, 7) |
| E3_cnn_holdout_ravdess | cnn | ok | ok | (1, 7) |
| colab_results/best_model.pt | cnn (choice.json winner) | ok | n/a | byte-identical to E2_cnn/best.pt |

Every architecture matches its folder name. The E2 metrics.json files record the same arch.

## 2. classes.json

`["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]`. Matches the expected list.

## 3. Parsing

All of these parsed: every run's `metrics.json`, `done.json`, `history.csv`, `compare_E2.csv`, `E3_summary.csv` and `choice.json`. Nothing was unparseable.

Results as recorded in the files:
- E2 macro-F1 (mean over the four in-dataset tests): cnn 0.614, resnet18 0.585. Winner: cnn (choice.json).
- E3 held-out macro-F1: fer2013 0.088, jaffe 0.454, ravdess 0.412, crema_d 0.416.

## 4. Copied

- `results/<run>/`: metrics.json, history.csv, done.json, confusion_*.png (for the 6 runs)
- `models/<run>/`: best.pt, last.pt (for the 6 runs)
- `results/`: compare_E2.csv, E3_summary.csv, E3_summary.png, choice.json, classes.json
- `models/best_model.pt`: the top-level best_model.pt (the E2 cnn winner)

48 files copied in total. No existing file differed, so nothing was renamed or skipped. The full list with actions is in `logs/phase_5_import.json`.

## Notes

- `state.json` still has `"waiting_for": "colab_results"`. Nothing reads it, so it was left unchanged.
- `project/collab_results/` (note the spelling) holds a zip and a copy of the notebook. It was not imported, since the spec names only `colab_results/`.
- The E3 fer2013 held-out macro-F1 is 0.088 (accuracy 0.072). That is a very low score and is worth a look in Phase 6, though it is not an import error.
- Decisions taken: see `DECISIONS.md`, section Phase 5.
