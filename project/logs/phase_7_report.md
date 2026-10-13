# Phase 7 report

Status: **done**. `state.json` `phase_done` set to 7 (was 6).

## Deliverables

| # | Deliverable | Path |
|---|---|---|
| 1 | Model, state_dict (CNN, E2 best checkpoint) | `project/models/face_emotion_best.pt` |
| 1 | Class order | `project/models/classes.json` |
| 1 | Model card | `project/models/model_card.md` |
| 1 | Packaging script | `project/code/phase7_package.py` |
| 2 | Inference script (image or video, JSON output) | `project/code/infer.py` |
| 2 | Inference test script (3 images, 1 video from test split) | `project/code/phase7_test_infer.py` |
| 2 | Inference test output | `project/logs/phase7_infer_test.json` |
| 3 | Course report | `project/reports/FACE_MODEL_REPORT.md` |
| 4 | This checklist and the state update | `project/logs/phase_7_report.md`, `project/state.json` |

## Checks run

- `phase7_package.py`: the copied file equals `models/best_model.pt` byte for byte. It loads with `strict=True` into `SmallCNN(7)`. `classes.json` equals `results/classes.json`.
- `phase7_test_infer.py`: all 4 runs returned JSON with 7 probabilities that sum to about 1 (rounded to 4 decimals). Results:
  - FER-2013 image: true happy, predicted happy (0.869)
  - JAFFE crop: true neutral, predicted neutral (0.560)
  - CREMA-D crop: true fear, predicted fear (0.886)
  - RAVDESS clip, 5 frames, all with faces: true disgust, predicted happy (0.474). Wrong.

## Decisions

See `project/DECISIONS.md`, section Phase 7. The main ones:

- The packaged model is the CNN, by the Phase 6 mean E2 macro-F1 rule.
- That winner was picked on the test split. On validation, ResNet-18 is higher. This is stated in the report and the model card, and the rule was not changed.
- `waiting_for: "colab_results"` in state.json is stale and was left as it was.
