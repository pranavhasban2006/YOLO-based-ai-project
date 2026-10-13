# Phase 4 report

## Packaging (project/colab_upload/)

| file | size |
|---|---|
| faces.zip | 0.822 GB, 85,318 files (51,818 crops + 33,500 fer2013 originals used by the splits) |
| splits.csv, faces_index.csv | 85,318 rows each, zip-relative paths |
| experiments.json, class_weights.json, classes.json | small |
| train_colab.ipynb | 11 cells, nbformat 4, validated with nbformat |
| **total** | **0.836 GB** |

- Verification: `faces.zip` was extracted with `zipfile.extractall` into a temp folder outside the project, and every path in splits.csv and faces_index.csv was checked. Missing after extraction: **0**. Temp folder deleted.
- Missing source files on this PC: **0**.
- Dropped rows: **2,387** fer2013 rows that have no FER+ label (not in any split, so not in the zip).

## Notebook test (local, CPU, tiny configuration)

- Every code cell was run in order with `YOLO_TEST=1`: 12 rows per dataset and split, CNN 1 epoch, ResNet-18 2 epochs (head 1 + full 1), no Drive.
- E2 for both models and the E3 path (ResNet-18 held out on each of the 4 datasets, 1 epoch) were run. Result: **PASS**, ALL DONE printed. Outputs (metrics.json, history.csv, confusion PNGs, compare_E2.csv, E3_summary.csv/png, classes.json, best_model.pt) were all written.
- The test folder `project/colab_upload/test_run/` was deleted afterwards.
- Not exercised: resume from `last.pt`, the OOM batch-halving fallback, the time-budget skip, and Colab/Drive mounting. These run only on Colab.
- No full model was trained in this phase. The test trained on 48 images for 1 to 2 epochs to check the code.

## Next

Waiting for the Colab run. Follow project/COLAB_INSTRUCTIONS.md, then copy the results to project/colab_results/. Phase 5 starts when that folder exists.
