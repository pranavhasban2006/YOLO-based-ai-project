# Phase 6 report

Status: **done**. `state.json` `phase_done` set to 6 (was 5). No training in this phase.

Precondition: `phase_done` was 5, so the phase ran.

## Outputs

| File | What it holds |
|---|---|
| `results/comparison_all.csv` | CNN vs ResNet-18 for E1, E2, E3: accuracy and macro-F1 per dataset, plus the mean |
| `results/comparison_all.png` | Bar chart of the same, one panel per experiment |
| `results/E3_summary.csv` | Each held-out dataset next to its in-dataset (E2, CNN) score, with the drop |
| `results/E3_summary.png` | Bar chart of the same |
| `results/E3_error_pairs.csv` | Three most confused pairs per held-out dataset, each with a likely reason |
| `results/E3_confusion_transcribed.csv` | The four E3 confusion matrices as numbers (read from the PNGs, checked) |
| `logs/phase_6_compare.json` | Everything the script produced, including skip and check results |
| `code/phase6_confusion.py`, `code/phase6_compare.py` | The code. Run with `python -I` in the project venv |

## 1. CNN vs ResNet-18, E1 / E2 / E3

Macro-F1 (accuracy in the CSV). E1 is a proxy: the Phase 3 decision uses the E2 per-dataset test splits in place of E1, which was not run on Colab.

| Dataset | E1 (proxy = E2) CNN | E1 (proxy) ResNet-18 | E2 CNN | E2 ResNet-18 | E3 CNN (held out) | E3 ResNet-18 |
|---|---|---|---|---|---|---|
| fer2013 | 0.710 | 0.746 | 0.710 | 0.746 | 0.088 | not run |
| jaffe | 0.632 | 0.404 | 0.632 | 0.404 | 0.454 | not run |
| ravdess | 0.573 | 0.649 | 0.573 | 0.649 | 0.412 | not run |
| crema_d | 0.543 | 0.542 | 0.543 | 0.542 | 0.416 | not run |
| **mean** | **0.614** | **0.585** | **0.614** | **0.585** | **0.342** | not run |

Accuracy means: E2 CNN 0.645, E2 ResNet-18 0.615, E3 CNN 0.361.

The E2 winner is the CNN on the mean macro-F1 (`choice.json`). The ResNet-18 wins on fer2013 and ravdess. The CNN wins on jaffe, which has only 44 test images. The CNN's lead on the mean comes mostly from jaffe; crema_d is effectively tied (0.543 vs 0.542).

## 2. E3 summary (main result)

| Held out | In-dataset (E2, CNN) macro-F1 | E3 held-out macro-F1 | Drop | E3 macro-F1, 6 shared classes | E3 n |
|---|---|---|---|---|---|
| fer2013 | 0.710 | 0.088 | −0.622 | 0.086 | 33,500 |
| jaffe | 0.632 | 0.454 | −0.178 | 0.417 | 213 |
| ravdess | 0.573 | 0.412 | −0.161 | 0.472 | 14,400 |
| crema_d | 0.543 | 0.416 | −0.128 | 0.416 | 37,205 |

**Skipped E3 runs: none.** All four held-out runs have `done.json`, and the notebook output has no skip line. The E3 time guard did not fire.

The fer2013 drop is the largest, and the held-out result is below chance. Accuracy is 0.072 against a chance level of 0.143, and 60% of its predictions are disgust. This was not investigated further. It is the most important open point in the results.

## 3. Error analysis, held-out datasets

Top three confused pairs per held-out dataset, ranked by count of true→predicted cells. The row rate is the share of that true class. Reasons are hypotheses, not tested.

**fer2013** (held out; test = all 33,500 rows)

| Pair | Count | Row rate | Likely reason |
|---|---|---|---|
| neutral → disgust | 7,113 | 60% | acted vs spontaneous |
| happy → disgust | 5,264 | 57% | acted vs spontaneous |
| sad → disgust | 2,927 | 69% | acted vs spontaneous |

fer2013 is spontaneous, 48 px and web-sourced. Every training set (JAFFE, RAVDESS, CREMA-D) is acted. All three top pairs run into disgust, so the disgust prediction is the common failure.

**jaffe** (held out; 213 images)

| Pair | Count | Row rate | Likely reason |
|---|---|---|---|
| angry → neutral | 22 | 73% | acted vs spontaneous |
| disgust → sad | 17 | 59% | label noise |
| fear → sad | 14 | 44% | label noise |

Each JAFFE class has about 30 images, so these counts are small and noisy. None of the four reasons fits cleanly. The labels given are the closest fit.

**ravdess** (held out; 14,400 rows)

| Pair | Count | Row rate | Likely reason |
|---|---|---|---|
| angry → neutral | 733 | 38% | acted vs spontaneous |
| sad → disgust | 635 | 33% | label noise |
| neutral → happy | 632 | 22% | label noise |

The neutral → happy pair has a documented cause. RAVDESS calm is merged into neutral (Phase 1 decision), so the neutral class is not one emotion.

**crema_d** (held out; 37,205 rows, scored on 6 classes)

| Pair | Count | Row rate | Likely reason |
|---|---|---|---|
| sad → neutral | 2,258 | 36% | label noise |
| angry → neutral | 1,856 | 29% | acted vs spontaneous |
| fear → neutral | 1,589 | 25% | acted vs spontaneous |

Missing classes: CREMA-D has no surprise class, and its test rows are never surprise. The model still predicts surprise on 2,680 of its rows (fear → surprise alone is 782). That is the surprise output going to a class the dataset does not have. It is not one of the top three pairs, so it is noted here and not ranked.

**Grayscale vs colour** is not used as a reason anywhere. The CNN converts every input to 1-channel grayscale (`build_notebook.py:135`), so it cannot see the colour difference.

## 4. Checks run

- The four confusion matrices were checked against the saved `metrics.json`. Row sums equal the per-class support, and diagonal / row sum equals the saved recall. All pass, to within 0.0005 on recall.
- Both charts were opened and checked by eye. One label collision (a note over the JAFFE value) was moved.
- The palette slots used (1 and 2 of `palette.md`) pass `validate_palette.js` in light and dark mode.

## Decisions and open items

See `DECISIONS.md`, section Phase 6, for the full list. The main ones:

- E1 is a proxy: the E2 per-dataset test splits. E2 trains on all four datasets, so it is not a true in-dataset score.
- ResNet-18 has no E3 result, because Colab trained only the winner.
- The fer2013 held-out result is below chance and predicts disgust for 60% of rows. The cause is not established. It may be domain shift, but that is not tested. It needs a follow-up. Phase 6 does no training.
- `state.json` still has `"waiting_for": "colab_results"`. Phase 5 left it, and nothing reads it. It was left unchanged here.
