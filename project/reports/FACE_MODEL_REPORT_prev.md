# Facial emotion model: report for the course

Paths in this report are relative to the repository root `D:\Projects\YOLO`.

## 1. Datasets

| Dataset | Size on disk and content | Licence | How we used it | Classes used |
|---|---|---|---|---|
| FER-2013 (Kaggle, msambare/fer2013) | 35,887 images, 48x48 greyscale, folder `datasets/FER-2013/extracted/` (0.12 GB) | Kaggle dataset licence, research use (`datasets/FER-2013/README.md`) | Labels replaced by the FER+ majority vote. 33,500 rows kept (2,387 dropped: no FER+ label). Official `train/` split gives train and val (10% carved off). Official `test/` gives test (6,722 rows) | all 7 |
| FER+ (Microsoft, `fer2013new.csv`) | 14 files, 10-annotator votes for the FER-2013 rows | MIT (`datasets/ferplus/LICENSE.md`) | Relabelling only. A column `label_ferplus` on the FER-2013 rows, not a separate dataset | 7 (contempt, unknown and NF rows dropped) |
| JAFFE | 213 stills, 10 Japanese female subjects; 0.026 GB | Non-commercial scientific research only, no redistribution (`datasets/jaffe/README.md`) | Each still detected, cropped (20% margin), stored at 224x224. Subject split 6/2/2 | all 7, about 30 per class |
| RAVDESS, speech video (modality 01 and 02) | 2,880 mp4 clips from 24 actors. Whole RAVDESS folder is 29.3 GB, of which the speech video is a part. Song and audio-only files not used | CC BY-NC-SA 4.0 (`datasets/RAVDESS/README.md`) | 5 frames per clip at 20% to 80% of the length. MediaPipe face crop. Subject split 16/4/4. 14,400 rows | 7 (calm merged into neutral) |
| CREMA-D, video (`VideoFlash`) | 7,442 clips from 91 actors; 3.7 GB for the folder | Open Database License (ODbL), contents under DbCL (`datasets/crema_d/LICENSE.txt`) | 5 frames per clip, same crop rule. 7,441 clips used (one clip, `1076_MTI_SAD_XX`, had no detectable face in any frame). Subject split 63/14/14. 37,205 rows | 6 (no surprise) |

Totals: the index holds 46,422 files (35,887 FER-2013 images, 213 JAFFE images, 2,880 RAVDESS clips, 7,442 CREMA-D clips). After dropping 2,387 unlabelled FER rows and sampling 5 frames per video clip, `project/data/splits.csv` has 85,318 rows (33,500 FER-2013, 213 JAFFE, 14,400 RAVDESS, 37,205 CREMA-D).

## 2. Label unification and split method

**Label mapping** (all four datasets map to the same 7 classes; the order is angry, disgust,
fear, happy, neutral, sad, surprise):

- FER-2013: FER+ majority vote over the 7 target classes. Ties, "unknown", "NF" and "contempt" rows are dropped. The folder label is not used for training.
- JAFFE: file name code AN, DI, FE, HA, NE, SA, SU map to angry, disgust, fear, happy, neutral, sad, surprise.
- RAVDESS: 01 neutral, 02 calm (merged into neutral), 03 happy, 04 sad, 05 angry, 06 fearful (fear), 07 disgust, 08 surprised (surprise).
- CREMA-D: ANG, DIS, FEA, HAP, NEU, SAD. No surprise class exists in this dataset.

**Split (leakage control):**

- Video and JAFFE are split by subject, never by frame or clip. All frames of one person go to one split. Targets were 70/15/15 by subjects, stratified as far as the class balance allowed, seed 42.
- FER-2013 has no subject ID, so its official train/test folders are used, and 10% of train is carved off as val.
- The script asserts: no subject appears in two splits within a dataset, and no file path appears twice. Both checks pass (`project/logs/phase_3_report.md`).
- Known remaining risk: FER-2013 has no subject IDs, so we cannot prove that identities do not cross its train and test folders. The RAVDESS modality 01 and 02 files come from the same performance and are in the same split (same actor), which is allowed by the subject rule but makes those frames near-duplicates.

**Experiments** (`project/data/experiments.json`):

- E1, within-dataset: train and test on the same dataset. **Not run on Colab.** The Colab run did only E2 and E3, to stay in the time budget (see section 5). The E1 rows below use the per-dataset E2 test scores as a proxy, labelled as such.
- E2, combined: train on the train splits of all four datasets together. Report the test split of each dataset separately.
- E3, leave-one-dataset-out: train on three datasets, test on the fourth. Run for the CNN only (the E2 winner). Main result for the claim that the model transfers to a new dataset.

## 3. Models and training settings

Both models run on Google Colab with a **Tesla T4 GPU** and mixed precision (`project/colab_upload/train_colab.ipynb`; the saved notebook with the run output, which prints `GPU: Tesla T4` and `ALL DONE`, is `project/collab_results/Copy_of_train_colab.ipynb`). Seed 42.

| Setting | CNN (`SmallCNN`) | ResNet-18 |
|---|---|---|
| Input | 48x48 greyscale, normalised to [-1, 1] | 224x224 RGB, ImageNet mean and std |
| Architecture | 4 blocks of (Conv3x3-BN-ReLU x2, MaxPool), 64/128/256/512 channels, GAP, dropout 0.4, linear to 7. 4.69M parameters | torchvision ResNet-18, ImageNet weights, new 7-way head |
| Loss | cross-entropy, class weights, label smoothing 0.1 | same |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4 | head only for 3 epochs at lr 1e-3, then all layers at lr 1e-4 with cosine decay |
| Schedule | 2 warmup epochs, cosine | constant head phase, then cosine |
| Maximum epochs | 40 (E2 stopped at 25, best 19) | 25 (E2 stopped at 14, best 8) |
| Batch | 64 | 64 |
| Augmentation | flip, rotation 15, brightness/contrast jitter 0.3, random erasing p=0.25 | same |
| Early stopping | validation macro-F1, patience 6 | same |
| Gradient clip | 1.0 | 1.0 |
| Per-epoch sampler | each dataset at most 50% of an epoch | same |

E3 runs used the CNN, up to 20 epochs each. Epochs actually run: FER-2013 held out 16 (best 10), JAFFE held out 15 (best 9), RAVDESS held out 15 (best 9), CREMA-D held out 20 (best 14, the cap). No E3 run was skipped by the time guard (MAX_HOURS = 6).

## 4. Results

Macro-F1 is the main metric. Accuracy is in brackets. For CREMA-D, the score is over the 6 classes it has.

### 4.1 E1 (proxy), E2: CNN and ResNet-18

E1 was not trained. The E1 columns repeat the E2 per-dataset numbers, which is the proxy agreed in Phase 3.

| Dataset | Test n | E1 proxy CNN | E1 proxy ResNet-18 | E2 CNN | E2 ResNet-18 |
|---|---|---|---|---|---|
| FER-2013 | 6,722 | 0.710 (0.810) | 0.746 (0.827) | 0.710 (0.810) | 0.746 (0.827) |
| JAFFE | 44 | 0.632 (0.659) | 0.404 (0.432) | 0.632 (0.659) | 0.404 (0.432) |
| RAVDESS | 2,400 | 0.573 (0.570) | 0.649 (0.662) | 0.573 (0.570) | 0.649 (0.662) |
| CREMA-D | 5,710 | 0.543 (0.542) | 0.542 (0.538) | 0.543 (0.542) | 0.542 (0.538) |
| **Mean** | | **0.614 (0.645)** | **0.585 (0.615)** | **0.614 (0.645)** | **0.585 (0.615)** |

Winner by the Phase 6 rule (mean E2 macro-F1): **CNN**, which is the model packaged in `project/models/`. See the limitation on this choice in section 6.

### 4.2 E3 (leave-one-dataset-out)

| Held out | In-dataset E2 CNN macro-F1 | E3 CNN macro-F1 (accuracy) | Drop in macro-F1 | E3 macro-F1 on the 6 shared classes | E3 n | E3 ResNet-18 |
|---|---|---|---|---|---|---|
| FER-2013 | 0.710 | 0.088 (0.072) | -0.622 | 0.086 | 33,500 | not run |
| JAFFE | 0.632 | 0.454 (0.484) | -0.178 | 0.417 | 213 | not run |
| RAVDESS | 0.573 | 0.412 (0.468) | -0.161 | 0.472 | 14,400 | not run |
| CREMA-D | 0.543 | 0.416 (0.420) | -0.128 | 0.416 | 37,205 | not run |
| **Mean** | | **0.342 (0.361)** | | | | |

The FER-2013 result is **below chance** (1/7 = 0.143 accuracy; the accuracy is 0.072). The model predicts "disgust" for about 60% of FER-2013 rows. The cause is not established. It is most likely the acted-to-spontaneous domain gap, plus the 48 px scale, but this was not tested.

### 4.3 Confusion matrices and key figures

Confusion matrices (E2, per dataset test split):
- `project/results/E2_cnn/confusion_fer2013.png`, `confusion_jaffe.png`, `confusion_ravdess.png`, `confusion_crema_d.png`
- `project/results/E2_resnet18/confusion_fer2013.png`, `confusion_jaffe.png`, `confusion_ravdess.png`, `confusion_crema_d.png`

Confusion matrices (E3, CNN, held-out dataset):
- `project/results/E3_cnn_holdout_fer2013/confusion_fer2013.png`
- `project/results/E3_cnn_holdout_jaffe/confusion_jaffe.png`
- `project/results/E3_cnn_holdout_ravdess/confusion_ravdess.png`
- `project/results/E3_cnn_holdout_crema_d/confusion_crema_d.png`

The counts behind the E3 charts are in `project/results/E3_confusion_transcribed.csv`. They were read from the PNGs and checked against the saved per-class recall.

Key figures:
- `project/results/comparison_all.png`: CNN vs ResNet-18, E1 proxy, E2 and E3 per dataset.
- `project/results/E3_summary.png`: each held-out dataset next to its in-dataset score.
- `project/results/E3_summary.csv`, `project/results/E3_error_pairs.csv`: the top confused pairs per held-out dataset.
- `project/reports/class_counts_*.png`, `project/reports/contact_*.png`: data-side checks.

Main confusions (E3):
- FER-2013: neutral to disgust (7,113 rows), happy to disgust (5,264), sad to disgust (2,927).
- RAVDESS: angry to neutral (733), sad to disgust (635), neutral to happy (632). The neutral to happy pair is partly explained by merging calm into neutral.
- CREMA-D: sad to neutral (2,258), angry to neutral (1,856), fear to neutral (1,589). CREMA-D has no surprise samples, yet the model predicts surprise on 2,680 rows.

### 4.4 Inference check

`project/code/infer.py` was run on 3 images and 1 video from the held-out test split (seed 42). Output is in `project/logs/phase7_infer_test.json`.

| Sample | True label | Predicted | Top probability |
|---|---|---|---|
| FER-2013 image (48x48) | happy | happy | 0.869 |
| JAFFE face crop | neutral | neutral | 0.560 |
| CREMA-D face crop | fear | fear | 0.886 |
| RAVDESS clip (5 frames) | disgust | happy | 0.474 (disgust 0.234) |

Three of four correct on one small sample. This is not an accuracy estimate. The RAVDESS clip is wrong.

## 5. Process and what was not done

- Data preparation, splits, and the Colab notebook were built on this PC (Phases 0 to 4). Training ran only on Colab (T4). Results were copied back and checked (Phase 5). Analysis is in `project/logs/phase_6_report.md`.
- The Colab notebook was tested on CPU with a tiny configuration before the full run.
- Not done: E1 as a real within-dataset experiment; ResNet-18 E3; a second seed; a calibration check; a face-detector accuracy check on FER-2013 (its images were not run through the detector).

## 6. Honest limitations

1. **Model choice used the test split.** The Colab notebook picked the winner (CNN) from the E2 test macro-F1 (`choice.json`). The validation macro-F1 would have picked ResNet-18 (0.657 vs 0.611). The CNN's lead on the mean comes mostly from JAFFE (44 test images). Without JAFFE the mean is 0.609 for the CNN and 0.646 for ResNet-18. The packaged CNN follows the agreed rule, but the comparison is weaker than the table suggests.
2. **Acted data.** JAFFE, RAVDESS and CREMA-D are posed or acted. Real-world faces (FER-2013, the only spontaneous-face source) score very low under E3.
3. **Class imbalance.** Disgust is rare in FER-2013 (about 1.5% of rows, 436 in train) and in the combined train set (5,856, most from RAVDESS and CREMA-D). Neutral and happy dominate FER-2013. Class weights reduce but do not remove this.
4. **Missing class.** CREMA-D has no surprise samples. Its results are on 6 classes, and any surprise prediction for CREMA-D is a mismatch with the dataset.
5. **Label noise.** FER+ votes are noisy for FER-2013; 2,387 rows were dropped. RAVDESS calm is merged into neutral. Some JAFFE classes are noisy at this size.
6. **Small test sets.** JAFFE has 44 test images, and its score is very noisy.
7. **Frames not independent.** Five frames per clip are highly correlated, and RAVDESS modality 01 and 02 share a performance. Effective sample size is lower than the row count.
8. **Resolution.** The CNN uses 48x48 greyscale; colour and fine detail are lost. ResNet-18 uses 224x224 RGB but was not the packaged model.
9. **Single run.** One seed per configuration. Differences of a few points (for example, E2 CNN vs ResNet-18 on CREMA-D: 0.543 vs 0.542) are not meaningful.
10. **Face detection on the web input.** Videos and images go through MediaPipe and Haar. When no face is found, the whole frame is used, which gives weaker predictions. The RAVDESS demo clip is an example of a wrong prediction.

## 7. Abstract summary (5 lines)

1. We built a 7-class facial emotion model from four public datasets (FER-2013 with FER+ labels, JAFFE, RAVDESS, CREMA-D), with subject-level splits to prevent identity leakage.
2. A small CNN on 48x48 greyscale faces reached a mean macro-F1 of 0.61 across the four in-dataset test splits (ResNet-18: 0.59); the CNN is the packaged model.
3. When each dataset is held out in turn, performance falls sharply, from 0.09 on FER-2013 (spontaneous faces, below chance) to 0.41 to 0.45 on the acted datasets.
4. Training ran on Google Colab (Tesla T4). A command-line tool takes an image or video, samples five frames, detects the face, and returns averaged probabilities for the seven classes.
5. The results show acted-data bias, class imbalance, a missing surprise class in CREMA-D, and a choice of model made on the test split; the report lists these limits in full.

## 8. Files

- Model: `project/models/face_emotion_best.pt`, `project/models/classes.json`, `project/models/model_card.md`
- Inference: `project/code/infer.py`, test script `project/code/phase7_test_infer.py`, packaging script `project/code/phase7_package.py`
- Results: `project/results/` (CSVs, PNGs, per-run metrics), `project/colab_results/` (raw Colab output)
- Splits and experiments: `project/data/splits.csv`, `project/data/experiments.json`, `project/data/class_weights.json`
- Decisions: `project/DECISIONS.md`
