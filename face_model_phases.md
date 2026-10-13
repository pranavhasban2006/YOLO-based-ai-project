## Phase 0: Setup and audit

```
You are working unattended. Never ask me questions. When a choice is needed, pick the
default stated here, record it in project/DECISIONS.md with one line of reasoning, and
continue. Do not download or delete anything under ./datasets/.

GOAL: prepare a project folder and audit the datasets for a multi-dataset facial
emotion model.

1. Create ./project/ with subfolders: code, data, logs, models, results, reports.
   Create project/state.json = {"phase_done": -1} and project/DECISIONS.md.
2. Create a Python venv in ./project/.venv (or reuse the active env). Install and
   pin in project/requirements.txt: torch, torchvision, numpy, pandas,
   scikit-learn, opencv-python, pillow, tqdm, matplotlib, seaborn, mediapipe
   (if mediapipe fails to install, skip it and use OpenCV Haar cascades; record that).
3. Print GPU status. Record torch version, CUDA availability and free disk space in
   project/logs/phase_0_report.md.
4. Audit ./datasets/. For each of fer2013, ferplus, jaffe, ravdess, crema_d: report
   whether it exists, file count, total size, any .part/.tmp/zero-byte files, and
   whether zips are extracted. Test zips with unzip -t. Mark each COMPLETE,
   PARTIAL or MISSING. Read ./datasets/manifest.json if present.
5. Continue with every dataset that is COMPLETE. Mark PARTIAL or MISSING datasets as
   "skipped" in state.json (key "datasets_available") and keep going without them.
   Never stop the whole project because one dataset is absent. If fer2013 itself is
   missing, stop and write the reason into the report.
6. Write project/code/common.py with: CLASSES list
   [angry, disgust, fear, happy, neutral, sad, surprise], seed_everything(), get_device(),
   and path helpers. Use seed 42, pathlib, and DataLoader num_workers=0 on Windows.
7. Set state.json phase_done = 0. Print a short summary.
```

## Phase 1: Inspect the data and build a master index

```
Unattended. Never ask me questions; pick defaults, log them in project/DECISIONS.md.
Precondition: project/state.json has phase_done >= 0, otherwise stop and say why.

GOAL: understand each dataset's real structure and build one index table.

1. For every available dataset, discover the real layout by listing folders and file
   name patterns. Do not assume. Save a short description per dataset to
   project/logs/layout_<dataset>.txt.
2. Parse labels with these rules, but verify against the files and fix if different:
   - fer2013 (image folders train/test, one folder per emotion): label = folder
     name. Keep the official train/test split. If a PublicTest/PrivateTest split
     exists, use it.
   - ferplus: fer2013new.csv holds 10-annotator votes for FER-2013 rows. Label =
     majority vote among the 7 target classes. Drop rows where the majority is
     "unknown" or "NF" (not a face), or where the top two votes tie. Map
     "contempt" and any extra classes to dropped. FER+ uses the same images as
     FER-2013, so build it as a relabelled version of fer2013 (a column
     label_ferplus), not as a separate dataset.
   - jaffe: file names like KA.AN1.39: expression code AN, DI, FE, HA, NE, SA, SU.
     Subject = first two letters (KA, KL, ...).
   - ravdess: file name modality-channel-emotion-intensity-statement-repetition-actor.
     Emotion codes 01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful,
     07 disgust, 08 surprised. Merge calm into neutral. Subject = actor number.
     Use only video-containing files (modality 01 audio-video or 02 video-only).
   - crema_d: file name ActorID_Sentence_Emotion_Level, emotions ANG, DIS, FEA,
     HAP, NEU, SAD. No surprise class. Subject = ActorID. Use the video files.
3. Build project/data/index.parquet (or csv if parquet unavailable) with columns:
   dataset, path, label, label_id, subject, split_hint, kind ("image" or "video").
4. Write project/reports/dataset_stats.csv and PNG bar charts of class counts per
   dataset. Note imbalance and which classes a dataset lacks.
5. Save 8 random sample images per class per image dataset to
   project/reports/samples/ so label sanity can be checked later. Flag, but do not
   remove, any file that cannot be opened.
6. Update state.json phase_done = 1 and write project/logs/phase_1_report.md.
```

## Phase 2: Extract frames and crop faces

```
Unattended. Never ask me questions; pick defaults, log them in project/DECISIONS.md.
Precondition: phase_done >= 1.

GOAL: turn every video into cropped face images and standardize all images.

1. Use project/data/index.parquet. For video rows (ravdess, crema_d):
   - Sample 5 frames per clip at evenly spaced positions between 20% and 80% of
     the clip length (avoids neutral start and end).
   - Detect the largest face with MediaPipe face detection (fallback: OpenCV Haar).
     Add a 20% margin, crop, resize to 224x224 RGB, save as JPEG quality 95 in
     project/data/faces/<dataset>/<subject>/<file>_f<k>.jpg.
   - Skip frames where no face is found; log how many were skipped per dataset.
2. For image datasets (fer2013, jaffe): do not run detection on fer2013 (already
   cropped 48x48 grayscale). For jaffe run the detector, crop, resize to 224.
   For fer2013 convert to 3 channel and resize to 224 on the fly later, so store
   the original path only.
3. Write project/data/faces_index.parquet with the same columns as the index plus
   face_path, frame_id. Make it resumable: skip any output file that already exists
   and is a valid image.
4. Show progress with tqdm and use multiprocessing where it is safe
   (workers = min(4, cpu_count()); on Windows use the spawn-safe pattern).
5. Report counts per dataset and class after extraction, and save a contact sheet
   (project/reports/contact_<dataset>.png) of 40 random crops per dataset.
6. Update state.json phase_done = 2 and write project/logs/phase_2_report.md.
```

## Phase 3: Splits that avoid person leakage

```
Unattended. Never ask me questions; pick defaults, log them in project/DECISIONS.md.
Precondition: phase_done >= 2.

GOAL: create train/val/test splits and the cross-dataset experiment plan.

1. Within each dataset, split BY SUBJECT (all frames of one person go to one split):
   ravdess, crema_d, jaffe -> 70% train, 15% val, 15% test subjects, stratified as
   far as the class balance allows, seed 42.
   fer2013 has no subject ids: use its official split, and carve 10% of train as val.
2. Save project/data/splits.csv with columns face_path/path, dataset, split.
3. Assert programmatically that no subject appears in two splits within a dataset
   and that no path appears twice. Fail loudly if violated.
4. Define these experiments in project/data/experiments.json:
   E1 within-dataset: train and test on the same dataset (one per dataset).
   E2 combined: train on train splits of all available datasets together, report the
      test split of EACH dataset separately.
   E3 leave-one-dataset-out: train on all datasets except D, test on all of D
      (for D in each available dataset). This is the main "real dataset" result.
   For crema_d, which has no surprise class, report metrics on the shared classes
   too, and say so.
5. Compute class weights for the combined training set and save them.
6. Update state.json phase_done = 3 and write project/logs/phase_3_report.md.
```

## Phase 4: Colab handoff (prepare upload, no training on this PC)

```
Unattended. Never ask me questions; pick defaults, log them in project/DECISIONS.md.
Precondition: phase_done >= 3.
Do NOT train any model in this phase. Training happens on Google Colab.
Do NOT delete or modify anything under ./datasets/ or the original files under ./project/data/.

GOAL: package the face crops, splits and experiment definitions for Colab, and write a notebook that trains everything there.

1. Create project/colab_upload/ and build faces.zip containing:
   - every cropped face image under project/data/faces/ (keep the folder structure),
   - the fer2013 images actually referenced in project/data/faces_index.parquet, under fer2013/<split>/<label>/<file>. Copy them, do not move them.
   Next to faces.zip write: splits.csv, faces_index.csv, experiments.json, class_weights.json and classes.json.
   Rewrite every path in these files to be relative to the zip root (for example faces/ravdess/01/...jpg or fer2013/train/angry/...jpg) so the same paths work on Colab.
   Verify: unzip faces.zip into a temp folder outside the project, check that every path in splits.csv and faces_index.csv exists, then delete the temp folder. If some paths are missing, list them in the report and drop those rows from the CSVs. Do not stop for this.
   Print the total size of project/colab_upload/ in GB.
2. Write project/colab_upload/train_colab.ipynb: a valid nbformat 4 notebook with markdown and code cells that runs top to bottom with Runtime > Run all. It must:
   - print the GPU name, or stop with a clear message if no GPU is attached (and say to change the runtime type);
   - mount Google Drive at /content/drive;
   - look for data at /content/drive/MyDrive/YOLO/colab_upload/, and unzip faces.zip into /content/data only if /content/data/faces does not already exist;
   - implement training: weighted cross-entropy with label smoothing 0.1, AdamW, cosine schedule with 2 warmup epochs, mixed precision on CUDA, augmentation (horizontal flip, rotation 15, colour jitter, random erasing; none at validation), a sampler that caps each dataset at 50% of an epoch, early stopping on validation macro-F1 (patience 6), gradient clipping 1.0, seed 42;
   - model "cnn": 4 blocks of (Conv3x3-BN-ReLU x2, MaxPool) with 64, 128, 256, 512 channels, global average pool, dropout 0.4, linear to 7, 48x48 grayscale input, 40 epochs;
   - model "resnet18": torchvision ResNet-18 with ImageNet weights, 224x224 RGB, ImageNet normalisation, new 7-way head; 3 epochs head only at lr 1e-3, then all layers at lr 1e-4 with cosine decay; 25 epochs in total; batch 64 (halve it on out-of-memory);
   - run E2 for both models and compare on macro-F1; then run E3 (leave-one-dataset-out) with the better model, 20 epochs per held-out dataset. Use a variable MAX_HOURS (default 6): if the estimated time would exceed it, skip the remaining E3 datasets and list them in the output;
   - save a checkpoint every epoch to /content/drive/MyDrive/YOLO/colab_results/<run>/last.pt and best.pt, and resume from last.pt automatically if it exists;
   - when a run finishes, write done.json in its folder and skip that run on later runs;
   - write to /content/drive/MyDrive/YOLO/colab_results/: history.csv per run, metrics.json per run (accuracy, macro-F1, per-class precision, recall and F1), confusion matrix PNGs, compare_E2.csv, E3_summary.csv and E3_summary.png, classes.json, and the best model as best_model.pt;
   - finish with a cell that prints ALL DONE and lists the output folder.
   Test the notebook's code on this PC before finishing it: run every code cell with a tiny configuration (CPU, 1 epoch, 200 images, E2 only, no Drive). Use a temporary folder under project/colab_upload/test_run/ and delete it afterwards. Then keep the full settings in the notebook. Record the test result in the report.
3. Write project/COLAB_INSTRUCTIONS.md with exactly these five steps for the user:
   1) Upload the project/colab_upload folder to Google Drive at MyDrive/YOLO/colab_upload.
   2) Open train_colab.ipynb in Google Colab (File > Upload notebook, or open it from Drive).
   3) Choose Runtime > Change runtime type > T4 GPU.
   4) Choose Runtime > Run all. If Colab disconnects, reconnect and run all again; it resumes.
   5) When it prints ALL DONE, copy MyDrive/YOLO/colab_results/ back into project/colab_results/ on this PC.
4. Write project/logs/phase_4_report.md: packaging sizes, verification result, notebook test result, and any dropped rows.
5. Set state.json phase_done = 4 and add the key "waiting_for": "colab_results".
```

## Phase 5: Ingest Colab results (after the user copies them back)

```
Unattended. Never ask me questions; pick defaults, log them in project/DECISIONS.md.
Precondition: phase_done >= 4 AND project/colab_results/ exists and contains at least one best.pt and one metrics.json. If not, stop, write the reason in project/logs/phase_5_report.md, do not change phase_done, and do not fail the project.

GOAL: check and import the Colab results.

1. Load every best.pt on CPU. Confirm each loads and that its architecture matches the model name in its folder (cnn or resnet18).
2. Check that each classes.json equals [angry, disgust, fear, happy, neutral, sad, surprise]. If it differs, stop and report it.
3. Parse every metrics.json, compare_E2.csv and E3_summary.csv. Report any file that does not parse.
4. Copy the metrics, history, plots and CSVs into project/results/, and the checkpoints into project/models/<run>/. Do not overwrite an existing file that differs: save the copy under a new name with a suffix and note it in the report.
5. Set state.json phase_done = 5 and write project/logs/phase_5_report.md with what was found and copied.
```

## Phase 6: Results analysis (no training)

```
Unattended. Never ask me questions; pick defaults, log them in project/DECISIONS.md.
Precondition: phase_done >= 5. No training in this phase.

GOAL: build the comparison tables and charts from the Colab metrics.

1. Build a table of CNN vs ResNet-18 for E1, E2 and E3: accuracy and macro-F1 per dataset, with the mean. Save it as project/results/comparison_all.csv and a bar chart as project/results/comparison_all.png.
2. For E3 (the main result), show each held-out dataset next to its in-dataset (E1 or E2) score so the drop is visible. Save project/results/E3_summary.csv and project/results/E3_summary.png. List any E3 run the Colab output marked as skipped.
3. Error analysis for each held-out dataset: the three most confused class pairs from its confusion matrix, each with a likely reason from this list: acted vs spontaneous faces, grayscale vs colour, missing classes (crema_d has no surprise), label noise.
4. Set state.json phase_done = 6 and write project/logs/phase_6_report.md.
```

## Phase 7: Package the model and write the report

```
Unattended. Never ask me questions; pick defaults, log them in project/DECISIONS.md.
Precondition: phase_done >= 6.

GOAL: produce the final deliverables for the course report and for the web app.

1. Copy the best model (the one with the higher mean macro-F1 from Phase 6) as a state_dict to project/models/face_emotion_best.pt, with classes.json and a model_card.md: architecture, input size and normalisation, class order, training datasets, metrics, known limits.
2. Write project/code/infer.py: load the model, take an image path or a video path (sample 5 frames, detect the face, predict, average the probabilities), print the 7 class probabilities as JSON. Test it on 3 images and 1 video and show the output.
3. Write project/reports/FACE_MODEL_REPORT.md for the course report with:
   - datasets used (name, size, licence, how we used it, classes)
   - label unification and split method (leakage control)
   - models and training settings, and that training ran on Google Colab (T4 GPU)
   - result tables: E1, E2, E3 for CNN and ResNet-18
   - confusion matrices and the key figures (paths to PNGs)
   - honest limitations (acted data, class imbalance, missing classes, label noise)
   - a 5-line summary suitable for the abstract
4. Print a final checklist of every deliverable with its path. Set state.json phase_done = 7.
```
