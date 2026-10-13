# Colab steps: speech emotion model

Expected total time on a free T4: about 2 to 3 hours of training (v2 has 23k clips), plus 5 to 10 minutes of setup
(unzip and load audio), plus however long uploading the 1.6 GB zip takes on your connection.

1. (Use colab_upload_audio_v2 if present: 7 datasets, 16-bit.) In Google Drive create `MyDrive/emotion_audio/colab_upload/` and upload
   `project/colab_upload_audio/audio_16k.zip` into it.
2. Upload `project/colab_upload_audio/audio_colab_train.ipynb` to Colab (File > Upload notebook).
   Runtime > Change runtime type > T4 GPU, then Runtime > Run all.
   If the browser disconnects, reconnect and Run all again; it resumes from `last.pt` and skips finished runs.
3. When the last cell prints `ALL DONE`, download `MyDrive/emotion_audio/colab_results/` to
   `project/colab_results_audio/` on this PC (summary.csv, summary.png, E2_*.json, E3_*.json,
   *_best.pt, cm_*.png, best_model.pt, classes.json).
4. Then run audio Phase 4.

## What it trains
- Model: log-mel (64 bands) CNN, 4 blocks (32/64/128/256 channels), 4 s clips at 16 kHz, 7 classes.
- E2: all 4 datasets (crema_d, emodb, ravdess, tess) trained together, tested on each dataset's test split.
- E3: leave-one-dataset-out, 4 runs, tested on the whole held-out dataset.

## Known limits
- Only 4 datasets are in the zip (SAVEE, MELD, ESD are not).
- TESS is split by word, so it is not speaker-independent. EmoDB val/test are tiny (30 / 74 clips): treat its scores as noisy.
- CREMA-D and EmoDB have no surprise class; scores are macro-F1 over the classes present.
