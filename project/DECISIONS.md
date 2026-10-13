# Decisions

Unattended-run choices, one line of reasoning each.

## Phase 0

- **Python 3.13 (Anaconda base) for `project/.venv`.** It is the default `python` on PATH and has wheels for every listed package.
- **PyTorch from the cu128 wheel index.** The GTX 1650 Ti runs driver 581.42 (CUDA 13.0), which supports cu128 builds.
- **fer2013 canonical copy = `datasets/FER-2013/extracted`.** `manifest.json` names it as the complete copy (28,709 train / 7,178 test verified). The old `datasets/fer2013/` folder is partial (train 3,028 of 28,709) and is ignored, not deleted.
- **RAVDESS scope = speech only (Audio_Speech + Video_Speech).** The manifest says song files were skipped per instructions, so they are excluded from COMPLETE/PARTIAL. The `Video_Song_Actor_04.part` the manifest mentions is no longer on disk (the runner log shows it disappearing at about 23:02, outside this run). Nothing was deleted by this run.
- **Audit status rule.** MISSING = no files in the used subset. PARTIAL = any `.part/.tmp/.crdownload/.download`, zero-byte file, git-lfs pointer stub, failed `unzip -t`, or wrong expected count. Otherwise COMPLETE.
- **Datasets audited = the five named in the phase spec** (fer2013, ferplus, jaffe, ravdess, crema_d). `emodb`, `tess`, `goemotions` and `eeg_ds003478` are on disk and COMPLETE in the manifest but are outside this phase's scope, so they are not audited here.
- **Dataset audit is read-only.** `project/code/audit_datasets.py` only reads; it does not download, move or delete under `datasets/`.
- **Mediapipe.** Installed (1.1.0). It has no legacy `mp.solutions` API, so detection uses `mp.tasks` FaceDetector with the repo's `models/blaze_face_short_range.tflite`. The Haar fallback was not needed; OpenCV Haar cascades ship with the package anyway.
- **opencv-contrib-python replaces opencv-python.** mediapipe requires the contrib build, and having both installed breaks `cv2` (uninstalling one deletes the shared `cv2` files). The pin is `opencv-contrib-python==5.0.0.93`.
- **Torch install restarted.** A pip torch download from an earlier session died at about 23:43 (no torch in the venv). It was restarted here with the same cu128 index; the log is `project/logs/install_torch.log`. Orphaned `pip-unpack-*` and `tmp*` partials in `%TEMP%` were left alone.
- **FER+ is label-only.** The `ferplus` audit count (14 files) is the labels folder. Its 35,887 rows map to the FER-2013 images (`fer2013new.csv` has 35,887 data rows, matching FER-2013 train+test).
- **All five audited datasets are COMPLETE**, so `datasets_available` in `state.json` has no `skipped` entries. `phase_0_report.md` records the state.
- **Canonical RAVDESS speech files.** `Video_Speech` holds 2,880 mp4s (120 per actor: modality 01 audio-visual plus 02 video-only). `Audio_Speech` holds 1,440 wavs.
- **DataLoader `num_workers=0`** in `common.py` because the target OS is Windows.
- **Seed 42** in `common.seed_everything()`.
- **Zero-byte files under `.git` do not make a dataset PARTIAL.** A post-audit walk found 130 git-lfs temp files and a stale `crema_d/.git/index.lock`. They are git internals; the audit skips `.git` by design. CREMA-D data folders have no zero-byte or pointer files, so it stays COMPLETE. Listed in `phase_0_report.md`, not deleted.

## Phase 1

- **FER+ join goes through pixels, not file names.** The FER-2013 image names are `Training_<id>.jpg` with Kaggle ids, not row numbers, so FER+ votes cannot be attached by name. The original `fer2013.csv` (with pixels) is inside `datasets image video/challenges-...zip` as `fer2013.tar.gz`. It is extracted to `project/data/_src_fer2013_original/` (copy only, source zip untouched). Each image is matched to its nearest original row by L2 distance within the same Usage.
- **Row alignment is checked, not assumed.** `fer2013new.csv` and the original CSV both have 35,887 rows with identical Usage order. The FER+ confusion against original emotions is strongly diagonal (66% agreement), which would not happen if rows were misaligned.
- **Exact pixel duplicates.** 3,369 original rows share pixels with another row (1,516 groups). An image on such pixels cannot be tied to one row, so it gets `label_ferplus` only when every duplicate row has the same FER+ label. Otherwise `label_ferplus` is empty and the reason is in `ferplus_drop_reason`. The folder label is kept for every image (spec rule). 52 folder labels disagree with the original emotion; all 52 are on duplicate pixels.
- **FER+ majority rule.** Majority is taken over all 10 vote columns. Dropped when the top two counts tie, or when the winner is unknown, NF or contempt. A winner in the 7 target classes is mapped to the class (happiness→happy, anger→angry, sadness→sad). Result: 33,500 kept (tie 1,776; unknown 222; contempt 212; NF 177).
- **Index format = CSV, not parquet.** pyarrow is not installed in `project/.venv`. The phase spec allows csv in that case, and the pinned requirements are not changed for it.
- **`label_ferplus` is an extra column** beyond the spec's seven, because the spec asks for it explicitly. It is empty for every non-fer2013 row.
- **fer2013 split_hint** = `Training`, `PublicTest`, `PrivateTest` (from the file-name prefix). The folder `test/` mixes PublicTest and PrivateTest, so the split comes from the prefix. Other datasets use `none` (their splits are subject-based in Phase 3).
- **Index paths are relative to `datasets/`** (forward slashes), so the index works on any machine that has the same dataset layout.
- **RAVDESS indexed from `RAVDESS/Video_Speech` only.** Song (`Video_Song`) is out (Phase 0 decision). `datasets/ravdess/` is a lowercase duplicate with identical counts, so it is not indexed. Modality 01 and 02 are kept (video contains); modality 03 is audio-only and does not occur here. Calm (02) merges into neutral, so neutral = 192 + 384 = 576.
- **CREMA-D video = `VideoFlash/*.flv`** (7,442 clips). The spec says "video files"; no mp4 exists in this dataset. Surprise has no clips (the spec's "no surprise class" is confirmed).
- **Unreadable files are flagged, not removed.** Every image (PIL) and every video (cv2 open + one frame) was tested. Result: 0 unreadable, `reports/unreadable_files.csv` holds only the header.
- **Samples** for fer2013 and jaffe: 8 random per class (seed 42), saved as PNG under `reports/samples/<dataset>/<label>/`. fer2013 is upscaled 3x (NEAREST) so it can be read. Contact sheets are `reports/samples/contact_<dataset>.png`.
- **Chart colour:** classes with zero files are drawn red, the rest blue.
- **Phase 1 writes only the state fields it needs:** `phase_done` is set to 1, and `datasets_available` is unchanged.

## Phase 2

- **Index read from `data/index.csv`, not `index.parquet`.** Phase 1 wrote CSV. Phase 2 installs pyarrow 26.0.0 (pinned in `requirements.txt`) and writes `faces_index.parquet` as the spec asks.
- **Face paths are relative to the repo root** (forward slashes): `project/data/faces/...` for crops, `datasets/...` for fer2013 originals. The index keeps paths relative to `datasets/`.
- **Frame sampling = 5 frames at fractions 0.2, 0.35, 0.5, 0.65, 0.8** of the frame count, rounded to the nearest frame index. `frame_id` is that source frame index.
- **Frame count from metadata**; if the container reports 0 (FLV), a grab() pass counts frames.
- **Detector = MediaPipe `tasks` FaceDetector** (model `models/blaze_face_short_range.tflite`, confidence 0.5). The largest box wins. If MediaPipe finds nothing, OpenCV Haar frontal face runs on the grayscale frame (min size 30 px). Haar hits are counted in the report.
- **20% margin = 0.2 × the largest box side on each side.** Crop is square, centred on the box, zero-padded outside the frame, resized to 224x224 INTER_AREA, saved as RGB JPEG quality 95.
- **Atomic writes.** Each JPEG is written to `*.tmp.jpg` and then renamed. A file counts as valid when PIL verifies it and it is 224x224.
- **Resume.** A clip whose 5 frames already exist as valid images is not decoded. Frames that had no face are not stored, so they are retried on each run (they are counted again).
- **Missing-face frames are skipped, not replaced.** A clip can yield fewer than 5 rows. The counts are in the report.
- **JAFFE** = one still per image, detected and cropped the same way, stored as `<stem>_f0.jpg` with `frame_id` 0.
- **fer2013 = passthrough.** No detection, no crop. The original 48x48 path goes into `face_path`, `frame_id` is 0. The loader converts to 3-channel and resizes to 224 at load time.
- **Parallelism.** `multiprocessing` with the spawn context, `min(4, cpu_count())` workers. Each worker creates its own MediaPipe detector in the pool initializer. The main process does all parquet and report writing.
- **Errors per clip.** A failing clip is logged and counted as skipped. It does not stop the pool.
- **Contact sheets** use 40 random rows per dataset (seed 42), shown as 8×5 thumbnails at 160 px with the label under each thumbnail.

## Phase 2 (fixes after the first run)

- **Pool slowdown fixed.** The first pool ran at ~70 s per clip. Each spawned worker used all cores through OpenCV and TFLite. `OMP_NUM_THREADS=1`, `TF_NUM_*_THREADS=1` and `cv2.setNumThreads(1)` are set in the worker. A single clip takes 0.15 to 1.2 s alone; the full run took 29 min at ~6 clips/s.
- **JAFFE subject "NA" was lost.** `pd.read_csv` reads the string `NA` as missing. The 21 `NA.*` faces were written flat to `faces/jaffe/` and recorded with an empty subject. They were moved to `faces/jaffe/NA/`, and `faces_index.parquet` was corrected to match. The index itself was already right. Phase 2 now reads the index with `keep_default_na=False`.
- **Haar fallback never worked.** Every frame was found by MediaPipe (`haar=0` everywhere). One clip (`crema_d/VideoFlash/1076_MTI_SAD_XX.flv`) reached the fallback, and the cascade failed with `!empty()`, so the XML was not loaded. That clip lost its 5 frames (`skipped_no_face=5` for crema_d). Not fixed: the loss is 5 frames out of 87,705, and the fallback is not needed for any other frame.

## Phase 3

- **fer2013 labels come from `label_ferplus`.** FER+ is a relabelled view of FER-2013 (Phase 1 decision). Rows with no FER+ label (2,387, the duplicate-pixel rows with disagreeing votes) are excluded from every split. The folder label is not used for training. Reason: the FER+ votes are the cleaner label, and the spec asks for FER+ as a relabelling.
- **fer2013 split.** `Training` gives train, with 10% carved off as val (stratified, seed 42). `PublicTest` and `PrivateTest` together give test, which is the whole official `test/` folder (6,722 rows after the drop). Reason: the spec says "official split" without naming one; the folder test is the official test set.
- **Subject split for ravdess, crema_d and jaffe.** Three subjects go first, one per split. The rest are placed one by one into the split that is furthest below its row share, with class balance as the tie-break. Result: crema_d 63/14/14 subjects, ravdess 16/4/4, jaffe 6/2/2. Reason: a version that minimised distance to target starved train (one subject), so it was replaced.
- **Class weights** come from the combined train split: inverse frequency, normalised to mean 1 (`data/class_weights.json`).
- **E1 is defined in experiments.json but is not run on Colab.** The Colab steps in the spec list only E2 and E3. The per-dataset rows of E2 are the in-dataset comparison in its place, so Phase 6's E1 table will read from E2. Reason: keeps Colab time inside the 6-hour budget.
- **Macro-F1 per dataset** averages over the classes that appear in that dataset's test labels. crema_d is therefore scored on 6 classes, and its surprise column is excluded.

## Phase 4

- **faces.zip holds only the fer2013 images that are used** (33,500, which is the 35,887 minus the 2,387 unlabelled rows). Crops: 51,818 files under `faces/`. Reason: the unused rows are never read, so there is no point copying them.
- **CSV paths are zip-relative.** `face_path` is rewritten. The `path` column (a source path on this PC) is dropped from the uploaded CSVs.
- **Notebook choices (not fixed by the spec):** AdamW weight decay 1e-4. ResNet head phase at a constant lr 1e-3, then full training at 1e-4 with cosine decay and no warmup. CNN has 2 warmup epochs then cosine. Per-epoch sampler: each dataset may contribute at most 50% of the training set per epoch (this cap does not bind in E2). Early stopping on the val macro-F1 of the training datasets. E3 test set = all rows of the held-out dataset. DataLoader `num_workers=2` on Colab.
- **Model choice for E3** = the model with the higher mean of its four per-dataset E2 macro-F1 scores. `best_model.pt` = that model's E2 best checkpoint.
- **E3 time guard.** Before each held-out run, the estimated time is the winner's mean E2 epoch time x 20 epochs x the run's train size / E2 train size. If elapsed + estimate > MAX_HOURS (default 6), the run is skipped and listed.
- **OOM fallback.** On CUDA out-of-memory the batch is halved and the run resumes from `last.pt`. Not exercised by the local test.
- **Local test.** The notebook code cells were run on CPU with 12 rows per dataset and split, 1 to 2 epochs, with and without E3 (`code/test_notebook.py`). Both passed. Resume from `last.pt` and the OOM fallback were not exercised.

## Phase 5

- **Import script.** `code/phase5_import.py` checks, copies and writes `logs/phase_5_import.json`. It runs with the project venv under `python -I`, because `colab_results/` is downloaded data.
- **Architecture check.** Each `best.pt` is loaded with `weights_only=True` on CPU, strict-loaded into the matching class (`SmallCNN` or torchvision `resnet18` with a 7-way fc), and given one forward pass. Architecture comes from the folder name (`resnet18` or `cnn`). `last.pt` is checked the same way but does not block.
- **Blocking rules.** A checkpoint that fails to load, or a classes.json that differs, stops the import. Nothing is copied in that case. Unparseable files are reported, and any of them keeps phase_done at 4. This run had none.
- **Copies.** `metrics.json`, `history.csv`, `done.json`, `confusion_*.png` and both checkpoints go to `results/<run>/` and `models/<run>/`. The top-level CSVs, PNG, choice.json and classes.json go to `results/`. The top-level `best_model.pt` (byte-identical to `E2_cnn/best.pt`) goes to `models/best_model.pt`, since it is not a run folder.
- **Conflicts.** A differing existing file is never overwritten. The copy is saved as `<name>__colab_import<ext>`, and the report lists it. No conflicts occurred in this run.
- **state.json.** Only `phase_done` changed (4 to 5). The `waiting_for: "colab_results"` key was left as it was, because nothing reads it.
- **`collab_results/` (typo folder).** The zip and copy of the notebook in `project/collab_results/` were not imported. The spec names `colab_results/` only.

## Phase 6

- **Confusion matrices were read from the PNGs.** Colab saved `confusion_<ds>.png` only, with no numbers. `code/phase6_confusion.py` holds the counts read from the four E3 charts. Each matrix must pass two checks against the saved `metrics.json`: row sums equal the per-class support, and diagonal / row sum equals the saved recall. All four pass, so a misread digit would have failed. The counts are in `results/E3_confusion_transcribed.csv`.
- **E1 was not run on Colab.** Phase 3 decided the E1 table reads from the E2 per-dataset test splits. `comparison_all.csv` follows that, and every E1 row is marked `proxy_from_E2`. E2 trains on all four datasets, so it is not a true in-dataset score. It is the best available stand-in and it is labelled everywhere it appears.
- **E3 is CNN only.** Colab's E3 cell trains just the E2 winner (CNN). The ResNet-18 E3 rows are marked `not_run_winner_only` with empty values. No ResNet-18 E3 number was made up.
- **Skipped E3 runs: none.** All four E3 folders have `done.json`, and the saved notebook output has no skip line. The time guard did not fire.
- **Shared-class macro-F1 is derived.** `E3_macro_f1_6_shared` is the mean of the saved per-class F1 over angry..sad (surprise dropped), per the Phase 3 note to report the six shared classes. Nothing was retrained.
- **Top-3 confused pairs** are ranked by count of true→predicted cells, off-diagonal. Row rate is shown next to each count, because the held-out classes have different sizes.
- **Reasons come from the four allowed categories and are hypotheses.** None was tested; each is a reading of the counts plus the dataset facts logged earlier. Grayscale vs colour was not used for the CNN: `build_notebook.py:135` applies `Grayscale(1)` to every dataset, so the CNN never sees colour. Where no category fits cleanly (small JAFFE classes), the closest one is used and the note says so.
- **Charts use palette slots 1 and 2** from `palette.md`, the light values. Both pass `validate_palette.js` in light and dark mode. Only the light PNGs are rendered, since there is no HTML page. The validator does not check layout, so both PNGs were opened and checked by eye.
- **fer2013 held-out is below chance** (accuracy 0.072, chance 0.143). Of its predictions, 60% are disgust. This is reported, not fixed. Phase 6 does no training. The cause is not established.

## Phase 7

- **Packaged model = CNN.** The rule is the higher mean E2 macro-F1 (`results/comparison_all.csv`): CNN 0.614, ResNet-18 0.585. `models/face_emotion_best.pt` is a plain state_dict copied from `models/E2_cnn/best.pt` (same bytes as `models/best_model.pt`). It loads with `strict=True` into `SmallCNN(7)`. Code: `code/phase7_package.py`.
- **Caveat logged, rule not changed.** The winner was chosen on the E2 *test* macro-F1 (`choice.json`), not on validation. On validation, ResNet-18 is higher (0.657 vs 0.611 best val macro-F1). Without JAFFE (44 test images) the mean favours ResNet-18 (0.646 vs 0.609). The rule was followed as specified and the caveat is in the report (section 6, item 1) and the model card.
- **Model card.** `models/model_card.md`. Parameter count 4,689,735, computed from `SmallCNN`. Metrics are copied from the Phase 6 files.
- **`models/classes.json`** = `["angry","disgust","fear","happy","neutral","sad","surprise"]`, checked against `results/classes.json`.
- **infer.py pipeline** follows the training pipeline: MediaPipe BlazeFace (largest box, confidence 0.5), OpenCV Haar fallback, square crop with 20% margin resized to 224 (INTER_AREA), then grayscale, 48x48 bilinear, normalise to [-1,1]. Same as `phase2_faces.py` and the notebook eval transform.
- **Video sampling**: 5 frames at fractions linspace(0.2, 0.8, 5) of the frame count, the same as Phase 2. Averaged over the frames where a face was found.
- **No face at all** (Haar also fails): the whole frame is used and the JSON reports `face_found_frames: 0`. Chosen over failing, so the tool always returns an answer. The flag tells the caller the result is weaker.
- **Output** = JSON on stdout with the 7 probabilities (4 decimals), `top_label`, `frames_sampled`, `frames_in_clip`, `face_found_frames`. Chosen so the web app can parse it directly.
- **Inference test samples** (`code/phase7_test_infer.py`): seed 42, drawn from the test split only. 3 images (fer2013 48px original, jaffe crop, crema_d crop) and 1 video (RAVDESS speech clip). Ground truth is printed beside the output and not passed to infer.py. Output in `logs/phase7_infer_test.json`. The RAVDESS clip is predicted wrong (happy, true disgust). Reported as is, not re-picked.
- **Test images are from the E2 test split.** Models were not trained on them, but the E2 test split was used for choosing the winner (see the first caveat).
- **E1 in the report** is labelled a proxy (the E2 per-dataset scores). E1 was never trained.
- **E3 in the report** is CNN only. ResNet-18 E3 was not run.
- **Colab evidence**: the GPU line (`GPU: Tesla T4`) is in `collab_results/Copy_of_train_colab.ipynb`, the saved notebook with output. That is the source cited in the report. `collab_results/` is the typo folder from Phase 5, which was not imported; it is cited here only as evidence.
- **Report paths** are repo-relative (`project/...`). Figures: `results/comparison_all.png`, `results/E3_summary.png`, the per-run `confusion_*.png` files, and the E3 counts in `results/E3_confusion_transcribed.csv`.
- **state.json** `waiting_for: "colab_results"` is left as it was (Phase 5 and 6 left it too). Nothing reads it. Removing it is a cleanup, not required.

## Web app (./app), added after the training phases

- **Existing app is not MERN.** The root already has `backend/` (FastAPI) and `frontend/` (React). Neither has Express or Mongoose, so `./app/` was created as the MERN app. `backend/`, `frontend/` and `project/` were not changed.
- **Python env: reuse `project/.venv`.** It already has torch, OpenCV and MediaPipe, and a second CPU torch install would cost time and disk. FastAPI, uvicorn, python-multipart, pytest, httpx and websockets were added to it with pip (add only; nothing removed). Logged in `app/model_service/requirements.txt`.
- **Model code is imported, not copied.** `app/model_service/analyzer.py` imports `project/code/infer.py` (SmallCNN, crop, preprocessing, FaceFinder), so the web app and the offline tool use the same code. Checked at startup that `classes.json` equals the model's class order.
- **Model:** `project/models/face_emotion_best.pt` (the CNN, the E2 winner per the Phase 7 rule). The ResNet-18 is not used.
- **Face detection:** MediaPipe BlazeFace is the active detector. The OpenCV Haar fallback cannot load in this venv: `haarcascade_frontalface_default.xml` is missing from `cv2/data` (opencv-python 5.0). MediaPipe found a face in 100 of 100 RAVDESS frames and 3 of 3 test images, so the fallback did not come into play. Not fixed, because the fix would need a file outside the venv.
- **No face means no guess.** The image and stream endpoints return `face_found: false` with null probabilities. `infer.py`'s whole-frame fallback is not used by the service, because it would give a prediction without a face.
- **Several faces: predict the largest one.** `infer.FaceFinder` returns one box, and running the CNN on every face is not needed for this demo.
- **Tracking is implemented but off by default.** `DETECT_EVERY` (default 1) sets how often the detector runs; in between, the last box is reused. Default 1 because detection is about 20 ms per 320 px frame, which already meets the 5-10 FPS target. Raising it is a speed-for-accuracy trade that was not measured.
- **Live frame size 320 px wide, JPEG quality 0.7, every 100 ms, one frame in flight.** A camera frame that comes due while the previous result is pending is dropped, not queued. Measured: 10 FPS in the browser, 22-24 ms round trip (see `app/README.md`).
- **Smoothing:** EMA with alpha = 2/(5+1) = 0.333 over the probabilities of the last ~5 frames. A frame with no face resets the EMA. The label is "uncertain" when the smoothed top probability is below 0.5. The raw top label is shown in small text.
- **Live page is the home tab.** Camera permission is requested on load. Start turns detection on, Stop turns detection off and releases the camera (so the "Camera is on" light goes off). Denied or no camera: a message plus a button to the Upload tab.
- **Save session:** only summaries (timestamp, probabilities, top label, confidence, model, face flag) are stored, at most one per second (`SAVE_INTERVAL_MS`). No frame or image is ever written to disk or the database. Uploads are always saved, as the original brief asked.
- **Database:** no local MongoDB is installed on this PC. The default is `MONGO_MODE=auto`: local `mongodb://127.0.0.1:27017`, else `mongodb-memory-server` (downloads about 600 MB once; data is lost when the server stops). Stated in the README.
- **Reconnect:** exponential backoff from 0.5 s up to 4 s, with a "Reconnecting…" state. Tested by killing and restarting Express (page back to live in 6.2 s).
- **Placeholders:** Audio, Text and EEG show "Model not trained yet" and no data.
- **Stream test:** the client simulates a 100 ms camera clock with 100 real RAVDESS frames, and the browser test uses a RAVDESS clip as the webcam (Chrome fake device). The stream test measures the server path; the browser test measures the page. The no-camera message was NOT verified in the browser: this PC has a real camera, so the no-device case cannot be reproduced here.
- **Ports:** model service 8001, Express 4000 (`/api`, `/ws/live`), Vite 5173 (proxies both).
- **Test accuracy is reported, not tuned.** On the three sample images the model gets 2 of 3 right (neutral is predicted angry at 0.91). Not changed to pass.

## Speech emotion pipeline (audio phases 0-2)

- **Datasets.** All four present on disk and COMPLETE (no zip extraction needed): RAVDESS speech (`datasets/RAVDESS/Audio_Speech`, 1440), CREMA-D (`datasets/crema_d/AudioWAV`, 7442), TESS (`datasets/tess`, 2800), EMO-DB (`datasets/emodb/wav`, 535). The notebook `prompts/audio/audio_colab_train.ipynb` does not exist on disk and was not supplied, so it could not be copied into the Colab package. Logged in the phase reports.
- **Venv.** `soundfile 0.14.0`, `librosa 1.0.0` installed; `torchaudio==2.11.0+cu128` matches torch 2.11.0+cu128 (torch not changed). Added to `project/requirements.txt`.
- **Classes.** The same seven as the face model, in the same order.
- **RAVDESS.** Vocal channel 01 (speech) only. `calm` (02) merged into `neutral`, as the brief says. Speaker = actor id (01-24).
- **CREMA-D.** No surprise clips exist, so surprise is absent for this dataset. Speaker = ActorID.
- **TESS.** Speakers OAF and YAF. The stray file `OA_bite_neutral.wav` (prefix `OA`) is kept as its own speaker. `ps` maps to surprise.
- **EMO-DB.** Boredom (`L`, 81 clips) has no class in the seven-class set. Mapping it to neutral would blur a distinct emotion, so it is EXCLUDED and logged as `boredom_excluded`. Surprise is absent (no such class in EMO-DB). Other letters: W angry, E disgust, A fear, F happy, T sad, N neutral.
- **Silent clips.** Clips with peak < 1e-4 are excluded and logged. One clip (CREMA-D) was excluded this way.
- **Speaker splits (seed 42).** RAVDESS 24 actors, CREMA-D 91 actors: about 70/15/15 by speaker (round to nearest, at least one speaker per split). EMO-DB 10 speakers: 7/1/2 as the brief specifies.
- **TESS is split by WORD, not speaker.** TESS has only two real speakers, so a speaker split would leave one speaker per split. Word split: no word is in two splits. Both speakers appear in every split, so TESS results are NOT speaker-independent. Reported wherever TESS results are shown.
- **Conversion.** 16 kHz mono float32 WAV (`soundfile` FLOAT subtype), trimmed with `librosa.effects.trim(top_db=30)`, centre-cropped to 4.0 s when longer, NOT padded (the notebook pads). Clips that become empty after trimming are skipped.
- **Colab zip.** `project/colab_upload_audio/audio_16k.zip` is stored uncompressed (ZIP_STORED), because WAV data does not compress and stored zips are faster to read on Colab. It holds `index.csv` plus all WAVs.

## Audio phase 0 (rerun 2026-10-12)

- **Phase 0 audited six datasets, plus EMO-DB as an extra.** ravdess, crema_d, tess, savee, meld, esd are COMPLETE and used. EMO-DB is COMPLETE too but was not in the phase 0 list, so it is recorded in `audio_state.json` only.
- **Audit is read-only and single-pass.** `code/audit_audio_datasets.py` walks each dataset once. It takes the first matching folder per dataset, because Windows folder names are case-insensitive.
- **Zero-byte status.** Zero-byte files under `.git`, log files, and non-audio files (`silb/13a01Wb.silb`) do not make a dataset PARTIAL. The audio folders have no zero-byte or partial files.
- **No zip extraction was needed.** RAVDESS speech, CREMA-D, TESS, SAVEE, MELD, and EMO-DB are already extracted. ESD's English speakers are extracted; `ESD.zip` is kept.
- **Unused RAVDESS song zips are left as they are.** Phase 0 does not extract them, and they are not in the speech subset.
- **Stale runner.** The headless `claude -p` run from `run_audio_phases.ps1` (PID 48636) was hung from 08:31, and it was not killed. Setting `phase_done: 0` means that when it exits, the runner moves on to phase 1 automatically. Stop the runner before it does if phase 1 should not start yet.
- **Existing state files kept.** `audio_state.json` already had `index_clips` and `audio_16k_clips` from the earlier phase 1-2 run; they were kept, not recomputed.

## Audio phases 4-5 (ingest, report, app)

- **Input folder was wrong.** `project/colab_results_audio/` held the FACE results (a copy of `collab_results`: the face zip, notebook and `extracted/`). The audio results are `project/colab_results-20261012T155842Z-1-001.zip` (E2_combined, E3_holdout_{crema_d,emodb,ravdess,tess}, summary.csv). It was extracted to the new folder `project/colab_results_audio_ingest/` and the result files copied to `project/results/audio/`. `colab_results_audio/` was left untouched.
- **Which notebook.** `colab_upload_audio_v2/audio_colab_train.ipynb` (byte-identical size to the v1 copy) is the training notebook. Architecture, features (16 kHz, 4 s, mel 64, n_fft 512, hop 160, per-clip normalisation inside `forward`) and class order were read from it.
- **This run used the 4-corpus package, not the 7-corpus one.** Result datasets: crema_d, emodb, ravdess, tess; test n = 1,881, which equals the test split of `data/audio_16k/index.csv` (12,135 clips). `data/audio_16k_pcm/` (7 corpora, 23,174 clips, incl. MELD, ESD, SAVEE) was NOT what was trained on. `audio_state.json` still lists `v2_clips: 23174`; left as it was. The report says plainly that MELD, ESD and SAVEE were not used.
- **Packaging.** `audio_best.pt` = `best_model.pt` = `E2_combined/best.pt` (hash-checked). Classes saved as `models/audio_classes.json` (not `classes.json`, to avoid clashing with the face model's file, which has identical content). Order matches angry, disgust, fear, happy, neutral, sad, surprise.
- **Preprocessing for raw input.** The training WAVs were already silence-trimmed (top_db 30) and centre-cropped at dataset preparation, then zero-padded / cut to the first 4 s in the notebook. `predict_audio` applies trim + centre-crop only to input that is not already in `data/audio_16k` (`prepare` defaults on for uploads and arrays). Verification: all 1,881 test clips give 0.5795 accuracy, identical to the Colab E2 ALL figure; the requested 20-clip sample (seed 42) gave 0.70, reported only as a small-sample number.
- **Browser recordings are re-encoded to WAV in the browser** (AudioContext decode, 16-bit mono), so ffmpeg is not required for the Record button. If that step fails the raw WebM is sent and ffmpeg decodes it. ffmpeg exists on this PC (`C:\Program Files (x86)\ffmpeg\bin`); a WebM clip produced with it gave the same prediction as the WAV (0.484 vs 0.482 happy; small difference from the Opus codec).
- **Mongoose.** `Prediction.face_found` is now optional (default false), `source` also allows `audio_upload` and `audio_record`, and a new `audio` sub-document holds top_label, confidence, duration_s and the 7 probabilities. Existing documents stay valid. The History tab shows "n/a" in the Face column for audio rows.
- **Client tabs.** A new "Audio" tab; the old "Audio · Text · EEG" placeholder tab became "Text · EEG".
- **Tests used separate ports** (model service 8011, Express 4010) so the stack already running on 8001/4000 was not touched. Two test-split clips (a CREMA-D happy clip and a TESS happy clip) gave happy at 0.48 (flagged uncertain) and 0.96. Node tests: 12 pass. pytest: 14 pass. `npm run build` succeeds.
- **Not verified:** the Record button and page layout in a real browser; the app was not driven with a microphone.
- **E3 results are noisy** (see the report, section 5): E3 CREMA-D's best checkpoint is from epoch 1. Not rerun, as instructed.
- **Old stack still running old code.** The services already listening on 8001, 4000 and 5173 were started before these changes. Restart them (`npm run dev`) to get the audio endpoint and tab.

## Wellbeing phase W1 (Gemini check-in and safety layer)

- **A real Gemini API key was found in `app/server/.env.example`.** It was moved to `app/server/.env` (git-ignored; `git check-ignore` confirms) and `.env.example` now has an empty `GEMINI_API_KEY=`. The `app/` folder had never been committed, so the key was not in git history. It was in a plain file, so consider rotating it.
- **Model pick.** The list-models endpoint (called once with the key) shows many flash models. `gemini-3.8-flash` (newest) answered 503 "high demand" on the test call; `gemini-2.5-flash` answered correctly with a JSON schema. Default `GEMINI_MODEL=gemini-2.5-flash` (stable, not a preview), `GEMINI_FALLBACK_MODEL=gemini-2.5-flash-lite`. The client retries once on 429/5xx/network, then moves to the fallback model. `thinkingBudget: 0` is sent for 2.5 models to keep replies fast.
- **Key handling.** The key is read from `.env` only and sent only in the `x-goog-api-key` header (never in a URL or a log; a test asserts this).
- **No key means MOCK mode:** canned questions, a keyword heuristic for mood and emotions (no PHQ items), and `mock: true` plus a `notice` on every response, `/status` and the final summary. The safety layer is identical in mock mode.
- **Session length.** `WELLBEING_MAX_TURNS` (default 8, clamped 6 to 10) counts user messages. The check-in closes after the last one; `POST /end` closes it early. After a safety trigger the session is not closed on that message, and may run up to 4 extra messages.
- **Opening.** The first assistant message is exactly "How are you feeling today?". The AI-not-a-clinician disclosure is a separate `intro` field returned by `POST /sessions`.
- **Topic weaving.** Order: mood, sleep, interest, energy, appetite, concentration, self-worth, restlessness, self-harm. One topic per reply at most, never repeated (`asked_topics`), skipped when the user already addressed it (analysis returned a value for it). With 8 messages not all nine can fit, so some are not asked. The self-harm question is asked only after 4 messages and only if there are signs of distress (mean mood at or below 6, any other PHQ item at 2 or more, or a "possible" signal), and never after the tier is `needs_support_now`.
- **Safety order.** Per message: regex detector first, then the separate analysis call (still run when the regex already fired, so the turn is recorded), then the combination (highest of the two levels). `explicit` returns the fixed message and never calls the chat model. Tested with Gemini fully down.
- **Regex design.** Explicit: first-person suicide or self-harm intent, strong hopelessness ("no reason to live", "better off without me", "tired of living"), and Hinglish forms (jeena nahi, mar jana chahta, khud ko khatam, etc.). Negated forms ("I'm not suicidal", "I don't want to die") and soft hopelessness ("hopeless", "can't take it anymore") are `possible`, which appends a fixed one-line helpline note and sets `possible_self_harm`, but not the human-followup flag. Idioms ("this exam is killing me", "dying to see", "want to die of embarrassment", suicide in an academic sentence) do not trigger. The detector deliberately errs towards triggering: "I'll kill myself if I fail" fires. The Gemini `explicit` signal can also fire on an ambiguous message; that is accepted (a false alarm costs one supportive message).
- **After a trigger.** Tier `needs_support_now`, `needs_human_followup` true, `safety_triggered_at` set. Later non-explicit messages get Gemini replies under a "stay gentle, no screening, no new topics" prompt; a further explicit message gets a shorter fixed message with the helplines.
- **Output guard.** A Gemini reply that mentions methods (overdose, noose and similar) is discarded and replaced by a canned line (`kind: fallback`).
- **Final score.** `wellbeing_score` 0 to 100 (higher is better): 60% PHQ distress (sum of the highest reported value per answered item, scaled to 0-3 per item) plus 40% mood distress; if only one is present it counts fully. Tiers: 70 or more `doing_well`, 50 to 69 `some_stress`, below 50 `needs_attention`; `possible_self_harm` raises the floor to `needs_attention`; the human-followup flag forces `needs_support_now`. A rough summary, not a clinical score, and the response says so. Modality (face or audio) samples are stored but not used in the score yet.
- **Storage.** Sessions are cached in memory and written to MongoDB through the existing connection when it is up. Conversation text is stored (needed for the check-in); no images or audio are.
- **Routes.** `createApp()` mounts `/api/wellbeing` before the existing `/api` router. The existing routes and tests are unchanged. 79 Node tests pass (67 new), pytest still passes.
- **Live check (real key).** One 4-message conversation: replies were natural and short, the analysis JSON validated every time (no fallback), "this exam is killing me lol" was a normal reply, and "...I feel like I want to die" returned the fixed safety message (both regex and Gemini said explicit). Not tested live: a Gemini outage (covered by mocks).
- **Not done in this phase:** any client UI for the check-in, and fusion of face/audio samples into the score.

## Wellbeing phases W2 to W4 (scoring, UI, voice interview, model-driven score, safety review)

- **Folder.** The spec says `app/server/wellbeing/`; the module lives in `app/server/src/wellbeing/` next to the rest of the server code (`scoring.js`, `config.json`, `model_reliability.json` are there).
- **Endpoints.** The singular routes from W2 (`/session`, `/session/:id/message`, `/finish`, `/signal`, `/history`) are the main ones; the W1 plural routes still work. User identity is a random id in the `X-Wellbeing-User` header (no login).
- **Model fallback.** `gemini-2.5-flash` hit its free-tier quota during evaluation, and `gemini-2.5-flash-lite` is "no longer available to new users" (404). Defaults are now `gemini-3.5-flash` (primary) and `gemini-3.5-flash-lite` (fallback); 3.x models get `thinkingLevel: low`. A 429 or 5xx is retried once, then the fallback model is tried.
- **Tier names** changed to `doing_okay`, `struggling`, `needs_support_now`. Tier labels shown to users are plain language ("Doing okay", "Having a harder time", "Support would help now").
- **Model-fusion thresholds 45 / 30** (not 65 / 40) because the valence mapping puts neutral at 50. Unvalidated.
- **Low confidence** is defined as fusion confidence under 0.4 (config). "Avoid the top tier" was read as: never `needs_support_now` from models alone; only the rule-based safety flag or item 9 can set it.
- **No text or EEG model exists.** Their slots are in the config and reliability file with `null`; they are skipped. A chat without camera or voice therefore has no score.
- **Reply streaming** from Gemini was not implemented; replies are short and are spoken sentence by sentence once complete.
- **Voice interview defaults.** Local faster-whisper `base.en` (int8, CPU) via the model service; browser Web Speech and Gemini transcription are optional settings. First answer = personal baseline; deltas are relative changes (a single baseline sample has no standard deviation, so no true z-score). Camera is optional in the interview (voice-only if it is missing); the microphone is required, otherwise it falls back to chat.
- **Face sampling in the interview** reuses the existing live WebSocket pipeline; only the samples taken while listening are aggregated and sent, as numbers.
- **Retention**: sessions expire after 90 days (`WELLBEING_RETENTION_DAYS`). `Prediction` summaries from the upload and live pages have no user id and are not removed by "Delete my data".
- **Rate limit** 30 messages per minute per user id (`/message`, `/voice-answer`, `/skip`, `/tts`).
- **Safety review fixes**: error logging reduced to the error name; transcript only returned to its owner; reply guard against diagnosis wording; Gemini disclosure added to both consent screens; the API key moved out of `.env.example` (see W1).
- **Evaluation caveat**: the 35 transcripts and their expected tiers were written and labelled by the same author; the mock heuristic was tuned once after a first run on them. The Gemini run (3.5-flash) was not tuned.
- **Test environment**: browser tests use system Chrome via `playwright-core` (no browser download), a fake camera (still JAFFE face) and a spoken WAV made with Windows speech synthesis as the microphone. Running them needs the MOCK server, the model service and `vite preview` (ports 4030, 8011, 5190 were used so the normal stack was not disturbed).
