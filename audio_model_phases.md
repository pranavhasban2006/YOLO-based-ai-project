# Audio model phases

## Phase 0: Setup and audit

```
Phase 0: Setup and audit
Unattended. Never ask me questions. Pick the default, log it in project/DECISIONS.md,
and continue. Do not download or delete anything under ./datasets/.

GOAL: prepare for a multi-dataset speech emotion model.

1. Ensure ./project/ exists with code, data, logs, models, results, reports. Create
   project/audio_state.json = {"phase_done": -1} if missing.
2. Reuse the existing venv. Install if missing and append to project/requirements.txt:
   soundfile, librosa, torchaudio (match the installed torch), pandas, scikit-learn,
   tqdm, matplotlib, seaborn.
3. Audit ./datasets/ for ravdess, crema_d, tess, savee, meld, esd (also accept names like
   ravdess_speech, MELD.Raw). Report existence, file count, size, any .part/.tmp/
   zero-byte files, unextracted zips (extract them into their own dataset folder if
   the extract does not exist yet; never delete the zip). Mark COMPLETE, PARTIAL or
   MISSING. Save to project/logs/audio_phase_0_report.md and set
   audio_state.json "datasets_available".
4. Continue with COMPLETE datasets only. Stop only if none of the four is usable.
5. Set phase_done = 0.
```

## Phase 1: Inspect and build the master index

```
Phase 1: Inspect and build the master index
Unattended. Precondition: project/audio_state.json phase_done >= 0, else stop and say why.

GOAL: one index table of every usable audio clip with label and speaker.

1. For each available dataset, look at real file names and folders (do not assume) and
   save a short description to project/logs/audio_layout_<dataset>.txt.
2. Parse labels and speakers from file names using the mapping in this prompt:
   RAVDESS: modality-channel-emotion-intensity-statement-repetition-actor.wav, emotion
   01 neutral 02 calm(->neutral) 03 happy 04 sad 05 angry 06 fearful 07 disgust
   08 surprised; vocal channel 01 = speech only; speaker = actor id.
   CREMA-D: ActorID_Sentence_Emotion_Level.wav, speaker = ActorID.
   TESS: OAF/YAF prefix is the speaker; emotion is the last token of the file name or
   the folder name; ps -> surprise.
   SAVEE, MELD and ESD-English: use the mapping at the top. For MELD run ffmpeg to
   extract 16 kHz mono wav from each mp4 into project/data/meld_wav/ first (skip files
   that already exist) and read the csv files for emotion, speaker and split.
3. Write project/data/audio_index.csv with columns: path (relative to datasets/), dataset,
   speaker, label, duration_sec, sample_rate. Read duration and rate with soundfile.
   Flag unreadable or silent clips (peak < 1e-4) in project/logs and exclude them.
4. Print and save a dataset x label count table to project/logs/audio_phase_1_report.md,
   noting which classes are absent per dataset (CREMA-D has no surprise; ESD has no fear or disgust).
5. Set phase_done = 1.
```

## Phase 2: Preprocess, speaker-level splits, Colab package

```
Phase 2: Preprocess, speaker-level splits, Colab package
Unattended. Precondition: audio_state.json phase_done >= 1.

GOAL: clean 16 kHz clips, leakage-free splits, and a single zip for Colab.

1. Using project/data/audio_index.csv, convert every clip to 16 kHz mono 16-bit PCM wav (subtype PCM_16, to keep the zip small) in
   project/data/audio_16k/<dataset>/<name>.wav. Trim leading/trailing silence
   (librosa.effects.trim top_db=30), keep it at most 4.0 s (centre crop), do NOT pad
   here (the notebook pads). Skip clips already converted.
2. Size cap: if a dataset has more than 6000 clips (MELD, ESD, CREMA-D can), keep a
   stratified random subset of 6000 (by label, seed 42) so Colab training stays within a
   few hours. Record original and kept counts in DECISIONS.md. Do the cap BEFORE the
   split step below, and keep all clips of rarer classes first (fear, disgust, surprise).
   Assign splits BY SPEAKER inside each dataset so the same speaker never appears in
   two splits: about 70% train, 15% val, 15% test of speakers, seed 42. Make sure each
   split has at least one speaker for every dataset; for TESS (2 speakers) split by
   sentence/word list instead and record that in DECISIONS.md because it is not
   speaker-independent. SAVEE has 4 speakers: use 2/1/1. ESD-English has 10: use 7/1/2. MELD: use its official
   train/dev/test split (dev = val); speakers overlap there, record that in DECISIONS.md.
3. Write project/data/audio_16k/index.csv with columns: path (relative to that folder,
   forward slashes), dataset, speaker, label, split. Print the dataset x split x label
   table to project/logs/audio_phase_2_report.md and check there is no speaker overlap
   across splits (assert).
4. Create project/colab_upload_audio/audio_16k.zip containing index.csv and all wavs
   (store, no heavy compression). Also copy ../prompts/audio/audio_colab_train.ipynb
   next to it if that path exists, otherwise copy it from wherever it was supplied.
5. Print the zip size. Set phase_done = 2.
```

## Phase 3: Colab handoff (training happens in Colab)

```
Phase 3: Colab handoff (training happens in Colab)
Unattended. Precondition: audio_state.json phase_done >= 2.

Do NOT train anything locally. Write project/COLAB_AUDIO_INSTRUCTIONS.md with exact
steps for me:
 1. In Google Drive create MyDrive/emotion_audio/colab_upload/ and upload
    project/colab_upload_audio/audio_16k.zip into it.
 2. Upload audio_colab_train.ipynb to Colab, Runtime > Change runtime type > T4 GPU,
    then Runtime > Run all. If the browser disconnects, reconnect and Run all again; it
    resumes and skips finished runs.
 3. When the cell prints ALL DONE, download MyDrive/emotion_audio/colab_results/ to
    project/colab_results_audio/ on this PC (summary.csv, E2_*.json, E3_*.json,
    *_best.pt, cm_*.png).
 4. Then run Phase 4.
State clearly that expected total Colab time is about 2 to 3 hours on a T4.
Set phase_done = 3 and stop.
```
