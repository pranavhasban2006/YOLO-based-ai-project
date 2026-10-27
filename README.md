# Multimodal Emotion Detection: web app

> **Student research prototype. Not a medical or diagnostic tool.**

A FastAPI backend and a React frontend for the multimodal emotion detection project.
Face (ResNet-18 / CNN on FER-2013), speech (HuBERT + Transformer) and text (RoBERTa + BiLSTM +
Attention on GoEmotions) are real models. EEG is a stub that returns clearly labelled placeholder
data (`"mock": true`); the physiological models were evaluated offline on WESAD (see
`project/reports/final/`). Audio and text fall back to the mock automatically if their checkpoint
in `models/` is missing.

## Run it

You need two terminals, one command each (after the one-time setup below).

**Backend** (http://localhost:8000, API docs at http://localhost:8000/docs):

```powershell
cd backend
.venv\Scripts\python -m uvicorn app.main:app --port 8000
```

**Frontend** (http://localhost:5173):

```powershell
cd frontend
npm run dev
```

**Tests:**

```powershell
cd backend
.venv\Scripts\python -m pytest
```

### One-time setup

1. Put the checkpoints in `models/`: `resnet18_fer.pt` and/or `cnn_baseline.pt`.
   `models/blaze_face_short_range.tflite` (the MediaPipe face detector) is already there; to
   re-download it, run:
   `curl -L -o models/blaze_face_short_range.tflite https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite`
2. Backend (Python 3.12; mediapipe doesn't support every newer Python release):
   ```powershell
   cd backend
   py -3.12 -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
   ```
3. Frontend (Node 20+): `cd frontend` then `npm install`.

## Configuration (environment variables, read by the backend)

| Variable | Default | Meaning |
|---|---|---|
| `ACTIVE_FACE_MODEL` | `resnet18` | `resnet18` or `cnn`. Falls back to the other checkpoint if the file is missing. |
| `MODELS_DIR` | `./models` | Where checkpoints live |
| `DB_PATH` | `backend/data/sessions.db` | SQLite file for session results |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Frontend dev origins |
| `MAX_IMAGE_MB` / `MAX_VIDEO_MB` / `MAX_AUDIO_MB` | `10` / `100` / `25` | Upload limits |

The frontend reads `VITE_API_URL` (default `http://localhost:8000`).

## Folder layout

```
models/                      checkpoints (not committed) + MediaPipe face detector
backend/
  app/
    main.py                  app factory, CORS, error handlers, /health
    config.py                all settings, class order, limits
    schemas.py               the common Prediction shape and its extensions
    errors.py                error codes returned as {"error": ..., "message": ...}
    face/pipeline.py         decode → MediaPipe largest face → square crop with 20% margin
    modalities/
      base.py                Predictor interface + MockPredictor
      registry.py            the ONE place modalities are registered
      face.py                EmotionCNN / ResNet-18 definitions, checkpoint loading, preprocessing
      audio.py, text.py, eeg.py   placeholders (mock)
    video.py                 5 fps sampling (PyAV, in memory), 5-frame moving average
    fusion.py                confidence-weighted late fusion
    db.py                    SQLite: sessions + predictions
    routes/                  predict.py, ws.py, sessions.py, uploads.py
  tests/                     pytest suite
frontend/
  src/
    App.jsx                  tabs, health polling, fused card
    api.js                   API client, class order
    theme.js                 chart palette (light/dark)
    components/              charts, result card, fused card, notices
    tabs/                    Live, Upload, Audio/Text/EEG (StubTabs), History
```

## API

Every prediction has the shape
`{"modality", "probs": [7 floats], "label", "confidence", "timestamp", "mock"}`
with `probs` in the order `["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]`.
Errors return `{"error": "<code>", "message": "<human readable>"}`.

| Endpoint | Notes |
|---|---|
| `GET /health` | Loaded and unavailable checkpoints, active model, which modalities are mocked |
| `POST /predict/image` | multipart `file`. Adds `faces_found`, `face_box` (normalised), `model`, `session_id` |
| `POST /predict/video` | multipart `file`. Adds `faces_found`, `frames_sampled`, `frames_without_face`, `duration`, `timeline` |
| `WS /ws/webcam` | send binary JPEG frames, one JSON reply each: `prediction`, `no_face` or `error`. Wait for the reply before sending the next frame. |
| `POST /predict/audio` | multipart `file` (mock for now) |
| `POST /predict/text` | `{"text": "..."}` (mock for now; the text itself is not stored) |
| `WS /eeg/stream` | pushes one prediction per second; accepts `{"samples": [...]}` (mock for now) |
| `POST /predict/fused` | `{"predictions": [<prediction>, ...]}`. Confidence-weighted average; skips mock, unknown and duplicate entries. Adds `contributing`, `skipped`, `weights`. |
| `GET /sessions`, `GET /sessions/{id}` | stored results |

Error codes: `no_face_detected` (422), `unsupported_file_type` (415), `file_too_large` (413),
`corrupt_video` (422), `invalid_image` (400), `model_unavailable` (503), `no_real_modalities` (422).

### How the face pipeline works

1. The image is decoded in memory, and EXIF orientation is applied.
2. MediaPipe's short-range face detector finds faces; the largest one is used.
3. A square crop is taken around it: side = longer box side × (1 + 2 × 0.20), i.e. a 20% margin on
   each side, clamped to the image. Square, because the models were trained on square FER-2013 crops.
4. The active model's preprocessing is applied:
   - CNN: grayscale (1 channel), 48×48, normalise with mean 0.5 and std 0.5
   - ResNet-18: grayscale to 3 channels, 224×224, ImageNet normalisation

Video: frames are sampled at 5 fps, frames with a face are classified, and the probabilities are
smoothed with a trailing 5-frame moving average. The clip result is the mean of the smoothed
probabilities.

Fusion: `fused = Σ cₘ·pₘ / Σ cₘ`, where `cₘ` is the confidence (max probability) of modality m.

## Adding a real model (e.g. audio)

1. In `backend/app/modalities/audio.py`, write a `Predictor` subclass: set `modality = "audio"`,
   load weights in `load()`, and return `Prediction.from_probs("audio", probs)` from `predict(data)`.
   Keep the class order above.
2. In `backend/app/modalities/registry.py`, register it instead of `MockAudioPredictor`.

Nothing else changes: `/health` stops listing audio as mocked, the frontend removes the
"coming soon" label, and fusion starts including audio.

## Privacy

Webcam frames and uploads are processed in memory and never written to disk. Only prediction
results (label, probabilities, timestamps, model name, file name) are stored in SQLite. The text
tab stores only the character count, not the text.
