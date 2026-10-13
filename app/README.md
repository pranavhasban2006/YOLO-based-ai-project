# Face Emotion Live (MERN + Python model service)

Real-time facial emotion detection in the browser. The home page is a live webcam view with a face box,
emotion label, 7 probability bars and a 30-second history chart. Photo upload and history are secondary tabs.

> **Experimental. Trained on acted datasets; accuracy drops on real-world photos (held-out FER-2013 macro-F1 0.09, RAVDESS 0.41, CREMA-D 0.42).**
> Research prototype. Not a medical or diagnostic tool.

```
browser (React + Vite, :5173)
   │  /api/*  and  WebSocket /ws/live   (Vite proxies both to :4000)
   ▼
Express server (Node, :4000)  ── MongoDB (summaries only)
   │  HTTP /predict/image, /predict/frames      WebSocket /predict/stream
   ▼
model service (Python, FastAPI + uvicorn, :8001)
   ├ project/models/face_emotion_best.pt  +  MediaPipe BlazeFace (Haar fallback)
   └ project/models/audio_best.pt  (speech emotion, POST /predict/audio)
```

## Quick start (Windows, PowerShell)

Open the `app` folder. You need **Node 20+** and the project's Python venv (`project\.venv`, already set up for training).

```powershell
# 1. Node packages (root scripts, server, client)
cd D:\Projects\YOLO\app
npm run install:all

# 2. Python packages for the model service, into the existing project venv (adds only)
D:\Projects\YOLO\project\.venv\Scripts\python.exe -m pip install -r model_service\requirements.txt

# 3. Start everything: model service, Express server, Vite client
npm run dev
```

Then open **http://localhost:5173**. Press **Start** and allow camera access.

If the venv is somewhere else, set `PYTHON` first: `$env:PYTHON="C:\path\to\python.exe"; npm run dev`.

Each service also runs alone:

| What | Command (from `app`) | URL |
|---|---|---|
| Model service | `npm run dev:model` | http://127.0.0.1:8001/health |
| Express server | `npm run dev --prefix server` | http://localhost:4000/api/health |
| Client | `npm run dev --prefix client` | http://localhost:5173 |

### Tests

```powershell
npm test                                        # Node tests (146) + model service pytest (26)
npm run stream-test                             # 100 frames through /ws/live (needs the stack running)
```

`scripts/make_stream_frames.py` (project venv) extracts the 100 test frames from a RAVDESS clip into `scripts/.cache/frames`.

## Database

The server looks for a **local MongoDB** at `mongodb://127.0.0.1:27017/emotion_app` first. This PC does not
have MongoDB installed, so the default is `MONGO_MODE=auto`: if the local server is not reachable, the server
starts **mongodb-memory-server**. That downloads a MongoDB binary (about 600 MB) the first time, and the data
lives only while the server runs. For data you keep, install MongoDB Community and it is used automatically.
The header of the page shows which one is connected (`MongoDB (local)` or `MongoDB (memory)`).

## Configuration

Copy `server/.env.example` to `server/.env` to change anything. Defaults:

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `4000` | Express port |
| `MODEL_SERVICE_URL` | `http://127.0.0.1:8001` | Python service |
| `MONGODB_URI` | `mongodb://127.0.0.1:27017/emotion_app` | local MongoDB |
| `MONGO_MODE` | `auto` | `local`, `memory`, or `auto` (local, else memory) |
| `MAX_IMAGE_MB` | `10` | upload limit |
| `SAVE_INTERVAL_MS` | `1000` | webcam summaries stored at most this often (only with "Save session") |

Model service (`model_service/.env.example`): `MODEL_PATH` (default `project\models\face_emotion_best.pt`) and
`DETECT_EVERY` (default `1`: detect on every frame).

## What works

- **Live camera (home):** camera permission on load; Start and Stop; a face box in the emotion colour with label and
  confidence; 7 live probability bars; a 30-second chart; FPS, round-trip latency, model name and dropped frames in
  the corner; a "Camera is on" light; "Save session" checkbox.
- **Smoothing:** exponential moving average over about the last 5 frames (alpha 1/3). Shows "uncertain" when the top
  probability is below 0.5, "No face" when there is no face, and the raw top label in small text.
- **Backpressure:** one frame is in flight at a time. A camera frame that comes due while the previous result is
  still pending is dropped, not queued. The count is shown as "Dropped".
- **Reconnect:** if the socket drops, the page shows "Reconnecting…" and retries with backoff (0.5 s up to 4 s).
- **Upload tab:** drag and drop or choose a JPEG/PNG/WebP/BMP; shows the face box, the 7 probabilities, and the label.
  No face gives a message, not a guess.
- **History tab:** the last 50 stored summaries.
- **Privacy:** frames are processed in memory and never written to disk or the database. Only summaries are stored,
  and webcam summaries only while "Save session" is ticked.
- **Placeholders:** Audio, Text and EEG show "Model not trained yet". They show no data.

### Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /api/predict/image` | multipart `file`. Stored as `upload`. |
| `POST /api/predict/frames` | JSON `{"frames": [base64 images]}`, averaged over frames with a face. Stored as `frames`. |
| `GET /api/history` | last 50 summaries (503 `db_unavailable` when no database) |
| `GET /api/health` | model service state and database state |
| `WS /ws/live` | browser frames (binary JPEG) in, JSON results out. Control: `{"type":"config","save":true}` |
| model `GET /health`, `POST /predict/image`, `POST /predict/frames`, `WS /predict/stream` | the Python service |

Result shape: `{face_found, box:{x,y,w,h} (0..1), probabilities:{7 classes}, top_label, confidence, ms, model}`.
Class order everywhere: `angry, disgust, fear, happy, neutral, sad, surprise` (from `project/models/classes.json`).

## Measured results

All numbers below come from the saved runs in `app/scripts/out/` and `app/scripts/.cache/browser/`.

### 1. Real images through the Express API (`scripts/out/api-e2e.json`)

Expected labels are the FER-2013 folder names. Request and response, saved as returned by `POST /api/predict/image`:

```json
[
{"request":{"file":"datasets/FER-2013/extracted/test/happy/PrivateTest_47631580.jpg","expected_label":"happy"},"response":{"face_found":true,"box":{"x":0.0625,"y":0.0625,"w":0.8958333333333334,"h":0.8958333333333334},"probabilities":{"angry":0.0171,"disgust":0.0173,"fear":0.0173,"happy":0.8691,"neutral":0.007,"sad":0.0089,"surprise":0.0633},"top_label":"happy","confidence":0.8691,"tracked":false,"ms":68,"model":"face_emotion_best.pt","saved":true}}
,
{"request":{"file":"datasets/FER-2013/extracted/test/angry/PrivateTest_10131363.jpg","expected_label":"angry"},"response":{"face_found":true,"box":{"x":0.10416666666666667,"y":0.125,"w":0.7916666666666666,"h":0.7916666666666666},"probabilities":{"angry":0.618,"disgust":0.091,"fear":0.025,"happy":0.1934,"neutral":0.0096,"sad":0.0382,"surprise":0.0249},"top_label":"angry","confidence":0.618,"tracked":false,"ms":14.5,"model":"face_emotion_best.pt","saved":true}}
,
{"request":{"file":"datasets/FER-2013/extracted/test/neutral/PrivateTest_10086748.jpg","expected_label":"neutral"},"response":{"face_found":true,"box":{"x":0,"y":0.125,"w":0.8333333333333334,"h":0.8333333333333334},"probabilities":{"angry":0.9145,"disgust":0.0061,"fear":0.0224,"happy":0.0028,"neutral":0.0277,"sad":0.006,"surprise":0.0206},"top_label":"angry","confidence":0.9145,"tracked":false,"ms":20.3,"model":"face_emotion_best.pt","saved":true}}
]
```

**Correct: 2 of 3.** The neutral image is predicted angry at 0.91. This is reported, not tuned away.

### 2. 100 frames through `/ws/live` (`npm run stream-test`)

Source: 100 consecutive frames of RAVDESS `01-01-05-01-01-01-01.mp4` (Actor 01), 320 px wide, JPEG 0.7 (about 5 KB each).

| Camera clock | Sent | Dropped (backpressure) | Results/s | Avg round trip | p95 round trip | Avg model time |
|---|---|---|---|---|---|---|
| 100 ms (browser rate) | 100 | 0 | 9.1 | 24 ms | 54 ms | 20.5 ms |
| 10 ms (faster than the round trip) | 48 | 52 | 30.5 | 21 ms | 33 ms | 18.1 ms |

Faces found: 100 of 100 at 100 ms. Labels seen: angry 62, neutral 23, surprise 15 (this clip is angry; this run is not
an accuracy measurement). Stream test output: `scripts/out/stream-test-tick100.json` and `stream-test-tick10.json`.

### 3. The live page in headless Chrome (`scripts/browser-check.mjs`)

Chrome's fake webcam played the RAVDESS clip, so the page saw a real face. Stats read from the page:

- Status **Live**, **10.0 FPS**, **latency 44 ms**, **0 dropped**, model `face_emotion_best.pt`.
- No console errors.
- Stop turns the camera off ("Camera is on" disappears).
- Upload of a real photo shows the top label, and the upload appears in History as saved.

### 4. Behaviour checks (`scripts/ui-behaviour-check.mjs`)

| Check | Result |
|---|---|
| Save session off | 0 webcam rows added in 5 s |
| Save session on | 6 webcam rows added in 6 s |
| Express killed while live | page shows "Reconnecting…" |
| Express restarted | page back to live after **6.2 s** |
| Camera permission denied | message shown, "Go to Upload tab" offered |
| No camera at all | **not verified**: see Known limits |

### 5. Test suites

- `npm test` (Express, `node:test`): **9 of 9 pass.** Covers validation (400, 413, 415), `bad_json`, `db_unavailable` (503), `model_service_unavailable` (503 when down), and the 404 JSON.
- Model service (`pytest`): **10 of 10 pass.** Covers class order, probability sums, face-less image (no guess), invalid image (400), frames averaging, WebSocket stream, and the 3-image accuracy report (2 of 3 correct, above).

## Performance

- Detection and classification run at about 20 ms per 320 px frame on this laptop CPU, which gives 10 FPS in the browser
  at a 100 ms frame clock. That meets the 5-10 FPS target, so the frame size and detection rate were not lowered.
- `DETECT_EVERY` exists (reuse the last face box between detections) but is set to 1. Raising it was not measured.

## Known limits and what did not work

- **No-camera message not verified in a browser.** This PC has a real webcam, so headless Chrome cannot show the "no
  camera" case. The code path maps `NotFoundError` to the message, but it has not been seen running.
- **OpenCV Haar fallback does not load** in `project\.venv` (the cascade XML is missing from `cv2/data`). MediaPipe is the detector that runs.
- **Accuracy:** the model is an experimental CNN trained on acted data (see the warning at the top). Probabilities are
  not calibrated. On the 3 sample images it is 2 of 3. Its live label on the RAVDESS clip is not an accuracy result (no frame-level labels were used).
- **Several faces:** only the largest face is classified.
- **Multimodal panels:** Audio, Text and EEG are placeholders. Nothing is predicted for them.
- **MongoDB memory mode** loses its data when the server stops.
- **The old `backend/` and `frontend/` folders** (FastAPI app) are untouched and are not part of this app.

## Folder layout

```
app/
  package.json            root scripts (npm run dev, install:all, test, stream-test)
  model_service/          Python FastAPI service (analyzer.py, main.py, tests/, requirements.txt)
  server/                 Express + ws + Mongoose (src/index.js, app.js, routes/, ws/live.js, models/, services/)
  client/                 React + Vite + Tailwind (src/App.jsx, components/, hooks/, lib/)
  scripts/                run-model.mjs, stream-test.mjs, browser-check.mjs, ui-behaviour-check.mjs, make_stream_frames.py
  scripts/out/            saved test results (JSON)
```

## Audio (speech emotion)

The **Audio** tab classifies a short speech clip into the same 7 classes (angry, disgust, fear, happy, neutral, sad, surprise).
Drop or choose a clip, or press **Record 4 seconds** (MediaRecorder; the browser re-encodes the recording to a 16 kHz-class WAV so the
server does not need ffmpeg). The result shows the label ("uncertain" if the top probability is under 50%), the confidence and
7 probability bars. Only a summary (label, probabilities, clip length) is stored in history; no audio is kept.

- Model: `project/models/audio_best.pt`, a small CNN on a log-mel spectrogram, loaded through `project/code/infer_audio.py` (the same code as the offline tool).
- API: `POST /api/predict/audio` (Express, field `file`, optional field `source` = `record` or `upload`, sent before the file) proxies to the model service `POST /predict/audio`. Response: `top_label`, `confidence`, `probabilities`, `uncertain`, `duration_s`, `model`, `ms`, `saved`.
- Formats: WAV always. WebM, OGG, MP3, M4A and FLAC are decoded with **ffmpeg** when it is on PATH (`/health` reports `audio.ffmpeg`); without it the service answers 415 `ffmpeg_missing`. Limits: 10 MB (`MAX_AUDIO_MB`), 0.5 to 60 s, silent clips rejected. Raw clips are silence-trimmed and centre-cropped to 4 s, as the training clips were.
- Try it: `curl -F "file=@clip.wav;type=audio/wav" http://localhost:4000/api/predict/audio`
- **Accuracy is limited.** Trained on four acted corpora (RAVDESS, CREMA-D, TESS, EMO-DB; not MELD, ESD or SAVEE). Combined test accuracy 0.58 (0.99 on TESS, whose two speakers appear in every split; about 0.46 without it). On a corpus it never saw, macro-F1 is only 0.27 to 0.47 (chance 0.14). See `project/reports/AUDIO_MODEL_REPORT.md`.
- Not verified in a browser: the Record button and the page layout were built and compiled (`npm run build`) but not exercised with a real microphone.

## Wellbeing check-in (backend, `/api/wellbeing`)

A short chat (6 to 10 user messages) that opens with "How are you feeling today?", follows up naturally, and gently
touches the nine PHQ-9 topics (each at most once). It is a research prototype, **not a diagnostic or medical tool**.
Code: `server/src/wellbeing/`.

- **Gemini.** Set `GEMINI_API_KEY` in `server/.env` (git-ignored; never in `.env.example`). `GEMINI_MODEL` defaults to `gemini-3.5-flash`
  (a current fast "flash" model from the list-models endpoint), `GEMINI_FALLBACK_MODEL` to `gemini-3.5-flash-lite`. **Without a key the module runs in a
  labelled MOCK mode** (canned questions, keyword heuristic analysis); every response then carries `mock: true` and a `notice`.
- **Per message:** (1) a regex safety detector (English plus Hinglish) runs first, with no LLM; (2) a separate Gemini call returns a
  schema-validated analysis (mood 1-10, PHQ items 0-3 or null, themes, 7 emotions, self-harm signal, rationale), retried once, then nulls;
  (3) if either signal is *explicit*, the server returns a fixed, pre-written message with helplines (Tele-MANAS 14416, AASRA 9820466726,
  emergency 112) and `COUNSELLOR_CONTACT`, sets `tier = needs_support_now` and `needs_human_followup`, and the chat stays open;
  (4) otherwise the Gemini (or mock) reply, with a fixed one-line helpline note when the signal is only "possible".
- **API:** `GET /status`; `POST /sessions` (optional `userId` or `anonymousId`); `POST /sessions/:id/messages` `{text, source?}`;
  `POST /sessions/:id/modality` (optional face/audio summary); `POST /sessions/:id/end`; `GET /sessions/:id`.
- **Storage:** `WellbeingSession` in the existing MongoDB setup; in memory when there is no database (the module still works).
- **Tests:** `npm test` includes `server/test/wellbeing.test.mjs` (mocked Gemini; safety triggers, harmless phrases, JSON fallback, mock mode).

## Wellbeing app: setup, screens and tests

Mindful is a research prototype, **not a medical device**. Full design, privacy, limits and evaluation: `project/reports/WELLBEING_DESIGN.md`.

### Setup
1. `npm run install:all`, and install the Python packages into the project venv:
   `project\.venv\Scripts\python.exe -m pip install -r app\model_service\requirements.txt` (adds `faster-whisper`; the `base.en` speech model, about 150 MB, downloads on first use and is cached).
2. Copy `server/.env.example` to `server/.env`. Set **`GEMINI_API_KEY`** there (the file is git-ignored; never put the key in the example). Optional: `GEMINI_MODEL`, `GEMINI_FALLBACK_MODEL`, `GEMINI_TTS_MODEL`, `COUNSELLOR_CONTACT` (shown in the safety message), `WELLBEING_MAX_TURNS`, `WELLBEING_RETENTION_DAYS`.
3. **No key = MOCK mode**: canned questions and a keyword heuristic, labelled in the UI ("Demo mode") and in every API response. The safety layer, scoring, voice interview and local speech recognition work the same in mock mode.
4. `npm run dev`, then open http://localhost:5173. ffmpeg on PATH is needed for non-WAV uploads and for the speech models' audio decoding.

### Screens
Home (score ring, last 7 days, streak, trend, tools) · Check-in (chat, quick replies, optional camera emotion check, dictation) · **Voice Interview** (camera and mic, the AI asks out loud, captions, Listening / Thinking / Speaking, Done / Skip / Switch to chat / End) · Live Emotion · Upload · Audio · Text (not trained yet) · History (score line with tier bands, EMA, per-session cards, emotion mix, detection log, **Delete my data**). Light and dark mode, responsive, keyboard accessible, reduced-motion aware.

### How the score is made
Late fusion of the trained face and voice emotion models (weights from their cross-dataset reliability; a text model and EEG slot are reserved). A chat with no camera or voice has no score; turn on the emotion check or use the voice interview. Gemini's own mood guesses are shown, not scored (`config.json`: `scoreUsesQuestionnaire`). All weights and thresholds are **unvalidated defaults** in `server/src/wellbeing/config.json` and `model_reliability.json`.

### Safety
A rule-based detector (English and common Hinglish) runs before any model reply and cannot be switched off. An explicit statement returns a fixed pre-written message with helplines (Tele-MANAS 14416, AASRA 9820466726, 112) and the counsellor contact, and sets the top tier regardless of score.

### Tests
```powershell
npm test               # Node: safety, scoring, conversation, voice, privacy, client logic; Python: models, speech, prosody
npm run build          # production client build
npm run ui-shots       # screenshots of every screen, light/dark/mobile (needs a MOCK server + `vite preview` on 5190, see the script header)
npm run voice-e2e      # full voice interview in Chrome with fake devices (needs the model service on 8011 too)
node server/eval/run-eval.mjs gemini   # 35 synthetic transcripts through the real pipeline (needs a key)
```
Screenshots are in `project/reports/ui_screenshots/`.