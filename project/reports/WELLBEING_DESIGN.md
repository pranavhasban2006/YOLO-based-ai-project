# Wellbeing check-in: design, safety review and evaluation plan

Status: research prototype. **Not a medical device.** It is a screening aid that shows a rough, model-based signal from a short chat or voice interview. It cannot diagnose anything, and every number below is an unvalidated default.

## 1. Architecture

```
 Browser (React + Vite, :5173)                              Local machine / server
 ┌────────────────────────────────────────────┐
 │ Home · Check-in (chat) · Voice interview   │   /api/*          ┌───────────────────────────┐
 │ Live · Upload · Audio · Text · History     │ ───────────────►  │ Express (Node, :4000)     │
 │                                            │                   │  /api/wellbeing/*         │
 │ camera ─► face frames ─► (WebSocket) ──────┼───────────────►   │   safety.js   (rules)     │
 │ mic ─► VAD ─► answer audio (webm) ─────────┼───────────────►   │   service.js  (flow)      │
 │ TTS (browser voice, or Gemini voice)       │                   │   scoring.js  (fusion)    │
 └────────────────────────────────────────────┘                   │   gemini.js   (client)    │
                                                                  │   store.js ─► MongoDB     │
                                                                  └───────┬───────────┬───────┘
                                                      HTTP (local)        │           │ HTTPS
                                                                          ▼           ▼
                                                  ┌──────────────────────────┐   ┌────────────────────┐
                                                  │ model_service (FastAPI)  │   │ Google Gemini API  │
                                                  │  face CNN   /predict/*   │   │  chat replies      │
                                                  │  audio CNN  /predict/audio│  │  per-message JSON  │
                                                  │  faster-whisper /transcribe│ │  optional TTS / STT│
                                                  │  prosody (librosa)       │   └────────────────────┘
                                                  │  /analyse/answer         │
                                                  └──────────────────────────┘
```

Code: `app/server/src/wellbeing/` (server), `app/model_service/` (models, speech), `app/client/src/` (UI).

## 2. Data flow

**Chat check-in** (6 to 10 user messages)
1. `POST /session` opens a session; the first question is the fixed "How are you feeling today?".
2. For each user message the server runs, in this order: (a) the rule-based safety detector (no LLM); (b) a separate Gemini call that returns a schema-checked JSON analysis; (c) the combination of both safety signals; (d) if neither is *explicit*, the Gemini reply with a short "director" note that weaves in the PHQ-9 topics one at a time, each at most once.
3. Optional emotion check: the browser reads the webcam on the local model service and sends only 7-class probabilities every 4 s (`/signal`).
4. `POST /finish` scores the session and returns score, tier, reasons, suggestions and a "how it was made" breakdown.

**Voice interview**
1. The AI asks out loud (browser speech synthesis, or optional Gemini voice). The first question, "How has your day been?", gives the person's own baseline.
2. The mic opens only after the AI has finished. Energy-based voice-activity detection ends the answer after about 1.5 s of silence (60 s maximum; "Done answering" overrides). Echo cancellation and noise suppression are requested.
3. During the answer, face probabilities are sampled about twice a second and aggregated in the browser (mean, dominant emotion, variability). The audio clip is sent to the local model service, which returns the transcript (faster-whisper `base.en`, CPU), prosody features and the voice-emotion probabilities, then discards the audio.
4. The transcript goes through the same safety layer and Gemini flow as the chat, **before** the next question. Replies are spoken sentence by sentence.

## 3. How the score and tier are computed

### 3.1 Default: late fusion of the trained models
- **Per-modality value** from a 7-class distribution `p`: `valence = p_happy + 0.25·p_surprise − (p_sad + p_fear + p_angry + p_disgust)`; neutral counts 0; value = `(valence + 1) × 50`. All happy = 100, all neutral = 50, all sad/fear/angry/disgust = 0. This is a **heuristic mapping from emotion to wellbeing, not a clinical measure.** Coefficients are in `config.json`.
- **Aggregation**: mean of probabilities within an answer first, then across answers.
- **Weights = measured reliability.** `model_reliability.json` holds each model's cross-dataset (leave-one-dataset-out, E3) mean macro-F1 and where the number came from:

| Model | E3 mean macro-F1 | Source | Share of weight (all present) |
|---|---|---|---|
| Face CNN | 0.342 | `FACE_MODEL_REPORT.md`, section 3.2 | 48.7% |
| Voice (audio) CNN | 0.360 | `AUDIO_MODEL_REPORT.md`, section 4.2 | 51.3% |
| Text emotion model | none yet | no model exists | skipped |
| EEG model | none yet | slot reserved | skipped |

  Weight of a present modality = its reliability divided by the sum for the modalities present; missing ones are dropped and the rest renormalised.
- **Confidence** = the sum of the base weights of the modalities actually present (face only 0.49, voice only 0.51, both 1.0).
- **Gemini's own mood and PHQ-style guesses are shown but not scored** (`scoreUsesQuestionnaire = false`). **Prosody** (speech rate, pauses, latency, pitch and energy variability, relative to the person's first answer) is display-only (weight 0).
- A chat with no camera and no voice has no model signal, so it gets **no score**; the result says how to get one.

### 3.2 Tiers (unvalidated defaults)
| Tier | Model-fusion score | Questionnaire-mode score |
|---|---|---|
| `doing_okay` | 45 or more | 65 or more |
| `struggling` | 30 to under 45 | 40 to under 65 |
| `needs_support_now` | under 30 | under 40 |

The model-fusion cut-offs are lower because neutral maps to 50. **These thresholds, the valence coefficients and the weights are developer choices that have not been checked against clinical data.**

Rules on top of the score:
- **Trend**: the tier is taken from an exponential moving average over the user's last sessions (alpha 0.5), so one bad day moves it only partly. A 7-day least-squares slope is reported.
- **Low confidence (under 0.4)**: the result is labelled "low confidence" and the model-based score alone cannot give `needs_support_now` (shown as `struggling`). A low-confidence session also moves the tier at most one step from the user's usual tier.
- **Hard overrides, independent of score, models and config**: the rule-based safety layer flagging an explicit statement, or PHQ-style item 9 at 2 or more, gives `needs_support_now`. A softer "possible" flag lifts `doing_okay` to `struggling`.
- Every result carries `reasons` (what drove it), `modalities` (value, weight, reliability per model), `confidence`, `suggestions` (2 or 3), and `unvalidated: true`.

### 3.3 Optional questionnaire mode (`scoreUsesQuestionnaire: true`)
PHQ-style items 0.35, mood 0.25, text emotion 0.10, face 0.15, voice 0.15 (voice includes a capped prosody penalty), renormalised over what exists, with the 65 / 40 cut-offs. Used for the transcript evaluation in section 7 because it is the only mode that can read text.

## 4. Safety layer
- **Runs first, in code, with no LLM**: `detectSafety()` on every user message and every transcript, before any model reply is produced. English and common Hinglish spellings (for example "jeena nahi", "mar jana chahta", "khud ko khatam"), strong hopelessness phrases, negation handling ("I'm not suicidal" is only "possible"), and idiom guards ("this exam is killing me" is not triggered).
- **Combined** with the structured `self_harm_signal` from Gemini (highest of the two). Either one saying *explicit* is enough.
- **Explicit result**: a fixed, pre-written message (never LLM text) with Tele-MANAS 14416, AASRA 9820466726, emergency 112 and the configured counsellor contact; tier `needs_support_now`; `needs_human_followup` set. In the voice interview the same text is shown and spoken, the banner offers "Continue talking" or "End session", and no further screening questions are asked.
- **Cannot be switched off**: there is no config key or environment variable for it, and tests assert this. It works with Gemini fully down. Face, voice and tone never decide it; they only feed the score.
- **Output guards**: model replies that discuss methods, or label the user with a condition, are discarded and replaced by a canned line.
- The detector deliberately errs towards triggering ("I'll kill myself if I fail the exam" fires). It can still miss indirect or non-English phrasing. The helpline numbers were supplied by the project owner and must be verified before real use. Nothing notifies a counsellor automatically: `needs_human_followup` is a flag in the database.

## 5. Privacy and consent
Where user data goes:

| Data | Where | Kept? |
|---|---|---|
| Chat and interview transcript, per-turn analysis, per-answer aggregates, scores | `WellbeingSession` in MongoDB (or server memory when no database) | Yes, 90 days (TTL), deletable any time |
| Chat text, last 14 turns | Sent to Google Gemini for the reply; last 4 turns plus the message for the analysis | By Google, under its API terms |
| Assistant text for the optional natural voice | Sent to Gemini TTS | By Google |
| Answer audio | Local model service; **only** sent to Google if the user picks "Gemini transcription" | Local audio is decoded in memory and discarded; ffmpeg briefly writes a temp file that is deleted at once |
| Face frames | Local model service (WebSocket) for the live preview | Not stored. Only probabilities are sent to the session |
| Raw video, raw audio | Never stored | n/a |
| Browser storage | random anonymous id, theme, consent flags, voice settings | In the browser only; no chat text |
| Upload and live-detection summaries (`Prediction`) | MongoDB | Probabilities only, no user id; **not** removed by "Delete my data" |

- **Consent screens** (first use of the chat and of the voice interview) say: not a medical device; text goes to Gemini; what is analysed; what is stored; raw video and audio are not stored; 90-day retention; how to delete.
- **Delete my data** (History page) calls `DELETE /api/wellbeing/data`, removing every session stored for the browser's anonymous id. A single session can be deleted with `DELETE /session/:id`. A transcript is only returned to its owner (matching id).
- **Logging**: no API key and no user text in logs (error handlers log only the error name); a test checks this.
- The anonymous id is not a login. Anyone who has it can read that history; there are no accounts yet.

## 6. Limitations
- **Not a medical device and not a diagnosis.** It is a screening aid; a low or high tier says nothing about a condition.
- **The emotion models generalise weakly.** The face CNN's cross-dataset macro-F1 is 0.34 (and below chance on in-the-wild faces), the voice model's 0.36 (chance 0.14), both trained on acted or other-domain data. Facial expression and tone of voice are not the same as how a person feels. The confidence value matters: with one model the score is a rough hint.
- **The emotion-to-wellbeing mapping and every threshold are heuristics** chosen by the developers. Prosody indicators are heuristic.
- **The language model can misjudge**: it can over-read distress (see the evaluation), invent PHQ items, or phrase things badly. Its replies are filtered but not verified.
- **Bias across languages and cultures**: the voice and face models were trained on a few English (and one German) acted corpora, with mostly North American, European and Japanese faces. Speech recognition is `base.en` and degrades with accents and code-mixed speech (Hinglish); the safety word list covers common Hinglish only. Display of emotion differs by culture and by person.
- Browser text-to-speech quality varies; CPU transcription adds a few seconds per answer; the interview needs a quiet room.
- Short sessions (6 to 10 messages) cannot cover all nine PHQ-style topics.
- No accounts, no clinician review, no automatic escalation.

## 7. Evaluation plan and results so far

### (a) Unit and scenario tests (automated)
Run with `npm test` in `app/`: 146 Node tests plus 26 Python tests, all passing at the time of writing.
- Safety: 21 explicit phrases fire, 15 harmless phrases (including "this exam is killing me") do not, negation, Hinglish, works with Gemini down, no switch to disable it.
- Scoring and fusion: valence mapping, weight renormalisation with missing modalities, confidence rule, low-confidence cap, end-to-end fixed model outputs to score and tier, overrides, EMA, slope, streak, history, prosody deltas.
- Conversation: mocked Gemini, JSON fallback, MOCK mode, topic weaving, output guards.
- Voice: prosody extraction on synthetic audio (fast vs slow speech, long pauses, pitch, latency), end-of-answer detection, face aggregation, voice-answer route.
- Privacy: no secrets or user text in logs, deletion, ownership, rate limit, retention index, no diagnosis wording in anything shown to the user.
- Browser: `npm run ui-shots` and `npm run voice-e2e` drive Chrome through a chat check-in and a full voice interview (fake camera and a spoken sample as the microphone) in light and dark mode, and a safety trigger.

### (b) 35 written synthetic transcripts with expected tiers
`app/server/eval/transcripts.mjs`: 10 expected `doing_okay`, 13 `struggling` (10 plus 3 edge cases), 12 `needs_support_now` (7 with an explicit statement, 5 severe without risk words). **The expected tiers were assigned by the author, not by a clinician**, so agreement measures consistency with the author's intent only. The transcripts are text only, so they are scored in questionnaire mode with Gemini (`gemini-3.5-flash`) analysing each message (`node eval/run-eval.mjs gemini`); the model-fusion score has no input for text-only data.

| Expected \ got | doing_okay | struggling | needs_support_now |
|---|---|---|---|
| doing_okay (10) | **10** | 0 | 0 |
| struggling (13) | 0 | **2** | 11 |
| needs_support_now (12) | 0 | 0 | **12** |

- Exact match **24 of 35 (69%)**; within one tier 35 of 35.
- **All 7 explicit statements were caught by the rule-based layer** (first explicit turn 1 to 4), including Hinglish. The 5 severe transcripts without risk words reached the top tier through the score (no safety flag).
- **Where it does not match**: the model over-reads distress. 11 of 13 "struggling" transcripts landed in the top tier (E01 through the safety flag, see below; the other 10 through the score), including the negated-risk edge case ("I am not suicidal") and the friend-in-distress case, because Gemini's per-message mood and emotion read sad or worried for stress and tiredness. In the default model-fusion mode this text analysis is not scored, which avoids this particular error but also leaves chat sessions without a score unless a camera or voice is used.
- **False safety alarm**: edge case E01 ("I could kill myself if I fail this exam lol") triggered the explicit layer. This is the intended trade-off (a false alarm costs one supportive message), but it shows the detector cannot read irony.
- MOCK mode (keyword heuristic, no model) matched 26 of 35, but its heuristic was adjusted once after seeing a first run on these same transcripts, so treat that figure as a demo, not evidence. The Gemini run was not tuned on these transcripts.
- Not covered: the model-fusion score on real faces and voices, which needs recorded data (see c).

### (c) Later, only with ethics approval
- Validate a Reddit-depression text classifier (text emotion or risk model) and wire it into the fusion as the `text` modality once its cross-dataset reliability is measured.
- Compare the score and tiers with PHQ-9 (and a clinician's judgement) on **real participants**, with informed consent, an ethics committee's approval, a data-protection plan, and a clear escalation path for anyone who shows risk. Fit or recalibrate the valence mapping, weights and thresholds on that data, and report sensitivity, specificity and calibration, per language and group.
- A second seed and confidence intervals for the underlying emotion models; speech recognition accuracy on the target population's accents.

## 8. How to reproduce
```powershell
npm test                       # app/: Node tests + model service pytest
npm run build                  # production client build
node eval/run-eval.mjs gemini  # app/server: transcript evaluation (needs GEMINI_API_KEY)
node eval/run-eval.mjs mock    # same, MOCK mode
npm run ui-shots               # app/client: screenshots (MOCK server + `vite preview` on 5190)
npm run voice-e2e              # app/client: full voice interview with fake devices
```
