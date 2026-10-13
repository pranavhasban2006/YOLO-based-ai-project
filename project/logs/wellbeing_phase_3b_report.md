# Wellbeing phase W3B report: live voice interview

Status: done (`wellbeing_state.json` phase_done = 3). Code: `app/client/src/components/InterviewRoom.jsx`, `lib/vad.js`, `lib/speech.js`, `lib/interview.js`; `app/model_service/speech_service.py`, `prosody.py`; `app/server/src/wellbeing/routes.js` (voice-answer, skip, tts).

## What was built
- **Interview room** (Voice Interview tab): camera preview with face box and an emotion-coloured mood ring, an AI orb (pulses while speaking, follows the mic level while listening, spins while thinking), live captions (AI sentence, then the person's transcript), status chips Listening / Thinking / Speaking, mic level meter, buttons Done answering, Skip question, Switch to chat, End session, a voice settings panel. A consent screen explains what is analysed and stored. If the microphone is denied or missing it falls back to chat; if only the camera is missing it continues voice-only.
- **Turn-taking**: AI speaks (sentence by sentence) -> mic opens only afterwards -> energy-based VAD ends the answer after 1.5 s of silence (60 s cap, 12 s no-speech timeout, Done overrides) -> local transcription -> safety check -> Gemini reply -> speak. `echoCancellation`, `noiseSuppression` and `autoGainControl` are requested. The mic is not analysed while the AI speaks.
- **Speech out**: browser SpeechSynthesis by default (calm English voice preferred, rate 0.92, helpline numbers read digit by digit). Optional "natural voice" through Gemini TTS (`gemini-3.8-flash-tts`, tested) with automatic fall-back to the browser voice.
- **Speech in**: each answer is recorded (MediaRecorder, webm/opus) and sent to the model service, which runs **faster-whisper `base.en`** (CPU, int8; model downloaded once to the Hugging Face cache) and returns text, word timings and duration. About 3 s for a 10 s clip on this PC. Settings also offer the browser Web Speech API (when present; Brave disables it) and Gemini transcription. Local Whisper needs no internet.
- **Emotion while answering**: face probabilities sampled about twice a second during listening only, aggregated in the browser (mean, dominant, variability, count) and sent as numbers only. Voice emotion from the audio model per answer. Prosody per answer (`/analyse/prosody`, librosa): speech rate, pause share and length, time to start speaking, pitch mean and variability (pyin), energy mean and variability, duration. The first answer is the personal baseline; later answers are reported as relative changes against it, and labelled heuristic.
- **Safety**: the rule-based detector runs on every transcript before the next question. If it fires, the fixed message is shown as text and spoken, the support banner offers Continue talking or End session, and probing stops. Tone and face never decide this.
- **Result**: score gauge and tier card (W3), a chart of face and voice emotion per answer, prosody table with deltas, transcript with per-answer emotion chips, "How your score was made" panel, suggestions. No diagnosis wording.
- **Scoring**: superseded by Phase W3C (model-driven fusion); the questionnaire-style weights requested here (PHQ 0.35, mood 0.25, text emotion 0.10, face 0.15, voice 0.15, with a capped prosody penalty on voice) are implemented behind `scoreUsesQuestionnaire`.

## Privacy
Raw video never leaves the browser (only probabilities are sent). Raw audio goes only to the local model service, is decoded in memory and discarded (ffmpeg briefly writes a temp file that is deleted at once), unless the person picks Gemini transcription, which sends the clip to Google. Stored: transcript, per-answer aggregates, scores. "Delete my data" removes them.

## Tests
- `model_service/tests/test_speech.py` (12): prosody on synthetic audio (latency, pauses, short gaps, fast vs slow speech, pitch mean and variability, energy, silence, resampling), prosody / transcribe / answer endpoints, real speech sample transcription.
- `test/client_logic.test.mjs` (15): VAD end-of-answer logic (1.5 s rule, short pause inside an answer, no-speech timeout, cough, 60 s cap, noisy room, stays ended), face aggregation, sentence splitting for speech.
- `test/voice.test.mjs`: voice session and answer records, skipped question, safety on a spoken transcript, voice-answer route with a mocked model service, no raw audio or frames in the stored session.
- Browser end to end: `client/scripts/voice-e2e.mjs`, Chrome with `--use-fake-ui-for-media-stream`, `--use-fake-device-for-media-stream`, a spoken WAV (Windows speech synthesis) as the microphone and a still face as the camera, MOCK server, real Whisper. Three full interviews (light, dark, safety trigger): transcripts correct, replies spoken, result page shows all panels, no page errors, no horizontal overflow. Screenshots: `project/reports/ui_screenshots/voice_*.png`.
- Bugs found by the browser run and fixed: the safety banner buttons were ignored while the message was still being spoken (the click is now remembered and cancels speech); long bullet-style safety text was read badly (now split into sentences, numbers spoken digit by digit).

## Known limits
- Browser TTS voice quality depends on the installed voices; headless Chrome has none, so the e2e run used timed fall-backs rather than real audio output. Natural voice needs a Gemini key.
- Speech recognition accuracy drops with accents, noise and Hinglish; `base.en` is English only. In the e2e run an answer that starts mid-sentence (looping fake microphone) gave partial transcripts, which is expected.
- CPU transcription adds a few seconds per answer (about 3 s for 10 s of audio here; longer answers take longer).
- The reply is not streamed from Gemini: the whole reply (at most a few short sentences) is requested, then spoken sentence by sentence. This was simpler and the replies are short; true streaming is a possible later improvement.
- The Web Speech option, the natural voice in a real browser, and a real microphone and camera were not exercised by the automated run.
- Prosody features are heuristic indicators; there is no validation that they track wellbeing.
