# Wellbeing phase W3C report: score from our trained models

Status: done (`wellbeing_state.json` phase_done = 3). Code: `app/server/src/wellbeing/scoring.js`, `config.json`, `model_reliability.json`; UI `ResultDetails.jsx`.

## What changed
- **Default score = late fusion of the trained emotion models only.** Gemini's mood and PHQ-style guesses are still computed and shown ("noted from the conversation") but not scored; `scoreUsesQuestionnaire` is `false` in `config.json`. Prosody is display-only (weight 0, optional switch `fusion.use_prosody`).
- **Models used**: face CNN (per-answer mean probabilities, or webcam samples in a chat), audio emotion CNN (per answer or an attached clip), text model slot (`/signal` with modality `text`), EEG slot (ignored). There is **no trained text model and no EEG model** in this project yet (no `infer_text.py`, no `TEXT_MODEL_REPORT.md`), so those slots are empty; the code skips them and the weights renormalise.
- **Valence mapping** (config): `p_happy + 0.25 p_surprise - (p_sad + p_fear + p_angry + p_disgust)`, neutral 0, mapped from [-1, 1] to [0, 100]. A heuristic, not a clinical measure.
- **Reliability weights** in `model_reliability.json`: face 0.342 (FACE_MODEL_REPORT.md section 3.2, E3 mean macro-F1), audio 0.360 (AUDIO_MODEL_REPORT.md section 4.2), text null, EEG null. weight = reliability / sum over present modalities. With both: face 48.7%, voice 51.3%.
- **Confidence** = sum of the base weights of the modalities present (face only 0.49, voice only 0.51, both 1.0). Under 0.4 the result is labelled "low confidence" and cannot reach the top-concern tier without the rule-based safety layer. Per-modality values, weights, reliabilities and the confidence are returned for every session and shown in the result screen panel "How your score was made".
- **Temporal handling**: mean within an answer, then across answers; the Phase W2 EMA across days and the 7-day slope are unchanged.
- **Tiers**: three tiers kept. Because neutral maps to 50, the questionnaire cut-offs (65 / 40) would call every neutral-looking person "struggling", so the model-fusion score uses 45 / 30 (`tiers_models`). This is a deviation from "keep the thresholds" and is an unvalidated default.
- **Safety layer** unchanged: rule-based, independent of the models, cannot be disabled.

## Consequences to know about
- A text-only chat with no camera and no voice has **no score** (nothing for the models to read). The chat UI says the emotion check is needed for a score; the voice interview gives both face and voice.
- The earlier text-based evaluation (35 synthetic transcripts) can only be run in questionnaire mode; see WELLBEING_DESIGN.md section 7.

## Tests
`test/scoring.test.mjs` (34): valence mapping (all happy high, sad / fear low, neutral 50, surprise 62.5, mixtures), weight renormalisation with missing modalities, text and EEG ignored, confidence rule, low-confidence top-tier rule with a reliability override, an end-to-end test with fixed model outputs checking score, weights, confidence and tier, Gemini mood and PHQ not changing the score, overrides unaffected by config, questionnaire mode, prosody deltas and penalty cap, EMA, slope, history.

## Limits
The models were trained on acted or other-domain data and generalise weakly (E3 macro-F1 0.34 and 0.36, chance about 0.14), so the score is a screening indicator and the confidence value matters. See WELLBEING_DESIGN.md section 6.
