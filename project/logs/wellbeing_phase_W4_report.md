# Wellbeing phase W4 report: safety review, docs and evaluation plan

Status: done (`wellbeing_state.json` phase_done = 4). Main document: `project/reports/WELLBEING_DESIGN.md`.

## 1. Safety review of the work so far

**Where user text is stored**: `WellbeingSession` (turns, per-turn Gemini analysis, answers, final result) in MongoDB or, without a database, server memory (bounded to 500 cached sessions); deleted after 90 days (TTL index) or on request. Browser `localStorage` holds only a random id, theme, consent flags and voice settings, never chat text. Upload and live-detection summaries (`Prediction`) hold probabilities only, with no user id.

**Where user text is sent**: to Google Gemini (reply call: last 14 turns; analysis call: last 4 turns plus the message); optionally the assistant's own sentences to Gemini TTS, and the answer audio to Gemini only if the person chooses Gemini transcription. Local Whisper and the audio and face models stay on the machine.

**Where it is logged**: nowhere. The only `console.*` calls print the database status, server start lines and error names. Two handlers printed whole errors (`err.message` / the error object), which can contain request content; they now print the error name only. A test runs a full session including model failures and bad requests with a unique string and a fake key, and asserts neither appears in any console output.

**Explicit-risk path**: runs first, in code, with no LLM; verified by tests (works with every Gemini call failing; source order check that detection precedes the analysis call and the explicit branch precedes the reply call). There is no environment variable or config key that disables it (a test scans the source and `config.json`). Item 9 and the safety flag override the score and the models.

**UI wording**: a test scans all client source and every generated user-facing text (fixed messages, reasons, suggestions, tier labels) for diagnosis words ("suicid", "depress", "bipolar", "disorder", "mentally ill" and similar); none. Assistant replies that label the user with a condition are discarded.

**Issues found and fixed**
1. Error handlers could log request content -> name only.
2. `GET /session/:id` returned a transcript to anyone holding the id -> owner only (403 otherwise); `DELETE /session/:id` too.
3. No retention limit -> 90-day TTL index; memory cache bounded.
4. No rate limit on endpoints that call Gemini -> 30 per minute per user id (429).
5. Replies could in principle label the user with a condition -> output guard.
6. Consent screens did not say that text goes to Google -> added to both.
7. A Gemini key sat in plain text in `.env.example` (W1) -> moved to the git-ignored `.env`; consider rotating it.
8. Deletion: endpoint `DELETE /api/wellbeing/data` and the "Delete my data" button on History (with confirmation) were added in W3/W4; tested.

**Remaining risks (not fixed)**: no accounts, so whoever has the anonymous id can read that history; `Prediction` summaries are not removed by "Delete my data"; ffmpeg briefly writes the encoded answer to a temp directory (removed immediately); the safety word list cannot cover every phrasing or language; nothing alerts a counsellor; the helpline numbers need verifying.

## 2. Docs
`project/reports/WELLBEING_DESIGN.md` (architecture, data flow, score and tiers, thresholds marked unvalidated, privacy and consent, safety layer, limitations, evaluation plan with results) and the setup, screens and tests section of `app/README.md`.

## 3. Evaluation (see WELLBEING_DESIGN.md section 7)
35 synthetic transcripts with author-assigned expected tiers, run through the real pipeline with Gemini analysis: exact match 24 of 35 (69%), within one tier 35 of 35; all 7 explicit risk statements caught by the rule-based layer; the model over-reads distress on stress-only transcripts (11 of 13 "struggling" cases landed in the top tier); one false safety alarm on an ironic exam remark. The text-only transcripts cannot be scored by the default model-fusion mode (no text model yet).

## 4. Final run
- `npm test` (app): 146 Node tests and 26 Python tests pass.
- `npm run build` (client): passes.
- Browser runs: `ui-shots` (light, dark, 390 px) and `voice-e2e` (light, dark, safety) finish with no page errors and no horizontal overflow.
