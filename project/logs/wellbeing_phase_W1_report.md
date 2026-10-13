# Wellbeing phase W1 report: backend, Gemini check-in and safety layer

Status: done (`wellbeing_state.json` phase_done = 1). Backend only; there is no client UI yet.

## What was built (`app/server/src/wellbeing/`)
| File | Role |
|---|---|
| `safety.js` | Regex detector (English plus Hinglish), level combination, the fixed safety message and soft support line, output guard |
| `gemini.js` | REST client: key in a header, retry once, fallback model |
| `analysis.js` | JSON schema, validation, retry once, null fallback, mock heuristic |
| `conversation.js` | System prompt, PHQ-9 topic weaving, canned mock questions, final scoring |
| `service.js` | Per-message flow, session completion, modality samples |
| `WellbeingSession.js`, `store.js` | Mongoose model and memory-plus-database store |
| `routes.js` | `/api/wellbeing/*` router (status, sessions, messages, modality, end) |

Config: `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_FALLBACK_MODEL`, `COUNSELLOR_CONTACT`, `WELLBEING_MAX_TURNS` in `app/server/.env` (examples in `.env.example`, `.env` is git-ignored).

## Flow per user message
1. Regex safety detector (no LLM).
2. Separate Gemini call with a response schema: mood 1-10, PHQ items 0-3 or null, themes, 7 emotions (normalised to 1), self-harm signal, rationale. Invalid output is retried once, then replaced by nulls.
3. Highest of the regex level and the structured signal. `explicit` returns the fixed message (helplines 14416, 9820466726, 112, `COUNSELLOR_CONTACT`), sets tier `needs_support_now` and `needs_human_followup`, and generates no LLM reply. The chat stays open.
4. Otherwise the Gemini reply (or the canned mock reply), with the director note for the topic to weave in, and a fixed helpline line when the level is only `possible`.

## Tests
`app/server/test/wellbeing.test.mjs`, 67 tests, Gemini mocked: 21 explicit phrases trigger (English, Hinglish, typos); 15 harmless messages do not (including "this exam is killing me", which is level `none`); negation gives `possible`; JSON validation, retry and null fallback; the safety layer fires with Gemini fully down; a structured-only `explicit` fires; behaviour after a trigger; the unsafe-reply guard; topic weaving (at most once, self-harm only when distressed); completion and final summary; mock mode; the Gemini client (header key, fallback model); HTTP routes. Whole suite: 79 Node tests and 14 pytest pass.

## Live check with the real key
`GET /status` reports `gemini-2.5-flash`. A 4-message conversation worked as designed (see DECISIONS.md). The structured analysis was sometimes generous (first message gave restlessness 2 and self-worth 1 although the user only mentioned exam stress), so the PHQ items from one message are noisy; the final score takes the highest value per item, which leans towards over-reporting distress. This is a tuning issue for a later phase, not changed here.

## Limits and risks
- A regex and a model are not a clinical risk assessment. The detector can miss indirect or non-English phrasing and can fire on dark jokes. Negated or academic mentions are handled only by simple rules.
- The helpline numbers were supplied by the project owner and should be verified before real use.
- `needs_human_followup` is only a flag in the database; nothing notifies a counsellor yet.
- Conversation text is stored with the session. Privacy, consent and retention rules have not been decided.
- The Gemini API key was found in plain text in `.env.example`; moved to `.env`, but consider rotating it.
- MOCK mode is for development only.
- No client UI yet; face and audio samples are stored but not used in the score.
