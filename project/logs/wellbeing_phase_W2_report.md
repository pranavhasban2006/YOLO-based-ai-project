# Wellbeing phase W2 report: scoring, tiers and trends

Status: done (`wellbeing_state.json` phase_done = 2). Code: `app/server/src/wellbeing/scoring.js` (pure functions) and `config.json`.

## Thresholds and weights chosen (UNVALIDATED DEFAULTS)
These numbers were set by the developers. They were not fitted to, or checked against, any clinical or real-user data, and the tiers are not a screening result. They live in `config.json` so they can be changed in one place.

| Setting | Value |
|---|---|
| Weights: PHQ / mood / emotion | 0.45 / 0.30 / 0.25 (renormalised over the components that exist) |
| PHQ component | mean of the highest reported 0-3 value per answered item, scaled so 100 = no symptoms; needs 4 answered items, otherwise `phq.confidence = low` |
| Mood component | mean mood 1-10 mapped to 0-100 |
| Emotion component | per source (text analysis, face samples, audio samples, text-model samples): 50 + 50 x (happy - (angry + disgust + fear + sad)); neutral and surprise are 50; sources averaged equally; missing sources ignored |
| Tiers | score >= 65 `doing_okay`; 40 to under 65 `struggling`; under 40 `needs_support_now` |
| Hard overrides to `needs_support_now` | explicit safety flag, or PHQ item 9 (self-harm item) >= 2, whatever the score |
| Soft flag | a "possible" safety signal lifts `doing_okay` to `struggling` |
| Trend | EMA over the user's last sessions, alpha 0.5; tier is taken from the smoothed score; 7-day least-squares slope in points per day |
| Low-confidence cap | confidence is low when fewer than 4 PHQ items were answered and fewer than 4 messages were sent; a low-confidence session can move the tier at most one step down from the user's usual tier (the tier of the EMA of earlier sessions, or `doing_okay` for a first session). Overrides are not capped. |

## Results always carry reasons
Each result has `reasons` (kind or component, effect raises/lowers/neutral, plain-language text) naming what drove the tier, plus `components` with weights, `trend` (EMA, series, slope), `suggestions` (2 or 3), and `unvalidated: true`.

## Endpoints (under `/api/wellbeing`)
`POST /session`, `POST /session/:id/message`, `POST /session/:id/signal` (face, audio or text samples), `POST /session/:id/finish` (score, tier, reasons, suggestions, support block with helplines for the top tier), `GET /history` (sessions, scores, EMA, slope, streak; per user via the `X-Wellbeing-User` header). The W1 plural routes still work. Tier names changed from W1 (`doing_well`, `some_stress`, `needs_attention`) to `doing_okay`, `struggling`, `needs_support_now`.

## Tests
`test/scoring.test.mjs` (30 tests): components, all-positive, mixed, clearly low, missing PHQ items, missing modalities, explicit flag overriding a great score, item 9, override not removable through config, thresholds, smoothing, low-confidence cap, EMA, slope, streak, history. Plus a route test in `wellbeing.test.mjs`. All Node tests pass (108 before the route test was added, 109 with it).

## Caveats
- A tier from one short chat is a coarse signal; with 8 messages not all PHQ topics are covered, so `phq.confidence` is often low.
- The Gemini-derived items can over-read distress (seen in the W1 live check). The tier rule that takes the highest value per item makes this worse, not better.
- The synthetic tests check that the arithmetic and rules behave as designed. They say nothing about whether the thresholds are right.
