# Wellbeing phase W3 report: UI redesign and check-in experience

Status: done (`wellbeing_state.json` phase_done = 3). Code: `app/client/src/`.

## What was built
- **Design system** (`index.css`): teal / soft violet / coral palette as CSS variables, light and dark values, glass panels, rounded cards, Tailwind v4 `@utility` classes (`card`, `btn-primary`, `chip`, `field` ...). Dark/light toggle remembered in `localStorage` (`wb_theme`), default from the OS setting.
- **Navigation**: left sidebar on large screens; top bar with a scrollable pill nav on phones. Pages: Home, Check-in, Live Emotion, Upload, Audio, Text, History. Skip-to-content link, visible focus rings, `aria-current`, labelled controls.
- **Home**: greeting, animated gauge for today's score (skeleton while loading, empty state), "Start check-in", last-7-days area chart, streak counter, smoothed trend and 7-day slope, tool cards.
- **Check-in**: consent screen on first use (not a medical device, text goes to Gemini, 90-day retention, delete button), chat bubbles, typing indicator, quick-reply chips, dictation button when the browser supports speech recognition, progress bar, optional "emotion check" (webcam probabilities every 4 s to `/signal`, nothing else leaves the page), calm support banner after a safety trigger.
- **Result**: animated gauge, tier card with a plain-language non-diagnostic message, "What shaped this" reasons, 2 or 3 suggestions, and for the top tier a prominent coral banner with helplines and the counsellor contact. No diagnosis wording (a test scans the client source and every generated text for banned words).
- **History**: score line chart with tier bands and EMA, per-session cards, stacked emotion-mix chart, detection log (previous History tab), "Delete my data" with confirmation.
- **Restyled**: Live Emotion (face box, animated bars, big emoji label), Upload, Audio, Text. Toasts for errors, skeleton loaders, empty states, `MotionConfig reducedMotion="user"` plus a CSS reduced-motion rule.
- Libraries added: framer-motion, recharts, lucide-react (client), playwright-core (dev only; uses the installed Chrome).

## Verified
`app/client/scripts/ui-shots.mjs` drives the app in Chrome through consent, a full mock-mode check-in, three sessions, History, a safety-trigger conversation and the other pages, in light and dark and at 390 px width, and fails on page errors or horizontal overflow. Result: no page errors, no overflow. 48 screenshots are in `project/reports/ui_screenshots/`. Bugs found and fixed from the screenshots: blank space under the Live page card (grid alignment), emotion chart axis showing 102 percent, repeated day labels on the 7-day chart, a noisy toast when the face-log database is offline (now inline). Production build passes (charts and motion are split into their own chunks).

## Notes and limits
- In the full-page screenshots the sticky mobile header appears mid-page; that is a capture artifact, not a layout bug.
- The Record button, dictation and the emotion check need a real camera or microphone and were not exercised by the script.
- Text tab is a "not trained yet" page; no text model is wired in yet.
