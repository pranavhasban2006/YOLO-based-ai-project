// Structured analysis of one user message: a SEPARATE Gemini call with a JSON response schema.
// The output is validated; it is retried once on failure; after that it falls back to nulls and the chat goes on.
export const PHQ_KEYS = ['interest', 'mood', 'sleep', 'energy', 'appetite', 'self_worth', 'concentration', 'restlessness', 'self_harm'];
export const EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'];
export const SIGNALS = ['none', 'possible', 'explicit'];

const nullableInt = { type: 'INTEGER', nullable: true };

export const ANALYSIS_SCHEMA = {
  type: 'OBJECT',
  properties: {
    mood_1_10: { type: 'INTEGER', nullable: true },
    phq_items: { type: 'OBJECT', properties: Object.fromEntries(PHQ_KEYS.map((k) => [k, nullableInt])) },
    themes: { type: 'ARRAY', items: { type: 'STRING' } },
    emotions: { type: 'OBJECT', properties: Object.fromEntries(EMOTIONS.map((e) => [e, { type: 'NUMBER' }])) },
    self_harm_signal: { type: 'STRING', enum: SIGNALS },
    rationale: { type: 'STRING' },
  },
  required: ['mood_1_10', 'phq_items', 'themes', 'emotions', 'self_harm_signal', 'rationale'],
};

export const ANALYSIS_SYSTEM = `You analyse ONE message from a college student in a wellbeing check-in. You are not diagnosing anything.
Return JSON only, with these fields:
- mood_1_10: integer 1 (very low) to 10 (very good) for the mood expressed in this message, or null if it says nothing about mood.
- phq_items: for each of interest, mood, sleep, energy, appetite, self_worth, concentration, restlessness, self_harm: an integer 0-3 (0 = not at all, 1 = several days, 2 = more than half the days, 3 = nearly every day) ONLY if the user actually addressed that topic in this message; otherwise null. Never guess.
- themes: up to 5 short lowercase topic words (for example "exams", "sleep", "friends", "family", "homesick", "money").
- emotions: probabilities for angry, disgust, fear, happy, neutral, sad, surprise that sum to 1, describing the emotional tone of the message.
- self_harm_signal: "explicit" if the user states thoughts of suicide or self-harm or wanting to die; "possible" if there are hints (hopelessness, being a burden, not wanting to be here); otherwise "none". Everyday exaggeration ("this exam is killing me", "I'm dead tired") is "none".
- rationale: one short sentence explaining the main signals.`;

export function nullAnalysis(reason = 'analysis unavailable') {
  return {
    mood_1_10: null,
    phq_items: Object.fromEntries(PHQ_KEYS.map((k) => [k, null])),
    themes: [],
    emotions: null,
    self_harm_signal: 'none',
    rationale: reason,
    fallback: true,
  };
}

// Returns a clean analysis object, or null when the shape is wrong.
export function validateAnalysis(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
  const out = { fallback: false };

  if (raw.mood_1_10 === null || raw.mood_1_10 === undefined) out.mood_1_10 = null;
  else if (typeof raw.mood_1_10 === 'number' && Number.isFinite(raw.mood_1_10) && raw.mood_1_10 >= 1 && raw.mood_1_10 <= 10) {
    out.mood_1_10 = Math.round(raw.mood_1_10);
  } else return null;

  const items = raw.phq_items;
  if (!items || typeof items !== 'object') return null;
  out.phq_items = {};
  for (const k of PHQ_KEYS) {
    const v = items[k];
    if (v === null || v === undefined) out.phq_items[k] = null;
    else if (typeof v === 'number' && Number.isInteger(v) && v >= 0 && v <= 3) out.phq_items[k] = v;
    else return null;
  }

  if (!Array.isArray(raw.themes)) return null;
  out.themes = raw.themes.filter((t) => typeof t === 'string' && t.trim()).map((t) => t.trim().toLowerCase().slice(0, 40)).slice(0, 5);

  const em = raw.emotions;
  if (!em || typeof em !== 'object') return null;
  const vals = EMOTIONS.map((e) => em[e]);
  if (!vals.every((v) => typeof v === 'number' && Number.isFinite(v) && v >= 0)) return null;
  const sum = vals.reduce((a, b) => a + b, 0);
  if (sum <= 0) return null;
  out.emotions = Object.fromEntries(EMOTIONS.map((e, i) => [e, vals[i] / sum]));

  if (!SIGNALS.includes(raw.self_harm_signal)) return null;
  out.self_harm_signal = raw.self_harm_signal;
  out.rationale = typeof raw.rationale === 'string' ? raw.rationale.trim().slice(0, 300) : '';
  return out;
}

function parseJson(text) {
  const cleaned = text.replace(/^```(?:json)?\s*/i, '').replace(/```\s*$/i, '').trim();
  try {
    return JSON.parse(cleaned);
  } catch {
    return null;
  }
}

// history: previous turns [{role:'user'|'assistant', text}] used only as context.
export async function analyseMessage({ gemini, history = [], text }) {
  const context = history
    .slice(-4)
    .map((t) => `${t.role === 'user' ? 'Student' : 'Assistant'}: ${t.text}`)
    .join('\n');
  const prompt = `${context ? `Earlier in the conversation:\n${context}\n\n` : ''}Analyse this new message from the student:\n"""${text}"""`;
  let lastError = 'unknown';
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const { text: out } = await gemini.generate({
        system: ANALYSIS_SYSTEM,
        contents: [{ role: 'user', text: prompt }],
        schema: ANALYSIS_SCHEMA,
        temperature: 0.1,
        maxOutputTokens: 600,
      });
      const valid = validateAnalysis(parseJson(out));
      if (valid) return { analysis: valid, ok: true };
      lastError = 'invalid JSON from the model';
    } catch (err) {
      lastError = err.message || 'request failed';
    }
  }
  return { analysis: nullAnalysis(lastError), ok: false, error: lastError };
}

// Keyword heuristic used only in MOCK mode, so the rest of the app has something to show. Not a model.
const NEG_WORDS = /\b(sad|tired|exhausted|stress(?:ed)?|anxious|worried|worry|bad|awful|terrible|lonely|overwhelmed|can'?t sleep|hopeless|low|down|angry|upset)\b/gi;
const POS_WORDS = /\b(good|great|happy|fine|okay|ok|well|excited|relaxed|calm|better|energetic|fun)\b/gi;
export function mockAnalysis(text) {
  const neg = (text.match(NEG_WORDS) || []).length;
  const pos = (text.match(POS_WORDS) || []).length;
  const mood = Math.max(1, Math.min(10, 7 + pos * 0.7 - neg * 1.5));
  const a = nullAnalysis('MOCK mode: keyword heuristic, not a model');
  a.fallback = false;
  a.mock = true;
  a.mood_1_10 = Math.round(mood);
  a.emotions = neg > pos
    ? { angry: 0.05, disgust: 0.02, fear: 0.18, happy: 0.05, neutral: 0.2, sad: 0.48, surprise: 0.02 }
    : { angry: 0.02, disgust: 0.01, fear: 0.04, happy: pos > 0 ? 0.55 : 0.3, neutral: pos > 0 ? 0.33 : 0.58, sad: 0.04, surprise: 0.02 };
  return a;
}
