// Class order is fixed by project/models/classes.json. Keep it identical everywhere.
export const CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'];

export const COLORS = {
  angry: '#ef4444',
  disgust: '#22c55e',
  fear: '#a855f7',
  happy: '#facc15',
  neutral: '#94a3b8',
  sad: '#3b82f6',
  surprise: '#f97316',
};

export const EMOJI = {
  angry: '😠',
  disgust: '🤢',
  fear: '😨',
  happy: '😊',
  neutral: '😐',
  sad: '😢',
  surprise: '😮',
};

// Below this top probability the label is shown as "uncertain".
export const LOW_CONFIDENCE = 0.5;

// Smoothing: EMA with alpha = 2 / (N + 1). N = 5 gives the "about the last 5 frames" window.
export const EMA_ALPHA = 2 / (5 + 1);

export const SERIES_WINDOW_MS = 30_000;

export function topOf(probs) {
  let best = CLASSES[0];
  for (const c of CLASSES) if (probs[c] > probs[best]) best = c;
  return { label: best, confidence: probs[best] };
}

export function emaUpdate(prev, probs, alpha = EMA_ALPHA) {
  if (!prev) return Object.fromEntries(CLASSES.map((c) => [c, probs[c]]));
  return Object.fromEntries(CLASSES.map((c) => [c, alpha * probs[c] + (1 - alpha) * prev[c]]));
}

// Label shown to the user, given smoothed probabilities.
export function displayLabel(probs) {
  const { label, confidence } = topOf(probs);
  return confidence < LOW_CONFIDENCE ? { text: 'uncertain', color: '#64748b', top: label } : { text: label, color: COLORS[label], top: label };
}

export const pct = (p) => `${Math.round(p * 100)}%`;
