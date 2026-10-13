import { CLASSES } from './emotion.js';

// Aggregate the face probabilities sampled while one answer was being given.
// Only these numbers are ever sent to the server: mean probabilities, the dominant emotion, how much they varied,
// and the sample count. No frames, no images.
export function faceAggregate(samples) {
  const valid = samples.filter((s) => s && CLASSES.every((c) => Number.isFinite(s[c])));
  if (!valid.length) return null;
  const mean = {};
  for (const c of CLASSES) mean[c] = valid.reduce((a, s) => a + s[c], 0) / valid.length;
  const sum = CLASSES.reduce((a, c) => a + mean[c], 0) || 1;
  for (const c of CLASSES) mean[c] /= sum;
  const dominant = CLASSES.reduce((best, c) => (mean[c] > mean[best] ? c : best), CLASSES[0]);
  const std = CLASSES.map((c) => {
    const m = valid.reduce((a, s) => a + s[c], 0) / valid.length;
    return Math.sqrt(valid.reduce((a, s) => a + (s[c] - m) ** 2, 0) / valid.length);
  });
  return { mean, dominant, variability: Math.round((std.reduce((a, b) => a + b, 0) / std.length) * 1000) / 1000, n: valid.length };
}

// Split a reply into sentences so they can be spoken one at a time (short pause before the first words).
export function splitSentences(text) {
  const parts = String(text || '')
    .replace(/\s*\n+\s*-?\s*/g, '. ') // line breaks and list bullets become sentence breaks
    .replace(/([:;])\.\s/g, '$1 ')
    .replace(/\s+/g, ' ')
    .trim()
    .match(/[^.!?]+[.!?]+(?=\s|$)|[^.!?]+$/g);
  return (parts || []).map((s) => s.trim()).filter(Boolean);
}

// Rough speaking time, used only as a fallback when the browser never reports that speech finished.
export const estimateSpeechMs = (text, rate = 0.92) => Math.max(1200, (String(text).split(/\s+/).length / (2.6 * rate)) * 1000);

// Make a sentence easier to say aloud: helpline numbers digit by digit, a few symbols spelled out.
export function speakable(sentence) {
  return String(sentence)
    .replace(/\b(\d{3,})\b/g, (m) => m.split('').join(' '))
    .replace(/(\d)x(\d)/gi, '$1 by $2')
    .replace(/Tele-MANAS/g, 'Tele Manas')
    .replace(/AASRA/g, 'Aasra')
    .replace(/[()]/g, ',')
    .replace(/\s+,/g, ',');
}