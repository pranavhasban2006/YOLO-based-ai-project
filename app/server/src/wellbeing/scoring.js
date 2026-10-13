// Scoring, tiers and trends. Pure functions: no I/O, no clock unless a time is passed in.
//
// Default ("models") mode: the score is a late fusion of the TRAINED emotion models (face, audio, text, EEG slot).
// Each model's 7-class output becomes a 0-100 value with a heuristic valence mapping, and modalities are weighted by
// their measured cross-dataset reliability (model_reliability.json). Gemini's own mood / PHQ-style guesses are shown
// but not scored unless config.scoreUsesQuestionnaire is true.
// All weights and thresholds come from config.json and are UNVALIDATED defaults.
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
export const DEFAULT_CONFIG = require('./config.json');
export const DEFAULT_RELIABILITY = require('./model_reliability.json');

export const PHQ_KEYS = ['interest', 'mood', 'sleep', 'energy', 'appetite', 'self_worth', 'concentration', 'restlessness', 'self_harm'];
export const EMOTION_KEYS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'];
export const MODALITIES = ['face', 'audio', 'text', 'eeg'];
export const TIER_ORDER = ['doing_okay', 'struggling', 'needs_support_now']; // best to worst
export const TIER_LABEL = {
  doing_okay: 'Doing okay',
  struggling: 'Having a harder time',
  needs_support_now: 'Support would help now',
};
const MODEL_NAME = { face: 'face model', audio: 'voice model', text: 'text model', eeg: 'EEG model' };
const TOPIC_PHRASE = {
  interest: 'enjoying the things you usually like', mood: 'your mood', sleep: 'your sleep', energy: 'your energy',
  appetite: 'your appetite', self_worth: 'how you feel about yourself', concentration: 'focus and concentration',
  restlessness: 'feeling restless or on edge', self_harm: 'some very heavy thoughts',
};

const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
const round1 = (x) => (x === null || x === undefined ? null : Math.round(x * 10) / 10);
const round3 = (x) => (x === null || x === undefined ? null : Math.round(x * 1000) / 1000);

// ---------- valence mapping (heuristic, not a clinical measure) ----------

export function valence(p, cfg = DEFAULT_CONFIG) {
  let v = 0;
  for (const k of EMOTION_KEYS) v += (cfg.valence[k] ?? 0) * (p[k] || 0);
  return clamp(v, -1, 1);
}
// 7-class distribution -> 0..100. neutral = 50, all happy = 100, all sad / fear / angry / disgust = 0.
export function emotionScore(p, cfg = DEFAULT_CONFIG) {
  return (valence(p, cfg) + 1) * 50;
}

export function meanDistribution(list) {
  const valid = list.filter((p) => p && typeof p === 'object');
  if (!valid.length) return null;
  const out = {};
  for (const k of EMOTION_KEYS) out[k] = mean(valid.map((p) => (Number.isFinite(p[k]) ? p[k] : 0)));
  const s = EMOTION_KEYS.reduce((a, k) => a + out[k], 0);
  if (s <= 0) return null;
  for (const k of EMOTION_KEYS) out[k] /= s;
  return out;
}

// ---------- questionnaire-style values (always computed and shown; scored only if the switch is on) ----------

export function bestItems(turns) {
  const best = Object.fromEntries(PHQ_KEYS.map((k) => [k, null]));
  for (const t of turns) {
    for (const k of PHQ_KEYS) {
      const v = t.analysis?.phq_items?.[k];
      if (typeof v === 'number') best[k] = best[k] === null ? v : Math.max(best[k], v);
    }
  }
  return best;
}

export function phqComponent(items, cfg = DEFAULT_CONFIG) {
  const q = cfg.questionnaire.phq;
  const answered = PHQ_KEYS.filter((k) => typeof items?.[k] === 'number');
  const sum = answered.reduce((a, k) => a + items[k], 0);
  return {
    value: answered.length ? round1(100 * (1 - sum / (answered.length * q.max_item_value))) : null, // 100 = no symptoms
    answered: answered.length,
    partial_sum: answered.length ? sum : null,
    confidence: answered.length >= q.min_answered_items ? 'ok' : 'low',
    items,
  };
}

export function moodComponent(turns) {
  const moods = turns.map((t) => t.analysis?.mood_1_10).filter((m) => typeof m === 'number');
  const m = mean(moods);
  return { value: m === null ? null : round1(((m - 1) / 9) * 100), mean_mood: round1(m), n: moods.length };
}

// ---------- prosody (heuristic indicators, relative to the person's own first answer) ----------

const REL_FIELDS = ['speech_rate_wps', 'pause_ratio', 'pause_mean_s', 'latency_s', 'pitch_mean_hz', 'pitch_std_st', 'energy_std_db', 'speech_s'];

export function prosodyDeltas(f, base) {
  if (!f || !base) return null;
  const d = {};
  for (const k of REL_FIELDS) {
    const a = f[k];
    const b = base[k];
    const eps = k === 'pause_ratio' || k === 'pause_mean_s' ? 0.1 : 1e-6;
    d[k] = typeof a === 'number' && typeof b === 'number' ? round3((a - b) / (Math.abs(b) + eps)) : null;
  }
  d.energy_mean_db = typeof f.energy_mean_db === 'number' && typeof base.energy_mean_db === 'number' ? round3(f.energy_mean_db - base.energy_mean_db) : null;
  return d;
}

// Capped penalty (0..max_penalty points) for lower pitch variability, slower rate, longer pauses, lower energy.
export function prosodyPenalty(deltas, cfg = DEFAULT_CONFIG) {
  if (!deltas) return 0;
  const p = cfg.questionnaire.prosody;
  let pen = 0;
  if (typeof deltas.speech_rate_wps === 'number' && deltas.speech_rate_wps < p.slower_rate_below) pen += Math.min(4, (-deltas.speech_rate_wps - 0.15) * 10);
  if (typeof deltas.pause_ratio === 'number' && deltas.pause_ratio > p.pause_ratio_above) pen += Math.min(4, (deltas.pause_ratio - 0.25) * 8);
  if (typeof deltas.pause_mean_s === 'number' && deltas.pause_mean_s > p.pause_mean_above) pen += Math.min(3, (deltas.pause_mean_s - 0.25) * 6);
  if (typeof deltas.pitch_std_st === 'number' && deltas.pitch_std_st < p.pitch_std_below) pen += Math.min(4, (-deltas.pitch_std_st - 0.25) * 8);
  if (typeof deltas.energy_mean_db === 'number' && deltas.energy_mean_db < p.energy_db_below) pen += Math.min(3, (-deltas.energy_mean_db - 3) * 0.6);
  return clamp(round1(pen), 0, p.max_penalty);
}

// ---------- model modalities ----------

function groupsFromSession(session) {
  const answers = session.answers || [];
  const samples = session.modality_samples || [];
  const g = { face: [], audio: [], text: [], eeg: [] };

  for (const a of answers) {
    if (a.face?.mean) g.face.push(a.face.mean); // aggregate within an answer first
    if (a.voice_emotion?.probabilities) g.audio.push(a.voice_emotion.probabilities);
    if (a.text_model?.probabilities) g.text.push(a.text_model.probabilities);
  }
  // chat check-in: all webcam samples form one group; each audio clip or text result is its own group
  const faceSamples = samples.filter((s) => s.modality === 'face').map((s) => s.probabilities);
  if (faceSamples.length) g.face.push(meanDistribution(faceSamples));
  for (const s of samples) {
    if (s.modality === 'audio' && s.probabilities) g.audio.push(s.probabilities);
    if (s.modality === 'text' && s.probabilities) g.text.push(s.probabilities);
  }
  const nSamples = {
    face: faceSamples.length + answers.filter((a) => a.face?.mean).reduce((n, a) => n + (a.face.n || 1), 0),
    audio: g.audio.length, text: g.text.length, eeg: 0,
  };
  return { groups: g, nSamples };
}

// Returns {face:{distribution, value, n_groups, n_samples}, ...} for the modalities that have data.
export function modalityValues(session, cfg = DEFAULT_CONFIG) {
  const { groups, nSamples } = groupsFromSession(session);
  const out = {};
  for (const m of ['face', 'audio', 'text']) {
    const dist = meanDistribution(groups[m]); // then across answers
    if (!dist) continue;
    out[m] = { distribution: dist, value: round1(emotionScore(dist, cfg)), n_groups: groups[m].length, n_samples: nSamples[m] };
  }
  return out; // eeg: reserved, always ignored for now
}

// Weights from measured reliability. base = share among every modality that has a reliability number;
// confidence = sum of base weights of the modalities present; weight = renormalised over the present ones.
export function fusionWeights(present, reliability = DEFAULT_RELIABILITY) {
  const known = MODALITIES.filter((m) => typeof reliability[m]?.value === 'number' && reliability[m].value > 0);
  const total = known.reduce((a, m) => a + reliability[m].value, 0);
  const usable = present.filter((m) => known.includes(m));
  const presentTotal = usable.reduce((a, m) => a + reliability[m].value, 0);
  const base = {};
  const weights = {};
  for (const m of known) base[m] = round3(reliability[m].value / total);
  for (const m of usable) weights[m] = round3(reliability[m].value / presentTotal);
  const confidence = usable.reduce((a, m) => a + reliability[m].value / total, 0);
  return { base, weights, confidence: usable.length ? round3(confidence) : 0, ignored: present.filter((m) => !known.includes(m)) };
}

// ---------- tiers ----------

export function tierFromScore(score, cfg = DEFAULT_CONFIG, models = false) {
  if (score === null || score === undefined) return null;
  const t = models ? cfg.tiers_models : cfg.tiers;
  if (score >= t.doing_okay_min) return 'doing_okay';
  if (score >= t.struggling_min) return 'struggling';
  return 'needs_support_now';
}
export const worseOf = (a, b) => (TIER_ORDER.indexOf(a) >= TIER_ORDER.indexOf(b) ? a : b);

// ---------- trend ----------

export function ema(scores, alpha = DEFAULT_CONFIG.trend.ema_alpha) {
  let e = null;
  const series = [];
  for (const s of scores) {
    if (typeof s !== 'number') {
      series.push(e);
      continue;
    }
    e = e === null ? s : alpha * s + (1 - alpha) * e;
    series.push(round1(e));
  }
  return { series, current: e === null ? null : round1(e) };
}

export function slopePerDay(points, nowMs, windowDays = DEFAULT_CONFIG.trend.slope_window_days) {
  const from = nowMs - windowDays * 86400000;
  const pts = points.filter((p) => typeof p.score === 'number' && p.t >= from && p.t <= nowMs + 1000);
  if (pts.length < 2) return null;
  const xs = pts.map((p) => (p.t - from) / 86400000);
  const mx = mean(xs);
  const my = mean(pts.map((p) => p.score));
  const den = xs.reduce((a, x) => a + (x - mx) ** 2, 0);
  if (den < 1e-9) return null;
  const num = xs.reduce((a, x, i) => a + (x - mx) * (pts[i].score - my), 0);
  return Math.round((num / den) * 100) / 100;
}

// ---------- the whole session ----------

export function computeSessionResult(session, { prior = [], nowMs = Date.now(), cfg = DEFAULT_CONFIG, reliability = DEFAULT_RELIABILITY } = {}) {
  const userTurns = session.turns.filter((t) => t.role === 'user');
  const analysed = userTurns.filter((t) => t.analysis && !t.analysis.fallback);
  const items = bestItems(analysed);
  const phq = phqComponent(items, cfg);
  const mood = moodComponent(analysed);
  const textEmotionDist = meanDistribution(analysed.map((t) => t.analysis?.emotions));
  const textEmotionValue = textEmotionDist ? round1(emotionScore(textEmotionDist, cfg)) : null;
  const modal = modalityValues(session, cfg);
  const useQ = Boolean(cfg.scoreUsesQuestionnaire);

  // prosody per answer, relative to the first answer (the personal baseline)
  const answers = session.answers || [];
  const baseline = answers[0]?.prosody || null;
  const timeline = answers.map((a, i) => {
    const deltas = i === 0 ? null : prosodyDeltas(a.prosody, baseline);
    return {
      index: a.index ?? i,
      question: a.question || null,
      duration_s: a.duration_s ?? a.prosody?.duration_s ?? null,
      face: a.face?.mean ? { value: round1(emotionScore(a.face.mean, cfg)), dominant: a.face.dominant || null, variability: a.face.variability ?? null, n: a.face.n ?? null } : null,
      voice: a.voice_emotion?.probabilities ? { value: round1(emotionScore(a.voice_emotion.probabilities, cfg)), top_label: a.voice_emotion.top_label || null } : null,
      prosody: a.prosody || null,
      deltas,
      is_baseline: i === 0,
    };
  });
  const penalties = timeline.map((t) => prosodyPenalty(t.deltas, cfg)).filter((x) => x > 0);
  const prosodyPen = penalties.length ? mean(penalties) : 0;

  // ---- the score ----
  let score = null;
  const components = {};
  let fusion = null;
  let confidenceValue = null;
  let confidence;
  let lowConfidence = false;

  if (!useQ) {
    fusion = fusionWeights(Object.keys(modal), reliability);
    const used = Object.keys(fusion.weights);
    if (used.length) {
      score = round1(used.reduce((a, m) => a + fusion.weights[m] * modal[m].value, 0));
      if (cfg.fusion.use_prosody && cfg.fusion.prosody_weight > 0 && prosodyPen > 0) {
        score = round1(clamp(score - prosodyPen * cfg.fusion.prosody_weight, 0, 100));
      }
    }
    for (const m of used) components[m] = { value: modal[m].value, weight: fusion.weights[m] };
    confidenceValue = fusion.confidence;
    lowConfidence = score !== null && confidenceValue < cfg.fusion.low_confidence_below;
    confidence = score === null ? 'none' : lowConfidence ? 'low' : confidenceValue < 0.75 ? 'medium' : 'high';
  } else {
    const w = cfg.questionnaire.weights;
    const parts = [];
    if (phq.value !== null) parts.push(['phq', phq.value]);
    if (mood.value !== null) parts.push(['mood', mood.value]);
    if (textEmotionValue !== null) parts.push(['text_emotion', textEmotionValue]);
    if (modal.face) parts.push(['face', modal.face.value]);
    if (modal.audio) parts.push(['voice', clamp(modal.audio.value - prosodyPen, 0, 100)]);
    const wTotal = parts.reduce((a, [k]) => a + w[k], 0);
    for (const [k, v] of parts) components[k] = { value: v, weight: round3(w[k] / wTotal) };
    score = parts.length ? round1(parts.reduce((a, [k, v]) => a + (w[k] / wTotal) * v, 0)) : null;
    if (score === null) confidence = 'none';
    else if (phq.confidence === 'ok' && userTurns.length >= cfg.confidence.high_min_user_turns) confidence = 'high';
    else if (phq.confidence === 'ok' || userTurns.length >= cfg.confidence.high_min_user_turns) confidence = 'medium';
    else confidence = 'low';
    lowConfidence = confidence === 'low';
  }

  // ---- trend ----
  const priorScores = prior.slice(-cfg.trend.ema_last_sessions).map((p) => p.score).filter((s) => typeof s === 'number');
  const priorEma = ema(priorScores, cfg.trend.ema_alpha).current;
  const trendEma = ema(score === null ? priorScores : [...priorScores, score], cfg.trend.ema_alpha);
  const smoothed = score === null ? null : trendEma.current;
  const slope = slopePerDay([...prior.map((p) => ({ t: p.t, score: p.score })), ...(score === null ? [] : [{ t: nowMs, score }])], nowMs, cfg.trend.slope_window_days);

  // ---- tier ----
  const reasons = [];
  let tier = tierFromScore(smoothed, cfg, !useQ);
  const rawTier = tierFromScore(score, cfg, !useQ);
  if (tier && rawTier && tier !== rawTier) {
    reasons.push({ kind: 'smoothing', text: 'Your recent check-ins were taken into account, so one day on its own moves the result only partly.' });
  }
  if (tier && confidence === 'low') {
    const refTier = tierFromScore(priorEma, cfg, !useQ) || 'doing_okay';
    const maxIdx = Math.min(TIER_ORDER.length - 1, TIER_ORDER.indexOf(refTier) + 1);
    if (TIER_ORDER.indexOf(tier) > maxIdx) {
      tier = TIER_ORDER[maxIdx];
      reasons.push({ kind: 'low_confidence_cap', text: 'There was not much to go on in this session, so the result is kept close to your usual level.' });
    }
  }
  // low confidence: never the highest-concern tier from the models alone (only the rule-based safety layer sets it)
  if (!useQ && lowConfidence && tier === 'needs_support_now') {
    tier = 'struggling';
    reasons.push({ kind: 'low_confidence_top_tier', text: 'Only part of the picture was available, so this is shown as a gentler level. A fuller check-in would be more reliable.' });
  }

  // ---- hard overrides: independent of the score, the models and the config ----
  const selfHarmItem = items.self_harm;
  const explicitFlag = Boolean(session.flags?.needs_human_followup);
  const itemFlag = typeof selfHarmItem === 'number' && selfHarmItem >= cfg.overrides.self_harm_item_min;
  if (explicitFlag) {
    tier = 'needs_support_now';
    reasons.push({ kind: 'safety_flag', text: 'You shared some very heavy thoughts during the chat. Talking to a real person soon is the best next step.' });
  } else if (itemFlag) {
    tier = 'needs_support_now';
    reasons.push({ kind: 'item_9', text: 'Some of your answers pointed to very heavy thoughts. Talking to a real person soon would help.' });
  }
  const flaggedOverride = explicitFlag || itemFlag;
  if (session.flags?.possible_self_harm && !flaggedOverride && tier === 'doing_okay') {
    tier = 'struggling';
    reasons.push({ kind: 'possible_flag', text: 'Some of what you wrote sounded heavy, so it is worth checking in with someone you trust.' });
  }

  // ---- reasons shown to the user ----
  const drivers = [];
  const who = { face: 'facial expressions', audio: 'voice', text: 'words' };
  if (!useQ) {
    for (const m of Object.keys(components)) {
      const v = components[m].value;
      const top = modal[m].top ?? Object.entries(modal[m].distribution).sort((a, b) => b[1] - a[1])[0][0];
      const effect = v < 40 ? 'lowers' : v >= 60 ? 'raises' : 'neutral';
      const lean = v < 40 ? 'leaned towards sadness, worry or tension' : v >= 60 ? 'leaned positive' : 'looked mostly calm or mixed';
      drivers.push({ component: m, effect, text: `Your ${who[m]} ${lean} (${MODEL_NAME[m]}: mostly ${top}).` });
    }
    if (!Object.keys(components).length) {
      drivers.push({ component: 'all', effect: 'neutral', text: 'No face, voice or text emotion data was available, so there is no score. Turn on the emotion check in a chat, or try the voice interview.' });
    } else if (lowConfidence) {
      drivers.push({ component: 'all', effect: 'neutral', text: 'Only some of the signals were available, so treat this as a rough picture.' });
    }
  } else {
    if (phq.value !== null) {
      const worst = PHQ_KEYS.filter((k) => k !== 'self_harm' && typeof items[k] === 'number' && items[k] >= 2).sort((a, b) => items[b] - items[a]).slice(0, 3);
      if (worst.length) drivers.push({ component: 'phq', effect: 'lowers', text: `Things that felt heavier lately: ${worst.map((k) => TOPIC_PHRASE[k]).join(', ')}.` });
      else if (phq.value >= 80) drivers.push({ component: 'phq', effect: 'raises', text: 'Your answers about sleep, energy and daily life were mostly on the lighter side.' });
      if (phq.confidence === 'low') drivers.push({ component: 'phq', effect: 'neutral', text: 'Only a few topics came up, so this part is a rough picture.' });
    }
    if (mood.value !== null) {
      drivers.push({ component: 'mood', effect: mood.value < 40 ? 'lowers' : mood.value >= 65 ? 'raises' : 'neutral', text: mood.value < 40 ? 'The mood you described was on the low side.' : mood.value >= 65 ? 'The mood you described was fairly good.' : 'The mood you described was in the middle.' });
    }
    for (const m of ['face', 'audio']) {
      if (modal[m]) drivers.push({ component: m, effect: modal[m].value < 40 ? 'lowers' : modal[m].value >= 60 ? 'raises' : 'neutral', text: `The ${m === 'audio' ? 'voice' : 'facial'} emotion model leaned ${modal[m].value < 40 ? 'towards sadness or tension' : modal[m].value >= 60 ? 'positive' : 'calm or mixed'}.` });
    }
    if (score === null && !flaggedOverride) drivers.push({ component: 'all', effect: 'neutral', text: 'Not enough was shared to give a result. Try a slightly longer chat next time.' });
  }

  const questionnaire = {
    used_in_score: useQ,
    phq: { value: phq.value, answered: phq.answered, partial_sum: phq.partial_sum, confidence: phq.confidence, items },
    mood: { value: mood.value, mean_mood: mood.mean_mood },
    text_emotion: { value: textEmotionValue, distribution: textEmotionDist, note: 'Language-model reading of the text. Displayed, not scored by default.' },
  };
  const modalities = {};
  for (const m of MODALITIES) {
    const known = typeof reliability[m]?.value === 'number';
    const dist = modal[m]?.distribution;
    modalities[m] = {
      available: Boolean(modal[m]),
      value: modal[m]?.value ?? null,
      weight: fusion?.weights?.[m] ?? null,
      base_weight: fusion?.base?.[m] ?? null,
      reliability: known ? reliability[m].value : null,
      n_answers: modal[m]?.n_groups ?? 0,
      n_samples: modal[m]?.n_samples ?? 0,
      top: dist ? Object.entries(dist).sort((a, b) => b[1] - a[1])[0][0] : null,
      note: known ? null : 'no model yet',
    };
  }
  const distribution = meanDistribution([modal.face?.distribution, modal.audio?.distribution, modal.text?.distribution]) || textEmotionDist;

  return {
    mode: useQ ? 'questionnaire' : 'models',
    score,
    smoothed_score: smoothed,
    tier,
    tier_label: tier ? TIER_LABEL[tier] : null,
    raw_tier: rawTier,
    confidence,
    confidence_value: confidenceValue,
    low_confidence: lowConfidence,
    components,
    modalities,
    fusion: fusion ? { weights: fusion.weights, base_weights: fusion.base, confidence: fusion.confidence, reliability: Object.fromEntries(MODALITIES.map((m) => [m, reliability[m]?.value ?? null])) } : null,
    questionnaire,
    phq: questionnaire.phq,
    mood: questionnaire.mood,
    emotion: { value: score, distribution, sources: Object.keys(modal) },
    prosody: { baseline, indicators_are_heuristic: true, used_in_score: useQ || (cfg.fusion.use_prosody && cfg.fusion.prosody_weight > 0), penalty: round1(prosodyPen) },
    timeline,
    overrides: { safety_flag: explicitFlag, item_9: itemFlag, possible: Boolean(session.flags?.possible_self_harm) },
    reasons: [...reasons, ...drivers],
    trend: { ema: smoothed, ema_series: trendEma.series, slope_per_day: slope, n_prior_sessions: priorScores.length },
    themes: topThemes(analysed),
    suggestions: suggestionsFor(tier, items, mood.value),
    mock: session.mode === 'mock',
    unvalidated: true,
    disclaimer: 'A rough summary from emotion models and one short session. It is not a medical test or a diagnosis.',
  };
}

function topThemes(turns) {
  const c = {};
  for (const t of turns) for (const th of t.analysis?.themes || []) c[th] = (c[th] || 0) + 1;
  return Object.entries(c).sort((a, b) => b[1] - a[1]).slice(0, 6).map(([k]) => k);
}

export function suggestionsFor(tier, items = {}) {
  const out = [];
  if (tier === 'needs_support_now') {
    out.push('Reach out to someone you trust today, such as a friend, family member or hostel warden, and tell them how you have been feeling.');
    out.push('Contact your college counsellor, or call a free helpline (Tele-MANAS 14416) to talk with a trained person.');
  }
  if ((items.sleep ?? 0) >= 2) out.push('Try a steady sleep and wake time, and keep screens away for the last 30 minutes before bed.');
  if ((items.energy ?? 0) >= 2 || (items.appetite ?? 0) >= 2) out.push('Small basics help: a short walk in daylight, water, and regular meals.');
  if ((items.concentration ?? 0) >= 2) out.push('Study in short blocks (25 minutes, then a 5 minute break) and write down the one task to finish first.');
  if ((items.interest ?? 0) >= 2 || (items.restlessness ?? 0) >= 2) out.push('Pick one small thing you usually enjoy and do it for ten minutes, or try a few slow breaths when you feel on edge.');
  if (tier === 'struggling') out.push('Talk to a friend or someone you trust about this week. Saying it out loud often makes it lighter.');
  if (tier === 'doing_okay') {
    out.push('Keep the habits that are working for you, like regular sleep, some movement and time with friends.');
    out.push('Check in again in a few days to see how things are going.');
  }
  if (out.length < 2) out.push('Check in again in a few days. A few notes over time show more than a single day.');
  const seen = new Set();
  return out.filter((s) => !seen.has(s) && seen.add(s)).slice(0, 3);
}

// ---------- history ----------

export function buildHistory(sessions, { nowMs = Date.now(), cfg = DEFAULT_CONFIG } = {}) {
  const rows = sessions
    .filter((s) => s.final && s.status !== 'active')
    .map((s) => ({
      sessionId: s.sessionId,
      t: new Date(s.completed_at || s.started_at).getTime(),
      score: s.final.score ?? null,
      tier: s.final.tier ?? null,
      confidence: s.final.confidence ?? null,
      confidence_value: s.final.confidence_value ?? null,
      mock: Boolean(s.final.mock),
      themes: s.final.themes || [],
      mood: s.final.mood?.mean_mood ?? null,
      emotion_value: s.final.emotion?.value ?? null,
      emotion_distribution: s.final.emotion?.distribution ?? null,
      kind: (s.answers || []).length ? 'voice' : 'chat',
      final: s.final,
    }))
    .sort((a, b) => a.t - b.t);
  const e = ema(rows.map((r) => r.score), cfg.trend.ema_alpha);
  const out = rows.map((r, i) => ({ ...r, ema: e.series[i] }));
  return {
    sessions: out,
    ema: e.current,
    slope_per_day: slopePerDay(rows, nowMs, cfg.trend.slope_window_days),
    streak_days: streakDays(rows.map((r) => r.t), nowMs),
    thresholds: cfg.scoreUsesQuestionnaire ? cfg.tiers : cfg.tiers_models,
  };
}

export function streakDays(times, nowMs = Date.now()) {
  const day = (t) => Math.floor(t / 86400000);
  const days = new Set(times.map(day));
  let d = day(nowMs);
  if (!days.has(d)) d -= 1;
  let n = 0;
  while (days.has(d)) {
    n += 1;
    d -= 1;
  }
  return n;
}
