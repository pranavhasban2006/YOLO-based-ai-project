// Scoring, fusion, tier and trend tests on synthetic sessions. Pure functions: no network, no database.
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  DEFAULT_CONFIG, DEFAULT_RELIABILITY, bestItems, buildHistory, computeSessionResult, ema, emotionScore, fusionWeights,
  meanDistribution, modalityValues, moodComponent, phqComponent, prosodyDeltas, prosodyPenalty, slopePerDay, streakDays, tierFromScore, valence,
} from '../src/wellbeing/scoring.js';

const PHQ = ['interest', 'mood', 'sleep', 'energy', 'appetite', 'self_worth', 'concentration', 'restlessness', 'self_harm'];
const items = (o = {}) => Object.fromEntries(PHQ.map((k) => [k, o[k] ?? null]));
const emo = (o) => ({ angry: 0, disgust: 0, fear: 0, happy: 0, neutral: 0, sad: 0, surprise: 0, ...o });
const HAPPY = emo({ happy: 1 });
const SAD = emo({ sad: 1 });
const NEUTRAL = emo({ neutral: 1 });
const turn = ({ mood = null, phq = {}, emotions = NEUTRAL, fallback = false } = {}) => ({
  role: 'user', text: 'x', analysis: { mood_1_10: mood, phq_items: items(phq), themes: ['exams'], emotions, self_harm_signal: 'none', rationale: '', fallback },
});
const session = (turns, { samples = [], answers = [], flags = {}, mode = 'gemini' } = {}) => ({
  mode, turns, modality_samples: samples, answers, flags: { needs_human_followup: false, possible_self_harm: false, ...flags },
});
const face = (p, n = 10) => ({ modality: 'face', probabilities: p, n });
const DAY = 86400000;
const NOW = Date.UTC(2026, 9, 12, 12);
const Q = { ...DEFAULT_CONFIG, scoreUsesQuestionnaire: true }; // questionnaire mode, for the tests that need it
const answer = (faceMean, voiceProbs, prosody = null) => ({
  face: faceMean ? { mean: faceMean, dominant: Object.entries(faceMean).sort((a, b) => b[1] - a[1])[0][0], variability: 0.1, n: 20 } : null,
  voice_emotion: voiceProbs ? { probabilities: voiceProbs, top_label: Object.entries(voiceProbs).sort((a, b) => b[1] - a[1])[0][0] } : null,
  prosody,
});

describe('valence mapping (heuristic)', () => {
  it('all happy is high, all sad / fear / angry / disgust is low, all neutral is about 50', () => {
    assert.equal(emotionScore(HAPPY), 100);
    for (const k of ['sad', 'fear', 'angry', 'disgust']) assert.equal(emotionScore(emo({ [k]: 1 })), 0, k);
    assert.equal(emotionScore(NEUTRAL), 50);
  });
  it('surprise counts a little positive: valence 0.25 -> 62.5', () => {
    assert.equal(valence(emo({ surprise: 1 })), 0.25);
    assert.equal(emotionScore(emo({ surprise: 1 })), 62.5);
  });
  it('mixtures: valence = happy + 0.25 surprise - (sad + fear + angry + disgust)', () => {
    const p = emo({ happy: 0.4, surprise: 0.2, sad: 0.2, fear: 0.1, neutral: 0.1 });
    assert.ok(Math.abs(valence(p) - (0.4 + 0.05 - 0.3)) < 1e-9);
    assert.ok(Math.abs(emotionScore(p) - (0.15 + 1) * 50) < 1e-9);
  });
  it('the coefficients come from config.json', () => {
    assert.deepEqual(DEFAULT_CONFIG.valence, { happy: 1, surprise: 0.25, neutral: 0, sad: -1, fear: -1, angry: -1, disgust: -1 });
    const cfg = { ...DEFAULT_CONFIG, valence: { ...DEFAULT_CONFIG.valence, happy: 0.5 } };
    assert.equal(emotionScore(HAPPY, cfg), 75);
  });
});

describe('fusion weights and confidence', () => {
  const REL = { face: { value: 0.342 }, audio: { value: 0.36 }, text: { value: null }, eeg: { value: null } };
  it('weights are reliability shares', () => {
    const f = fusionWeights(['face', 'audio'], REL);
    assert.ok(Math.abs(f.weights.face - 0.342 / 0.702) < 1e-3);
    assert.ok(Math.abs(f.weights.audio - 0.36 / 0.702) < 1e-3);
    assert.ok(Math.abs(f.weights.face + f.weights.audio - 1) < 2e-3);
    assert.equal(f.confidence, 1);
  });
  it('missing modalities are dropped and the weights renormalised; confidence = sum of base weights present', () => {
    const only = fusionWeights(['face'], REL);
    assert.deepEqual(only.weights, { face: 1 });
    assert.ok(Math.abs(only.confidence - 0.342 / 0.702) < 1e-3);
    assert.equal(fusionWeights([], REL).confidence, 0);
  });
  it('a modality without a model (text, eeg) is ignored even if data arrives', () => {
    const f = fusionWeights(['face', 'text', 'eeg'], REL);
    assert.deepEqual(f.weights, { face: 1 });
    assert.deepEqual(f.ignored.sort(), ['eeg', 'text']);
  });
  it('once a text model has a reliability number, the shares change accordingly', () => {
    const rel = { ...REL, text: { value: 0.5 } };
    const f = fusionWeights(['face', 'audio', 'text'], rel);
    assert.ok(Math.abs(f.weights.text - 0.5 / 1.202) < 1e-3);
    assert.ok(Math.abs(fusionWeights(['face'], rel).confidence - 0.342 / 1.202) < 1e-3);
  });
  it('the shipped reliability file holds the cross-dataset numbers with their sources', () => {
    assert.equal(DEFAULT_RELIABILITY.face.value, 0.342);
    assert.equal(DEFAULT_RELIABILITY.audio.value, 0.36);
    assert.equal(DEFAULT_RELIABILITY.text.value, null);
    for (const m of ['face', 'audio', 'text', 'eeg']) assert.ok(DEFAULT_RELIABILITY[m].source);
  });
});

describe('model-driven score (default mode)', () => {
  it('end to end with fixed model outputs: score, weights, confidence and tier', () => {
    const s = session([turn({ mood: 2, phq: { mood: 3 } })], {
      answers: [answer(emo({ happy: 0.6, neutral: 0.4 }), emo({ happy: 0.2, neutral: 0.6, sad: 0.2 })), answer(emo({ happy: 0.8, neutral: 0.2 }), emo({ neutral: 0.8, happy: 0.2 }))],
    });
    const r = computeSessionResult(s, { nowMs: NOW });
    const faceV = emotionScore(meanDistribution([emo({ happy: 0.6, neutral: 0.4 }), emo({ happy: 0.8, neutral: 0.2 })])); // 0.7 happy -> 85
    const voiceV = emotionScore(meanDistribution([emo({ happy: 0.2, neutral: 0.6, sad: 0.2 }), emo({ neutral: 0.8, happy: 0.2 })])); // 0.2 happy 0.1 sad -> 55
    assert.equal(r.mode, 'models');
    assert.ok(Math.abs(r.modalities.face.value - faceV) < 0.06);
    assert.ok(Math.abs(r.modalities.audio.value - voiceV) < 0.06);
    const wf = 0.342 / 0.702;
    assert.ok(Math.abs(r.score - (wf * faceV + (1 - wf) * voiceV)) < 0.2, `${r.score}`);
    assert.equal(r.confidence_value, 1);
    assert.equal(r.confidence, 'high');
    assert.equal(r.tier, 'doing_okay');
    assert.equal(r.low_confidence, false);
    assert.ok(r.reasons.length >= 2);
  });
  it("Gemini's own mood and PHQ-style guesses do not change the score by default, but are still shown", () => {
    const answers = [answer(emo({ neutral: 0.7, happy: 0.3 }), null)];
    const a = computeSessionResult(session([turn({ mood: 9, phq: { mood: 0, sleep: 0, energy: 0, interest: 0 } })], { answers }), { nowMs: NOW });
    const b = computeSessionResult(session([turn({ mood: 1, phq: { mood: 3, sleep: 3, energy: 3, interest: 3 } })], { answers }), { nowMs: NOW });
    assert.equal(a.score, b.score);
    assert.equal(a.questionnaire.used_in_score, false);
    assert.equal(a.questionnaire.mood.mean_mood, 9);
    assert.equal(b.questionnaire.phq.value, 0);
  });
  it('a text-only chat has no model signal: no score, and it says how to get one', () => {
    const r = computeSessionResult(session([turn({ mood: 8 }), turn({ mood: 8 })]), { nowMs: NOW });
    assert.equal(r.score, null);
    assert.equal(r.tier, null);
    assert.ok(r.reasons.some((x) => /voice interview|emotion check/.test(x.text)));
  });
  it('chat check-in: webcam samples are one group; an audio clip result counts too', () => {
    const s = session([turn()], { samples: [face(emo({ sad: 0.8, neutral: 0.2 })), face(emo({ sad: 0.6, neutral: 0.4 })), { modality: 'audio', probabilities: emo({ happy: 1 }) }] });
    const m = modalityValues(s);
    assert.equal(m.face.n_groups, 1);
    assert.equal(m.face.n_samples, 2);
    assert.ok(m.face.value < 25);
    assert.equal(m.audio.value, 100);
  });
  it('averages inside an answer first, then across answers', () => {
    const s = session([turn()], { answers: [answer(emo({ happy: 1 }), null), answer(emo({ sad: 1 }), null), answer(emo({ neutral: 1 }), null)] });
    assert.equal(modalityValues(s).face.value, 50);
    assert.equal(modalityValues(s).face.n_groups, 3);
  });
  it('a text-model result and an EEG sample are ignored until those models exist', () => {
    const s = session([turn()], { samples: [face(HAPPY), { modality: 'text', probabilities: SAD }, { modality: 'eeg', probabilities: SAD }] });
    const r = computeSessionResult(s, { nowMs: NOW });
    assert.equal(r.score, 100);
    assert.deepEqual(Object.keys(r.components), ['face']);
    assert.equal(r.modalities.text.note, 'no model yet');
    assert.ok(r.fusion.weights.face === 1);
  });
  it('low confidence (under 0.4): labelled low and never the top-concern tier without the safety layer', () => {
    // a reliability table where the face model alone carries only 15% of the total
    const reliability = { face: { value: 0.1 }, audio: { value: 0.57 }, text: { value: null }, eeg: { value: null } };
    const s = session([turn()], { samples: [face(SAD)] });
    const r = computeSessionResult(s, { nowMs: NOW, reliability });
    assert.ok(r.confidence_value < 0.4);
    assert.equal(r.low_confidence, true);
    assert.equal(r.confidence, 'low');
    assert.ok(r.score < 30, 'the raw score is very low');
    assert.notEqual(r.tier, 'needs_support_now');
    // the same data with enough weight is allowed to reach the top-concern tier
    const full = computeSessionResult(s, { nowMs: NOW });
    assert.equal(full.low_confidence, false);
    assert.equal(full.tier, 'needs_support_now');
    // and the rule-based flag always wins
    const flagged = computeSessionResult({ ...s, flags: { needs_human_followup: true } }, { nowMs: NOW, reliability });
    assert.equal(flagged.tier, 'needs_support_now');
  });
  it('model-mode thresholds: 45 and 30 (neutral = 50 is "doing okay")', () => {
    assert.equal(computeSessionResult(session([turn()], { samples: [face(NEUTRAL)] }), { nowMs: NOW }).tier, 'doing_okay');
    assert.equal(tierFromScore(44.9, DEFAULT_CONFIG, true), 'struggling');
    assert.equal(tierFromScore(29.9, DEFAULT_CONFIG, true), 'needs_support_now');
  });
});

describe('safety overrides', () => {
  it('an explicit safety flag forces needs_support_now even with a great score', () => {
    const s = session([turn()], { samples: [face(HAPPY)], flags: { needs_human_followup: true } });
    const r = computeSessionResult(s, { nowMs: NOW });
    assert.equal(r.score, 100);
    assert.equal(r.tier, 'needs_support_now');
    assert.ok(r.reasons.some((x) => x.kind === 'safety_flag'));
  });
  it('PHQ item 9 at 2 or more forces needs_support_now; at 1 it does not', () => {
    const base = (v) => session([turn({ phq: { self_harm: v } })], { samples: [face(HAPPY)] });
    assert.equal(computeSessionResult(base(2), { nowMs: NOW }).tier, 'needs_support_now');
    assert.equal(computeSessionResult(base(1), { nowMs: NOW }).tier, 'doing_okay');
  });
  it('works with no model data at all, and cannot be switched off by the config', () => {
    const s = session([turn({ fallback: true })], { flags: { needs_human_followup: true } });
    assert.equal(computeSessionResult(s, { nowMs: NOW }).tier, 'needs_support_now');
    const cfg = { ...DEFAULT_CONFIG, tiers_models: { doing_okay_min: 0, struggling_min: 0 }, fusion: { ...DEFAULT_CONFIG.fusion, low_confidence_below: 0 }, overrides: {} };
    assert.equal(computeSessionResult(s, { nowMs: NOW, cfg }).tier, 'needs_support_now');
  });
  it('a "possible" flag lifts doing_okay to struggling, never higher', () => {
    const ok = session([turn()], { samples: [face(HAPPY)], flags: { possible_self_harm: true } });
    assert.equal(computeSessionResult(ok, { nowMs: NOW }).tier, 'struggling');
  });
});

describe('questionnaire mode (switch on): PHQ, mood, text emotion, face, voice', () => {
  const positive = () => session([
    turn({ mood: 8, phq: { mood: 0, sleep: 0 }, emotions: emo({ happy: 0.7, neutral: 0.3 }) }),
    turn({ mood: 9, phq: { energy: 0, interest: 0 }, emotions: emo({ happy: 0.8, neutral: 0.2 }) }),
    turn({ mood: 8, phq: { appetite: 0, concentration: 0 }, emotions: emo({ happy: 0.6, neutral: 0.4 }) }),
    turn({ mood: 8, phq: { self_worth: 0, restlessness: 0 }, emotions: emo({ happy: 0.7, neutral: 0.3 }) }),
  ]);
  const low = () => session([
    turn({ mood: 2, phq: { mood: 3, sleep: 3 }, emotions: emo({ sad: 0.7, fear: 0.2, neutral: 0.1 }) }),
    turn({ mood: 2, phq: { energy: 3, interest: 3 }, emotions: emo({ sad: 0.8, neutral: 0.2 }) }),
    turn({ mood: 3, phq: { appetite: 2, concentration: 3 }, emotions: emo({ sad: 0.6, fear: 0.2, neutral: 0.2 }) }),
    turn({ mood: 2, phq: { self_worth: 3, restlessness: 2 }, emotions: emo({ sad: 0.7, angry: 0.1, neutral: 0.2 }) }),
  ]);
  it('PHQ: 100 is no symptoms, needs 4 answered items for normal confidence', () => {
    assert.equal(phqComponent(items({ mood: 0, sleep: 0, energy: 0, interest: 0 })).value, 100);
    assert.equal(phqComponent(items({ mood: 3, sleep: 3, energy: 3, interest: 3 })).value, 0);
    assert.equal(phqComponent(items({ mood: 1, sleep: 1, energy: 1 })).confidence, 'low');
    assert.equal(phqComponent(items()).value, null);
    assert.equal(bestItems([turn({ phq: { sleep: 1 } }), turn({ phq: { sleep: 3 } }), turn({ phq: { sleep: 0 } })]).sleep, 3);
  });
  it('mood: 1 maps to 0 and 10 maps to 100', () => {
    assert.equal(moodComponent([turn({ mood: 1 })]).value, 0);
    assert.equal(moodComponent([turn({ mood: 10 })]).value, 100);
    assert.equal(moodComponent([turn()]).value, null);
  });
  it('all-positive, and clearly low, give doing_okay and needs_support_now', () => {
    const p = computeSessionResult(positive(), { nowMs: NOW, cfg: Q });
    const l = computeSessionResult(low(), { nowMs: NOW, cfg: Q });
    assert.equal(p.mode, 'questionnaire');
    assert.ok(p.score >= 85, p.score);
    assert.equal(p.tier, 'doing_okay');
    assert.ok(l.score < 40, l.score);
    assert.equal(l.tier, 'needs_support_now');
    assert.ok(l.reasons.some((x) => x.component === 'phq' && x.effect === 'lowers'));
  });
  it('default weights are PHQ 0.35, mood 0.25, text emotion 0.10, face 0.15, voice 0.15, renormalised over what exists', () => {
    assert.deepEqual(DEFAULT_CONFIG.questionnaire.weights, { phq: 0.35, mood: 0.25, text_emotion: 0.1, face: 0.15, voice: 0.15 });
    const none = computeSessionResult(positive(), { nowMs: NOW, cfg: Q });
    assert.deepEqual(Object.keys(none.components).sort(), ['mood', 'phq', 'text_emotion']);
    assert.ok(Math.abs(none.components.phq.weight - 0.35 / 0.7) < 1e-3);
    const all = positive();
    all.answers = [answer(emo({ happy: 1 }), emo({ happy: 1 }))];
    const full = computeSessionResult(all, { nowMs: NOW, cfg: Q });
    assert.deepEqual(Object.keys(full.components).sort(), ['face', 'mood', 'phq', 'text_emotion', 'voice']);
    assert.ok(Math.abs(full.components.face.weight - 0.15) < 1e-3);
  });
  it('missing PHQ items: low confidence, still scored; no PHQ at all: component dropped', () => {
    const s = session([turn({ mood: 7, phq: { sleep: 0 } }), turn({ mood: 7 })]);
    const r = computeSessionResult(s, { nowMs: NOW, cfg: Q });
    assert.equal(r.phq.confidence, 'low');
    assert.equal(r.confidence, 'low');
    const none = computeSessionResult(session([turn({ mood: 6 }), turn({ mood: 6 })]), { nowMs: NOW, cfg: Q });
    assert.equal(none.components.phq, undefined);
  });
  it('a single low-confidence bad session moves at most one tier down from the usual level', () => {
    const s = session([turn({ mood: 1, phq: { mood: 3 }, emotions: SAD })]);
    const r = computeSessionResult(s, { nowMs: NOW, cfg: Q });
    assert.ok(r.score < 40);
    assert.equal(r.confidence, 'low');
    assert.equal(r.tier, 'struggling');
    assert.equal(computeSessionResult(s, { nowMs: NOW, cfg: Q, prior: [{ t: NOW - 3 * DAY, score: 50 }, { t: NOW - 2 * DAY, score: 48 }] }).tier, 'needs_support_now');
  });
  it('prosody lowers the voice component by a capped amount', () => {
    const base = { speech_rate_wps: 3, pause_ratio: 0.1, pause_mean_s: 0.3, latency_s: 0.5, pitch_mean_hz: 180, pitch_std_st: 3, energy_mean_db: -20, energy_std_db: 5, speech_s: 8 };
    const slow = { ...base, speech_rate_wps: 1.5, pause_ratio: 0.6, pause_mean_s: 1.2, pitch_std_st: 1, energy_mean_db: -30 };
    const mk = (second) => { const s = positive(); s.answers = [answer(null, emo({ neutral: 1 }), base), answer(null, emo({ neutral: 1 }), second)]; return s; };
    const calm = computeSessionResult(mk(base), { nowMs: NOW, cfg: Q });
    const heavy = computeSessionResult(mk(slow), { nowMs: NOW, cfg: Q });
    assert.ok(heavy.components.voice.value < calm.components.voice.value);
    assert.ok(calm.components.voice.value - heavy.components.voice.value <= Q.questionnaire.prosody.max_penalty);
  });
});

describe('prosody deltas are relative to the personal baseline', () => {
  const base = { speech_rate_wps: 3, pause_ratio: 0.1, pause_mean_s: 0.3, latency_s: 0.5, pitch_mean_hz: 200, pitch_std_st: 4, energy_mean_db: -20, energy_std_db: 5, speech_s: 10 };
  it('computes relative changes', () => {
    const d = prosodyDeltas({ ...base, speech_rate_wps: 1.5, pitch_mean_hz: 220, energy_mean_db: -26 }, base);
    assert.equal(d.speech_rate_wps, -0.5);
    assert.equal(d.pitch_mean_hz, 0.1);
    assert.equal(d.energy_mean_db, -6);
    assert.equal(prosodyDeltas(null, base), null);
  });
  it('no change from baseline means no penalty; heavy slowing is penalised but capped', () => {
    assert.equal(prosodyPenalty(prosodyDeltas(base, base)), 0);
    const pen = prosodyPenalty(prosodyDeltas({ ...base, speech_rate_wps: 0.5, pause_ratio: 0.9, pause_mean_s: 2, pitch_std_st: 0.2, energy_mean_db: -40 }, base));
    assert.ok(pen > 5 && pen <= DEFAULT_CONFIG.questionnaire.prosody.max_penalty);
  });
  it('the first answer is the baseline; later answers carry deltas; prosody is display-only in model mode', () => {
    const s = session([turn()], { samples: [face(NEUTRAL)], answers: [answer(null, null, base), answer(null, null, { ...base, speech_rate_wps: 1.5 })] });
    const r = computeSessionResult(s, { nowMs: NOW });
    assert.equal(r.timeline[0].is_baseline, true);
    assert.equal(r.timeline[0].deltas, null);
    assert.equal(r.timeline[1].deltas.speech_rate_wps, -0.5);
    assert.equal(r.prosody.used_in_score, false);
    assert.equal(r.prosody.indicators_are_heuristic, true);
    assert.equal(r.score, 50);
  });
});

describe('trend: smoothing, EMA, slope, streak, history', () => {
  it('EMA smoothing: one bad day after good days does not drop the tier on its own', () => {
    const prior = [{ t: NOW - 3 * DAY, score: 70 }, { t: NOW - 2 * DAY, score: 68 }, { t: NOW - 1 * DAY, score: 70 }];
    const s = session([turn()], { samples: [face(SAD)] });
    const r = computeSessionResult(s, { nowMs: NOW, prior });
    assert.equal(r.score, 0);
    assert.ok(r.smoothed_score > 30);
    assert.equal(r.raw_tier, 'needs_support_now');
    assert.notEqual(r.tier, 'needs_support_now');
    assert.ok(r.reasons.some((x) => x.kind === 'smoothing'));
  });
  it('EMA with alpha 0.5, slope per day, streak', () => {
    assert.deepEqual(ema([80, 60, 40]).series, [80, 70, 55]);
    assert.equal(ema([]).current, null);
    assert.equal(slopePerDay([{ t: NOW - 6 * DAY, score: 40 }, { t: NOW - 3 * DAY, score: 55 }, { t: NOW, score: 70 }], NOW), 5);
    assert.equal(slopePerDay([{ t: NOW, score: 50 }], NOW), null);
    assert.equal(slopePerDay([{ t: NOW - 30 * DAY, score: 50 }, { t: NOW - 20 * DAY, score: 60 }], NOW), null);
    assert.equal(streakDays([NOW, NOW - DAY, NOW - 2 * DAY], NOW), 3);
    assert.equal(streakDays([NOW - 3 * DAY], NOW), 0);
  });
  it('history: ordered sessions with EMA and slope; active sessions are left out; thresholds follow the mode', () => {
    const mk = (i, score) => ({ sessionId: `s${i}`, status: 'completed', started_at: new Date(NOW - (5 - i) * DAY), completed_at: new Date(NOW - (5 - i) * DAY), final: { score, tier: 'struggling', confidence: 'high', themes: [], mood: { mean_mood: 6 }, emotion: { value: 50, distribution: NEUTRAL } } });
    const h = buildHistory([mk(5, 70), mk(1, 50), mk(3, 60), { sessionId: 'a', status: 'active', final: null, started_at: new Date(NOW) }], { nowMs: NOW });
    assert.deepEqual(h.sessions.map((s) => s.sessionId), ['s1', 's3', 's5']);
    assert.deepEqual(h.sessions.map((s) => s.ema), [50, 55, 62.5]);
    assert.equal(h.slope_per_day, 5);
    assert.equal(h.thresholds.doing_okay_min, 45);
    assert.equal(buildHistory([], { cfg: Q }).thresholds.doing_okay_min, 65);
  });
});
