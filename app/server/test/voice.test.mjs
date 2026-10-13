// Voice interview backend: session kind, per-answer records, the voice-answer route with a mocked model service.
import assert from 'node:assert/strict';
import { after, before, describe, it } from 'node:test';
import { once } from 'node:events';
import http from 'node:http';
import { createApp } from '../src/app.js';
import { createGeminiClient } from '../src/wellbeing/gemini.js';
import { createWellbeingRouter } from '../src/wellbeing/routes.js';
import { createWellbeingService } from '../src/wellbeing/service.js';
import { createStore } from '../src/wellbeing/store.js';

const FACE = { mean: { angry: 0, disgust: 0, fear: 0, happy: 0.6, neutral: 0.4, sad: 0, surprise: 0 }, dominant: 'happy', variability: 0.1, n: 20 };
const VOICE = { top_label: 'neutral', confidence: 0.6, probabilities: { angry: 0, disgust: 0, fear: 0, happy: 0.2, neutral: 0.6, sad: 0.2, surprise: 0 } };
const PROSODY = { duration_s: 5, latency_s: 0.6, speech_rate_wps: 3, pause_ratio: 0.1, pause_mean_s: 0.3, pitch_mean_hz: 190, pitch_std_st: 3, energy_mean_db: -22, energy_std_db: 5 };

describe('voice session (service)', () => {
  it('opens with "How has your day been?", records one aggregate per answer, and scores from face and voice models', async () => {
    const svc = createWellbeingService({ gemini: createGeminiClient({ apiKey: '', model: 'm' }), store: createStore({ isDbReady: () => false }), maxTurns: 6 });
    const start = await svc.start({ kind: 'voice' });
    assert.equal(start.reply, 'How has your day been?');
    assert.equal(start.kind, 'voice');
    let r;
    for (let i = 0; i < 3; i += 1) r = await svc.message(start.sessionId, `answer ${i} it was fine`, { source: 'voice', voice: { prosody: { ...PROSODY, speech_rate_wps: 3 - i * 0.5 }, voice_emotion: VOICE, face: FACE } });
    assert.equal(r.kind, 'mock');
    const fin = await svc.finish(start.sessionId);
    assert.equal(fin.kind, 'voice');
    assert.equal(fin.timeline.length, 3);
    assert.equal(fin.timeline[0].is_baseline, true);
    assert.ok(fin.timeline[2].deltas.speech_rate_wps < 0);
    assert.equal(fin.transcript.length, 3);
    assert.equal(fin.transcript[0].question, 'How has your day been?');
    assert.ok(fin.modalities.face.available && fin.modalities.audio.available && !fin.modalities.text.available);
    assert.equal(fin.confidence_value, 1);
    assert.ok(fin.score > 50);
    const session = await svc.get(start.sessionId);
    const dump = JSON.stringify(session);
    assert.ok(!/base64|audio_data|frame/i.test(dump.replace(/audio_model|voice_emotion/g, '')), 'no raw audio or frames in the stored session');
  });

  it('a skipped question is recorded without analysis data', async () => {
    const svc = createWellbeingService({ gemini: createGeminiClient({ apiKey: '', model: 'm' }), store: createStore({ isDbReady: () => false }) });
    const { sessionId } = await svc.start({ kind: 'voice' });
    await svc.message(sessionId, '(skipped this question)', { source: 'voice', voice: { skipped: true } });
    const s = await svc.get(sessionId);
    assert.equal(s.turns_detail.filter((t) => t.role === 'user').length, 1);
  });

  it('the safety layer runs on a spoken transcript before the next question', async () => {
    const calls = { chat: 0 };
    const gemini = { available: true, model: 'x', models: ['x'], generate: async ({ schema }) => { if (!schema) calls.chat += 1; return { text: schema ? 'bad' : 'ok' }; } };
    const svc = createWellbeingService({ gemini, store: createStore({ isDbReady: () => false }), counsellorContact: 'Counsellor Z' });
    const { sessionId } = await svc.start({ kind: 'voice' });
    const r = await svc.message(sessionId, 'honestly I do not want to live anymore', { source: 'voice', voice: { prosody: PROSODY, face: FACE, voice_emotion: VOICE } });
    assert.equal(r.kind, 'safety');
    assert.equal(calls.chat, 0);
    assert.ok(r.reply.includes('14416') && r.reply.includes('Counsellor Z'));
    const fin = await svc.finish(sessionId);
    assert.equal(fin.tier, 'needs_support_now'); // set by the rule-based layer, not by tone or face
  });
});

describe('voice-answer route (mock model service)', () => {
  let server;
  let base;
  let analysis;
  before(async () => {
    analysis = { transcript: { text: 'I have been tired but okay', words: [], duration_s: 4 }, prosody: PROSODY, voice_emotion: VOICE };
    const modelClient = { health: async () => ({ stt: { available: true, model: 'base.en' }, audio: {} }), analyseAnswer: async () => analysis };
    const router = createWellbeingRouter({ gemini: createGeminiClient({ apiKey: '', model: 'm' }), store: createStore({ isDbReady: () => false }), modelClient });
    server = http.createServer(createApp({ wellbeing: router }));
    server.listen(0, '127.0.0.1');
    await once(server, 'listening');
    base = `http://127.0.0.1:${server.address().port}/api/wellbeing`;
  });
  after(() => server?.close());
  const H = { 'X-Wellbeing-User': 'voice-user' };
  const form = (extra = {}) => {
    const f = new FormData();
    f.append('audio', new Blob([new Uint8Array(2000)], { type: 'audio/webm' }), 'a.webm');
    for (const [k, v] of Object.entries(extra)) f.append(k, v);
    return f;
  };

  it('status reports speech-to-text and voice availability', async () => {
    const j = await (await fetch(`${base}/status`)).json();
    assert.equal(j.stt.available, true);
    assert.equal(j.tts.gemini, false);
  });

  it('transcribes, stores the answer aggregates and replies; empty speech is not a turn', async () => {
    const s = await (await fetch(`${base}/session`, { method: 'POST', headers: { ...H, 'Content-Type': 'application/json' }, body: JSON.stringify({ kind: 'voice' }) })).json();
    assert.equal(s.reply, 'How has your day been?');
    const r1 = await (await fetch(`${base}/session/${s.sessionId}/voice-answer`, { method: 'POST', headers: H, body: form({ face: JSON.stringify(FACE) }) })).json();
    assert.equal(r1.transcript, 'I have been tired but okay');
    assert.equal(r1.kind, 'mock');
    assert.equal(r1.turn, 1);
    assert.equal(r1.face.dominant, 'happy');
    analysis = { transcript: { text: '', words: [] }, prosody: null, voice_emotion: null };
    const empty = await (await fetch(`${base}/session/${s.sessionId}/voice-answer`, { method: 'POST', headers: H, body: form() })).json();
    assert.equal(empty.empty, true);
    const state = await (await fetch(`${base}/session/${s.sessionId}`, { headers: H })).json();
    assert.equal(state.user_turns, 1, 'an empty answer is not counted');
    const skip = await (await fetch(`${base}/session/${s.sessionId}/skip`, { method: 'POST', headers: H })).json();
    assert.equal(skip.turn, 2);
  });

  it('rejects malformed face data instead of storing it', async () => {
    analysis = { transcript: { text: 'hello there', words: [] }, prosody: PROSODY, voice_emotion: null };
    const s = await (await fetch(`${base}/session`, { method: 'POST', headers: { ...H, 'Content-Type': 'application/json' }, body: JSON.stringify({ kind: 'voice' }) })).json();
    const r = await (await fetch(`${base}/session/${s.sessionId}/voice-answer`, { method: 'POST', headers: H, body: form({ face: JSON.stringify({ mean: { happy: 'x' }, frame: 'data:image/jpeg;base64,AAAA' }) }) })).json();
    assert.equal(r.face, null);
    const full = await (await fetch(`${base}/session/${s.sessionId}`, { headers: H })).json();
    assert.ok(!JSON.stringify(full).includes('base64'));
  });
});
