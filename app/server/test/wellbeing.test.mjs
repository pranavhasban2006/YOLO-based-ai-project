// Wellbeing module tests. Gemini is always mocked: no network, no key needed.
import assert from 'node:assert/strict';
import { after, before, describe, it } from 'node:test';
import { once } from 'node:events';
import http from 'node:http';
import { createApp } from '../src/app.js';
import { createWellbeingRouter } from '../src/wellbeing/routes.js';
import { createGeminiClient, GeminiError } from '../src/wellbeing/gemini.js';
import { ANALYSIS_SCHEMA, analyseMessage, validateAnalysis } from '../src/wellbeing/analysis.js';
import { chooseNextTopic } from '../src/wellbeing/conversation.js';
import { createStore } from '../src/wellbeing/store.js';
import { createWellbeingService } from '../src/wellbeing/service.js';
import { detectSafety, replyIsUnsafe, safetyMessage } from '../src/wellbeing/safety.js';

const memoryStore = () => createStore({ isDbReady: () => false });

const EMOTIONS = { angry: 0.05, disgust: 0.05, fear: 0.1, happy: 0.1, neutral: 0.4, sad: 0.25, surprise: 0.05 };
const goodAnalysis = (over = {}) => ({
  mood_1_10: 5, themes: ['exams'], emotions: EMOTIONS, self_harm_signal: 'none', rationale: 'test',
  phq_items: { interest: null, mood: null, sleep: null, energy: null, appetite: null, self_worth: null, concentration: null, restlessness: null, self_harm: null },
  ...over,
});

// A fake Gemini. `analysis` is a function (text) -> object | string | Error; chat replies are canned.
function fakeGemini({ analysis = () => goodAnalysis(), chat = () => 'That sounds like a lot. How have you been sleeping?', model = 'fake-flash' } = {}) {
  const calls = { analysis: 0, chat: 0, systems: [] };
  return {
    available: true, model, models: [model], calls,
    async generate({ system, contents, schema }) {
      if (schema) {
        calls.analysis += 1;
        const last = contents[contents.length - 1].text;
        const out = analysis(last, calls.analysis);
        if (out instanceof Error) throw out;
        return { text: typeof out === 'string' ? out : JSON.stringify(out), model };
      }
      calls.chat += 1;
      calls.systems.push(system);
      const out = chat(contents, calls.chat);
      if (out instanceof Error) throw out;
      return { text: out, model };
    },
  };
}

const svc = (gemini, extra = {}) =>
  createWellbeingService({ gemini, store: memoryStore(), maxTurns: 6, counsellorContact: 'Dr Rao, Counselling Cell (ext 123)', ...extra });

describe('safety layer: explicit phrases trigger', () => {
  const explicit = [
    'I want to kill myself',
    'i think i should end my life',
    'I want to die',
    "i don't want to live anymore",
    'I wish I was dead',
    "everyone would be better off without me",
    'no one would miss me',
    'there is no reason to live',
    'I have been thinking about suicide a lot',
    "I'm suicidal",
    'I want to hurt myself',
    "i've been cutting myself",
    "I'm planning to end it all",
    'kms',
    'mujhe jeena nahi hai',
    'mar jana chahta hu',
    'main marna chahti hoon',
    'jeene ka mann nahi hai',
    'mujhe suicide karna hai',
    'khud ko khatam kar lunga',
    'sucide karne ka mann hai',
  ];
  for (const t of explicit) {
    it(`explicit: "${t}"`, () => assert.equal(detectSafety(t).level, 'explicit', JSON.stringify(detectSafety(t))));
  }
});

describe('safety layer: harmless messages do not trigger explicit', () => {
  const harmless = [
    'this exam is killing me',
    "I'm dead tired after the lab",
    "I'm dying to see the results",
    'my parents will kill me if I fail',
    'I could kill my roommate for snoring',
    'I am so stressed about the viva',
    'we studied suicide statistics in sociology class',
    'I read an article about suicide prevention for my assignment',
    'I cut myself a slice of cake',
    'I want to die of embarrassment lol',
    'I was laughing so hard I wanted to kill myself laughing',
    'The deadline is going to murder my sleep schedule',
    'I feel great today, thanks',
    'exam mein bahut tension hai',
    'kal test hai, bahut thak gaya hu',
  ];
  for (const t of harmless) {
    it(`not explicit: "${t}"`, () => assert.notEqual(detectSafety(t).level, 'explicit', JSON.stringify(detectSafety(t))));
  }
  it('"this exam is killing me" is level none', () => assert.equal(detectSafety('this exam is killing me').level, 'none'));
});

describe('safety layer: soft signals', () => {
  it('negated mentions are only "possible"', () => {
    for (const t of ["I'm not suicidal", "I don't want to die", 'I would never kill myself']) {
      assert.equal(detectSafety(t).level, 'possible', t);
    }
  });
  it('hopelessness words are "possible"', () => {
    assert.equal(detectSafety('everything feels pointless and hopeless').level, 'possible');
  });
  it('fixed message has the helplines and the counsellor contact', () => {
    const m = safetyMessage({ counsellorContact: 'Dr Rao (ext 123)' });
    for (const s of ['14416', '9820466726', '112', 'Dr Rao (ext 123)']) assert.ok(m.includes(s), s);
    assert.ok(/can't promise/i.test(m));
  });
  it('replies that discuss methods are flagged', () => {
    assert.ok(replyIsUnsafe('You could take an overdose of tablets'));
    assert.ok(!replyIsUnsafe('That sounds really hard. How is your sleep?'));
  });
});

describe('structured analysis validation and fallback', () => {
  it('accepts a valid object and normalises emotions to sum 1', () => {
    const v = validateAnalysis(goodAnalysis({ emotions: { angry: 1, disgust: 1, fear: 1, happy: 1, neutral: 4, sad: 1, surprise: 1 } }));
    assert.ok(v);
    assert.ok(Math.abs(Object.values(v.emotions).reduce((a, b) => a + b, 0) - 1) < 1e-9);
  });
  it('rejects bad values', () => {
    assert.equal(validateAnalysis(goodAnalysis({ mood_1_10: 11 })), null);
    assert.equal(validateAnalysis(goodAnalysis({ self_harm_signal: 'maybe' })), null);
    assert.equal(validateAnalysis(goodAnalysis({ phq_items: { ...goodAnalysis().phq_items, sleep: 5 } })), null);
    assert.equal(validateAnalysis(null), null);
  });
  it('the response schema lists the required fields', () => {
    assert.deepEqual(ANALYSIS_SCHEMA.required.sort(), ['emotions', 'mood_1_10', 'phq_items', 'rationale', 'self_harm_signal', 'themes']);
  });
  it('retries once after invalid JSON, then succeeds', async () => {
    const g = fakeGemini({ analysis: (_t, n) => (n === 1 ? 'not json {' : goodAnalysis({ mood_1_10: 3 })) });
    const r = await analyseMessage({ gemini: g, text: 'hello' });
    assert.equal(r.ok, true);
    assert.equal(r.analysis.mood_1_10, 3);
    assert.equal(g.calls.analysis, 2);
  });
  it('falls back to nulls after two failures', async () => {
    const g = fakeGemini({ analysis: () => 'garbage' });
    const r = await analyseMessage({ gemini: g, text: 'hello' });
    assert.equal(r.ok, false);
    assert.equal(r.analysis.fallback, true);
    assert.equal(r.analysis.mood_1_10, null);
    assert.equal(r.analysis.emotions, null);
    assert.equal(r.analysis.self_harm_signal, 'none');
    assert.ok(Object.values(r.analysis.phq_items).every((v) => v === null));
    assert.equal(g.calls.analysis, 2);
  });
  it('falls back to nulls when the request itself throws', async () => {
    const g = fakeGemini({ analysis: () => new GeminiError('http', 'boom', 500) });
    assert.equal((await analyseMessage({ gemini: g, text: 'hi' })).analysis.fallback, true);
  });
});

describe('conversation with a mocked Gemini', () => {
  it('opens with the fixed question and keeps chatting when analysis fails', async () => {
    const g = fakeGemini({ analysis: () => 'garbage' });
    const s = svc(g);
    const start = await s.start();
    assert.equal(start.reply, 'How are you feeling today?');
    const r = await s.message(start.sessionId, 'Pretty stressed about exams');
    assert.equal(r.kind, 'llm');
    assert.equal(r.reply, 'That sounds like a lot. How have you been sleeping?');
    assert.equal(r.analysis.fallback, true);
    assert.equal(r.safety.level, 'none');
    assert.equal(r.degraded, true);
  });

  it('explicit regex phrase: fixed message, no LLM chat reply, tier and flag set', async () => {
    const g = fakeGemini();
    const s = svc(g);
    const { sessionId } = await s.start();
    const r = await s.message(sessionId, 'honestly I want to kill myself');
    assert.equal(r.kind, 'safety');
    assert.equal(r.tier, 'needs_support_now');
    assert.equal(r.needs_human_followup, true);
    assert.equal(g.calls.chat, 0, 'no LLM reply may be generated for an explicit message');
    for (const x of ['14416', '9820466726', '112', 'Dr Rao, Counselling Cell (ext 123)']) assert.ok(r.reply.includes(x), x);
    assert.equal(r.complete, false);
  });

  it('safety fires even when Gemini is completely down', async () => {
    const g = fakeGemini({ analysis: () => new GeminiError('network', 'down', 0), chat: () => new GeminiError('network', 'down', 0) });
    const s = svc(g);
    const { sessionId } = await s.start();
    const r = await s.message(sessionId, 'I want to die');
    assert.equal(r.kind, 'safety');
    assert.equal(r.needs_human_followup, true);
  });

  it('explicit structured signal alone also triggers the fixed message', async () => {
    const g = fakeGemini({ analysis: () => goodAnalysis({ self_harm_signal: 'explicit' }) });
    const s = svc(g);
    const { sessionId } = await s.start();
    const r = await s.message(sessionId, 'things are quiet and grey and I keep wishing for it to stop');
    assert.equal(detectSafety('things are quiet and grey and I keep wishing for it to stop').level, 'none');
    assert.equal(r.kind, 'safety');
    assert.equal(r.tier, 'needs_support_now');
    assert.equal(g.calls.chat, 0);
  });

  it('a harmless idiom is a normal reply', async () => {
    const g = fakeGemini();
    const s = svc(g);
    const { sessionId } = await s.start();
    const r = await s.message(sessionId, 'this exam is killing me');
    assert.equal(r.kind, 'llm');
    assert.notEqual(r.tier, 'needs_support_now');
    assert.equal(r.needs_human_followup, false);
  });

  it('stays open and supportive after a safety trigger (no screening questions, repeat message on repeat)', async () => {
    const g = fakeGemini();
    const s = svc(g);
    const { sessionId } = await s.start();
    await s.message(sessionId, 'I want to end my life');
    const r2 = await s.message(sessionId, 'I just feel so alone');
    assert.equal(r2.kind, 'llm');
    assert.match(g.calls.systems.at(-1), /Stay gentle and present/);
    assert.equal(r2.tier, 'needs_support_now');
    const r3 = await s.message(sessionId, 'I want to die');
    assert.equal(r3.kind, 'safety');
    assert.ok(r3.reply.includes('14416'));
  });

  it('an unsafe model reply is replaced', async () => {
    const g = fakeGemini({ chat: () => 'You could try an overdose.' });
    const s = svc(g);
    const { sessionId } = await s.start();
    const r = await s.message(sessionId, 'rough week');
    assert.ok(!/overdose/i.test(r.reply));
    assert.equal(r.kind, 'fallback');
  });

  it('model failure falls back to a canned question and the chat continues', async () => {
    const g = fakeGemini({ chat: () => new GeminiError('http', 'overloaded', 503) });
    const s = svc(g);
    const { sessionId } = await s.start();
    const r = await s.message(sessionId, 'tired lately');
    assert.equal(r.kind, 'fallback');
    assert.ok(r.reply.length > 10);
    assert.equal(r.degraded, true);
  });

  it('a "possible" signal adds a fixed soft support line without setting the human-followup flag', async () => {
    const g = fakeGemini();
    const s = svc(g);
    const { sessionId } = await s.start();
    const r = await s.message(sessionId, 'everything feels hopeless lately');
    assert.equal(r.safety.level, 'possible');
    assert.ok(r.reply.includes('14416'));
    assert.equal(r.needs_human_followup, false);
    assert.notEqual(r.tier, 'doing_well');
  });

  it('completes after the max number of messages with a final summary', async () => {
    const g = fakeGemini({ analysis: () => goodAnalysis({ mood_1_10: 8, phq_items: { ...goodAnalysis().phq_items, sleep: 0 } }) });
    const s = svc(g);
    const { sessionId } = await s.start();
    await s.addModality(sessionId, { modality: 'face', top_label: 'happy', confidence: 0.7, probabilities: { happy: 0.7, neutral: 0.3 } });
    let r;
    for (let i = 0; i < 6; i += 1) r = await s.message(sessionId, `message ${i} I am doing fine`);
    assert.equal(r.complete, true);
    assert.equal(r.kind, 'closing');
    assert.ok(r.final.score >= 65);
    assert.equal(r.final.tier, 'doing_okay');
    await assert.rejects(s.message(sessionId, 'one more'), /finished/);
  });

  it('weaves each PHQ topic at most once, skips addressed ones, and asks about self-harm only when distressed', async () => {
    const g = fakeGemini({ analysis: () => goodAnalysis({ mood_1_10: 8 }) });
    const s = svc(g, { maxTurns: 10 });
    const { sessionId } = await s.start();
    for (let i = 0; i < 9; i += 1) await s.message(sessionId, `fine day ${i}`);
    const session = await s.get(sessionId);
    const asked = session.turns_detail.filter((t) => t.role === 'user').length;
    assert.equal(asked, 9);
    const sys = g.calls.systems.join('\n');
    assert.ok(!/thoughts of hurting themselves/.test(sys), 'no self-harm question when the mood is fine');

    const g2 = fakeGemini({ analysis: () => goodAnalysis({ mood_1_10: 3 }) });
    const s2 = svc(g2, { maxTurns: 10 });
    const st = await s2.start();
    for (let i = 0; i < 9; i += 1) await s2.message(st.sessionId, `a heavy day ${i}`);
    const sh = g2.calls.systems.filter((x) => /thoughts of hurting themselves/.test(x));
    assert.equal(sh.length, 1, 'the self-harm topic is asked exactly once');
    for (const key of ['how they have been sleeping', 'how their appetite']) {
      assert.ok(g2.calls.systems.filter((x) => x.includes(key)).length <= 1, key);
    }
  });

  it('chooseNextTopic skips topics the user already addressed', () => {
    const session = {
      asked_topics: [], tier: null, flags: { possible_self_harm: false },
      turns: [{ role: 'user', text: 'x', analysis: goodAnalysis({ phq_items: { ...goodAnalysis().phq_items, mood: 2, sleep: 1 } }) }],
    };
    assert.equal(chooseNextTopic(session).key, 'interest');
  });

  it('validates input', async () => {
    const s = svc(fakeGemini());
    const { sessionId } = await s.start();
    await assert.rejects(s.message(sessionId, '   '), /non-empty/);
    await assert.rejects(s.message(sessionId, 'x'.repeat(2001)), /under 2000/);
    await assert.rejects(s.message('nope', 'hi'), /No such/);
  });
});

describe('MOCK mode (no API key)', () => {
  it('runs a whole check-in with canned questions and clear labelling', async () => {
    const s = svc(createGeminiClient({ apiKey: '', model: 'm' }));
    assert.equal(s.mode, 'mock');
    assert.equal(s.status().mock, true);
    const start = await s.start();
    assert.equal(start.reply, 'How are you feeling today?');
    assert.equal(start.mock, true);
    assert.match(start.notice, /MOCK MODE/);
    let r;
    for (let i = 0; i < 6; i += 1) r = await s.message(start.sessionId, i % 2 ? 'I am tired and stressed' : 'it was okay');
    assert.equal(r.mock, true);
    assert.equal(r.complete, true);
    assert.ok(r.final.mock);
    const session = await s.get(start.sessionId);
    const replies = session.turns_detail.filter((t) => t.role === 'assistant' && t.kind === 'mock');
    assert.ok(replies.length >= 4);
  });

  it('the safety layer works in mock mode too', async () => {
    const s = svc(createGeminiClient({ apiKey: '', model: 'm' }));
    const { sessionId } = await s.start();
    const r = await s.message(sessionId, 'mujhe jeena nahi hai');
    assert.equal(r.kind, 'safety');
    assert.equal(r.needs_human_followup, true);
  });
});

describe('Gemini client', () => {
  it('sends the key in a header, falls back to the second model on 503, and never puts the key in the URL', async () => {
    const urls = [];
    const fetchImpl = async (url, init) => {
      urls.push({ url, key: init.headers['x-goog-api-key'] });
      if (url.includes('main-model')) return { ok: false, status: 503, json: async () => ({ error: { message: 'busy' } }) };
      return { ok: true, status: 200, json: async () => ({ candidates: [{ content: { parts: [{ text: 'hello' }] } }] }) };
    };
    const c = createGeminiClient({ apiKey: 'SECRET', model: 'main-model', fallbackModel: 'backup-model', fetchImpl, retryDelayMs: 1 });
    const out = await c.generate({ system: 's', contents: [{ role: 'user', text: 'hi' }] });
    assert.equal(out.text, 'hello');
    assert.equal(out.model, 'backup-model');
    assert.ok(urls.every((u) => u.key === 'SECRET' && !u.url.includes('SECRET')));
    assert.equal(urls.filter((u) => u.url.includes('main-model')).length, 2, 'one retry on the main model');
  });
});

describe('HTTP routes', () => {
  let server;
  let base;
  before(async () => {
    const router = createWellbeingRouter({ gemini: createGeminiClient({ apiKey: '', model: 'm' }), store: memoryStore(), maxTurns: 6, counsellorContact: 'Counsellor X' });
    server = http.createServer(createApp({ wellbeing: router }));
    server.listen(0, '127.0.0.1');
    await once(server, 'listening');
    base = `http://127.0.0.1:${server.address().port}/api/wellbeing`;
  });
  after(() => server?.close());
  const post = (path, body) => fetch(`${base}${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body ?? {}) });

  it('status reports mock mode', async () => {
    const j = await (await fetch(`${base}/status`)).json();
    assert.equal(j.mode, 'mock');
    assert.match(j.notice, /MOCK/);
  });

  it('start, message, safety, modality, end', async () => {
    const start = await (await post('/sessions', { anonymousId: 'anon-1' })).json();
    assert.ok(start.sessionId);
    const m1 = await (await post(`/sessions/${start.sessionId}/messages`, { text: 'a bit tired' })).json();
    assert.equal(m1.kind, 'mock');
    const m2 = await (await post(`/sessions/${start.sessionId}/messages`, { text: 'I want to kill myself' })).json();
    assert.equal(m2.kind, 'safety');
    assert.ok(m2.reply.includes('Counsellor X'));
    const mod = await (await post(`/sessions/${start.sessionId}/modality`, { modality: 'face', top_label: 'sad', confidence: 0.6, probabilities: { sad: 0.6 } })).json();
    assert.equal(mod.samples, 1);
    const end = await (await post(`/sessions/${start.sessionId}/end`)).json();
    assert.equal(end.status, 'completed');
    assert.equal(end.tier, 'needs_support_now');
    assert.equal(end.flags.needs_human_followup, true);
  });

  it('W2 routes: session, message, signal, finish, history (with a user header), delete my data', async () => {
    const H = { 'Content-Type': 'application/json', 'X-Wellbeing-User': 'anon-w2' };
    const call = (method, path, body) => fetch(`${base}${path}`, { method, headers: H, body: body ? JSON.stringify(body) : undefined }).then(async (r) => ({ status: r.status, json: await r.json() }));
    const start = (await call('POST', '/session')).json;
    for (const text of ['it was okay', 'a bit tired', 'fine really']) assert.equal((await call('POST', `/session/${start.sessionId}/message`, { text })).status, 200);
    assert.equal((await call('POST', `/session/${start.sessionId}/signal`, { modality: 'face', top_label: 'happy', confidence: 0.7, probabilities: { happy: 0.7, neutral: 0.3 } })).json.samples, 1);
    const fin = (await call('POST', `/session/${start.sessionId}/finish`)).json;
    assert.ok('score' in fin && 'tier' in fin && Array.isArray(fin.reasons) && Array.isArray(fin.suggestions));
    assert.equal(fin.unvalidated, true);
    const hist = (await call('GET', '/history')).json;
    assert.equal(hist.sessions.length, 1);
    assert.equal(hist.sessions[0].sessionId, start.sessionId);
    assert.equal((await call('GET', '/history', undefined)).status, 200);
    const other = await fetch(`${base}/history`, { headers: { 'X-Wellbeing-User': 'someone-else' } }).then((r) => r.json());
    assert.equal(other.sessions.length, 0, 'history is per user');
    assert.equal((await fetch(`${base}/history`)).status, 400);
    const del = (await call('DELETE', '/data')).json;
    assert.equal(del.deleted_sessions, 1);
    assert.equal((await call('GET', '/history')).json.sessions.length, 0);
    assert.equal((await call('GET', `/session/${start.sessionId}`)).status, 404);
  });
  it('errors are clean JSON', async () => {
    assert.equal((await post('/sessions/nope/messages', { text: 'hi' })).status, 404);
    const start = await (await post('/sessions')).json();
    const empty = await post(`/sessions/${start.sessionId}/messages`, { text: '' });
    assert.equal(empty.status, 400);
    assert.equal((await empty.json()).error, 'empty_message');
    const bad = await fetch(`${base}/sessions`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{bad' });
    assert.equal(bad.status, 400);
  });
});
