// Safety review checks (phase W4): logging, deletion, ownership, rate limit, config cannot disable safety,
// and no diagnosis labels in anything shown to the user.
import assert from 'node:assert/strict';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import http from 'node:http';
import { once } from 'node:events';
import path from 'node:path';
import { after, before, describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';
import { createApp } from '../src/app.js';
import { GeminiError, createGeminiClient } from '../src/wellbeing/gemini.js';
import { createWellbeingRouter } from '../src/wellbeing/routes.js';
import { computeSessionResult, suggestionsFor } from '../src/wellbeing/scoring.js';
import { replyHasDiagnosis, safetyMessage, softSupportLine } from '../src/wellbeing/safety.js';
import { createWellbeingService } from '../src/wellbeing/service.js';
import { createStore } from '../src/wellbeing/store.js';
import { WellbeingSession } from '../src/wellbeing/WellbeingSession.js';
import { CLOSING_FALLBACK, INTRO, OPENING, TOPICS, mockReply } from '../src/wellbeing/conversation.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const srcDir = path.resolve(here, '..', 'src');
const clientSrc = path.resolve(here, '..', '..', 'client', 'src');
const walk = (dir) => readdirSync(dir).flatMap((f) => {
  const p = path.join(dir, f);
  return statSync(p).isDirectory() ? walk(p) : [p];
});

const BANNED = /suicid|depress|bipolar|anxiety disorder|mentally ill|mental illness|you have a disorder|schizo|ptsd/i;

describe('no secrets or user text in plain logs', () => {
  it('a full session, including model failures, writes neither the key nor the user text to the console', async () => {
    const lines = [];
    const orig = { log: console.log, warn: console.warn, error: console.error, info: console.info };
    for (const k of Object.keys(orig)) console[k] = (...a) => lines.push(a.map(String).join(' '));
    try {
      const KEY = 'SECRET-KEY-DO-NOT-LOG-12345';
      const UNIQUE = 'zebra-moon-unique-user-text-98765';
      const failing = createGeminiClient({
        apiKey: KEY, model: 'm', retryDelayMs: 1,
        fetchImpl: async () => ({ ok: false, status: 500, json: async () => ({ error: { message: 'server blew up' } }) }),
      });
      const svc = createWellbeingService({ gemini: failing, store: createStore({ isDbReady: () => false }), maxTurns: 6 });
      const { sessionId } = await svc.start();
      await svc.message(sessionId, `${UNIQUE} I am tired`);
      await svc.message(sessionId, `${UNIQUE} I want to die`);
      await svc.end(sessionId);
      // the HTTP error path too
      const server = http.createServer(createApp({ wellbeing: createWellbeingRouter({ gemini: failing, store: createStore({ isDbReady: () => false }) }) }));
      server.listen(0, '127.0.0.1');
      await once(server, 'listening');
      const base = `http://127.0.0.1:${server.address().port}/api/wellbeing`;
      await fetch(`${base}/session/x/message`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: `{"text":"${UNIQUE}"` });
      await fetch(`${base}/session/x/message`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: UNIQUE }) });
      server.close();
      const all = lines.join('\n');
      assert.ok(!all.includes(KEY), 'API key found in logs');
      assert.ok(!all.includes(UNIQUE), 'user text found in logs');
    } finally {
      Object.assign(console, orig);
    }
  });

  it('source: only the known console calls exist and none logs a message body', () => {
    const calls = [];
    for (const f of walk(srcDir).filter((p) => p.endsWith('.js'))) {
      readFileSync(f, 'utf8').split('\n').forEach((line, i) => {
        if (/console\.(log|warn|error|info|debug)/.test(line)) calls.push(`${path.relative(srcDir, f)}:${i + 1}: ${line.trim()}`);
      });
    }
    for (const c of calls) assert.ok(!/req\.body|\.text\b|message\b.*err\.message(?!.*\[db\])/.test(c) || /\[db\]|\[live\]/.test(c), c);
    const wb = calls.filter((c) => c.startsWith('wellbeing'));
    assert.ok(wb.every((c) => /err\.name/.test(c)), wb.join('\n'));
  });
});

describe('the explicit-risk path cannot be disabled', () => {
  it('no environment variable or config key mentions turning safety off', () => {
    const env = new Set();
    for (const f of walk(srcDir).filter((p) => p.endsWith('.js'))) {
      for (const m of readFileSync(f, 'utf8').matchAll(/process\.env\.([A-Z0-9_]+)/g)) env.add(m[1]);
    }
    assert.ok([...env].every((k) => !/SAFETY|DISABLE|SKIP|BYPASS/.test(k)), [...env].join(','));
    const cfg = readFileSync(path.join(srcDir, 'wellbeing', 'config.json'), 'utf8');
    assert.ok(!/safety|disable/i.test(cfg.replace(/"_comment":.*\n/, '')), 'config.json must not hold a safety switch');
  });

  it('safety.js imports nothing from Gemini, and the service calls it before the model', () => {
    const safety = readFileSync(path.join(srcDir, 'wellbeing', 'safety.js'), 'utf8');
    assert.ok(!/import .*gemini|fetch\(/i.test(safety));
    const service = readFileSync(path.join(srcDir, 'wellbeing', 'service.js'), 'utf8');
    assert.ok(service.indexOf('detectSafety(text)') < service.indexOf('analyseMessage('), 'regex must run before the analysis call');
    assert.ok(service.indexOf("if (combined === 'explicit')") < service.indexOf('await llmReply('), 'explicit branch must precede the LLM reply');
  });

  it('with every Gemini call failing, an explicit message still gets the fixed message and flags', async () => {
    const dead = { available: true, model: 'x', models: ['x'], generate: async () => { throw new GeminiError('network', 'down', 0); } };
    const svc = createWellbeingService({ gemini: dead, store: createStore({ isDbReady: () => false }), counsellorContact: 'Counsellor Y' });
    const { sessionId } = await svc.start();
    const r = await svc.message(sessionId, 'I want to end my life');
    assert.equal(r.kind, 'safety');
    assert.ok(r.reply.includes('Counsellor Y') && r.reply.includes('14416'));
    assert.equal(r.needs_human_followup, true);
  });
});

describe('deletion, ownership, retention, rate limit', () => {
  let server;
  let base;
  before(async () => {
    const router = createWellbeingRouter({
      gemini: createGeminiClient({ apiKey: '', model: 'm' }),
      store: createStore({ isDbReady: () => false }),
      rateLimit: { max: 4, windowMs: 60_000 },
    });
    server = http.createServer(createApp({ wellbeing: router }));
    server.listen(0, '127.0.0.1');
    await once(server, 'listening');
    base = `http://127.0.0.1:${server.address().port}/api/wellbeing`;
  });
  after(() => server?.close());
  const call = (method, p, user, body) =>
    fetch(`${base}${p}`, { method, headers: { 'Content-Type': 'application/json', ...(user ? { 'X-Wellbeing-User': user } : {}) }, body: body ? JSON.stringify(body) : undefined })
      .then(async (r) => ({ status: r.status, json: await r.json() }));

  it("a session's transcript is only returned to its owner", async () => {
    const s = (await call('POST', '/session', 'alice')).json;
    assert.equal((await call('GET', `/session/${s.sessionId}`, 'alice')).status, 200);
    assert.equal((await call('GET', `/session/${s.sessionId}`, 'mallory')).status, 403);
    assert.equal((await call('GET', `/session/${s.sessionId}`)).status, 403);
    assert.equal((await call('DELETE', `/session/${s.sessionId}`, 'mallory')).status, 403);
    assert.equal((await call('DELETE', `/session/${s.sessionId}`, 'alice')).status, 200);
    assert.equal((await call('GET', `/session/${s.sessionId}`, 'alice')).status, 404);
  });

  it('delete my data removes only that user\'s sessions, including chat text', async () => {
    const a = (await call('POST', '/session', 'bob')).json;
    await call('POST', `/session/${a.sessionId}/message`, 'bob', { text: 'a private thing' });
    const c = (await call('POST', '/session', 'carol')).json;
    assert.equal((await call('DELETE', '/data', 'bob')).json.deleted_sessions, 1);
    assert.equal((await call('GET', `/session/${a.sessionId}`, 'bob')).status, 404);
    assert.equal((await call('GET', `/session/${c.sessionId}`, 'carol')).status, 200);
    assert.equal((await call('DELETE', '/data')).status, 400, 'needs a user id');
  });

  it('messages are rate limited', async () => {
    const s = (await call('POST', '/session', 'dave')).json;
    const codes = [];
    for (let i = 0; i < 6; i += 1) codes.push((await call('POST', `/session/${s.sessionId}/message`, 'dave', { text: 'hello' })).status);
    assert.deepEqual(codes.slice(0, 4), [200, 200, 200, 200]);
    assert.ok(codes.slice(4).every((c) => c === 429), codes.join(','));
  });

  it('stored sessions expire automatically (TTL index on started_at)', () => {
    const idx = WellbeingSession.schema.indexes().find(([f]) => f.started_at === 1);
    assert.ok(idx, 'no TTL index');
    assert.ok(idx[1].expireAfterSeconds >= 86400);
  });
});

describe('nothing shown to the user labels them with a diagnosis', () => {
  it('fixed texts, topics prompts shown as questions, scoring reasons and suggestions are free of banned words', () => {
    const shown = [
      OPENING, INTRO, CLOSING_FALLBACK, safetyMessage({ counsellorContact: 'X' }), safetyMessage({ repeat: true }), softSupportLine({}),
      ...TOPICS.map((t) => mockReply({ turn: 1, topic: t })),
    ];
    const turn = (mood, phq, emo) => ({ role: 'user', text: 'x', analysis: { mood_1_10: mood, phq_items: phq, themes: [], emotions: emo, self_harm_signal: 'none', rationale: '' } });
    const all = { interest: 3, mood: 3, sleep: 3, energy: 3, appetite: 3, self_worth: 3, concentration: 3, restlessness: 3, self_harm: 3 };
    const emo = { angry: 0, disgust: 0, fear: 0, happy: 0, neutral: 0, sad: 1, surprise: 0 };
    for (const flags of [{}, { needs_human_followup: true }, { possible_self_harm: true }]) {
      const r = computeSessionResult({ mode: 'gemini', flags, turns: [turn(1, all, emo), turn(1, all, emo)], modality_samples: [] });
      shown.push(r.tier_label ?? '', r.disclaimer, ...r.reasons.map((x) => x.text), ...r.suggestions);
    }
    for (const t of ['doing_okay', 'struggling', 'needs_support_now']) shown.push(...suggestionsFor(t, all, 1));
    for (const s of shown) assert.ok(!BANNED.test(s), `banned word in: ${s}`);
  });

  it('client source has no diagnosis labels or banned words in user-visible text', () => {
    const hits = [];
    for (const f of walk(clientSrc).filter((p) => /\.(jsx|js)$/.test(p))) {
      readFileSync(f, 'utf8').split('\n').forEach((line, i) => {
        if (/^\s*(\/\/|\*|\/\*)/.test(line)) return;
        if (BANNED.test(line)) hits.push(`${path.relative(clientSrc, f)}:${i + 1}: ${line.trim()}`);
      });
    }
    assert.deepEqual(hits, []);
  });

  it('a model reply that labels the user with a condition is replaced', async () => {
    assert.ok(replyHasDiagnosis('It sounds like you have depression.'));
    assert.ok(replyHasDiagnosis('You are clinically anxious and probably have bipolar disorder'));
    assert.ok(!replyHasDiagnosis('That sounds like a heavy week. How has your sleep been?'));
    const g = { available: true, model: 'x', models: ['x'], generate: async ({ schema }) => ({ text: schema ? 'garbage' : 'You are depressed and you have depression.' }) };
    const svc = createWellbeingService({ gemini: g, store: createStore({ isDbReady: () => false }) });
    const { sessionId } = await svc.start();
    const r = await svc.message(sessionId, 'rough week');
    assert.ok(!/depress/i.test(r.reply), r.reply);
    assert.equal(r.kind, 'fallback');
  });
});
