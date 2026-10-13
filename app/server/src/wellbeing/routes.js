import express from 'express';
import multer from 'multer';
import * as modelClient from '../services/modelClient.js';
import { config } from '../config.js';
import { createGeminiClient } from './gemini.js';
import { createStore } from './store.js';
import { WellbeingError, createWellbeingService } from './service.js';

// Builds the /api/wellbeing router. Dependencies can be replaced in tests (mocked Gemini, in-memory store).
export function createWellbeingRouter(deps = {}) {
  const gemini = deps.gemini ?? createGeminiClient({
    apiKey: config.wellbeing.apiKey,
    model: config.wellbeing.model,
    fallbackModel: config.wellbeing.fallbackModel,
  });
  const service = createWellbeingService({
    gemini,
    store: deps.store ?? createStore(),
    maxTurns: deps.maxTurns ?? config.wellbeing.maxTurns,
    counsellorContact: deps.counsellorContact ?? config.wellbeing.counsellorContact,
  });

  // Simple in-memory rate limit for the endpoints that call Gemini (cost and abuse control).
  const limit = deps.rateLimit ?? { max: 30, windowMs: 60_000 };
  const hits = new Map();
  const limited = (req, res, next) => {
    const key = String(req.get('x-wellbeing-user') || req.ip);
    const now = Date.now();
    const recent = (hits.get(key) || []).filter((t) => now - t < limit.windowMs);
    if (recent.length >= limit.max) {
      return res.status(429).json({ error: 'rate_limited', message: 'Too many messages. Please wait a moment.' });
    }
    recent.push(now);
    hits.set(key, recent);
    if (hits.size > 5000) hits.clear();
    return next();
  };

  const router = express.Router();
  router.use(express.json({ limit: '100kb' }));
  const wrap = (fn) => (req, res, next) => Promise.resolve(fn(req, res)).then((out) => res.json(out), next);

  // Face probabilities for one answer, aggregated in the browser. Only numbers are accepted.
  const parseFace = (raw) => {
    try {
      const f = typeof raw === 'string' ? JSON.parse(raw) : raw;
      const keys = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'];
      if (!f || typeof f.mean !== 'object' || !keys.every((k) => Number.isFinite(f.mean[k]))) return null;
      return { mean: Object.fromEntries(keys.map((k) => [k, f.mean[k]])), dominant: keys.includes(f.dominant) ? f.dominant : null, variability: Number.isFinite(f.variability) ? f.variability : null, n: Number.isFinite(f.n) ? Math.min(f.n, 10000) : null };
    } catch {
      return null;
    }
  };

  // The client keeps a random anonymous id in localStorage and sends it in this header.
  const who = (req) => String(req.get('x-wellbeing-user') || '').trim().slice(0, 80) || null;
  const startBody = (req) => ({ ...(req.body || {}), anonymousId: req.body?.anonymousId || who(req) });
  const send = (req) => ({ source: req.body?.source === 'voice' ? 'voice' : 'text' });

  const audioUpload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 10 * 1024 * 1024, files: 1 } });
  const ttsModel = deps.ttsModel ?? config.wellbeing.ttsModel;
  const modelSvc = deps.modelClient ?? modelClient;

  router.get('/status', wrap(async () => {
    const base = service.status();
    let stt = { available: false };
    try {
      const h = await modelSvc.health();
      stt = { available: Boolean(h.stt?.available), model: h.stt?.model ?? null, voice_emotion: Boolean(h.audio) };
    } catch {
      /* model service offline */
    }
    return { ...base, stt, tts: { gemini: gemini.available && typeof gemini.tts === 'function', model: ttsModel } };
  }));

  // Optional "natural voice": Gemini text-to-speech (the default voice is the browser's, which needs no key).
  router.post('/tts', limited, async (req, res, next) => {
    try {
      const text = String(req.body?.text ?? '').trim().slice(0, 600);
      if (!text) return res.status(400).json({ error: 'empty_text', message: 'Send {"text": "..."}.' });
      if (!gemini.available || typeof gemini.tts !== 'function') return res.status(503).json({ error: 'tts_unavailable', message: 'Natural voice needs GEMINI_API_KEY.' });
      const { wav } = await gemini.tts(text, { model: ttsModel });
      res.set('Content-Type', 'audio/wav').send(wav);
    } catch (err) {
      res.status(503).json({ error: 'tts_unavailable', message: 'Natural voice is not available right now.' });
    }
  });

  // Voice interview: one spoken answer. The audio goes to the local model service for transcript, prosody and voice
  // emotion and is then discarded; only the transcript and per-answer aggregates are kept.
  router.post('/session/:id/voice-answer', limited, audioUpload.single('audio'), async (req, res, next) => {
    try {
      const sttMode = ['local', 'browser', 'gemini'].includes(req.body?.stt) ? req.body.stt : 'local';
      let analysis = { transcript: { text: '', words: [] }, prosody: null, voice_emotion: null };
      if (req.file) {
        try {
          analysis = await modelSvc.analyseAnswer(req.file.buffer, req.file.originalname, (req.file.mimetype || '').split(';')[0] || 'audio/webm');
        } catch (err) {
          if (sttMode === 'local' && !String(req.body?.browser_text || '').trim()) throw err;
        }
      }
      let text = (analysis.transcript?.text || '').trim();
      if (sttMode === 'browser') text = String(req.body?.browser_text || '').trim().slice(0, 2000) || text;
      if (sttMode === 'gemini' && req.file && gemini.available && typeof gemini.transcribe === 'function') {
        text = (await gemini.transcribe(req.file.buffer, (req.file.mimetype || 'audio/webm').split(';')[0]).catch(() => '')) || text;
      }
      if (!text) return res.json({ empty: true, message: 'I could not hear anything. Could you say that again?', transcript: '' });

      const face = parseFace(req.body?.face);
      const out = await service.message(req.params.id, text, {
        source: 'voice',
        voice: {
          prosody: analysis.prosody,
          voice_emotion: analysis.voice_emotion,
          face,
          duration_s: analysis.prosody?.duration_s ?? analysis.transcript?.duration_s ?? null,
          skipped: false,
        },
      });
      res.json({ ...out, transcript: text, prosody: analysis.prosody, voice_emotion: analysis.voice_emotion, face });
    } catch (err) {
      next(err);
    }
  });

  // "Skip question": the next question is asked without an answer (no analysis data is stored for it).
  router.post('/session/:id/skip', limited, wrap(async (req) => service.message(req.params.id, '(skipped this question)', { source: 'voice', voice: { skipped: true } })));
  // Current routes (singular) and the plural W1 aliases.
  for (const base of ['/session', '/sessions']) {
    router.post(base, wrap(async (req) => service.start(startBody(req))));
    router.get(`${base}/:id`, wrap(async (req) => service.get(req.params.id, who(req))));
    router.post(`${base}/:id/message`, limited, wrap(async (req) => service.message(req.params.id, req.body?.text, send(req))));
    router.post(`${base}/:id/signal`, wrap(async (req) => service.addModality(req.params.id, req.body || {})));
    router.post(`${base}/:id/finish`, wrap(async (req) => service.finish(req.params.id)));
    router.delete(`${base}/:id`, wrap(async (req) => service.deleteSession(req.params.id, who(req))));
  }
  router.post('/sessions/:id/messages', limited, wrap(async (req) => service.message(req.params.id, req.body?.text, send(req))));
  router.post('/sessions/:id/modality', wrap(async (req) => service.addModality(req.params.id, req.body || {})));
  router.post('/sessions/:id/end', wrap(async (req) => service.finish(req.params.id)));
  router.get('/history', wrap(async (req) => service.history(who(req) || String(req.query.user || '').slice(0, 80))));
  // "Delete my data": removes every session stored for this anonymous id.
  router.delete('/data', wrap(async (req) => service.deleteData(who(req))));
  // eslint-disable-next-line no-unused-vars
  router.use((err, _req, res, _next) => {
    if (err instanceof WellbeingError) return res.status(err.status).json({ error: err.code, message: err.message });
    if (err.type === 'entity.parse.failed') return res.status(400).json({ error: 'bad_json', message: 'Request body is not valid JSON.' });
    if (err.type === 'entity.too.large') return res.status(413).json({ error: 'payload_too_large', message: 'Request body is too large.' });
    console.error('[wellbeing] unexpected error:', err.name); // name only: messages can contain user text
    return res.status(500).json({ error: 'server_error', message: 'Unexpected server error.' });
  });

  router.service = service;
  return router;
}
