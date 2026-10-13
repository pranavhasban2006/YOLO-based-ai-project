// Check-in orchestration. Order of work for each user message:
//   1. regex safety detector (no LLM)       2. structured analysis (separate Gemini call)
//   3. combine both signals -> explicit => fixed message, no LLM reply
//   4. otherwise the LLM reply (or the canned MOCK reply)
import { randomUUID } from 'node:crypto';
import { analyseMessage, mockAnalysis, nullAnalysis } from './analysis.js';
import {
  CLOSING_FALLBACK, INTRO, OPENING, VOICE_OPENING, buildSystemPrompt, chooseNextTopic, mockReply, userTurns,
} from './conversation.js';
import { DEFAULT_CONFIG, computeSessionResult, buildHistory } from './scoring.js';
import { HELPLINES, combineSignals, detectSafety, replyHasDiagnosis, replyIsUnsafe, safetyMessage, softSupportLine } from './safety.js';

export class WellbeingError extends Error {
  constructor(status, code, message) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

const MAX_TEXT = 2000;
const EXTRA_TURNS_AFTER_SAFETY = 4;

export function createWellbeingService({ gemini, store, maxTurns = 8, counsellorContact = '', scoringConfig = DEFAULT_CONFIG }) {
  const mode = gemini?.available ? 'gemini' : 'mock';
  const busy = new Set();
  const clampTurns = Math.min(10, Math.max(6, Number(maxTurns) || 8));

  const publicSession = (s) => ({
    sessionId: s.sessionId,
    mode: s.mode,
    status: s.status,
    tier: s.tier,
    flags: s.flags,
    turns: s.turns.length,
    user_turns: userTurns(s),
    max_turns: s.max_turns,
    final: s.final,
  });

  function notice() {
    return mode === 'mock'
      ? 'MOCK MODE: no GEMINI_API_KEY is set, so questions are canned and the analysis is a keyword heuristic.'
      : null;
  }

  async function start({ userId = null, anonymousId = null, kind = 'chat' } = {}) {
    const voice = kind === 'voice';
    const opening = voice ? VOICE_OPENING : OPENING;
    const session = {
      kind: voice ? 'voice' : 'chat',
      answers: [],
      sessionId: randomUUID(),
      userId: userId ? String(userId).slice(0, 80) : null,
      anonymousId: userId ? null : String(anonymousId || randomUUID()).slice(0, 80),
      mode,
      model: mode === 'gemini' ? gemini.model : null,
      status: 'active',
      max_turns: clampTurns,
      turns: [{ role: 'assistant', text: opening, ts: new Date(), kind: 'opening' }],
      modality_samples: [],
      asked_topics: [],
      tier: null,
      flags: { needs_human_followup: false, possible_self_harm: false, safety_triggered_at: null, degraded: false },
      final: null,
      started_at: new Date(),
      completed_at: null,
    };
    const saved = await store.save(session);
    return { ...publicSession(session), counsellor_contact: counsellorContact || null, intro: INTRO, reply: opening, kind: session.kind, mock: mode === 'mock', notice: notice(), saved };
  }

  function chatHistory(session) {
    return session.turns.map((t) => ({ role: t.role, text: t.text }));
  }

  async function llmReply(session, directives) {
    const contents = chatHistory(session)
      .slice(-14)
      .map((t) => ({ role: t.role === 'user' ? 'user' : 'model', text: t.text }));
    // Gemini wants the conversation to start with a user turn; the fixed opening question is context only.
    while (contents.length && contents[0].role !== 'user') contents.shift();
    const { text } = await gemini.generate({
      system: buildSystemPrompt(session, directives),
      contents,
      temperature: 0.7,
      maxOutputTokens: 300,
    });
    return text.replace(/\s+/g, ' ').trim();
  }

  // `voice` (optional): per-answer data from the voice interview: { prosody, voice_emotion, face, duration_s, skipped }.
  async function message(sessionId, rawText, { source = 'text', voice = null } = {}) {
    const text = String(rawText ?? '').trim();
    if (!text) throw new WellbeingError(400, 'empty_message', 'Send a non-empty "text".');
    if (text.length > MAX_TEXT) throw new WellbeingError(413, 'message_too_long', `Keep messages under ${MAX_TEXT} characters.`);
    const session = await store.get(sessionId);
    if (!session) throw new WellbeingError(404, 'session_not_found', 'No such check-in session.');
    if (busy.has(sessionId)) throw new WellbeingError(409, 'session_busy', 'The previous message is still being processed.');

    // 1. safety regex first, before anything else, and independent of Gemini
    const regex = detectSafety(text);
    if (session.status === 'completed' && regex.level !== 'explicit') {
      throw new WellbeingError(409, 'session_complete', 'This check-in is finished. Start a new one.');
    }

    busy.add(sessionId);
    try {
      // 2. structured analysis (separate call); failures never stop the chat
      const analysisResult = mode === 'gemini'
        ? await analyseMessage({ gemini, history: chatHistory(session), text })
        : { analysis: mockAnalysis(text), ok: true };
      const analysis = analysisResult.analysis || nullAnalysis();
      if (mode === 'gemini' && !analysisResult.ok) session.flags.degraded = true;

      // 3. combine the signals
      const combined = combineSignals(regex.level, analysis.self_harm_signal);
      const now = new Date();
      const lastAssistant = [...session.turns].reverse().find((t) => t.role === 'assistant');
      if (voice) {
        // one record per spoken answer: aggregates only (no audio, no frames)
        (session.answers ||= []).push({
          index: session.answers.length,
          question: lastAssistant?.text || null,
          ts: now,
          duration_s: voice.duration_s ?? voice.prosody?.duration_s ?? null,
          prosody: voice.prosody || null,
          voice_emotion: voice.voice_emotion || null,
          face: voice.face || null,
          text_model: voice.text_model || null,
          skipped: Boolean(voice.skipped),
        });
      }
      session.turns.push({
        role: 'user', text, ts: now, source,
        analysis,
        safety: { level: regex.level, matches: regex.matches, structured: analysis.self_harm_signal, combined },
      });

      let reply;
      let kind;
      let complete = false;
      const n = userTurns(session);

      if (combined === 'explicit') {
        const repeat = session.flags.needs_human_followup;
        reply = safetyMessage({ counsellorContact, repeat });
        kind = 'safety';
        session.tier = 'needs_support_now';
        session.flags.needs_human_followup = true;
        session.flags.possible_self_harm = true;
        session.flags.safety_triggered_at ||= now;
      } else {
        if (combined === 'possible') session.flags.possible_self_harm = true;
        const supportMode = session.flags.needs_human_followup;
        const closing = n >= session.max_turns;
        const topic = closing || supportMode ? null : chooseNextTopic(session);
        if (topic) session.asked_topics.push(topic.key);
        kind = mode === 'gemini' ? 'llm' : 'mock';
        try {
          if (mode !== 'gemini') throw new Error('mock');
          reply = await llmReply(session, { topic, closing, possible: combined === 'possible', supportMode });
          if (replyIsUnsafe(reply) || replyHasDiagnosis(reply)) throw new Error('unsafe reply discarded');
        } catch (err) {
          if (mode === 'gemini') {
            session.flags.degraded = true;
            kind = 'fallback';
          }
          reply = closing ? CLOSING_FALLBACK : supportMode
            ? 'I am still here with you. Would you like to tell me more about how things are right now?'
            : mockReply({ turn: n, topic });
        }
        if (closing) kind = kind === 'llm' ? 'closing' : kind;
        if (combined === 'possible' && !closing) reply = `${reply} ${softSupportLine({ counsellorContact })}`;
      }

      session.turns.push({ role: 'assistant', text: reply, ts: new Date(), kind });

      // finish after the last turn, but never straight after an explicit message: stay supportive
      if (combined !== 'explicit' && (n >= session.max_turns || n >= session.max_turns + EXTRA_TURNS_AFTER_SAFETY)) complete = true;
      if (n >= session.max_turns + EXTRA_TURNS_AFTER_SAFETY) complete = true;
      if (!session.flags.needs_human_followup) session.tier = computeSessionResult(session, { cfg: scoringConfig }).tier;
      if (complete) await finish(session);

      const saved = await store.save(session);
      return {
        reply, kind,
        mock: mode === 'mock',
        notice: notice(),
        safety: { level: combined, regex: regex.level, structured: analysis.self_harm_signal },
        tier: session.tier,
        needs_human_followup: session.flags.needs_human_followup,
        analysis,
        turn: n,
        max_turns: session.max_turns,
        complete,
        final: session.final,
        degraded: session.flags.degraded,
        saved,
      };
    } finally {
      busy.delete(sessionId);
    }
  }

  const userKey = (s) => s.userId || s.anonymousId;

  // Scores the session against the user's earlier finished sessions and stores the result.
  async function finish(session) {
    const earlier = (await store.list(userKey(session)))
      .filter((s) => s.sessionId !== session.sessionId && s.status === 'completed' && s.final && typeof s.final.score === 'number')
      .map((s) => ({ t: new Date(s.completed_at || s.started_at).getTime(), score: s.final.score }))
      .sort((a, b) => a.t - b.t);
    const result = computeSessionResult(session, { prior: earlier, cfg: scoringConfig });
    if ((session.answers || []).length) {
      const userTexts = session.turns.filter((t) => t.role === 'user');
      result.transcript = session.answers.map((a, i) => ({ index: i, question: a.question, answer: userTexts[i]?.text ?? '', face_top: a.face?.dominant ?? null, voice_top: a.voice_emotion?.top_label ?? null, skipped: a.skipped }));
    }
    result.kind = session.kind || 'chat';
    if (result.tier === 'needs_support_now') {
      result.support = {
        helplines: HELPLINES,
        counsellor_contact: counsellorContact || 'your college counselling centre',
      };
    }
    session.final = result;
    session.tier = result.tier;
    session.status = 'completed';
    session.completed_at = new Date();
    return result;
  }

  async function end(sessionId) {
    const session = await store.get(sessionId);
    if (!session) throw new WellbeingError(404, 'session_not_found', 'No such check-in session.');
    if (session.status !== 'completed') {
      await finish(session);
      await store.save(session);
    }
    return { ...publicSession(session), ...session.final, reply: CLOSING_FALLBACK, notice: notice(), mock: mode === 'mock' };
  }

  async function history(key) {
    if (!key) throw new WellbeingError(400, 'missing_user', 'Send your anonymous id in the X-Wellbeing-User header.');
    const sessions = await store.list(key);
    return buildHistory(sessions);
  }

  async function deleteData(key) {
    if (!key) throw new WellbeingError(400, 'missing_user', 'Send your anonymous id in the X-Wellbeing-User header.');
    return { deleted_sessions: await store.removeByUser(key) };
  }

  async function deleteSession(sessionId, key) {
    const session = await store.get(sessionId);
    if (!session) throw new WellbeingError(404, 'session_not_found', 'No such check-in session.');
    if (key && userKey(session) !== key) throw new WellbeingError(403, 'not_your_session', 'This session belongs to someone else.');
    await store.remove(sessionId);
    return { deleted_sessions: 1 };
  }
  // The transcript is only returned to the owner (matching anonymous id), not to anyone who has the session id.
  async function get(sessionId, key) {
    const session = await store.get(sessionId);
    if (!session) throw new WellbeingError(404, 'session_not_found', 'No such check-in session.');
    if (key !== undefined && userKey(session) !== key) throw new WellbeingError(403, 'not_your_session', 'This session belongs to someone else.');
    return { ...publicSession(session), turns_detail: session.turns, modality_samples: session.modality_samples };
  }

  async function addModality(sessionId, sample) {
    const session = await store.get(sessionId);
    if (!session) throw new WellbeingError(404, 'session_not_found', 'No such check-in session.');
    if (!['face', 'audio', 'text'].includes(sample?.modality)) throw new WellbeingError(400, 'bad_modality', 'modality must be face, audio or text.');
    if (session.modality_samples.length >= 500) throw new WellbeingError(413, 'too_many_samples', 'Too many samples for one session.');
    session.modality_samples.push({
      ts: new Date(),
      modality: sample.modality,
      top_label: typeof sample.top_label === 'string' ? sample.top_label.slice(0, 20) : null,
      confidence: Number.isFinite(sample.confidence) ? sample.confidence : null,
      probabilities: sample.probabilities && typeof sample.probabilities === 'object' ? sample.probabilities : null,
    });
    await store.save(session);
    return { ok: true, samples: session.modality_samples.length };
  }

  return {
    mode,
    status: () => ({
      mode,
      mock: mode === 'mock',
      model: mode === 'gemini' ? gemini.model : null,
      fallback_model: mode === 'gemini' ? gemini.models[1] || null : null,
      max_turns: clampTurns,
      counsellor_contact_set: Boolean(counsellorContact),
      persistent: store.persistent(),
      notice: notice(),
    }),
    start, message, end, finish: end, get, addModality, history, deleteData, deleteSession,
  };
}
