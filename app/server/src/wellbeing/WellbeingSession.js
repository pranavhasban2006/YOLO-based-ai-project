import mongoose from 'mongoose';

// One check-in conversation. The structured analysis is stored per turn. Camera or audio samples are optional
// summaries (label and probabilities), never images or recordings.
const { Mixed } = mongoose.Schema.Types;

const turnSchema = new mongoose.Schema(
  {
    role: { type: String, enum: ['user', 'assistant'], required: true },
    text: { type: String, required: true },
    ts: { type: Date, default: Date.now },
    kind: { type: String, default: null }, // assistant: opening | llm | mock | safety | closing | fallback
    source: { type: String, default: null }, // user: text | voice
    analysis: { type: Mixed, default: undefined },
    safety: { type: Mixed, default: undefined }, // { level, matches, structured, combined }
  },
  { _id: false },
);

const modalitySchema = new mongoose.Schema(
  {
    ts: { type: Date, default: Date.now },
    modality: { type: String, enum: ['face', 'audio', 'text'], required: true },
    top_label: String,
    confidence: Number,
    probabilities: { type: Mixed },
  },
  { _id: false },
);

const wellbeingSessionSchema = new mongoose.Schema(
  {
    sessionId: { type: String, required: true, unique: true, index: true },
    userId: { type: String, default: null },
    anonymousId: { type: String, default: null, index: true },
    mode: { type: String, enum: ['gemini', 'mock'], required: true },
    model: { type: String, default: null },
    status: { type: String, enum: ['active', 'completed'], default: 'active' },
    max_turns: { type: Number, default: 8 },
    turns: { type: [turnSchema], default: [] },
    modality_samples: { type: [modalitySchema], default: [] },
    asked_topics: { type: [String], default: [] },
    kind: { type: String, enum: ['chat', 'voice'], default: 'chat' },
    // voice interview: one aggregate record per spoken answer (prosody, voice emotion, face probabilities). No audio or video.
    answers: { type: [Mixed], default: [] },
    tier: { type: String, enum: ['doing_okay', 'struggling', 'needs_support_now', null], default: null },
    flags: {
      needs_human_followup: { type: Boolean, default: false },
      possible_self_harm: { type: Boolean, default: false },
      safety_triggered_at: { type: Date, default: null },
      degraded: { type: Boolean, default: false }, // the model failed at least once and canned text was used
    },
    final: { type: Mixed, default: null },
    started_at: { type: Date, default: Date.now },
    completed_at: { type: Date, default: null },
  },
  { versionKey: false, minimize: false },
);

// Sessions (including chat text) are deleted automatically after WELLBEING_RETENTION_DAYS (default 90).
const RETENTION_DAYS = Number(process.env.WELLBEING_RETENTION_DAYS) > 0 ? Number(process.env.WELLBEING_RETENTION_DAYS) : 90;
wellbeingSessionSchema.index({ started_at: 1 }, { expireAfterSeconds: RETENTION_DAYS * 86400 });

export const WellbeingSession = mongoose.models.WellbeingSession || mongoose.model('WellbeingSession', wellbeingSessionSchema);
