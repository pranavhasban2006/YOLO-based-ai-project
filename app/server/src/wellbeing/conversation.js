// Conversation design: system prompt, PHQ-9 topic weaving, canned MOCK replies, final scoring.
import { PHQ_KEYS } from './analysis.js';

export const OPENING = 'How are you feeling today?';
export const VOICE_OPENING = 'How has your day been?'; // the first spoken answer is the person's own baseline
export const INTRO =
  "Hi, I'm an automated check-in assistant, not a doctor or counsellor. This is a short, friendly chat about how you've been. You can stop any time.";

// Order in which topics are woven in. Each is asked at most once. `hint` is guidance for the model, not a script.
export const TOPICS = [
  { key: 'mood', label: 'mood', hint: 'how their mood has been over the last couple of weeks, whether they have felt down or low' },
  { key: 'sleep', label: 'sleep', hint: 'how they have been sleeping lately (trouble falling asleep, waking up, or sleeping too much)' },
  { key: 'interest', label: 'interest', hint: 'whether they still enjoy the things they usually like, such as hobbies, friends or classes' },
  { key: 'energy', label: 'energy', hint: 'how their energy levels have been through the day' },
  { key: 'appetite', label: 'appetite', hint: 'how their appetite and eating have been lately' },
  { key: 'concentration', label: 'concentration', hint: 'whether it has been hard to focus on studies or reading' },
  { key: 'self_worth', label: 'self-worth', hint: 'how they have been feeling about themselves, for example being hard on themselves or feeling like a letdown' },
  { key: 'restlessness', label: 'restlessness', hint: 'whether they have felt restless, on edge, or the opposite, slowed down' },
  {
    key: 'self_harm',
    label: 'self-harm',
    hint:
      'whether, when things have been this hard, they have had any thoughts of hurting themselves or that life is not worth it. Ask it once, gently and directly, with care. Do not ask about methods',
  },
];

const MOCK_QUESTIONS = {
  mood: 'Over the last couple of weeks, has your mood been mostly okay, or have there been low days?',
  sleep: 'How has your sleep been lately?',
  interest: 'Do you still enjoy the things you usually like doing?',
  energy: 'How are your energy levels through the day?',
  appetite: 'How has your appetite been? Are you eating as usual?',
  concentration: 'Is it easy or hard to focus on your studies right now?',
  self_worth: 'How have you been feeling about yourself recently?',
  restlessness: 'Have you felt restless or on edge lately, or the opposite, slowed down?',
  self_harm:
    'When things feel this heavy, do you ever have thoughts of hurting yourself or that life is not worth it? You can tell me honestly.',
};
const MOCK_ACKS = ['Thanks for sharing that.', 'I hear you.', 'That makes sense.', 'Thank you for telling me.', 'I appreciate you being open.'];
const MOCK_GENERIC = ['Is there anything else on your mind that you would like to talk about?', 'How are things with friends or people around you?'];

export const CLOSING_FALLBACK =
  "Thank you for taking the time to check in. Whatever today looks like, it's okay to talk to a friend, family member or your college counsellor about how you've been feeling. Take care of yourself.";

export function mockReply({ turn, topic }) {
  const ack = MOCK_ACKS[turn % MOCK_ACKS.length];
  const q = topic ? MOCK_QUESTIONS[topic.key] : MOCK_GENERIC[turn % MOCK_GENERIC.length];
  return `${ack} ${q}`;
}

export function userTurns(session) {
  return session.turns.filter((t) => t.role === 'user').length;
}

export function addressedKeys(session) {
  const keys = new Set();
  for (const t of session.turns) {
    const items = t.analysis?.phq_items;
    if (items) for (const k of PHQ_KEYS) if (items[k] !== null && items[k] !== undefined) keys.add(k);
  }
  return keys;
}

function distressed(session) {
  const moods = session.turns.map((t) => t.analysis?.mood_1_10).filter((m) => typeof m === 'number');
  const meanMood = moods.length ? moods.reduce((a, b) => a + b, 0) / moods.length : null;
  const items = session.turns.flatMap((t) => Object.entries(t.analysis?.phq_items || {}).filter(([k, v]) => k !== 'self_harm' && v !== null));
  return (meanMood !== null && meanMood <= 6) || items.some(([, v]) => v >= 2) || session.flags.possible_self_harm;
}

// Next topic to weave into the reply, or null. Each topic is asked at most once and is skipped when the user
// has already talked about it. The self-harm question is only asked after 4 turns, and only if there are signs of
// distress, and never once the session is already in the "needs support now" tier.
export function chooseNextTopic(session) {
  const done = addressedKeys(session);
  for (const k of session.asked_topics) done.add(k);
  for (const t of TOPICS) {
    if (done.has(t.key)) continue;
    if (t.key === 'self_harm') {
      if (session.flags.needs_human_followup) continue;
      if (userTurns(session) < 4 || !distressed(session)) continue;
    }
    return t;
  }
  return null;
}

export const SYSTEM_PROMPT = `You are a warm, caring check-in companion for college students, talking in a short wellbeing chat.
Rules:
- Be warm, natural and brief: 1 to 3 short sentences. Ask ONE question at a time. Plain, everyday English; mirror the student's tone.
- Acknowledge what they just said before asking anything. Follow up on details they mention (college, sleep, friends, energy, worries).
- You are not a doctor or therapist. Never diagnose, never name a condition, never make medical claims, never give a score.
- Never give methods or details about suicide or self-harm, even if asked. Never promise secrecy or confidentiality; if asked, say honestly that you can't promise that.
- If they seem to be in distress, encourage them to talk to someone they trust, the college counsellor or a professional. Do not lecture.
- Do not repeat questions the student already answered. Do not list several questions. Do not use emojis or bullet points.
- Take figures of speech like "this exam is killing me" as stress, and do not repeat them literally back.
- Stay on the wellbeing check-in. Politely steer back if asked for unrelated things.`;

export function buildSystemPrompt(session, { topic, closing, possible, supportMode }) {
  const parts = [SYSTEM_PROMPT];
  if (session.kind === 'voice') {
    parts.push('\nThis is a SPOKEN conversation: your reply will be read aloud. Use at most two short, plain sentences, no lists, no symbols, and end with one gentle question unless this is the closing.');
  }
  const n = userTurns(session);
  parts.push(`\nThis is message ${n} of ${session.max_turns} from the student.`);
  if (closing) {
    parts.push(
      '\nThis is the end of the check-in. Thank them warmly, reflect back one or two things they shared in a sentence, and gently encourage talking to a trusted person or the college counsellor if things feel heavy. Do NOT ask a question.',
    );
  } else if (supportMode) {
    parts.push(
      '\nThe student earlier shared thoughts of self-harm or hopelessness and was given helpline information. Stay gentle and present. Do not ask more screening questions, do not bring up new topics, and do not mention self-harm methods. Respond to what they say, and softly remind them they can contact the college counsellor or a trusted person.',
    );
  } else {
    if (possible) {
      parts.push(
        '\nThe student may be hinting at hopelessness. Respond with extra warmth, ask gently how they are coping, and mention that talking to someone they trust or the college counsellor can help. Do not ask about methods.',
      );
    }
    if (topic) {
      parts.push(
        `\nAfter acknowledging what they said, gently weave in this topic as your one question: ${topic.hint}. Keep it natural and connected to what they said, in your own words.`,
      );
    } else {
      parts.push('\nFollow up naturally on something they said. Do not introduce a new health topic.');
    }
  }
  return parts.join('');
}
