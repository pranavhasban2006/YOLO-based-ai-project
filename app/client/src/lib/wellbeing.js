// API wrapper for /api/wellbeing. A random anonymous id (kept in localStorage) tells the server whose
// sessions to list or delete. It is not a login and carries no personal data.
const KEY = 'wb_user';

export function userId() {
  try {
    let id = localStorage.getItem(KEY);
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem(KEY, id);
    }
    return id;
  } catch {
    return 'anonymous-no-storage';
  }
}

async function call(path, { method = 'GET', body } = {}) {
  let res;
  try {
    res = await fetch(`/api/wellbeing${path}`, {
      method,
      headers: { 'Content-Type': 'application/json', 'X-Wellbeing-User': userId() },
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new Error('Cannot reach the server. Is it running on port 4000?');
  }
  const json = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(json.message || `Request failed (${res.status})`);
  return json;
}

export const wellbeing = {
  status: () => call('/status'),
  start: (kind = 'chat') => call('/session', { method: 'POST', body: { kind } }),
  skip: (id) => call(`/session/${id}/skip`, { method: 'POST' }),
  // One spoken answer. Only the audio clip (analysed locally, then discarded) and aggregated face numbers are sent.
  voiceAnswer: async (id, blob, { face, stt = 'local', browserText = '' } = {}) => {
    const form = new FormData();
    form.append('audio', blob, 'answer.webm');
    form.append('stt', stt);
    if (face) form.append('face', JSON.stringify(face));
    if (browserText) form.append('browser_text', browserText);
    let res;
    try {
      res = await fetch(`/api/wellbeing/session/${id}/voice-answer`, { method: 'POST', headers: { 'X-Wellbeing-User': userId() }, body: form });
    } catch {
      throw new Error('Cannot reach the server. Is it running on port 4000?');
    }
    const json = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(json.message || `Request failed (${res.status})`);
    return json;
  },
  tts: async (text) => {
    const res = await fetch('/api/wellbeing/tts', { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Wellbeing-User': userId() }, body: JSON.stringify({ text }) });
    if (!res.ok) throw new Error('natural voice unavailable');
    return URL.createObjectURL(await res.blob());
  },
  message: (id, text, source = 'text') => call(`/session/${id}/message`, { method: 'POST', body: { text, source } }),
  signal: (id, sample) => call(`/session/${id}/signal`, { method: 'POST', body: sample }),
  finish: (id) => call(`/session/${id}/finish`, { method: 'POST' }),
  history: () => call('/history'),
  deleteMyData: () => call('/data', { method: 'DELETE' }),
};

export const TIER = {
  doing_okay: { label: 'Doing okay', color: 'accent', tone: 'text-accent', bg: 'bg-accent-soft', hex: 'accent' },
  struggling: { label: 'Having a harder time', color: 'violet', tone: 'text-violet', bg: 'bg-violet-soft', hex: 'violet' },
  needs_support_now: { label: 'Support would help now', color: 'coral', tone: 'text-coral', bg: 'bg-coral-soft', hex: 'coral' },
};

export const tierMessage = {
  doing_okay: 'Things look fairly steady right now. Keep doing what helps you, and check in again in a few days.',
  struggling: 'This week seems to have been heavier than usual. That is common, and small steps and talking to someone can help.',
  needs_support_now: 'It sounds like things are hard right now. You deserve support, and talking to a real person soon would be a good next step.',
};

export const HELPLINES = [
  { name: 'Tele-MANAS (free, 24x7)', number: '14416' },
  { name: 'AASRA', number: '9820466726' },
  { name: 'Emergency services', number: '112' },
];
