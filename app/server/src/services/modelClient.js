import { config } from '../config.js';

// HTTP calls to the Python model service. Every failure becomes a ModelError with a code.
export class ModelError extends Error {
  constructor(code, message, status = 503) {
    super(message);
    this.code = code;
    this.status = status;
  }
}

async function call(path, { timeoutMs, ...init } = {}) {
  let res;
  try {
    res = await fetch(`${config.modelServiceUrl}${path}`, {
      ...init,
      signal: AbortSignal.timeout(timeoutMs || config.upstreamTimeoutMs),
    });
  } catch (err) {
    const timeout = err.name === 'TimeoutError' || err.name === 'AbortError';
    throw new ModelError(
      timeout ? 'model_timeout' : 'model_service_unavailable',
      timeout ? 'The model service did not answer in time.' : 'The model service is not running.',
    );
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = body.detail || {};
    throw new ModelError(detail.error || 'model_error', detail.message || 'Model service error.', res.status);
  }
  return body;
}

export function health() {
  return call('/health');
}

export function predictImage(buffer, filename, mimetype) {
  const form = new FormData();
  form.append('file', new Blob([buffer], { type: mimetype }), filename || 'upload');
  return call('/predict/image', { method: 'POST', body: form });
}

export function predictAudio(buffer, filename, mimetype) {
  const form = new FormData();
  form.append('file', new Blob([buffer], { type: mimetype }), filename || 'audio');
  return call('/predict/audio', { method: 'POST', body: form });
}

export function predictFrames(frames) {
  return call('/predict/frames', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ frames }),
  });
}

// One spoken answer: transcript + prosody + voice emotion (the audio is decoded in memory by the model service).
export function analyseAnswer(buffer, filename, mimetype) {
  const form = new FormData();
  form.append('file', new Blob([buffer], { type: mimetype }), filename || 'answer.webm');
  return call('/analyse/answer', { method: 'POST', body: form, timeoutMs: 120000 }); // CPU transcription can take several seconds
}