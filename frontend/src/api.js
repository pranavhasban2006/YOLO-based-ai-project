export const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");
export const WS_URL = API_URL.replace(/^http/, "ws");

// Must match the backend (and the checkpoints' training order).
export const CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"];

export const LIMITS = { imageMB: 10, videoMB: 100, audioMB: 25 };

export class ApiError extends Error {
  constructor(status, code, message) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

async function request(path, options) {
  let res;
  try {
    res = await fetch(API_URL + path, options);
  } catch {
    throw new ApiError(0, "backend_offline", `Cannot reach the backend at ${API_URL}. Is it running?`);
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    throw new ApiError(res.status, body?.error ?? "http_error", body?.message ?? `Request failed (${res.status}).`);
  }
  return body;
}

function upload(path, file) {
  const form = new FormData();
  form.append("file", file);
  return request(path, { method: "POST", body: form });
}

const json = (body) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request("/health"),
  predictImage: (file) => upload("/predict/image", file),
  predictVideo: (file) => upload("/predict/video", file),
  predictAudio: (file) => upload("/predict/audio", file),
  predictText: (text) => request("/predict/text", json({ text })),
  fuse: (predictions) => request("/predict/fused", json({ predictions })),
  sessions: () => request("/sessions"),
  session: (id) => request(`/sessions/${id}`),
};

/** Strip extra fields so a prediction matches the common shape expected by /predict/fused. */
export function corePrediction(p) {
  const { modality, probs, label, confidence, timestamp, mock } = p;
  return { modality, probs, label, confidence, timestamp, mock };
}
