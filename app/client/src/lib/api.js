// Thin wrappers over the Express API. Errors carry the server's {error, message} body.
export async function apiJson(path, init) {
  let res;
  try {
    res = await fetch(path, init);
  } catch {
    throw new Error('Cannot reach the server. Is it running on port 4000?');
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.message || `Request failed (${res.status})`);
  return body;
}

export function getHealth() {
  return apiJson('/api/health');
}

export function uploadImage(file) {
  const form = new FormData();
  form.append('file', file);
  return apiJson('/api/predict/image', { method: 'POST', body: form });
}

export function getHistory() {
  return apiJson('/api/history');
}

export function uploadAudio(blob, filename, source = 'upload') {
  const form = new FormData();
  form.append('source', source); // before the file, so it is available to the server
  form.append('file', blob, filename);
  return apiJson('/api/predict/audio', { method: 'POST', body: form });
}
