// Express API tests. They run without MongoDB (history must say so) and without the model service
// (health must say so), so they check the error paths. The end-to-end run is scripts/stream-test.mjs.
import assert from 'node:assert/strict';
import { after, before, describe, it } from 'node:test';
import { once } from 'node:events';
import http from 'node:http';
import { createApp } from '../src/app.js';

let server;
let base;

before(async () => {
  server = http.createServer(createApp());
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  base = `http://127.0.0.1:${server.address().port}`;
});

after(() => server?.close());

describe('upload validation', () => {
  it('rejects a request without a file', async () => {
    const res = await fetch(`${base}/api/predict/image`, { method: 'POST', body: new FormData() });
    assert.equal(res.status, 400);
    assert.equal((await res.json()).error, 'missing_file');
  });

  it('rejects a non-image file type with 415', async () => {
    const form = new FormData();
    form.append('file', new Blob(['hello'], { type: 'text/plain' }), 'x.txt');
    const res = await fetch(`${base}/api/predict/image`, { method: 'POST', body: form });
    assert.equal(res.status, 415);
    assert.equal((await res.json()).error, 'unsupported_file_type');
  });

  it('returns 413 for an image over the size limit', async () => {
    const form = new FormData();
    form.append('file', new Blob([new Uint8Array(11 * 1024 * 1024)], { type: 'image/jpeg' }), 'big.jpg');
    const res = await fetch(`${base}/api/predict/image`, { method: 'POST', body: form });
    assert.equal(res.status, 413);
    assert.equal((await res.json()).error, 'file_too_large');
  });

  it('returns a clean 503 when the model service is down', async () => {
    const form = new FormData();
    form.append('file', new Blob([new Uint8Array([0xff, 0xd8])], { type: 'image/jpeg' }), 'a.jpg');
    const res = await fetch(`${base}/api/predict/image`, { method: 'POST', body: form });
    const body = await res.json();
    if (body.error === 'model_service_unavailable') {
      assert.equal(res.status, 503);
    } else {
      // Model service is running in this environment: it should reject the fake JPEG with 400.
      assert.equal(res.status, 400);
      assert.equal(body.error, 'invalid_image');
    }
  });
});

describe('frames endpoint', () => {
  it('requires a non-empty frames array', async () => {
    const res = await fetch(`${base}/api/predict/frames`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frames: [] }),
    });
    assert.equal(res.status, 400);
    assert.equal((await res.json()).error, 'bad_frames');
  });

  it('rejects malformed JSON with 400', async () => {
    const res = await fetch(`${base}/api/predict/frames`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{not json',
    });
    assert.equal(res.status, 400);
    assert.equal((await res.json()).error, 'bad_json');
  });
});

describe('history and health', () => {
  it('history says the database is unavailable when not connected', async () => {
    const res = await fetch(`${base}/api/history`);
    assert.equal(res.status, 503);
    assert.equal((await res.json()).error, 'db_unavailable');
  });

  it('health always answers and reports the model service state', async () => {
    const res = await fetch(`${base}/api/health`);
    assert.equal(res.status, 200);
    const body = await res.json();
    assert.equal(typeof body.modelService.reachable, 'boolean');
    assert.ok('connected' in body.db);
  });

  it('unknown API routes return JSON 404', async () => {
    const res = await fetch(`${base}/api/nope`);
    assert.equal(res.status, 404);
    assert.equal((await res.json()).error, 'not_found');
  });
});

describe('audio endpoint', () => {
  it('rejects a request without a file', async () => {
    const res = await fetch(`${base}/api/predict/audio`, { method: 'POST', body: new FormData() });
    assert.equal(res.status, 400);
    assert.equal((await res.json()).error, 'missing_file');
  });

  it('rejects a non-audio file type with 415', async () => {
    const form = new FormData();
    form.append('file', new Blob(['hello'], { type: 'text/plain' }), 'x.txt');
    const res = await fetch(`${base}/api/predict/audio`, { method: 'POST', body: form });
    assert.equal(res.status, 415);
    assert.equal((await res.json()).error, 'unsupported_file_type');
  });

  it('returns 413 for audio over the size limit', async () => {
    const form = new FormData();
    form.append('file', new Blob([new Uint8Array(11 * 1024 * 1024)], { type: 'audio/wav' }), 'big.wav');
    const res = await fetch(`${base}/api/predict/audio`, { method: 'POST', body: form });
    assert.equal(res.status, 413);
    assert.equal((await res.json()).error, 'file_too_large');
  });
});
