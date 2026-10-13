// End-to-end test of /ws/live. Plays 100 real RAVDESS frames like the browser does:
//  - a camera clock ticks every 100 ms;
//  - a tick sends the next frame only if no result is pending, otherwise that frame is DROPPED;
//  - latency = time from send to the matching result.
// Needs the model service (port 8001) and Express server (port 4000) running.
//
// Usage: node scripts/stream-test.mjs [framesDir] [url]
import { readFile, readdir, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const framesDir = process.argv[2] || path.join(here, '.cache', 'frames');
const url = process.argv[3] || process.env.LIVE_URL || 'ws://localhost:4000/ws/live';
const TICK_MS = Number(process.env.TICK_MS || 100);
const WAIT_MS = 8000;

const files = (await readdir(framesDir)).filter((f) => f.endsWith('.jpg')).sort();
if (files.length === 0) throw new Error(`no .jpg frames in ${framesDir}; run make_stream_frames.py first`);
const frames = await Promise.all(files.map((f) => readFile(path.join(framesDir, f))));

const ws = new WebSocket(url);
ws.binaryType = 'arraybuffer';
await new Promise((resolve, reject) => {
  ws.onopen = resolve;
  ws.onerror = () => reject(new Error(`cannot connect to ${url}`));
});

// Wait until the server says the model service is up.
const modelUp = await new Promise((resolve) => {
  const t = setTimeout(() => resolve(false), WAIT_MS);
  ws.addEventListener('message', function onStatus(ev) {
    const m = JSON.parse(ev.data);
    if (m.type === 'status') {
      clearTimeout(t);
      ws.removeEventListener('message', onStatus);
      resolve(m.model_service === 'up');
    }
  });
});
if (!modelUp) throw new Error('model service did not report up');

const results = [];
const errors = [];
let inFlight = false;
let sentAt = 0;
let busyReplies = 0;
let pendingIdx = -1;
ws.addEventListener('message', (ev) => {
  const m = JSON.parse(ev.data);
  if (m.type === 'result') {
    const now = performance.now();
    results.push({
      frame: pendingIdx,
      latencyMs: now - sentAt,
      modelMs: m.ms,
      serverMs: m.server_ms,
      face_found: m.face_found,
      top_label: m.top_label,
      confidence: m.confidence,
    });
    inFlight = false;
  } else if (m.type === 'busy') {
    busyReplies += 1;
  } else if (m.type === 'error') {
    errors.push(m.code);
    inFlight = false;
  }
});

let sent = 0;
let dropped = 0;
const sentFrames = [];
const t0 = performance.now();
await new Promise((resolve) => {
  let i = 0;
  const clock = setInterval(() => {
    if (i >= frames.length) {
      clearInterval(clock);
      resolve();
      return;
    }
    if (inFlight) {
      dropped += 1;
    } else {
      inFlight = true;
      sentAt = performance.now();
      pendingIdx = i;
      sentFrames.push(i);
      sent += 1;
      ws.send(frames[i]);
    }
    i += 1;
  }, TICK_MS);
});
// Let the last in-flight frame finish.
const drainStart = performance.now();
while (inFlight && performance.now() - drainStart < WAIT_MS) await new Promise((r) => setTimeout(r, 10));
const totalMs = performance.now() - t0;
ws.close();

const lat = results.map((r) => r.latencyMs).sort((a, b) => a - b);
const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
const pick = (p) => (lat.length ? lat[Math.min(lat.length - 1, Math.floor((p / 100) * lat.length))] : null);
const labels = {};
for (const r of results) if (r.top_label) labels[r.top_label] = (labels[r.top_label] || 0) + 1;

const report = {
  url,
  source_frames: frames.length,
  source_clip: 'RAVDESS 01-01-05-01-01-01-01.mp4 (Actor 01, emotion code 05 = angry)',
  frame_size_bytes_avg: Math.round(frames.reduce((a, b) => a + b.length, 0) / frames.length),
  tick_ms: TICK_MS,
  sent,
  dropped_by_backpressure: dropped,
  busy_replies: busyReplies,
  results: results.length,
  errors,
  faces_found: results.filter((r) => r.face_found).length,
  wall_clock_s: Math.round(totalMs) / 1000,
  results_per_second: Math.round((results.length / (totalMs / 1000)) * 10) / 10,
  avg_round_trip_ms: mean(lat) == null ? null : Math.round(mean(lat)),
  p50_round_trip_ms: pick(50) == null ? null : Math.round(pick(50)),
  p95_round_trip_ms: pick(95) == null ? null : Math.round(pick(95)),
  avg_model_ms: mean(results.map((r) => r.modelMs)) == null ? null : Math.round(mean(results.map((r) => r.modelMs)) * 10) / 10,
  top_label_counts: labels,
};

const outDir = path.join(here, 'out');
await mkdir(outDir, { recursive: true });
await writeFile(path.join(outDir, 'stream-test.json'), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
if (results.length === 0) process.exitCode = 1;
