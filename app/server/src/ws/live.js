import { WebSocket, WebSocketServer } from 'ws';
import { config } from '../config.js';
import { Prediction } from '../models/Prediction.js';
import { dbReady } from '../db.js';

const MODEL_WS = config.modelServiceUrl.replace(/^http/, 'ws') + '/predict/stream';
const MAX_FRAME_BYTES = 1024 * 1024; // a 320 px JPEG is around 10-30 KB

// Browser <-> /ws/live <-> model service /predict/stream.
// Each browser connection gets its own upstream socket. At most one frame is in flight;
// a frame that arrives while one is in flight is answered with {type:"busy"} and dropped.
export function attachLive(server) {
  const wss = new WebSocketServer({ server, path: '/ws/live', maxPayload: MAX_FRAME_BYTES });
  wss.on('connection', (client) => new LiveSession(client));
  return wss;
}

class LiveSession {
  constructor(client) {
    this.client = client;
    this.upstream = null;
    this.inFlight = false;
    this.sentAt = 0;
    this.save = false;
    this.lastSaveAt = 0;
    this.closed = false;
    this.timer = null;

    client.on('message', (data, isBinary) => this.onClientMessage(data, isBinary));
    client.on('close', () => this.close());
    client.on('error', () => this.close());
    this.connectUpstream();
  }

  send(obj) {
    if (this.client.readyState === WebSocket.OPEN) this.client.send(JSON.stringify(obj));
  }

  connectUpstream() {
    if (this.closed) return;
    const up = new WebSocket(MODEL_WS, { maxPayload: 1024 * 1024 });
    this.upstream = up;
    up.on('open', () => this.send({ type: 'status', model_service: 'up' }));
    up.on('message', (raw) => this.onUpstreamMessage(raw));
    up.on('close', () => {
      if (this.upstream === up) this.upstream = null;
      this.inFlight = false;
      this.send({ type: 'status', model_service: 'down' });
      if (!this.closed) setTimeout(() => this.connectUpstream(), 1000);
    });
    up.on('error', () => {}); // close handler reports and retries
  }

  onClientMessage(data, isBinary) {
    if (!isBinary) {
      let msg;
      try {
        msg = JSON.parse(data.toString());
      } catch {
        return this.send({ type: 'error', code: 'bad_json', message: 'Control messages must be JSON.' });
      }
      if (msg.type === 'config') this.save = msg.save === true;
      return;
    }
    if (!this.upstream || this.upstream.readyState !== WebSocket.OPEN) {
      return this.send({ type: 'error', code: 'model_unavailable', message: 'The model service is not running.' });
    }
    if (this.inFlight) {
      return this.send({ type: 'busy' });
    }
    this.inFlight = true;
    this.sentAt = Date.now();
    this.timer = setTimeout(() => {
      // Upstream never answered: release the slot so the browser can keep going.
      if (this.inFlight) {
        this.inFlight = false;
        this.send({ type: 'error', code: 'model_timeout', message: 'The model service did not answer in time.' });
      }
    }, config.upstreamTimeoutMs);
    this.upstream.send(data);
  }

  onUpstreamMessage(raw) {
    clearTimeout(this.timer);
    this.inFlight = false;
    const serverMs = Date.now() - this.sentAt;
    let result;
    try {
      result = JSON.parse(raw.toString());
    } catch {
      return this.send({ type: 'error', code: 'bad_upstream', message: 'Model service sent invalid JSON.' });
    }
    if (result.error) {
      return this.send({ type: 'error', code: result.error, message: result.message });
    }
    this.send({ type: 'result', ...result, server_ms: serverMs, timestamp: new Date().toISOString() });
    this.maybeSave(result);
  }

  maybeSave(result) {
    if (!this.save || !dbReady()) return;
    const now = Date.now();
    if (now - this.lastSaveAt < config.saveIntervalMs) return; // store at most one summary per interval
    this.lastSaveAt = now;
    Prediction.create({
      source: 'webcam',
      face_found: result.face_found,
      probabilities: result.probabilities,
      top_label: result.top_label,
      confidence: result.confidence,
      model: result.model || null,
    }).catch((err) => console.warn('[live] save failed:', err.message));
  }

  close() {
    this.closed = true;
    clearTimeout(this.timer);
    if (this.upstream) this.upstream.close();
  }
}
