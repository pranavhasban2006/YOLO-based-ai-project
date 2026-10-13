import { useEffect, useRef, useState } from 'react';
import { EMA_ALPHA, SERIES_WINDOW_MS, emaUpdate, topOf } from '../lib/emotion.js';

const CAPTURE_MS = 100; // the camera "produces" a frame every 100 ms
const FRAME_WIDTH = 320; // width of the JPEG sent to the server
const JPEG_QUALITY = 0.7;
const RECONNECT_MIN_MS = 500;
const RECONNECT_MAX_MS = 4000;
const STATS_WINDOW_MS = 2000;

// Streams webcam frames to /ws/live while `enabled`. Only one frame is in flight: a camera frame
// that comes due while the previous result is still pending is dropped, not queued.
export function useLiveStream({ videoRef, enabled, save }) {
  const [link, setLink] = useState('idle'); // idle | connecting | open | reconnecting
  const [modelUp, setModelUp] = useState(null); // null until the server reports
  const [result, setResult] = useState(null); // latest raw result from the model
  const [smoothed, setSmoothed] = useState(null); // EMA probabilities, null when no face
  const [series, setSeries] = useState([]); // {t, probs} for the last 30 s
  const [stats, setStats] = useState({ fps: 0, latencyMs: null, dropped: 0, sent: 0 });
  const [error, setError] = useState('');

  const wsRef = useRef(null);
  const inFlightRef = useRef(false);
  const sentAtRef = useRef(0);
  const emaRef = useRef(null);
  const resultTimesRef = useRef([]);
  const countersRef = useRef({ dropped: 0, sent: 0 });
  const saveRef = useRef(save);
  saveRef.current = save;

  useEffect(() => {
    if (!enabled) {
      setLink('idle');
      return undefined;
    }

    let closedByUs = false;
    let retryTimer = null;
    let delay = RECONNECT_MIN_MS;
    const canvas = document.createElement('canvas');
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${window.location.host}/ws/live`;

    // Fresh counters for this run; the previous run's values are cleared.
    emaRef.current = null;
    resultTimesRef.current = [];
    countersRef.current = { dropped: 0, sent: 0 };
    setResult(null);
    setSmoothed(null);
    setSeries([]);
    setStats({ fps: 0, latencyMs: null, dropped: 0, sent: 0 });
    setError('');

    const onResult = (msg) => {
      const now = Date.now();
      const latencyMs = now - sentAtRef.current;
      resultTimesRef.current = [...resultTimesRef.current, now].filter((t) => now - t <= STATS_WINDOW_MS);
      setResult({ ...msg, clientLatency: latencyMs });
      if (msg.face_found) {
        emaRef.current = emaUpdate(emaRef.current, msg.probabilities, EMA_ALPHA);
        const s = emaRef.current;
        setSmoothed(s);
        setSeries((prev) => [...prev, { t: now, probs: s }].filter((p) => now - p.t <= SERIES_WINDOW_MS));
      } else {
        emaRef.current = null;
        setSmoothed(null);
      }
      setStats({
        fps: Math.round((resultTimesRef.current.length / (STATS_WINDOW_MS / 1000)) * 10) / 10,
        latencyMs,
        dropped: countersRef.current.dropped,
        sent: countersRef.current.sent,
      });
    };

    const connect = () => {
      setLink((prev) => (prev === 'open' || prev === 'reconnecting' ? 'reconnecting' : 'connecting'));
      const ws = new WebSocket(url);
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;

      ws.onopen = () => {
        delay = RECONNECT_MIN_MS;
        setLink('open');
        setError('');
        ws.send(JSON.stringify({ type: 'config', save: saveRef.current }));
      };

      ws.onmessage = (ev) => {
        let msg;
        try {
          msg = JSON.parse(ev.data);
        } catch {
          return;
        }
        if (msg.type === 'status') {
          setModelUp(msg.model_service === 'up');
          inFlightRef.current = false;
        } else if (msg.type === 'result') {
          inFlightRef.current = false;
          onResult(msg);
        } else if (msg.type === 'busy') {
          countersRef.current.dropped += 1;
        } else if (msg.type === 'error') {
          inFlightRef.current = false;
          setError(msg.message || msg.code);
          if (msg.code === 'model_unavailable') setModelUp(false);
        }
      };

      ws.onclose = () => {
        wsRef.current = null;
        inFlightRef.current = false;
        if (closedByUs) return;
        setLink('reconnecting');
        setModelUp(null);
        retryTimer = setTimeout(connect, delay);
        delay = Math.min(delay * 2, RECONNECT_MAX_MS);
      };
    };

    connect();

    // Frame clock: every CAPTURE_MS, grab the camera if the previous result is back.
    const capture = setInterval(() => {
      const ws = wsRef.current;
      const video = videoRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN || !video || video.readyState < 2 || !video.videoWidth) return;
      if (inFlightRef.current) {
        countersRef.current.dropped += 1;
        return;
      }
      const h = Math.round((FRAME_WIDTH * video.videoHeight) / video.videoWidth);
      canvas.width = FRAME_WIDTH;
      canvas.height = h;
      canvas.getContext('2d').drawImage(video, 0, 0, FRAME_WIDTH, h);
      inFlightRef.current = true;
      sentAtRef.current = Date.now();
      countersRef.current.sent += 1;
      canvas.toBlob(
        (blob) => {
          if (blob && wsRef.current === ws && ws.readyState === WebSocket.OPEN) ws.send(blob);
          else inFlightRef.current = false;
        },
        'image/jpeg',
        JPEG_QUALITY,
      );
    }, CAPTURE_MS);

    return () => {
      closedByUs = true;
      clearInterval(capture);
      clearTimeout(retryTimer);
      wsRef.current?.close();
      wsRef.current = null;
      inFlightRef.current = false;
    };
  }, [enabled, videoRef]);

  // Changing "Save session" while live tells the server straight away.
  useEffect(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'config', save }));
  }, [save]);

  const top = smoothed ? topOf(smoothed) : null;
  return { link, modelUp, result, smoothed, top, series, stats, error };
}
