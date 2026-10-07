import { useEffect, useRef, useState } from "react";
import { api, WS_URL } from "../api";
import { EmotionTimeline } from "../components/Charts";
import { ComingSoon, ErrorBox } from "../components/Notices";
import ResultCard from "../components/ResultCard";

function useRequest(onResult) {
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  async function run(fn) {
    setBusy(true);
    setError(null);
    try {
      const r = await fn();
      setResult(r);
      onResult?.(r);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }
  return { result, error, busy, run, setError };
}

export function AudioTab({ onResult, mocked }) {
  const { result, error, busy, run, setError } = useRequest(onResult);
  const [file, setFile] = useState(null);
  return (
    <div className="space-y-4">
      <ComingSoon modality="audio" mocked={mocked} />
      <div className="card space-y-3">
        <label className="block text-sm font-medium">Speech clip (WAV, WebM, OGG, MP3, M4A)</label>
        <input type="file" accept=".wav,.webm,.ogg,.mp3,.m4a,audio/*" onChange={(e) => setFile(e.target.files[0] ?? null)} className="text-sm" />
        <button className="btn-primary" disabled={!file || busy} onClick={() => run(() => api.predictAudio(file))}>
          {busy ? "Analysing…" : "Analyse audio"}
        </button>
      </div>
      <ErrorBox error={error} onDismiss={() => setError(null)} />
      <ResultCard title="Audio result" result={result} />
    </div>
  );
}

export function TextTab({ onResult, mocked }) {
  const { result, error, busy, run, setError } = useRequest(onResult);
  const [text, setText] = useState("");
  return (
    <div className="space-y-4">
      <ComingSoon modality="text" mocked={mocked} />
      <div className="card space-y-3">
        <label htmlFor="text-input" className="block text-sm font-medium">Text to analyse</label>
        <textarea id="text-input" value={text} onChange={(e) => setText(e.target.value)} rows={5} maxLength={10000}
          className="w-full rounded-lg border border-black/15 bg-transparent p-2 text-sm dark:border-white/15"
          placeholder="Type or paste a few sentences…" />
        <p className="text-xs text-stone-500">Only the prediction is stored, not your text.</p>
        <button className="btn-primary" disabled={!text.trim() || busy} onClick={() => run(() => api.predictText(text))}>
          {busy ? "Analysing…" : "Analyse text"}
        </button>
      </div>
      <ErrorBox error={error} onDismiss={() => setError(null)} />
      <ResultCard title="Text result" result={result} />
    </div>
  );
}

export function EegTab({ onResult, mocked, backendOnline }) {
  const [connected, setConnected] = useState(false);
  const [result, setResult] = useState(null);
  const [points, setPoints] = useState([]);
  const [error, setError] = useState(null);
  const wsRef = useRef(null);
  const startRef = useRef(0);

  function disconnect() {
    const ws = wsRef.current;
    wsRef.current = null;
    if (ws) { ws.onclose = null; ws.close(); }
    setConnected(false);
  }
  useEffect(() => disconnect, []);

  function connect() {
    setError(null);
    setPoints([]);
    startRef.current = performance.now();
    const ws = new WebSocket(`${WS_URL}/eeg/stream`);
    wsRef.current = ws;
    ws.onopen = () => setConnected(true);
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type !== "prediction") return;
      const t = (performance.now() - startRef.current) / 1000;
      setResult(msg);
      onResult?.(msg);
      setPoints((ps) => [...ps.filter((p) => p.t >= t - 60), { t, probs: msg.probs }]);
    };
    ws.onclose = () => {
      if (wsRef.current !== ws) return;
      setError({ code: "backend_offline", message: "The EEG stream disconnected." });
      disconnect();
    };
  }

  return (
    <div className="space-y-4">
      <ComingSoon modality="EEG" mocked={mocked} />
      <div className="card flex flex-wrap items-center gap-3">
        {connected ? (
          <button className="btn-ghost" onClick={disconnect}>Disconnect</button>
        ) : (
          <button className="btn-primary" onClick={connect} disabled={!backendOnline}>Connect to EEG stream</button>
        )}
        <span className="text-xs text-stone-500">
          The IoT EEG sensor isn't connected yet. The stream sends one prediction per second.
        </span>
      </div>
      <ErrorBox error={error} onDismiss={() => setError(null)} />
      <ResultCard title="EEG result" result={result}>
        {points.length > 1 && <EmotionTimeline points={points} height={180} />}
      </ResultCard>
    </div>
  );
}
