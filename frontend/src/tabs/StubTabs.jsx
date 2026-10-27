import { useEffect, useRef, useState } from "react";
import { api, WS_URL } from "../api";
import { EmotionTimeline } from "../components/Charts";
import { AudioRecorder } from "../components/MediaCapture";
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

function Verdict({ title, p, accent }) {
  return (
    <div className={`rounded-lg border px-3 py-2 ${accent ? "border-blue-500" : "border-black/10 dark:border-white/10"}`}>
      <div className="text-xs uppercase tracking-wide text-stone-500">{title}</div>
      <div className="text-lg font-semibold capitalize">{p.label} <span className="text-sm font-normal tabular text-stone-500">{(p.confidence * 100).toFixed(0)}%</span></div>
    </div>
  );
}

/** Speech CNN (A1) -> GRU (A2): A1's answer, then the answer after A2 re-reads A1's output. */
function AudioCascadeCard({ onResult }) {
  const { result, error, busy, run, setError } = useRequest(onResult);
  const [url, setUrl] = useState(null);
  useEffect(() => () => url && URL.revokeObjectURL(url), [url]);
  function analyse(f) {
    if (!f) return;
    setUrl(URL.createObjectURL(f));
    run(() => api.predictAudioCascade(f));
  }
  return (
    <div className="card space-y-3">
      <div>
        <h3 className="font-semibold">Two-stage model: CNN (A1) → GRU (A2)</h3>
        <p className="text-xs text-stone-500">
          Stage 1: the CNN alone reads the spectrogram and answers. Stage 2: the CNN's output is fed to the GRU, which
          keeps or corrects that answer. Test-set numbers for every hybrid are in the "A1 → A2 Results" tab.
        </p>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <AudioRecorder onRecorded={analyse} disabled={busy} />
        <label className="btn-ghost cursor-pointer text-sm">
          or choose a file
          <input type="file" className="hidden" accept=".wav,.webm,.ogg,.mp3,.m4a,.mp4,audio/*"
            onChange={(e) => { analyse(e.target.files[0]); e.target.value = ""; }} />
        </label>
      </div>
      {url && <audio src={url} controls className="w-full max-w-md" />}
      {busy && <p className="text-sm text-stone-500" role="status">Running CNN → GRU…</p>}
      <ErrorBox error={error} onDismiss={() => setError(null)} />
      {result && (
        <div className="grid items-center gap-2 sm:grid-cols-[1fr_auto_1fr]">
          <Verdict title="Stage 1 · CNN alone (A1)" p={result.a1} />
          <span className="text-center text-stone-400">→</span>
          <Verdict title="Stage 2 · after GRU (A2)" p={result} accent />
          <p className="text-sm sm:col-span-3">
            {result.changed
              ? <>The GRU <b>changed</b> the CNN's answer from <b>{result.a1.label}</b> to <b>{result.label}</b>.</>
              : <>The GRU <b>confirmed</b> the CNN's answer ({result.label}).</>}
          </p>
        </div>
      )}
    </div>
  );
}

export function AudioTab({ onResult, mocked }) {
  const { result, error, busy, run, setError } = useRequest(onResult);
  const [file, setFile] = useState(null);
  const [recUrl, setRecUrl] = useState(null);
  useEffect(() => () => recUrl && URL.revokeObjectURL(recUrl), [recUrl]);
  function analyseRecording(f) {
    setFile(f);
    setRecUrl(URL.createObjectURL(f));
    run(() => api.predictAudio(f));
  }
  return (
    <div className="space-y-4">
      <ComingSoon modality="audio" mocked={mocked} />
      <div className="card space-y-3">
        <label className="block text-sm font-medium">Record your voice</label>
        <AudioRecorder onRecorded={analyseRecording} disabled={busy} />
        {recUrl && <audio src={recUrl} controls className="w-full max-w-md" />}
      </div>
      <div className="card space-y-3">
        <label className="block text-sm font-medium">Speech clip (WAV, WebM, OGG, MP3, M4A, MP4)</label>
        <input type="file" accept=".wav,.webm,.ogg,.mp3,.m4a,.mp4,audio/*" onChange={(e) => setFile(e.target.files[0] ?? null)} className="text-sm" />
        <button className="btn-primary" disabled={!file || busy} onClick={() => run(() => api.predictAudio(file))}>
          {busy ? "Analysing…" : "Analyse audio"}
        </button>
      </div>
      <ErrorBox error={error} onDismiss={() => setError(null)} />
      <ResultCard title="Audio result" result={result} />
      <AudioCascadeCard onResult={onResult} />
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
