import { useEffect, useRef, useState } from 'react';
import { uploadAudio } from '../lib/api.js';
import { Mic, Upload } from 'lucide-react';
import { COLORS, EMOJI, LOW_CONFIDENCE, pct } from '../lib/emotion.js';
import { useToast } from '../lib/toast.jsx';
import ProbBars from './ProbBars.jsx';
import { PageHeader, Skeleton } from './ui.jsx';

const ACCEPT = 'audio/wav,audio/x-wav,audio/webm,audio/ogg,audio/mpeg,audio/mp4,audio/flac,.wav,.webm,.ogg,.mp3,.m4a,.flac';
const RECORD_SECONDS = 4;

// Re-encode a recorded clip as 16-bit mono WAV in the browser, so the server does not need ffmpeg.
async function toWav(blob) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  try {
    const buf = await ctx.decodeAudioData(await blob.arrayBuffer());
    const ch = buf.getChannelData(0);
    const out = new DataView(new ArrayBuffer(44 + ch.length * 2));
    const str = (o, s) => [...s].forEach((c, i) => out.setUint8(o + i, c.charCodeAt(0)));
    str(0, 'RIFF');
    out.setUint32(4, 36 + ch.length * 2, true);
    str(8, 'WAVEfmt ');
    out.setUint32(16, 16, true);
    out.setUint16(20, 1, true);
    out.setUint16(22, 1, true);
    out.setUint32(24, buf.sampleRate, true);
    out.setUint32(28, buf.sampleRate * 2, true);
    out.setUint16(32, 2, true);
    out.setUint16(34, 16, true);
    str(36, 'data');
    out.setUint32(40, ch.length * 2, true);
    ch.forEach((v, i) => out.setInt16(44 + i * 2, Math.max(-1, Math.min(1, v)) * 0x7fff, true));
    return new Blob([out], { type: 'audio/wav' });
  } finally {
    ctx.close();
  }
}

// Speech emotion: upload a clip or record 4 seconds. Shows the top label, confidence and 7 probability bars.
export default function AudioTab() {
  const [result, setResult] = useState(null);
  const toast = useToast();
  const setError = (m) => m && toast.error(m);
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(false);
  const [left, setLeft] = useState(0);
  const [dragOver, setDragOver] = useState(false);
  const [audioUrl, setAudioUrl] = useState('');
  const inputRef = useRef(null);
  const recRef = useRef(null);

  useEffect(
    () => () => {
      recRef.current?.stream?.getTracks().forEach((t) => t.stop());
    },
    [],
  );
  useEffect(() => () => audioUrl && URL.revokeObjectURL(audioUrl), [audioUrl]);

  const send = async (blob, name, source) => {
    setError('');
    setResult(null);
    setLoading(true);
    setAudioUrl(URL.createObjectURL(blob));
    try {
      setResult(await uploadAudio(blob, name, source));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const record = async () => {
    setError('');
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      setError('This browser cannot record audio. Upload a WAV file instead.');
      return;
    }
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setError('Microphone access was denied or no microphone was found. You can upload a file instead.');
      return;
    }
    const rec = new MediaRecorder(stream);
    recRef.current = rec;
    const chunks = [];
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    rec.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      setRecording(false);
      const raw = new Blob(chunks, { type: rec.mimeType || 'audio/webm' });
      try {
        await send(await toWav(raw), 'recording.wav', 'record');
      } catch {
        await send(raw, 'recording.webm', 'record'); // server decodes with ffmpeg if it has it
      }
    };
    rec.start();
    setRecording(true);
    setLeft(RECORD_SECONDS);
    const tick = setInterval(() => setLeft((s) => Math.max(0, s - 1)), 1000);
    setTimeout(() => {
      clearInterval(tick);
      if (rec.state !== 'inactive') rec.stop();
    }, RECORD_SECONDS * 1000);
  };

  const choose = (f) => f && send(f, f.name, 'upload');
  const confident = result && result.confidence >= LOW_CONFIDENCE;

  return (
    <div>
      <PageHeader title="Speech emotion" subtitle="Upload a short speech clip or record 4 seconds. Experimental: tone of voice is a weak signal." />
      <div className="grid gap-5 lg:grid-cols-2">
        <section className="space-y-4">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              choose(e.dataTransfer.files?.[0]);
            }}
            onClick={() => inputRef.current?.click()}
            onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && inputRef.current?.click()}
            role="button"
            tabIndex={0}
            aria-label="Choose a speech clip to analyse"
            className={`glass flex min-h-40 cursor-pointer flex-col items-center justify-center gap-2 rounded-3xl border-2 border-dashed p-6 text-center transition ${
              dragOver ? 'border-accent bg-accent-soft' : 'border-line hover:border-accent'
            }`}
          >
            <Upload className="h-8 w-8 text-accent" aria-hidden />
            <p className="font-medium">Drag a speech clip here, or click to choose one</p>
            <p className="text-xs text-muted">WAV works everywhere; other formats need ffmpeg on the server. Up to 10 MB.</p>
            <input ref={inputRef} type="file" accept={ACCEPT} className="hidden" onChange={(e) => choose(e.target.files?.[0])} />
          </div>

          <button type="button" onClick={record} disabled={recording || loading} className="btn-primary w-full">
            <Mic className="h-4 w-4" aria-hidden />
            {recording ? `Recording… ${left}s` : `Record ${RECORD_SECONDS} seconds`}
          </button>
          {audioUrl && <audio controls src={audioUrl} className="w-full" />}
        </section>

        <section className="card space-y-4">
          {loading && (
            <div className="space-y-3" aria-busy="true">
              <Skeleton className="h-8 w-1/2" />
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-5/6" />
            </div>
          )}
          {!result && !loading && <p className="text-sm text-muted">Results appear here.</p>}
          {result && (
            <>
              <div className="flex items-center gap-4">
                <span className="text-6xl" aria-hidden>{confident ? EMOJI[result.top_label] : '🤔'}</span>
                <div>
                  <div className="eyebrow">Top label</div>
                  <div className="text-3xl font-semibold capitalize" style={{ color: confident ? COLORS[result.top_label] : 'var(--muted)' }}>
                    {confident ? result.top_label : 'uncertain'}
                  </div>
                  <div className="tabular-nums text-muted">{pct(result.confidence)}</div>
                </div>
              </div>
              {!confident && <p className="text-xs text-muted">Top probability is below 50% (it leans {result.top_label}), so no confident label is shown.</p>}
              <ProbBars probs={result.probabilities} highlight={result.top_label} />
              <p className="text-xs text-muted">
                {result.saved ? 'Saved to the detection log.' : 'Not saved: the database is not connected.'} Model: {result.model}, {result.duration_s} s clip, {result.ms} ms.
              </p>
            </>
          )}
          <p className="rounded-2xl bg-violet-soft p-3 text-xs">
            Experimental. Trained on four acted corpora (English and one German). On a corpus it never saw, macro-F1 is only 0.27 to 0.47 (chance 0.14), so expect many wrong answers on everyday speech.
          </p>
        </section>
      </div>
    </div>
  );
}