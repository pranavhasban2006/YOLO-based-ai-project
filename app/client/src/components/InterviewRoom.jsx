import { AnimatePresence, motion } from 'framer-motion';
import { ArrowLeftRight, Check, Loader2, Mic, MicOff, Settings, ShieldCheck, SkipForward, Square } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useCamera } from '../hooks/useCamera.js';
import { useLiveStream } from '../hooks/useLiveStream.js';
import { COLORS, EMOJI, displayLabel } from '../lib/emotion.js';
import { faceAggregate } from '../lib/interview.js';
import { createSpeaker } from '../lib/speech.js';
import { useToast } from '../lib/toast.jsx';
import { createVad, rms } from '../lib/vad.js';
import { wellbeing } from '../lib/wellbeing.js';
import FaceOverlay from './FaceOverlay.jsx';
import Result, { SupportBanner } from './Result.jsx';
import { PageHeader } from './ui.jsx';

const CONSENT_KEY = 'wb_voice_consent_v1';
const SETTINGS_KEY = 'wb_voice_settings';
const FRAME_MS = 50;
const FACE_SAMPLE_MS = 500; // about 2 samples a second, only while listening
const MAX_EMPTY_RETRIES = 2;

const loadJson = (k, d) => {
  try {
    return { ...d, ...JSON.parse(localStorage.getItem(k) || '{}') };
  } catch {
    return d;
  }
};

function Consent({ onAccept, onChat }) {
  return (
    <div className="card mx-auto max-w-2xl space-y-4">
      <span className="grid h-12 w-12 place-items-center rounded-2xl bg-accent-soft text-accent"><ShieldCheck className="h-6 w-6" aria-hidden /></span>
      <h2 className="text-xl font-semibold">Before the voice interview</h2>
      <p className="text-sm text-muted">The assistant asks a few questions out loud and you answer by speaking. Your browser will ask for the microphone and the camera.</p>
      <div className="grid gap-4 text-sm sm:grid-cols-2">
        <div>
          <h3 className="mb-1 font-semibold">What is analysed</h3>
          <ul className="list-disc space-y-1 pl-5">
            <li>Your words (turned into text on this computer).</li>
            <li>Your facial expression, a few times a second while you answer.</li>
            <li>How you speak: tone, pace and pauses, compared with your own first answer.</li>
          </ul>
        </div>
        <div>
          <h3 className="mb-1 font-semibold">What is stored</h3>
          <ul className="list-disc space-y-1 pl-5">
            <li>The transcript, and per-answer numbers (emotion percentages, speech measures) and your score.</li>
            <li><strong>Raw video and raw audio are not stored.</strong> Video never leaves your browser. Each answer's audio goes only to the local analysis service and is thrown away straight after.</li>
            <li>Stored for 90 days at most. Delete everything any time from History.</li>
          </ul>
        </div>
      </div>
      <p className="rounded-2xl bg-violet-soft px-4 py-3 text-sm">
        <strong>This is not a medical device.</strong> It cannot diagnose anything. The replies are written by a language model (Google Gemini), so your words are sent to it. If you say something that sounds like you may be in danger, the app shows helplines.
      </p>
      <div className="flex flex-wrap gap-3">
        <button type="button" onClick={onAccept} className="btn-primary">I understand, start</button>
        <button type="button" onClick={onChat} className="btn-soft">Use text chat instead</button>
      </div>
    </div>
  );
}

function Orb({ status, level }) {
  const speaking = status === 'speaking';
  const listening = status === 'listening';
  const thinking = status === 'thinking';
  return (
    <div className="relative grid h-44 w-44 place-items-center sm:h-56 sm:w-56" aria-hidden>
      <motion.div
        className="absolute inset-0 rounded-full bg-accent/20"
        animate={speaking ? { scale: [1, 1.18, 1], opacity: [0.5, 0.9, 0.5] } : listening ? { scale: 1 + Math.min(level * 7, 0.35), opacity: 0.8 } : { scale: 1, opacity: 0.5 }}
        transition={speaking ? { duration: 1.6, repeat: Infinity, ease: 'easeInOut' } : { type: 'spring', stiffness: 200, damping: 20 }}
      />
      <motion.div
        className="absolute inset-6 rounded-full bg-gradient-to-br from-accent to-violet shadow-2xl"
        animate={speaking ? { scale: [1, 1.06, 1] } : thinking ? { rotate: 360 } : { scale: 1 }}
        transition={speaking ? { duration: 1.2, repeat: Infinity, ease: 'easeInOut' } : thinking ? { duration: 3, repeat: Infinity, ease: 'linear' } : { duration: 0.3 }}
      />
      <div className="relative text-3xl text-accent-ink">{thinking ? <Loader2 className="h-8 w-8 animate-spin" /> : listening ? <Mic className="h-8 w-8" /> : <span>🌿</span>}</div>
    </div>
  );
}

const STATUS_TEXT = { idle: 'Getting ready', speaking: 'Speaking', listening: 'Listening', thinking: 'Thinking' };

function Room({ settings, setSettings, caps, onChat, onResult }) {
  const toast = useToast();
  const camera = useCamera();
  const videoRef = useRef(null);
  const live = useLiveStream({ videoRef, enabled: camera.status === 'on', save: false });
  const latestFace = useRef(null);
  const [status, setStatus] = useState('idle');
  const [aiCaption, setAiCaption] = useState('');
  const [userCaption, setUserCaption] = useState('');
  const [level, setLevel] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [micError, setMicError] = useState('');
  const [safety, setSafety] = useState(false);
  const [counsellor, setCounsellor] = useState(null);
  const [answers, setAnswers] = useState(0);
  const [showSettings, setShowSettings] = useState(false);
  const [ending, setEnding] = useState(false);
  const [history, setHistory] = useState([]);

  const sessionRef = useRef(null);
  const micRef = useRef(null);
  const ctxRef = useRef(null);
  const analyserRef = useRef(null);
  const speakerRef = useRef(null);
  const finishListenRef = useRef(null);
  const choiceRef = useRef(null);
  const cancelledRef = useRef(false);
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const historyRef = useRef([]);
  const endRef = useRef(null);

  useEffect(() => {
    if (videoRef.current && videoRef.current.srcObject !== camera.stream) videoRef.current.srcObject = camera.stream;
  }, [camera.stream]);
  useEffect(() => {
    latestFace.current = live.result?.face_found ? live.result.probabilities : null;
  }, [live.result]);

  const addHistory = (role, text, kind) => {
    historyRef.current = [...historyRef.current, { id: `${role}${historyRef.current.length}`, role, text, kind }];
    setHistory(historyRef.current);
  };

  // ---- microphone ----
  const openMic = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('This browser cannot record audio.');
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    micRef.current = stream;
    const Ctx = window.AudioContext || window.webkitAudioContext;
    const ctx = new Ctx();
    ctxRef.current = ctx;
    const src = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    src.connect(analyser);
    analyserRef.current = analyser;
  }, []);

  // ---- one answer: record + voice activity detection + face sampling ----
  const listen = useCallback(
    () =>
      new Promise((resolve) => {
        setStatus('listening');
        setUserCaption('');
        setElapsed(0);
        const chunks = [];
        let rec;
        try {
          rec = new MediaRecorder(micRef.current, MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? { mimeType: 'audio/webm;codecs=opus' } : undefined);
        } catch {
          resolve({ error: 'recorder' });
          return;
        }
        rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
        const vad = createVad();
        const buf = new Float32Array(analyserRef.current.fftSize);
        const faceSamples = [];
        let lastFace = 0;
        let browserText = '';
        let recog = null;
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (settingsRef.current.stt === 'browser' && SR) {
          try {
            recog = new SR();
            recog.lang = 'en-US';
            recog.interimResults = true;
            recog.continuous = true;
            recog.onresult = (e) => {
              let t = '';
              for (let i = 0; i < e.results.length; i += 1) t += e.results[i][0].transcript;
              browserText = t.trim();
              setUserCaption(browserText);
            };
            recog.onerror = () => {};
            recog.start();
          } catch {
            recog = null;
          }
        }
        let reason = null;
        const timer = setInterval(() => {
          analyserRef.current.getFloatTimeDomainData(buf);
          const lv = rms(buf);
          setLevel(lv);
          const st = vad.push(lv, FRAME_MS);
          setElapsed(st.elapsedMs / 1000);
          const now = Date.now();
          if (now - lastFace >= FACE_SAMPLE_MS) {
            lastFace = now;
            if (latestFace.current) faceSamples.push({ ...latestFace.current }); // numbers only; the frame is never kept
          }
          if (st.ended) finish(st.ended);
        }, FRAME_MS);
        const finish = (why) => {
          if (reason) return;
          reason = why;
          clearInterval(timer);
          try { recog?.stop(); } catch { /* ignore */ }
          setLevel(0);
          rec.onstop = () => {
            const blob = new Blob(chunks, { type: rec.mimeType || 'audio/webm' });
            resolve({ blob, reason: why, face: faceAggregate(faceSamples), browserText, speechMs: vad.state().speechMs });
          };
          if (rec.state !== 'inactive') rec.stop();
          else rec.onstop();
        };
        finishListenRef.current = finish;
        rec.start();
      }),
    [],
  );

  const say = useCallback(async (text, kind) => {
    setStatus('speaking');
    setAiCaption(text);
    addHistory('assistant', text, kind);
    await speakerRef.current.speak(text, (sentence) => setAiCaption(sentence));
  }, []);

  // The two buttons are visible while the safety message is still being spoken, so a click is remembered even if
  // nobody is waiting for it yet.
  const choiceValueRef = useRef(null);
  const choose = (value) => {
    choiceValueRef.current = value;
    setSafety(false);
    speakerRef.current?.cancel();
    choiceRef.current?.(value);
  };
  const waitChoice = () =>
    new Promise((resolve) => {
      if (choiceValueRef.current) return resolve(choiceValueRef.current);
      choiceRef.current = resolve;
      return undefined;
    });

  // ---- the interview loop: AI speaks -> mic opens -> VAD ends the answer -> transcribe -> safety -> next question ----
  const converse = useCallback(async () => {
    let reply = sessionRef.current.reply;
    let kind = 'opening';
    let empties = 0;
    let skipListenSay = false;
    while (!cancelledRef.current) {
      if (!skipListenSay) await say(reply, kind);
      skipListenSay = false;
      if (cancelledRef.current) return;
      const ans = await listen();
      if (cancelledRef.current) return;
      if (ans.error) {
        toast.error('Recording is not available in this browser.');
        return onChat();
      }
      if (ans.reason === 'skip') {
        setStatus('thinking');
        try {
          const r = await wellbeing.skip(sessionRef.current.sessionId);
          addHistory('user', '(skipped)');
          reply = r.reply;
          kind = r.kind;
          setAnswers((n) => n + 1);
          if (r.complete) return onResult(r.final);
        } catch (err) {
          toast.error(err.message);
          return;
        }
        continue;
      }
      if (ans.reason === 'no_speech' && !ans.blob?.size) {
        reply = 'I did not hear anything. Take your time. You can also press Skip question.';
        kind = 'fallback';
        continue;
      }
      setStatus('thinking');
      let r;
      try {
        r = await wellbeing.voiceAnswer(sessionRef.current.sessionId, ans.blob, { face: ans.face, stt: settingsRef.current.stt, browserText: ans.browserText });
      } catch (err) {
        toast.error(err.message);
        reply = 'Sorry, something went wrong on my side. Could you say that again?';
        kind = 'fallback';
        if ((empties += 1) > MAX_EMPTY_RETRIES) return;
        continue;
      }
      if (r.empty) {
        empties += 1;
        reply = empties > MAX_EMPTY_RETRIES ? 'I am still not catching it. You can try again, press Skip question, or switch to chat.' : 'Sorry, I did not catch that. Could you say it again?';
        kind = 'fallback';
        continue;
      }
      empties = 0;
      setUserCaption(r.transcript);
      addHistory('user', r.transcript);
      setAnswers((n) => n + 1);
      reply = r.reply;
      kind = r.kind;
      if (r.complete) {
        await say(r.reply, r.kind);
        return onResult(r.final);
      }
      if (r.kind === 'safety') {
        // fixed pre-written message: shown as text AND spoken, then no more probing; the person chooses what happens next
        choiceValueRef.current = null;
        setSafety(true);
        await say(r.reply, 'safety');
        const choice = await waitChoice();
        choiceValueRef.current = null;
        if (choice === 'end') return endRef.current?.();
        skipListenSay = true; // "continue talking": just listen, do not ask a new question
      }
    }
  }, [say, listen, onChat, onResult, toast]);

  // ---- start up once ----
  useEffect(() => {
    cancelledRef.current = false;
    speakerRef.current = createSpeaker({ natural: settingsRef.current.voice === 'natural' });
    let alive = true;
    (async () => {
      try {
        await openMic();
      } catch (err) {
        if (!alive) return;
        setMicError(err.name === 'NotAllowedError' ? 'Microphone permission was denied.' : err.name === 'NotFoundError' ? 'No microphone was found.' : err.message);
        return;
      }
      try {
        const s = await wellbeing.start('voice');
        sessionRef.current = s;
        setCounsellor(s.counsellor_contact);
        if (alive) await converse();
      } catch (err) {
        if (alive) toast.error(err.message);
      }
    })();
    return () => {
      alive = false;
      cancelledRef.current = true;
      speakerRef.current?.cancel();
      finishListenRef.current?.('cancel');
      micRef.current?.getTracks().forEach((t) => t.stop());
      ctxRef.current?.close?.();
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // fall back to chat when the microphone is not usable
  useEffect(() => {
    if (micError) {
      toast.error(`${micError} Switching to text chat.`);
      const t = setTimeout(onChat, 1200);
      return () => clearTimeout(t);
    }
    return undefined;
  }, [micError]); // eslint-disable-line react-hooks/exhaustive-deps

  const label = live.smoothed ? displayLabel(live.smoothed) : null;
  const ring = label && label.text !== 'uncertain' ? COLORS[label.top] : 'var(--line)';
  const cameraOn = camera.status === 'on';

  const endSession = async () => {
    setEnding(true);
    cancelledRef.current = true;
    speakerRef.current?.cancel();
    finishListenRef.current?.('cancel');
    try {
      const r = await wellbeing.finish(sessionRef.current.sessionId);
      onResult(r);
    } catch (err) {
      toast.error(err.message);
      setEnding(false);
    }
  };
  endRef.current = endSession;
  const switchToChat = () => {
    cancelledRef.current = true;
    speakerRef.current?.cancel();
    finishListenRef.current?.('cancel');
    onChat(sessionRef.current, historyRef.current);
  };

  return (
    <div className="mx-auto grid max-w-5xl gap-5 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]" data-testid="interview-room">
      {/* camera + mood ring */}
      <section className="space-y-3">
        <div className="relative overflow-hidden rounded-3xl bg-slate-900 p-1.5 transition-colors" style={{ boxShadow: `0 0 0 4px ${ring}, 0 10px 30px -12px rgba(0,0,0,.5)` }}>
          <video ref={videoRef} autoPlay playsInline muted className="block h-auto min-h-40 w-full rounded-[1.25rem]" aria-label="Your camera preview (not recorded)" />
          {!cameraOn && (
            <div className="absolute inset-0 grid place-items-center p-4 text-center text-sm text-slate-200">{camera.status === 'requesting' ? 'Waiting for camera permission…' : camera.message || 'Camera is off: continuing with voice only.'}</div>
          )}
          {cameraOn && <FaceOverlay targetRef={videoRef} box={live.result?.face_found && live.result.box} label={label} confidence={live.top?.confidence} />}
          {cameraOn && label && label.text !== 'uncertain' && (
            <div className="absolute bottom-3 left-3 rounded-full bg-black/60 px-3 py-1 text-sm text-white">{EMOJI[label.top]} <span className="capitalize">{label.text}</span></div>
          )}
        </div>
        <p className="text-xs text-muted">Video stays in your browser. Only emotion percentages from the listening moments are sent.</p>
        <div className="flex items-center gap-2 text-xs text-muted" aria-label="Microphone level">
          {status === 'listening' ? <Mic className="h-4 w-4 text-accent" aria-hidden /> : <MicOff className="h-4 w-4" aria-hidden />}
          <div className="h-2 flex-1 overflow-hidden rounded-full bg-line" role="progressbar" aria-label="Microphone level" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(Math.min(level * 400, 100))}>
            <div className="h-full rounded-full bg-accent transition-[width] duration-75" style={{ width: `${Math.min(level * 400, 100)}%` }} />
          </div>
          <span className="w-10 text-right tabular-nums">{status === 'listening' ? `${elapsed.toFixed(0)}s` : ''}</span>
        </div>
      </section>

      {/* orb, captions, controls */}
      <section className="card flex flex-col items-center gap-4 text-center">
        <div className="flex flex-wrap items-center justify-center gap-2" role="status" aria-live="polite">
          {['listening', 'thinking', 'speaking'].map((s) => (
            <span key={s} className={`rounded-full px-3 py-1 text-xs font-semibold transition ${status === s ? 'bg-accent text-accent-ink' : 'bg-line text-muted'}`}>{STATUS_TEXT[s]}</span>
          ))}
        </div>
        <Orb status={status} level={level} />
        <div className="min-h-20 w-full space-y-3" aria-live="polite">
          <AnimatePresence mode="wait">
            <motion.p key={aiCaption} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="text-lg font-medium leading-snug" data-testid="ai-caption">{aiCaption || '…'}</motion.p>
          </AnimatePresence>
          {userCaption && <p className="rounded-2xl bg-accent-soft px-4 py-2 text-sm text-muted" data-testid="user-caption">“{userCaption}”</p>}
        </div>

        {safety && (
          <div className="w-full space-y-3 text-left">
            <SupportBanner support={{ counsellor_contact: counsellor }} />
            <div className="flex flex-wrap justify-center gap-3">
              <button type="button" className="btn-primary" onClick={() => choose('continue')}>Continue talking</button>
              <button type="button" className="btn-soft" onClick={() => choose('end')}>End session</button>
            </div>
          </div>
        )}

        <div className="flex flex-wrap items-center justify-center gap-2">
          <button type="button" className="btn-primary" disabled={status !== 'listening'} onClick={() => finishListenRef.current?.('done')}><Check className="h-4 w-4" aria-hidden /> Done answering</button>
          <button type="button" className="btn-soft" disabled={status !== 'listening'} onClick={() => finishListenRef.current?.('skip')}><SkipForward className="h-4 w-4" aria-hidden /> Skip question</button>
          <button type="button" className="btn-soft" onClick={switchToChat}><ArrowLeftRight className="h-4 w-4" aria-hidden /> Switch to chat</button>
          <button type="button" className="btn-danger" disabled={ending || answers < 1} onClick={endSession}><Square className="h-4 w-4" aria-hidden /> End session</button>
          <button type="button" className="btn-soft !p-2.5" aria-label="Voice settings" aria-expanded={showSettings} onClick={() => setShowSettings((s) => !s)}><Settings className="h-4 w-4" aria-hidden /></button>
        </div>

        {showSettings && (
          <div className="w-full rounded-2xl border border-line bg-solid p-4 text-left text-sm">
            <label className="mb-3 block">
              <span className="eyebrow">Voice</span>
              <select className="field mt-1" value={settings.voice} onChange={(e) => setSettings({ ...settings, voice: e.target.value })}>
                <option value="browser">Browser voice (default, no key)</option>
                <option value="natural" disabled={!caps.tts}>Natural voice (Gemini){caps.tts ? '' : ' - needs a key'}</option>
              </select>
              <span className="text-xs text-muted">Applies from the next session.</span>
            </label>
            <label className="block">
              <span className="eyebrow">Speech to text</span>
              <select className="field mt-1" value={settings.stt} onChange={(e) => setSettings({ ...settings, stt: e.target.value })}>
                <option value="local">Local Whisper (default, works offline)</option>
                <option value="browser" disabled={!caps.webSpeech}>Browser speech recognition{caps.webSpeech ? '' : ' - not available here'}</option>
                <option value="gemini" disabled={!caps.tts}>Gemini transcription{caps.tts ? '' : ' - needs a key'}</option>
              </select>
            </label>
          </div>
        )}
        <p className="text-xs text-muted">Not a medical device. Press Done answering any time, or just stop talking for a moment.</p>
      </section>
      <div className="sr-only" aria-hidden>{history.length}</div>
    </div>
  );
}

export default function InterviewRoom({ onChat, onHome, onFinished }) {
  const [phase, setPhase] = useState(() => {
    try {
      return localStorage.getItem(CONSENT_KEY) === '1' ? 'room' : 'consent';
    } catch {
      return 'consent';
    }
  });
  const [result, setResult] = useState(null);
  const [settings, setSettingsState] = useState(() => loadJson(SETTINGS_KEY, { voice: 'browser', stt: 'local' }));
  const [caps, setCaps] = useState({ tts: false, webSpeech: Boolean(window.SpeechRecognition || window.webkitSpeechRecognition), stt: true });
  const [key, setKey] = useState(0);

  const setSettings = (s) => {
    setSettingsState(s);
    try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(s)); } catch { /* ignore */ }
  };
  useEffect(() => {
    wellbeing.status().then((s) => setCaps((c) => ({ ...c, tts: Boolean(s.tts?.gemini), stt: s.stt?.available !== false }))).catch(() => {});
  }, []);

  const accept = () => {
    try { localStorage.setItem(CONSENT_KEY, '1'); } catch { /* ignore */ }
    setPhase('room');
  };

  if (phase === 'result' && result) {
    return (
      <div>
        <PageHeader title="Your result" />
        <Result result={result} onHome={onHome} onAgain={() => { setResult(null); setKey((k) => k + 1); setPhase('room'); }} />
      </div>
    );
  }
  return (
    <div>
      <PageHeader title="Voice interview" subtitle="The assistant asks, you answer out loud. Camera and microphone on this device only." />
      {phase === 'consent' ? (
        <Consent onAccept={accept} onChat={() => onChat(null, [])} />
      ) : (
        <Room
          key={key}
          settings={settings}
          setSettings={setSettings}
          caps={caps}
          onChat={onChat}
          onResult={(r) => { setResult(r); setPhase('result'); onFinished?.(); }}
        />
      )}
    </div>
  );
}
