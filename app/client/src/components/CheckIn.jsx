import { AnimatePresence, motion } from 'framer-motion';
import { Camera, CameraOff, Mic, MicOff, Send, ShieldCheck } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useCamera } from '../hooks/useCamera.js';
import { useLiveStream } from '../hooks/useLiveStream.js';
import { useToast } from '../lib/toast.jsx';
import { wellbeing } from '../lib/wellbeing.js';
import Result, { SupportBanner } from './Result.jsx';
import { PageHeader, Skeleton } from './ui.jsx';

const CONSENT_KEY = 'wb_consent_v1';
const CHIPS = ['Pretty good', 'Tired', 'Stressed', 'Not great'];
const SAMPLE_EVERY_MS = 4000;

const readConsent = () => {
  try {
    return localStorage.getItem(CONSENT_KEY) === '1';
  } catch {
    return false;
  }
};

function Consent({ onAccept }) {
  return (
    <div className="card mx-auto max-w-xl space-y-4">
      <span className="grid h-12 w-12 place-items-center rounded-2xl bg-accent-soft text-accent"><ShieldCheck className="h-6 w-6" aria-hidden /></span>
      <h2 className="text-xl font-semibold">Before we start</h2>
      <ul className="list-disc space-y-2 pl-5 text-sm">
        <li><strong>This is not a medical device.</strong> It cannot diagnose anything or replace a doctor or counsellor.</li>
        <li>You will chat with an automated assistant for a few minutes. <strong>What you write is sent to a third-party language model (Google Gemini)</strong> to write replies and to pick out mood and themes, so it leaves this computer. In demo mode nothing is sent anywhere.</li>
        <li>The chat text and a summary are stored on the app's server under a random id (kept in this browser, not linked to your name) so you can see your trend. They are deleted automatically after 90 days, and you can delete everything now with "Delete my data" on the History page.</li>
        <li>The optional emotion check reads your face on this computer and sends only emotion percentages. No video or photo is ever saved.</li>
        <li>If you say something that sounds like you may be in danger, the app shows helplines. It cannot call anyone for you, and it cannot promise to keep things private.</li>
      </ul>
      <button type="button" onClick={onAccept} className="btn-primary">I understand, continue</button>
    </div>
  );
}

function Bubble({ m }) {
  const mine = m.role === 'user';
  const safety = m.kind === 'safety';
  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }} className={`flex ${mine ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[88%] whitespace-pre-line rounded-3xl px-4 py-3 text-[15px] leading-relaxed sm:max-w-[75%] ${
          mine ? 'rounded-br-lg bg-accent text-accent-ink' : safety ? 'rounded-bl-lg border border-coral/40 bg-coral-soft text-ink' : 'rounded-bl-lg border border-line bg-solid text-ink'
        }`}
      >
        {m.text}
      </div>
    </motion.div>
  );
}

const Typing = () => (
  <div className="flex justify-start" role="status" aria-label="The assistant is typing">
    <div className="flex gap-1.5 rounded-3xl rounded-bl-lg border border-line bg-solid px-4 py-3.5">
      <span className="dot h-2 w-2 rounded-full bg-muted" /><span className="dot h-2 w-2 rounded-full bg-muted" /><span className="dot h-2 w-2 rounded-full bg-muted" />
    </div>
  </div>
);

// Hidden-by-default emotion check: samples the live face emotion every few seconds and sends only probabilities.
function EmotionCheck({ sessionId, onSent }) {
  const toast = useToast();
  const camera = useCamera();
  const videoRef = useRef(null);
  const live = useLiveStream({ videoRef, enabled: camera.status === 'on', save: false });
  const latest = useRef(null);

  useEffect(() => {
    if (videoRef.current && videoRef.current.srcObject !== camera.stream) videoRef.current.srcObject = camera.stream;
  }, [camera.stream]);
  useEffect(() => {
    latest.current = live.result;
  }, [live.result]);
  useEffect(() => {
    if (camera.status !== 'on') return undefined;
    const t = setInterval(() => {
      const r = latest.current;
      if (r?.face_found && r.probabilities) {
        wellbeing
          .signal(sessionId, { modality: 'face', top_label: r.top_label, confidence: r.confidence, probabilities: r.probabilities })
          .then(() => onSent?.())
          .catch(() => {});
      }
    }, SAMPLE_EVERY_MS);
    return () => clearInterval(t);
  }, [camera.status, sessionId, onSent]);
  useEffect(() => {
    if (['denied', 'no_camera', 'unsupported', 'error'].includes(camera.status)) toast.error(camera.message || 'Camera not available.');
  }, [camera.status]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="flex items-center gap-3 rounded-2xl bg-accent-soft p-3 text-sm">
      {camera.status === 'on' ? (
        <>
          <video ref={videoRef} autoPlay playsInline muted className="h-14 w-20 rounded-lg bg-slate-900 object-cover" aria-label="Camera preview" />
          <div>
            <div className="flex items-center gap-1.5 font-medium"><span className="inline-block h-2 w-2 animate-pulse rounded-full bg-red-500" aria-hidden /> Emotion check on</div>
            <div className="text-xs text-muted">Only emotion percentages leave this page. Turn it off any time.</div>
          </div>
          <button type="button" onClick={camera.stop} className="btn-soft ml-auto !px-3 !py-1.5">Turn off</button>
        </>
      ) : (
        <>
          <CameraOff className="h-5 w-5 text-muted" aria-hidden />
          <span className="text-muted">{camera.status === 'requesting' ? 'Waiting for camera permission…' : 'Starting the camera…'}</span>
        </>
      )}
      {camera.status === 'idle' && <AutoStart start={camera.start} />}
    </div>
  );
}
function AutoStart({ start }) {
  useEffect(() => {
    start();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  return null;
}

export default function CheckIn({ onHome, onFinished, onVoice, resume }) {
  const toast = useToast();
  const [consent, setConsent] = useState(readConsent);
  const [phase, setPhase] = useState('intro'); // intro | chat | result
  const [session, setSession] = useState(null);
  const [intro, setIntro] = useState('');
  const [counsellor, setCounsellor] = useState(null);
  const [mockNotice, setMockNotice] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [progress, setProgress] = useState({ turn: 0, max: 8 });
  const [result, setResult] = useState(null);
  const [supportOpen, setSupportOpen] = useState(false);
  const [emotionOn, setEmotionOn] = useState(false);
  const [samples, setSamples] = useState(0);
  const [listening, setListening] = useState(false);
  const [viaVoice, setViaVoice] = useState(false);
  const scrollRef = useRef(null);
  const recRef = useRef(null);
  const SpeechRec = typeof window !== 'undefined' ? window.SpeechRecognition || window.webkitSpeechRecognition : null;

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, sending]);
  useEffect(() => () => recRef.current?.stop?.(), []);
  // switching from the voice interview: continue the same session as a text chat
  useEffect(() => {
    if (!resume?.session) return;
    setSession(resume.session);
    setIntro('');
    setCounsellor(resume.session.counsellor_contact);
    setMessages(resume.messages.length ? resume.messages : [{ id: 'open', role: 'assistant', text: resume.session.reply }]);
    setProgress({ turn: resume.messages.filter((m) => m.role === 'user').length, max: resume.session.max_turns });
    setPhase('chat');
  }, [resume]);

  const accept = () => {
    try {
      localStorage.setItem(CONSENT_KEY, '1');
    } catch {
      /* ignore */
    }
    setConsent(true);
  };

  const start = async () => {
    setSending(true);
    try {
      const s = await wellbeing.start();
      setSession(s);
      setIntro(s.intro);
      setCounsellor(s.counsellor_contact);
      setMockNotice(s.mock ? s.notice : null);
      setProgress({ turn: 0, max: s.max_turns });
      setMessages([{ id: 'open', role: 'assistant', text: s.reply }]);
      setPhase('chat');
    } catch (err) {
      toast.error(err.message);
    } finally {
      setSending(false);
    }
  };

  const finish = useCallback(async () => {
    if (!session) return;
    setSending(true);
    try {
      const r = await wellbeing.finish(session.sessionId);
      setResult(r);
      setPhase('result');
      onFinished?.();
    } catch (err) {
      toast.error(err.message);
    } finally {
      setSending(false);
    }
  }, [session, toast, onFinished]);

  const send = async (text) => {
    const t = text.trim();
    if (!t || sending || !session) return;
    setInput('');
    setMessages((m) => [...m, { id: `u${m.length}`, role: 'user', text: t }]);
    setSending(true);
    try {
      const r = await wellbeing.message(session.sessionId, t, viaVoice ? 'voice' : 'text');
      setViaVoice(false);
      setMessages((m) => [...m, { id: `a${m.length}`, role: 'assistant', text: r.reply, kind: r.kind }]);
      setProgress({ turn: r.turn, max: r.max_turns });
      if (r.needs_human_followup) setSupportOpen(true);
      if (r.complete) {
        setResult(r.final);
        setPhase('result');
        onFinished?.();
      }
    } catch (err) {
      toast.error(err.message);
    } finally {
      setSending(false);
    }
  };

  const toggleMic = () => {
    if (!SpeechRec) return;
    if (listening) {
      recRef.current?.stop();
      return;
    }
    const rec = new SpeechRec();
    rec.lang = navigator.language || 'en-IN';
    rec.interimResults = false;
    rec.onresult = (e) => {
      setInput((cur) => `${cur} ${e.results[0][0].transcript}`.trim());
      setViaVoice(true);
    };
    rec.onerror = () => toast.error('The microphone could not be used.');
    rec.onend = () => setListening(false);
    recRef.current = rec;
    setListening(true);
    rec.start();
  };

  if (!consent) {
    return (
      <div>
        <PageHeader title="Check-in" subtitle="A short, friendly chat about how you have been." />
        <Consent onAccept={accept} />
      </div>
    );
  }

  if (phase === 'result' && result) {
    return (
      <div>
        <PageHeader title="Your result" />
        <Result result={result} onHome={onHome} onAgain={() => { setPhase('intro'); setResult(null); setSession(null); setMessages([]); setSupportOpen(false); setEmotionOn(false); }} />
      </div>
    );
  }

  if (phase === 'intro') {
    return (
      <div>
        <PageHeader title="Check-in" subtitle="A short, friendly chat about how you have been. About 5 minutes." />
        <div className="card mx-auto max-w-xl space-y-4 text-center">
          <div className="text-5xl" aria-hidden>🌿</div>
          <h2 className="text-xl font-semibold">Ready when you are</h2>
          <p className="text-sm text-muted">You can stop at any time. Nothing you type is judged.</p>
          <div className="flex flex-wrap justify-center gap-3">
            <button type="button" onClick={start} disabled={sending} className="btn-primary">{sending ? 'Starting…' : 'Start check-in'}</button>
            {onVoice && <button type="button" onClick={onVoice} className="btn-soft">Try the voice interview</button>}
          </div>
          {sending && <Skeleton className="mx-auto h-4 w-1/2" />}
        </div>
      </div>
    );
  }

  const pctDone = Math.min(100, Math.round((progress.turn / progress.max) * 100));
  return (
    <div className="mx-auto max-w-3xl">
      <PageHeader title="Check-in" subtitle={intro} />
      {mockNotice && <p className="mb-3 rounded-2xl bg-violet-soft px-4 py-2 text-xs" role="status">{mockNotice}</p>}

      <div className="mb-3" aria-label={`Progress: message ${progress.turn} of ${progress.max}`}>
        <div className="mb-1 flex justify-between text-xs text-muted">
          <span>Message {Math.min(progress.turn, progress.max)} of {progress.max}</span>
          <span>{pctDone}%</span>
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-line" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pctDone}>
          <motion.div className="h-full rounded-full bg-accent" initial={false} animate={{ width: `${pctDone}%` }} transition={{ duration: 0.5 }} />
        </div>
      </div>

      <div className="card flex h-[min(62vh,560px)] flex-col !p-0">
        <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto p-4 sm:p-5" aria-live="polite">
          <AnimatePresence initial={false}>
            {messages.map((m) => <Bubble key={m.id} m={m} />)}
          </AnimatePresence>
          {sending && messages[messages.length - 1]?.role === 'user' && <Typing />}
          {supportOpen && <SupportBanner support={{ counsellor_contact: counsellor }} />}
        </div>

        <div className="space-y-3 border-t border-line p-3 sm:p-4">
          {progress.turn < 2 && (
            <div className="flex flex-wrap gap-2" aria-label="Quick replies">
              {CHIPS.map((c) => (
                <button key={c} type="button" className="chip" disabled={sending} onClick={() => send(c)}>{c}</button>
              ))}
            </div>
          )}
          {emotionOn && session && <EmotionCheck sessionId={session.sessionId} onSent={() => setSamples((n) => n + 1)} />}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              send(input);
            }}
            className="flex items-end gap-2"
          >
            <label htmlFor="chat-input" className="sr-only">Your message</label>
            <textarea
              id="chat-input"
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  send(input);
                }
              }}
              placeholder="Type here…"
              maxLength={2000}
              className="field max-h-32 resize-none"
            />
            {SpeechRec && (
              <button type="button" onClick={toggleMic} aria-pressed={listening} aria-label={listening ? 'Stop dictation' : 'Dictate with the microphone'} className="btn-soft !px-3.5 !py-3">
                {listening ? <MicOff className="h-5 w-5 text-coral" aria-hidden /> : <Mic className="h-5 w-5" aria-hidden />}
              </button>
            )}
            <button type="submit" disabled={sending || !input.trim()} aria-label="Send" className="btn-primary !px-4 !py-3"><Send className="h-5 w-5" aria-hidden /></button>
          </form>
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
            <label className="flex cursor-pointer items-center gap-2 text-muted">
              <input type="checkbox" checked={emotionOn} onChange={(e) => setEmotionOn(e.target.checked)} className="h-4 w-4 accent-[var(--accent)]" />
              <Camera className="h-4 w-4" aria-hidden /> Optional emotion check (camera): needed for a score{samples > 0 && ` · ${samples} samples`}
            </label>
            <button type="button" onClick={finish} disabled={sending || progress.turn < 1} className="font-semibold text-accent underline-offset-2 hover:underline disabled:opacity-50">
              Finish and see my result
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
