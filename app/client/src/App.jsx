import { AnimatePresence, MotionConfig, motion } from 'framer-motion';
import { AudioLines, History, Home as HomeIcon, ImagePlus, MessageCircleHeart, MessageSquareText, Mic, Moon, ScanFace, Sun } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';
import AudioTab from './components/AudioTab.jsx';
import CheckIn from './components/CheckIn.jsx';
import Home from './components/Home.jsx';
import InterviewRoom from './components/InterviewRoom.jsx';
import HistoryPage from './components/HistoryPage.jsx';
import LiveTab from './components/LiveTab.jsx';
import TextTab from './components/TextTab.jsx';
import UploadTab from './components/UploadTab.jsx';
import { ThemeProvider, useTheme } from './lib/theme.jsx';
import { ToastProvider, useToast } from './lib/toast.jsx';
import { wellbeing } from './lib/wellbeing.js';

const TABS = [
  { id: 'home', label: 'Home', icon: HomeIcon },
  { id: 'checkin', label: 'Check-in', icon: MessageCircleHeart },
  { id: 'voice', label: 'Voice Interview', icon: AudioLines },
  { id: 'live', label: 'Live Emotion', icon: ScanFace },
  { id: 'upload', label: 'Upload', icon: ImagePlus },
  { id: 'audio', label: 'Audio', icon: Mic },
  { id: 'text', label: 'Text', icon: MessageSquareText },
  { id: 'history', label: 'History', icon: History },
];

const NOTE =
  'Experimental research prototype, not a medical device. The emotion models were trained on acted datasets and are weak on real-world data (held-out FER-2013 macro-F1 0.09, RAVDESS 0.41, CREMA-D 0.42). Scores and tiers use unvalidated defaults.';

function ThemeToggle() {
  const { dark, toggle } = useTheme();
  return (
    <button type="button" onClick={toggle} aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'} aria-pressed={dark} className="btn-soft !p-2.5">
      {dark ? <Sun className="h-4 w-4" aria-hidden /> : <Moon className="h-4 w-4" aria-hidden />}
    </button>
  );
}

function Shell() {
  const toast = useToast();
  const [tab, setTab] = useState('home');
  const [health, setHealth] = useState(null);
  const [history, setHistory] = useState(null);
  const [histLoading, setHistLoading] = useState(true);
  const [resume, setResume] = useState(null); // a voice session continued as text chat

  useEffect(() => {
    let alive = true;
    const load = () =>
      fetch('/api/health')
        .then((r) => r.json())
        .then((h) => alive && setHealth(h))
        .catch(() => alive && setHealth({ error: true }));
    load();
    const t = setInterval(load, 15000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  const loadHistory = useCallback(async () => {
    setHistLoading(true);
    try {
      setHistory(await wellbeing.history());
    } catch (err) {
      setHistory({ sessions: [], ema: null, slope_per_day: null, streak_days: 0 });
      if (!/reach the server/.test(err.message)) toast.error(err.message);
    } finally {
      setHistLoading(false);
    }
  }, [toast]);
  useEffect(() => {
    loadHistory();
  }, [loadHistory]);
  useEffect(() => {
    if (tab === 'home' || tab === 'history') loadHistory();
  }, [tab, loadHistory]);

  const modelInfo = health?.modelService?.reachable ? { model: health.modelService.model, detector: health.modelService.detector } : null;
  const dbText = health?.db?.connected ? `Database: ${health.db.mode}` : 'Database offline';

  const page = {
    home: <Home history={history} loading={histLoading} onNavigate={setTab} />,
    checkin: <CheckIn onHome={() => setTab('home')} onFinished={loadHistory} onVoice={() => setTab('voice')} resume={resume} />,
    voice: (
      <InterviewRoom
        onHome={() => setTab('home')}
        onFinished={loadHistory}
        onChat={(session, messages) => {
          setResume(session ? { session, messages: messages || [] } : null);
          setTab('checkin');
        }}
      />
    ),
    live: <LiveTab modelInfo={modelInfo} onGoUpload={() => setTab('upload')} />,
    upload: <UploadTab />,
    audio: <AudioTab />,
    text: <TextTab />,
    history: <HistoryPage history={history} loading={histLoading} onReload={loadHistory} onNavigate={setTab} />,
  }[tab];

  const navButton = (t, vertical) => (
    <button
      key={t.id}
      type="button"
      onClick={() => setTab(t.id)}
      aria-current={tab === t.id ? 'page' : undefined}
      className={`flex items-center gap-3 whitespace-nowrap rounded-2xl px-3.5 py-2.5 text-sm font-medium transition ${
        vertical ? 'w-full' : ''
      } ${tab === t.id ? 'bg-accent text-accent-ink shadow' : 'text-muted hover:bg-accent-soft hover:text-ink'}`}
    >
      <t.icon className="h-[18px] w-[18px]" aria-hidden />
      {t.label}
    </button>
  );

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[15.5rem_1fr]">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-50 focus:rounded-full focus:bg-solid focus:px-4 focus:py-2">Skip to content</a>

      {/* Sidebar (large screens) */}
      <aside className="glass sticky top-0 hidden h-screen flex-col gap-6 border-y-0 border-l-0 p-5 lg:flex" aria-label="Main navigation">
        <div className="flex items-center gap-3">
          <span className="grid h-10 w-10 place-items-center rounded-2xl bg-accent text-lg text-accent-ink" aria-hidden>🌿</span>
          <div>
            <div className="font-semibold leading-tight">Mindful</div>
            <div className="text-xs text-muted">Emotion and wellbeing</div>
          </div>
        </div>
        <nav className="flex flex-col gap-1">{TABS.map((t) => navButton(t, true))}</nav>
        <div className="mt-auto space-y-2 text-xs text-muted">
          <div className={`rounded-xl px-3 py-1.5 ${modelInfo ? 'bg-accent-soft text-accent' : 'bg-coral-soft text-coral'}`}>{modelInfo ? `Model: ${modelInfo.model}` : 'Model service offline'}</div>
          <div className="rounded-xl bg-line px-3 py-1.5">{health ? dbText : 'checking…'}</div>
          <ThemeToggle />
        </div>
      </aside>

      <div className="min-w-0">
        {/* Top bar (small screens) */}
        <header className="glass sticky top-0 z-30 border-x-0 border-t-0 lg:hidden">
          <div className="flex items-center justify-between px-4 py-3">
            <div className="flex items-center gap-2 font-semibold"><span aria-hidden>🌿</span> Mindful</div>
            <ThemeToggle />
          </div>
          <nav className="flex gap-1 overflow-x-auto px-3 pb-2" aria-label="Main navigation">{TABS.map((t) => navButton(t, false))}</nav>
        </header>

        <main id="main" className="mx-auto max-w-6xl px-4 py-6 sm:px-6 lg:py-8">
          <AnimatePresence mode="wait">
            <motion.div key={tab} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}>
              {page}
            </motion.div>
          </AnimatePresence>
        </main>

        <footer className="mx-auto max-w-6xl px-4 pb-8 sm:px-6">
          <p className="rounded-2xl border border-line bg-violet-soft p-3 text-xs"><strong>Not a medical or diagnostic tool.</strong> {NOTE}</p>
        </footer>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <MotionConfig reducedMotion="user">
        <ToastProvider>
          <Shell />
        </ToastProvider>
      </MotionConfig>
    </ThemeProvider>
  );
}
