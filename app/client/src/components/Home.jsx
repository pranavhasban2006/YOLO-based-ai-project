import { motion } from 'framer-motion';
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { AudioLines, Flame, History as HistoryIcon, ImagePlus, Mic, MessageCircleHeart, MessageSquareText, ScanFace } from 'lucide-react';
import { chartColors, useTheme } from '../lib/theme.jsx';
import { TIER } from '../lib/wellbeing.js';
import { EmptyState, Gauge, Pill, Skeleton } from './ui.jsx';

const DAY = 86400000;

function greeting() {
  const h = new Date().getHours();
  return h < 5 ? 'Hello, night owl' : h < 12 ? 'Good morning' : h < 18 ? 'Good afternoon' : 'Good evening';
}

const TOOLS = [
  { id: 'voice', title: 'Voice interview', text: 'Talk it through out loud; the app listens.', icon: AudioLines },
  { id: 'live', title: 'Live emotion', text: 'See expressions in real time from your camera.', icon: ScanFace },
  { id: 'upload', title: 'Photo upload', text: 'Analyse the expression in a photo.', icon: ImagePlus },
  { id: 'audio', title: 'Speech emotion', text: 'Record or upload a short voice clip.', icon: Mic },
  { id: 'text', title: 'Text', text: 'Planned. Not available yet.', icon: MessageSquareText },
  { id: 'history', title: 'History', text: 'Your trend and past check-ins.', icon: HistoryIcon },
];

export default function Home({ history, loading, onNavigate }) {
  const { dark } = useTheme();
  const colors = chartColors(dark);
  const now = Date.now();
  const sessions = history?.sessions || [];
  const today = sessions.filter((s) => now - s.t < DAY && new Date(s.t).getDate() === new Date(now).getDate()).at(-1);
  const week = sessions.filter((s) => now - s.t < 7 * DAY && typeof s.score === 'number').map((s) => ({ t: s.t, score: s.score, day: new Date(s.t).toLocaleString(undefined, { weekday: 'short', hour: 'numeric', minute: '2-digit' }) }));
  const tier = today?.tier ? TIER[today.tier] : null;

  return (
    <div className="space-y-5">
      <motion.section initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="card grid items-center gap-6 md:grid-cols-[auto_1fr]">
        <div className="flex justify-center">
          {loading ? (
            <Skeleton className="h-40 w-40 !rounded-full" />
          ) : (
            <Gauge value={today?.score} empty={!today || today.score == null} color={tier ? `var(--${tier.color})` : 'var(--accent)'} label={today ? 'today' : 'no check-in yet'} />
          )}
        </div>
        <div>
          <div className="eyebrow">{new Date().toLocaleDateString(undefined, { weekday: 'long', month: 'long', day: 'numeric' })}</div>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight">{greeting()} 👋</h1>
          <p className="mt-2 max-w-xl text-muted">
            {today ? `Your latest check-in today: ${tier?.label || 'done'}. You can check in again whenever you like.` : 'A short chat can show how the week is going. It takes about five minutes.'}
          </p>
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <button type="button" onClick={() => onNavigate('checkin')} className="btn-primary !px-6 !py-3 text-base">
              <MessageCircleHeart className="h-5 w-5" aria-hidden /> Start check-in
            </button>
            <button type="button" onClick={() => onNavigate('voice')} className="btn-soft !px-5 !py-3"><AudioLines className="h-5 w-5" aria-hidden /> Voice interview</button>
            <Pill tone="muted"><Flame className="mr-1 h-3.5 w-3.5" aria-hidden /> {history?.streak_days ?? 0} day streak</Pill>
          </div>
        </div>
      </motion.section>

      <div className="grid gap-5 lg:grid-cols-[2fr_1fr]">
        <section className="card">
          <h2 className="text-lg font-semibold">Last 7 days</h2>
          {loading ? (
            <Skeleton className="mt-4 h-40" />
          ) : week.length === 0 ? (
            <EmptyState icon={MessageCircleHeart} title="No check-ins this week" text="Your scores show up here after you finish a check-in." />
          ) : (
            <div className="mt-3 h-44 w-full" aria-label="Wellbeing scores over the last 7 days" role="img">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={week} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="homeFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={colors.accent} stopOpacity={0.45} />
                      <stop offset="100%" stopColor={colors.accent} stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="day" stroke={colors.text} tick={{ fill: colors.text, fontSize: 12 }} tickLine={false} axisLine={false} />
                  <YAxis domain={[0, 100]} stroke={colors.text} tick={{ fill: colors.text, fontSize: 12 }} tickLine={false} axisLine={false} />
                  <Tooltip contentStyle={{ background: colors.surface, border: `1px solid ${colors.grid}`, borderRadius: 12, color: colors.text }} />
                  <Area type="monotone" dataKey="score" stroke={colors.accent} strokeWidth={2.5} fill="url(#homeFill)" isAnimationActive />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </section>

        <section className="card">
          <h2 className="text-lg font-semibold">Trend</h2>
          {loading ? (
            <Skeleton className="mt-4 h-24" />
          ) : history?.ema == null ? (
            <p className="mt-3 text-sm text-muted">Do two or three check-ins and a smoothed trend appears here.</p>
          ) : (
            <div className="mt-3 space-y-2 text-sm">
              <div className="text-4xl font-semibold tabular-nums">{Math.round(history.ema)}</div>
              <p className="text-muted">Smoothed score over your recent check-ins, so one rough day does not swing it.</p>
              {history.slope_per_day != null && (
                <Pill tone={history.slope_per_day >= 0 ? 'accent' : 'violet'}>{history.slope_per_day >= 0 ? 'Rising' : 'Falling'} {Math.abs(history.slope_per_day)} pts/day</Pill>
              )}
            </div>
          )}
        </section>
      </div>

      <section aria-label="Other tools">
        <h2 className="mb-3 text-lg font-semibold">More tools</h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
          {TOOLS.map((t, i) => (
            <motion.button
              key={t.id}
              type="button"
              onClick={() => onNavigate(t.id)}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.05 * i }}
              whileHover={{ y: -3 }}
              className="glass rounded-3xl p-4 text-left transition hover:border-accent"
            >
              <span className="mb-3 grid h-10 w-10 place-items-center rounded-2xl bg-violet-soft text-violet"><t.icon className="h-5 w-5" aria-hidden /></span>
              <div className="font-semibold">{t.title}</div>
              <div className="mt-1 text-xs text-muted">{t.text}</div>
            </motion.button>
          ))}
        </div>
      </section>
    </div>
  );
}
