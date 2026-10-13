import { Trash2, TrendingUp } from 'lucide-react';
import { useState } from 'react';
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceArea, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { COLORS, EMOJI, CLASSES } from '../lib/emotion.js';
import { chartColors, useTheme } from '../lib/theme.jsx';
import { useToast } from '../lib/toast.jsx';
import { TIER, wellbeing } from '../lib/wellbeing.js';
import DetectionLog from './DetectionLog.jsx';
import { EmptyState, PageHeader, Pill, Skeleton } from './ui.jsx';

export default function HistoryPage({ history, loading, onReload, onNavigate }) {
  const { dark } = useTheme();
  const toast = useToast();
  const c = chartColors(dark);
  const [confirming, setConfirming] = useState(false);
  const sessions = history?.sessions || [];
  const th = history?.thresholds || { doing_okay_min: 65, struggling_min: 40 };
  const data = sessions.map((s) => ({ t: s.t, label: new Date(s.t).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }), score: s.score, ema: s.ema }));
  const emo = sessions
    .filter((s) => s.emotion_distribution)
    .map((s) => ({ label: new Date(s.t).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }), ...Object.fromEntries(CLASSES.map((k) => [k, Math.round(s.emotion_distribution[k] * 1000) / 10])) }));

  const del = async () => {
    try {
      const r = await wellbeing.deleteMyData();
      toast.success(`Deleted ${r.deleted_sessions} check-in${r.deleted_sessions === 1 ? '' : 's'} and their chat text.`);
      setConfirming(false);
      onReload();
    } catch (err) {
      toast.error(err.message);
    }
  };

  return (
    <div className="space-y-5">
      <PageHeader title="History" subtitle="Your check-ins over time. The shaded bands are rough zones, not diagnoses.">
        <button type="button" onClick={() => setConfirming(true)} disabled={loading} className="btn-soft">
          <Trash2 className="h-4 w-4" aria-hidden /> Delete my data
        </button>
      </PageHeader>

      {confirming && (
        <div role="alertdialog" aria-labelledby="del-title" className="card border-coral/40">
          <h2 id="del-title" className="text-lg font-semibold">Delete everything?</h2>
          <p className="mt-1 text-sm text-muted">This removes all your check-ins, chat text and scores from the server. It cannot be undone.</p>
          <div className="mt-4 flex gap-3">
            <button type="button" onClick={del} className="btn-danger">Yes, delete</button>
            <button type="button" onClick={() => setConfirming(false)} className="btn-soft">Keep my data</button>
          </div>
        </div>
      )}

      <section className="card">
        <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-lg font-semibold">Score over time</h2>
          {history?.ema != null && <Pill tone="accent">Smoothed trend {Math.round(history.ema)}</Pill>}
        </div>
        {loading ? (
          <Skeleton className="mt-3 h-64" />
        ) : data.length === 0 ? (
          <EmptyState icon={TrendingUp} title="No check-ins yet" text="After your first check-in, your score and trend appear here." action={<button type="button" className="btn-primary mt-2" onClick={() => onNavigate('checkin')}>Start check-in</button>} />
        ) : (
          <div className="mt-3 h-64 w-full" role="img" aria-label="Line chart of wellbeing score over time with tier bands">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data} margin={{ top: 8, right: 12, left: -16, bottom: 0 }}>
                <ReferenceArea y1={0} y2={th.struggling_min} fill={c.coral} fillOpacity={0.1} />
                <ReferenceArea y1={th.struggling_min} y2={th.doing_okay_min} fill={c.violet} fillOpacity={0.1} />
                <ReferenceArea y1={th.doing_okay_min} y2={100} fill={c.accent} fillOpacity={0.1} />
                <CartesianGrid stroke={c.grid} vertical={false} />
                <XAxis dataKey="label" stroke={c.text} tick={{ fill: c.text, fontSize: 12 }} tickLine={false} />
                <YAxis domain={[0, 100]} stroke={c.text} tick={{ fill: c.text, fontSize: 12 }} tickLine={false} axisLine={false} />
                <Tooltip contentStyle={{ background: c.surface, border: `1px solid ${c.grid}`, borderRadius: 12, color: c.text }} />
                <Legend wrapperStyle={{ color: c.text, fontSize: 12 }} />
                <Line type="monotone" dataKey="score" name="Score" stroke={c.accent} strokeWidth={2.5} dot={{ r: 4, fill: c.accent }} />
                <Line type="monotone" dataKey="ema" name="Trend (EMA)" stroke={c.violet} strokeWidth={2} strokeDasharray="5 4" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
        <p className="mt-2 text-xs text-muted">Bands: coral under {th.struggling_min}, violet {th.struggling_min} to {th.doing_okay_min}, teal {th.doing_okay_min} and above. These cut-offs are unvalidated defaults.</p>
      </section>

      <section className="card">
        <h2 className="text-lg font-semibold">Emotional tone per check-in</h2>
        {loading ? (
          <Skeleton className="mt-3 h-52" />
        ) : emo.length === 0 ? (
          <p className="mt-2 text-sm text-muted">The mix of emotions in what you wrote (and in your face or voice, if you used those) appears here.</p>
        ) : (
          <div className="mt-3 h-60 w-full" role="img" aria-label="Stacked bar chart of the emotion mix per check-in">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={emo} margin={{ top: 8, right: 12, left: -16, bottom: 0 }}>
                <CartesianGrid stroke={c.grid} vertical={false} />
                <XAxis dataKey="label" stroke={c.text} tick={{ fill: c.text, fontSize: 12 }} tickLine={false} />
                <YAxis unit="%" domain={[0, 100]} allowDataOverflow ticks={[0, 25, 50, 75, 100]} stroke={c.text} tick={{ fill: c.text, fontSize: 12 }} tickLine={false} axisLine={false} />
                <Tooltip contentStyle={{ background: c.surface, border: `1px solid ${c.grid}`, borderRadius: 12, color: c.text }} />
                <Legend wrapperStyle={{ color: c.text, fontSize: 12 }} formatter={(v) => `${EMOJI[v]} ${v}`} />
                {CLASSES.map((k) => <Bar key={k} dataKey={k} stackId="e" fill={COLORS[k]} />)}
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </section>

      <section>
        <h2 className="mb-3 text-lg font-semibold">Past check-ins</h2>
        {loading ? (
          <div className="grid gap-4 sm:grid-cols-2"><Skeleton className="h-28" /><Skeleton className="h-28" /></div>
        ) : sessions.length === 0 ? (
          <p className="text-sm text-muted">Nothing here yet.</p>
        ) : (
          <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {[...sessions].reverse().map((s) => {
              const tier = s.tier ? TIER[s.tier] : null;
              return (
                <li key={s.sessionId} className="glass rounded-3xl p-4">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-sm text-muted">{new Date(s.t).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })}</span>
                    {tier && <Pill tone={tier.color}>{tier.label}</Pill>}
                  </div>
                  <div className="mt-2 flex items-baseline gap-2">
                    <span className="text-3xl font-semibold tabular-nums">{s.score != null ? Math.round(s.score) : '--'}</span>
                    <span className="text-xs text-muted">/ 100{s.mood != null ? ` · mood ${s.mood}/10` : ''}</span>
                  </div>
                  {s.themes?.length > 0 && <div className="mt-2 flex flex-wrap gap-1.5">{s.themes.slice(0, 4).map((t) => <Pill key={t} tone="muted">{t}</Pill>)}</div>}
                </li>
              );
            })}
          </ul>
        )}
      </section>

      <DetectionLog />
    </div>
  );
}
