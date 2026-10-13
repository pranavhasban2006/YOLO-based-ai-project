import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { EMOJI } from '../lib/emotion.js';
import { chartColors, useTheme } from '../lib/theme.jsx';
import { Pill } from './ui.jsx';

const NAME = { face: 'Face', audio: 'Voice', text: 'Text', eeg: 'EEG' };
const fmt = (v, d = 0) => (v === null || v === undefined ? '--' : Number(v).toFixed(d));

// "How your score was made": per-model values, the weights used and the confidence.
export function ModelPanel({ result }) {
  const mods = result.modalities;
  if (!mods) return null;
  const conf = result.confidence_value;
  return (
    <section className="card" aria-labelledby="how-made">
      <h3 id="how-made" className="text-lg font-semibold">How your score was made</h3>
      <p className="mt-1 text-sm text-muted">
        {result.mode === 'models'
          ? 'The score blends the emotion models we trained (face, voice, text). Each model\'s output is turned into a 0 to 100 value, and models are weighted by how well they held up on data they had never seen.'
          : 'This score uses the questionnaire-style mode (conversation and model values together).'}
      </p>
      <div className="mt-3 overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead className="text-left text-xs uppercase tracking-wide text-muted">
            <tr><th className="py-1.5 pr-4">Model</th><th className="py-1.5 pr-4">Value</th><th className="py-1.5 pr-4">Weight</th><th className="py-1.5 pr-4">Reliability</th><th className="py-1.5">Seen as</th></tr>
          </thead>
          <tbody>
            {Object.entries(mods).map(([m, v]) => (
              <tr key={m} className="border-t border-line">
                <td className="py-2 pr-4 font-medium">{NAME[m]}</td>
                <td className="py-2 pr-4 tabular-nums">{v.available ? fmt(v.value) : '--'}</td>
                <td className="py-2 pr-4 tabular-nums">{v.weight != null ? `${Math.round(v.weight * 100)}%` : '--'}</td>
                <td className="py-2 pr-4 tabular-nums">{v.reliability != null ? v.reliability.toFixed(2) : v.note || '--'}</td>
                <td className="py-2">{v.available && v.top ? <span className="capitalize">{EMOJI[v.top]} {v.top}</span> : <span className="text-muted">{v.note || 'not used'}</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {conf != null && (
        <div className="mt-4">
          <div className="mb-1 flex items-center justify-between text-sm">
            <span>Confidence</span>
            <span className="tabular-nums">{Math.round(conf * 100)}% {result.low_confidence && <Pill tone="violet">low confidence</Pill>}</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-line" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(conf * 100)}>
            <div className="h-full rounded-full bg-accent" style={{ width: `${Math.round(conf * 100)}%` }} />
          </div>
          <p className="mt-1 text-xs text-muted">Confidence is the share of the model evidence that was actually available this time. With only one model, treat the score as a rough hint.</p>
        </div>
      )}
      <details className="mt-3 text-xs text-muted">
        <summary className="cursor-pointer font-medium text-ink">Read this first</summary>
        <p className="mt-2">
          The mapping from emotion to a wellbeing value is a simple heuristic, not a clinical measure. The models were trained on acted or other-domain data and generalise weakly to new people and settings. This is a screening aid and cannot tell you what is wrong.
        </p>
      </details>
      {result.questionnaire && (result.questionnaire.mood?.value != null || result.questionnaire.phq?.value != null) && !result.questionnaire.used_in_score && (
        <p className="mt-3 rounded-2xl bg-violet-soft px-3 py-2 text-xs">
          Also noted from the conversation (shown, not used in the score): mood about {fmt(result.questionnaire.mood?.mean_mood, 1)} out of 10
          {result.questionnaire.phq?.answered ? `, ${result.questionnaire.phq.answered} daily-life topics touched` : ''}.
        </p>
      )}
    </section>
  );
}

const PROS = [
  ['speech_rate_wps', 'Speech rate', 'words/s', 2],
  ['pause_ratio', 'Pause share', '', 2],
  ['pause_mean_s', 'Pause length', 's', 2],
  ['latency_s', 'Time to start', 's', 2],
  ['pitch_std_st', 'Pitch variability', 'st', 2],
  ['energy_mean_db', 'Loudness', 'dB', 1],
];

// Voice interview extras: timeline of face and voice emotion, prosody relative to the person's first answer, transcript.
export function VoiceDetails({ result }) {
  const { dark } = useTheme();
  const c = chartColors(dark);
  const tl = result.timeline || [];
  if (!tl.length) return null;
  const data = tl.map((t) => ({ answer: `A${t.index + 1}`, face: t.face?.value ?? null, voice: t.voice?.value ?? null }));
  const base = tl[0]?.prosody;
  return (
    <>
      <section className="card">
        <h3 className="text-lg font-semibold">Across your answers</h3>
        <p className="text-xs text-muted">Face and voice emotion per answer, as a 0 to 100 value (50 is neutral).</p>
        <div className="mt-3 h-56 w-full" role="img" aria-label="Line chart of face and voice emotion per answer">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 8, right: 12, left: -16, bottom: 0 }}>
              <CartesianGrid stroke={c.grid} vertical={false} />
              <XAxis dataKey="answer" stroke={c.text} tick={{ fill: c.text, fontSize: 12 }} tickLine={false} />
              <YAxis domain={[0, 100]} stroke={c.text} tick={{ fill: c.text, fontSize: 12 }} tickLine={false} axisLine={false} />
              <Tooltip contentStyle={{ background: c.surface, border: `1px solid ${c.grid}`, borderRadius: 12, color: c.text }} />
              <Legend wrapperStyle={{ color: c.text, fontSize: 12 }} />
              <Line type="monotone" dataKey="face" name="Face" stroke={c.accent} strokeWidth={2.5} connectNulls dot={{ r: 4 }} />
              <Line type="monotone" dataKey="voice" name="Voice" stroke={c.violet} strokeWidth={2.5} connectNulls dot={{ r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="card">
        <h3 className="text-lg font-semibold">How you spoke</h3>
        <p className="text-xs text-muted">
          Heuristic indicators only, shown for your information (they are not used in the score). Your first answer is your own baseline, so later answers are compared with <em>you</em>, not with other people.
        </p>
        <div className="mt-3 overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead className="text-left text-xs uppercase tracking-wide text-muted">
              <tr><th className="py-1.5 pr-3">Answer</th>{PROS.map((p) => <th key={p[0]} className="py-1.5 pr-3">{p[1]}</th>)}</tr>
            </thead>
            <tbody>
              {tl.map((t) => (
                <tr key={t.index} className="border-t border-line">
                  <td className="py-2 pr-3 font-medium">A{t.index + 1}{t.is_baseline && <span className="ml-1 text-xs text-muted">(baseline)</span>}</td>
                  {PROS.map(([k, , unit, d]) => {
                    const v = t.prosody?.[k];
                    const delta = t.deltas?.[k];
                    const dText = delta == null ? '' : k === 'energy_mean_db' ? `${delta > 0 ? '+' : ''}${delta.toFixed(1)} dB` : `${delta > 0 ? '+' : ''}${Math.round(delta * 100)}%`;
                    return (
                      <td key={k} className="py-2 pr-3 tabular-nums">
                        {v == null ? '--' : v.toFixed(d)} <span className="text-xs text-muted">{unit}</span>
                        {dText && <div className="text-xs text-muted">{dText}</div>}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!base && <p className="mt-2 text-xs text-muted">No prosody data was captured.</p>}
      </section>

      {result.transcript?.length > 0 && (
        <section className="card">
          <h3 className="text-lg font-semibold">Transcript</h3>
          <ol className="mt-3 space-y-4">
            {tl.map((t) => {
              const tr = result.transcript.find((x) => x.index === t.index);
              return (
                <li key={t.index} className="text-sm">
                  <div className="text-muted">{t.question}</div>
                  <div className="mt-1 rounded-2xl bg-accent-soft px-4 py-2">{tr?.skipped ? <em>(skipped)</em> : tr?.answer}</div>
                  <div className="mt-1.5 flex flex-wrap gap-1.5">
                    {t.face && <Pill tone="accent">face: {EMOJI[t.face.dominant]} {t.face.dominant}</Pill>}
                    {t.voice && <Pill tone="violet">voice: {EMOJI[t.voice.top_label]} {t.voice.top_label}</Pill>}
                    {t.duration_s != null && <Pill tone="muted">{Number(t.duration_s).toFixed(1)} s</Pill>}
                  </div>
                </li>
              );
            })}
          </ol>
        </section>
      )}
    </>
  );
}
