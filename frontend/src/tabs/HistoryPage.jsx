import { useEffect, useState } from "react";
import { api } from "../api";
import { EmotionTimeline, ProbBars } from "../components/Charts";
import { ErrorBox, MockBadge } from "../components/Notices";

const fmt = (iso) => (iso ? new Date(iso).toLocaleString() : "–");

function SessionDetail({ id, onClose }) {
  const [s, setS] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    setS(null);
    api.session(id).then(setS).catch(setError);
  }, [id]);
  if (error) return <ErrorBox error={error} onDismiss={onClose} />;
  if (!s) return <p className="text-sm text-stone-500">Loading session {id}…</p>;

  // Video sessions store a timeline; streaming sessions store one prediction per frame.
  const start = s.predictions[0] ? new Date(s.predictions[0].timestamp).getTime() : 0;
  const points = s.meta.timeline
    ? s.meta.timeline.map((p) => ({ t: p.t, probs: p.probs }))
    : s.predictions.length > 1
      ? s.predictions.map((p) => ({ t: (new Date(p.timestamp).getTime() - start) / 1000, probs: p.probs }))
      : null;
  const summary = s.meta.mean_probs ?? (s.predictions.length === 1 ? s.predictions[0].probs : null);

  return (
    <section className="card space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold">Session #{s.id} · <span className="capitalize">{s.kind}</span></h3>
        <button className="btn-ghost" onClick={onClose}>Close</button>
      </div>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm sm:grid-cols-4">
        <dt className="text-stone-500">Started</dt><dd>{fmt(s.created_at)}</dd>
        <dt className="text-stone-500">Ended</dt><dd>{fmt(s.ended_at)}</dd>
        <dt className="text-stone-500">Result</dt>
        <dd className="capitalize">{s.label ?? "–"} {s.confidence != null && `(${(s.confidence * 100).toFixed(0)}%)`}</dd>
        <dt className="text-stone-500">Predictions</dt><dd>{s.predictions.length}</dd>
        {s.meta.model && (<><dt className="text-stone-500">Model</dt><dd>{s.meta.model}</dd></>)}
        {s.meta.contributing && (<><dt className="text-stone-500">Fused from</dt><dd>{s.meta.contributing.join(", ")}</dd></>)}
      </dl>
      {s.mock && <MockBadge />}
      {summary && <ProbBars probs={summary} dimmed={s.mock} height={200} />}
      {points && <EmotionTimeline points={points} height={200} />}
    </section>
  );
}

export default function HistoryPage() {
  const [sessions, setSessions] = useState(null);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(null);

  function load() {
    setError(null);
    api.sessions().then(setSessions).catch(setError);
  }
  useEffect(load, []);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Past sessions</h2>
        <button className="btn-ghost" onClick={load}>Refresh</button>
      </div>
      <ErrorBox error={error} />
      {selected && <SessionDetail id={selected} onClose={() => setSelected(null)} />}
      {sessions && sessions.length === 0 && <p className="text-sm text-stone-500">No sessions yet. Results appear here after you analyse something.</p>}
      {sessions && sessions.length > 0 && (
        <div className="card overflow-x-auto p-0">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-black/10 text-xs uppercase tracking-wide text-stone-500 dark:border-white/10">
              <tr>
                <th className="px-3 py-2">#</th><th className="px-3 py-2">Type</th><th className="px-3 py-2">When</th>
                <th className="px-3 py-2">Result</th><th className="px-3 py-2 text-right">Predictions</th><th className="px-3 py-2" />
              </tr>
            </thead>
            <tbody>
              {sessions.map((s) => (
                <tr key={s.id} onClick={() => setSelected(s.id)}
                  className={`cursor-pointer border-b border-black/5 hover:bg-black/5 dark:border-white/5 dark:hover:bg-white/5 ${selected === s.id ? "bg-blue-50 dark:bg-blue-950/30" : ""}`}>
                  <td className="px-3 py-2 tabular">{s.id}</td>
                  <td className="px-3 py-2 capitalize">{s.kind}</td>
                  <td className="px-3 py-2 tabular">{fmt(s.created_at)}</td>
                  <td className="px-3 py-2 capitalize">
                    {s.label ?? <span className="text-stone-500">–</span>}
                    {s.confidence != null && <span className="text-stone-500"> · {(s.confidence * 100).toFixed(0)}%</span>}
                  </td>
                  <td className="px-3 py-2 text-right tabular">{s.n_predictions}</td>
                  <td className="px-3 py-2">{s.mock && <MockBadge />}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
