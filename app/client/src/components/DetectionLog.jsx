import { History } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';
import { getHistory } from '../lib/api.js';
import { COLORS, pct } from '../lib/emotion.js';
import { EmptyState, Skeleton } from './ui.jsx';

// Last 50 stored detection summaries (newest first). Images are never stored, so none are shown.
export default function DetectionLog() {
  const [items, setItems] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      setItems((await getHistory()).items);
    } catch (err) {
      setItems([]);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <section className="card">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">Detection log</h2>
          <p className="text-xs text-muted">Face and audio results from uploads, and from Live sessions when "Save session" is ticked.</p>
        </div>
        <button type="button" onClick={load} disabled={loading} className="btn-soft">
          {loading ? 'Loading…' : 'Refresh'}
        </button>
      </div>
      {items === null && (
        <div className="space-y-2" aria-busy="true">
          <Skeleton className="h-8" />
          <Skeleton className="h-8" />
        </div>
      )}
      {error && <p className="mb-3 rounded-2xl bg-coral-soft px-4 py-2 text-sm" role="status">{error}</p>}
      {items && items.length === 0 && !error && <EmptyState icon={History} title="Nothing logged yet" text="Uploads are always logged; live results only when you tick Save session." />}
      {items && items.length > 0 && (
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead className="text-left text-xs uppercase tracking-wide text-muted">
              <tr>
                <th className="py-2 pr-3">Time</th>
                <th className="py-2 pr-3">Source</th>
                <th className="py-2 pr-3">Top label</th>
                <th className="py-2 pr-3">Confidence</th>
                <th className="py-2">Face</th>
              </tr>
            </thead>
            <tbody>
              {items.map((it) => (
                <tr key={it._id} className="border-t border-line">
                  <td className="whitespace-nowrap py-2 pr-3 tabular-nums">{new Date(it.timestamp).toLocaleString()}</td>
                  <td className="py-2 pr-3">{it.source}</td>
                  <td className="py-2 pr-3 capitalize" style={{ color: COLORS[it.top_label] || 'var(--muted)' }}>{it.top_label || '-'}</td>
                  <td className="py-2 pr-3 tabular-nums">{it.confidence != null ? pct(it.confidence) : '-'}</td>
                  <td className="py-2">{it.audio ? 'n/a' : it.face_found ? 'yes' : 'no'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
