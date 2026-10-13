import { CLASSES, COLORS, SERIES_WINDOW_MS } from '../lib/emotion.js';

const W = 300;
const H = 120;

// Smoothed probability of each class over the last 30 seconds. Pure SVG, redrawn on each result.
export default function RollingChart({ series }) {
  const end = series.length ? series[series.length - 1].t : Date.now();
  const start = end - SERIES_WINDOW_MS;
  const x = (t) => ((t - start) / SERIES_WINDOW_MS) * W;
  const y = (p) => H - 4 - p * (H - 8);

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="h-32 w-full rounded-2xl bg-solid ring-1 ring-line" role="img" aria-label="Probabilities over the last 30 seconds">
        {[0.25, 0.5, 0.75].map((g) => (
          <line key={g} x1="0" x2={W} y1={y(g)} y2={y(g)} stroke="var(--line)" strokeWidth="1" />
        ))}
        {series.length > 1 &&
          CLASSES.map((c) => (
            <polyline
              key={c}
              fill="none"
              stroke={COLORS[c]}
              strokeWidth="1.75"
              strokeLinejoin="round"
              points={series.map((s) => `${x(s.t).toFixed(1)},${y(s.probs[c]).toFixed(1)}`).join(' ')}
            />
          ))}
        {series.length <= 1 && (
          <text x={W / 2} y={H / 2} textAnchor="middle" fontSize="11" fill="var(--muted)">
            waiting for face data
          </text>
        )}
      </svg>
      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted">
        {CLASSES.map((c) => (
          <span key={c} className="inline-flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-full" style={{ background: COLORS[c] }} />
            {c}
          </span>
        ))}
      </div>
    </div>
  );
}
