import {
  Bar, BarChart, CartesianGrid, Cell, LabelList, Legend, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { CLASSES } from "../api";
import { useScheme } from "../theme";

const pct = (v) => (v == null ? "–" : `${(v * 100).toFixed(1)}%`);

function TooltipBox({ title, rows, chrome }) {
  return (
    <div className="rounded-md border border-black/10 bg-white px-3 py-2 text-xs shadow-md dark:border-white/10 dark:bg-neutral-900">
      {title && <div className="mb-1 font-semibold">{title}</div>}
      {rows.map((r) => (
        <div key={r.name} className="flex items-center gap-2 tabular">
          <span className="inline-block h-2 w-2 rounded-full" style={{ background: r.color }} />
          <span style={{ color: chrome.text }} className="w-16">{r.name}</span>
          <span className="ml-auto font-medium">{pct(r.value)}</span>
        </div>
      ))}
    </div>
  );
}

/** Horizontal bars: one row per emotion, labelled with its name and value. */
export function ProbBars({ probs, height = 230, dimmed = false }) {
  const { emotion, chrome } = useScheme();
  const data = CLASSES.map((name, i) => ({ name, value: probs ? probs[i] : 0, i }));
  const top = probs ? probs.indexOf(Math.max(...probs)) : -1;
  return (
    <div style={{ height, opacity: dimmed ? 0.45 : 1 }} aria-label="Emotion probabilities">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 52, left: 4, bottom: 4 }} barCategoryGap={4}>
          <CartesianGrid horizontal={false} stroke={chrome.grid} />
          <XAxis type="number" domain={[0, 1]} tickFormatter={(v) => `${v * 100}%`} stroke={chrome.axis}
            tick={{ fill: chrome.muted, fontSize: 11 }} ticks={[0, 0.25, 0.5, 0.75, 1]} />
          <YAxis type="category" dataKey="name" width={64} stroke={chrome.axis}
            tick={({ x, y, payload }) => (
              <text x={x - 4} y={y} dy={4} textAnchor="end" fontSize={12}
                fontWeight={CLASSES.indexOf(payload.value) === top ? 700 : 400} fill={chrome.text}>{payload.value}</text>
            )} />
          <Tooltip cursor={{ fill: chrome.grid, opacity: 0.4 }}
            content={({ active, payload }) => active && payload?.length ? (
              <TooltipBox chrome={chrome} rows={[{ name: payload[0].payload.name, value: payload[0].value, color: emotion[payload[0].payload.i] }]} />
            ) : null} />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} isAnimationActive={false}>
            {data.map((d) => <Cell key={d.name} fill={emotion[d.i]} />)}
            <LabelList dataKey="value" position="right" formatter={pct}
              style={{ fill: chrome.text, fontSize: 11, fontVariantNumeric: "tabular-nums" }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/**
 * One line per emotion over time. `points` are {t, probs | null}; a null probs
 * leaves a gap (no face in that frame).
 */
export function EmotionTimeline({ points, domain, cursor, onSeek, height = 220, unit = "s" }) {
  const { emotion, chrome } = useScheme();
  const data = points.map((p) => {
    const row = { t: p.t };
    CLASSES.forEach((c, i) => { row[c] = p.probs ? p.probs[i] : null; });
    return row;
  });
  return (
    <div style={{ height }} className={onSeek ? "cursor-pointer" : ""}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 12, left: -8, bottom: 0 }}
          onClick={(e) => { if (onSeek && e?.activeLabel != null) onSeek(Number(e.activeLabel)); }}>
          <CartesianGrid vertical={false} stroke={chrome.grid} />
          <XAxis dataKey="t" type="number" domain={domain ?? ["dataMin", "dataMax"]} allowDataOverflow
            tickFormatter={(v) => `${Math.round(v * 10) / 10}${unit}`} stroke={chrome.axis}
            tick={{ fill: chrome.muted, fontSize: 11 }} />
          <YAxis domain={[0, 1]} ticks={[0, 0.5, 1]} tickFormatter={(v) => `${v * 100}%`} stroke={chrome.axis}
            tick={{ fill: chrome.muted, fontSize: 11 }} />
          <Tooltip
            content={({ active, payload, label }) => active && payload?.length ? (
              <TooltipBox chrome={chrome} title={`${Number(label).toFixed(1)}${unit}`}
                rows={[...payload].sort((a, b) => (b.value ?? -1) - (a.value ?? -1))
                  .map((p) => ({ name: p.dataKey, value: p.value, color: p.color }))} />
            ) : null} />
          <Legend iconType="plainline" wrapperStyle={{ fontSize: 12, color: chrome.text }} />
          {CLASSES.map((c, i) => (
            <Line key={c} dataKey={c} stroke={emotion[i]} strokeWidth={2} dot={false}
              isAnimationActive={false} connectNulls={false} />
          ))}
          {cursor != null && <ReferenceLine x={cursor} stroke={chrome.text} strokeDasharray="3 3" />}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
