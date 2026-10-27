import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import results from "../data/cascadeResults.json";
import { useScheme } from "../theme";

// Exported by project/code/export_cascade_for_app.py from the cascade.py results (Cascade_results/reports).

const pct = (v, d = 1) => (v == null ? "–" : `${(v * 100).toFixed(d)}%`);
const pts = (a, b) => `${b >= a ? "+" : ""}${((b - a) * 100).toFixed(2)}`;
const name = (r) => `${r.A1} + ${r.A2}`;
const FILTERS = [
  { id: "all", label: "All 20" }, { id: "speech", label: "Speech" }, { id: "face", label: "Face" },
  { id: "text", label: "Text" },
];

function Steps() {
  const box = "rounded-lg border border-black/10 px-3 py-2 dark:border-white/10";
  return (
    <div className="grid gap-2 text-sm sm:grid-cols-[1fr_auto_1fr_auto_1fr] sm:items-center">
      <div className={box}><b>Stage 1 · A1 alone</b><br /><span className="text-stone-500">A1 is trained and tested by itself, e.g. the CNN gets 84 of 100 right.</span></div>
      <span className="hidden text-center text-stone-400 sm:block">→</span>
      <div className={box}><b>A1's wrong ones → A2</b><br /><span className="text-stone-500">Only the samples A1 got wrong are fed (as A1's output) to A2, e.g. the GRU.</span></div>
      <span className="hidden text-center text-stone-400 sm:block">→</span>
      <div className={box}><b>Final = A1 right + A2 fixed</b><br /><span className="text-stone-500">A1's right answers are kept; A2 corrects some of the rest.</span></div>
    </div>
  );
}

function Kpis({ rows }) {
  const gains = rows.map((r) => r.wrongOnly.acc - r.a1.acc);
  const best = rows[gains.indexOf(Math.max(...gains))];
  const fixed = rows.reduce((s, r) => s + r.wrongOnly.fixed, 0);
  const fed = rows.reduce((s, r) => s + r.wrongOnly.fed, 0);
  const tiles = [
    ["Hybrids improved", `${gains.filter((g) => g > 0).length} / ${rows.length}`],
    ["Average gain", `+${(100 * gains.reduce((a, b) => a + b, 0) / gains.length).toFixed(2)} pts`],
    ["A1 mistakes fixed by A2", `${fixed.toLocaleString()} / ${fed.toLocaleString()}`],
    ["Largest gain", `${name(best)} (${pts(best.a1.acc, best.wrongOnly.acc)} pts)`],
  ];
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      {tiles.map(([k, v]) => (
        <div key={k} className="card py-3">
          <div className="text-xs uppercase tracking-wide text-stone-500">{k}</div>
          <div className="mt-1 font-semibold tabular">{v}</div>
        </div>
      ))}
    </div>
  );
}

function GainChart({ rows }) {
  const { emotion, chrome } = useScheme();
  const data = rows.map((r) => ({ name: `${name(r)}${r.modality === "text" ? ` (${r.dataset})` : ""}`, a1: r.a1.acc, final: r.wrongOnly.acc }));
  const lo = Math.floor(Math.min(...rows.map((r) => r.a1.acc)) * 10) / 10;
  return (
    <div style={{ height: 44 + rows.length * 30 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, left: 4, bottom: 4 }} barGap={1} barCategoryGap={6}>
          <CartesianGrid horizontal={false} stroke={chrome.grid} />
          <XAxis type="number" domain={[lo, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} stroke={chrome.axis}
            tick={{ fill: chrome.muted, fontSize: 11 }} />
          <YAxis type="category" dataKey="name" width={250} stroke={chrome.axis} tick={{ fill: chrome.text, fontSize: 11 }} />
          <Tooltip cursor={{ fill: chrome.grid, opacity: 0.4 }} formatter={(v) => pct(v, 2)}
            contentStyle={{ background: chrome.surface, border: `1px solid ${chrome.grid}`, fontSize: 12 }} />
          <Legend wrapperStyle={{ fontSize: 12, color: chrome.text }} />
          <Bar dataKey="a1" name="Stage 1: A1 alone" fill={chrome.muted} isAnimationActive={false} />
          <Bar dataKey="final" name="A1 + A2 (A1's wrong ones → A2)" fill={emotion[0]} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function Confusion({ title, m, classes }) {
  if (!m) return <div className="text-sm text-stone-500">{title}: not available</div>;
  const max = Math.max(...m.flat());
  return (
    <div className="min-w-0">
      <div className="mb-1 text-sm font-medium">{title}</div>
      <div className="overflow-x-auto">
        <table className="text-[11px] tabular">
          <thead>
            <tr><th className="px-1 text-left text-stone-500">true ↓ / pred →</th>
              {classes.map((c) => <th key={c} className="px-1 font-normal text-stone-500">{c.slice(0, 4)}</th>)}</tr>
          </thead>
          <tbody>
            {m.map((row, i) => (
              <tr key={classes[i]}>
                <td className="pr-2 text-stone-500">{classes[i]}</td>
                {row.map((v, j) => (
                  <td key={j} className="h-7 w-10 text-center"
                    style={{ background: `rgba(42,120,214,${(v / max) * 0.85})`, color: v / max > 0.5 ? "#fff" : undefined,
                      outline: i === j ? "1px solid rgba(0,0,0,0.25)" : undefined }}>{v}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Detail({ r, onClose }) {
  const final = r.a1.correct + r.wrongOnly.fixed;
  return (
    <section className="card space-y-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="font-semibold">{name(r)}</h3>
          <p className="text-xs text-stone-500">{r.modality} · {r.dataset} · split: {r.split}</p>
        </div>
        <button className="btn-ghost" onClick={onClose}>Close</button>
      </div>
      <ol className="space-y-1 text-sm">
        <li><b>Stage 1.</b> {r.A1} alone, on {r.testSize.toLocaleString()} test samples: {r.a1.correct.toLocaleString()} right, {r.a1.wrong.toLocaleString()} wrong → <b>{pct(r.a1.acc, 2)}</b></li>
        <li><b>Stage 2.</b> The {r.wrongOnly.fed.toLocaleString()} wrong ones are fed to {r.A2}, which fixes <b>{r.wrongOnly.fixed.toLocaleString()}</b> of them.</li>
        <li><b>Final.</b> {r.a1.correct.toLocaleString()} + {r.wrongOnly.fixed.toLocaleString()} = {final.toLocaleString()} right → <b>{pct(r.wrongOnly.acc, 2)}</b> ({pts(r.a1.acc, r.wrongOnly.acc)} points)</li>
        <li className="text-stone-500">For comparison, if A2 re-checks every sample (no labels used): {pct(r.allSamples.acc, 2)}; it fixes {r.allSamples.fixed} but also changes {r.allSamples.broken} right answers.</li>
      </ol>
      <div className="grid gap-4 lg:grid-cols-2">
        <Confusion title={`${r.A1} alone`} m={r.confusionA1} classes={r.classes} />
        <Confusion title={`After A2 (wrong ones only)`} m={r.confusionFinal} classes={r.classes} />
      </div>
    </section>
  );
}

export default function CascadeTab() {
  const [filter, setFilter] = useState("all");
  const [selected, setSelected] = useState(null);
  const rows = useMemo(() => results.filter((r) => filter === "all" || r.modality === filter), [filter]);
  const sel = results.find((r) => r.id === selected);

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold">A1 → A2 results: every hybrid as a two-stage model</h2>
        <p className="text-sm text-stone-500">Each hybrid "A1 + A2" is run in two stages on its held-out test set.</p>
      </div>
      <Steps />
      <Kpis rows={results} />

      <div className="flex flex-wrap gap-1" role="tablist">
        {FILTERS.map((f) => (
          <button key={f.id} role="tab" aria-selected={filter === f.id} onClick={() => setFilter(f.id)}
            className={`btn ${filter === f.id ? "bg-stone-900 text-white dark:bg-white dark:text-black" : "hover:bg-black/5 dark:hover:bg-white/10"}`}>
            {f.label}
          </button>
        ))}
      </div>

      <section className="card">
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-stone-500">Test accuracy: A1 alone vs A1 + A2</h3>
        <GainChart rows={rows} />
      </section>

      {sel && <Detail r={sel} onClose={() => setSelected(null)} />}

      <div className="card overflow-x-auto p-0">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-black/10 text-xs uppercase tracking-wide text-stone-500 dark:border-white/10">
            <tr>
              <th className="px-3 py-2">Hybrid (A1 + A2)</th><th className="px-3 py-2">Dataset</th>
              <th className="px-3 py-2 text-right">Test</th><th className="px-3 py-2 text-right">A1 alone</th>
              <th className="px-3 py-2 text-right">A1 wrong → A2</th><th className="px-3 py-2 text-right">Fixed</th>
              <th className="px-3 py-2 text-right">A1 + A2</th><th className="px-3 py-2 text-right">Gain (pts)</th>
              <th className="px-3 py-2 text-right">F1 A1 → A1+A2</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} onClick={() => setSelected(r.id === selected ? null : r.id)}
                className={`cursor-pointer border-b border-black/5 hover:bg-black/5 dark:border-white/5 dark:hover:bg-white/5 ${selected === r.id ? "bg-blue-50 dark:bg-blue-950/30" : ""}`}>
                <td className="px-3 py-2">{name(r)}</td>
                <td className="px-3 py-2 text-stone-500">{r.dataset}</td>
                <td className="px-3 py-2 text-right tabular">{r.testSize.toLocaleString()}</td>
                <td className="px-3 py-2 text-right tabular">{pct(r.a1.acc, 2)}</td>
                <td className="px-3 py-2 text-right tabular">{r.wrongOnly.fed.toLocaleString()}</td>
                <td className="px-3 py-2 text-right tabular">{r.wrongOnly.fixed.toLocaleString()}</td>
                <td className="px-3 py-2 text-right font-semibold tabular">{pct(r.wrongOnly.acc, 2)}</td>
                <td className="px-3 py-2 text-right tabular text-green-700 dark:text-green-400">{pts(r.a1.acc, r.wrongOnly.acc)}</td>
                <td className="px-3 py-2 text-right tabular text-stone-500">{r.a1.f1?.toFixed(3)} → {r.wrongOnly.f1 == null ? "–" : r.wrongOnly.f1.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-stone-500">
        Click a row for the step-by-step numbers and confusion matrices. Splits: face FER-2013 90/10 train/validation +
        official test set · speech 70/15/15 · text official train/validation/test. Choosing A1's wrong samples uses the true
        labels, as in the evaluation protocol; in live use A2 would have to re-check every sample (shown per hybrid).
      </p>
    </div>
  );
}
