import { ProbBars } from "./Charts";
import { MockBadge } from "./Notices";

const pct = (v) => `${(v * 100).toFixed(1)}%`;

export function ResultHeadline({ result }) {
  return (
    <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
      <span className="text-3xl font-semibold capitalize">{result.label}</span>
      <span className="text-sm text-stone-500 dark:text-stone-400">{pct(result.confidence)} confidence</span>
      {result.mock && <MockBadge />}
    </div>
  );
}

export default function ResultCard({ title, result, children }) {
  if (!result) return null;
  return (
    <section className={`card space-y-3 ${result.mock ? "border-2 border-dashed !border-amber-500" : ""}`}>
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold uppercase tracking-wide text-stone-500 dark:text-stone-400">{title}</h3>
        <span className="text-xs text-stone-500 tabular">{new Date(result.timestamp).toLocaleTimeString()}</span>
      </div>
      <ResultHeadline result={result} />
      <ProbBars probs={result.probs} dimmed={result.mock} />
      {children}
    </section>
  );
}
