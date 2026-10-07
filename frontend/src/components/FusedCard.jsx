import { useState } from "react";
import { api, corePrediction } from "../api";
import { ProbBars } from "./Charts";
import { ErrorBox, MockBadge } from "./Notices";
import { ResultHeadline } from "./ResultCard";

const MODALITIES = ["face", "audio", "text", "eeg"];
const REASONS = {
  mock: "placeholder data",
  not_supplied: "no result yet",
  duplicate: "duplicate",
  unknown_modality: "unknown modality",
};

/** Fuses the most recent result from each modality on the backend. */
export default function FusedCard({ latest }) {
  const [fused, setFused] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const supplied = MODALITIES.filter((m) => latest[m]);
  const usable = supplied.filter((m) => !latest[m].mock);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      setFused(await api.fuse(supplied.map((m) => corePrediction(latest[m]))));
    } catch (e) {
      setFused(null);
      setError(e);
    } finally {
      setBusy(false);
    }
  }

  const contributed = new Set(fused?.contributing ?? []);
  const skipped = Object.fromEntries((fused?.skipped ?? []).map((s) => [s.modality, s.reason]));

  return (
    <section className="card space-y-3">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-sm font-semibold uppercase tracking-wide text-stone-500 dark:text-stone-400">Fused result</h3>
        <button className="btn-primary" onClick={run} disabled={busy || supplied.length === 0}>
          {busy ? "Fusing…" : "Fuse latest"}
        </button>
      </div>

      <ul className="grid grid-cols-2 gap-2 text-sm">
        {MODALITIES.map((m) => {
          const p = latest[m];
          const state = fused ? (contributed.has(m) ? "used" : "skipped") : p ? (p.mock ? "mock" : "ready") : "none";
          const styles = {
            used: "border-green-600 bg-green-50 dark:bg-green-950/40",
            ready: "border-black/15 dark:border-white/15",
            skipped: "border-dashed border-black/20 opacity-70 dark:border-white/20",
            mock: "border-dashed border-amber-500",
            none: "border-dashed border-black/20 opacity-60 dark:border-white/20",
          }[state];
          return (
            <li key={m} className={`rounded-lg border p-2 ${styles}`}>
              <div className="flex items-center justify-between">
                <span className="font-medium capitalize">{m}</span>
                {state === "used" && <span className="text-xs font-semibold text-green-700 dark:text-green-400">✓ used</span>}
              </div>
              <div className="text-xs text-stone-500 dark:text-stone-400">
                {p ? <span className="capitalize">{p.label} · {(p.confidence * 100).toFixed(0)}%</span> : "no result yet"}
                {fused && skipped[m] && <span> — skipped: {REASONS[skipped[m]] ?? skipped[m]}</span>}
                {!fused && p?.mock && <span> — placeholder, will be skipped</span>}
              </div>
            </li>
          );
        })}
      </ul>

      {supplied.length > 0 && usable.length === 0 && !fused && (
        <p className="text-xs text-stone-500">Only placeholder results so far. Fusion needs at least one real modality (e.g. face).</p>
      )}
      <ErrorBox error={error} />
      {fused && (
        <div className="space-y-2">
          <ResultHeadline result={fused} />
          <p className="text-xs text-stone-500 dark:text-stone-400">
            Confidence-weighted average of: {fused.contributing.join(", ")}
            {Object.keys(fused.weights).length > 1 &&
              ` (weights ${Object.entries(fused.weights).map(([m, w]) => `${m} ${(w * 100).toFixed(0)}%`).join(", ")})`}
          </p>
          <ProbBars probs={fused.probs} height={200} />
          {fused.mock && <MockBadge />}
        </div>
      )}
    </section>
  );
}
