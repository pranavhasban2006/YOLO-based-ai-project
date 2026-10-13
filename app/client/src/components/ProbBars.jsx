import { motion } from 'framer-motion';
import { CLASSES, COLORS, EMOJI, pct } from '../lib/emotion.js';

// Seven animated horizontal bars, one per class, in classes.json order.
export default function ProbBars({ probs, highlight }) {
  return (
    <ul className="space-y-2" aria-label="Class probabilities">
      {CLASSES.map((c) => {
        const p = probs?.[c] ?? 0;
        const isTop = highlight === c;
        return (
          <li key={c} className="grid grid-cols-[6.5rem_1fr_3rem] items-center gap-2 text-sm">
            <span className={`flex items-center gap-1.5 ${isTop ? 'font-semibold text-ink' : 'text-muted'}`}>
              <span aria-hidden>{EMOJI[c]}</span>
              {c}
            </span>
            <div className="h-3 overflow-hidden rounded-full bg-line">
              <motion.div
                className="h-full rounded-full"
                style={{ background: COLORS[c] }}
                initial={false}
                animate={{ width: `${(probs ? p : 0) * 100}%` }}
                transition={{ type: 'spring', stiffness: 160, damping: 22 }}
              />
            </div>
            <span className="text-right tabular-nums text-muted">{probs ? pct(p) : '-'}</span>
          </li>
        );
      })}
    </ul>
  );
}
