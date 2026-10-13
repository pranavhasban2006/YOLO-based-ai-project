import { animate, motion, useReducedMotion } from 'framer-motion';
import { useEffect, useState } from 'react';

export function PageHeader({ title, subtitle, children }) {
  return (
    <header className="mb-5 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">{title}</h1>
        {subtitle && <p className="mt-1 max-w-2xl text-sm text-muted">{subtitle}</p>}
      </div>
      {children}
    </header>
  );
}

export const Skeleton = ({ className = 'h-6 w-full' }) => <div className={`skeleton ${className}`} aria-hidden />;

export function EmptyState({ icon: Icon, title, text, action }) {
  return (
    <div className="flex flex-col items-center gap-2 px-4 py-10 text-center">
      {Icon && (
        <span className="mb-1 grid h-14 w-14 place-items-center rounded-2xl bg-accent-soft text-accent">
          <Icon className="h-7 w-7" aria-hidden />
        </span>
      )}
      <h3 className="text-lg font-semibold">{title}</h3>
      {text && <p className="max-w-sm text-sm text-muted">{text}</p>}
      {action}
    </div>
  );
}

// Animated count-up that respects prefers-reduced-motion.
export function useCountUp(value, duration = 1.1) {
  const reduce = useReducedMotion();
  const [shown, setShown] = useState(reduce ? value : 0);
  useEffect(() => {
    if (value === null || value === undefined) return undefined;
    if (reduce) {
      setShown(value);
      return undefined;
    }
    const c = animate(0, value, { duration, ease: 'easeOut', onUpdate: (v) => setShown(v) });
    return () => c.stop();
  }, [value, duration, reduce]);
  return shown;
}

// Animated ring gauge, 0-100. `color` is a CSS colour (use a var()).
export function Gauge({ value, color = 'var(--accent)', size = 168, label, sublabel, empty = false }) {
  const reduce = useReducedMotion();
  const r = 52;
  const c = 2 * Math.PI * r;
  const v = value === null || value === undefined ? 0 : Math.max(0, Math.min(100, value));
  const shown = useCountUp(empty ? null : value);
  return (
    <div className="relative inline-block" style={{ width: size, height: size }} role="img" aria-label={empty ? 'No score yet' : `Wellbeing score ${Math.round(v)} out of 100`}>
      <svg viewBox="0 0 120 120" className="h-full w-full -rotate-90">
        <circle cx="60" cy="60" r={r} fill="none" stroke="var(--line)" strokeWidth="10" />
        {!empty && (
          <motion.circle
            cx="60" cy="60" r={r} fill="none" stroke={color} strokeWidth="10" strokeLinecap="round"
            strokeDasharray={c}
            initial={{ strokeDashoffset: reduce ? c * (1 - v / 100) : c }}
            animate={{ strokeDashoffset: c * (1 - v / 100) }}
            transition={{ duration: reduce ? 0 : 1.1, ease: 'easeOut' }}
          />
        )}
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
        <span className="text-4xl font-semibold tabular-nums">{empty ? '--' : Math.round(shown)}</span>
        {label && <span className="text-xs text-muted">{label}</span>}
        {sublabel && <span className="px-4 text-[11px] leading-tight text-muted">{sublabel}</span>}
      </div>
    </div>
  );
}

export function Pill({ children, tone = 'accent' }) {
  const map = { accent: 'bg-accent-soft text-accent', violet: 'bg-violet-soft text-violet', coral: 'bg-coral-soft text-coral', muted: 'bg-line text-muted' };
  return <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ${map[tone]}`}>{children}</span>;
}
