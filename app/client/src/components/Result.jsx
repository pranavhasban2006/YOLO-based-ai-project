import { motion } from 'framer-motion';
import { Heart, Lightbulb, Phone, RotateCcw } from 'lucide-react';
import { HELPLINES, TIER, tierMessage } from '../lib/wellbeing.js';
import { ModelPanel, VoiceDetails } from './ResultDetails.jsx';
import { Gauge, Pill } from './ui.jsx';

const EFFECT_TONE = { raises: 'accent', lowers: 'violet', neutral: 'muted' };
const EFFECT_TEXT = { raises: 'helps', lowers: 'weighs on it', neutral: 'neutral' };

// Support block shown for the highest tier: calm, prominent, with helplines and the counsellor contact.
export function SupportBanner({ support }) {
  const lines = support?.helplines || HELPLINES;
  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      role="region"
      aria-label="Support"
      className="rounded-3xl border border-coral/40 bg-coral-soft p-5 sm:p-6"
    >
      <div className="flex items-start gap-3">
        <span className="grid h-10 w-10 shrink-0 place-items-center rounded-2xl bg-coral text-white"><Heart className="h-5 w-5" aria-hidden /></span>
        <div>
          <h2 className="text-lg font-semibold">You do not have to carry this alone</h2>
          <p className="mt-1 text-sm">
            Please reach out to someone today: a friend, a family member, or a person at the college you trust. Trained people are also available for free, any time.
          </p>
        </div>
      </div>
      <ul className="mt-4 grid gap-2 sm:grid-cols-3">
        {lines.map((h) => (
          <li key={h.number} className="rounded-2xl bg-solid p-3">
            <div className="text-xs text-muted">{h.name}</div>
            <a href={`tel:${h.number}`} className="flex items-center gap-2 text-xl font-semibold tabular-nums text-coral">
              <Phone className="h-4 w-4" aria-hidden /> {h.number}
            </a>
          </li>
        ))}
      </ul>
      <p className="mt-4 text-sm">
        College counsellor: <strong>{support?.counsellor_contact || 'your college counselling centre'}</strong>. If you might act on how you feel, call 112 now.
      </p>
    </motion.section>
  );
}

export default function Result({ result, onAgain, onHome }) {
  const tier = result.tier ? TIER[result.tier] : null;
  const colorVar = tier ? `var(--${tier.color})` : 'var(--muted)';
  const showSupport = result.tier === 'needs_support_now';
  return (
    <div className="space-y-5">
      <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="card grid items-center gap-6 sm:grid-cols-[auto_1fr]">
        <div className="flex justify-center">
          <Gauge value={result.score} empty={result.score === null || result.score === undefined} color={colorVar} label="out of 100" />
        </div>
        <div>
          <div className="eyebrow">Your check-in</div>
          <h2 className="mt-1 text-2xl font-semibold">{tier ? tier.label : 'Not enough to say yet'}</h2>
          <p className="mt-2 text-sm text-muted">
            {tier ? tierMessage[result.tier] : 'There was nothing for the emotion models to read this time, so there is no score. In a chat, turn on the emotion check; or try the voice interview.'}
          </p>
          <div className="mt-3 flex flex-wrap gap-2">
            {result.confidence && <Pill tone="muted">Confidence: {result.confidence}{result.confidence_value != null ? ` (${Math.round(result.confidence_value * 100)}%)` : ''}</Pill>}
            {result.low_confidence && <Pill tone="violet">low confidence</Pill>}
            {result.trend?.slope_per_day != null && (
              <Pill tone={result.trend.slope_per_day >= 0 ? 'accent' : 'violet'}>
                7-day trend {result.trend.slope_per_day >= 0 ? '+' : ''}{result.trend.slope_per_day}/day
              </Pill>
            )}
            {result.mock && <Pill tone="violet">Demo mode</Pill>}
          </div>
        </div>
      </motion.section>

      {showSupport && <SupportBanner support={result.support} />}

      <section className="card">
        <h3 className="text-lg font-semibold">What shaped this</h3>
        <ul className="mt-3 space-y-2">
          {(result.reasons || []).map((r, i) => (
            <li key={`${r.kind || r.component}-${i}`} className="flex items-start gap-3 text-sm">
              <Pill tone={EFFECT_TONE[r.effect] || 'muted'}>{r.component ? r.component.replace('phq', 'daily life').replace('emotion', 'tone') : 'note'}{r.effect ? ` · ${EFFECT_TEXT[r.effect]}` : ''}</Pill>
              <span>{r.text}</span>
            </li>
          ))}
        </ul>
      </section>

      <ModelPanel result={result} />
      {result.kind === 'voice' && <VoiceDetails result={result} />}

      {result.suggestions?.length > 0 && (
        <section className="card">
          <h3 className="flex items-center gap-2 text-lg font-semibold"><Lightbulb className="h-5 w-5 text-accent" aria-hidden /> Small things that may help</h3>
          <ul className="mt-3 space-y-2 text-sm">
            {result.suggestions.map((s) => (
              <li key={s} className="rounded-2xl bg-accent-soft px-4 py-3">{s}</li>
            ))}
          </ul>
        </section>
      )}

      <p className="text-xs text-muted">
        {result.disclaimer || 'A rough summary of one short chat.'} This is not a medical device, and it cannot tell you what is wrong. If you are worried about yourself, please talk to a professional.
      </p>
      <div className="flex flex-wrap gap-3">
        <button type="button" onClick={onHome} className="btn-primary">Back to home</button>
        <button type="button" onClick={onAgain} className="btn-soft"><RotateCcw className="h-4 w-4" aria-hidden /> New check-in</button>
      </div>
    </div>
  );
}
