// Runs the synthetic transcripts through the real pipeline and compares the tier with the expected one.
//   node eval/run-eval.mjs mock    -> MOCK mode (keyword heuristic), no network
//   node eval/run-eval.mjs gemini  -> real Gemini for the per-message analysis (needs GEMINI_API_KEY); chat replies are skipped
// Scoring mode: the transcripts are TEXT ONLY, so the default model-fusion score (face / voice / text models) has no input
// for them. This script therefore scores them in questionnaire mode (Gemini's per-message analysis), which is the only
// mode that can read text today, and also reports the rule-based safety layer separately.
// Writes eval/results-<mode>.json. Nothing here is a clinical validation.
import { writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { config } from '../src/config.js';
import { createGeminiClient } from '../src/wellbeing/gemini.js';
import { createWellbeingService } from '../src/wellbeing/service.js';
import { createStore } from '../src/wellbeing/store.js';
import { DEFAULT_CONFIG } from '../src/wellbeing/scoring.js';
import { TRANSCRIPTS } from './transcripts.mjs';

const mode = process.argv[2] === 'gemini' ? 'gemini' : 'mock';
const here = path.dirname(fileURLToPath(import.meta.url));

let gemini;
if (mode === 'gemini') {
  if (!config.wellbeing.apiKey) throw new Error('GEMINI_API_KEY is not set');
  const real = createGeminiClient({ apiKey: config.wellbeing.apiKey, model: config.wellbeing.model, fallbackModel: config.wellbeing.fallbackModel });
  // Only the analysis call is real; chat replies are canned, since the tier depends on the analysis only.
  gemini = { ...real, generate: (args) => (args.schema ? real.generate(args) : Promise.resolve({ text: 'Thank you for sharing. How has your sleep been?' })) };
} else {
  gemini = createGeminiClient({ apiKey: '', model: 'none' });
}

async function runOne(t) {
  const svc = createWellbeingService({ gemini, store: createStore({ isDbReady: () => false }), maxTurns: 10, counsellorContact: 'eval', scoringConfig: { ...DEFAULT_CONFIG, scoreUsesQuestionnaire: true } });
  const { sessionId } = await svc.start({ anonymousId: `eval-${t.id}` });
  let safetyAt = null;
  let regexAt = null;
  let degraded = false;
  for (let i = 0; i < t.turns.length; i += 1) {
    const r = await svc.message(sessionId, t.turns[i]);
    degraded ||= Boolean(r.degraded);
    if (r.safety.level === 'explicit' && safetyAt === null) safetyAt = i + 1;
    if (r.safety.regex === 'explicit' && regexAt === null) regexAt = i + 1;
  }
  const fin = await svc.finish(sessionId);
  return {
    id: t.id, kind: t.kind, expected: t.expected, tier: fin.tier, score: fin.score, smoothed: fin.smoothed_score,
    confidence: fin.confidence, phq_answered: fin.phq.answered, mood: fin.mood.mean_mood,
    safety_first_explicit_turn: safetyAt, regex_first_explicit_turn: regexAt, degraded,
    overrides: fin.overrides, note: t.note || null,
  };
}

const results = [];
const queue = [...TRANSCRIPTS];
const workers = Array.from({ length: mode === 'gemini' ? 2 : 1 }, async () => {
  while (queue.length) {
    const t = queue.shift();
    results.push(await runOne(t));
  }
});
await Promise.all(workers);
results.sort((a, b) => a.id.localeCompare(b.id));

const tiers = ['doing_okay', 'struggling', 'needs_support_now'];
const matrix = Object.fromEntries(tiers.map((e) => [e, Object.fromEntries([...tiers, 'none'].map((p) => [p, 0]))]));
for (const r of results) matrix[r.expected][r.tier || 'none'] += 1;
const exact = results.filter((r) => r.tier === r.expected).length;
const adjacent = results.filter((r) => Math.abs(tiers.indexOf(r.tier) - tiers.indexOf(r.expected)) <= 1 && r.tier).length;
const flaggedByRisk = results.filter((r) => r.expected === 'needs_support_now' && r.kind === 'high' && r.overrides.safety_flag);
const nonRisk = results.filter((r) => r.expected !== 'needs_support_now');
const summary = {
  mode, model: mode === 'gemini' ? config.wellbeing.model : null, n: results.length,
  exact_match: exact, exact_match_rate: +(exact / results.length).toFixed(3),
  within_one_tier: adjacent, within_one_tier_rate: +(adjacent / results.length).toFixed(3),
  per_expected: Object.fromEntries(tiers.map((e) => {
    const rows = results.filter((r) => r.expected === e);
    return [e, { n: rows.length, matched: rows.filter((r) => r.tier === e).length }];
  })),
  matrix_expected_by_predicted: matrix,
  explicit_risk_statements_flagged: `${flaggedByRisk.length} of ${results.filter((r) => ['N01', 'N02', 'N03', 'N04', 'N05', 'N06', 'N07'].includes(r.id)).length} transcripts with an explicit statement`,
  false_safety_flags_on_non_risk: nonRisk.filter((r) => r.overrides.safety_flag).map((r) => r.id),
  degraded_runs: results.filter((r) => r.degraded).map((r) => r.id),
};
writeFileSync(path.join(here, `results-${mode}.json`), JSON.stringify({ summary, results }, null, 1));
console.log(JSON.stringify(summary, null, 1));
console.log('mismatches:');
for (const r of results.filter((x) => x.tier !== x.expected)) console.log(`  ${r.id} expected ${r.expected}, got ${r.tier} (score ${r.score}, smoothed ${r.smoothed}, phq ${r.phq_answered}, conf ${r.confidence})`);
