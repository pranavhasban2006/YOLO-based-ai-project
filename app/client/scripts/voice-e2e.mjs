// End-to-end run of the voice interview in Chrome with fake devices (mock mode, local Whisper).
//   Fake microphone = a spoken sample (client/scripts/.cache/speech.wav), fake camera = a still face (face.y4m).
// Usage: node scripts/voice-e2e.mjs            (needs: model service, server in MOCK mode, `vite preview` on 5190)
import { chromium } from 'playwright-core';
import { mkdirSync } from 'node:fs';
import path from 'node:path';

const BASE = process.env.UI_URL || 'http://localhost:5190';
const OUT = path.resolve('..', '..', 'project', 'reports', 'ui_screenshots');
const CACHE = path.resolve('scripts', '.cache');
mkdirSync(OUT, { recursive: true });
const problems = [];

async function run({ wav, theme, prefix, safety }) {
  const browser = await chromium.launch({
    channel: 'chrome',
    headless: true,
    args: [
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream',
      `--use-file-for-fake-audio-capture=${path.join(CACHE, wav)}`,
      `--use-file-for-fake-video-capture=${path.join(CACHE, 'face.y4m')}`,
    ],
  });
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 }, permissions: ['camera', 'microphone'], reducedMotion: 'reduce' });
  await ctx.addInitScript(([t, id]) => { localStorage.setItem('wb_theme', t); localStorage.setItem('wb_user', id); }, [theme, `voice-e2e-${Date.now()}`]);
  const page = await ctx.newPage();
  page.on('pageerror', (e) => problems.push(`${prefix} pageerror: ${e.message}`));
  page.on('console', (m) => m.type() === 'error' && !/favicon|Failed to load resource/.test(m.text()) && problems.push(`${prefix} console: ${m.text().slice(0, 200)}`));
  page.on('response', async (r) => { if (r.url().includes('voice-answer')) { const j = await r.json().catch(() => ({})); console.log(prefix, 'voice-answer ->', r.status(), j.kind, JSON.stringify(j.safety), j.complete, (j.reply || '').slice(0, 60)); } });
  const shot = (name) => page.screenshot({ path: path.join(OUT, `voice_${prefix}${name}.png`), fullPage: true });
  const chip = (name) => page.locator('span.rounded-full', { hasText: new RegExp(`^${name}$`) }).first();

  await page.goto(BASE);
  await page.getByRole('button', { name: 'Voice Interview', exact: true }).filter({ visible: true }).first().click();
  await page.getByText('Before the voice interview').waitFor();
  await shot('1-consent');
  await page.getByRole('button', { name: /I understand, start/ }).click();

  await page.getByTestId('ai-caption').filter({ hasText: 'How has your day been?' }).waitFor({ timeout: 30000 });
  await shot('2-room-speaking');
  await page.waitForFunction(() => [...document.querySelectorAll('span.rounded-full')].some((s) => s.textContent === 'Listening' && s.className.includes('bg-accent ')), null, { timeout: 60000 });
  await page.waitForTimeout(1500);
  await shot('3-listening');

  const answers = safety ? 1 : 3;
  for (let i = 0; i < answers; i += 1) {
    await page.getByTestId('user-caption').waitFor({ timeout: 90000 });
    const said = await page.getByTestId('user-caption').innerText();
    console.log(`${prefix} answer ${i + 1}: ${said}`);
    if (i === 0) await shot('4-captions');
    if (safety) break;
    await page.waitForFunction((n) => document.querySelector('[data-testid="ai-caption"]')?.textContent && !document.querySelector('[data-testid="user-caption"]'), i, { timeout: 90000 }).catch(() => {});
    await page.waitForTimeout(500);
  }

  if (safety) {
    // the fake microphone loops, so an answer can start mid-sentence; the interview simply goes on until a full phrase is heard
    try {
      await page.getByText('You do not have to carry this alone').first().waitFor({ timeout: Number(process.env.SAFETY_WAIT || 300000) });
    } catch (err) {
      await shot('x-nobanner');
      console.log('caption now:', await page.getByTestId('ai-caption').innerText().catch(() => ''), '| user:', await page.getByTestId('user-caption').innerText().catch(() => ''));
      console.log('PAGE PROBLEMS:', JSON.stringify(problems));
      throw err;
    }
    await shot('5-safety');
    const text = await page.getByTestId('ai-caption').innerText();
    console.log(`${prefix} safety caption: ${text.slice(0, 80)}`);
    await page.getByRole('button', { name: 'End session' }).first().click();
  } else {
    await page.getByRole('button', { name: /End session/ }).first().waitFor();
    await page.getByRole('button', { name: /End session/ }).first().click();
  }
  try {
    await page.getByText('How your score was made').waitFor({ timeout: 30000 });
  } catch (e) {
    await shot('x-failure');
    throw e;
  }
  await page.waitForTimeout(1200);
  await shot('6-result');
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  if (overflow > 1) problems.push(`${prefix}: horizontal overflow ${overflow}px`);
  await browser.close();
}

if (process.env.ONLY !== 'safety') {
  await run({ wav: 'speech.wav', theme: 'light', prefix: 'light_' });
  await run({ wav: 'speech.wav', theme: 'dark', prefix: 'dark_' });
}
await run({ wav: 'speech_safety.wav', theme: 'light', prefix: 'safety_', safety: true });
console.log(problems.length ? `PROBLEMS:\n${problems.join('\n')}` : 'voice e2e: no page errors, no overflow');
process.exit(problems.length ? 1 : 0);




