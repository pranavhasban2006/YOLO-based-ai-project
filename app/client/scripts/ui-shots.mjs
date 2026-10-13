// Drives the app in a real browser (system Chrome via playwright-core) in light and dark mode and saves screenshots.
// Needs the Express server in MOCK mode (empty GEMINI_API_KEY) and the built client served by `vite preview`:
//   see project/reports/ui_screenshots/README.md for the exact commands.
import { chromium } from 'playwright-core';
import { mkdirSync } from 'node:fs';
import path from 'node:path';

const BASE = process.env.UI_URL || 'http://localhost:5190';
const OUT = process.env.OUT_DIR || path.resolve('..', '..', 'project', 'reports', 'ui_screenshots');
mkdirSync(OUT, { recursive: true });

// fake camera = a still face, so the optional emotion check produces real model outputs (the score is model-driven)
const CACHE = path.resolve('scripts', '.cache');
const launch = (face) => chromium.launch({ channel: 'chrome', headless: true, args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-video-capture=${path.join(CACHE, face)}`] });
let browser;
const problems = [];

async function run(theme, { width = 1366, height = 900, suffix = '' } = {}) {
  browser = await launch(theme === 'dark' ? 'face_happy.y4m' : 'face.y4m');
  const ctx = await browser.newContext({ viewport: { width, height }, reducedMotion: 'reduce', permissions: ['camera', 'microphone'] });
  await ctx.addInitScript(([t, id]) => {
    localStorage.setItem('wb_theme', t);
    localStorage.setItem('wb_user', id);
  }, [theme, `shots-${theme}${suffix}-${Date.now()}`]);
  const page = await ctx.newPage();
  page.on('pageerror', (e) => problems.push(`${theme}${suffix} pageerror: ${e.message}`));
  page.on('console', (m) => m.type() === 'error' && !/favicon|Failed to load resource/.test(m.text()) && problems.push(`${theme}${suffix} console: ${m.text().slice(0, 160)}`));
  const shot = (name) => page.screenshot({ path: path.join(OUT, `${name}-${theme}${suffix}.png`), fullPage: true });
  const nav = async (label) => page.getByRole('button', { name: label, exact: true }).filter({ visible: true }).first().click();
  const say = async (text) => {
    await page.locator('#chat-input').fill(text);
    await page.getByRole('button', { name: 'Send' }).click();
    await page.waitForFunction((n) => document.querySelectorAll('[aria-live="polite"] > div').length >= n, 0).catch(() => {});
    await page.waitForTimeout(500);
  };

  await page.goto(BASE);
  await page.waitForTimeout(800);
  await shot('1-home-empty');

  // check-in: consent, chat, result
  await page.getByRole('button', { name: /Start check-in/ }).first().click();
  await page.getByRole('button', { name: /I understand/ }).click();
  await shot('2-checkin-intro');
  await page.getByRole('button', { name: 'Start check-in' }).click();
  await page.locator('#chat-input').waitFor();
  await page.getByLabel(/Optional emotion check/).check();
  await page.waitForTimeout(6000);
  await page.getByRole('button', { name: 'Stressed', exact: true }).click();
  await page.waitForTimeout(700);
  await say('Exams are next week and I keep putting off studying');
  await say('I have been sleeping badly and I am tired all day');
  await say('A bit better when I talk to my friends though');
  await shot('3-checkin-chat');
  await page.getByRole('button', { name: /Finish and see my result/ }).click();
  await page.getByText('What shaped this').waitFor();
  await page.waitForTimeout(500);
  await shot('4-result');

  // history after a few sessions
  for (let i = 0; i < 2; i += 1) {
    await page.getByRole('button', { name: /New check-in/ }).click();
    await page.getByRole('button', { name: 'Start check-in' }).click();
    await page.locator('#chat-input').waitFor();
    await page.getByLabel(/Optional emotion check/).check();
    await page.waitForTimeout(6000);
    await say(i ? 'Feeling good today, had a nice walk' : 'Pretty tired and a little down');
    await say('Nothing much else really');
    await page.getByRole('button', { name: /Finish and see my result/ }).click();
    await page.getByText('What shaped this').waitFor();
  }
  await page.getByRole('button', { name: /Back to home/ }).click();
  await page.waitForTimeout(1200);
  await shot('5-home');
  await nav('History');
  await page.waitForTimeout(1200);
  await shot('6-history');

  // safety path: calm banner with helplines
  await nav('Check-in');
  await page.getByRole('button', { name: 'Start check-in' }).click();
  await page.locator('#chat-input').waitFor();
  await say('honestly I want to die');
  await shot('7-checkin-support');
  await page.getByRole('button', { name: /Finish and see my result/ }).click();
  await page.getByText('You do not have to carry this alone').first().waitFor();
  await shot('8-result-support');

  // other pages
  for (const [label, name] of [['Live Emotion', '9-live'], ['Upload', '10-upload'], ['Audio', '11-audio'], ['Text', '12-text']]) {
    await nav(label);
    await page.waitForTimeout(600);
    await shot(name);
  }

  // horizontal overflow check on this viewport
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  if (overflow > 1) problems.push(`${theme}${suffix}: horizontal overflow ${overflow}px`);
  await ctx.close();
  await browser.close();
}

await run('light');
await run('dark');
await run('light', { width: 390, height: 844, suffix: '-mobile' });
await run('dark', { width: 390, height: 844, suffix: '-mobile' });
console.log(problems.length ? `PROBLEMS:\n${problems.join('\n')}` : 'no page errors, no horizontal overflow');
process.exit(problems.length ? 1 : 0);


