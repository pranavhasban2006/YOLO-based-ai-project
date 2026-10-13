// Real-browser check of the live page. Chrome's fake webcam plays a RAVDESS clip (y4m file),
// so the page sees a real face. Reads what the UI shows, takes screenshots, reports console errors.
//
// Needs: Vite on :5173, Express on :4000, model service on :8001, and a y4m clip.
// Usage: node scripts/browser-check.mjs <clip.y4m> <outDir>
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const puppeteer = require(process.env.PUPPETEER_PATH || 'puppeteer-core');

const clip = process.argv[2];
const outDir = process.argv[3] || path.join(path.dirname(new URL(import.meta.url).pathname), '.cache', 'browser');
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const APP = process.env.APP_URL || 'http://localhost:5173/';
await mkdir(outDir, { recursive: true });

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  args: [
    '--use-fake-ui-for-media-stream',
    '--use-fake-device-for-media-stream',
    `--use-file-for-fake-video-capture=${clip}`,
    '--autoplay-policy=no-user-gesture-required',
    '--no-sandbox',
  ],
  defaultViewport: { width: 1280, height: 900 },
});

const report = { console_errors: [], steps: {} };
try {
  const page = await browser.newPage();
  page.on('console', (m) => {
    if (m.type() === 'error') report.console_errors.push(m.text());
  });
  page.on('pageerror', (e) => report.console_errors.push(`pageerror: ${e.message}`));

  await page.goto(APP, { waitUntil: 'networkidle0' });
  // Camera permission is granted by the fake-UI flag; the page asks on load.
  await page.waitForFunction(() => document.body.innerText.includes('Camera is on'), { timeout: 15000 });
  report.steps.camera_on_before_start = true;

  const clickButton = (text) =>
    page.evaluate((t) => {
      const b = [...document.querySelectorAll('button')].find((x) => x.textContent.trim() === t);
      if (!b) return false;
      b.click();
      return true;
    }, text);

  report.steps.clicked_start = await clickButton('Start');
  await new Promise((r) => setTimeout(r, 8000));

  const live = await page.evaluate(() => {
    const txt = document.body.innerText;
    const grab = (re) => (txt.match(re) || [])[1] ?? null;
    return {
      status: grab(/(Live|Connecting…|Reconnecting…)\n/) ?? (txt.includes('Live') ? 'Live' : null),
      fps: grab(/FPS ([\d.]+)/),
      latency: grab(/Latency (\d+) ms/),
      dropped: grab(/Dropped (\d+)/),
      model: grab(/Model (\S+)/),
      raw: grab(/Raw: ([a-z]+ [\d%]+)/),
      label_block: txt.slice(txt.indexOf('Smoothed label'), txt.indexOf('Smoothed label') + 120).replace(/\n/g, ' | '),
      no_face_text: txt.includes('No face'),
      stopped: txt.includes('Stopped'),
    };
  });
  report.steps.live_stats = live;
  await page.screenshot({ path: path.join(outDir, 'live.png'), fullPage: false });

  report.steps.clicked_stop = await clickButton('Stop');
  await new Promise((r) => setTimeout(r, 1000));
  report.steps.camera_off_after_stop = await page.evaluate(() => !document.body.innerText.includes('Camera is on'));

  // Upload tab with a real photo.
  await clickButton('Image upload');
  await new Promise((r) => setTimeout(r, 500));
  const input = await page.$('input[type=file]');
  await input.uploadFile(process.env.UPLOAD_IMAGE || '');
  await page.waitForFunction(() => document.body.innerText.toLowerCase().includes('top label'), { timeout: 20000 });
  report.steps.upload_result = await page.evaluate(() => document.body.innerText.match(/Top label[\s\S]{0,120}/)?.[0]);
  await page.screenshot({ path: path.join(outDir, 'upload.png'), fullPage: true });

  await clickButton('History');
  await new Promise((r) => setTimeout(r, 1500));
  report.steps.history_text = await page.evaluate(() => document.querySelector('main')?.innerText.slice(0, 300));
  await page.screenshot({ path: path.join(outDir, 'history.png'), fullPage: false });
} catch (err) {
  report.error = err.message;
} finally {
  await browser.close();
}
await writeFile(path.join(outDir, 'browser-check.json'), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
