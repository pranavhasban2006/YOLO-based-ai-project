// Real-browser checks for the live page's behaviour paths:
//  1. "Save session" stores webcam summaries in MongoDB (and nothing when unticked).
//  2. Express restart: the page shows "Reconnecting", then goes back to "Live" by itself.
//  3. Camera permission denied: a clear message and the Upload-tab button.
//  4. No camera at all: a clear message.
// Needs Vite :5173, Express :4000 and the model service :8001. Express is restarted by this script.
//
// Usage: node scripts/ui-behaviour-check.mjs <clip.y4m>
import { spawn, execSync } from 'node:child_process';
import { writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const puppeteer = require(process.env.PUPPETEER_PATH || 'puppeteer-core');
const here = path.dirname(fileURLToPath(import.meta.url));
const appDir = path.resolve(here, '..');
const clip = process.argv[2];
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const APP = 'http://localhost:5173/';
const out = path.join(here, '.cache', 'browser');
await mkdir(out, { recursive: true });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const report = {};
const log = (k, v) => {
  report[k] = v;
  console.log(k, JSON.stringify(v));
};

const launch = (extra = []) =>
  puppeteer.launch({
    executablePath: CHROME,
    headless: true,
    args: ['--no-sandbox', '--autoplay-policy=no-user-gesture-required', ...extra],
    defaultViewport: { width: 1280, height: 900 },
  });

const clickButton = (page, text) =>
  page.evaluate((t) => {
    const b = [...document.querySelectorAll('button')].find((x) => x.textContent.trim() === t);
    if (!b) return false;
    b.click();
    return true;
  }, text);

const pageText = (page) => page.evaluate(() => document.body.innerText);

async function countWebcam() {
  const body = await (await fetch('http://localhost:4000/api/history')).json();
  return body.items ? body.items.filter((i) => i.source === 'webcam').length : null;
}

// ---- 1 and 2: live page with fake camera -------------------------------------------------------
let browser = await launch([
  '--use-fake-ui-for-media-stream',
  '--use-fake-device-for-media-stream',
  `--use-file-for-fake-video-capture=${clip}`,
]);
try {
  const page = await browser.newPage();
  await page.goto(APP, { waitUntil: 'networkidle0' });
  await page.waitForFunction(() => document.body.innerText.includes('Camera is on'), { timeout: 15000 });

  // 1. Save session OFF: nothing stored.
  const before = await countWebcam();
  await clickButton(page, 'Start');
  await sleep(5000);
  const afterOff = await countWebcam();
  log('save_off_webcam_rows_added', afterOff - before);

  // Tick "Save session" while live, then run 6 seconds.
  await page.evaluate(() => document.querySelector('input[type=checkbox]').click());
  await sleep(6000);
  const afterOn = await countWebcam();
  log('save_on_webcam_rows_added', afterOn - afterOff);
  await clickButton(page, 'Stop');
  await sleep(500);

  // 2. Express restart while live.
  await clickButton(page, 'Start');
  await sleep(3000);
  const pidLine = execSync('netstat -ano | findstr :4000 | findstr LISTENING', { shell: 'cmd.exe' }).toString().trim();
  const pid = pidLine.split(/\s+/).pop();
  execSync(`taskkill /PID ${pid} /F`, { shell: 'cmd.exe' });
  await sleep(1500);
  const duringOutage = await pageText(page);
  log('shows_reconnecting_during_outage', /Reconnecting/.test(duringOutage));

  const server = spawn(process.execPath, ['src/index.js'], {
    cwd: path.join(appDir, 'server'),
    detached: true,
    stdio: ['ignore', 'ignore', 'ignore'],
  });
  server.unref();
  const t0 = Date.now();
  await page.waitForFunction(() => document.body.innerText.includes('Live'), { timeout: 20000 }).catch(() => {});
  let back = false;
  while (Date.now() - t0 < 20000) {
    const txt = await pageText(page);
    if (/FPS [\d.]+/.test(txt) && !/Reconnecting/.test(txt)) {
      back = true;
      break;
    }
    await sleep(500);
  }
  log('reconnected_after_restart', back);
  log('reconnect_seconds', back ? Math.round((Date.now() - t0) / 100) / 10 : null);
  await clickButton(page, 'Stop');
} finally {
  await browser.close();
}

// ---- 3 and 4: denied permission, and no camera -------------------------------------------------
browser = await launch([]);
try {
  const ctx = browser.defaultBrowserContext();
  await ctx.overridePermissions(APP.slice(0, -1), []);
  const page = await browser.newPage();
  await page.goto(APP, { waitUntil: 'networkidle0' });
  await sleep(1500);
  const txt = await pageText(page);
  log('denied_message_shown', /permission was denied/i.test(txt));
  log('denied_offers_upload', txt.includes('Go to Upload tab'));
  await page.screenshot({ path: path.join(out, 'denied.png') });
} finally {
  await browser.close();
}

browser = await launch([]);
try {
  const page = await browser.newPage();
  await page.goto(APP, { waitUntil: 'networkidle0' });
  await sleep(1500);
  const txt = await pageText(page);
  // Headless Chrome with no fake device reports NotFoundError, or NotReadable on some builds.
  log('no_camera_message_shown', /No camera was found|camera is in use|Camera error/i.test(txt));
  log('no_camera_text', txt.match(/(No camera was found[^\n]*|Camera error[^\n]*|camera is in use[^\n]*)/i)?.[0] ?? null);
} finally {
  await browser.close();
}

await writeFile(path.join(out, 'ui-behaviour-check.json'), JSON.stringify(report, null, 2));
console.log('WROTE', path.join(out, 'ui-behaviour-check.json'));
