// Pure client-side logic used by the voice interview: end-of-answer detection and face aggregation.
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { VAD_DEFAULTS, createVad, rms } from '../../client/src/lib/vad.js';
import { estimateSpeechMs, faceAggregate, speakable, splitSentences } from '../../client/src/lib/interview.js';

const DT = 50; // ms per frame
const QUIET = 0.004;
const LOUD = 0.12;

function run(vad, frames) {
  let s;
  for (const level of frames) {
    s = vad.push(level, DT);
    if (s.ended) return s;
  }
  return s;
}
const rep = (level, ms) => Array(Math.round(ms / DT)).fill(level);

describe('VAD end-of-answer logic', () => {
  it('rms of silence and of a full-scale square wave', () => {
    assert.equal(rms(new Float32Array(100)), 0);
    assert.ok(Math.abs(rms(new Float32Array(100).fill(0.5)) - 0.5) < 1e-9);
  });
  it('ends about 1.5 s after speech stops', () => {
    const s = run(createVad(), [...rep(QUIET, 500), ...rep(LOUD, 2000), ...rep(QUIET, 3000)]);
    assert.equal(s.ended, 'silence');
    assert.ok(s.silenceMs >= VAD_DEFAULTS.silenceMs && s.silenceMs < VAD_DEFAULTS.silenceMs + 2 * DT);
    assert.ok(Math.abs(s.elapsedMs - (500 + 2000 + 1500)) <= 2 * DT);
  });
  it('a short pause inside the answer does not end it', () => {
    const s = run(createVad(), [...rep(QUIET, 500), ...rep(LOUD, 1500), ...rep(QUIET, 1000), ...rep(LOUD, 1500), ...rep(QUIET, 300)]);
    assert.equal(s.ended, false);
    assert.equal(s.started, true);
  });
  it('does not end before the person has started talking, and gives up after 12 s of silence', () => {
    const early = run(createVad(), rep(QUIET, 5000));
    assert.equal(early.ended, false);
    const none = run(createVad(), rep(QUIET, 15000));
    assert.equal(none.ended, 'no_speech');
  });
  it('a click or cough shorter than the minimum speech length does not start an answer', () => {
    const s = run(createVad(), [...rep(QUIET, 500), ...rep(LOUD, 100), ...rep(QUIET, 3000)]);
    assert.equal(s.started, false);
    assert.equal(s.ended, false);
  });
  it('stops at the 60 s maximum even if the person keeps talking', () => {
    const s = run(createVad(), [...rep(QUIET, 500), ...rep(LOUD, 70000)]);
    assert.equal(s.ended, 'max');
    assert.ok(s.elapsedMs >= 60000 && s.elapsedMs < 60000 + 2 * DT);
  });
  it('adapts to a noisy room: steady background noise is not speech', () => {
    const NOISY = 0.03;
    const s = run(createVad(), [...rep(NOISY, 800), ...rep(NOISY, 3000)]);
    assert.equal(s.started, false);
    const talk = run(createVad(), [...rep(NOISY, 800), ...rep(0.25, 1500), ...rep(NOISY, 2500)]);
    assert.equal(talk.ended, 'silence');
  });
  it('once ended it stays ended', () => {
    const vad = createVad();
    run(vad, [...rep(QUIET, 500), ...rep(LOUD, 1000), ...rep(QUIET, 2500)]);
    assert.equal(vad.push(LOUD, DT).ended, 'silence');
  });
  it('options override the defaults (shorter silence window)', () => {
    const s = run(createVad({ silenceMs: 600 }), [...rep(QUIET, 500), ...rep(LOUD, 1000), ...rep(QUIET, 2000)]);
    assert.equal(s.ended, 'silence');
    assert.ok(s.silenceMs < 800);
  });
});

describe('face aggregation per answer', () => {
  const p = (o) => ({ angry: 0, disgust: 0, fear: 0, happy: 0, neutral: 0, sad: 0, surprise: 0, ...o });
  it('mean, dominant emotion, variability and count; only numbers come out', () => {
    const a = faceAggregate([p({ happy: 0.8, neutral: 0.2 }), p({ happy: 0.6, neutral: 0.4 }), p({ happy: 0.7, neutral: 0.3 })]);
    assert.equal(a.dominant, 'happy');
    assert.equal(a.n, 3);
    assert.ok(Math.abs(a.mean.happy - 0.7) < 1e-9);
    assert.ok(Math.abs(Object.values(a.mean).reduce((x, y) => x + y, 0) - 1) < 1e-9);
    assert.ok(a.variability > 0);
    assert.deepEqual(Object.keys(a).sort(), ['dominant', 'mean', 'n', 'variability']);
  });
  it('steady faces vary less than changing faces', () => {
    const steady = faceAggregate(Array(6).fill(p({ neutral: 1 })));
    const changing = faceAggregate([p({ happy: 1 }), p({ sad: 1 }), p({ happy: 1 }), p({ sad: 1 })]);
    assert.equal(steady.variability, 0);
    assert.ok(changing.variability > steady.variability);
  });
  it('ignores malformed samples and returns null for none', () => {
    assert.equal(faceAggregate([]), null);
    assert.equal(faceAggregate([{ happy: 'x' }, null]), null);
    assert.equal(faceAggregate([null, p({ sad: 1 })]).n, 1);
  });
});

describe('speech splitting', () => {
  it('splits into sentences to be spoken one by one', () => {
    assert.deepEqual(splitSentences('That sounds hard. How has your sleep been?  Take your time.'), ['That sounds hard.', 'How has your sleep been?', 'Take your time.']);
    assert.deepEqual(splitSentences('No punctuation here'), ['No punctuation here']);
    assert.deepEqual(splitSentences(''), []);
  });
  it('line breaks and bullets in the fixed safety message become sentences, and numbers are spoken digit by digit', () => {
    const parts = splitSentences('You can talk any time:\nTele-MANAS (free): 14416\n- AASRA: 9820466726\n- Emergency: 112\nPlease tell someone.');
    assert.ok(parts.length >= 4);
    assert.ok(parts.every((p) => !p.startsWith('-')));
    assert.equal(speakable('Call 14416 now'), 'Call 1 4 4 1 6 now');
    assert.equal(speakable('free, 24x7'), 'free, 24 by 7');
  });
  it('fallback speaking time grows with length', () => {
    assert.ok(estimateSpeechMs('one two three four five six seven eight nine ten eleven twelve') > estimateSpeechMs('hello'));
  });
});
