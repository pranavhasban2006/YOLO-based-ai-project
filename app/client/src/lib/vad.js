// Energy-based voice activity detection for the interview. Pure logic: feed it one RMS value per frame.
//
//   const vad = createVad();
//   const s = vad.push(rms, dtMs);   // s.ended is false, 'silence', 'max' or 'no_speech'
//
// The noise floor is learned from the first frames and from quiet frames afterwards. Speech starts when the level is
// well above the floor for a short run; the answer ends after `silenceMs` of quiet following speech.
export const VAD_DEFAULTS = {
  silenceMs: 1500, // quiet after speech that ends the answer
  maxMs: 60000, // hard limit for one answer
  noSpeechMs: 12000, // give up if nobody starts talking
  minSpeechMs: 250, // speech needed before the answer counts as started
  calibrateMs: 400, // initial frames used to learn the noise floor
  startFactor: 3, // speech if level > floor x factor
  minThreshold: 0.012, // and never below this absolute level
};

export function rms(samples) {
  let s = 0;
  for (let i = 0; i < samples.length; i += 1) s += samples[i] * samples[i];
  return Math.sqrt(s / Math.max(1, samples.length));
}

export function createVad(options = {}) {
  const o = { ...VAD_DEFAULTS, ...options };
  let elapsed = 0;
  let floor = null;
  let calibrated = 0;
  let speechRun = 0;
  let speechTotal = 0;
  let started = false;
  let silence = 0;
  let ended = false;
  let startedAt = null;

  return {
    // Returns the current state after this frame.
    push(level, dtMs) {
      if (ended) return this.state();
      elapsed += dtMs;
      if (calibrated < o.calibrateMs) {
        floor = floor === null ? level : Math.min(floor, level) * 0.5 + floor * 0.5;
        calibrated += dtMs;
        return this.state();
      }
      const thr = Math.max(o.minThreshold, (floor ?? 0) * o.startFactor);
      const speaking = level > thr;
      if (speaking) {
        speechRun += dtMs;
        speechTotal += dtMs;
        silence = 0;
        if (!started && speechRun >= o.minSpeechMs) {
          started = true;
          startedAt = elapsed;
        }
      } else {
        speechRun = 0;
        if (started) silence += dtMs;
        else floor = floor * 0.95 + level * 0.05; // keep tracking the room noise while nobody talks
      }
      if (started && silence >= o.silenceMs) ended = 'silence';
      else if (elapsed >= o.maxMs) ended = 'max';
      else if (!started && elapsed >= o.noSpeechMs) ended = 'no_speech';
      return this.state();
    },
    state() {
      return { speaking: speechRun > 0, started, ended, elapsedMs: elapsed, speechMs: speechTotal, silenceMs: silence, startedAtMs: startedAt, floor };
    },
  };
}
