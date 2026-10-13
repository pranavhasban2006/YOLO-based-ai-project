// Speech output for the interview. Default: the browser's SpeechSynthesis (free, no key). Optional: Gemini "natural voice".
import { estimateSpeechMs, speakable, splitSentences } from './interview.js';
import { wellbeing } from './wellbeing.js';

const PREFERRED = /(natural|neural|aria|jenny|libby|sonia|samantha|zira|google uk english female|google us english|karen|moira)/i;

export function pickVoice() {
  const voices = window.speechSynthesis?.getVoices?.() || [];
  const en = voices.filter((v) => /^en[-_]/i.test(v.lang));
  return en.find((v) => PREFERRED.test(v.name)) || en.find((v) => /female/i.test(v.name)) || en[0] || null;
}

export function createSpeaker({ natural = false } = {}) {
  let cancelled = false;
  let currentAudio = null;
  let currentTimer = null;

  const speakBrowser = (text) =>
    new Promise((resolve) => {
      const synth = window.speechSynthesis;
      const rate = 0.92; // a little slower than default: calmer
      if (!synth || typeof SpeechSynthesisUtterance === 'undefined') {
        currentTimer = setTimeout(resolve, Math.min(estimateSpeechMs(text, rate), 4000));
        return;
      }
      const u = new SpeechSynthesisUtterance(speakable(text));
      const v = pickVoice();
      if (v) u.voice = v;
      u.lang = v?.lang || 'en-US';
      u.rate = rate;
      u.pitch = 0.98;
      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        clearTimeout(currentTimer);
        resolve();
      };
      u.onend = finish;
      u.onerror = finish;
      // some browsers never fire onend (no voices, headless): never wait longer than the estimate allows
      currentTimer = setTimeout(finish, estimateSpeechMs(text, rate) * 1.8 + 1500);
      synth.speak(u);
    });

  const speakNatural = async (text) => {
    const url = await wellbeing.tts(text);
    await new Promise((resolve, reject) => {
      const a = new Audio(url);
      currentAudio = a;
      a.onended = resolve;
      a.onerror = () => reject(new Error('playback failed'));
      a.play().catch(reject);
    });
    URL.revokeObjectURL(url);
  };

  return {
    // Speaks sentence by sentence; onSentence(sentence, index) lets the caller show captions as it goes.
    async speak(text, onSentence) {
      cancelled = false;
      const sentences = splitSentences(text);
      let useNatural = natural;
      for (let i = 0; i < sentences.length && !cancelled; i += 1) {
        onSentence?.(sentences[i], i);
        if (useNatural) {
          try {
            await speakNatural(speakable(sentences[i]));
            continue;
          } catch {
            useNatural = false; // fall back to the browser voice for the rest
          }
        }
        await speakBrowser(sentences[i]);
      }
    },
    cancel() {
      cancelled = true;
      clearTimeout(currentTimer);
      window.speechSynthesis?.cancel?.();
      currentAudio?.pause?.();
    },
  };
}
