// Minimal Gemini REST client (generateContent). The key is only ever sent in the x-goog-api-key header
// and is never logged. A model that is overloaded (429/5xx) or missing (404) falls back to the next model.

export class GeminiError extends Error {
  constructor(code, message, status) {
    super(message);
    this.code = code;
    this.status = status;
  }
}

const RETRYABLE = new Set([429, 500, 502, 503, 504]);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export function createGeminiClient({
  apiKey,
  model,
  fallbackModel,
  baseUrl = 'https://generativelanguage.googleapis.com/v1beta',
  fetchImpl = globalThis.fetch,
  timeoutMs = 20000,
  retryDelayMs = 600,
} = {}) {
  const models = [model, fallbackModel].filter((m, i, a) => m && a.indexOf(m) === i);

  async function callOnce(m, body) {
    let res;
    try {
      res = await fetchImpl(`${baseUrl}/models/${m}:generateContent`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'x-goog-api-key': apiKey },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(timeoutMs),
      });
    } catch (err) {
      throw new GeminiError('network', `Gemini request failed: ${err.name}`, 0);
    }
    const json = await res.json().catch(() => ({}));
    if (!res.ok) throw new GeminiError('http', json?.error?.message || `HTTP ${res.status}`, res.status);
    const cand = json.candidates?.[0];
    const text = (cand?.content?.parts || []).map((p) => p.text || '').join('').trim();
    if (!text) throw new GeminiError('empty', `Gemini returned no text (${cand?.finishReason || json.promptFeedback?.blockReason || 'unknown'})`, 200);
    return { text, model: m };
  }

  return {
    available: Boolean(apiKey),
    model,
    models,
    // generate({ system, contents:[{role:'user'|'model', text}], schema?, temperature, maxOutputTokens }) -> { text, model }
    async generate({ system, contents, schema, temperature = 0.7, maxOutputTokens = 400, inlineAudio = null }) {
      let lastErr;
      for (const m of models) {
        const generationConfig = { temperature, maxOutputTokens };
        if (schema) Object.assign(generationConfig, { responseMimeType: 'application/json', responseSchema: schema });
        // keep thinking short: 2.5 models take a token budget, 3.x models a level
        if (/^gemini-2\.5/.test(m)) generationConfig.thinkingConfig = { thinkingBudget: 0 };
        else if (/^gemini-3/.test(m)) generationConfig.thinkingConfig = { thinkingLevel: 'low' };
        const body = {
          systemInstruction: { parts: [{ text: system }] },
          contents: contents.map((c, i) => ({ role: c.role, parts: i === contents.length - 1 && inlineAudio ? [{ inlineData: inlineAudio }, { text: c.text }] : [{ text: c.text }] })),
          generationConfig,
        };
        for (let attempt = 0; attempt < 2; attempt += 1) {
          try {
            return await callOnce(m, body);
          } catch (err) {
            lastErr = err;
            const retry = err.code === 'network' || RETRYABLE.has(err.status);
            if (retry && attempt === 0) {
              await sleep(retryDelayMs);
              continue;
            }
            break; // next model
          }
        }
      }
      throw lastErr || new GeminiError('no_model', 'No Gemini model configured', 0);
    },

    // Optional "natural voice": Gemini text-to-speech. Returns { wav: Buffer }.
    async tts(text, { model: ttsModel = 'gemini-3.8-flash-tts', voice = 'Kore' } = {}) {
      const body = {
        contents: [{ parts: [{ text: `Say calmly and warmly: ${text}` }] }],
        generationConfig: { responseModalities: ['AUDIO'], speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: voice } } } },
      };
      const res = await fetchImpl(`${baseUrl}/models/${ttsModel}:generateContent`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'x-goog-api-key': apiKey },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(timeoutMs),
      }).catch(() => { throw new GeminiError('network', 'Gemini TTS request failed', 0); });
      const json = await res.json().catch(() => ({}));
      if (!res.ok) throw new GeminiError('http', json?.error?.message || `HTTP ${res.status}`, res.status);
      const part = json.candidates?.[0]?.content?.parts?.find((p) => p.inlineData?.data);
      if (!part) throw new GeminiError('empty', 'Gemini TTS returned no audio', 200);
      const raw = Buffer.from(part.inlineData.data, 'base64');
      const mime = part.inlineData.mimeType || '';
      if (/wav/i.test(mime)) return { wav: raw };
      const rate = Number(/rate=(\d+)/i.exec(mime)?.[1]) || 24000; // audio/L16 PCM, mono
      const header = Buffer.alloc(44);
      header.write('RIFF', 0); header.writeUInt32LE(36 + raw.length, 4); header.write('WAVEfmt ', 8);
      header.writeUInt32LE(16, 16); header.writeUInt16LE(1, 20); header.writeUInt16LE(1, 22);
      header.writeUInt32LE(rate, 24); header.writeUInt32LE(rate * 2, 28); header.writeUInt16LE(2, 32); header.writeUInt16LE(16, 34);
      header.write('data', 36); header.writeUInt32LE(raw.length, 40);
      return { wav: Buffer.concat([header, raw]) };
    },

    // Optional alternative to local Whisper: Gemini audio transcription (text only, no word timings).
    async transcribe(buffer, mimeType = 'audio/webm') {
      const { text } = await this.generate({
        system: 'You transcribe speech. Return only the words that were spoken, verbatim, with normal punctuation. If there is no speech, return an empty string.',
        contents: [{ role: 'user', text: '(audio follows)' }],
        temperature: 0,
        maxOutputTokens: 400,
        inlineAudio: { mimeType, data: buffer.toString('base64') },
      });
      return text.trim();
    },
  };
}
