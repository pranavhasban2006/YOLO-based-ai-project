import 'dotenv/config';

const num = (v, d) => (v === undefined || v === '' ? d : Number(v));

export const config = {
  port: num(process.env.PORT, 4000),
  modelServiceUrl: (process.env.MODEL_SERVICE_URL || 'http://127.0.0.1:8001').replace(/\/$/, ''),
  // 'local' = MONGODB_URI only, 'memory' = mongodb-memory-server only, 'auto' = local, else memory.
  mongoMode: process.env.MONGO_MODE || 'auto',
  mongoUri: process.env.MONGODB_URI || 'mongodb://127.0.0.1:27017/emotion_app',
  maxImageMb: num(process.env.MAX_IMAGE_MB, 10),
  maxAudioMb: num(process.env.MAX_AUDIO_MB, 10),
  // Webcam results are stored at most once per this many ms, when the user ticks "Save session".
  saveIntervalMs: num(process.env.SAVE_INTERVAL_MS, 1000),
  // Upstream calls to the model service time out after this many ms.
  upstreamTimeoutMs: num(process.env.UPSTREAM_TIMEOUT_MS, 15000),
  // Wellbeing check-in. The key lives in server/.env only. Without it the module runs in MOCK mode.
  wellbeing: {
    apiKey: (process.env.GEMINI_API_KEY || '').trim(),
    model: process.env.GEMINI_MODEL || 'gemini-3.5-flash',
    fallbackModel: process.env.GEMINI_FALLBACK_MODEL || 'gemini-3.5-flash-lite',
    counsellorContact: process.env.COUNSELLOR_CONTACT || 'your college counselling centre',
    maxTurns: num(process.env.WELLBEING_MAX_TURNS, 8),
    ttsModel: process.env.GEMINI_TTS_MODEL || 'gemini-3.8-flash-tts',
  },
};