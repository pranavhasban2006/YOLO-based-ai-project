import express from 'express';
import multer from 'multer';
import { config } from '../config.js';
import { Prediction } from '../models/Prediction.js';
import { dbReady, dbStatus } from '../db.js';
import * as model from '../services/modelClient.js';
import { ModelError } from '../services/modelClient.js';

export const api = express.Router();

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: config.maxImageMb * 1024 * 1024, files: 1 },
});

const IMAGE_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp', 'image/bmp']);

// Shape of one stored summary; the probabilities come straight from the model service.
function toDoc(result, source) {
  return {
    source,
    face_found: result.face_found,
    probabilities: result.probabilities,
    top_label: result.top_label,
    confidence: result.confidence,
    model: result.model || null,
  };
}

async function save(result, source) {
  if (!dbReady()) return false;
  await Prediction.create(toDoc(result, source));
  return true;
}

api.post('/predict/image', upload.single('file'), async (req, res, next) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'missing_file', message: 'Send the image as form field "file".' });
    if (!IMAGE_TYPES.has(req.file.mimetype)) {
      return res.status(415).json({ error: 'unsupported_file_type', message: 'Use JPEG, PNG, WebP or BMP.' });
    }
    const result = await model.predictImage(req.file.buffer, req.file.originalname, req.file.mimetype);
    const saved = await save(result, 'upload');
    res.json({ ...result, saved });
  } catch (err) {
    next(err);
  }
});

// WAV, plus the formats a browser recorder or a phone produces (decoded by ffmpeg in the model service).
const AUDIO_TYPES = new Set([
  'audio/wav', 'audio/x-wav', 'audio/wave', 'audio/vnd.wave',
  'audio/webm', 'video/webm', 'audio/ogg', 'audio/mpeg', 'audio/mp4', 'audio/x-m4a', 'audio/flac',
]);
const audioUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: config.maxAudioMb * 1024 * 1024, files: 1 },
});

api.post('/predict/audio', audioUpload.single('file'), async (req, res, next) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'missing_file', message: 'Send the audio as form field "file".' });
    // "audio/webm;codecs=opus" -> "audio/webm"
    const type = (req.file.mimetype || '').split(';')[0].trim();
    if (!AUDIO_TYPES.has(type)) {
      return res.status(415).json({ error: 'unsupported_file_type', message: 'Use WAV (or WebM, OGG, MP3, M4A, FLAC if ffmpeg is installed).' });
    }
    const result = await model.predictAudio(req.file.buffer, req.file.originalname, type);
    const source = req.body?.source === 'record' ? 'audio_record' : 'audio_upload';
    let saved = false;
    if (dbReady()) {
      await Prediction.create({
        source,
        model: result.model || null,
        top_label: result.top_label,
        confidence: result.confidence,
        audio: {
          top_label: result.top_label,
          confidence: result.confidence,
          duration_s: result.duration_s,
          probabilities: result.probabilities,
        },
      });
      saved = true;
    }
    res.json({ ...result, saved });
  } catch (err) {
    next(err);
  }
});

api.post('/predict/frames', express.json({ limit: '20mb' }), async (req, res, next) => {
  try {
    const frames = req.body?.frames;
    if (!Array.isArray(frames) || frames.length === 0) {
      return res.status(400).json({ error: 'bad_frames', message: 'Send {"frames": [base64 images]}.' });
    }
    const result = await model.predictFrames(frames);
    const saved = await save(result, 'frames');
    res.json({ ...result, saved });
  } catch (err) {
    next(err);
  }
});

api.get('/history', async (_req, res, next) => {
  try {
    if (!dbReady()) {
      return res.status(503).json({ error: 'db_unavailable', message: 'MongoDB is not connected.' });
    }
    const items = await Prediction.find({}, { __v: 0 }).sort({ timestamp: -1 }).limit(50).lean();
    res.json({ items });
  } catch (err) {
    next(err);
  }
});

api.get('/health', async (_req, res) => {
  let modelService;
  try {
    modelService = { reachable: true, ...(await model.health()) };
  } catch (err) {
    modelService = { reachable: false, error: err.code || 'model_service_unavailable', message: err.message };
  }
  res.json({ status: 'ok', modelService, db: dbStatus() });
});

// Error handler for this router: maps known errors to clean JSON responses.
export function errorHandler(err, _req, res, _next) {
  if (err instanceof multer.MulterError && err.code === 'LIMIT_FILE_SIZE') {
    const isAudio = _req.path === '/predict/audio';
    return res.status(413).json({
      error: 'file_too_large',
      message: isAudio ? `Audio is larger than ${config.maxAudioMb} MB.` : `Image is larger than ${config.maxImageMb} MB.`,
    });
  }
  if (err instanceof ModelError) {
    return res.status(err.status).json({ error: err.code, message: err.message });
  }
  if (err.type === 'entity.too.large') {
    return res.status(413).json({ error: 'payload_too_large', message: 'Request body is too large.' });
  }
  if (err.type === 'entity.parse.failed') {
    return res.status(400).json({ error: 'bad_json', message: 'Request body is not valid JSON.' });
  }
  console.error('[api] unexpected error:', err.name); // name only, so request content never reaches the logs
  res.status(500).json({ error: 'server_error', message: 'Unexpected server error.' });
}
