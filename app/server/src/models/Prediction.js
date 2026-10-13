import mongoose from 'mongoose';

// One stored result summary. No image or frame data is ever stored.
const predictionSchema = new mongoose.Schema(
  {
    timestamp: { type: Date, default: Date.now, index: true },
    source: { type: String, enum: ['upload', 'frames', 'webcam', 'audio_upload', 'audio_record'], required: true },
    // Face fields are unused for audio rows; the audio result lives in `audio`.
    face_found: { type: Boolean, default: false },
    probabilities: {
      type: {
        angry: Number,
        disgust: Number,
        fear: Number,
        happy: Number,
        neutral: Number,
        sad: Number,
        surprise: Number,
      },
      default: null,
    },
    top_label: { type: String, default: null },
    confidence: { type: Number, default: null },
    model: { type: String, default: null },
    // Speech result summary (no audio is stored).
    audio: {
      type: {
        top_label: String,
        confidence: Number,
        duration_s: Number,
        probabilities: {
          angry: Number,
          disgust: Number,
          fear: Number,
          happy: Number,
          neutral: Number,
          sad: Number,
          surprise: Number,
        },
      },
      default: undefined,
    },
  },
  { versionKey: false },
);

export const Prediction = mongoose.model('Prediction', predictionSchema);
