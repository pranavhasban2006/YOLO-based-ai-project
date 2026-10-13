import { AnimatePresence, motion } from 'framer-motion';
import { Camera, CameraOff, Video } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useCamera } from '../hooks/useCamera.js';
import { useLiveStream } from '../hooks/useLiveStream.js';
import { EMOJI, displayLabel, pct } from '../lib/emotion.js';
import FaceOverlay from './FaceOverlay.jsx';
import ProbBars from './ProbBars.jsx';
import RollingChart from './RollingChart.jsx';
import { PageHeader } from './ui.jsx';

const LINK_TEXT = { idle: 'Not detecting', connecting: 'Connecting…', open: 'Live', reconnecting: 'Reconnecting…' };

// Live emotion: camera with real-time detection. Frames are processed in memory and never saved;
// only a result summary is stored, and only while "Save session" is ticked.
export default function LiveTab({ modelInfo, onGoUpload }) {
  const camera = useCamera();
  const videoRef = useRef(null);
  const [detecting, setDetecting] = useState(false);
  const [save, setSave] = useState(false);
  const [busy, setBusy] = useState(false);

  const live = useLiveStream({ videoRef, enabled: detecting && camera.status === 'on', save });

  useEffect(() => {
    if (videoRef.current && videoRef.current.srcObject !== camera.stream) videoRef.current.srcObject = camera.stream;
  }, [camera.stream]);

  const start = async () => {
    setBusy(true);
    const ok = await camera.start();
    setBusy(false);
    if (ok) setDetecting(true);
  };
  const stop = () => {
    setDetecting(false);
    camera.stop(); // releases the camera, so the "camera on" light goes off
  };

  const cameraOn = camera.status === 'on';
  const label = live.smoothed ? displayLabel(live.smoothed) : null;
  const noFace = detecting && live.link === 'open' && live.result && !live.result.face_found;
  const lowConf = label && label.text === 'uncertain';
  const headline = !detecting ? 'Stopped' : live.link !== 'open' ? LINK_TEXT[live.link] : label ? label.text : 'No face';
  const emoji = detecting && label && !lowConf ? EMOJI[label.top] : '🙂';

  return (
    <div>
      <PageHeader title="Live emotion" subtitle="Your camera is processed on this computer, frame by frame, and never recorded." />
      <div className="grid items-start gap-5 lg:grid-cols-[minmax(0,3fr)_minmax(280px,2fr)]">
        <section className="card overflow-hidden !p-0">
          <div className="relative bg-slate-900">
            <video ref={videoRef} autoPlay playsInline muted className="block h-auto min-h-56 w-full" aria-label="Webcam preview" />
            {!cameraOn && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-6 text-center text-slate-200">
                <CameraOff className="h-9 w-9 opacity-70" aria-hidden />
                <p className="max-w-md text-sm">
                  {camera.status === 'requesting' && 'Waiting for camera permission…'}
                  {camera.status === 'idle' && 'Camera is off. Press Start to turn it on and begin detection.'}
                  {['denied', 'no_camera', 'unsupported', 'error'].includes(camera.status) && camera.message}
                </p>
              </div>
            )}
            {detecting && cameraOn && (
              <FaceOverlay
                targetRef={videoRef}
                box={live.result?.face_found && live.result.box}
                label={live.smoothed ? label : live.result?.face_found ? { text: 'uncertain', color: '#64748b' } : null}
                confidence={live.smoothed ? live.top.confidence : live.result?.confidence}
              />
            )}
            {cameraOn && (
              <div className="pointer-events-none absolute left-3 top-3 flex items-center gap-2 rounded-full bg-black/60 px-3 py-1 text-xs text-white">
                <span className="inline-block h-2.5 w-2.5 animate-pulse rounded-full bg-red-500" aria-hidden />
                Camera is on
              </div>
            )}
            {detecting && (
              <div className="pointer-events-none absolute right-3 top-3 space-y-0.5 rounded-xl bg-black/60 px-2.5 py-1.5 text-xs tabular-nums text-white">
                <div>FPS {live.stats.fps.toFixed(1)}</div>
                <div>Latency {live.stats.latencyMs != null ? `${live.stats.latencyMs} ms` : '-'}</div>
                <div>Model {modelInfo?.model || '…'}</div>
                <div>Dropped {live.stats.dropped}</div>
              </div>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-3 p-4 sm:p-5">
            {!detecting ? (
              <button type="button" onClick={start} disabled={busy} className="btn-primary">
                <Camera className="h-4 w-4" aria-hidden /> {busy ? 'Starting…' : 'Start'}
              </button>
            ) : (
              <button type="button" onClick={stop} className="btn-danger">
                <Video className="h-4 w-4" aria-hidden /> Stop
              </button>
            )}
            <label className="flex items-center gap-2 text-sm text-ink">
              <input type="checkbox" checked={save} onChange={(e) => setSave(e.target.checked)} className="h-4 w-4 accent-[var(--accent)]" />
              Save session
              <span className="text-xs text-muted">(timestamp and probabilities only, at most once a second; no images)</span>
            </label>
          </div>
          {['denied', 'no_camera', 'unsupported', 'error'].includes(camera.status) && (
            <div className="space-y-2 border-t border-line bg-coral-soft p-4 text-sm">
              <p>{camera.message}</p>
              <button type="button" onClick={onGoUpload} className="btn-soft">
                Go to Upload
              </button>
            </div>
          )}
        </section>

        <aside className="space-y-5">
          <div className="card text-center">
            <div className="eyebrow">Smoothed label (about 5 frames)</div>
            <AnimatePresence mode="wait">
              <motion.div
                key={emoji}
                initial={{ scale: 0.6, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ type: 'spring', stiffness: 260, damping: 18 }}
                className="mt-2 text-7xl leading-none"
                aria-hidden
              >
                {emoji}
              </motion.div>
            </AnimatePresence>
            <div className="mt-3 text-3xl font-semibold capitalize" style={{ color: detecting && label && !lowConf ? label.color : 'var(--ink)' }}>
              {headline}
            </div>
            {label && live.top && detecting && <div className="tabular-nums text-muted">{pct(live.top.confidence)}</div>}
            <div className="mt-1 text-xs text-muted">
              {detecting && live.result?.face_found && `Raw: ${live.result.top_label} ${pct(live.result.confidence)}`}
              {detecting && noFace && 'Raw: no face in this frame'}
              {lowConf && ' · below 50% confidence, so not shown as a label'}
            </div>
            {live.error && <p className="mt-2 text-xs text-amber">{live.error}</p>}
            {detecting && live.link !== 'open' && (
              <p className="mt-2 rounded-xl bg-coral-soft px-3 py-2 text-xs" role="status">
                {live.link === 'reconnecting' ? 'Connection lost. Reconnecting to the server…' : 'Connecting to the server…'}
                {live.modelUp === false && ' The model service is not running.'}
              </p>
            )}
          </div>

          <div className="card">
            <div className="eyebrow mb-3">Probabilities</div>
            <ProbBars probs={live.smoothed} highlight={label && !lowConf ? label.top : null} />
          </div>

          <div className="card">
            <div className="eyebrow mb-3">Last 30 seconds</div>
            <RollingChart series={live.series} />
          </div>
        </aside>
      </div>
      <p className="mt-4 text-xs text-muted">
        FPS and latency are measured in the browser over the last two seconds. Dropped counts camera frames skipped because the previous result had not come back yet.
      </p>
    </div>
  );
}
