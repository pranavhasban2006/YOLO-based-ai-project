import { useEffect, useRef, useState } from "react";
import { WS_URL } from "../api";
import { EmotionTimeline, ProbBars } from "../components/Charts";
import { ErrorBox } from "../components/Notices";
import { ResultHeadline } from "../components/ResultCard";

const FRAME_INTERVAL_MS = 200; // ~5 fps
const WINDOW_S = 30;
const SEND_WIDTH = 480; // frames are downscaled before sending

const CAMERA_ERRORS = {
  denied: {
    title: "Camera permission denied",
    body: "Allow camera access for this site in your browser's address bar (camera icon), then press Start again.",
  },
  nocamera: { title: "No camera found", body: "Connect a webcam, or check that no other app is using it." },
  insecure: { title: "Camera unavailable", body: "Browsers only allow camera access on https or localhost." },
};

export default function LiveTab({ onResult, backendOnline }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const timerRef = useRef(null);
  const startRef = useRef(0);

  const [status, setStatus] = useState("idle"); // idle | starting | running | denied | nocamera | insecure
  const [result, setResult] = useState(null); // last face prediction
  const [noFace, setNoFace] = useState(false);
  const [points, setPoints] = useState([]);
  const [now, setNow] = useState(0);
  const [error, setError] = useState(null);

  function stop() {
    clearTimeout(timerRef.current);
    const ws = wsRef.current;
    wsRef.current = null;
    if (ws) {
      ws.onclose = null;
      if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "end" }));
      ws.close();
    }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setStatus((s) => (s === "running" || s === "starting" ? "idle" : s));
  }

  useEffect(() => stop, []);

  function sendFrame() {
    const ws = wsRef.current;
    const video = videoRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN || !video?.videoWidth) {
      timerRef.current = setTimeout(sendFrame, FRAME_INTERVAL_MS);
      return;
    }
    const canvas = canvasRef.current;
    canvas.width = SEND_WIDTH;
    canvas.height = Math.round((video.videoHeight / video.videoWidth) * SEND_WIDTH);
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    const sentAt = performance.now();
    canvas.toBlob((blob) => {
      if (!blob || wsRef.current !== ws) return;
      ws._sentAt = sentAt;
      ws.send(blob); // in memory only; the backend never writes frames to disk
    }, "image/jpeg", 0.8);
  }

  function handleMessage(ws, event) {
    const msg = JSON.parse(event.data);
    const t = (performance.now() - startRef.current) / 1000;
    setNow(t);
    if (msg.type === "prediction") {
      setResult(msg);
      setNoFace(false);
      setError(null);
      onResult?.(msg);
      setPoints((ps) => [...ps.filter((p) => p.t >= t - WINDOW_S - 1), { t, probs: msg.probs }]);
    } else if (msg.type === "no_face") {
      setNoFace(true);
      setPoints((ps) => [...ps.filter((p) => p.t >= t - WINDOW_S - 1), { t, probs: null }]);
    } else if (msg.type === "error") {
      setError({ code: msg.error, message: msg.message });
      if (msg.error === "model_unavailable") return stop();
    }
    // Wait for each reply before sending the next frame, so the backend is never flooded.
    const elapsed = performance.now() - (ws._sentAt ?? 0);
    timerRef.current = setTimeout(sendFrame, Math.max(0, FRAME_INTERVAL_MS - elapsed));
  }

  async function start() {
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia) return setStatus("insecure");
    setStatus("starting");
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    } catch (e) {
      if (e.name === "NotAllowedError" || e.name === "SecurityError") return setStatus("denied");
      if (e.name === "NotFoundError" || e.name === "OverconstrainedError" || e.name === "NotReadableError")
        return setStatus("nocamera");
      setStatus("idle");
      return setError({ code: "camera_error", message: `Could not start the camera: ${e.message}` });
    }
    streamRef.current = stream;
    videoRef.current.srcObject = stream;
    await videoRef.current.play().catch(() => {});

    setPoints([]);
    setResult(null);
    setNoFace(false);
    startRef.current = performance.now();
    const ws = new WebSocket(`${WS_URL}/ws/webcam`);
    wsRef.current = ws;
    ws.onopen = () => { setStatus("running"); sendFrame(); };
    ws.onmessage = (e) => handleMessage(ws, e);
    ws.onclose = () => {
      if (wsRef.current !== ws) return;
      setError({ code: "backend_offline", message: "Lost connection to the backend." });
      stop();
    };
  }

  const running = status === "running" || status === "starting";
  const camError = CAMERA_ERRORS[status];
  const box = !noFace && result?.face_box;
  const domain = [Math.max(0, now - WINDOW_S), Math.max(WINDOW_S, now)];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        {!running ? (
          <button className="btn-primary" onClick={start} disabled={!backendOnline}>Start camera</button>
        ) : (
          <button className="btn-ghost" onClick={stop}>Stop</button>
        )}
        {!backendOnline && !running && <span className="text-sm text-stone-500">Backend offline. Start it to use the live view.</span>}
        <span className="text-xs text-stone-500">Frames are analysed in memory and never saved. Only predictions are stored.</span>
      </div>

      {camError && (
        <div role="alert" className="rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-900 dark:border-red-800 dark:bg-red-950 dark:text-red-100">
          <p className="font-semibold">{camError.title}</p>
          <p className="mt-1 opacity-80">{camError.body}</p>
        </div>
      )}
      <ErrorBox error={error} onDismiss={() => setError(null)} />

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="card p-2">
          <div className="relative overflow-hidden rounded-lg bg-black">
            {/* The preview is mirrored like a selfie view; the box is mirrored with it. */}
            <div className="relative -scale-x-100">
              <video ref={videoRef} muted playsInline className="block aspect-[4/3] w-full object-cover" />
              {box && (
                <div className="absolute rounded-md border-2 border-green-400 shadow-[0_0_0_2px_rgba(0,0,0,0.4)]"
                  style={{ left: `${box.x * 100}%`, top: `${box.y * 100}%`, width: `${box.w * 100}%`, height: `${box.h * 100}%` }} />
              )}
            </div>
            {box && (
              <div className="absolute rounded bg-green-500 px-1.5 text-xs font-semibold text-black capitalize"
                style={{ left: `${(1 - box.x - box.w) * 100}%`, top: `calc(${box.y * 100}% - 1.25rem)` }}>
                {result.label} {(result.confidence * 100).toFixed(0)}%
              </div>
            )}
            {running && noFace && (
              <div className="absolute inset-x-0 bottom-0 bg-black/70 p-2 text-center text-sm text-white">
                No face detected. Face the camera in good light.
              </div>
            )}
            {!running && (
              <div className="absolute inset-0 flex items-center justify-center text-sm text-white/70">
                Camera off
              </div>
            )}
          </div>
          <canvas ref={canvasRef} className="hidden" />
        </div>

        <div className="card space-y-3">
          {result ? <ResultHeadline result={result} /> : <p className="text-sm text-stone-500">Waiting for a face…</p>}
          <ProbBars probs={result?.probs} dimmed={noFace} />
        </div>
      </div>

      <div className="card">
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-stone-500 dark:text-stone-400">
          Last {WINDOW_S} seconds
        </h3>
        {points.length ? (
          <EmotionTimeline points={points.filter((p) => p.t >= domain[0])} domain={domain} />
        ) : (
          <p className="py-10 text-center text-sm text-stone-500">The timeline fills in once the camera is running. Gaps mean no face was detected.</p>
        )}
      </div>
    </div>
  );
}
