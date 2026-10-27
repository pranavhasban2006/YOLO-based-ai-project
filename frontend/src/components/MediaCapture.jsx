import { useEffect, useRef, useState } from "react";

function stopStream(stream) {
  stream?.getTracks().forEach((t) => t.stop());
}

function mediaError(e, what) {
  if (!navigator.mediaDevices?.getUserMedia) return `${what} access needs http://localhost or https.`;
  if (e?.name === "NotAllowedError") return `${what} permission was denied. Allow it from the icon in the address bar.`;
  if (e?.name === "NotFoundError") return `No ${what.toLowerCase()} found.`;
  return `Could not open the ${what.toLowerCase()}: ${e?.message ?? e}`;
}

/** Webcam preview with a "Take photo" button. Calls onCapture(File) with a JPEG of the frame. */
export function CameraCapture({ onCapture, onClose }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [error, setError] = useState(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    if (!navigator.mediaDevices?.getUserMedia) {
      setError(mediaError(null, "Camera"));
      return undefined;
    }
    navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false })
      .then((s) => {
        if (cancelled) return stopStream(s);
        streamRef.current = s;
        if (videoRef.current) videoRef.current.srcObject = s;
      })
      .catch((e) => setError(mediaError(e, "Camera")));
    return () => { cancelled = true; stopStream(streamRef.current); };
  }, []);

  function snap() {
    const v = videoRef.current;
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext("2d").drawImage(v, 0, 0);
    c.toBlob((b) => onCapture(new File([b], `camera-${Date.now()}.jpg`, { type: "image/jpeg" })), "image/jpeg", 0.92);
  }

  return (
    <div className="card space-y-3">
      {error ? <p className="text-sm text-red-600">{error}</p> : (
        <video ref={videoRef} autoPlay playsInline muted onLoadedData={() => setReady(true)}
          className="block w-full max-w-xl rounded-lg bg-black" style={{ transform: "scaleX(-1)" }} />
      )}
      <div className="flex gap-2">
        <button className="btn-primary" disabled={!ready} onClick={snap}>Take photo &amp; analyse</button>
        <button className="btn-ghost" onClick={onClose}>Close camera</button>
      </div>
    </div>
  );
}

/** Microphone recorder. Calls onRecorded(File) with the recording when stopped. */
export function AudioRecorder({ onRecorded, disabled }) {
  const recRef = useRef(null);
  const timerRef = useRef(null);
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [error, setError] = useState(null);

  useEffect(() => () => {
    clearInterval(timerRef.current);
    if (recRef.current?.state === "recording") recRef.current.stop();
  }, []);

  async function start() {
    setError(null);
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      return setError(mediaError(e, "Microphone"));
    }
    const chunks = [];
    const rec = new MediaRecorder(stream);
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    rec.onstop = () => {
      stopStream(stream);
      clearInterval(timerRef.current);
      setRecording(false);
      const type = (rec.mimeType || "audio/webm").split(";")[0];
      const ext = type.includes("ogg") ? "ogg" : type.includes("mp4") ? "m4a" : "webm";
      onRecorded(new File(chunks, `recording-${Date.now()}.${ext}`, { type }));
    };
    recRef.current = rec;
    rec.start();
    setSeconds(0);
    setRecording(true);
    timerRef.current = setInterval(() => setSeconds((s) => {
      if (s + 1 >= 30 && rec.state === "recording") rec.stop();   // hard limit: 30 s
      return s + 1;
    }), 1000);
  }

  return (
    <div className="flex flex-wrap items-center gap-3">
      {recording ? (
        <button className="btn-primary !bg-red-600" onClick={() => recRef.current.stop()}>
          ■ Stop &amp; analyse ({seconds}s)
        </button>
      ) : (
        <button className="btn-primary" disabled={disabled} onClick={start}>● Record from microphone</button>
      )}
      {recording && <span className="animate-pulse text-sm text-red-600">Recording… speak for 3–5 seconds</span>}
      {error && <span className="text-sm text-red-600">{error}</span>}
    </div>
  );
}
