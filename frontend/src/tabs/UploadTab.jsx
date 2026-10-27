import { useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError, LIMITS } from "../api";
import { EmotionTimeline } from "../components/Charts";
import { CameraCapture } from "../components/MediaCapture";
import { ErrorBox } from "../components/Notices";
import ResultCard from "../components/ResultCard";

const IMAGE_EXT = [".jpg", ".jpeg", ".png", ".webp"];
const VIDEO_EXT = [".mp4", ".webm", ".mov", ".avi"];

function kindOf(file) {
  const name = file.name.toLowerCase();
  if (file.type.startsWith("image/") || IMAGE_EXT.some((e) => name.endsWith(e))) return "image";
  if (file.type.startsWith("video/") || VIDEO_EXT.some((e) => name.endsWith(e))) return "video";
  return null;
}

function FaceBox({ box }) {
  if (!box) return null;
  return (
    <div className="pointer-events-none absolute rounded-md border-2 border-green-400 shadow-[0_0_0_2px_rgba(0,0,0,0.4)]"
      style={{ left: `${box.x * 100}%`, top: `${box.y * 100}%`, width: `${box.w * 100}%`, height: `${box.h * 100}%` }} />
  );
}

export default function UploadTab({ onResult }) {
  const [file, setFile] = useState(null);
  const [kind, setKind] = useState(null);
  const [url, setUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [time, setTime] = useState(0);
  const [camera, setCamera] = useState(false);
  const videoRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => () => url && URL.revokeObjectURL(url), [url]);

  async function handle(f) {
    if (!f) return;
    setError(null);
    setResult(null);
    const k = kindOf(f);
    if (!k) {
      return setError(new ApiError(415, "unsupported_file_type", `"${f.name}" isn't a supported image or video.`));
    }
    const limit = k === "image" ? LIMITS.imageMB : LIMITS.videoMB;
    if (f.size > limit * 1024 * 1024) {
      return setError(new ApiError(413, "file_too_large", `${k === "image" ? "Images" : "Videos"} must be under ${limit} MB.`));
    }
    setFile(f);
    setKind(k);
    setUrl(URL.createObjectURL(f));
    setTime(0);
    setBusy(true);
    try {
      const r = k === "image" ? await api.predictImage(f) : await api.predictVideo(f);
      setResult(r);
      onResult?.(r);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }

  const timeline = result?.timeline ?? [];
  // Nearest sampled frame to the player's current time, for the face box and readout.
  const current = useMemo(() => {
    if (!timeline.length) return null;
    return timeline.reduce((best, p) => (Math.abs(p.t - time) < Math.abs(best.t - time) ? p : best));
  }, [timeline, time]);

  function seek(t) {
    setTime(t);
    if (videoRef.current) videoRef.current.currentTime = t;
  }

  return (
    <div className="space-y-4">
      {camera ? (
        <CameraCapture onCapture={(f) => { setCamera(false); handle(f); }} onClose={() => setCamera(false)} />
      ) : (
        <button className="btn-primary" onClick={() => setCamera(true)}>📷 Take a photo with the camera</button>
      )}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); handle(e.dataTransfer.files[0]); }}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        className={`card flex cursor-pointer flex-col items-center justify-center gap-1 border-2 border-dashed py-10 text-center transition ${dragging ? "!border-blue-500 bg-blue-50 dark:bg-blue-950/30" : ""}`}
      >
        <p className="font-medium">Drop an image or video here, or click to choose</p>
        <p className="text-xs text-stone-500">
          Images: JPG, PNG, WebP up to {LIMITS.imageMB} MB · Videos: MP4, WebM, MOV, AVI up to {LIMITS.videoMB} MB
        </p>
        <input ref={inputRef} type="file" className="hidden" accept={[...IMAGE_EXT, ...VIDEO_EXT].join(",")}
          onChange={(e) => { handle(e.target.files[0]); e.target.value = ""; }} />
      </div>

      <ErrorBox error={error} onDismiss={() => setError(null)} />
      {busy && (
        <p className="text-sm text-stone-500" role="status">
          Analysing {file?.name}… {kind === "video" && "Videos are processed on CPU at 5 frames per second, so this can take a while."}
        </p>
      )}

      {url && (
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="card p-2">
            <div className="relative overflow-hidden rounded-lg bg-black">
              <div className="relative">
                {kind === "image" ? (
                  <img src={url} alt="Uploaded" className="block w-full" />
                ) : (
                  <video ref={videoRef} src={url} controls className="block w-full"
                    onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
                    onSeeked={(e) => setTime(e.currentTarget.currentTime)} />
                )}
                {kind === "image" && <FaceBox box={result?.face_box} />}
                {kind === "video" && current?.face && <FaceBox box={current.face_box} />}
              </div>
            </div>
            {kind === "video" && current && (
              <p className="mt-2 px-1 text-sm tabular">
                At {current.t.toFixed(1)}s:{" "}
                {current.face ? <strong className="capitalize">{current.label} ({(current.confidence * 100).toFixed(0)}%)</strong>
                  : <span className="text-stone-500">no face in this frame</span>}
              </p>
            )}
          </div>

          <ResultCard title={kind === "video" ? "Clip result (smoothed)" : "Image result"} result={result}>
            {result && (
              <p className="text-xs text-stone-500 dark:text-stone-400">
                Model: {result.model}
                {kind === "image" && ` · ${result.faces_found} face${result.faces_found === 1 ? "" : "s"} found (largest used)`}
                {kind === "video" && ` · ${result.frames_sampled} frames sampled, ${result.faces_found} with a face, ${result.frames_without_face} without`}
              </p>
            )}
          </ResultCard>
        </div>
      )}

      {kind === "video" && timeline.length > 0 && (
        <div className="card space-y-2">
          <h3 className="text-sm font-semibold uppercase tracking-wide text-stone-500 dark:text-stone-400">
            Timeline (5-frame moving average) · click the chart or use the slider to scrub
          </h3>
          <EmotionTimeline points={timeline} domain={[0, Math.max(result.duration, 0.1)]} cursor={time} onSeek={seek} />
          <input type="range" min={0} max={result.duration} step={0.05} value={Math.min(time, result.duration)}
            onChange={(e) => seek(Number(e.target.value))} className="w-full" aria-label="Scrub video" />
        </div>
      )}
    </div>
  );
}
