const ERROR_HINTS = {
  no_face_detected: "Try a well-lit, front-facing photo with the face clearly visible and not too small.",
  unsupported_file_type: "Images: JPG, PNG or WebP. Videos: MP4, WebM, MOV or AVI.",
  file_too_large: "Trim or compress the file and try again.",
  corrupt_video: "Re-export the clip as an H.264 MP4 and try again.",
  invalid_image: "The file may be damaged. Try re-saving it as a JPG or PNG.",
  backend_offline: "Start the backend (see README) and try again.",
  model_unavailable: "Copy the checkpoint files into models/ and restart the backend.",
};

export function Disclaimer() {
  return (
    <div className="sticky top-0 z-30 border-b border-amber-300 bg-amber-50 px-4 py-2 text-center text-sm text-amber-900 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-100">
      <strong>Student research prototype.</strong> Not a medical or diagnostic tool. Do not use it to
      assess anyone's mental health.
    </div>
  );
}

export function MockBadge({ className = "" }) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border border-dashed border-amber-500 bg-amber-100 px-2 py-0.5 text-xs font-semibold uppercase tracking-wide text-amber-900 dark:bg-amber-900/40 dark:text-amber-100 ${className}`}
      title="This result comes from a placeholder model and is random. It is not a real prediction."
    >
      ⚠ Placeholder data
    </span>
  );
}

export function ErrorBox({ error, onDismiss }) {
  if (!error) return null;
  const hint = ERROR_HINTS[error.code];
  return (
    <div role="alert" className="rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-900 dark:border-red-800 dark:bg-red-950 dark:text-red-100">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="font-semibold">{error.message}</p>
          {hint && <p className="mt-1 opacity-80">{hint}</p>}
        </div>
        {onDismiss && (
          <button className="opacity-60 hover:opacity-100" onClick={onDismiss} aria-label="Dismiss">✕</button>
        )}
      </div>
    </div>
  );
}

export function ComingSoon({ modality, mocked }) {
  if (mocked === false) return null;
  return (
    <div className="rounded-lg border border-dashed border-amber-500 bg-amber-50 p-3 text-sm text-amber-900 dark:bg-amber-950/50 dark:text-amber-100">
      <strong>Coming soon.</strong> The {modality} model hasn't been added yet, so the backend returns
      random placeholder data. This tab is already wired to the real endpoint and will show real
      predictions as soon as a model is registered.
    </div>
  );
}

export function BackendBanner({ health }) {
  if (health.state === "online" && health.data?.status === "ok") return null;
  if (health.state === "checking") return null;
  if (health.state === "offline") {
    return (
      <div role="alert" className="border-b border-red-300 bg-red-50 px-4 py-2 text-sm text-red-900 dark:border-red-800 dark:bg-red-950 dark:text-red-100">
        <strong>Backend offline.</strong> Can't reach the API. Start it with the backend command in
        the README. Retrying every 5 seconds…
      </div>
    );
  }
  return (
    <div role="alert" className="border-b border-amber-300 bg-amber-50 px-4 py-2 text-sm text-amber-900 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-100">
      <strong>No facial emotion model loaded.</strong> Copy resnet18_fer.pt or cnn_baseline.pt into
      models/ and restart the backend. Face predictions are unavailable until then.
    </div>
  );
}
