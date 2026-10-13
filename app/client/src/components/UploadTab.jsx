import { ImagePlus } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { uploadImage } from '../lib/api.js';
import { EMOJI, displayLabel, pct } from '../lib/emotion.js';
import { useToast } from '../lib/toast.jsx';
import FaceOverlay from './FaceOverlay.jsx';
import ProbBars from './ProbBars.jsx';
import { PageHeader, Skeleton } from './ui.jsx';

const ACCEPT = 'image/jpeg,image/png,image/webp,image/bmp';

// Drag and drop a photo, or pick one. Shows the face box, the 7 probabilities and the top label.
export default function UploadTab() {
  const toast = useToast();
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const imgRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (!file) return undefined;
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const choose = async (f) => {
    if (!f) return;
    setResult(null);
    if (!ACCEPT.split(',').includes(f.type)) {
      toast.error('Use a JPEG, PNG, WebP or BMP image.');
      return;
    }
    setFile(f);
    setLoading(true);
    try {
      setResult(await uploadImage(f));
    } catch (err) {
      toast.error(err.message);
    } finally {
      setLoading(false);
    }
  };

  const label = result?.face_found ? displayLabel(result.probabilities) : null;

  return (
    <div>
      <PageHeader title="Photo upload" subtitle="Pick a photo with a clear, front-facing face. The result is a guess about the expression, not about how a person feels." />
      <div className="grid gap-5 lg:grid-cols-2">
        <section className="space-y-4">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              choose(e.dataTransfer.files?.[0]);
            }}
            onClick={() => inputRef.current?.click()}
            onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && inputRef.current?.click()}
            role="button"
            tabIndex={0}
            aria-label="Choose an image to analyse"
            className={`glass flex min-h-48 cursor-pointer flex-col items-center justify-center gap-2 rounded-3xl border-2 border-dashed p-6 text-center transition ${
              dragOver ? 'border-accent bg-accent-soft' : 'border-line hover:border-accent'
            }`}
          >
            <ImagePlus className="h-8 w-8 text-accent" aria-hidden />
            <p className="font-medium">Drag a photo here, or click to choose one</p>
            <p className="text-xs text-muted">JPEG, PNG, WebP or BMP, up to 10 MB</p>
            <input ref={inputRef} type="file" accept={ACCEPT} className="hidden" onChange={(e) => choose(e.target.files?.[0])} />
          </div>

          {preview && (
            <div className="overflow-hidden rounded-3xl bg-slate-900 shadow">
              <div className="relative inline-block w-full">
                <img ref={imgRef} src={preview} alt="Selected upload" className="block h-auto w-full" onLoad={() => setResult((r) => (r ? { ...r } : r))} />
                {result?.face_found && <FaceOverlay targetRef={imgRef} box={result.box} label={label} confidence={result.confidence} />}
              </div>
            </div>
          )}
        </section>

        <section className="card space-y-4">
          {loading && (
            <div className="space-y-3" aria-busy="true">
              <Skeleton className="h-8 w-1/2" />
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-5/6" />
              <Skeleton className="h-4 w-2/3" />
            </div>
          )}
          {!result && !loading && <p className="text-sm text-muted">Results appear here.</p>}
          {result && !result.face_found && (
            <div className="rounded-2xl bg-coral-soft p-4 text-sm">
              <strong>No face found.</strong> No prediction is shown, because guessing from a frame without a face would be misleading. Try a photo with a clearly visible, front-facing face.
            </div>
          )}
          {result?.face_found && (
            <>
              <div className="flex items-center gap-4">
                <span className="text-6xl" aria-hidden>{label.text === 'uncertain' ? '🤔' : EMOJI[label.top]}</span>
                <div>
                  <div className="eyebrow">Top label</div>
                  <div className="text-3xl font-semibold capitalize" style={{ color: label.color }}>{label.text}</div>
                  <div className="tabular-nums text-muted">{pct(result.confidence)}</div>
                </div>
              </div>
              {label.text === 'uncertain' && <p className="text-xs text-muted">Top probability is below 50%, so no confident label is shown.</p>}
              <ProbBars probs={result.probabilities} highlight={label.top} />
            </>
          )}
          {result && (
            <p className="text-xs text-muted">
              {result.saved ? 'Saved to the detection log.' : 'Not saved: the database is not connected.'} Model: {result.model}, {result.ms} ms.
            </p>
          )}
        </section>
      </div>
    </div>
  );
}
