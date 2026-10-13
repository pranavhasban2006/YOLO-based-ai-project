import { useEffect, useRef } from 'react';
import { pct } from '../lib/emotion.js';

// Draws the face box and label on a canvas sitting on top of a video or image.
// `box` is normalised {x, y, w, h}; `label` is {text, color} from displayLabel; `confidence` in 0..1.
export default function FaceOverlay({ targetRef, box, label, confidence }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const target = targetRef.current;
    if (!canvas || !target) return;
    const W = target.clientWidth;
    const H = target.clientHeight;
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);
    if (!box || !label) return;

    const x = box.x * W;
    const y = box.y * H;
    const w = box.w * W;
    const h = box.h * H;
    ctx.lineWidth = 3;
    ctx.strokeStyle = label.color;
    ctx.strokeRect(x, y, w, h);

    const text = `${label.text} ${confidence != null ? pct(confidence) : ''}`.trim();
    ctx.font = '600 14px system-ui, sans-serif';
    const tw = ctx.measureText(text).width + 12;
    const ty = y - 22 < 0 ? y + 4 : y - 22;
    ctx.fillStyle = label.color;
    ctx.fillRect(x, ty, tw, 20);
    ctx.fillStyle = '#0f172a';
    ctx.fillText(text, x + 6, ty + 15);
  });

  return <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true" />;
}
