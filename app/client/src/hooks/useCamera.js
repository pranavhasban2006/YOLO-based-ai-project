import { useCallback, useEffect, useRef, useState } from 'react';

// Owns the webcam stream. `status` is one of: idle, requesting, on, denied, no_camera, unsupported, error.
export function useCamera() {
  const [stream, setStream] = useState(null);
  const [status, setStatus] = useState('idle');
  const [message, setMessage] = useState('');
  const streamRef = useRef(null);
  const pendingRef = useRef(null);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setStream(null);
    setStatus('idle');
  }, []);

  const request = async () => {
    if (!navigator.mediaDevices?.getUserMedia || !window.isSecureContext) {
      setStatus('unsupported');
      setMessage('This browser cannot open the camera here. Use Chrome, Edge or Firefox on http://localhost.');
      return false;
    }
    setStatus('requesting');
    setMessage('');
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false,
      });
      streamRef.current = s;
      setStream(s);
      setStatus('on');
      return true;
    } catch (err) {
      if (err.name === 'NotAllowedError' || err.name === 'SecurityError') {
        setStatus('denied');
        setMessage('Camera permission was denied. Allow camera access for this site in the browser address bar, then press Start again.');
      } else if (err.name === 'NotFoundError' || err.name === 'OverconstrainedError') {
        setStatus('no_camera');
        setMessage('No camera was found. Connect a webcam, or use the Upload tab to analyse photos.');
      } else if (err.name === 'NotReadableError') {
        setStatus('error');
        setMessage('The camera is in use by another app. Close it and press Start again.');
      } else {
        setStatus('error');
        setMessage(`Camera error: ${err.message}`);
      }
      return false;
    }
  };

  // Concurrent calls (for example StrictMode's second mount) share one permission request.
  const start = useCallback(() => {
    if (streamRef.current) return Promise.resolve(true);
    if (!pendingRef.current) {
      pendingRef.current = request().finally(() => {
        pendingRef.current = null;
      });
    }
    return pendingRef.current;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Ask for permission as soon as the page loads.
  useEffect(() => {
    start();
  }, [start]);

  // Release the camera only when the page is really closed.
  useEffect(
    () => () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    },
    [],
  );

  return { stream, status, message, start, stop };
}
