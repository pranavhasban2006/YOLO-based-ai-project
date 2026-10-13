import { AnimatePresence, motion } from 'framer-motion';
import { AlertCircle, CheckCircle2, X } from 'lucide-react';
import { createContext, useCallback, useContext, useMemo, useRef, useState } from 'react';

const ToastCtx = createContext({ error: () => {}, success: () => {} });

// Small toasts for errors and confirmations. They announce themselves to screen readers (role="status" / "alert").
export function ToastProvider({ children }) {
  const [items, setItems] = useState([]);
  const nextId = useRef(1);
  const remove = useCallback((id) => setItems((xs) => xs.filter((x) => x.id !== id)), []);
  const push = useCallback(
    (kind, message) => {
      const id = nextId.current++;
      setItems((xs) => [...xs.slice(-3), { id, kind, message }]);
      setTimeout(() => remove(id), kind === 'error' ? 7000 : 4000);
    },
    [remove],
  );
  const api = useMemo(() => ({ error: (m) => push('error', m), success: (m) => push('success', m) }), [push]);

  return (
    <ToastCtx.Provider value={api}>
      {children}
      <div className="pointer-events-none fixed inset-x-0 bottom-4 z-50 flex flex-col items-center gap-2 px-4">
        <AnimatePresence>
          {items.map((t) => (
            <motion.div
              key={t.id}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 8 }}
              role={t.kind === 'error' ? 'alert' : 'status'}
              className={`pointer-events-auto flex max-w-md items-start gap-2 rounded-2xl border px-4 py-3 text-sm shadow-lg backdrop-blur ${
                t.kind === 'error' ? 'border-coral/40 bg-solid text-ink' : 'border-accent/40 bg-solid text-ink'
              }`}
            >
              {t.kind === 'error' ? <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-coral" aria-hidden /> : <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-accent" aria-hidden />}
              <span>{t.message}</span>
              <button type="button" onClick={() => remove(t.id)} aria-label="Dismiss" className="ml-1 text-muted hover:text-ink">
                <X className="h-4 w-4" />
              </button>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </ToastCtx.Provider>
  );
}

export const useToast = () => useContext(ToastCtx);
