import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';

const KEY = 'wb_theme';
const ThemeCtx = createContext({ dark: false, toggle: () => {} });

function initial() {
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === 'dark' || saved === 'light') return saved === 'dark';
  } catch {
    /* storage can be blocked */
  }
  return typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: dark)').matches;
}

// Light/dark toggle, remembered in localStorage. The class is set on <html> so Tailwind's dark: variant works.
export function ThemeProvider({ children }) {
  const [dark, setDark] = useState(initial);
  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark);
    try {
      localStorage.setItem(KEY, dark ? 'dark' : 'light');
    } catch {
      /* ignore */
    }
  }, [dark]);
  const toggle = useCallback(() => setDark((d) => !d), []);
  const value = useMemo(() => ({ dark, toggle }), [dark, toggle]);
  return <ThemeCtx.Provider value={value}>{children}</ThemeCtx.Provider>;
}

export const useTheme = () => useContext(ThemeCtx);

// Chart colours as plain values (SVG attributes), matching the CSS variables.
export function chartColors(dark) {
  return dark
    ? { accent: '#2dd4bf', violet: '#a89cff', coral: '#ff8a82', amber: '#f0b45a', grid: 'rgba(190,225,232,0.14)', text: '#9db4bc', surface: '#15242b' }
    : { accent: '#0b7f78', violet: '#6b5bd6', coral: '#cf4a43', amber: '#9a6208', grid: 'rgba(18,48,58,0.12)', text: '#4f6a75', surface: '#ffffff' };
}
