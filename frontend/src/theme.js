import { useEffect, useState } from "react";

// Categorical palette, one fixed slot per emotion in class order.
// Validated with the dataviz skill's validator (adjacent CVD and normal-vision checks pass
// in both modes). Light-mode contrast relief: every chart labels emotions directly.
const EMOTION_COLORS = {
  light: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"],
  dark: ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"],
};

const CHROME = {
  light: { grid: "#e1e0d9", axis: "#c3c2b7", muted: "#898781", text: "#52514e", surface: "#fcfcfb" },
  dark: { grid: "#2c2c2a", axis: "#383835", muted: "#898781", text: "#c3c2b7", surface: "#1a1a19" },
};

export function useScheme() {
  const query = "(prefers-color-scheme: dark)";
  const [dark, setDark] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const mq = window.matchMedia(query);
    const on = (e) => setDark(e.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  const mode = dark ? "dark" : "light";
  return { emotion: EMOTION_COLORS[mode], chrome: CHROME[mode] };
}
