import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import FusedCard from "./components/FusedCard";
import { BackendBanner, Disclaimer } from "./components/Notices";
import HistoryPage from "./tabs/HistoryPage";
import LiveTab from "./tabs/LiveTab";
import { AudioTab, EegTab, TextTab } from "./tabs/StubTabs";
import UploadTab from "./tabs/UploadTab";

const TABS = [
  { id: "live", label: "Live" },
  { id: "upload", label: "Upload" },
  { id: "audio", label: "Audio", modality: "audio" },
  { id: "text", label: "Text", modality: "text" },
  { id: "eeg", label: "EEG", modality: "eeg" },
  { id: "history", label: "History" },
];

function useHealth() {
  const [health, setHealth] = useState({ state: "checking", data: null });
  useEffect(() => {
    let alive = true;
    const check = () =>
      api.health()
        .then((data) => alive && setHealth({ state: "online", data }))
        .catch(() => alive && setHealth({ state: "offline", data: null }));
    check();
    const id = setInterval(check, 5000);
    return () => { alive = false; clearInterval(id); };
  }, []);
  return health;
}

export default function App() {
  const [tab, setTab] = useState("live");
  const [latest, setLatest] = useState({});
  const health = useHealth();
  const online = health.state === "online";
  const mocked = new Set(health.data?.mocked ?? ["audio", "text", "eeg"]);
  const isMocked = (m) => (health.data ? mocked.has(m) : undefined);

  // Most recent result per modality, used by the fused card.
  const onResult = useCallback((p) => setLatest((l) => ({ ...l, [p.modality]: p })), []);

  return (
    <div className="min-h-screen">
      <Disclaimer />
      <BackendBanner health={health} />
      <header className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 pt-5">
        <div>
          <h1 className="text-xl font-semibold">Multimodal Emotion Detection</h1>
          <p className="text-xs text-stone-500">
            Face model: {health.data?.face?.active_model ?? "—"} ·{" "}
            <span className={online ? "text-green-700 dark:text-green-400" : "text-red-600"}>
              {online ? "backend online" : health.state === "checking" ? "connecting…" : "backend offline"}
            </span>
          </p>
        </div>
        <nav className="flex flex-wrap gap-1" role="tablist">
          {TABS.map((t) => (
            <button key={t.id} role="tab" aria-selected={tab === t.id} onClick={() => setTab(t.id)}
              className={`btn ${tab === t.id ? "bg-stone-900 text-white dark:bg-white dark:text-black" : "hover:bg-black/5 dark:hover:bg-white/10"}`}>
              {t.label}
              {t.modality && isMocked(t.modality) !== false && (
                <span className="rounded bg-amber-200 px-1 text-[10px] font-semibold uppercase text-amber-900">soon</span>
              )}
            </button>
          ))}
        </nav>
      </header>

      <main className="mx-auto grid max-w-7xl gap-6 px-4 py-5 xl:grid-cols-[1fr_340px]">
        <div className="min-w-0">
          {tab === "live" && <LiveTab onResult={onResult} backendOnline={online} />}
          {tab === "upload" && <UploadTab onResult={onResult} />}
          {tab === "audio" && <AudioTab onResult={onResult} mocked={isMocked("audio")} />}
          {tab === "text" && <TextTab onResult={onResult} mocked={isMocked("text")} />}
          {tab === "eeg" && <EegTab onResult={onResult} mocked={isMocked("eeg")} backendOnline={online} />}
          {tab === "history" && <HistoryPage />}
        </div>
        <aside className="space-y-4">
          <FusedCard latest={latest} />
        </aside>
      </main>
    </div>
  );
}
