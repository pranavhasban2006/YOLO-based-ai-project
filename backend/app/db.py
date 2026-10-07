"""SQLite storage for session results. Only predictions are stored; never images or frames."""
import json
import sqlite3
import threading
from pathlib import Path

from .schemas import now_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,          -- image | video | webcam | audio | text | eeg | fused
    created_at  TEXT NOT NULL,
    ended_at    TEXT,
    label       TEXT,                   -- session-level result, if any
    confidence  REAL,
    mock        INTEGER NOT NULL DEFAULT 0,
    meta        TEXT                    -- JSON: model, file name, timeline, fusion details...
);
CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    modality    TEXT NOT NULL,
    label       TEXT NOT NULL,
    confidence  REAL NOT NULL,
    probs       TEXT NOT NULL,          -- JSON list of 7 floats
    mock        INTEGER NOT NULL,
    timestamp   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_session ON predictions(session_id);
"""


class Store:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._lock = threading.Lock()

    def close(self):
        self._conn.close()

    def create_session(self, kind: str, meta: dict | None = None, mock: bool = False) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO sessions (kind, created_at, mock, meta) VALUES (?, ?, ?, ?)",
                (kind, now_iso(), int(mock), json.dumps(meta or {})))
            return cur.lastrowid

    def add_prediction(self, session_id: int, pred) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO predictions (session_id, modality, label, confidence, probs, mock, timestamp)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, pred.modality, pred.label, pred.confidence,
                 json.dumps(pred.probs), int(pred.mock), pred.timestamp))

    def finish_session(self, session_id: int, label: str | None = None,
                       confidence: float | None = None, meta: dict | None = None) -> None:
        with self._lock, self._conn:
            if meta is not None:
                self._conn.execute("UPDATE sessions SET meta = ? WHERE id = ?",
                                   (json.dumps(meta), session_id))
            self._conn.execute(
                "UPDATE sessions SET ended_at = ?, label = ?, confidence = ? WHERE id = ?",
                (now_iso(), label, confidence, session_id))

    def record(self, kind: str, pred, meta: dict | None = None) -> int:
        """Store a one-shot session with a single prediction."""
        sid = self.create_session(kind, meta, mock=pred.mock)
        self.add_prediction(sid, pred)
        self.finish_session(sid, pred.label, pred.confidence)
        return sid

    def list_sessions(self, limit: int = 100, offset: int = 0) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT s.id, s.kind, s.created_at, s.ended_at, s.label, s.confidence, s.mock,
                          COUNT(p.id) AS n_predictions
                   FROM sessions s LEFT JOIN predictions p ON p.session_id = s.id
                   GROUP BY s.id ORDER BY s.id DESC LIMIT ? OFFSET ?""",
                (limit, offset)).fetchall()
        return [{**dict(r), "mock": bool(r["mock"])} for r in rows]

    def get_session(self, session_id: int) -> dict | None:
        with self._lock:
            s = self._conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if s is None:
                return None
            preds = self._conn.execute(
                "SELECT modality, label, confidence, probs, mock, timestamp FROM predictions"
                " WHERE session_id = ? ORDER BY id", (session_id,)).fetchall()
        out = dict(s)
        out["mock"] = bool(out["mock"])
        out["meta"] = json.loads(out["meta"] or "{}")
        out["predictions"] = [
            {**dict(p), "probs": json.loads(p["probs"]), "mock": bool(p["mock"])} for p in preds]
        return out
