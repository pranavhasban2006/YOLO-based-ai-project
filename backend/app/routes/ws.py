"""WebSocket endpoints.

/ws/webcam   client sends binary JPEG frames; server replies with one JSON message
             per frame: {"type": "prediction", ...} | {"type": "no_face", ...} |
             {"type": "error", ...}. Clients should wait for the reply before
             sending the next frame. Frames are decoded in memory and discarded.
/eeg/stream  server pushes a prediction every second. Clients may send JSON
             {"samples": [...]} windows; the latest one is passed to the EEG
             predictor (ignored while it is a mock).
"""
import asyncio
import json
import logging

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

from .. import config
from ..config import CLASSES
from ..errors import AppError
from ..face.pipeline import decode_image
from ..modalities import registry
from ..schemas import ImagePrediction, now_iso

router = APIRouter(tags=["stream"])
log = logging.getLogger(__name__)


def _analyse_frame(data: bytes) -> dict:
    face = registry.get("face")
    rgb = decode_image(data, allowed_formats=("JPEG", "PNG", "WEBP"))
    result = face.analyse(rgb)
    if result is None:
        return {"type": "no_face", "faces_found": 0, "timestamp": now_iso()}
    probs, box, n = result
    pred = ImagePrediction.from_probs("face", probs, faces_found=n, face_box=box, model=face.model_name)
    return {"type": "prediction", **pred.model_dump()}


@router.websocket("/ws/webcam")
async def webcam(ws: WebSocket):
    await ws.accept()
    store = ws.app.state.store
    session_id = None
    sums = np.zeros(len(CLASSES))
    n_pred = 0
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data is None:
                if msg.get("text") and json.loads(msg["text"]).get("type") == "end":
                    break
                continue
            if len(data) > config.MAX_WS_FRAME_BYTES:
                await ws.send_json({"type": "error", "error": "file_too_large",
                                    "message": "Frame is larger than 2 MB."})
                continue
            try:
                reply = await run_in_threadpool(_analyse_frame, data)
            except AppError as e:
                await ws.send_json({"type": "error", "error": e.code, "message": e.message})
                if e.code == "model_unavailable":
                    break
                continue
            if reply["type"] == "prediction":
                if session_id is None:  # create lazily so reconnects don't leave empty sessions
                    session_id = store.create_session("webcam", {"model": reply["model"]})
                pred = ImagePrediction(**{k: v for k, v in reply.items() if k != "type"})
                store.add_prediction(session_id, pred)
                sums += np.asarray(pred.probs)
                n_pred += 1
                reply["session_id"] = session_id
            await ws.send_json(reply)
    except WebSocketDisconnect:
        pass
    finally:
        if session_id is not None:
            mean = sums / max(n_pred, 1)
            k = int(mean.argmax())
            meta = {"model": registry.get("face").model_name, "frames": n_pred,
                    "mean_probs": [float(v) for v in mean]}
            store.finish_session(session_id, CLASSES[k], float(mean[k]), meta)
        try:
            await ws.close()
        except RuntimeError:
            pass  # already closed


@router.websocket("/eeg/stream")
async def eeg_stream(ws: WebSocket):
    await ws.accept()
    predictor = registry.get("eeg")
    latest_window = None
    session_id = None
    store = ws.app.state.store
    try:
        while True:
            try:
                text = await asyncio.wait_for(ws.receive_text(), timeout=1.0)
                try:
                    latest_window = json.loads(text).get("samples")
                except (json.JSONDecodeError, AttributeError):
                    await ws.send_json({"type": "error", "error": "bad_message",
                                        "message": 'Expected JSON like {"samples": [...]}'})
                continue
            except asyncio.TimeoutError:
                pass
            pred = await run_in_threadpool(predictor.predict, latest_window)
            if session_id is None:
                session_id = store.create_session("eeg", mock=pred.mock)
            store.add_prediction(session_id, pred)
            await ws.send_json({"type": "prediction", **pred.model_dump(), "session_id": session_id})
    except WebSocketDisconnect:
        pass
    finally:
        if session_id is not None:
            store.finish_session(session_id)
