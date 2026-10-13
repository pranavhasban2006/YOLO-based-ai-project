"""Model service (FastAPI). Loads the face emotion CNN once at startup and serves predictions.

Run from app/model_service:  uvicorn main:app --port 8001
"""
import asyncio
import base64
import binascii
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from analyzer import Analyzer
from audio_service import AudioAnalyzer, AudioError
import prosody as prosody_mod
import speech_service

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_AUDIO_BYTES = 10 * 1024 * 1024
MAX_FRAMES = 10
analyzer: Analyzer | None = None
audio_analyzer: AudioAnalyzer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer, audio_analyzer
    analyzer = Analyzer(detect_every=int(os.environ.get("DETECT_EVERY", "1")))
    audio_analyzer = AudioAnalyzer()
    yield


app = FastAPI(title="Face emotion model service", lifespan=lifespan)


class FramesIn(BaseModel):
    frames: list[str]  # base64 JPEG or PNG, optionally as a data: URL


def _decode_b64(s: str) -> bytes:
    if s.startswith("data:") and "," in s:
        s = s.split(",", 1)[1]
    try:
        return base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(400, detail={"error": "invalid_image", "message": "frame is not base64"}) from exc


@app.get("/health")
def health():
    return {**analyzer.health(), "audio": audio_analyzer.health(), "stt": {"available": speech_service.available(), "model": speech_service.MODEL_NAME}}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Speech to text (faster-whisper, CPU). The audio is decoded in memory and discarded."""
    data = await file.read(MAX_AUDIO_BYTES + 1)
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, detail={"error": "file_too_large", "message": "audio is larger than 10 MB"})
    if not speech_service.available():
        raise HTTPException(503, detail={"error": "stt_unavailable", "message": "faster-whisper is not installed"})
    try:
        return await asyncio.to_thread(lambda: speech_service.transcribe_wave(speech_service.to_16k(data)))
    except AudioError as exc:
        raise HTTPException(exc.status, detail={"error": exc.code, "message": str(exc)}) from exc


@app.post("/analyse/prosody")
async def analyse_prosody(file: UploadFile = File(...), words: str = Form("")):
    """Heuristic prosody indicators for one answer. `words` is an optional JSON list of {word,start,end}."""
    data = await file.read(MAX_AUDIO_BYTES + 1)
    try:
        ws = json.loads(words) if words else None
        return await asyncio.to_thread(lambda: prosody_mod.extract(speech_service.to_16k(data), ws))
    except AudioError as exc:
        raise HTTPException(exc.status, detail={"error": exc.code, "message": str(exc)}) from exc
    except ValueError as exc:
        raise HTTPException(400, detail={"error": "bad_words", "message": "words must be JSON"}) from exc


@app.post("/analyse/answer")
async def analyse_answer(file: UploadFile = File(...)):
    """Transcript + prosody + voice emotion for one spoken answer, in one decode. Nothing is stored."""
    data = await file.read(MAX_AUDIO_BYTES + 1)
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, detail={"error": "file_too_large", "message": "audio is larger than 10 MB"})
    if not speech_service.available():
        raise HTTPException(503, detail={"error": "stt_unavailable", "message": "faster-whisper is not installed"})
    try:
        return await asyncio.to_thread(speech_service.analyse_answer, data, audio_analyzer)
    except AudioError as exc:
        raise HTTPException(exc.status, detail={"error": exc.code, "message": str(exc)}) from exc


@app.post("/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    data = await file.read(MAX_AUDIO_BYTES + 1)
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, detail={"error": "file_too_large", "message": "audio is larger than 10 MB"})
    try:
        return await asyncio.to_thread(audio_analyzer.predict_bytes, data)
    except AudioError as exc:
        raise HTTPException(exc.status, detail={"error": exc.code, "message": str(exc)}) from exc


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    data = await file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, detail={"error": "file_too_large", "message": "image is larger than 10 MB"})
    try:
        return await asyncio.to_thread(analyzer.predict_image_bytes, data)
    except ValueError as exc:
        raise HTTPException(400, detail={"error": "invalid_image", "message": str(exc)}) from exc


@app.post("/predict/frames")
async def predict_frames(body: FramesIn):
    if not body.frames or len(body.frames) > MAX_FRAMES:
        raise HTTPException(400, detail={"error": "bad_frames",
                                         "message": f"send between 1 and {MAX_FRAMES} frames"})
    images = [_decode_b64(f) for f in body.frames]
    try:
        return await asyncio.to_thread(analyzer.predict_frames, images)
    except ValueError as exc:
        raise HTTPException(400, detail={"error": "invalid_image", "message": str(exc)}) from exc


@app.websocket("/predict/stream")
async def predict_stream(ws: WebSocket):
    """Binary messages are JPEG frames. Each one gets exactly one JSON reply, in order."""
    await ws.accept()
    try:
        while True:
            data = await ws.receive_bytes()
            try:
                result = await asyncio.to_thread(analyzer.predict_jpeg_bytes, data)
            except ValueError as exc:
                result = {"error": "invalid_image", "message": str(exc)}
            await ws.send_json(result)
    except WebSocketDisconnect:
        return
