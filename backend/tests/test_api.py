import io
from datetime import datetime

import av
import numpy as np
import pytest

from app import config
from app.config import CLASSES


def assert_prediction_shape(body, modality, mock):
    assert body["modality"] == modality
    assert len(body["probs"]) == 7
    assert abs(sum(body["probs"]) - 1) < 1e-4
    assert body["label"] in CLASSES
    assert body["label"] == CLASSES[int(np.argmax(body["probs"]))]
    assert body["confidence"] == pytest.approx(max(body["probs"]))
    assert body["mock"] is mock
    datetime.fromisoformat(body["timestamp"])


# ---- image

def test_image_with_face(client, face_jpeg):
    r = client.post("/predict/image", files={"file": ("face.jpg", face_jpeg, "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert_prediction_shape(body, "face", mock=False)
    assert body["faces_found"] >= 1
    box = body["face_box"]
    assert 0 <= box["x"] < 1 and 0 <= box["y"] < 1 and 0 < box["w"] <= 1 and 0 < box["h"] <= 1
    # The astronaut's face is in the upper-middle of the photo.
    assert 0.3 < box["x"] + box["w"] / 2 < 0.7 and box["y"] + box["h"] / 2 < 0.5


def test_image_without_face(client, no_face_jpeg):
    r = client.post("/predict/image", files={"file": ("scene.jpg", no_face_jpeg, "image/jpeg")})
    assert r.status_code == 422
    assert r.json()["error"] == "no_face_detected"


def test_unsupported_file_type(client):
    r = client.post("/predict/image", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415
    assert r.json()["error"] == "unsupported_file_type"


def test_corrupt_image(client):
    r = client.post("/predict/image", files={"file": ("x.jpg", b"not really a jpeg", "image/jpeg")})
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_image"


def test_oversized_upload(client, face_jpeg, monkeypatch):
    monkeypatch.setattr(config, "MAX_IMAGE_BYTES", 1000)
    r = client.post("/predict/image", files={"file": ("face.jpg", face_jpeg, "image/jpeg")})
    assert r.status_code == 413
    assert r.json()["error"] == "file_too_large"


# ---- video

def make_video(frames, fps=10) -> bytes:
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as out:
        s = out.add_stream("mpeg4", rate=fps)
        s.height, s.width = frames[0].shape[:2]
        s.pix_fmt = "yuv420p"
        for f in frames:
            for pkt in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                out.mux(pkt)
        for pkt in s.encode():
            out.mux(pkt)
    return buf.getvalue()


def test_video_timeline(client, face_rgb):
    blank = np.full_like(face_rgb, 90)
    # 2 s with a face, then 1 s without, at 10 fps -> ~15 samples at 5 fps.
    clip = make_video([face_rgb] * 20 + [blank] * 10)
    r = client.post("/predict/video", files={"file": ("clip.mp4", clip, "video/mp4")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert_prediction_shape(body, "face", mock=False)
    assert body["frames_sampled"] == len(body["timeline"])
    assert 13 <= body["frames_sampled"] <= 16
    assert body["faces_found"] >= 8
    assert body["frames_without_face"] >= 4
    assert body["faces_found"] + body["frames_without_face"] == body["frames_sampled"]
    ts = [p["t"] for p in body["timeline"]]
    assert ts == sorted(ts) and all(b - a > 0.15 for a, b in zip(ts, ts[1:]))
    with_face = [p for p in body["timeline"] if p["face"]]
    assert all(len(p["probs"]) == 7 for p in with_face)


def test_corrupt_video(client):
    r = client.post("/predict/video", files={"file": ("clip.mp4", b"\x00garbage" * 500, "video/mp4")})
    assert r.status_code == 422
    assert r.json()["error"] == "corrupt_video"


# ---- stubs and fusion

def test_stub_endpoints_are_labelled_mock(client):
    r = client.post("/predict/text", json={"text": "I feel fine"})
    assert r.status_code == 200
    assert_prediction_shape(r.json(), "text", mock=True)
    r = client.post("/predict/audio", files={"file": ("a.wav", b"RIFF....WAVE", "audio/wav")})
    assert r.status_code == 200
    assert_prediction_shape(r.json(), "audio", mock=True)
    with client.websocket_connect("/eeg/stream") as ws:
        msg = ws.receive_json()
    assert msg["type"] == "prediction" and msg["mock"] is True and msg["modality"] == "eeg"


def test_fused_ignores_mocked_modalities(client, face_jpeg):
    face = client.post("/predict/image", files={"file": ("f.jpg", face_jpeg, "image/jpeg")}).json()
    audio = client.post("/predict/audio", files={"file": ("a.wav", b"RIFF", "audio/wav")}).json()
    text = client.post("/predict/text", json={"text": "hello"}).json()
    # A client claiming mock=false for a modality the backend still mocks must also be ignored.
    text_spoofed = {**text, "mock": False}
    r = client.post("/predict/fused", json={"predictions": [face, audio, text_spoofed]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert_prediction_shape(body, "fused", mock=False)
    assert body["contributing"] == ["face"]
    assert body["probs"] == pytest.approx(face["probs"], abs=1e-6)
    reasons = {s["modality"]: s["reason"] for s in body["skipped"]}
    assert reasons == {"audio": "mock", "text": "mock", "eeg": "not_supplied"}


def test_fused_weights_by_confidence(client):
    from app.fusion import fuse
    from app.modalities import registry
    from app.modalities.base import Predictor
    from app.schemas import Prediction

    class RealText(Predictor):
        modality = "text"
        def predict(self, data):  # pragma: no cover
            raise NotImplementedError

    registry.register(RealText())
    a = Prediction.from_probs("face", [0.9, 0.1, 0, 0, 0, 0, 0])
    b = Prediction.from_probs("text", [0, 0.5, 0.5, 0, 0, 0, 0])
    out = fuse([a, b])
    expected = (0.9 * np.array(a.probs) + 0.5 * np.array(b.probs)) / 1.4
    assert out.probs == pytest.approx(expected.tolist())
    assert set(out.contributing) == {"face", "text"}


def test_fused_with_only_mocks_is_an_error(client):
    text = client.post("/predict/text", json={"text": "hello"}).json()
    r = client.post("/predict/fused", json={"predictions": [text]})
    assert r.status_code == 422
    assert r.json()["error"] == "no_real_modalities"


# ---- webcam websocket and sessions

def test_webcam_ws_and_sessions(client, face_jpeg, no_face_jpeg):
    with client.websocket_connect("/ws/webcam") as ws:
        ws.send_bytes(face_jpeg)
        msg = ws.receive_json()
        assert msg["type"] == "prediction"
        assert_prediction_shape(msg, "face", mock=False)
        assert msg["face_box"] is not None
        ws.send_bytes(no_face_jpeg)
        assert ws.receive_json()["type"] == "no_face"
        ws.send_bytes(b"not an image")
        err = ws.receive_json()
        assert err["type"] == "error" and err["error"] == "invalid_image"
        ws.send_text('{"type": "end"}')
    sid = msg["session_id"]

    sessions = client.get("/sessions").json()
    assert any(s["id"] == sid and s["kind"] == "webcam" for s in sessions)
    detail = client.get(f"/sessions/{sid}").json()
    assert detail["kind"] == "webcam" and len(detail["predictions"]) == 1
    assert detail["ended_at"] is not None
    assert client.get("/sessions/999999").status_code == 404
