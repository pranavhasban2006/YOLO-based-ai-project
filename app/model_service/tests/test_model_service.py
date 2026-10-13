"""Tests for the model service, run on real images from datasets/ (FER-2013 test split).

Run from app/model_service:  ..\\..\\..\\project\\.venv\\Scripts\\python.exe -m pytest -s
Assertions check shapes, probability sums and class order. Accuracy is printed, never asserted,
so a wrong prediction makes a test print a miss, not fail.
"""
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

ROOT = HERE.parents[2]  # tests -> model_service -> app -> repo root
FER_TEST = ROOT / "datasets" / "FER-2013" / "extracted" / "test"
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# One real image per emotion, first file in each folder of the official FER-2013 test split.
SAMPLES = {
    "happy": FER_TEST / "happy" / "PrivateTest_47631580.jpg",
    "angry": FER_TEST / "angry" / "PrivateTest_10131363.jpg",
    "neutral": FER_TEST / "neutral" / "PrivateTest_10086748.jpg",
}


@pytest.fixture(scope="module")
def client():
    import main
    with TestClient(main.app) as c:
        yield c


def test_sample_files_exist():
    for label, path in SAMPLES.items():
        assert path.is_file(), f"missing sample for {label}: {path}"


def test_health_reports_class_order(client):
    body = client.get("/health").json()
    assert body["classes"] == CLASSES
    assert body["status"] == "ok"
    assert body["device"] == "cpu"
    assert body["model"].endswith(".pt")


@pytest.mark.parametrize("label", list(SAMPLES))
def test_predict_image_shape_and_sum(client, label):
    with open(SAMPLES[label], "rb") as f:
        r = client.post("/predict/image", files={"file": (SAMPLES[label].name, f, "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) >= {"face_found", "box", "probabilities", "top_label", "ms", "model"}
    if body["face_found"]:
        probs = body["probabilities"]
        assert list(probs) == CLASSES, "class order must match classes.json"
        assert abs(sum(probs.values()) - 1.0) < 1e-2
        assert body["top_label"] == max(probs, key=probs.get)
        b = body["box"]
        assert 0 <= b["x"] <= 1 and 0 <= b["y"] <= 1 and 0 < b["w"] <= 1 and 0 < b["h"] <= 1
    else:
        assert body["probabilities"] is None and body["top_label"] is None


def test_no_face_is_not_guessed(client):
    import cv2
    import numpy as np
    blank = np.full((240, 320, 3), 127, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", blank)
    r = client.post("/predict/image", files={"file": ("blank.jpg", buf.tobytes(), "image/jpeg")})
    body = r.json()
    assert body["face_found"] is False
    assert body["probabilities"] is None


def test_invalid_image_is_400(client):
    r = client.post("/predict/image", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "invalid_image"


def test_frames_endpoint_averages(client):
    import base64
    frames = []
    for label in SAMPLES:
        frames.append(base64.b64encode(SAMPLES[label].read_bytes()).decode())
    r = client.post("/predict/frames", json={"frames": frames})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["frames_total"] == 3
    if body["face_found"]:
        assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-2


def test_websocket_stream_returns_one_result_per_frame(client):
    import cv2
    img = cv2.imread(str(SAMPLES["happy"]))
    ok, buf = cv2.imencode(".jpg", img)
    with client.websocket_connect("/predict/stream") as ws:
        for _ in range(3):
            ws.send_bytes(buf.tobytes())
            msg = ws.receive_json()
            assert "face_found" in msg and "ms" in msg


def test_accuracy_report(client, capsys):
    """Prints how many of the three sample predictions match their folder label (no assertion)."""
    correct, lines = 0, []
    for label, path in SAMPLES.items():
        with open(path, "rb") as f:
            body = client.post("/predict/image", files={"file": (path.name, f, "image/jpeg")}).json()
        pred = body.get("top_label") or "no face"
        correct += int(pred == label)
        lines.append({"file": path.name, "true": label, "pred": pred,
                      "top_prob": body.get("confidence")})
    print("\nACCURACY_REPORT", json.dumps({"correct": correct, "total": len(SAMPLES), "rows": lines}))


# --- speech emotion (/predict/audio) ---
AUDIO_DIR = ROOT / "project" / "data" / "audio_16k"
AUDIO_SAMPLES = {"happy": AUDIO_DIR / "tess" / "OAF_bar_happy.wav",
                 "happy_crema": AUDIO_DIR / "crema_d" / "1002_DFA_HAP_XX.wav"}


@pytest.mark.parametrize("name", list(AUDIO_SAMPLES))
def test_predict_audio_shape_and_sum(client, name):
    with open(AUDIO_SAMPLES[name], "rb") as f:
        r = client.post("/predict/audio", files={"file": ("a.wav", f, "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert list(body["probabilities"]) == CLASSES
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-4
    assert body["top_label"] in CLASSES
    print(f"audio {name}: true=happy pred={body['top_label']} conf={body['confidence']:.2f}")


def test_invalid_audio_is_400(client):
    r = client.post("/predict/audio", files={"file": ("a.wav", b"not audio", "audio/wav")})
    assert r.status_code in (400, 415)
    assert r.json()["detail"]["error"] in ("invalid_audio", "ffmpeg_missing")


def test_silent_audio_is_rejected(client):
    import io
    import numpy as np
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000, dtype="float32"), 16000, format="WAV")
    r = client.post("/predict/audio", files={"file": ("s.wav", buf.getvalue(), "audio/wav")})
    assert r.status_code == 400 and r.json()["detail"]["error"] == "silent_audio"
