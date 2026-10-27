from fastapi import APIRouter, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from .. import config
from ..errors import no_face
from ..face.pipeline import decode_image
from ..fusion import fuse
from ..modalities import registry
from ..errors import AppError
from ..modalities.audio_cascade import cascade as audio_cascade
from ..schemas import (CascadePrediction, FusedPrediction, FusedRequest, ImagePrediction, Prediction,
                       TextRequest, VideoPrediction)
from ..video import predict_video
from .uploads import read_upload

router = APIRouter(prefix="/predict", tags=["predict"])

AUDIO_TYPES = {"audio/wav", "audio/x-wav", "audio/wave", "audio/webm", "audio/ogg",
               "audio/mpeg", "audio/mp4", "audio/x-m4a", "video/mp4"}   # .mp4: the audio track is used
AUDIO_EXTS = {".wav", ".webm", ".ogg", ".mp3", ".m4a", ".mp4"}


def _image_prediction(rgb) -> ImagePrediction:
    face = registry.get("face")
    result = face.analyse(rgb)
    if result is None:
        raise no_face()
    probs, box, n = result
    return ImagePrediction.from_probs("face", probs, faces_found=n, face_box=box, model=face.model_name)


@router.post("/image", response_model=ImagePrediction)
async def predict_image(request: Request, file: UploadFile = File(...)):
    data = await read_upload(file, "image", config.IMAGE_TYPES, config.IMAGE_EXTS, config.MAX_IMAGE_BYTES)
    rgb = decode_image(data)
    pred = await run_in_threadpool(_image_prediction, rgb)
    pred.session_id = request.app.state.store.record(
        "image", pred, {"model": pred.model, "faces_found": pred.faces_found, "file": file.filename})
    return pred


@router.post("/video", response_model=VideoPrediction)
async def predict_video_route(request: Request, file: UploadFile = File(...)):
    data = await read_upload(file, "video", config.VIDEO_TYPES, config.VIDEO_EXTS, config.MAX_VIDEO_BYTES)
    pred = await run_in_threadpool(predict_video, registry.get("face"), data)
    meta = {"model": pred.model, "file": file.filename, "duration": pred.duration,
            "faces_found": pred.faces_found, "frames_without_face": pred.frames_without_face,
            "timeline": [p.model_dump() for p in pred.timeline]}
    pred.session_id = request.app.state.store.record("video", pred, meta)
    return pred


@router.post("/audio", response_model=Prediction)
async def predict_audio(request: Request, file: UploadFile = File(...)):
    data = await read_upload(file, "audio", AUDIO_TYPES, AUDIO_EXTS, config.MAX_AUDIO_BYTES)
    pred = await run_in_threadpool(registry.get("audio").predict, data)
    request.app.state.store.record("audio", pred, {"file": file.filename})
    return pred


@router.post("/audio/cascade", response_model=CascadePrediction)
async def predict_audio_cascade(request: Request, file: UploadFile = File(...)):
    """Speech CNN (A1) -> GRU (A2): returns A1's answer and the final answer after A2."""
    if not audio_cascade.available:
        raise AppError(503, "model_unavailable", "The CNN -> GRU checkpoints are not in models/cascade_live/models.")
    data = await read_upload(file, "audio", AUDIO_TYPES, AUDIO_EXTS, config.MAX_AUDIO_BYTES)
    out = await run_in_threadpool(audio_cascade.predict, data)
    a1, final = out["a1"], out["final"]
    pred = CascadePrediction(**final.model_dump(), a1=a1, changed=a1.label != final.label,
                             model=audio_cascade.model_name)
    pred.session_id = request.app.state.store.record("audio", pred, {"file": file.filename, "model": pred.model})
    return pred


@router.post("/text", response_model=Prediction)
async def predict_text(request: Request, body: TextRequest):
    pred = await run_in_threadpool(registry.get("text").predict, body.text)
    # Store the result only, not the user's text.
    request.app.state.store.record("text", pred, {"chars": len(body.text)})
    return pred


@router.post("/fused", response_model=FusedPrediction)
async def predict_fused(request: Request, body: FusedRequest):
    pred = fuse(body.predictions)
    pred.session_id = request.app.state.store.record(
        "fused", pred, {"contributing": pred.contributing, "skipped": pred.skipped, "weights": pred.weights})
    return pred
