"""FastAPI entry point. Run from backend/:  python -m uvicorn app.main:app --port 8000"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.formparsers import MultiPartParser

from . import config
from .db import Store
from .errors import AppError, too_large
from .modalities import registry

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

# Keep uploads in memory rather than letting Starlette spool large files to a temp file.
MultiPartParser.spool_max_size = config.MAX_VIDEO_BYTES + 1024 * 1024


def create_app(db_path=None, load_models: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        registry.register_defaults()
        if load_models:
            for p in registry.all_predictors().values():
                p.load()
        app.state.store = Store(db_path or config.DB_PATH)
        face = registry.get("face")
        log.info("Face model: %s (loaded: %s)", face.model_name, face.loaded)
        yield
        app.state.store.close()

    app = FastAPI(title="Multimodal Emotion Detection (research prototype)", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS,
                       allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def reject_huge_bodies(request: Request, call_next):
        # Reject obviously oversized uploads before reading the body.
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > config.MAX_VIDEO_BYTES + 1024 * 1024:
            e = too_large(config.MAX_VIDEO_BYTES)
            return JSONResponse(status_code=413, content=e.detail)
        return await call_next(request)

    @app.exception_handler(AppError)
    async def app_error(_: Request, e: AppError):
        return JSONResponse(status_code=e.status_code, content=e.detail)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, e: RequestValidationError):
        return JSONResponse(status_code=422, content={
            "error": "invalid_request", "message": "The request was not valid.",
            "details": [{"loc": err["loc"], "msg": err["msg"]} for err in e.errors()]})

    @app.exception_handler(Exception)
    async def unexpected(_: Request, e: Exception):
        log.exception("Unhandled error")
        return JSONResponse(status_code=500, content={
            "error": "internal_error", "message": "Unexpected server error. See the backend log."})

    @app.get("/health")
    def health():
        preds = registry.all_predictors()
        face = preds["face"].info
        return {
            "status": "ok" if face["ready"] else "degraded",
            "face": face,
            "modalities": {m: {"mock": p.is_mock} for m, p in preds.items()},
            "mocked": [m for m, p in preds.items() if p.is_mock],
            "classes": config.CLASSES,
        }

    from .routes import predict, sessions, ws
    app.include_router(predict.router)
    app.include_router(sessions.router)
    app.include_router(ws.router)
    return app


app = create_app()
