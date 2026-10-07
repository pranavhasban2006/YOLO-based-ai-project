from pathlib import Path

from fastapi import UploadFile

from ..errors import AppError, too_large, unsupported_type


async def read_upload(file: UploadFile, kind: str, types: set[str], exts: set[str], limit: int) -> bytes:
    """Validate type and size, then read the upload into memory.

    Accepts the file if either its MIME type or its extension is allowed, since
    browsers sometimes send application/octet-stream. Content is verified again
    when decoded.
    """
    ext = Path(file.filename or "").suffix.lower()
    if file.content_type not in types and ext not in exts:
        raise unsupported_type(kind, exts)
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise too_large(limit)
    if not data:
        raise AppError(400, "empty_file", "The uploaded file is empty.")
    return data
