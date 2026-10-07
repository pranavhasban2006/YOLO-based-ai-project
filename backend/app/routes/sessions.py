from fastapi import APIRouter, Query, Request

from ..errors import AppError

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("")
def list_sessions(request: Request, limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)):
    return request.app.state.store.list_sessions(limit, offset)


@router.get("/{session_id}")
def get_session(request: Request, session_id: int):
    s = request.app.state.store.get_session(session_id)
    if s is None:
        raise AppError(404, "session_not_found", f"No session with id {session_id}.")
    return s
