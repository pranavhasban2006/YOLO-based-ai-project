from fastapi import HTTPException


class AppError(HTTPException):
    """HTTP error with a machine-readable code and a human-readable message.

    Serialised as {"error": code, "message": message}.
    """

    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(status_code=status_code, detail={"error": code, "message": message})
        self.code = code
        self.message = message


def no_face() -> AppError:
    return AppError(422, "no_face_detected",
                    "No face was detected. Use a well-lit, front-facing photo with the face clearly visible.")


def unsupported_type(kind: str, allowed) -> AppError:
    return AppError(415, "unsupported_file_type",
                    f"Unsupported {kind} type. Allowed: {', '.join(sorted(allowed))}.")


def too_large(limit_bytes: int) -> AppError:
    return AppError(413, "file_too_large",
                    f"File is larger than the {limit_bytes // (1024 * 1024)} MB limit.")
