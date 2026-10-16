"""Exceptions raised by the inference core, each with a stable error code."""

from __future__ import annotations


class InferenceError(Exception):
    """Base class; ``code`` is a stable machine-readable identifier."""

    code = "inference_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidImageError(InferenceError):
    """The upload is not an acceptable image."""

    code = "invalid_image"


class UnsupportedImageTypeError(InvalidImageError):
    code = "unsupported_image_type"


class ImageTooLargeError(InvalidImageError):
    code = "image_too_large"


class ModelArtifactError(InferenceError):
    """Model files are missing, corrupted or inconsistent."""

    code = "model_unavailable"
