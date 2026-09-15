"""Public API error types for the S4.6 error-handling boundary."""

from __future__ import annotations


class APIError(Exception):
    """Base class for expected, user-facing API failures."""

    code = "API_ERROR"
    status_code = 500
    default_message = "The request could not be completed."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.default_message)
        self.message = message or self.default_message


class MissingFileError(APIError):
    """Raised when the multipart image field is missing."""

    code = "MISSING_FILE"
    status_code = 400
    default_message = "An image file is required."


class InvalidImageError(APIError):
    """Raised when uploaded content is not a valid supported image."""

    code = "INVALID_IMAGE"
    status_code = 400
    default_message = "The uploaded file is not a supported image."


class FileTooLargeError(APIError):
    """Raised when an uploaded file exceeds the configured byte limit."""

    code = "FILE_TOO_LARGE"
    status_code = 413
    default_message = "The uploaded file is too large."


class ModelUnavailableError(APIError):
    """Raised when the inference model cannot be loaded or used."""

    code = "MODEL_UNAVAILABLE"
    status_code = 503
    default_message = "The search model is currently unavailable."


class GalleryUnavailableError(APIError):
    """Raised when the retrieval gallery cannot be loaded or used."""

    code = "GALLERY_UNAVAILABLE"
    status_code = 503
    default_message = "The search gallery is currently unavailable."


class NoResultsError(APIError):
    """Raised when a valid query produces no searchable products."""

    code = "NO_RESULTS"
    status_code = 404
    default_message = "No similar products were found."


class InternalAPIError(APIError):
    """Safe public wrapper for unexpected server failures."""

    code = "INTERNAL_ERROR"
    status_code = 500
    default_message = "An unexpected error occurred while processing the request."
