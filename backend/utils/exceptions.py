"""Domain exceptions. Each maps to one of the app's edge cases."""


class CropClassificationError(Exception):
    """Base class for all handled pipeline errors."""


class InsufficientObservationsError(CropClassificationError):
    """Raised when GEE returns too few NDVI points to build a lifecycle."""

    def __init__(self, message: str = "Not enough NDVI observations."):
        super().__init__(message)


class GEEExtractionError(CropClassificationError):
    """Raised when Earth Engine cannot complete NDVI extraction."""

    def __init__(
        self,
        message: str = "Earth Engine could not extract NDVI for this request. Try a smaller date window or another point.",
    ):
        super().__init__(message)


class InvalidLocationError(CropClassificationError):
    """Raised when the queried point does not look like agricultural land."""

    def __init__(
        self,
        message: str = "Location does not appear to be agricultural land.",
    ):
        super().__init__(message)


class NoLifecycleFoundError(CropClassificationError):
    """Raised when no candidate lifecycle passes validation for the query."""

    def __init__(self, message: str = "No valid crop lifecycle could be extracted for this location and date."):
        super().__init__(message)


class ModelArtifactNotFoundError(CropClassificationError):
    """Raised when a configured model/label-encoder file is missing on disk."""
