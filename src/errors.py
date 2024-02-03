"""Custom exceptions."""


class NoFacesDetectedError(Exception):
    """Exception raised when no faces are detected in the image."""

    message = "No faces detected in the image."

    def __init__(self) -> None:
        super().__init__(self.message)


class NoEyesDetectedError(Exception):
    """Exception raised when no eyes are detected in the image."""

    message = "No eyes detected in the image."

    def __init__(self) -> None:
        super().__init__(self.message)
