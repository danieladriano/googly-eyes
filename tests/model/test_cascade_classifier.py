"""Tests for the CascadeClassifier class."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
from pytest import fixture

from src.config import CascadeClassifierConfig
from src.model.cascade_classifier import CascadeClassifier


class MockCascadeClassifier:
    def detectMultiScale(self, image, scaleFactor, minNeighbors) -> np.ndarray:
        return np.zeros((3, 4), dtype=np.int32)


@fixture
def mock_cascade_classifier() -> CascadeClassifier:
    with patch(
        "src.model.cascade_classifier.cv2.CascadeClassifier",
        return_value=MockCascadeClassifier(),
    ):
        face_config = CascadeClassifierConfig(
            path="haarcascade_frontalface_default.xml",
            scale_factor=1.3,
            min_neighbors=5,
        )
        eyes_config = CascadeClassifierConfig(
            path="haarcascade_eye.xml",
            scale_factor=1.2,
            min_neighbors=5,
        )
        return CascadeClassifier(face_config=face_config, eyes_config=eyes_config)


def test_detect_eyes(mock_cascade_classifier: CascadeClassifier) -> None:
    """
    Test the detect_eyes method of the CascadeClassifier class.

    Args:
        mock_cascade_classifier (CascadeClassifier): The mock cascade classifier object.
    """
    gray_image = np.zeros((100, 100), dtype=np.uint8)
    eyes = mock_cascade_classifier.detect_eyes(gray_image=gray_image)
    assert isinstance(eyes, list)
    assert isinstance(eyes[0], np.ndarray)
    assert len(eyes) == 2
    assert all(len(eye) == 4 for eye in eyes)


def test_detect_faces(mock_cascade_classifier: CascadeClassifier) -> None:
    """
    Test the detect_faces method of the CascadeClassifier class.

    Args:
        mock_cascade_classifier (CascadeClassifier): The mock cascade classifier object.
    """
    gray_image = np.zeros((100, 100), dtype=np.uint8)
    eyes = mock_cascade_classifier.detect_faces(gray_image=gray_image)
    assert isinstance(eyes, np.ndarray)
    assert isinstance(eyes[0], np.ndarray)
    assert len(eyes) == 3
    assert all(len(eye) == 4 for eye in eyes)
