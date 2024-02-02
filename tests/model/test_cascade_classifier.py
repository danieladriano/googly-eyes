from pathlib import Path
from unittest.mock import patch

import numpy as np
from pytest import fixture

from model.cascade_classifier import CascadeClassifier


class MockCascadeClassifier:
    def detectMultiScale(self, image, scaleFactor, minNeighbors) -> np.ndarray:
        return np.zeros((3, 4), dtype=np.int32)


@fixture
def mock_cascade_classifier() -> CascadeClassifier:
    with patch(
        "model.cascade_classifier.cv2.CascadeClassifier",
        return_value=MockCascadeClassifier(),
    ):
        return CascadeClassifier(face_model_path=Path(), eyes_model_path=Path())


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
