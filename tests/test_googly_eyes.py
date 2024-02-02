from unittest.mock import patch

import numpy as np

from src.googly_eyes import Googlify


def test_get_face_image(mock_googlify: Googlify) -> None:
    """Test Googlify class."""
    face = np.array([10, 20, 30, 40])
    image = np.random.randint(100, size=(100, 100))
    result = mock_googlify._get_face_image(face, image)

    assert np.array_equal(result, image[20:60, 10:40])


def test_get_eyes(mock_googlify: Googlify) -> None:
    """Test Googlify class."""
    face = np.array([10, 20, 30, 40])
    gray_image = np.random.randint(100, size=(100, 100))
    result = mock_googlify._get_eyes(face, gray_image)

    assert len(result) == 2
    assert all(len(eye) == 4 for eye in result)


def test_draw_googly_eyes(mock_googlify: Googlify) -> None:
    """Test Googlify class."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :] = (255, 0, 0)
    eyes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])

    mock_googlify._draw_googly_eyes(image, eyes)

    assert image.shape == (100, 100, 3)


def test_googlify(mock_googlify: Googlify) -> None:
    """Test Googlify class."""
    image = np.zeros((255, 255, 3), dtype=np.uint8)
    with patch(
        "src.googly_eyes.CascadeClassifier.detect_faces",
        return_value=np.array([[10, 20, 50, 60]]),
    ):
        result = mock_googlify.googlify(image)

    assert result.shape == (1651,)
