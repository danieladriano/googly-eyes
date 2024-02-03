from unittest.mock import patch

import cv2
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


def test_transform_eyes_to_face_coordinates(mock_googlify: Googlify) -> None:
    """Test _transform_eyes_to_face_coordinates method."""
    face = np.array([10, 20, 30, 40])
    eyes = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    expected_result = [(11, 22, 3, 4), (15, 26, 7, 8)]

    result = mock_googlify._transform_eyes_to_face_coordinates(face, eyes)

    assert result == expected_result


def test_draw_eyeball(mock_googlify: Googlify) -> None:
    """Test Googlify class."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    eye = np.array([10, 20, 30, 40])

    mock_googlify._draw_eyeball(image=image, eye=eye)

    assert image.shape == (100, 100, 3)
    assert np.array_equal(image[20:60, 10:40], np.full((40, 30, 3), 255))


def test_draw_pupil(mock_googlify: Googlify) -> None:
    """Test Googlify class."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :] = (255, 0, 0)
    blue_pixels = image.shape[0] * image.shape[1]
    eye = np.array([10, 20, 30, 40])

    mock_googlify._draw_pupil(image=image, eye=eye)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    assert cv2.countNonZero(image_gray) < blue_pixels


def test_googlify(mock_googlify: Googlify) -> None:
    """Test Googlify class."""
    image = np.zeros((255, 255, 3), dtype=np.uint8)
    with patch(
        "src.googly_eyes.CascadeClassifier.detect_faces",
        return_value=np.array([[10, 20, 50, 60]]),
    ):
        result = mock_googlify.googlify(image)

    assert result.shape == (1651,)
