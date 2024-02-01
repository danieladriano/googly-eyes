"""Cascade classifier to detect faces and eyes."""

from pathlib import Path

import cv2
import numpy as np


class CascadeClassifier:
    """Cascade classifier to detect faces and eyes."""

    def __init__(self, face_model_path: Path, eyes_model_path: Path) -> None:
        self._face_cascade = cv2.CascadeClassifier(str(face_model_path))
        self._eyes_cascade = cv2.CascadeClassifier(str(eyes_model_path))

    def detect_faces(self, gray_image: np.ndarray) -> np.ndarray:
        """Detect faces in a gray image.

        Args:
            gray_image (np.ndarray): A gray image to detect faces.

        Returns:
            np.ndarray: An array of faces (x, y, w, h)
        """
        return self._face_cascade.detectMultiScale(
            image=gray_image, scaleFactor=1.3, minNeighbors=5
        )

    def detect_eyes(self, gray_image: np.ndarray) -> np.ndarray:
        """Detect eyes in a gray image, then return the two biggest eyes.

        Args:
            gray_image (np.ndarray): A gray image to detect eyes.

        Returns:
            np.ndarray: An array of eyes (x, y, w, h)
        """
        eyes = self._eyes_cascade.detectMultiScale(
            image=gray_image, scaleFactor=1.2, minNeighbors=5
        )

        eyes = sorted(eyes, key=lambda x: x[2], reverse=True)
        return eyes[:2]
