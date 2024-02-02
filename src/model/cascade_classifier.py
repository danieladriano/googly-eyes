"""Cascade classifier to detect faces and eyes."""

from typing import List

import cv2
import numpy as np

from src.config import CascadeClassifierConfig


class CascadeClassifier:
    """Cascade classifier to detect faces and eyes."""

    def __init__(
        self,
        face_config: CascadeClassifierConfig,
        eyes_config: CascadeClassifierConfig,
    ) -> None:
        self._face_config = face_config
        self._eyes_config = eyes_config
        self._face_cascade = cv2.CascadeClassifier(self._face_config.path)
        self._eyes_cascade = cv2.CascadeClassifier(self._eyes_config.path)

    def detect_faces(self, gray_image: np.ndarray) -> np.ndarray:
        """Detect faces in a gray image.

        Args:
            gray_image (np.ndarray): A gray image to detect faces.

        Returns:
            np.ndarray: An array of faces (x, y, w, h)
        """
        return self._face_cascade.detectMultiScale(
            image=gray_image,
            scaleFactor=self._face_config.scale_factor,
            minNeighbors=self._face_config.min_neighbors,
        )

    def detect_eyes(self, gray_image: np.ndarray) -> List[np.ndarray]:
        """Apply a blur filter on gray image, detect eyes,
        and then return the two biggest eyes.

        Args:
            gray_image (np.ndarray): A gray image to detect eyes.

        Returns:
            np.ndarray: An array of eyes (x, y, w, h)
        """
        blur_gray_image = cv2.blur(gray_image, ksize=(3, 3))
        eyes = self._eyes_cascade.detectMultiScale(
            image=blur_gray_image,
            scaleFactor=self._eyes_config.scale_factor,
            minNeighbors=self._eyes_config.min_neighbors,
        )

        eyes = sorted(eyes, key=lambda x: x[2], reverse=True)
        return eyes[:2]
