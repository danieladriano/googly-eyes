"""Googlify eyes"""

import random
from typing import List

import cv2
import numpy as np
from werkzeug.datastructures import FileStorage

from src.config import AppConfig
from src.model.cascade_classifier import CascadeClassifier


class Googlify:
    """Googlify eyes images"""

    def __init__(self, config: AppConfig) -> None:
        self._cascade_classifier = CascadeClassifier(
            face_config=config.face_cascade_classifier,
            eyes_config=config.eyes_cascade_classifier,
        )

    def _draw_googly_eyes(
        self, image: np.ndarray, face: List[np.ndarray], eyes: np.ndarray
    ) -> None:
        for x, y, w, h in eyes:
            x = x + face[0]
            y = y + face[1]
            try:
                center = (int(x + w / 2), int(y + h / 2))
                radius = int(w * (0.4 + random.random()))
            except ArithmeticError:
                center = (x, y)
                radius = w

            cv2.circle(
                img=image,
                center=center,
                radius=radius,
                color=(255, 255, 255),
                thickness=-1,
            )
            cv2.circle(
                img=image,
                center=center,
                radius=int(radius / 2),
                color=(0, 0, 0),
                thickness=-1,
            )

    def _get_face_image(self, face: np.ndarray, image: np.ndarray) -> np.ndarray:
        (x, y, w, h) = face
        return image[y : y + h, x : x + w]

    def _get_eyes(self, face: np.ndarray, gray_image: np.ndarray) -> List[np.ndarray]:
        roi_gray_image = self._get_face_image(face=face, image=gray_image)
        return self._cascade_classifier.detect_eyes(gray_image=roi_gray_image)

    def convert_file_storage_to_image(self, image_file: FileStorage) -> np.ndarray:
        """Convert a file storage to an image.

        Args:
            image_file (FileStorage): image file storage.

        Returns:
            np.ndarray: image.
        """
        image_bytes = np.fromfile(image_file, np.uint8)
        return cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED)

    def googlify(self, image: np.ndarray) -> np.ndarray:
        """Detect faces and eyes in an image and draw googly eyes.

        Args:
            image (np.ndarray): image.
        """
        image_copy = image.copy()
        gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        faces = self._cascade_classifier.detect_faces(gray_image=gray_image)
        for face in faces:
            eyes = self._get_eyes(face=face, gray_image=gray_image)
            self._draw_googly_eyes(image=image_copy, face=face, eyes=eyes)

        _, buffer = cv2.imencode(".jpeg", image_copy)
        return buffer
