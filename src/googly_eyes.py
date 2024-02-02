"""Googlify eyes"""

import logging
import random

import cv2
import numpy as np

from src.model.cascade_classifier import CascadeClassifier

cascade_classifier = CascadeClassifier(
    face_model_path="./data/haarcascade_frontalface_default.xml",
    eyes_model_path="./data/haarcascade_eye.xml",
)


def draw_googly_eyes(image: np.ndarray, eyes: np.ndarray) -> None:
    """Draw googly eyes in an image.

    Args:
        image (np.ndarray): image to draw googly eyes.
        eyes (np.ndarray): eyes coordinates.
    """
    for x, y, w, h in eyes:
        try:
            center = (int(x + w / 2), int(y + h / 2))
            radius = int(w * (0.7 + random.random()))
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
        except Exception as ex:
            logging.error("Error drawing eyes: %s", ex)


def googlify(image: np.ndarray) -> np.ndarray:
    """Detect faces and eyes in an image and draw googly eyes.

    Args:
        image (np.ndarray): image.
    """
    image_copy = image.copy()
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detect_faces(gray_image=gray_image)
    for x, y, w, h in faces:
        roi_gray_image = gray_image[y : y + h, x : x + w]
        roi_image = image_copy[y : y + h, x : x + w]
        eyes = cascade_classifier.detect_eyes(gray_image=roi_gray_image)
        draw_googly_eyes(image=roi_image, eyes=eyes)
    return image_copy
