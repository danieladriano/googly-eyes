"""Flask app to detect faces in an image and add googly eyes to them."""

import logging
import random

import cv2
import numpy as np
from flask import Flask, make_response, request

from model.cascade_classifier import CascadeClassifier

app = Flask(__name__)

cascade_classifier = CascadeClassifier(
    face_model_path="./data/haarcascade_frontalface_default.xml",
    eyes_model_path="./data/haarcascade_eye.xml",
)


def draw_googly_eyes(image: np.ndarray, eyes: np.ndarray) -> np.ndarray:
    """Draw googly eyes in an image.

    Args:
        image (np.ndarray): image to draw googly eyes.
        eyes (np.ndarray): eyes coordinates.

    Returns:
        np.ndarray: image with googly eyes.
    """
    image_h = np.size(image, 0)
    for x, y, w, h in eyes:
        try:
            if y + h > image_h / 2:
                pass
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


def get_image_with_faces(image: np.ndarray) -> np.ndarray:
    """Return the input image with faces detected.

    Args:
        image (np.ndarray): image to detect faces.

    Returns:
        np.ndarray: image with faces detected.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detect_faces(gray_image=gray_image)
    for x, y, w, h in faces:
        roi_gray_image = gray_image[y : y + h, x : x + w]
        roi_image = image[y : y + h, x : x + w]
        eyes = cascade_classifier.detect_eyes(gray_image=roi_gray_image)
        draw_googly_eyes(image=roi_image, eyes=eyes)
    return image


@app.route("/googly_eyes", methods=["POST"])
def googly_eyes():
    """Receive an image and return the same image with googly eyes."""
    file = request.files["file"]

    file_bytes = np.fromfile(file, np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

    image_with_faces = get_image_with_faces(image=image)
    _, buffer = cv2.imencode(".jpg", image_with_faces)

    response = make_response(buffer.tobytes())
    response.headers["Content-Type"] = "image/jpeg"
    return response
