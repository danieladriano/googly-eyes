"""Flask app to detect faces in an image and add googly eyes to them."""

import logging
import random

import cv2
import numpy as np
from flask import Flask, Response, make_response, request

from model.cascade_classifier import CascadeClassifier

ALLOWED_EXTENSIONS = ("png", "jpg", "jpeg")

app = Flask(__name__)

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


def get_googly_eyes(image: np.ndarray) -> np.ndarray:
    """Detect faces and eyes in an image and draw googly eyes.

    Args:
        image (np.ndarray): image.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detect_faces(gray_image=gray_image)
    for x, y, w, h in faces:
        roi_gray_image = gray_image[y : y + h, x : x + w]
        roi_image = image[y : y + h, x : x + w]
        eyes = cascade_classifier.detect_eyes(gray_image=roi_gray_image)
        draw_googly_eyes(image=roi_image, eyes=eyes)


def allowed_file(filename: str) -> bool:
    """Verify if the file extension is allowed

    Args:
        filename (str): filename

    Returns:
        bool: return true if the file extension is allowed
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/googly_eyes", methods=["POST"])
def googly_eyes():
    """Receive an image and return the same image with googly eyes."""
    try:
        file = request.files["image"]
        app.logger.info("Received image: %s", file.filename)
    except KeyError:
        app.logger.error("Image not present at request.")
        return Response("It's necessary to send a image.", status=500)

    if not allowed_file(file.filename):
        app.logger.error("Not a %s file", ALLOWED_EXTENSIONS)
        return Response(
            response=f"Only {ALLOWED_EXTENSIONS} formarts are allowed!",
            status=415,
        )

    file_bytes = np.fromfile(file, np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    get_googly_eyes(image=image)
    _, buffer = cv2.imencode(".jpg", image)

    response = make_response(buffer.tobytes())
    response.headers["Content-Type"] = "image/jpeg"
    return response
