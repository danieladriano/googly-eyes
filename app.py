"""Flask app to detect faces in an image and add googly eyes to them."""

import cv2
import numpy as np
from flask import Flask, make_response, request

app = Flask(__name__)
face_cascade = cv2.CascadeClassifier("./model/haarcascade_frontalface_default.xml")


def detect_faces(gray_image: np.ndarray) -> np.ndarray:
    """Detect faces in a gray image.

    Args:
        gray_image (np.ndarray): A gray image to detect faces.

    Returns:
        np.ndarray: An array of faces (x, y, w, h)
    """
    return face_cascade.detectMultiScale(
        image=gray_image, scaleFactor=1.3, minNeighbors=5
    )


def get_image_with_faces(image: np.ndarray) -> np.ndarray:
    """Return the input image with faces detected.

    Args:
        image (np.ndarray): image to detect faces.

    Returns:
        np.ndarray: image with faces detected.
    """
    faces = detect_faces(gray_image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    for x, y, w, h in faces:
        cv2.rectangle(
            img=image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=3
        )
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
