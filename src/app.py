"""Flask app to detect faces in an image and add googly eyes to them."""

import cv2
import numpy as np
from flask import Flask, Response, make_response, request

from src.googly_eyes import get_googly_eyes

ALLOWED_EXTENSIONS = ("png", "jpg", "jpeg")

app = Flask(__name__)


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
