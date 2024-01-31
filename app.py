import io

import cv2
import numpy as np
from flask import Flask, make_response, request

app = Flask(__name__)


@app.route("/googly_eyes", methods=["POST"])
def googly_eyes():
    """Receive an image and return the same image with googly eyes."""
    file = request.files["file"]

    file_bytes = np.fromfile(file, np.uint8)
    gray_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    _, buffer = cv2.imencode(".jpg", gray_image)

    response = make_response(buffer.tobytes())
    response.headers["Content-Type"] = "image/jpeg"
    return response
