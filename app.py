"""Flask app to detect faces in an image and add googly eyes to them."""

from flask import Flask, Response, make_response, request

from src.config import AppConfig
from src.googly_eyes import Googlify

ALLOWED_EXTENSIONS = ("png", "jpg", "jpeg")

app = Flask(__name__)
googlify = Googlify(config=AppConfig.load())


@app.route("/googly-eyes", methods=["POST"])
def googly_eyes():
    """Receive an image and return the same image with googly eyes."""
    if "image" not in request.files:
        app.logger.error("Image not present at request.")
        return Response("It's necessary to send a image.", status=500)

    file = request.files["image"]
    app.logger.info("Received image: %s", file.filename)

    if not file.content_type in ("image/png", "image/jpeg"):
        app.logger.error("Not a %s file", ALLOWED_EXTENSIONS)
        return Response(
            response=f"Only {ALLOWED_EXTENSIONS} formarts are allowed!",
            status=415,
        )

    image = googlify.convert_file_storage_to_image(image_file=file)
    buffer = googlify.googlify(image=image)
    response = make_response(buffer.tobytes())
    response.headers["Content-Type"] = "image/jpeg"
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0")
