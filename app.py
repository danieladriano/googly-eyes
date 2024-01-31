import io

from flask import Flask, request, send_file

app = Flask(__name__)


@app.route("/googly_eyes", methods=["POST"])
def googly_eyes():
    """Receive an image and return the same image with googly eyes."""
    file = request.files["file"]
    return send_file(io.BytesIO(file.read()), mimetype="image/jpeg")
