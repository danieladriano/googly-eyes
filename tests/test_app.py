"""Test googly eyes app"""

import io

from flask.testing import FlaskClient


def test_image_post_key_not_present(client: FlaskClient) -> None:
    """Test if image key is not present on post request

    Args:
        client (FlaskClient): The flask test client.
    """
    file_name = "fake_file.txt"
    data = {"fake": (io.BytesIO(b"Some text"), file_name)}
    response = client.post("/googly_eyes", data=data)
    assert response.status_code == 500


def test_upload_text_file(client: FlaskClient) -> None:
    """
    Test case for uploading a text file.

    Args:
        client (FlaskClient): The Flask test client.
    """

    file_name = "fake_file.txt"
    data = {"image": (io.BytesIO(b"Some text"), file_name)}
    response = client.post("/googly_eyes", data=data)
    assert response.status_code == 415


def test_upload_image(client: FlaskClient) -> None:
    """Test case for uploading a jpg image

    Args:
        client (FlaskClient): The flask test client
    """
    file_name = "photo.jpg"
    data = {
        "image": (
            open("./tests/resources/photo.jpg", "rb"),
            file_name,
        )
    }
    response = client.post("/googly_eyes", data=data)
    assert response.status_code == 200
