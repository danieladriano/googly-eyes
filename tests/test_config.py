"""Test App config"""

from src.config import AppConfig


def test_app_config():
    """Test load app config from yaml file."""
    config = AppConfig.load("./tests/resources/config.yml")
    assert config == AppConfig(
        face_model_path="haarcascade_frontalface_default.xml",
        eyes_model_path="haarcascade_eye.xml",
    )
