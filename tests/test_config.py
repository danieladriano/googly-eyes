"""Test App config"""

from src.config import AppConfig


def test_app_config():
    """Test load app config from yaml file."""
    config = AppConfig.load("./tests/resources/config.yml")
    assert config.face_cascade_classifier.path == "haarcascade_frontalface_default.xml"
    assert config.face_cascade_classifier.scale_factor == 1.3
    assert config.face_cascade_classifier.min_neighbors == 5
    assert config.eyes_cascade_classifier.path == "haarcascade_eye.xml"
    assert config.eyes_cascade_classifier.scale_factor == 1.2
    assert config.eyes_cascade_classifier.min_neighbors == 5
