"""Conftest"""

from unittest.mock import patch

import numpy as np
from flask.testing import FlaskClient
from pytest import fixture

from app import app
from src.config import AppConfig
from src.googly_eyes import Googlify
from src.model.cascade_classifier import CascadeClassifier


@fixture
def client() -> FlaskClient:
    """Get flask test client

    Yields:
        FlaskClient
    """
    yield app.test_client()


class MockCascadeClassifier:
    def detectMultiScale(self, image, scaleFactor, minNeighbors) -> np.ndarray:
        return np.zeros((3, 4), dtype=np.int32)


@fixture
def mock_cascade_classifier() -> CascadeClassifier:
    """Mock cascade classifier"""
    with patch(
        "src.model.cascade_classifier.cv2.CascadeClassifier",
        return_value=MockCascadeClassifier(),
    ):
        app_config = AppConfig.load("./tests/resources/config.yml")
        return CascadeClassifier(
            face_config=app_config.face_cascade_classifier,
            eyes_config=app_config.eyes_cascade_classifier,
        )


@fixture
def mock_googlify(mock_cascade_classifier: CascadeClassifier) -> Googlify:
    """Mock Googlify class."""
    config = AppConfig.load("./tests/resources/config.yml")
    with patch(
        "src.googly_eyes.CascadeClassifier", return_value=mock_cascade_classifier
    ):
        return Googlify(config=config)
