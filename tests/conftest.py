"""Conftest"""

from flask.testing import FlaskClient
from pytest import fixture

from app import app


@fixture
def client() -> FlaskClient:
    """Get flask test client

    Yields:
        FlaskClient
    """
    yield app.test_client()
