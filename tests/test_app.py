
"""Tests for the Flask application."""

from app import app
from io import BytesIO


def test_homepage():
    """Test that the home page loads successfully."""
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200


def test_valid_image_upload(monkeypatch):
    """Test that a valid image upload returns a prediction."""

    monkeypatch.setattr(
        "app.preprocess_img",
        lambda image: "processed_image"
    )

    monkeypatch.setattr(
        "app.predict_result",
        lambda image: 2
    )

    client = app.test_client()

    response = client.post(
        "/prediction",
        data={
            "file": (BytesIO(b"fake_image_data"), "test.jpg")
        },
        content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert b"2" in response.data


def test_missing_uploaded_file():
    """Test that a missing uploaded file is handled."""

    client = app.test_client()

    response = client.post(
        "/prediction",
        data={},
        content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert b"File cannot be processed." in response.data


def test_invalid_image_upload():
    """Test that uploading a non-image file is handled."""

    client = app.test_client()

    response = client.post(
        "/prediction",
        data={
            "file": (BytesIO(b"This is not an image"), "invalid.txt")
        },
        content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert b"File cannot be processed." in response.data
