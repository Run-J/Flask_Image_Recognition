
"""Tests for image preprocessing."""

from io import BytesIO

import numpy as np
from PIL import Image

from model import preprocess_img


def test_preprocess_img():
    """Test image resizing, shape, and normalization."""

    image = Image.new("RGB", (100, 100), color="white")

    image_file = BytesIO()
    image.save(image_file, format="PNG")
    image_file.seek(0)

    result = preprocess_img(image_file)

    assert result.shape == (1, 224, 224, 3)
    assert np.all(result >= 0)
    assert np.all(result <= 1)
