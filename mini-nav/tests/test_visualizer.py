"""Tests for visualizer app image upload similarity search."""

import base64
import io

import numpy as np
from PIL import Image


class TestImageUploadSimilaritySearch:
    """Test suite for image upload similarity search functionality."""

    def test_base64_to_pil_image(self):
        """Test conversion from base64 string to PIL Image."""
        # Create a test image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)

        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Add data URI prefix (as Dash provides)
        img_base64_with_prefix = f"data:image/png;base64,{img_base64}"

        # Parse base64 to PIL Image
        # Remove prefix
        base64_str = img_base64_with_prefix.split(",")[1]
        img_bytes = base64.b64decode(base64_str)
        parsed_img = Image.open(io.BytesIO(img_bytes))

        # Verify the image is valid
        assert parsed_img.size == (224, 224)
        assert parsed_img.mode == "RGB"
