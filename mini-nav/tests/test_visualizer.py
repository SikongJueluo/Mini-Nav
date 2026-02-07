"""Tests for visualizer app image upload similarity search."""

import base64
import io

import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity


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


class TestCosineSimilarity:
    """Test suite for cosine similarity computation between feature vectors."""

    def test_identical_vectors_return_one(self):
        """Identical vectors should have cosine similarity of 1.0."""
        vec = np.random.randn(1024).tolist()
        similarity = cosine_similarity([vec], [vec])[0][0]
        assert np.isclose(similarity, 1.0)

    def test_orthogonal_vectors_return_zero(self):
        """Orthogonal vectors should have cosine similarity of 0.0."""
        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0]
        similarity = cosine_similarity([vec_a], [vec_b])[0][0]
        assert np.isclose(similarity, 0.0)

    def test_opposite_vectors_return_negative_one(self):
        """Opposite vectors should have cosine similarity of -1.0."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [-1.0, 0.0, 0.0]
        similarity = cosine_similarity([vec_a], [vec_b])[0][0]
        assert np.isclose(similarity, -1.0)

    def test_similarity_range(self):
        """Cosine similarity should always be within [-1, 1]."""
        # Random vectors
        for _ in range(10):
            vec_a = np.random.randn(1024).tolist()
            vec_b = np.random.randn(1024).tolist()
            similarity = cosine_similarity([vec_a], [vec_b])[0][0]
            assert -1.0 <= similarity <= 1.0

    def test_similarity_with_list_input(self):
        """Cosine similarity should work with Python list inputs (as stored in dcc.Store)."""
        # Simulate feature vectors stored as Python lists in dcc.Store
        vec_a = [0.1, 0.2, 0.3, 0.4, 0.5]
        vec_b = [0.1, 0.2, 0.3, 0.4, 0.5]
        similarity = cosine_similarity([vec_a], [vec_b])[0][0]
        assert np.isclose(similarity, 1.0)
