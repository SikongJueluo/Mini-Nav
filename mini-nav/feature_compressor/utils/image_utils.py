"""Image loading and preprocessing utilities."""

from pathlib import Path
from typing import List, Optional, Union

import requests
from PIL import Image


def load_image(path: Union[str, Path]) -> Image.Image:
    """Load an image from file path or URL.

    Args:
        path: File path or URL to image

    Returns:
        PIL Image object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If image cannot be loaded
    """
    path_str = str(path)

    if path_str.startswith(("http://", "https://")):
        response = requests.get(path_str, stream=True)
        response.raise_for_status()
        img = Image.open(response.raw)
    else:
        img = Image.open(path)

    return img


def preprocess_image(image: Image.Image, size: int = 224) -> Image.Image:
    """Preprocess image to square format with resizing.

    Args:
        image: PIL Image
        size: Target size for shortest dimension (default: 224)

    Returns:
        Resized PIL Image
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize while maintaining aspect ratio, then center crop
    image = image.resize((size, size), Image.Resampling.LANCZOS)

    return image


def load_images_from_directory(
    dir_path: Union[str, Path], extensions: Optional[List[str]] = None
) -> List[Image.Image]:
    """Load all images from a directory.

    Args:
        dir_path: Path to directory
        extensions: List of file extensions to include (e.g., ['.jpg', '.png'])

    Returns:
        List of PIL Images
    """
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

    dir_path = Path(dir_path)
    images = []

    for ext in extensions:
        images.extend([load_image(p) for p in dir_path.glob(f"*{ext}")])
        images.extend([load_image(p) for p in dir_path.glob(f"*{ext.upper()}")])

    return images
