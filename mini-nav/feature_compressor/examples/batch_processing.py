"""Batch processing example for DINOv2 Feature Compressor."""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dino_feature_compressor import DINOv2FeatureExtractor


def main():
    # Initialize extractor
    print("Initializing DINOv2FeatureExtractor...")
    extractor = DINOv2FeatureExtractor()

    # Create a test directory with sample images
    # In practice, use your own directory
    image_dir = "/tmp/test_images"
    Path(image_dir).mkdir(parents=True, exist_ok=True)

    # Create 3 test images
    print("Creating test images...")
    import numpy as np
    from PIL import Image

    for i in range(3):
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(f"{image_dir}/test_{i}.jpg")
    print(f"Created 3 test images in {image_dir}")

    # Process batch
    print("\nProcessing images in batch...")
    results = extractor.process_batch(image_dir, batch_size=2, save_features=True)

    print(f"\n=== Batch Processing Results ===")
    print(f"Processed {len(results)} images")

    for i, result in enumerate(results):
        print(f"\nImage {i + 1}: {result['metadata']['image_path']}")
        print(f"  Compressed shape: {result['compressed_features'].shape}")
        print(f"  Feature norm: {result['metadata']['feature_norm']:.4f}")

    print("\nDone! Features saved to outputs/ directory.")


if __name__ == "__main__":
    main()
