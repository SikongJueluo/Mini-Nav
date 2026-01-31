"""Basic usage example for DINOv2 Feature Compressor."""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import requests
from PIL import Image
import io

from dino_feature_compressor import DINOv2FeatureExtractor, FeatureVisualizer


def main():
    # Initialize extractor
    print("Initializing DINOv2FeatureExtractor...")
    extractor = DINOv2FeatureExtractor()

    # Download and save test image
    print("Downloading test image...")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content))

    test_image_path = "/tmp/test_image.jpg"
    img.save(test_image_path)
    print(f"Image saved to {test_image_path}")

    # Extract features
    print("Extracting features...")
    result = extractor.process_image(test_image_path)

    print(f"\n=== Feature Extraction Results ===")
    print(f"Original features shape: {result['original_features'].shape}")
    print(f"Compressed features shape: {result['compressed_features'].shape}")
    print(f"Processing time: {result['metadata']['processing_time']:.3f}s")
    print(f"Compression ratio: {result['metadata']['compression_ratio']:.2f}x")
    print(f"Feature norm: {result['metadata']['feature_norm']:.4f}")
    print(f"Device: {result['metadata']['device']}")

    # Visualize
    print("\nGenerating visualization...")
    viz = FeatureVisualizer()

    fig = viz.plot_histogram(
        result["compressed_features"], title="Compressed Features Distribution"
    )

    output_path = (
        Path(__file__).parent.parent.parent / "outputs" / "basic_usage_histogram"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    viz.save(fig, str(output_path), formats=["html"])
    print(f"Visualization saved to {output_path}.html")

    print("\nDone!")


if __name__ == "__main__":
    main()
