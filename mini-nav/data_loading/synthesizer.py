"""Image synthesizer for generating synthetic object detection datasets."""

import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.Image import Resampling
from rich.progress import track


class ImageSynthesizer:
    """Synthesizes composite images from background and object images with masks."""

    def __init__(
        self,
        dataset_root: Path,
        output_dir: Path,
        num_objects_range: tuple[int, int] = (3, 8),
        num_scenes: int = 1000,
        object_scale_range: tuple[float, float] = (0.1, 0.4),
        rotation_range: tuple[int, int] = (-30, 30),
        overlap_threshold: float = 0.3,
        seed: int = 42,
    ):
        """
        Initialize the image synthesizer.

        Args:
            dataset_root: Root directory of the dataset (InsDet-FULL)
            output_dir: Directory to save synthesized images
            num_objects_range: Range of number of objects per scene
            num_scenes: Number of scenes to generate
            object_scale_range: Range of object scale relative to background
            rotation_range: Range of rotation angles in degrees
            overlap_threshold: Maximum allowed overlap ratio
            seed: Random seed for reproducibility
        """
        self.dataset_root = Path(dataset_root)
        self.output_dir = Path(output_dir)
        self.num_objects_range = num_objects_range
        self.num_scenes = num_scenes
        self.object_scale_range = object_scale_range
        self.rotation_range = rotation_range
        self.overlap_threshold = overlap_threshold
        self.seed = seed

        self.background_dir = self.dataset_root / "Background"
        self.objects_dir = self.dataset_root / "Objects"
        self.scenes_dir = self.dataset_root / "Scenes"

        # Will be populated on first use
        self._background_categories: list[str] | None = None
        self._object_categories: list[str] | None = None

    @property
    def background_images(self) -> list[Path]:
        """List of background image paths."""
        if self._background_categories is None:
            self._background_categories = sorted(
                [
                    p.name
                    for p in self.background_dir.iterdir()
                    if p.suffix in [".jpg", ".jpeg", ".png"]
                ]
            )
        # Return as list of Path for type compatibility
        return [self.background_dir / name for name in self._background_categories]  # type: ignore[return-value]

    @property
    def object_categories(self) -> list[str]:
        """List of object categories."""
        if self._object_categories is None:
            self._object_categories = sorted(
                [d.name for d in self.objects_dir.iterdir() if d.is_dir()]
            )
        return self._object_categories

    def load_background(self, path: Path) -> Image.Image:
        """Load a background image.

        Args:
            path: Background image path

        Returns:
            PIL Image
        """
        return Image.open(path).convert("RGB")

    def load_object(self, category: str, angle: int) -> tuple[Image.Image, Image.Image]:
        """Load an object image and its mask.

        Args:
            category: Object category name (e.g., '099_mug_blue')
            angle: Angle index (1-24)

        Returns:
            Tuple of (image, mask) as PIL Images
        """
        img_path = self.objects_dir / category / "images" / f"{angle:03d}.jpg"
        mask_path = self.objects_dir / category / "masks" / f"{angle:03d}.png"
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        return image, mask

    def get_random_background(self) -> tuple[Image.Image, Path]:
        """Get a random background image.

        Returns:
            Tuple of (image, path)
        """
        path = random.choice(self.background_images)
        return self.load_background(path), path

    def get_random_object(self) -> tuple[Image.Image, Image.Image, str]:
        """Get a random object with its mask.

        Returns:
            Tuple of (image, mask, category_name)
        """
        category = random.choice(self.object_categories)
        angle = random.randint(1, 24)
        image, mask = self.load_object(category, angle)
        return image, mask, category

    def _rotate_image_and_mask(
        self, image: Image.Image, mask: Image.Image, angle: float
    ) -> tuple[Image.Image, Image.Image]:
        """Rotate image and mask together."""
        image = image.rotate(angle, resample=Resampling.BILINEAR, expand=True)
        mask = mask.rotate(angle, resample=Resampling.BILINEAR, expand=True)
        return image, mask

    def _compute_overlap(
        self, box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]
    ) -> float:
        """Compute overlap ratio between two boxes.

        Args:
            box1: (xmin, ymin, xmax, ymax)
            box2: (xmin, ymin, xmax, ymax)

        Returns:
            Overlap ratio (area of intersection / area of smaller box)
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Compute intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0

        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        min_area = min(box1_area, box2_area)

        return inter_area / (min_area if min_area > 0 else 0.0)

    def _has_overlap(
        self,
        new_box: tuple[int, int, int, int],
        existing_boxes: list[tuple[int, int, int, int]],
    ) -> bool:
        """Check if new_box overlaps with any existing boxes.

        Args:
            new_box: The new bounding box (xmin, ymin, xmax, ymax)
            existing_boxes: List of existing bounding boxes

        Returns:
            True if any overlap exceeds threshold, False otherwise
        """
        for existing_box in existing_boxes:
            if self._compute_overlap(new_box, existing_box) > self.overlap_threshold:
                return True
        return False

    def _place_object(
        self,
        background: Image.Image,
        obj_image: Image.Image,
        obj_mask: Image.Image,
        existing_boxes: list[tuple[int, int, int, int]],
        scale: float,
    ) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]] | None:
        """Place an object on the background without exceeding overlap threshold.

        Args:
            background: Background PIL Image
            obj_image: Object PIL Image (RGB)
            obj_mask: Object PIL Image (L)
            existing_boxes: List of existing object boxes
            scale: Scale factor for the object

        Returns:
            Tuple of (new_background, updated_mask, new_box) or None if placement failed
        """
        bg_w, bg_h = background.size

        # Scale object
        obj_w, obj_h = obj_image.size
        new_w = int(obj_w * scale)
        new_h = int(obj_h * scale)

        if new_w <= 0 or new_h <= 0 or new_w > bg_w or new_h > bg_h:
            return None

        obj_image = obj_image.resize((new_w, new_h), Resampling.LANCZOS)
        obj_mask = obj_mask.resize((new_w, new_h), Resampling.LANCZOS)

        # Try to find a valid position
        max_attempts = 50
        for _ in range(max_attempts):
            # Random position
            x = random.randint(0, bg_w - new_w)
            y = random.randint(0, bg_h - new_h)

            new_box = (x, y, x + new_w, y + new_h)

            # Check overlap with existing boxes
            if not self._has_overlap(new_box, existing_boxes):
                # Composite object onto background using Pillow's paste method
                background = background.copy()
                background.paste(obj_image, (x, y), mask=obj_mask)

                return background, obj_mask, new_box

        return None

    def synthesize_scene(
        self,
    ) -> tuple[Image.Image, list[tuple[str, int, int, int, int]]]:
        """Synthesize a single scene with random objects.

        Returns:
            Tuple of (synthesized_image, list of (category, xmin, ymin, xmax, ymax))
        """
        # Load background
        background, _ = self.get_random_background()

        # Determine number of objects
        num_objects = random.randint(*self.num_objects_range)

        # Place objects
        placed_boxes: list[tuple[int, int, int, int]] = []
        annotations: list[tuple[str, int, int, int, int]] = []

        for _ in range(num_objects):
            # Get random object
            obj_image, obj_mask, obj_category = self.get_random_object()

            # Get random scale
            scale = random.uniform(*self.object_scale_range)

            # Get random rotation
            angle = random.uniform(*self.rotation_range)
            obj_image, obj_mask = self._rotate_image_and_mask(
                obj_image, obj_mask, angle
            )

            # Try to place object
            result = self._place_object(
                background, obj_image, obj_mask, placed_boxes, scale
            )

            if result is not None:
                background, _, box = result
                placed_boxes.append(box)
                annotations.append((obj_category, box[0], box[1], box[2], box[3]))

        return background, annotations

    def generate(self) -> list[Path]:
        """Generate all synthesized scenes.

        Returns:
            List of paths to generated images
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        generated_files: list[Path] = []

        for i in track(range(self.num_scenes), description="Generating scenes"):
            # Update seed for each scene
            random.seed(self.seed + i)
            np.random.seed(self.seed + i)

            image, annotations = self.synthesize_scene()

            # Save image
            img_path = self.output_dir / f"synth_{i:04d}.jpg"
            image.save(img_path, quality=95)

            # Save annotation
            anno_path = self.output_dir / f"synth_{i:04d}.txt"
            with open(anno_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                for category, xmin, ymin, xmax, ymax in annotations:
                    writer.writerow([category, xmin, ymin, xmax, ymax])

            generated_files.append(img_path)

        return generated_files
