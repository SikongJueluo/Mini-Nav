"""Data loaders for synthetic and validation datasets using Hugging Face datasets."""

import xml.etree.ElementTree as ET
from pathlib import Path

from datasets import Dataset, Image

# Type alias for objects annotation
ObjectsDict = dict[str, list[list[float]] | list[str] | list[int] | list[float]]


def _parse_bbox_line(line: str) -> tuple[str, list[float], float] | None:
    """Parse a single line from synth annotation file.

    Args:
        line: Line in format "category xmin ymin xmax ymax"

    Returns:
        Tuple of (category, [xmin, ymin, width, height], area) or None if invalid
    """
    parts = line.split()
    if len(parts) != 5:
        return None

    category = parts[0]
    xmin, ymin, xmax, ymax = map(int, parts[1:])
    width = xmax - xmin
    height = ymax - ymin
    area = width * height

    return category, [float(xmin), float(ymin), float(width), float(height)], float(area)


def _get_element_text(element: ET.Element | None, default: str = "0") -> str:
    """Get text from XML element, returning default if element or text is None."""
    if element is None:
        return default
    return element.text if element.text is not None else default


def _parse_voc_xml(xml_path: Path) -> ObjectsDict:
    """Parse a VOC-format XML annotation file.

    Args:
        xml_path: Path to the XML annotation file

    Returns:
        Dictionary containing:
            - bbox: List of bounding boxes in [xmin, ymin, width, height] format
            - category: List of object category names
            - area: List of bounding box areas
            - id: List of object IDs (0-based indices)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    bboxes: list[list[float]] = []
    categories: list[str] = []
    areas: list[float] = []

    for obj in root.findall("object"):
        name_elem = obj.find("name")
        bndbox = obj.find("bndbox")

        if name_elem is None or bndbox is None:
            continue

        name = name_elem.text
        if name is None:
            continue

        xmin = int(_get_element_text(bndbox.find("xmin")))
        ymin = int(_get_element_text(bndbox.find("ymin")))
        xmax = int(_get_element_text(bndbox.find("xmax")))
        ymax = int(_get_element_text(bndbox.find("ymax")))

        width = xmax - xmin
        height = ymax - ymin

        bboxes.append([float(xmin), float(ymin), float(width), float(height)])
        categories.append(name)
        areas.append(float(width * height))

    return {
        "bbox": bboxes,
        "category": categories,
        "area": areas,
        "id": list(range(len(bboxes))),
    }


def load_synth_dataset(
    synth_dir: Path,
    annotations_suffix: str = ".txt",
) -> Dataset:
    """Load synthesized dataset for object detection training.

    Args:
        synth_dir: Directory containing synthesized images and annotations
        annotations_suffix: Suffix for annotation files (default: ".txt")

    Returns:
        Hugging Face Dataset with the following columns:
            - image: PIL Image
            - objects: dict containing:
                - bbox: List of bounding boxes in [xmin, ymin, width, height] format
                - category: List of object category names
                - area: List of bounding box areas
                - id: List of object IDs (0-based indices)

    Example:
        >>> dataset = load_synth_dataset(Path("outputs/synth"))
        >>> sample = dataset[0]
        >>> # sample["image"] - PIL Image
        >>> # sample["objects"]["bbox"] - [[xmin, ymin, width, height], ...]
        >>> # sample["objects"]["category"] - ["category_name", ...]
    """
    synth_dir = Path(synth_dir)
    image_files = sorted(synth_dir.glob("synth_*.jpg"))

    if not image_files:
        return Dataset.from_dict({"image": [], "objects": []}).cast_column("image", Image())

    image_paths: list[str] = []
    all_objects: list[ObjectsDict] = []

    for img_path in image_files:
        image_paths.append(str(img_path))

        anno_path = img_path.with_suffix(annotations_suffix)
        if not anno_path.exists():
            all_objects.append({"bbox": [], "category": [], "area": [], "id": []})
            continue

        bboxes: list[list[float]] = []
        categories: list[str] = []
        areas: list[float] = []
        obj_id = 0

        with open(anno_path, "r", encoding="utf-8") as f:
            for line in f:
                if not (line := line.strip()):
                    continue

                result = _parse_bbox_line(line)
                if result is None:
                    continue

                category, bbox, area = result
                bboxes.append(bbox)
                categories.append(category)
                areas.append(area)
                obj_id += 1

        all_objects.append({"bbox": bboxes, "category": categories, "area": areas, "id": list(range(len(bboxes)))})

    dataset = Dataset.from_dict({"image": image_paths, "objects": all_objects})
    return dataset.cast_column("image", Image())


def load_val_dataset(
    scenes_dir: Path,
    split: str = "easy",
) -> Dataset:
    """Load validation dataset from scene images with VOC-format XML annotations.

    Args:
        scenes_dir: Directory containing scene subdirectories
        split: Scene split to load ('easy' or 'hard')

    Returns:
        Hugging Face Dataset with the following columns:
            - image: PIL Image
            - image_id: Image identifier (filename stem without extension)
            - objects: dict containing (loaded from XML annotations):
                - bbox: List of bounding boxes in [xmin, ymin, width, height] format
                - category: List of object category names
                - area: List of bounding box areas
                - id: List of object IDs (0-based indices)

    Example:
        >>> dataset = load_val_dataset(Path("datasets/InsDet-FULL/Scenes"), "easy")
        >>> sample = dataset[0]
        >>> # sample["image"] - PIL Image
        >>> # sample["image_id"] - "rgb_000"
        >>> # sample["objects"]["bbox"] - [[xmin, ymin, width, height], ...]
        >>> # sample["objects"]["category"] - ["category_name", ...]
    """
    scenes_dir = Path(scenes_dir)
    split_dir = scenes_dir / split

    if not split_dir.exists():
        raise ValueError(f"Scene split directory not found: {split_dir}")

    rgb_files = sorted(split_dir.glob("*/rgb_*.jpg"))

    if not rgb_files:
        return Dataset.from_dict({"image": [], "image_id": [], "objects": []}).cast_column("image", Image())

    image_paths: list[str] = []
    image_ids: list[str] = []
    all_objects: list[ObjectsDict] = []

    for img_path in rgb_files:
        image_paths.append(str(img_path))
        image_ids.append(img_path.stem)

        xml_path = img_path.with_suffix(".xml")
        if xml_path.exists():
            objects: ObjectsDict = _parse_voc_xml(xml_path)
        else:
            objects = {"bbox": [], "category": [], "area": [], "id": []}

        all_objects.append(objects)

    dataset = Dataset.from_dict({
        "image": image_paths,
        "image_id": image_ids,
        "objects": all_objects,
    })
    return dataset.cast_column("image", Image())
