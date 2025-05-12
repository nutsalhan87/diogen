import json
import random
from pathlib import Path
from typing import List, Tuple
import yaml
from concurrent.futures import ProcessPoolExecutor

from PIL import Image

SOURCE_JSON = "data/carplates/train.json"
SOURCE_IMAGES_DIR = "data/carplates/train"
TARGET_BASE_DIR = "data/carplates_yolo"
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

CLASS_ID = 0
CLASS_NAME = "plate"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def quad_to_yolo_bbox(quad: List[List[int]], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    xc = (x_min + x_max) / 2 / img_w
    yc = (y_min + y_max) / 2 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return xc, yc, w, h


def process_image(entry, split_name):
    src_img_path = Path(entry["file"])
    ext = src_img_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return None

    full_src_path = Path(SOURCE_IMAGES_DIR) / src_img_path.name
    try:
        with Image.open(full_src_path) as img:
            img_w, img_h = img.size
            rgb_img = img.convert("RGB")

            img_out_dir = Path(TARGET_BASE_DIR) / "images" / split_name
            lbl_out_dir = Path(TARGET_BASE_DIR) / "labels" / split_name
            img_out_dir.mkdir(parents=True, exist_ok=True)
            lbl_out_dir.mkdir(parents=True, exist_ok=True)

            if ext == ".bmp":
                img_name = src_img_path.stem + ".jpg"
                rgb_img.save(img_out_dir / img_name, format="JPEG")
            else:
                img_name = src_img_path.name
                rgb_img.save(img_out_dir / img_name)

            label_lines = []
            for plate in entry["nums"]:
                bbox = quad_to_yolo_bbox(plate["box"], img_w, img_h)
                label_lines.append(f"{CLASS_ID} {' '.join(map(str, bbox))}")

            label_file_path = lbl_out_dir / (Path(img_name).stem + ".txt")
            with open(label_file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(label_lines))

            return True
    except Exception:
        return None


def process_image_with_split(args):
    entry, split_name = args
    return process_image(entry, split_name)


def prepare_dataset():
    with open(SOURCE_JSON, encoding="utf-8") as f:
        data = json.load(f)

    random.shuffle(data)
    total = len(data)
    counts = {
        "train": int(total * SPLIT_RATIOS["train"]),
        "val": int(total * SPLIT_RATIOS["val"]),
    }
    counts["test"] = total - counts["train"] - counts["val"]

    splits = {
        "train": data[:counts["train"]],
        "val": data[counts["train"]:counts["train"] + counts["val"]],
        "test": data[counts["train"] + counts["val"]:]
    }

    for split_name, split_data in splits.items():
        print(f"[{split_name.upper()}] Обрабатывается {len(split_data)} изображений...")

        with ProcessPoolExecutor() as executor:
            results = list(executor.map(process_image_with_split, [(entry, split_name) for entry in split_data]))

        success_count = sum(1 for r in results if r)
        print(f"  Обработано успешно: {success_count} / {len(split_data)}")

    yaml_path = Path(TARGET_BASE_DIR) / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump({
            "path": str(Path(TARGET_BASE_DIR).resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {CLASS_ID: CLASS_NAME}
        }, f, allow_unicode=True)


if __name__ == "__main__":
    prepare_dataset()
