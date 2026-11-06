# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub>=0.30",
#   "ultralytics>=8.4",
#   "PyYAML>=6.0",
# ]
# ///
"""Fine-tune a small YOLO model on a labeled Hugging Face traffic dataset."""

from __future__ import annotations

import argparse
import ctypes.util
import json
import os
import random
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path

DEFAULT_DATASET_REPO = "thundarstrom/traffic-vehicle-detection"
DEFAULT_MODEL_REPO = "Tawuff/tawwuf-traffic-yolov8n"
BASE_MODEL_REPO = "Ultralytics/YOLOv8"
BASE_MODEL_FILE = "yolov8n.pt"
EXPECTED_CLASSES = ("car", "bus", "truck", "van")
DATASET_LICENSE = "CC-BY-4.0"
DATASET_ATTRIBUTION = (
    "UA-DETRAC-10K-SAMPLE > v1-optimal, provided by a Roboflow user; "
    "dataset mirror: thundarstrom/traffic-vehicle-detection."
)


def ensure_opencv_system_libraries() -> None:
    if ctypes.util.find_library("GL"):
        return
    if os.geteuid() != 0:
        raise RuntimeError("OpenCV requires libGL, and system packages cannot be installed.")
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(
        ["apt-get", "install", "-y", "libgl1", "libglib2.0-0"],
        check=True,
    )


def split_training_images(
    image_paths: Sequence[Path],
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """Return a deterministic train/validation split."""
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be between zero and one.")
    if len(image_paths) < 2:
        raise ValueError("At least two labeled training images are required.")

    shuffled = sorted(image_paths)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    validation = sorted(shuffled[:validation_count])
    training = sorted(shuffled[validation_count:])
    return training, validation


def _paired_images(image_dir: Path, label_dir: Path) -> tuple[list[Path], int]:
    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    paired = [image for image in images if (label_dir / f"{image.stem}.txt").is_file()]
    if not paired:
        raise ValueError(f"No image/label pairs found in {image_dir}.")
    return paired, len(images) - len(paired)


def _link_split(
    images: Sequence[Path],
    source_label_dir: Path,
    output_root: Path,
    split_name: str,
) -> None:
    image_dir = output_root / split_name / "images"
    label_dir = output_root / split_name / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for image in images:
        for source, destination in (
            (image, image_dir / image.name),
            (
                source_label_dir / f"{image.stem}.txt",
                label_dir / f"{image.stem}.txt",
            ),
        ):
            try:
                destination.symlink_to(source.resolve())
            except OSError as error:
                if getattr(error, "winerror", None) != 1314:
                    raise
                shutil.copy2(source, destination)


def prepare_dataset(source_root: Path, output_root: Path, seed: int) -> dict:
    """Build a clean YOLO dataset layout, excluding orphaned label files."""
    train_images, missing_train_labels = _paired_images(
        source_root / "train" / "images",
        source_root / "train" / "labels",
    )
    test_images, missing_test_labels = _paired_images(
        source_root / "test" / "images",
        source_root / "test" / "labels",
    )
    train, validation = split_training_images(train_images, seed=seed)

    _link_split(train, source_root / "train" / "labels", output_root, "train")
    _link_split(
        validation,
        source_root / "train" / "labels",
        output_root,
        "valid",
    )
    _link_split(test_images, source_root / "test" / "labels", output_root, "test")

    yaml_path = output_root / "dataset.yaml"
    yaml_path.write_text(
        "path: {}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        "nc: {}\nnames:\n{}".format(
            output_root.resolve(),
            len(EXPECTED_CLASSES),
            "".join(f"  - {name}\n" for name in EXPECTED_CLASSES),
        ),
        encoding="utf-8",
    )
    return {
        "train_images": len(train),
        "validation_images": len(validation),
        "test_images": len(test_images),
        "train_images_without_labels": missing_train_labels,
        "test_images_without_labels": missing_test_labels,
        "orphan_train_labels": max(
            0,
            len(list((source_root / "train" / "labels").glob("*.txt")))
            - len(train_images),
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument(
        "--model-repo",
        default=os.getenv("HF_MODEL_REPO", DEFAULT_MODEL_REPO),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    ensure_opencv_system_libraries()
    import yaml
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from ultralytics import YOLO

    args = _parse_args()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("The HF_TOKEN secret is required to publish trained weights.")
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32:
        raise ValueError("Epochs, batch size, and image size must be positive.")

    api = HfApi(token=token)
    dataset_info = api.dataset_info(args.dataset_repo)
    license_name = (dataset_info.card_data or {}).get("license")
    if (license_name or "").lower() != "cc-by-4.0":
        raise ValueError(
            f"Expected a CC-BY-4.0 dataset card; found {license_name!r}."
        )

    with tempfile.TemporaryDirectory(prefix="tawwuf-yolo-train-") as work_dir:
        work_root = Path(work_dir)
        dataset_dir = Path(
            snapshot_download(
                repo_id=args.dataset_repo,
                repo_type="dataset",
                revision=dataset_info.sha,
                local_dir=work_root / "dataset-source",
                allow_patterns=[
                    "data.yaml",
                    "README.dataset.txt",
                    "README.md",
                    "README.roboflow.txt",
                    "train/images/**",
                    "train/labels/**",
                    "test/images/**",
                    "test/labels/**",
                ],
                token=token,
            )
        )
        source_yaml = yaml.safe_load(
            (dataset_dir / "data.yaml").read_text(encoding="utf-8")
        )
        class_names = source_yaml.get("names", [])
        if isinstance(class_names, dict):
            class_names = [class_names[key] for key in sorted(class_names)]
        normalized_names = tuple(str(name).strip().lower() for name in class_names)
        if normalized_names != EXPECTED_CLASSES:
            raise ValueError(
                f"Expected classes {EXPECTED_CLASSES}; found {normalized_names}."
            )

        prepared_dir = work_root / "dataset"
        dataset_counts = prepare_dataset(dataset_dir, prepared_dir, args.seed)
        print(
            "Prepared labeled dataset:",
            json.dumps(
                {
                    **dataset_counts,
                    "dataset_repo": args.dataset_repo,
                    "dataset_revision": dataset_info.sha,
                    "license": license_name,
                    "card_size_note": (
                        "The dataset card reports 23,319 images; this pinned "
                        "Hub revision currently contains fewer matched images. "
                        "Only matched image/label pairs are used."
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if dataset_counts["train_images"] < 100:
            raise ValueError("Fewer than 100 matched training images; refusing to train.")

        base_model_path = hf_hub_download(
            repo_id=BASE_MODEL_REPO,
            filename=BASE_MODEL_FILE,
            token=token,
        )
        run_root = work_root / "training-run"
        model = YOLO(base_model_path)
        model.train(
            data=str(prepared_dir / "dataset.yaml"),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            seed=args.seed,
            project=str(run_root),
            name="yolov8n-traffic",
            exist_ok=True,
            patience=5,
            workers=4,
            device=0,
            verbose=True,
        )

        best_weights = run_root / "yolov8n-traffic" / "weights" / "best.pt"
        if not best_weights.is_file():
            raise FileNotFoundError(f"Training did not produce {best_weights}.")

        trained_model = YOLO(str(best_weights))
        metrics = trained_model.val(
            data=str(prepared_dir / "dataset.yaml"),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=0,
            verbose=True,
        )
        metric_summary = {
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }
        model_card = f"""---
license: agpl-3.0
library_name: ultralytics
pipeline_tag: object-detection
tags:
  - traffic
  - vehicle-detection
  - yolov8
---

# Tawwuf YOLOv8n traffic detector

Fine-tuned from `{BASE_MODEL_REPO}/{BASE_MODEL_FILE}` on the public
[{args.dataset_repo}](https://huggingface.co/datasets/{args.dataset_repo})
dataset, pinned to revision `{dataset_info.sha}`. The dataset card declares
CC-BY-4.0. Attribution: {DATASET_ATTRIBUTION}

The Hub dataset card reports 23,319 images, but the pinned repository revision
contains fewer matched train/test image-label pairs and extra orphaned training
label files. Training used only matched pairs: {dataset_counts}.

Class order: `car`, `bus`, `truck`, `van`. Test-split metrics from this run:
`{json.dumps(metric_summary, sort_keys=True)}`.

This is a research/demo model trained on general UA-DETRAC traffic imagery. It
has not been validated on Qatar roads, government-camera footage, weather
conditions, or operational traffic control. Do not use it to control speed
signs or make safety-critical decisions. Commercial use requires independent
review of dataset/model terms and validation.
"""
        api.create_repo(args.model_repo, repo_type="model", private=False, exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(best_weights),
            path_in_repo="best.pt",
            repo_id=args.model_repo,
            repo_type="model",
            commit_message="Upload fine-tuned YOLOv8n traffic detector",
        )
        api.upload_file(
            path_or_fileobj=BytesIO(model_card.encode("utf-8")),
            path_in_repo="README.md",
            repo_id=args.model_repo,
            repo_type="model",
            commit_message="Document training data, attribution, and limitations",
        )
        print(
            json.dumps(
                {
                    "model_repo": args.model_repo,
                    "weights": f"https://huggingface.co/{args.model_repo}/blob/main/best.pt",
                    "metrics": metric_summary,
                    "dataset_counts": dataset_counts,
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
