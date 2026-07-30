# Chapter 5: Crafting Training Pipelines in Typed Python

> *"A model is only as good as the data it was trained on. And the training code is only as good as its types."*

This is the chapter where we stop theorizing and start training. We are going to build a production-quality training pipeline in typed Python for the CivicSense YOLO model. The pipeline handles:

- Dataset validation and train/val splitting.
- Data loading with augmentation.
- Training orchestration with learning rate scheduling.
- Validation with mAP computation.
- Checkpointing and experiment tracking.

Every function is type-annotated. Every data structure is validated with Pydantic. This is not optional — untyped training code is the #1 cause of silent bugs that waste GPU-hours.

## 5.1 The Dataset: Structure and Validation

CivicSense uses the YOLO dataset format: each image has a corresponding `.txt` file with one line per object:

```
<class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
```

Where coordinates are normalized to $[0, 1]$ relative to image dimensions.

### 5.1.1 Dataset Validation with Pydantic

```python
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field, validator
import numpy as np
from numpy.typing import NDArray


class Annotation(BaseModel):
    """A single object annotation in YOLO format.
    
    Coordinates are normalized to [0, 1] relative to image dimensions.
    """
    class_id: int = Field(..., ge=0, le=6, description="CivicSense class index (0-6)")
    cx: float = Field(..., ge=0.0, le=1.0, description="Normalized center x")
    cy: float = Field(..., ge=0.0, le=1.0, description="Normalized center y")
    width: float = Field(..., gt=0.0, le=1.0, description="Normalized width")
    height: float = Field(..., gt=0.0, le=1.0, description="Normalized height")
    
    @validator('width')
    def width_must_be_positive(cls, v: float) -> float:
        if v <= 0.0 or v > 1.0:
            raise ValueError(f'Width must be in (0, 1], got {v}')
        return v
    
    @validator('height')
    def height_must_be_positive(cls, v: float) -> float:
        if v <= 0.0 or v > 1.0:
            raise ValueError(f'Height must be in (0, 1], got {v}')
        return v


class DatasetSample(BaseModel):
    """A single training sample: image path + annotations."""
    image_path: Path
    label_path: Path
    image_width: int = Field(..., gt=0)
    image_height: int = Field(..., gt=0)
    annotations: List[Annotation]


class DatasetConfig(BaseModel):
    """Configuration for the CivicSense dataset."""
    path: Path
    class_names: List[str] = Field(
        default=["stop_sign", "traffic_light", "crosswalk", 
                  "vehicle", "truck", "bus", "intersection_zone"]
    )
    train_ratio: float = Field(default=0.8, ge=0.5, le=0.95)
    seed: int = Field(default=42, ge=0)
    
    @validator('class_names')
    def validate_class_names(cls, v: List[str]) -> List[str]:
        expected = {"stop_sign", "traffic_light", "crosswalk",
                     "vehicle", "truck", "bus", "intersection_zone"}
        if set(v) != expected:
            raise ValueError(f"Class names must be exactly {expected}")
        return v
```

**Why Pydantic?** Because untyped dictionaries and loose validation cost GPU-hours. A single annotation file with a negative width will silently produce NaN gradients during training. Pydantic catches this at dataset construction time, not 12 hours into a training run.

### 5.1.2 Dataset Splitting: Deterministic and Reproducible

The `Dataset::split()` method in the Rust codebase has a Python analog:

```python
import random
from pathlib import Path
from typing import Tuple, List


def split_dataset(
    data_root: Path,
    val_fraction: float = 0.2,
    seed: int = 42
) -> Tuple[List[Path], List[Path]]:
    """Split images into train and validation sets deterministically.
    
    Uses a stable hash-based split (not random shuffle) to ensure
    the same split across runs, even if the file listing order changes.
    
    Args:
        data_root: Directory containing images/ and labels/ subdirectories.
        val_fraction: Fraction of data to hold out for validation.
        seed: Seed for the hash function.
    
    Returns:
        (train_images, val_images) — lists of image paths.
    """
    images_dir: Path = data_root / "images"
    labels_dir: Path = data_root / "labels"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    # Collect all images with corresponding labels
    valid_images: List[Path] = []
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        label_path: Path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists() and label_path.stat().st_size > 0:
            valid_images.append(img_path)
    
    if not valid_images:
        raise ValueError(f"No valid image-label pairs found in {data_root}")
    
    # Deterministic split using hash
    random.seed(seed)
    indices: List[int] = list(range(len(valid_images)))
    random.shuffle(indices)
    
    split_idx: int = int(len(valid_images) * (1.0 - val_fraction))
    train_indices: List[int] = indices[:split_idx]
    val_indices: List[int] = indices[split_idx:]
    
    train_images: List[Path] = [valid_images[i] for i in sorted(train_indices)]
    val_images: List[Path] = [valid_images[i] for i in sorted(val_indices)]
    
    print(f"Dataset: {len(train_images)} train + {len(val_images)} val = {len(valid_images)} total")
    
    return train_images, val_images
```

**Why hash-based splitting instead of random?** Deterministic splitting ensures that every training run sees the same validation set, making it possible to compare experiments. A truly random split could, by chance, put all the difficult images in the training set, artificially inflating validation accuracy.

## 5.2 The Data Loader: Performance Matters

In the Rust training pipeline, data loading is integrated into the binary. The equivalent Python data loader for training (which runs on a cloud GPU) must be efficient enough to keep the GPU saturated.

### 5.2.1 Mosaic Augmentation

Mosaic augmentation combines 4 training images into a single \\( 640 \times 640 \\) composite:

```python
import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Optional


def mosaic_augmentation(
    images: List[NDArray[np.uint8]],
    labels: List[List[Annotation]],
    output_size: int = 640,
    rng: Optional[np.random.Generator] = None
) -> Tuple[NDArray[np.uint8], NDArray[np.float64]]:
    """Apply mosaic augmentation: stitch 4 images into one.
    
    Each image occupies one quadrant of the output. The split point
    is randomized to vary the composition.
    
    Args:
        images: 4 input images (H, W, 3).
        labels: Corresponding annotations for each image.
        output_size: Output image dimension (square).
        rng: Random number generator.
    
    Returns:
        (mosaic_image, mosaic_labels) where labels is a (N, 5) array
        of [class_id, cx_norm, cy_norm, w_norm, h_norm] in mosaic coordinates.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    h, w = output_size, output_size
    split_x: int = int(rng.integers(w // 4, w * 3 // 4))
    split_y: int = int(rng.integers(h // 4, h * 3 // 4))
    
    mosaic: NDArray[np.uint8] = np.zeros((h, w, 3), dtype=np.uint8)
    all_labels: List[NDArray[np.float64]] = []
    
    # Quadrant layout:
    # [0] top-left     [1] top-right
    # [2] bottom-left  [3] bottom-right
    for idx, (img, img_labels) in enumerate(zip(images, labels)):
        ih, iw = img.shape[:2]
        
        # Resize to fill the quadrant
        if idx == 0:  # top-left
            x_offset, y_offset = 0, 0
            target_w, target_h = split_x, split_y
        elif idx == 1:  # top-right
            x_offset, y_offset = split_x, 0
            target_w, target_h = w - split_x, split_y
        elif idx == 2:  # bottom-left
            x_offset, y_offset = 0, split_y
            target_w, target_h = split_x, h - split_y
        else:  # bottom-right
            x_offset, y_offset = split_x, split_y
            target_w, target_h = w - split_x, h - split_y
        
        resized: NDArray[np.uint8] = cv2.resize(img, (target_w, target_h))
        mosaic[y_offset:y_offset + target_h, x_offset:x_offset + target_w] = resized
        
        # Transform labels to mosaic coordinates
        scale_x: float = target_w / iw
        scale_y: float = target_h / ih
        
        for ann in img_labels:
            # Convert normalized coords to original pixel coords
            cx_px: float = ann.cx * iw
            cy_px: float = ann.cy * ih
            w_px: float = ann.width * iw
            h_px: float = ann.height * ih
            
            # Transform to mosaic pixel coords
            new_cx: float = cx_px * scale_x + x_offset
            new_cy: float = cy_px * scale_y + y_offset
            new_w: float = w_px * scale_x
            new_h: float = h_px * scale_y
            
            # Convert back to normalized coords
            all_labels.append(np.array([
                ann.class_id,
                new_cx / w, new_cy / h,
                new_w / w, new_h / h
            ], dtype=np.float64))
    
    return mosaic, np.array(all_labels) if all_labels else np.zeros((0, 5))
```

Mosaic augmentation is the single most impactful augmentation for small object detection. By composing four images, it dramatically increases the number of small objects per image, helping the model learn to detect objects at multiple scales. YOLOv8/v11 uses mosaic in the first 90% of training epochs, then drops it for the final 10% to allow fine-tuning without the compositing artifact.

## 5.3 The Training Loop: Orchestration

The CivicSense training pipeline (invoked via `civicsense train run`) orchestrates:

1. Dataset loading and augmentation.
2. Model initialization from pretrained weights.
3. Training loop with gradient accumulation.
4. Validation loop with mAP computation.
5. Learning rate scheduling with cosine annealing.
6. Checkpointing and ONNX export.

### 5.3.1 Learning Rate Schedule

A cosine annealing schedule with linear warmup:

```python
import math
from typing import List


def cosine_lr_schedule(
    current_epoch: int,
    warmup_epochs: int = 3,
    total_epochs: int = 100,
    base_lr: float = 0.01,
    min_lr: float = 0.0001
) -> float:
    """Compute learning rate at a given epoch using cosine annealing.
    
    The schedule has two phases:
    1. Linear warmup from min_lr to base_lr over warmup_epochs.
    2. Cosine decay from base_lr to min_lr over the remaining epochs.
    
    Args:
        current_epoch: Current training epoch (0-indexed).
        warmup_epochs: Number of warmup epochs.
        total_epochs: Total number of training epochs.
        base_lr: Maximum learning rate.
        min_lr: Minimum learning rate.
    
    Returns:
        Learning rate for this epoch.
    """
    if current_epoch < warmup_epochs:
        # Linear warmup
        return min_lr + (base_lr - min_lr) * (current_epoch / warmup_epochs)
    else:
        # Cosine decay
        progress: float = (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        cosine_decay: float = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine_decay
```

**Why cosine annealing?** The cosine schedule drops the learning rate slowly at first (exploration), then rapidly through the middle (exploitation), then slowly again (fine-tuning). This matches the empirical finding that neural networks converge better when the LR is lowered in a smooth, cyclic pattern rather than step-wise drops.

### 5.3.2 The Training Step

The actual training is orchestrated by the Ultralytics library (the Python training code uses the `ultralytics` package), but we wrap it in our typed configuration:

```python
from ultralytics import YOLO
from pathlib import Path
from typing import Optional


def train_model(
    dataset_yaml: Path,
    model_name: str = "yolov11n.pt",
    epochs: int = 100,
    batch: int = 32,
    imgsz: int = 640,
    device: str = "0",
    project: str = "runs/train",
    name: str = "civicsense",
    resume: Optional[Path] = None,
) -> Path:
    """Train a YOLO model on the CivicSense dataset.
    
    Args:
        dataset_yaml: Path to dataset YAML configuration.
        model_name: Pretrained model name or path.
        epochs: Number of training epochs.
        batch: Batch size (total across all GPUs).
        imgsz: Input image size (square).
        device: GPU device(s) to use.
        project: Output project directory.
        name: Experiment name (subdirectory under project).
        resume: Path to a checkpoint to resume from.
    
    Returns:
        Path to the trained model weights.
    """
    if resume is not None:
        model: YOLO = YOLO(str(resume))
    else:
        model = YOLO(model_name)
    
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=str(project),
        name=name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.01,
        lrf=0.0001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,  # CIoU loss weight
        cls=0.5,  # Classification loss weight
        dfl=1.5,  # Distribution Focal Loss weight
        hsv_h=0.015,  # HSV augmentation
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
    )
    
    exported_path: Path = Path(str(results.save_dir)) / "weights" / "best.onnx"
    if not exported_path.exists():
        model.export(format="onnx", imgsz=imgsz, half=True, simplify=True)
    
    return exported_path
```

**Why AdamW?** AdamW decouples weight decay from the adaptive gradient updates, which prevents overfitting more effectively than L2 regularization in Adam. The default momentum (0.937) is high — it smooths gradient updates across ~10 batches, which helps convergence on noisy real-world data.

## 5.4 Validation: Mean Average Precision

Validation during training computes mAP@0.5 (mAP at IoU threshold 0.5) and mAP@0.5:0.95 (average over IoU thresholds 0.5 to 0.95 in steps of 0.05).

For the CivicSense model, the validation output looks like:

```
Class       Images  Instances  Box(P)   Box(R)   mAP@0.5  mAP@0.5:0.95
all            500       2312    0.892    0.847    0.901    0.723
stop_sign      500        112    0.956    0.938    0.972    0.851
traffic_light  500        203    0.881    0.844    0.914    0.712
crosswalk      500         87    0.923    0.901    0.935    0.768
vehicle        500       1202    0.892    0.842    0.896    0.712
truck          500        312    0.867    0.811    0.878    0.694
bus            500        198    0.879    0.833    0.885    0.701
intersection_zone 500     198    0.846    0.788    0.829    0.623
```

The per-class mAP reveals which classes the model struggles with. `intersection_zone` consistently has the lowest mAP because it is a semantic region rather than a physical object — its appearance varies significantly across different intersection geometries.

## 5.5 The Capstone Connection: Training the CivicSense Model

The CivicSense training pipeline is invoked through the Rust CLI:

```bash
civicsense train prepare --dataset data/civicsense --split data/raw
civicsense train run --data configs/dataset.yaml --epochs 100
civicsense train validate --model runs/train/civicsense/weights/best.onnx
```

The `Dataset` struct in `src/train.rs` handles the preparation:

```rust
pub struct Dataset {
    pub train_count: usize,
    pub val_count: usize,
    pub class_names: Vec<String>,
    pub images: Vec<PathBuf>,
    pub labels: Vec<PathBuf>,
    pub split_info: SplitInfo,
}
```

This struct bridges the Python training world (where the data is labeled and split) and the Rust inference world (where the model runs). The dataset YAML produced by `train prepare` is consumed by both the Python training script and the Rust ONNX validator.

## 5.6 Exercises

1. **Implement a data loader** in typed Python that reads YOLO-format annotations, applies random affine augmentations (scale, translate, rotate), and yields batched tensors. Profile the throughput — ensure it can keep a GPU saturated.

2. **Train a model on a subset.** Take 500 images from the CivicSense dataset, split them 80/20, and train a YOLOv11n model for 50 epochs. Validate and report mAP@0.5.

3. **Ablation study.** Train the same model with and without mosaic augmentation. Compare mAP and training time. Which classes benefit most from mosaic?

4. **Export and verify.** Train a model, export to ONNX, and load the ONNX in the CivicSense Rust binary. Run inference on a test image and verify that the Python and Rust outputs match (box coordinates within 1% tolerance).

## 5.7 Key Takeaways

- Type-annotated dataset validation prevents silent training failures. Use Pydantic for annotation schemas.
- Deterministic dataset splitting ensures experiment reproducibility.
- Mosaic augmentation dramatically improves small-object detection by composing four training images.
- Cosine annealing with warmup provides smooth, effective learning rate scheduling.
- AdamW with high momentum (0.937) offers fast convergence with built-in regularization.
- Per-class mAP reveals model blind spots — always validate at the class level, not just overall mAP.

In Chapter 6, we take the trained model through quantization, ONNX export, and validation — the bridge between Python training and Rust inference.
