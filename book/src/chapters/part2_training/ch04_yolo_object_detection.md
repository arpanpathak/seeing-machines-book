# Chapter 4: YOLO  -  You Only Look Once

> *"The best time to detect an object is the first time you see it. The second best time is now."*  -  Adapted from an old proverb

Before YOLO, object detection was a two-stage affair. Region proposal networks (R-CNN, Fast R-CNN, Faster R-CNN) would first generate candidate regions where objects might exist, then classify each region. This was accurate but slow  -  5-7 FPS on a GPU.

YOLO reframed detection as a single regression problem: look at the image once, and directly predict bounding boxes and class probabilities. The name says it all. The original YOLO achieved 45 FPS on a Titan X GPU in 2015. By YOLOv11 (2024), models were running at 30+ FPS on edge devices with 2-3% of the GPU's power.

This chapter explains how YOLO works, from the architectural innovations to the loss function that drives convergence. We focus on YOLOv8/v11, the versions used in CivicSense, because they represent the sweet spot between accuracy, speed, and ease of ONNX export.

## 4.1 The Core Insight: Detection as Regression

YOLO divides the input image into an \\( S \times S \\) grid. Each grid cell predicts:

- **\(B\) bounding boxes**, each with 4 coordinates \\( (c_x, c_y, w, h) \\) and a confidence score.
- **\(C\) class probabilities** (one per class).

For YOLOv8 with 80 COCO classes, 3 anchors per cell, and three detection scales, the total number of predictions per image is:

\\[3 \times (4 + 1 + 80) \times (80 \times 80 + 40 \times 40 + 20 \times 20) = 3 \times 85 \times 8400 = 2,142,000\\]

But only a handful of these 2 million predictions correspond to real objects. The rest are background and are suppressed during post-processing.

### 4.1.1 The Anchor Grid

YOLOv8 is **anchor-free** in the traditional sense  -  it does not use predefined anchor box shapes. Instead, it directly predicts the center coordinates and dimensions at each grid cell. The "anchor" is the grid cell itself.

The grid is constructed in `src/detection/yolo.rs` through the `AnchorGrid`:

```rust
impl AnchorGrid {
    fn new(input_size: u32) -> Self {
        let anchors: Vec<_> = [8u32, 16, 32]
            .iter()
            .flat_map(|&stride| {
                let grid = input_size / stride;
                (0..grid)
                    .flat_map(move |gy| (0..grid).map(move |gx| (gx as f32, gy as f32, stride as f32)))
            })
            .collect();
        let n = anchors.len();
        Self { anchors, num_predictions: n }
    }
}
```

For a \(640\) input, the strides \(8, 16, 32\) produce grids of \\( 80 \times 80 \\), \\( 40 \times 40 \\), and \\( 20 \times 20 \\) cells  -  \(6400 + 1600 + 400 = 8400\) anchor points total. Each anchor stores its grid coordinates \\( (g_x, g_y) \\) in grid-space and its stride \(s\) in pixels.

**Why three strides?** Objects appear at different scales. A large truck close to the camera might be \\( 300 \times 300 \\) pixels and is best detected at stride 32 (the \\( 20 \times 20 \\) grid). A pedestrian 50 meters away might be only \\( 20 \times 30 \\) pixels and needs the stride-8 detection head (the \\( 80 \times 80 \\) grid). By combining predictions from multiple scales, YOLO detects objects across a wide size range.

## 4.2 The YOLO Output Decoder

The raw output from the ONNX model is a flat tensor. The decoder in `AnchorGrid::decode()` translates this tensor into meaningful bounding boxes.

### 4.2.1 The Decoding Equations

For each anchor at grid position \\( (g_x, g_y) \\) with stride \(s\):

```rust
let cx = (sigmoid(output[i]) * 2.0 - 0.5 + gx) * s;
let cy = (sigmoid(output[1 * stride + i]) * 2.0 - 0.5 + gy) * s;
let w = (sigmoid(output[2 * stride + i]) * 2.0).powi(2) * s;
let h = (sigmoid(output[3 * stride + i]) * 2.0).powi(2) * s;
```

Let us break this down:

- **Center coordinates**: \\( \sigma(t_x) \\) produces a value in \([0, 1]\). The expression \\( 2 \cdot \sigma(t_x) - 0.5 \\) maps this to \([-0.5, 1.5]\), allowing the predicted center to be slightly outside the grid cell. This handles objects whose center falls near cell boundaries. Adding \\( g_x \\) shifts to the correct grid cell, and multiplying by stride \(s\) converts to pixel coordinates.

- **Width and height**: YOLOv8 uses \\( (2 \cdot \sigma(t_w))^2 \\) as the width multiplier. The square ensures positive values. The factor of 2 allows the box to be up to \(4s\) pixels wide (since \\( (2 \times 1)^2 = 4 \\)). For stride 32, this means boxes up to \\( 4 \times 32 = 128 \\) pixels wide  -  roughly the size of a vehicle in dashcam footage.

### 4.2.2 The Sigmoid and Why It Is There

Each of these decoded values goes through a sigmoid first:

```rust
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
```

The sigmoid ensures numerical stability and bounds the raw network output. Without it, the network could predict arbitrarily large or negative values for box coordinates, producing erratic detections. By sigmoid-squashing, we ensure the predictions stay in a reasonable range.

### 4.2.3 Class Selection

For each anchor, the class with the highest sigmoid-mapped score is selected:

```rust
let (best_class, best_conf) = (0..num_classes)
    .map(|c| (c as u32, sigmoid(output[(4 + c) * stride + i])))
    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    .unwrap_or((0, 0.0));
```

This is a **multi-label** classification: each anchor can have multiple classes (sometimes a vehicle is also a truck). But for simplicity, we take the argmax. If you need multi-label detection (uncommon for traffic scenes), you would keep all classes above a threshold instead.

## 4.3 Non-Maximum Suppression: The Great Filter

After decoding, we have up to 8400 candidate boxes per scale, most overlapping heavily. Non-Maximum Suppression (NMS) reduces this to a clean set of detections.

### 4.3.1 The Greedy Algorithm

```rust
fn non_max_suppression(mut candidates: Vec<BBox>, iou_threshold: f32) -> Vec<BBox> {
    candidates.sort_unstable_by(|a, b| {
        b.confidence.partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut suppressed = vec![false; candidates.len()];
    let mut keep = Vec::new();

    for i in 0..candidates.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(candidates[i]);
        for j in (i + 1)..candidates.len() {
            if !suppressed[j] && box_iou(&candidates[i], &candidates[j]) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    keep
}
```

NMS is \\( O(n^2) \\) in the worst case (all \(n\) candidates overlapping with all others). For 8400 candidates, that is 70 million pairwise IoU computations. In practice, most candidates have low confidence and are filtered before NMS, reducing the effective \(n\) to 50-200 per frame.

**Why greedy?** The optimal NMS (finding the maximum-weight independent set of boxes) is NP-hard. Greedy NMS is a heuristic that works well in practice because objects rarely overlap heavily in natural scenes (you do not often see two stop signs occupying the same pixels).

### 4.3.2 The IoU Threshold

The `iou_threshold` (default 0.45 in CivicSense) controls the tradeoff between precision and recall:

- **Low threshold** (e.g., 0.3): more aggressive suppression, fewer detections, higher precision, lower recall.
- **High threshold** (e.g., 0.6): less suppression, more detections, lower precision, higher recall.

The choice depends on the application. For stop sign detection, you want high recall (do not miss a stop sign), so you use a higher threshold. For general object detection, 0.45-0.5 is standard.

### 4.3.3 The Minimum Box Size Filter

The CivicSense decoder includes an additional filter:

```rust
.filter(|(x1, y1, x2, y2, _, _)| (x2 - x1) >= 1.0 && (y2 - y1) >= 1.0)
```

Boxes smaller than 1 pixel are discarded. These arise from numerical noise in the decoder and are never meaningful detections. In practice, you may want a higher minimum (e.g., 3-5 pixels) to filter noise further.

## 4.4 The Loss Function: Training the Beast

YOLOv8/v11 uses a composite loss with three terms:

\\[\mathcal{L} = \lambda_{\text{box}} \cdot \mathcal{L}_{\text{CIoU}} + \lambda_{\text{cls}} \cdot \mathcal{L}_{\text{BCE}} + \lambda_{\text{DFL}} \cdot \mathcal{L}_{\text{DFL}}\\]

### 4.4.1 CIoU Loss: Better Than L1 or L2

Complete IoU loss directly optimizes the overlap between predicted and ground-truth boxes:

\\[\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(\mathbf{b}, \mathbf{b}^{\text{gt}})}{c^2} + \alpha v\\]

where:
- \\( \text{IoU} \\) is the Intersection-over-Union.
- \\( \rho \\) is the Euclidean distance between box centers \\( \mathbf{b} \\) and \\( \mathbf{b}^{\text{gt}} \\).
- \(c\) is the diagonal length of the smallest enclosing box.
- \(v\) measures the consistency of aspect ratios: \\( \frac{4}{\pi^2}(\arctan\frac{w^{\text{gt}}}{h^{\text{gt}}} - \arctan\frac{w}{h})^2 \\).
- \\( \alpha = \frac{v}{(1 - \text{IoU}) + v} \\) is a tradeoff parameter.

CIoU is superior to L1 or L2 loss because:
1. It is scale-invariant  -  a 10-pixel error on a small box is penalized more than on a large box (since IoU is relative).
2. It directly optimizes the evaluation metric (IoU leads to better mAP).
3. The center distance penalty helps boxes converge faster by pulling the predicted center toward the ground truth.

### 4.4.2 Binary Cross-Entropy for Classification

YOLO uses binary cross-entropy (not multi-class cross-entropy) because the same object can belong to multiple classes (a "vehicle" can also be a "truck"):

\\[\mathcal{L}_{\text{BCE}} = -\sum_{c=1}^{C} [y_c \log(\sigma(z_c)) + (1 - y_c) \log(1 - \sigma(z_c))]\\]

where \\( z_c \\) is the raw logit for class \(c\), \\( \sigma \\) is the sigmoid, and \\( y_c \in \{0, 1\} \\) is the ground-truth indicator.

### 4.4.3 Distribution Focal Loss (DFL)

The DFL reformulates bounding box regression as a classification problem. Instead of predicting a single value for each box coordinate, the network predicts a **distribution** over discrete positions:

\\[\mathcal{L}_{\text{DFL}} = -\sum_{k} \text{KL}(y_k \| \hat{y}_k)\\]

where \\( y_k \\) is the target distribution (a Dirac delta at the correct position) and \\( \hat{y}_k \\) is the predicted distribution.

The advantage: the network outputs a probability distribution over possible box positions, and you take the **expectation** (weighted average) as the final coordinate. This produces smoother, more accurate boxes because the network is penalized for being confidently wrong in a single position.

## 4.5 The Capstone Connection: CivicSense YOLO Model

The CivicSense model is trained on 7 classes specific to traffic analysis:

1. `stop_sign`  -  Red octagonal stop signs.
2. `traffic_light`  -  Red, yellow, and green traffic signals.
3. `crosswalk`  -  Pedestrian crossing markings.
4. `vehicle`  -  Cars, SUVs, vans.
5. `truck`  -  Large cargo vehicles.
6. `bus`  -  Public transit buses.
7. `intersection_zone`  -  Semantic region of the intersection ahead.

This limited class set (compared to COCO's 80) allows the model to be smaller and faster  -  the YOLOv11n variant used in CivicSense has only 2.6 million parameters versus 63 million for YOLOv11x. The detection head output is \(4 + 7 = 11\) channels per anchor (instead of 84 for COCO).

The model configuration is in `configs/default.yaml`:

```yaml
model:
  path: "weights/best-int8.onnx"
  conf_threshold: 0.5
  iou_threshold: 0.45
  input_width: 640
  input_height: 640
  classes:
    - stop_sign
    - traffic_light
    - crosswalk
    - vehicle
    - truck
    - bus
    - intersection_zone
```

## 4.6 Exercises

1. **Implement the YOLO decoder from scratch** in Python. Take a random tensor of shape \((1, 11, 8400)\) and decode it using the equations in Section 4.2. Verify that the output coordinates are in pixel space.

2. **NMS analysis.** Write a script that generates random overlapping boxes, runs NMS with different IoU thresholds, and plots the number of kept boxes versus threshold. Find the knee of the curve.

3. **Loss function visualization.** Plot the CIoU loss for two boxes as you vary their relative positions. Show that the loss is smooth and has a clear minimum when the boxes perfectly overlap.

4. **Scale analysis.** Take a 640×640 image, manually place a 100×100 box (vehicle), a 20×20 box (distant vehicle), and a 10×10 box (very distant). Determine which detection head (stride 8, 16, or 32) would fire for each.

## 4.7 Key Takeaways

- YOLO reframes object detection as a single regression problem  -  look at the image once and predict everything.
- The anchor grid with three strides (8, 16, 32) provides multi-scale detection coverage.
- The output decoder converts raw network outputs to bounding boxes using sigmoid squashing and grid-aware coordinate transforms.
- NMS reduces 8400 candidates to a clean detection set using greedy IoU suppression.
- The composite loss (CIoU + BCE + DFL) directly optimizes the evaluation metric and produces accurate boxes.
- A specialized 7-class model is smaller and faster than a generic 80-class model while being more accurate for the traffic domain.

In Chapter 5, we build the production training pipeline  -  dataset management, augmentation, training loops, and hyperparameter tuning  -  all in typed Python with strict type annotations.
