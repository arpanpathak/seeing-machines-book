# Chapter 8: Building the Inference Engine

> *"The model is the specification. The inference engine is the implementation."*

The inference engine is where the neural network meets the real world. It takes raw bytes from a camera sensor and produces structured detections that the rest of the pipeline can act on. This chapter examines the CivicSense inference engine — `YoloDetector` in `src/detection/yolo.rs` — in complete detail.

Every line of this code has been shaped by real hardware constraints: the model must run within a 25 ms budget on a 4-core ARM CPU while sharing memory with a Kalman filter tracker, an intersection analyzer, and a visualization module.

## 8.1 The Detector Interface

The `YoloDetector` wraps the ONNX Runtime session and provides a clean `detect()` interface:

```rust
pub struct YoloDetector {
    config: YoloConfig,
    session: Option<ort::session::Session>,
    anchor_grid: AnchorGrid,
}

impl YoloDetector {
    pub fn new(config: YoloConfig) -> Result<Self, String> { /* ... */ }
    
    pub fn detect(
        &mut self,
        frame: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<Detection>, String> { /* ... */ }
}
```

The `session` is `Option` because the model file may not exist during development. This is not an error — it allows the pipeline to be tested for integration and data collection before a model is trained. The `detect()` method returns an empty `Vec` when no model is loaded.

**Why `&mut self`?** The ONNX Runtime session is not mutated during inference (it is read-only), but the `ort` crate's API requires `&mut` for the `run()` method. This is a quirk of the Rust bindings, not a conceptual requirement. In the future, interior mutability (`RefCell` or `Mutex`) could make the interface shared.

## 8.2 Loading the Model

```rust
impl YoloDetector {
    pub fn new(config: YoloConfig) -> Result<Self, String> {
        let path = Path::new(&config.model_path);

        let session = match path.exists() {
            true => {
                log::info!("Loading ONNX model from '{}'", config.model_path);
                let s = ort::session::Session::builder()
                    .map_err(|e| format!("ort init: {e}"))?
                    .commit_from_file(path)
                    .map_err(|e| format!("Failed to load model '{}': {e}", config.model_path))?;
                log::info!("Model loaded: {}x{} input", config.input_width, config.input_height);
                Some(s)
            }
            false => {
                log::warn!(
                    "ONNX model not found at '{}'. Detector returns empty results. \
                     Train a model and place it at this path.",
                    config.model_path
                );
                None
            }
        };

        let anchor_grid = AnchorGrid::new(config.input_width);
        Ok(Self { config, session, anchor_grid })
    }
}
```

**Model loading is eager, not lazy.** When `YoloDetector::new()` is called, the entire ONNX model is loaded into memory and parsed. This takes approximately 200-500 ms for a 5 MB model on an ARM CPU. We do this at startup, not on the first inference call, because:

1. The pipeline initialization is a one-time cost. The user already waited for the camera to initialize.
2. Lazy loading would introduce a latency spike on the first frame, potentially missing a critical detection.
3. If the model file is corrupted, we catch the error at startup, not during a driving session.

## 8.3 Pre-processing: The Letterbox Transform

Before the model can process a frame, the frame must be transformed into the format the model expects:

1. **Resize** while preserving aspect ratio (letterbox).
2. **Pad** with gray pixels (value 114/255 = 0.447).
3. **Normalize** from $[0, 255]$ to $[0, 1]$.
4. **Rearrange** from HWC (Height-Width-Channel) to CHW (Channel-Height-Width).

```rust
fn letterbox(
    frame: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> LetterBox {
    let scale = (dst_w as f32 / src_w as f32).min(dst_h as f32 / src_h as f32);
    let new_w = (src_w as f32 * scale).round() as u32;
    let new_h = (src_h as f32 * scale).round() as u32;
    let pad_x = (dst_w - new_w) as f32 / 2.0;
    let pad_y = (dst_h - new_h) as f32 / 2.0;

    let src_img =
        image::RgbImage::from_raw(src_w, src_h, frame.to_vec()).expect("valid frame buffer");
    let resized = image::imageops::resize(
        &src_img, new_w, new_h, image::imageops::FilterType::CatmullRom,
    );

    let mut tensor = vec![114.0f32 / 255.0; (dst_w * dst_h * 3) as usize];
    let total = (dst_w * dst_h) as usize;

    // Copy resized image into the center of the padded tensor
    for y in 0..new_h {
        for x in 0..new_w {
            let pixel = resized.get_pixel(x, y);
            let idx = ((y as f32 + pad_y) as u32 * dst_w + (x as f32 + pad_x) as u32) as usize;
            tensor[idx] = pixel[0] as f32 / 255.0;
            tensor[total + idx] = pixel[1] as f32 / 255.0;
            tensor[2 * total + idx] = pixel[2] as f32 / 255.0;
        }
    }

    LetterBox { tensor, scale, pad_x, pad_y }
}
```

**Why is this on the hot path and yet we use `to_vec()` and `from_raw()`?** This is a valid concern. The `frame.to_vec()` call clones the entire camera frame — approximately 2.76 MB for a $1280 \times 720$ RGB frame. On a CPU with limited RAM bandwidth (Raspberry Pi), this can take 1-2 ms per frame.

The optimization path: pre-allocate a buffer at startup and reuse it for each frame, avoiding the `to_vec()` clone. This is tracked as an optimization item in the repository's performance budget.

**Why CatmullRom interpolation?** As discussed in Chapter 3, CatmullRom is a cubic interpolant that produces smoother results than bilinear interpolation, especially for fine details like text on stop signs. It is the default in the `image` crate and adds approximately 0.5 ms to pre-processing on a Pi 5 — acceptable for the 33 ms budget.

## 8.4 Running the Model

```rust
pub fn detect(
    &mut self,
    frame: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<Detection>, String> {
    let session = match &mut self.session {
        Some(s) => s,
        None => return Ok(Vec::new()),
    };

    let LetterBox { tensor, scale, pad_x, pad_y } =
        letterbox(frame, width, height, self.config.input_width, self.config.input_height);

    // Create a 4D array from the flat CHW tensor
    let array = ndarray::Array4::from_shape_vec(
        (1, 3, self.config.input_height as usize, self.config.input_width as usize),
        tensor,
    )
    .map_err(|e| format!("tensor shape: {e}"))?;

    let input_tensor = ort::value::Tensor::from_array(array)
        .map_err(|e| format!("tensor from array: {e}"))?;

    let outputs = session
        .run(ort::inputs![input_tensor])
        .map_err(|e| format!("inference failed: {e}"))?;

    let tensor_ref: ort::value::TensorRef<'_, f32> = outputs[0]
        .downcast_ref()
        .map_err(|e| format!("output downcast: {e}"))?;

    let (_shape, output_data) = tensor_ref
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("output data: {e}"))?;

    // Decode the output into bounding boxes
    let num_classes = self.config.class_names.len();
    let num_predictions = self.anchor_grid.num_predictions;
    let expected = 4 + num_classes;

    match output_data.len() >= expected * num_predictions {
        false => Err(format!(
            "Expected >= {} elements, got {}",
            expected * num_predictions,
            output_data.len()
        )),
        true => {
            let candidates = self.anchor_grid.decode(
                output_data, num_classes, self.config.conf_threshold,
                width, height, scale, pad_x, pad_y,
            );
            let kept = non_max_suppression(candidates, self.config.iou_threshold);
            let detections: Vec<Detection> = kept
                .into_iter()
                .map(|b| Detection {
                    x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2,
                    confidence: b.confidence, class_id: b.class_id,
                })
                .collect();

            log::debug!("detect() returned {} detections", detections.len());
            Ok(detections)
        }
    }
}
```

**The three-stage pipeline:**

1. **Pre-process** (letterbox + CHW conversion): ~2 ms on Pi 5.
2. **Inference** (ONNX Runtime forward pass): ~22 ms on Pi 5 (with Hailo-8L NPU: ~18 ms).
3. **Post-process** (decoding + NMS): ~1 ms on Pi 5.

Total: ~25 ms, within the 33 ms budget at 30 FPS.

## 8.5 Output Decoding: From Tensor to Boxes

The `AnchorGrid::decode()` method transforms the raw output tensor into bounding boxes. The key insight is that the output tensor is laid out as `[1, 11, 8400]` for our 7-class model, where each of the 8400 anchor points has 11 channels of data:

- Channels 0-3: box coordinates $(t_x, t_y, t_w, t_h)$ (raw logits).
- Channels 4-10: class logits.

```rust
fn decode(
    &self,
    output: &[f32],
    num_classes: usize,
    conf_threshold: f32,
    orig_w: u32,
    orig_h: u32,
    scale: f32,
    pad_x: f32,
    pad_y: f32,
) -> Vec<BBox> {
    let stride = self.num_predictions;  // 8400

    self.anchors
        .iter()
        .enumerate()
        .filter_map(|(i, &(gx, gy, s))| {
            let cx = (sigmoid(output[i]) * 2.0 - 0.5 + gx) * s;
            let cy = (sigmoid(output[1 * stride + i]) * 2.0 - 0.5 + gy) * s;
            let w = (sigmoid(output[2 * stride + i]) * 2.0).powi(2) * s;
            let h = (sigmoid(output[3 * stride + i]) * 2.0).powi(2) * s;

            let (best_class, best_conf) = (0..num_classes)
                .map(|c| (c as u32, sigmoid(output[(4 + c) * stride + i])))
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, 0.0));

            (best_conf >= conf_threshold).then_some((
                ((cx - w / 2.0 - pad_x) / scale).clamp(0.0, orig_w as f32),
                ((cy - h / 2.0 - pad_y) / scale).clamp(0.0, orig_h as f32),
                ((cx + w / 2.0 - pad_x) / scale).clamp(0.0, orig_w as f32),
                ((cy + h / 2.0 - pad_y) / scale).clamp(0.0, orig_h as f32),
                best_class,
                best_conf,
            ))
        })
        .filter(|(x1, y1, x2, y2, _, _)| (x2 - x1) >= 1.0 && (y2 - y1) >= 1.0)
        .map(|(x1, y1, x2, y2, class_id, confidence)| BBox {
            x1, y1, x2, y2, confidence, class_id,
        })
        .collect()
}
```

**The tensor layout is critical to understand.** The output is NOT transposed — it is in channel-first order. This means for anchor index `i`:
- Box coordinate 0 (tx) is at `output[i]`
- Box coordinate 1 (ty) is at `output[1 * stride + i]` where `stride = 8400`
- Box coordinate 2 (tw) is at `output[2 * stride + i]`
- Box coordinate 3 (th) is at `output[3 * stride + i]`
- Class 0 logit is at `output[4 * stride + i]`
- Class 1 logit is at `output[5 * stride + i]`
- etc.

A common bug is to misinterpret the layout as `[batch, anchors, channels]` when it is actually `[batch, channels, anchors]`. The difference is whether you index with `output[i * 11 + 4]` (wrong for YOLOv8) or `output[4 * 8400 + i]` (correct for YOLOv8). This bug cost me approximately 6 hours of debugging the first time I implemented a YOLO decoder.

## 8.6 Post-Processing: Non-Maximum Suppression

After decoding, we have anchor coordinates in the *model's* pre-processed coordinate system. We apply NMS to suppress overlapping detections:

```rust
fn non_max_suppression(mut candidates: Vec<BBox>, iou_threshold: f32) -> Vec<BBox> {
    candidates.sort_unstable_by(|a, b| {
        b.confidence.partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut suppressed = vec![false; candidates.len()];
    let mut keep = Vec::new();

    for i in 0..candidates.len() {
        if suppressed[i] { continue; }
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

**Performance note**: NMS is $O(n^2)$ in the number of candidates after confidence filtering. For typical traffic scenes, we have 50-200 candidates, so NMS runs in <0.1 ms. If you have 500+ candidates (congested scenes), consider the `torchvision::ops::nms()` equivalent or a more efficient NMS algorithm.

## 8.7 Graceful Degradation and Error Handling

The inference engine is designed for graceful degradation:

1. **No model file**: Returns empty detections. The pipeline continues running; tracking and analysis modules operate on empty input and produce no alerts.

2. **Inference failure**: If `session.run()` returns an error, the entire frame's processing fails. This is a hard error because an inference failure indicates a corrupted model or hardware issue.

3. **Output shape mismatch**: If the output tensor does not have the expected dimensions, we return an error with a descriptive message. This catches model version mismatches (e.g., loading a COCO model when the code expects a 7-class model).

4. **Empty detections**: The pipeline calls `process_frame()` with an empty detection list. The tracker handles this gracefully (tracks age out over time). The analysis modules return no alerts. No crash.

Error handling follows the Rust philosophy: errors that can be handled are handled (missing model, empty frame). Errors that should not happen are propagated (inference failure, shape mismatch).

## 8.8 The Capstone Connection: Inference in Context

The `process_frame` method in `src/main.rs` ties the inference engine to the rest of the pipeline:

```rust
fn process_frame(&mut self, frame_buffer: &[u8]) -> Result<bool, String> {
    let dt_secs = 1.0 / self.config.camera.fps as f32;
    let detections = self.detector.detect(frame_buffer, self.frame_width, self.frame_height)?;
    let tracks = self.tracker.update(&detections);

    let intersection_alerts = self.intersection_analyzer
        .analyze(&detections, self.ego_speed, dt_secs);
    let lane_alerts = self.lane_speed_analyzer
        .analyze(&tracks, self.ego_speed, dt_secs);

    log_intersection_alerts(&intersection_alerts);
    log_lane_alerts(&lane_alerts);

    if self.visualize && !detections.is_empty() {
        self.render_frame(&detections, &intersection_alerts, &lane_alerts, frame_buffer);
    }

    self.frame_count += 1;
    log::info!("Frame {}: {} detections, {} tracks",
        self.frame_count, detections.len(), tracks.len());

    Ok(true)
}
```

This is the orchestrator. It takes one frame through the entire pipeline:
1. Detect objects with YOLO.
2. Associate detections with existing tracks (or create new tracks).
3. Analyze intersection safety (stop signs, blocked intersections).
4. Analyze lane speed compliance.
5. Log alerts (and optionally render visualization).
6. Advance to the next frame.

The entire cycle must complete in under 33 ms (30 FPS). The per-component latency budget (from the coding standards):

| Stage | Budget | Typical |
|-------|--------|---------|
| Pre-processing | < 3 ms | 2 ms |
| Inference | < 22 ms | 22 ms |
| Post-processing | < 2 ms | 1 ms |
| Tracking | < 5 ms | 3 ms |
| Modules | < 6 ms | 4 ms |
| **Total** | **< 38 ms** | **32 ms** |

The safety margin (38 ms budget vs 33 ms frame time) allows for occasional spikes due to OS scheduling or memory pressure.

## 8.9 Exercises

1. **Add a new backend.** Implement a new `ObjectDetector` trait that both `YoloDetector` (ONNX) and a mock `NoopDetector` implement. Use trait objects (`Box<dyn ObjectDetector>`) in the pipeline.

2. **Profile the hot path.** Use `perf` or `cargo instruments` to profile the inference loop. Identify the bottleneck (pre-processing, inference, or post-processing). Verify it matches the latency budget.

3. **Add INT8 inference.** Build a version of `YoloDetector` that loads both FP32 and INT8 models and switches between them based on a runtime configuration. Compare output quality.

4. **Error recovery.** Add a mechanism to re-load the model if inference fails (e.g., if the model file was replaced during a hot update). Implement this using `watch` filesystem notifications.

## 8.10 Key Takeaways

- The inference engine converts raw bytes to structured detections in three stages: pre-process (letterbox), inference (ONNX forward pass), post-process (decoding + NMS).
- The letterbox transform must preserve aspect ratio, pad with the dataset mean, and convert HWC to CHW layout. A bug here silently breaks all downstream results.
- The YOLO output tensor is channel-first: `[batch, channels, anchors]`, not `[batch, anchors, channels]`. Index carefully.
- Graceful degradation ensures the pipeline runs even without a model file (returns empty detections).
- The latency budget is tight (33 ms/frame at 30 FPS). Each component must stay within its allocation.

In Chapter 9, we examine the video processing infrastructure — how frames are captured from cameras, video files, and directories, and how the pipeline is orchestrated.
