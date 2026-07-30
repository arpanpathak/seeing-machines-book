# Appendix A  -  CivicSense Full Source Reference

This appendix maps every chapter concept to its location in the CivicSense source tree.

## A.1 Repository Structure

```
driving-civicsense-vision-model/
├── Cargo.toml                  # Rust crate manifest
├── Makefile                    # Build / train / deploy targets
├── README.md                   # Project overview
├── CODING_STANDARDS.md         # Formal invariant system (Chapter 16)
├── CONTRIBUTING.md             # How to contribute
├── CLOUD_TRAINING.md           # Cloud GPU training guide (Chapter 5-6)
├── configs/
│   ├── default.yaml            # Default pipeline configuration
│   └── dataset.yaml            # Dataset metadata
├── scripts/
│   ├── download_test_model.sh  # Download YOLO test model
│   └── cross_compile.sh        # Cross-compilation script (Chapter 15)
├── src/
│   ├── main.rs                 # Binary entry, CLI, Pipeline orchestrator (Ch 9)
│   ├── lib.rs                  # Library exports (Ch 7)
│   ├── config.rs               # Configuration deserialization (Ch 8)
│   ├── detection/
│   │   ├── mod.rs              # ObjectDetector trait (Ch 8)
│   │   └── yolo.rs             # YOLO inference + NMS (Ch 4, 8)
│   ├── tracking/
│   │   ├── mod.rs              # Tracking module
│   │   └── deep_sort.rs        # Kalman filter + IoU association (Ch 10, 11)
│   ├── modules/
│   │   ├── mod.rs              # Module trait
│   │   ├── intersection.rs     # Stop sign + occupancy alerts (Ch 13)
│   │   └── lane_speed.rs       # Relative speed + lane courtesy (Ch 14)
│   ├── utils/
│   │   ├── mod.rs              # Utilities module
│   │   ├── geometry.rs         # Pinhole + IoU + filters (Ch 12)
│   │   └── visualization.rs    # Debug overlay rendering
│   ├── video.rs                # Frame I/O (Ch 9)
│   └── train.rs                # Dataset prep + ONNX validation (Ch 5-6)
└── assets/
    └── logo.svg                # Project logo
```

## A.2 Chapter-to-Source Mapping

| Chapter | Key Files | Key Concepts |
|---------|-----------|--------------|
| 1 (Mathematics) | `geometry.rs`, `yolo.rs:sigmoid()` | Pinhole model, covariance, chain rule |
| 2 (Neural Networks) | (Python training script) | Backprop, cross-entropy, dense layers |
| 3 (CNNs) | `yolo.rs:letterbox()` | Convolutions, SiLU, receptive field |
| 4 (YOLO) | `yolo.rs:AnchorGrid`, `decode()`, `nms()` | Grid decoding, NMS, CIoU loss |
| 5 (Training) | `train.rs`, config YAML files | Dataset splitting, mosaic, LR schedule |
| 6 (ONNX) | `train.rs:validate_onnx()` | Export, quantization, ONNX Runtime |
| 7 (Why Rust) | `lib.rs`, `Cargo.toml` | Zero-cost abstractions, memory safety |
| 8 (Inference Engine) | `detection/yolo.rs:YoloDetector` | Letterbox, session run, decoding |
| 9 (Video Pipeline) | `video.rs`, `main.rs:Pipeline` | FrameIter, latency budget |
| 10 (Kalman) | `tracking/deep_sort.rs:KalmanFilter` | State vector, predict, update |
| 11 (Deep SORT) | `tracking/deep_sort.rs:MultiObjectTracker` | IoU matching, track lifecycle |
| 12 (Geometry) | `utils/geometry.rs` | Distance, velocity, low-pass filter |
| 13 (Intersection) | `modules/intersection.rs` | Stop sign, occupancy alerts |
| 14 (Lane Speed) | `modules/lane_speed.rs` | Lane assignment, hysteresis |
| 15 (Deployment) | `scripts/cross_compile.sh`, `Makefile` | Cross-compilation, systemd |
| 16 (Testing) | All `#[cfg(test)]` modules | Unit tests, proptest, benchmarks |

## A.3 Key Constants and Their Locations

| Constant | Value | File | Purpose |
|----------|-------|------|---------|
| `Q_VAR` | 0.01 | `deep_sort.rs` | Process noise for Kalman filter |
| `R_VAR` | 0.1 | `deep_sort.rs` | Measurement noise for Kalman filter |
| `P_INIT` | 10.0 | `deep_sort.rs` | Initial state covariance |
| `MIN_CONFIDENCE` | 0.5 | `intersection.rs` | Minimum confidence for stop sign detection |
| IoU threshold | 0.3 | `deep_sort.rs` | Gating threshold for track association |
| NMS threshold | 0.45 | `config.rs` | IoU threshold for NMS suppression |
| `ALPHA` (LPF) | 0.3 | `lane_speed.rs` | Low-pass filter smoothing factor |
| `stop_sign_width_m` | 0.75 | `intersection.rs` | Real-world stop sign diameter (US) |
| `vehicle_width_m` | 1.8 | `lane_speed.rs` | Average vehicle width |

## A.4 Configuration Defaults

The complete default configuration is in `configs/default.yaml`:

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

camera:
  focal_length: 650.0
  frame_width: 1280
  frame_height: 720
  fps: 30

tracking:
  max_cosine_distance: 0.2
  max_age: 30
  n_init: 3

intersection:
  stop_sign_warning_distance: 50.0
  stop_sign_warning_speed: 10.0
  blocked_intersection_speed: 15.0
  blocked_distance_to_stop: 30.0
  grid_resolution: 0.5
  grid_ahead_distance: 20.0

lane_speed:
  speed_diff_threshold: 5.0
  hysteresis_seconds: 3.0
```
