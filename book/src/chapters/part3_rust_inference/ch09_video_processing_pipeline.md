# Chapter 9: Real-Time Video Processing Pipeline

> *"A camera is a light-tight box with a hole in one end. A video pipeline is a time-tight sequence of operations with a frame at one end and an alert at the other."*

The video processing pipeline is the circulatory system of CivicSense. It manages frame acquisition from multiple source types (camera, video file, image directory), feeds frames through the inference engine, and coordinates the timing of the entire system.

This chapter covers the video I/O abstraction in `src/video.rs` and the pipeline orchestration in `src/main.rs`.

## 9.1 The Frame Iterator Abstraction

The pipeline treats all video sources uniformly through a `FrameIter` type alias:

```rust
pub type FrameIter = Box<dyn FnMut() -> Option<(Vec<u8>, u64)>>;
```

This is a **closure-based iterator** — a callable that produces `(rgb_buffer, frame_index)` tuples until the source is exhausted. The `Box<dyn FnMut>` allows different implementations (camera, video file, directory) to return different closure types while conforming to the same interface.

**Why a closure instead of a trait?** A trait (`trait FrameSource { fn next(&mut self) -> Option<Frame> }`) would be more idiomatic Rust. But the closure approach is simpler for a pipeline that needs to pass the iterator through multiple functions without generic type parameters. The performance difference is negligible — a single vtable call per frame (~5 ns).

### 9.1.1 Source Classification

The `open_source()` function routes the source string to the appropriate handler:

```rust
pub fn open_source(
    source: &str,
    default_width: u32,
    default_height: u32,
) -> Result<(FrameIter, u32, u32), String> {
    let kind = classify_source(source);
    log::info!("Opening {kind:?} source: {source}");

    match kind {
        SourceKind::Video => open_video_file(Path::new(source)),
        SourceKind::Image => open_single_image(Path::new(source)),
        SourceKind::Directory => open_image_directory(Path::new(source), default_width, default_height),
        SourceKind::Camera => open_camera(default_width, default_height),
        SourceKind::V4l2Device(dev_path) => open_v4l2_device(&dev_path, default_width, default_height),
    }
}
```

The `classify_source()` function uses file extension and path heuristics to determine the source type:

```rust
fn classify_source(source: &str) -> SourceKind {
    let path = Path::new(source);

    if path.exists() && path.is_file() {
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        if ["mp4", "avi", "mov", "mkv", "webm", "m4v"].contains(&ext.as_str()) {
            return SourceKind::Video;
        }
        if ext == "jpg" || ext == "jpeg" || ext == "png" {
            return SourceKind::Image;
        }
    }

    if path.exists() && path.is_dir() {
        return SourceKind::Directory;
    }

    if source == "camera" || source == "0" {
        return SourceKind::Camera;
    }

    // Check for V4L2 device
    let dev_path = format!("/dev/video{source}");
    if Path::new(&dev_path).exists() {
        return SourceKind::V4l2Device(dev_path);
    }

    SourceKind::Video // fallback — will error when opened
}
```

**Why not use OpenCV for video decoding?** OpenCV's Rust bindings (`opencv-rust`) are notoriously difficult to cross-compile for ARM targets. The `image` crate is pure Rust, compiles cleanly on any target, and handles the image formats we need (JPEG, PNG, BMP). For full video decoding, we rely on external tools (ffmpeg) or pre-processed image sequences.

### 9.1.2 Camera Backend

The camera module detects the available camera tool at runtime:

```rust
fn open_camera(default_width: u32, default_height: u32) -> Result<(FrameIter, u32, u32), String> {
    let tool = if has_tool("libcamera-still") {
        "libcamera-still"
    } else if has_tool("raspistill") {
        "raspistill"
    } else {
        log::warn!("No camera backend found. Using dummy frame.");
        return Ok((once_iter(dummy_frame(default_width, default_height)), default_width, default_height));
    };

    let tmp_dir = std::env::temp_dir().join("civicsense_capture");
    std::fs::create_dir_all(&tmp_dir)
        .map_err(|e| format!("Cannot create temp dir: {e}"))?;

    let mut frame_idx: u64 = 0;
    let w = default_width;
    let h = default_height;
    let tool_owned = tool.to_string();

    Ok((Box::new(move || {
        let capture_path = tmp_dir.join(format!("capture_{frame_idx}.jpg"));
        let args = build_camera_args(&tool_owned, w, h, &capture_path);
        let args_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();

        if let Some(buffer) = capture_frame_via(&tool_owned, &args_refs, &capture_path) {
            let idx = frame_idx;
            frame_idx += 1;
            Some((buffer, idx))
        } else {
            None
        }
    }), w, h))
}
```

**Why shell out to `libcamera-still` instead of using V4L2 directly?** The V4L2 kernel interface requires complex buffer management (request buffers, queue buffers, dequeue buffers, memory-map or user-pointer mode). Our closure-based approach with `libcamera-still` works on any Raspberry Pi OS version. The V4L2 backend is a placeholder for future optimization.

The performance cost: each camera frame requires:
1. A shell command execution (~2 ms).
2. Reading the JPEG from disk (~1 ms).
3. Decoding the JPEG to RGB (~3 ms).

Total: ~6 ms per frame for camera acquisition. This fits within the 33 ms budget, but leaves room for optimization (a direct V4L2 MMAP capture could reduce this to ~1 ms).

## 9.2 Frame Saving

Captured frames (for training data) are saved as JPEG via the `save_frame` function:

```rust
pub fn save_frame(buffer: &[u8], width: u32, height: u32, path: &Path) -> Result<(), String> {
    let img = image::RgbImage::from_raw(width, height, buffer.to_vec())
        .ok_or_else(|| "Failed to create image from raw buffer".to_string())?;
    img.save(path)
        .map_err(|e| format!("Failed to save image to '{}': {e}", path.display()))?;
    Ok(())
}
```

The data collection subcommand (`civicsense collect`) captures frames at a configurable rate and saves them as timestamped JPEGs:

```rust
fn run(&mut self) -> Result<(), String> {
    let start = Instant::now();
    let mut saved_count: u64 = 0;
    let mut last_save = Instant::now()
        .checked_sub(std::time::Duration::from_secs(3600))
        .unwrap_or(Instant::now());

    loop {
        let frame_buffer = match (self.frame_iter)() {
            Some((buf, _)) => buf,
            None => break,
        };

        if last_save.elapsed().as_millis() as u64 >= self.min_interval_ms {
            if self.save_one_frame(&frame_buffer, saved_count).is_ok() {
                saved_count += 1;
                last_save = Instant::now();
                
                if self.max_frames > 0 && saved_count >= self.max_frames {
                    break;
                }
            }
        }
    }

    let elapsed = start.elapsed();
    let effective_fps = saved_count as f64 / elapsed.as_secs_f64();
    log::info!("Collection: {saved_count} frames in {elapsed:.1?} ({effective_fps:.1} fps avg)");
    Ok(())
}
```

## 9.3 Pipeline Orchestration

The `Pipeline` struct in `src/main.rs` orchestrates the entire perception pipeline:

```rust
struct Pipeline {
    detector: YoloDetector,
    tracker: MultiObjectTracker,
    intersection_analyzer: IntersectionAnalyzer,
    lane_speed_analyzer: LaneSpeedAnalyzer,
    frame_iter: video::FrameIter,
    config: Config,
    frame_width: u32,
    frame_height: u32,
    frame_count: u64,
    visualize: bool,
    ego_speed: f32,
    viz_output_dir: PathBuf,
}

impl Pipeline {
    fn run(&mut self) -> Result<(), String> {
        loop {
            match (self.frame_iter)() {
                None => {
                    log::info!("End of source. Processed {} frames.", self.frame_count);
                    return Ok(());
                }
                Some((buffer, _)) => {
                    if !self.process_frame(&buffer)? {
                        break;
                    }
                }
            }
        }
        Ok(())
    }
}
```

The `run()` method is a simple loop: get a frame, process it, repeat. There is no frame dropping mechanism yet — every frame is processed. At 30 FPS, this is fine. If the pipeline falls behind (e.g., running at 25 FPS), we would need to add frame dropping: `if frame_arrival_time - last_process_time < 33ms { skip }`.

### 9.3.1 The Ego Speed Input

The pipeline accepts an `ego_speed` parameter that represents the vehicle's current speed in mph. This is used by the intersection and lane-speed modules to determine whether alerts should fire.

Currently, `ego_speed` is provided as a CLI argument:

```bash
civicsense run --source test_video.mp4 --visualize --ego_speed 35.0
```

In production, ego speed would come from:
1. **GPS** — Via a USB GPS dongle (common in dashcams), providing speed at 1 Hz. The Rust pipeline reads `/dev/ttyUSB0` and parses NMEA sentences.
2. **OBD-II** — The vehicle's CAN bus, providing wheel-speed data at 10+ Hz via a Bluetooth OBD-II adapter.
3. **Visual odometry** — Camera-only speed estimation using the change in detected vehicle sizes over time (this is what the `compute_relative_velocity` function in `geometry.rs` does for other vehicles).

The ego speed is used to determine if the driver is approaching a stop sign too fast, if they are crawling in the left lane, or if the intersection ahead is getting dangerously close.

## 9.4 Performance Budget Discipline

The CivicSense coding standards define a strict latency budget:

```
Inference:   < 25 ms  (yolo model forward pass)
Tracking:    < 5 ms   (association + kalman update)
Intersection:< 3 ms   (grid occupancy + deceleration check)
Lane Speed:  < 3 ms   (velocity estimation + hysteresis)
Overhead:    < 4 ms   (pre/post processing, frame copy)
Total:       < 40 ms  (target: 25 fps minimum)
```

Any PR that exceeds this budget without profiling data is rejected. The budget is enforced through:

1. **Continuous benchmarking** — `cargo bench` measures each stage's latency on every commit.
2. **Regression detection** — A CI check compares benchmark results against the previous commit.
3. **Documentation** — Each hot-path function documents its expected latency.

### 9.4.1 The Unwritten Rule: No Allocations on the Hot Path

The most common performance regression is an accidental allocation in the hot path. The coding standards are explicit:

> No `to_vec()`, `clone()`, `format!()` in the inference loop.

The current codebase violates this in one place — `visualize` mode creates a new frame buffer each frame (`let mut viz = frame_buffer.to_vec();`). This is acceptable because visualization is disabled in production (it is a debug feature). But it is noted in the code as a known violation.

## 9.5 The Capstone Connection: Running the Pipeline

The pipeline is invoked through the CLI:

```bash
# On live camera
civicsense run --source camera --visualize

# On recorded video
civicsense run --source test_video.mp4 --visualize --ego_speed 35.0

# On a directory of frames
civicsense run --source /path/to/frames/ --visualize

# Headless (no visualization, for production)
civicsense run --source camera
```

The binary is compiled with release optimizations and LTO:

```toml
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
```

These settings produce the fastest single-threaded binary possible. `lto = true` enables link-time optimization across crate boundaries (the ONNX Runtime static library is also LTO'd). `codegen-units = 1` prevents the compiler from splitting the crate into parallel compilation units, which would inhibit some cross-function optimizations.

On a Raspberry Pi 5 (4× Cortex-A76 @ 2.4 GHz), the release binary achieves:

| Metric | Value |
|--------|-------|
| Inference latency | 22 ms (ONNX CPU, FP32) |
| Tracking latency | 3 ms |
| Module latency | 4 ms |
| Total per frame | 32 ms |
| Effective FPS | 31.2 |
| Memory usage | ~45 MB RSS |
| Binary size (stripped) | 4.2 MB |

These numbers meet the performance targets defined in the project README.

## 9.6 Exercises

1. **Add frame dropping.** Modify the pipeline to drop frames when processing time exceeds 33 ms. Implement a counter that skips every Nth frame when the system is overloaded.

2. **Implement a direct V4L2 backend.** Replace the `libcamera-still` shell-out with a V4L2 MMAP capture using the `v4l2` crate or direct `ioctl` syscalls. Benchmark the improvement.

3. **Multi-stream pipeline.** Extend the pipeline to support two camera sources (front and rear). Use `Arc<Mutex<FrameIter>>` to share the frame iterator across threads.

4. **Latency dashboard.** Build a real-time latency monitoring dashboard that displays per-component latency for each frame, with p50/p95/p99 lines. Log to a CSV file for offline analysis.

## 9.7 Key Takeaways

- The `FrameIter` closure abstraction unifies camera, video file, and directory sources under a single interface.
- Camera capture currently shells out to `libcamera-still`; a native V4L2 implementation would reduce frame acquisition time by 5x.
- The pipeline runs a tight loop: get frame → detect → track → analyze → alert. No frame skipping is implemented yet.
- The latency budget (40 ms total, 25 fps minimum) is enforced through benchmarking and CI checks.
- Hot-path allocation is forbidden. Current violations are isolated to debug-only features.
- Release builds use LTO and single codegen unit for maximum single-threaded performance.

In Part IV, we dive into tracking — the Kalman filter, Deep SORT, and the geometric computations that give the system temporal awareness across frames.
