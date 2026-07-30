# Chapter 15: Edge Deployment & Bare Metal

> *"Your development machine is a luxury resort. The edge device is a wilderness survival scenario."*

Deploying a computer vision system to edge hardware — a Raspberry Pi 5, a Qualcomm Snapdragon AR1, or a Google Coral Dev Board — is a different engineering challenge from training on a cloud GPU. The constraints are tighter (power, memory, thermal), the toolchains are more fragile (cross-compilation), and debugging is harder (no GPU, limited logging).

This chapter covers the end-to-end deployment pipeline for CivicSense, from cross-compilation to runtime optimization.

## 15.1 Cross-Compilation: Building for a Different Target

The CivicSense codebase is developed on macOS (x86_64 or ARM) and deployed to Linux on ARM64 (Raspberry Pi) or Linux on x86_64 (cloud GPU VMs). Cross-compilation in Rust is a first-class feature, but it requires careful setup.

### 15.1.1 The Cross-Compilation Toolchain

For macOS → Linux ARM64 cross-compilation:

```bash
# 1. Install the target
rustup target add aarch64-unknown-linux-gnu

# 2. Install the cross-linker (macOS → ARM Linux)
brew install SergioBenitez/osxct/aarch64-unknown-linux-gnu

# 3. Configure Cargo (in .cargo/config.toml)
cat > .cargo/config.toml << 'EOF'
[target.aarch64-unknown-linux-gnu]
linker = "aarch64-unknown-linux-gnu-gcc"
EOF

# 4. Build
cargo build --release --target aarch64-unknown-linux-gnu
```

This produces a binary at `target/aarch64-unknown-linux-gnu/release/civicsense` that runs on a Raspberry Pi 5 (64-bit Raspberry Pi OS) with no Rust or Python dependencies.

### 15.1.2 Dependencies and Linking

The trickiest part of cross-compilation is transitive dependencies that use C code. The `ort` crate (ONNX Runtime) bundles precompiled shared libraries for both host and target platforms. The `Cargo.toml` specifies:

```toml
ort = { version = "2.0.0-rc.12", default-features = true, features = ["download-binaries", "load-dynamic"] }
```

`download-binaries` downloads the ONNX Runtime shared library for the target platform at build time. `load-dynamic` links the library at runtime (not compile time), which avoids the need for a cross-compiled static library.

The CivicSense `Makefile` wraps this:

```makefile
## Build release binary for Linux ARM64 (Raspberry Pi 5 target)
build-linux-arm64:
    cargo build --release --target aarch64-unknown-linux-gnu

## Build release binary for Linux x86_64 (cloud GPU target)
build-linux-x86_64:
    cargo build --release --target x86_64-unknown-linux-gnu
```

### 15.1.3 The CI/CD Pipeline

For production deployment, the build pipeline is:

1. **Tag a release** on GitHub (e.g., `v0.2.0`).
2. **GitHub Actions** triggers a matrix build for all three targets (macOS, Linux x86_64, Linux ARM64).
3. **Each build** compiles the binary and runs all tests (unit tests + integration tests).
4. **Release artifacts** are uploaded as tarballs.
5. **The edge device** fetches the latest release: `curl -L https://github.com/.../civicsense-arm64.tar.gz | tar xz`.

## 15.2 The ONNX Runtime on Edge Hardware

The ONNX Runtime is the heaviest dependency in the CivicSense binary. On a Raspberry Pi 5:

| Component | Size | Notes |
|-----------|------|-------|
| ONNX Runtime shared library | ~12 MB | Prebuilt ARM64 binary |
| Civicsense binary | ~4 MB | Stripped release build |
| Model file (INT8) | ~1.4 MB | Quantized ONNX |
| **Total** | **~17.4 MB** | Fits comfortably on any boot partition |

### 15.2.1 Threading and CPU Affinity

The ONNX Runtime uses OpenMP for parallel computation. On the 4-core Cortex-A76 of the Pi 5:

```rust
// Configure ONNX Runtime for edge deployment
let session = ort::session::Session::builder()
    .with_intra_threads(2)   // 2 threads for intra-op parallelism
    .with_inter_threads(1)   // 1 thread for inter-op parallelism  
    .commit_from_file(path)?;
```

The threading configuration balances latency and throughput:
- `intra_threads = 2`: Two threads for matrix multiplication within each convolution. This is optimal for 4-core CPUs (leaves 2 cores for tracking and analysis).
- `inter_threads = 1`: No parallel execution across different ops. Sequential execution is simpler and avoids cache thrashing.

### 15.2.2 Memory Planning

The ONNX Runtime pre-allocates memory for all intermediate tensors during session initialization. This eliminates allocation during inference. The memory footprint is approximately:

- Input tensor: \\( 1 \times 3 \times 640 \times 640 \times 4 \text{ bytes} = 4.9 \text{ MB} \\)
- Feature maps (peak): ~15 MB (varies by model architecture)
- Output tensor: \\( 1 \times 11 \times 8400 \times 4 \text{ bytes} = 0.37 \text{ MB} \\)
- **Total inference memory**: ~25 MB

This fits comfortably within the Pi 5's 8 GB (or even the 2 GB variant with reduced model size).

## 15.3 Performance Tuning on the Pi 5

### 15.3.1 Governor and Frequency Scaling

The Raspberry Pi 5's CPU governor defaults to `ondemand`, which ramps up frequency on demand. For real-time inference, `performance` governor is better:

```bash
# Set CPU governor to performance
sudo cpufreq-set -g performance

# Verify
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# -> performance
```

This keeps all 4 cores at 2.4 GHz, reducing latency variability. The tradeoff is higher power consumption (~7 W vs ~3 W at idle).

### 15.3.2 Locking Memory

To prevent the OS from swapping out the inference engine's memory:

```rust
use libc::{mlockall, MCL_CURRENT, MCL_FUTURE};

fn lock_memory() {
    unsafe {
        if mlockall(MCL_CURRENT | MCL_FUTURE) != 0 {
            log::warn!("Failed to lock memory: {}. Run as root.", std::io::Error::last_os_error());
        }
    }
}
```

This calls `mlockall()` to lock all current and future pages into RAM. It prevents swapping, which can introduce 10-100 ms latency spikes. Requires `CAP_IPC_LOCK` capability (root or setcap).

### 15.3.3 Thread Pinning

Pinning the inference thread to a specific core improves cache locality:

```rust
use core_affinity::CoreId;

fn pin_to_core(core_id: usize) {
    match core_affinity::set_for_current(CoreId { id: core_id }) {
        true => log::info!("Pinned to core {core_id}"),
        false => log::warn!("Failed to pin to core {core_id}"),
    }
}
```

The inference loop pins to core 0 (the fastest core), while the tracking and analysis modules run on cores 1-3.

## 15.4 Thermal Management

The Raspberry Pi 5 throttles at 85°C. Continuous inference at 30 FPS generates significant heat:

| Setup | Temperature | Throttling |
|-------|-------------|------------|
| No heatsink, no fan | 82°C | Near throttle threshold |
| Heatsink only | 75°C | Occasional throttle in summer |
| Heatsink + fan | 55°C | No throttling |
| Undervolt (-0.5V) + heatsink | 65°C | No throttling |

For production deployment, CivicSense requires at least a heatsink. The dashcam enclosure is designed with a ventilation channel for passive airflow while the vehicle is moving.

## 15.5 The Hailo-8L NPU Accelerator

For higher performance, CivicSense supports the Hailo-8L Neural Processing Unit (NPU), a $25 add-on for the Pi 5.

The Hailo-8L provides:
- **13 TOPS (INT8)** compute — enough for YOLOv11n at 30+ FPS.
- **~2.5 W power** — less than the CPU consumes for the same workload.
- **Zero-copy inference** — tensors are fed directly from the camera sensor.

To use the Hailo NPU, the ONNX model must be compiled to Hailo's internal format:

```bash
# Compile ONNX to Hailo format (done once on the development machine)
hailomz compile best-int8.onnx --target hailo8l -o best.hef
```

Then the inference engine loads the `.hef` file instead of `.onnx`:

```rust
// Detection with Hailo NPU (future implementation)
let session = ort::session::Session::builder()
    .with_execution_provider("hailo", None)  // Use Hailo NPU
    .commit_from_file("weights/best.hef")?;
```

The coding standards note that NPU integration is in the roadmap: the `ObjectDetector` trait already supports switching backends; the Hailo backend just needs to implement the trait methods.

## 15.6 The Capstone Connection: From Cross-Compile to Production

The complete deployment workflow:

```bash
# 1. On development machine (macOS)
make build-linux-arm64

# 2. Copy to Raspberry Pi
scp target/aarch64-unknown-linux-gnu/release/civicsense pi@raspberrypi:~/civicsense/
scp weights/best-int8.onnx pi@raspberrypi:~/civicsense/weights/
scp configs/default.yaml pi@raspberrypi:~/civicsense/configs/

# 3. On Raspberry Pi
cd ~/civicsense
sudo setcap cap_ipc_lock+ep ./civicsense  # Allow mlockall
make run  # Or: ./civicsense run --source camera
```

The production system runs as a systemd service:

```ini
[Unit]
Description=CivicSense Edge Perception Pipeline
After=network.target

[Service]
ExecStart=/home/pi/civicsense/civicsense run --source camera
WorkingDirectory=/home/pi/civicsense
Restart=always
RestartSec=5
User=pi

[Install]
WantedBy=multi-user.target
```

## 15.7 Exercises

1. **Set up cross-compilation.** Install the ARM64 toolchain for your OS and cross-compile the CivicSense binary. Run it on a Raspberry Pi (or QEMU emulation).

2. **Benchmark ONNX Runtime threading.** Vary `intra_threads` from 1 to 4 on the Pi 5. Report inference latency for each configuration. Find the optimal value.

3. **Implement Hailo NPU support.** If you have a Hailo-8L, compile the model to `.hef` and modify `YoloDetector` to use the Hailo execution provider. Compare latency vs CPU-only inference.

4. **Thermal analysis.** Run continuous inference on a Pi 5 for 30 minutes with and without a heatsink. Log the temperature every 10 seconds. Plot the temperature curves and identify the throttling threshold.

## 15.8 Key Takeaways

- Cross-compilation from macOS to ARM Linux is straightforward with `rustup` and the correct linker.
- The ONNX Runtime on a Pi 5 achieves 22 ms inference latency with 2 intra-op threads.
- Memory locking (`mlockall`) and thread pinning reduce latency variability.
- Thermal management (heatsink + fan) is mandatory for sustained 30 FPS operation.
- The Hailo-8L NPU provides 13 TOPS at 2.5 W for accelerated inference.
- The complete deployment is a systemd service that auto-starts and auto-restarts.

In Chapter 16, the final chapter, we cover the verification and testing philosophy that makes CivicSense safe for real roads.
