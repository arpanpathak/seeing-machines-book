# Chapter 6: Advanced Training, Quantization & ONNX Export

> *"A model in PyTorch is a prototype. A model in ONNX is a product."*

The gap between training and deployment is a chasm filled with precision formats, runtime dependencies, and surprising numerical differences. This chapter builds the bridge: we take a trained PyTorch YOLO model and convert it into an ONNX file that can run on edge devices with no Python dependency, minimal memory, and accelerated inference.

## 6.1 Why ONNX? The Interoperability Standard

Open Neural Network Exchange (ONNX) is an open format for representing machine learning models. It is the assembly language of deep learning — a low-level, hardware-agnostic representation of the computation graph.

The key properties that make ONNX the right choice for CivicSense:

1. **No Python dependency** — ONNX models are loaded and run by the ONNX Runtime, a C++ library with Rust bindings (`ort`). The edge device does not need PyTorch, torchvision, or even Python itself.

2. **Graph optimizations** — The ONNX Runtime applies dozens of graph-level optimizations: constant folding, operator fusion (combining Conv + BatchNorm + SiLU into a single kernel), and memory planning. These optimizations can reduce inference latency by 30-50% compared to PyTorch eager mode.

3. **Hardware acceleration** — ONNX Runtime supports execution providers for CUDA (NVIDIA GPUs), TensorRT (NVIDIA inference optimization), CoreML (Apple Neural Engine), OpenVINO (Intel), and the Hailo-8L NPU (used in CivicSense's dashcam hardware).

4. **INT8 quantization** — ONNX supports INT8 quantization, which reduces model size by 4x and increases inference speed by 2-3x with minimal accuracy loss.

## 6.2 Exporting YOLO to ONNX

The export process converts the PyTorch model's computation graph into the ONNX intermediate representation.

### 6.2.1 The Export Command

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/train/civicsense/weights/best.pt")

# Export to ONNX
model.export(
    format="onnx",
    imgsz=640,
    half=True,           # FP16 weights (half precision)
    simplify=True,       # ONNX graph simplification
    opset=12,            # ONNX opset version
    dynamic=False,        # Static batch size (batch=1)
)
```

**What `simplify=True` does:** The ONNX graph simplifier applies symbolic reasoning to eliminate redundant operations. For example, `Conv -> BatchNorm -> Mul(1.0) -> Add(0.0)` becomes just `Conv` with the BN weights folded into the convolution weights. This reduces the model file size and eliminates unnecessary computation.

**Why `dynamic=False`:** Dynamic batch size allows the same ONNX model to handle different batch sizes. For edge deployment (single-image inference), static batch = 1 produces a simpler, more aggressively optimized graph.

### 6.2.2 The ONNX File Structure

An ONNX file is a protobuf binary containing:

1. **Graph definition** — The computation graph (nodes, edges, tensor shapes).
2. **Weight tensors** — All learned parameters (convolution weights, BN parameters, biases).
3. **Input/output specifications** — Tensor names, shapes, data types.

For the CivicSense YOLO model, the input is `images` (float32, \\( \text{batch} \times 3 \times 640 \times 640 \\)) and the output is a single tensor of shape \\( \text{batch} \times 11 \times 8400 \\) (11 channels: 4 box coords + 7 class logits, one per anchor).

### 6.2.3 Verifying the ONNX Export

The CivicSense binary includes a validation subcommand that loads the ONNX model, runs a dummy input, and checks the output shape:

```rust
pub fn validate_onnx(path: &Path) -> Result<(), String> {
    let session = ort::session::Session::builder()
        .map_err(|e| format!("ort init: {e}"))?
        .commit_from_file(path)
        .map_err(|e| format!("Failed to load '{:?}': {e}", path))?;

    // Create a dummy input (batch=1, channels=3, height=640, width=640)
    let dummy = ndarray::Array4::<f32>::zeros((1, 3, 640, 640));
    let input_tensor = ort::value::Tensor::from_array(dummy)
        .map_err(|e| format!("tensor creation: {e}"))?;

    let outputs = session
        .run(ort::inputs![input_tensor])
        .map_err(|e| format!("inference: {e}"))?;

    let tensor_ref: ort::value::TensorRef<'_, f32> = outputs[0]
        .downcast_ref()
        .map_err(|e| format!("downcast: {e}"))?;

    let (shape, _data) = tensor_ref
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("extract: {e}"))?;

    log::info!("ONNX model validated. Output shape: {:?}", shape);

    // Expected: [1, 11, 8400] for 7-class + 4-coord
    if shape.len() != 3 || shape[0] != 1 {
        return Err(format!("Unexpected output shape: {:?}", shape));
    }

    Ok(())
}
```

## 6.3 INT8 Quantization: The Edge Device Enabler

A YOLOv11n model in FP32 is approximately 5.5 MB. In INT8, it is ~1.4 MB. The 4x reduction comes from quantizing each weight from 32-bit floating point to 8-bit integer.

### 6.3.1 How Quantization Works

Quantization maps a range of floating-point values \\( [r_{\min}, r_{\max}] \\) to integer values $[0, 255]$:

\\[r = S \cdot (q - Z)\\]

where:
- $r$ is the real (float) value.
- $q$ is the quantized (integer) value.
- $S$ is the scale factor (a float).
- $Z$ is the zero-point (an integer representing the quantized value of 0).

For **per-tensor quantization**, a single $(S, Z)$ pair is used for the entire tensor. For **per-channel quantization**, each output channel has its own $(S, Z)$ pair, which is more accurate but requires more storage.

### 6.3.2 Post-Training Quantization vs Quantization-Aware Training

**Post-training quantization (PTQ):** Take a trained FP32 model and calibrate the quantization parameters using a small calibration dataset. Fast (minutes), but accuracy can degrade by 1-3%.

**Quantization-aware training (QAT):** Simulate quantization during training by inserting fake quantization nodes. The model learns to be robust to quantization errors. Slower (requires retraining), but accuracy loss is typically <0.5%.

For CivicSense, we use PTQ because:
- The accuracy loss (typically 1-2% mAP) is acceptable for the traffic domain.
- Retraining with QAT requires additional GPU time and hyperparameter tuning.
- The model is small enough that INT8 errors do not accumulate catastrophically.

### 6.3.3 Quantizing with ONNX Runtime

```python
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path


def quantize_onnx_model(
    input_path: Path,
    output_path: Path,
    weight_type: QuantType = QuantType.QUInt8
) -> None:
    """Quantize an ONNX model to INT8 using dynamic quantization.
    
    Dynamic quantization computes the quantization ranges on-the-fly
    during inference, based on the actual observed activation values.
    This avoids needing a calibration dataset.
    
    Args:
        input_path: Path to the FP32 ONNX model.
        output_path: Path to save the INT8 quantized model.
        weight_type: Quantization type (QUInt8 or QInt8).
    """
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=weight_type,
        per_channel=True,  # Per-channel quantization for convolutions
        reduce_range=False,  # Use full [0, 255] range
    )
    print(f"Quantized model saved to {output_path}")
    print(f"Size reduction: {input_path.stat().st_size / 1024:.1f} KB -> "
          f"{output_path.stat().st_size / 1024:.1f} KB")
```

**Why dynamic quantization?** Static quantization requires a calibration dataset to determine optimal quantization ranges. Dynamic quantization computes ranges on the fly, which introduces slight overhead but eliminates the calibration step. For edge deployment, we eventually want static quantization (pre-computed ranges for maximum speed), but dynamic quantization is a good first step.

## 6.4 Numerical Accuracy: The FP32 vs INT8 Gap

The quantization process introduces noise. A typical FP32 to INT8 conversion changes the model's outputs by about 1-3% in relative terms. The impact on detection accuracy depends on:

1. **The layer type**: Convolutions are relatively robust to quantization (the spatial averaging smooths out errors). Batch normalization folds during export help by centering activations near zero. Fully-connected layers are more sensitive.

2. **The activation distribution**: If activations are uniformly distributed across the full range, quantization works well. If activations are concentrated in a narrow range (e.g., always near 0.1), the effective precision is much lower because only a few quantization levels are used.

3. **The model size**: Smaller models (like YOLOv11n) are more sensitive to quantization than larger models because each parameter represents a greater fraction of the model's capacity.

To validate quantization accuracy:

```bash
# Run inference with FP32 model
civicsense train validate --model runs/train/civicsense/weights/best.onnx

# Run inference with INT8 model
civicsense train validate --model runs/train/civicsense/weights/best-int8.onnx
```

Compare the output tensors for the same input. A root-mean-square error (RMSE) below \\( 5 \times 10^{-3} \\) across all 8400 anchors indicates a successful quantization.

## 6.5 Model Optimization: Beyond Quantization

### 6.5.1 Graph Surgery

The ONNX graph can be further optimized by:

- **Constant folding**: Pre-compute subgraphs that depend only on constant inputs.
- **Operator fusion**: Combine adjacent operations into a single kernel (e.g., Conv + BN + SiLU → fused kernel).
- **Dead code elimination**: Remove operations whose outputs are never used.
- **Memory layout optimization**: Reorder memory accesses for cache efficiency.

The ONNX Runtime applies all of these automatically.

### 6.5.2 NMS Integration

In the current CivicSense pipeline, NMS runs in Rust on the CPU, outside the ONNX model. But ONNX Runtime supports custom operators, and some deployers fuse NMS into the ONNX graph to avoid copying tensors between CPU and GPU memory.

The tradeoff: fusing NMS into ONNX reduces host-device transfers but makes the model less portable (NMS is not a standard ONNX operator). For the edge deployment scenario (CPU-only inference), separating NMS into Rust code is simpler and equally performant.

## 6.6 The Capstone Connection: From PyTorch to ONNX to Rust

The complete export pipeline for CivicSense is:

```
best.pt (PyTorch, FP32, 5.5 MB)
    │
    ├── YOLO.export(format="onnx")
    │
    ├── best.onnx (ONNX, FP32, 5.5 MB)
    │
    ├── onnxruntime.quantization.quantize_dynamic()
    │
    └── best-int8.onnx (ONNX, INT8, 1.4 MB)
         │
         └── Loaded by Rust `ort` crate
              │
              └── Inference on Raspberry Pi 5 / Hailo-8L
```

This pipeline is automated in the Makefile:

```makefile
## Train YOLO model on cloud GPU (run this on the VM)
train-run:
    $(CARGO) run --release -- train run --data configs/dataset.yaml --epochs 100

## Validate an exported ONNX model with ort
train-validate:
    $(CARGO) run --release -- train validate
```

The `train run` subcommand executes the Python training script (via shell), then the `train validate` subcommand loads the resulting ONNX file and verifies it with the Rust inference engine. This creates a tight feedback loop: train → export → verify → deploy.

## 6.7 Exercises

1. **Export and compare.** Train a YOLO model, export to ONNX, quantize to INT8. Compare the output of FP32 vs INT8 on 100 validation images. Compute the mean IoU between FP32 and INT8 detections.

2. **Quantization sensitivity analysis.** Quantize the model with different settings (per-tensor vs per-channel, UINT8 vs INT8, dynamic vs static). Measure mAP for each variant on the validation set.

3. **ONNX graph visualization.** Use Netron (a model visualizer) to open the exported ONNX file. Identify the backbone, neck, and detection head sections of the graph.

4. **Latency benchmark.** Write a Rust benchmark that loads both FP32 and INT8 models and runs 1000 inference iterations. Report the mean, median, and 99th percentile latency for each.

## 6.8 Key Takeaways

- ONNX is the bridge between training (Python/PyTorch) and deployment (Rust/edge). It eliminates the Python dependency at inference time.
- INT8 quantization reduces model size by 4x and increases inference speed by 2-3x with 1-2% mAP loss.
- The ONNX Runtime applies automatic graph optimizations (operator fusion, constant folding) that improve performance beyond what is possible in PyTorch eager mode.
- Per-channel quantization preserves more accuracy than per-tensor quantization, especially for convolutional layers.
- Always validate the ONNX export by comparing FP32 and quantized model outputs on real data.

In Part III, we shift from training to deployment — building the Rust inference engine that loads this ONNX model and runs it on edge hardware.
