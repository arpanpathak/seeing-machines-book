# Chapter 10: Kalman Filters & State Estimation

> *"Prediction is very difficult, especially about the future."*  -  Niels Bohr

The Kalman filter is the mathematical workhorse of multi-object tracking. It solves a deceptively simple problem: given a sequence of noisy measurements (bounding boxes from YOLO), estimate the true state of each object (position, velocity, dimensions) and predict where it will be in the next frame.

This chapter derives the Kalman filter from first principles, implements it in Rust, and integrates it into the CivicSense tracking pipeline. If you only read one chapter on mathematics, read this one. The Kalman filter is the most beautiful algorithm in engineering.

## 10.1 The Problem: Seeing Through Noise

Your YOLO detector produces bounding boxes. But each box has noise: the detector might jitter by a few pixels between frames, occasionally miss a detection, or momentarily swap two objects with similar positions.

The Kalman filter addresses this by maintaining a **belief state**  -  a Gaussian probability distribution over the true state of each tracked object. At each frame, it:

1. **Predicts** where the object should be now, based on where it was before and how it was moving.
2. **Updates** that prediction with the new measurement (the YOLO detection), weighted by how much we trust each source of information.

The result: a smooth, filtered trajectory that is more accurate than either the prediction or the measurement alone.

## 10.2 The Mathematical Derivation

### 10.2.1 The State Space

We model each tracked object with an 8-dimensional state vector:

\\[\mathbf{x} = [c\_x, c\_y, w, h, v\_x, v\_y, v\_w, v\_h]^T\\]

where:
- \\( (c\_x, c\_y) \\) is the bounding box center in pixels.
- \((w, h)\) is the width and height in pixels.
- \\( (v\_x, v\_y, v\_w, v\_h) \\) are the velocities (rate of change per frame).

The state evolves according to a linear **process model**:

\\[\mathbf{x}\_t = \mathbf{F} \cdot \mathbf{x}\_{t-1} + \mathbf{w}\_t\\]

where \\( \mathbf{F} \\) is the state transition matrix and \\( \mathbf{w}\_t \sim \mathcal{N}(0, \mathbf{Q}) \\) is process noise.

For a constant-velocity model (the default in CivicSense and Deep SORT):

\\[\mathbf{F} = \begin{bmatrix}
1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}\\]

This matrix says: the new position = old position + velocity (dt = 1 frame). The new velocity = old velocity (constant velocity assumption).

### 10.2.2 The Measurement Model

The measurement (YOLO detection) is a 4-element vector:

\\[\mathbf{z} = [c\_x, c\_y, w, h]^T\\]

The measurement model relates the state to the measurement:

\\[\mathbf{z}\_t = \mathbf{H} \cdot \mathbf{x}\_t + \mathbf{v}\_t\\]

where \\( \mathbf{H} \\) extracts the position components (first 4 elements) from the state:

\\[\mathbf{H} = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0
\end{bmatrix}\\]

And \\( \mathbf{v}\_t \sim \mathcal{N}(0, \mathbf{R}) \\) is measurement noise.

### 10.2.3 The Full Kalman Equations

The Kalman filter proceeds in two steps at each frame:

**Predict step:**

\\[\hat{\mathbf{x}}\_{t|t-1} = \mathbf{F} \cdot \hat{\mathbf{x}}\_{t-1|t-1}\\]

\\[\mathbf{P}\_{t|t-1} = \mathbf{F} \cdot \mathbf{P}\_{t-1|t-1} \cdot \mathbf{F}^T + \mathbf{Q}\\]

Where \\( \mathbf{P} \\) is the state covariance matrix  -  our uncertainty about the state. The predict step increases uncertainty (adds \\( \mathbf{Q} \\)).

**Update step:**

\\[\mathbf{K}\_t = \mathbf{P}\_{t|t-1} \cdot \mathbf{H}^T \cdot (\mathbf{H} \cdot \mathbf{P}\_{t|t-1} \cdot \mathbf{H}^T + \mathbf{R})^{-1}\\]

\\[\hat{\mathbf{x}}\_{t|t} = \hat{\mathbf{x}}\_{t|t-1} + \mathbf{K}\_t \cdot (\mathbf{z}\_t - \mathbf{H} \cdot \hat{\mathbf{x}}\_{t|t-1})\\]

\\[\mathbf{P}\_{t|t} = (\mathbf{I} - \mathbf{K}\_t \cdot \mathbf{H}) \cdot \mathbf{P}\_{t|t-1}\\]

Where \\( \mathbf{K}\_t \\) is the **Kalman gain**  -  it determines how much we trust the measurement vs the prediction:
- If measurement noise \\( \mathbf{R} \\) is large, \\( \mathbf{K}\_t \\) is small, and we trust the prediction more.
- If process noise \\( \mathbf{Q} \\) is large (our model is uncertain), \\( \mathbf{K}\_t \\) is larger, and we trust the measurement more.

### 10.2.4 The Scalar-Gain Approximation

The full Kalman update requires inverting a \\( 4 \times 4 \\) matrix \\( (\mathbf{H}\mathbf{P}\mathbf{H}^T + \mathbf{R}) \\). On edge hardware, this inversion is expensive and can introduce numerical instability.

The CivicSense Kalman filter uses a **scalar-gain approximation**: instead of computing the full Kalman gain matrix, we update each state dimension independently:

**Innovation:**

\\[y\_i = z\_i - H\_i \cdot \mathbf{x} \quad \text{for } i = 0, 1, 2, 3\\]

**Scalar gain:**

\\[K\_i = \frac{P\_{i,i}}{P\_{i,i} + R} \quad \text{for } i = 0, 1, 2, 3\\]

**Update:**

\\[x\_i \leftarrow x\_i + K\_i \cdot y\_i\\]

\\[P\_{i,i} \leftarrow (1 - K\_i) \cdot P\_{i,i}\\]

This approximation assumes the state variables are uncorrelated (off-diagonal covariances are zero). In practice, bounding box center \\( (c\_x, c\_y) \\) and dimensions \((w, h)\) are approximately independent for typical traffic scenes, so the approximation is good.

## 10.3 The Rust Implementation

### 10.3.1 The KalmanFilter Struct

```rust
struct KalmanFilter {
    /// State mean: 8-element vector [cx, cy, w, h, vx, vy, vw, vh].
    mean: [f32; 8],
    /// State covariance: 8×8 matrix stored flattened row-major (64 elements).
    /// Only the diagonal is actively maintained.
    cov: [f32; 64],
}
```

Note: we store the full \\( 8 \times 8 \\) matrix (64 elements) as a flat array for cache efficiency. The \\( n \times n \\) matrix stored row-major means element \((i,j)\) is at index \\( i \times n + j \\). The diagonal element \\( P\_{i,i} \\) is at index \\( i \times 9 \\) (since \\( i \times 8 + i = i \times 9 \\)).

### 10.3.2 Initialization

```rust
impl KalmanFilter {
    fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        let cx = (x1 + x2) / 2.0;
        let cy = (y1 + y2) / 2.0;
        let w = (x2 - x1).abs();
        let h = (y2 - y1).abs();
        let mean = [cx, cy, w, h, 0.0, 0.0, 0.0, 0.0];
        
        // Initial covariance: high uncertainty for velocity terms
        let mut cov = [0.0f32; 64];
        for i in 0..4 {
            cov[i * 9] = P_INIT;  // Position terms: covariance = P_INIT
        }
        for i in 4..8 {
            cov[i * 9] = P_INIT * 100.0;  // Velocity terms: higher uncertainty
        }
        
        Self { mean, cov }
    }
}
```

The velocity terms start with 100x the position uncertainty because we have no information about velocity at initialization. The filter will learn the velocity over the first few frames (after `n_init = 3` matches, the velocity estimates stabilize).

### 10.3.3 Predict

```rust
fn predict(&mut self) {
    // x += velocity (in place)
    self.mean[0] += self.mean[4];  // cx += vx
    self.mean[1] += self.mean[5];  // cy += vy
    self.mean[2] += self.mean[6];  // w  += vw
    self.mean[3] += self.mean[7];  // h  += vh

    // P += Q (add process noise to diagonal)
    for i in 0..8 {
        self.cov[i * 9] += Q_VAR;
    }
}
```

The predict step:
1. Adds velocity to position (constant-velocity model). Note: dt is implicitly 1 frame. A full implementation would multiply velocity by the actual time delta.
2. Increases covariance by the process noise \\( \mathbf{Q} \\). This represents the uncertainty added by our simplified motion model  -  vehicles can accelerate, brake, or turn, and our constant-velocity model cannot capture that.

### 10.3.4 Update

```rust
fn update(&mut self, x1: f32, y1: f32, x2: f32, y2: f32) {
    let cx = (x1 + x2) / 2.0;
    let cy = (y1 + y2) / 2.0;
    let w = (x2 - x1).abs();
    let h = (y2 - y1).abs();
    let z = [cx, cy, w, h];

    // Innovation: y = z - H * x  (H extracts first 4 elements)
    let y0 = z[0] - self.mean[0];
    let y1 = z[1] - self.mean[1];
    let y2 = z[2] - self.mean[2];
    let y3 = z[3] - self.mean[3];

    // Scalar-gain approximation
    for i in 0..4 {
        let p = self.cov[i * 9];
        let gain = p / (p + R_VAR);
        self.mean[i] += gain * [y0, y1, y2, y3][i];
        self.cov[i * 9] *= 1.0 - gain;
    }
}
```

The update step:
1. Computes the **innovation** (residual): difference between what we measured and what we predicted.
2. Computes the **scalar gain** for each position dimension.
3. Updates the state mean: pulls it toward the measurement by the gain-weighted innovation.
4. Decreases the covariance: the measurement has reduced our uncertainty.

## 10.4 Tuning the Filter

The Kalman filter has three key parameters that must be tuned for the traffic domain:

### Q_VAR: Process Noise (Default: 0.01)

Controls how much we trust the constant-velocity model. Higher values = more uncertainty in the model = filter relies more on measurements. In traffic surveillance:
- **Highway**: Vehicles move predictably, low Q_VAR (~0.005) gives smoother tracks.
- **Intersections**: Vehicles accelerate/brake/turn unpredictably, higher Q_VAR (~0.05) gives faster response.

### R_VAR: Measurement Noise (Default: 0.1)

Controls how much we trust each YOLO detection. Higher values = noisier measurements = filter relies more on the motion model. Factors affecting R_VAR:
- **High confidence detections (conf > 0.9)**: Lower R_VAR (0.05)  -  trust the measurement.
- **Low confidence detections (conf < 0.5)**: Higher R_VAR (0.2)  -  the box is likely noisy.
- **Small objects (far away)**: Higher R_VAR (0.15)  -  small boxes have more relative noise.
- **Large objects (close)**: Lower R_VAR (0.08)  -  large boxes are more stable.

The current implementation uses a single R_VAR for all detections. An adaptive version would scale R_VAR based on detection confidence and object size.

### P_INIT: Initial Covariance (Default: 10.0)

Controls how uncertain we are about a new track's state. Higher values = faster initial adaptation but more jitter in the first few frames. The default 10.0 means the initial position uncertainty is ~3 pixels (standard deviation = \\( \sqrt{10.0} \approx 3.2 \\) px), which is reasonable for a new detection.

## 10.5 Numerical Stability Considerations

The Kalman filter is famous for numerical instability in its naive form. The covariance matrix \\( \mathbf{P} \\) must remain symmetric positive-definite. Due to floating-point arithmetic, it can drift into asymmetry or lose positive definiteness over time.

The CivicSense implementation avoids these issues through the scalar-gain approximation (which only operates on the diagonal) and the use of `f32` (which, while less precise than `f64`, is sufficient for pixel-level tracking and avoids the larger memory footprint).

For reference, a numerically robust implementation would enforce:

```rust
// Enforce symmetry: P = (P + P^T) / 2
for i in 0..8 {
    for j in 0..i {
        let avg = (self.cov[i * 8 + j] + self.cov[j * 8 + i]) / 2.0;
        self.cov[i * 8 + j] = avg;
        self.cov[j * 8 + i] = avg;
    }
}

// Check positive definiteness (simplified: all diagonal elements > 0)
for i in 0..8 {
    debug_assert!(self.cov[i * 9] > 0.0, "Covariance diagonal must be positive");
}
```

This is documented in the code as an invariant:

```rust
/// INVARIANT: `covariance` must always be symmetric positive-definite.
/// Violating this produces incorrect (and possibly NaN) state estimates.
```

## 10.6 The Capstone Connection: Tracking in CivicSense

The Kalman filter is embedded in the `Track` struct:

```rust
pub struct Track {
    pub track_id: u64,
    pub bbox: (f32, f32, f32, f32),
    pub age: u32,
    pub is_confirmed: bool,
    kalman: KalmanFilter,
    time_since_update: u32,
    hits: u32,
}
```

At each frame:

```rust
pub fn predict(&mut self) {
    self.kalman.predict();
    self.bbox = self.kalman.bbox();  // Update predicted bbox from state
    self.age += 1;
    self.time_since_update += 1;
}

pub fn update(&mut self, detection: &Detection) {
    self.kalman.update(detection.x1, detection.y1, detection.x2, detection.y2);
    self.bbox = self.kalman.bbox();
    self.time_since_update = 0;
    self.hits += 1;
}
```

The `predict()` is called for every track at the start of each frame (before association). The `update()` is called only for tracks that matched a detection. Tracks that did not match a detection do not update their Kalman state  -  they continue predicting with increasing uncertainty until they exceed `max_age` and are removed.

## 10.7 Exercises

1. **Implement the full matrix Kalman filter.** Replace the scalar-gain approximation with the full Kalman gain computation (including the \\( 4 \times 4 \\) matrix inversion). Compare tracking accuracy and latency against the scalar version on a benchmark dataset.

2. **Kalman filter visualization.** Track a single object through 100 frames of a video. Plot the predicted vs measured vs filtered trajectory for each coordinate. The filtered trajectory should be smoother than both.

3. **Parameter sensitivity.** Vary Q_VAR from 0.001 to 1.0 and measure the tracking MOTA (Multiple Object Tracking Accuracy) on a validation set. Find the optimal value for highway vs intersection scenes.

4. **Add adaptive noise.** Modify the update step to compute R_VAR from detection confidence: `r_var = R_BASE * (1.0 / detection.confidence)`. Measure the impact on tracking smoothness and latency.

## 10.8 Key Takeaways

- The Kalman filter combines a motion model (prediction) with noisy measurements (YOLO detections) to produce a smooth, optimal state estimate.
- The state vector is 8-dimensional: position and velocity for each of 4 bounding box parameters.
- The scalar-gain approximation avoids matrix inversion at the cost of ignoring cross-correlations between state variables.
- The filter has three tuning parameters (Q_VAR, R_VAR, P_INIT) that control the tradeoff between smoothness and responsiveness.
- Numerical stability requires the covariance matrix to remain symmetric positive-definite.
- The predict-update cycle integrates naturally into the per-frame tracking loop.

In Chapter 11, we build the full Deep SORT tracker around this Kalman filter, adding IoU-based association, track lifecycle management, and confirmed/tentative state transitions.
