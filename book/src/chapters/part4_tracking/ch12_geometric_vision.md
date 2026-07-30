# Chapter 12: Geometric Computer Vision & Sensor Fusion

> *"The world is 3D. Your camera sensor is 2D. The gap between them is geometry."*

Every alert in CivicSense requires converting pixel coordinates into real-world meaning. A stop sign that is 50 pixels wide in the image might be 10 meters away or 50 meters away, depending on the camera's focal length and the sign's physical size. A vehicle moving 10 pixels per frame across the image might be traveling at 5 mph or 50 mph, depending on the distance.

This chapter covers the geometric computations in `src/utils/geometry.rs` that bridge this gap: pinhole camera projection, distance estimation, relative velocity, and the low-pass filters that smooth out the noise.

## 12.1 The Pinhole Camera Model: A Refresher

The pinhole camera model describes the geometric relationship between 3D points in the world and their 2D projections onto the camera sensor. The core equation:

$$u = f_x \cdot \frac{X}{Z} + u_0, \quad v = f_y \cdot \frac{Y}{Z} + v_0$$

For the CivicSense use case (a forward-facing dashcam with a known camera configuration), we invert this relationship to estimate real-world distances from pixel measurements.

### 12.1.1 Distance from Known Width

If we know the real-world width $W$ of an object (e.g., a vehicle is approximately 1.8 meters wide, a stop sign is 0.75 meters in diameter), then:

$$Z = \frac{f \cdot W}{w}$$

where $w$ is the pixel width of the object's bounding box.

This is implemented in `geometry.rs`:

```rust
pub fn estimate_distance(pixel_width: f32, real_width: f32, focal_length: f32) -> f32 {
    if pixel_width <= 0.0 {
        return f32::MAX;
    }
    (focal_length * real_width) / pixel_width
}
```

**What $f = 650$ means in physical terms.** The default focal length of 650 pixels for a 1280-pixel-wide image corresponds to a horizontal field of view of approximately:

$$\text{FOV} = 2 \cdot \arctan\left(\frac{1280/2}{650}\right) \approx 89^\circ$$

This is a typical wide-angle dashcam lens. The wide field of view captures more of the road scene but introduces perspective distortion at the edges.

### 12.1.2 Limitations of Monocular Depth Estimation

The pinhole method assumes:
1. **The object's real-world width is known.** A vehicle's width varies from 1.5 m (Smart car) to 2.6 m (F-150 pickup). The assumed value of 1.8 m is the average passenger vehicle width in the US. Trucks and buses use different assumed widths.

2. **The object is viewed perpendicularly.** If a vehicle is at a 45-degree angle (e.g., turning), its apparent width is $W \cdot \cos(45^\circ) \approx 0.7W$, making it appear farther than it is.

3. **The focal length is accurately calibrated.** A factory-calibrated camera module may have a slightly different focal length than specified.

These limitations mean distance estimates are approximate, not precise. The estimate is typically accurate to ±20-30% for vehicles viewed from behind (the most common dashcam scenario). For safety-critical alerts (stop sign distance), we maintain safety margins: we use the upper bound of the distance estimate to ensure we alert early rather than late.

## 12.2 Relative Velocity: Motion from Distance Change

The relative velocity of a tracked vehicle is estimated from the change in distance between consecutive frames:

$$V_{\text{rel}} = \frac{Z_{t-1} - Z_t}{\Delta t}$$

This is positive when the object is approaching (distance decreasing) and negative when receding.

```rust
pub fn compute_relative_velocity(prev_distance: f32, curr_distance: f32, dt: f32) -> f32 {
    if dt <= 0.0 { return 0.0; }
    (prev_distance - curr_distance) / dt
}
```

**Why this works:** If a vehicle was 20 m away at frame $t-1$ and is 19 m away at frame $t$ (with $\Delta t = 0.033$ s at 30 FPS):

$$V_{\text{rel}} = \frac{20 - 19}{0.033} \approx 30 \text{ m/s} \approx 67 \text{ mph}$$

This is the closing speed. If the ego vehicle is traveling at 35 mph, the lead vehicle is traveling at 35 - 67 = -32 mph (we are rapidly approaching a slower vehicle).

**Why this is noisy:** The distance estimate itself has ±20% uncertainty. The difference between two uncertain estimates is even noisier. This is why the low-pass filter (Section 12.3) is essential before the velocity estimate is used for alerts.

## 12.3 Low-Pass Filter: Taming the Noise

The low-pass filter smooths noisy measurements by blending the current value with the previous filtered value:

$$y_t = \alpha \cdot x_t + (1 - \alpha) \cdot y_{t-1}$$

```rust
pub fn low_pass_filter(value: f32, prev_value: f32, alpha: f32) -> f32 {
    alpha * value + (1.0 - alpha) * prev_value
}
```

The parameter $\alpha$ (default 0.3) controls the smoothing strength:
- $\alpha = 1.0$: No smoothing (pass-through).
- $\alpha = 0.1$: Heavy smoothing (slow to respond to changes).
- $\alpha = 0.3$: Moderate smoothing (responsiveness ≈ 3 frames to reach 66% of a step change).

**The 66% response time:** For a step change in the input, the filtered output reaches 66% of the step in approximately $1/\alpha$ frames. At $\alpha = 0.3$, this is about 3 frames or 0.1 seconds. This is fast enough for traffic alerts (which require 1-3 seconds of sustained behavior) but smooth enough to eliminate frame-to-frame jitter.

## 12.4 IoU: The Geometry of Overlap

Intersection-over-Union is the fundamental geometric metric for detection and tracking:

$$IoU = \frac{|A \cap B|}{|A \cup B|} = \frac{\text{overlap area}}{\text{total area}}$$

For two axis-aligned bounding boxes, the overlap area is computed from the intersection of their coordinate ranges:

```rust
pub fn compute_iou(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
    let (ax1, ay1, ax2, ay2) = a;
    let (bx1, by1, bx2, by2) = b;

    let ix1 = ax1.max(bx1);
    let iy1 = ay1.max(by1);
    let ix2 = ax2.min(bx2);
    let iy2 = ay2.min(by2);

    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;

    let a_area = (ax2 - ax1) * (ay2 - ay1);
    let b_area = (bx2 - bx1) * (by2 - by1);
    let union = a_area + b_area - inter;

    if union <= 0.0 { 0.0 } else { inter / union }
}
```

In tracking, we use IoU as a **similarity metric** between a predicted track position and a detection. If IoU > 0.3, the detection likely belongs to this track. If IoU < 0.1, it is likely a different object or a false positive.

## 12.5 Bounding Box Format Conversion

The Kalman filter's state vector uses center-size format $(c_x, c_y, w, h)$, while the detector and analysis modules use corner format $(x_1, y_1, x_2, y_2)$. The conversion:

```rust
pub fn bbox_to_cxcywh(x1: f32, y1: f32, x2: f32, y2: f32) -> (f32, f32, f32, f32) {
    let cx = (x1 + x2) / 2.0;
    let cy = (y1 + y2) / 2.0;
    let w = (x2 - x1).abs();
    let h = (y2 - y1).abs();
    (cx, cy, w, h)
}
```

The conversion from center-size back to corner format:

```rust
fn bbox_from_cxcywh(cx: f32, cy: f32, w: f32, h: f32) -> (f32, f32, f32, f32) {
    (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
}
```

This conversion is hidden inside the `KalmanFilter::bbox()` method — no other code needs to know the internal format.

## 12.6 The Bird's Eye View Projection (BEP)

The intersection analyzer uses a Bird's Eye View (BEV) occupancy grid. The BEV projection maps image coordinates $(u, v)$ to ground-plane coordinates $(X, Z)$ under the assumption that all objects lie on a flat ground plane.

Given the camera's height $h$ above the ground and its pitch angle $\theta$ (downward tilt), the mapping is:

$$Z = \frac{h}{\tan(\theta + \arctan(\frac{v - v_0}{f_y}))}$$

$$X = \frac{Z \cdot (u - u_0)}{f_x}$$

where $(u_0, v_0)$ is the principal point (image center) and $(f_x, f_y)$ is the focal length.

This is currently a placeholder in the code — the BEV grid resolution and ahead-distance are configured, but the actual projection requires camera extrinsic calibration (height and pitch angle). The TODO comment in `src/modules/intersection.rs` notes this as future work.

## 12.7 The Capstone Connection: Geometry in the Modules

The geometric utilities are used throughout the analysis modules:

| Function | Used By | Purpose |
|----------|---------|---------|
| `estimate_distance()` | Intersection analyzer, Lane speed analyzer | Convert pixel width to meters |
| `compute_relative_velocity()` | Lane speed analyzer | Estimate vehicle speed from distance change |
| `low_pass_filter()` | Lane speed analyzer | Smooth noisy velocity estimates |
| `compute_iou()` | Deep SORT tracker | Match detections to tracks |
| `bbox_to_cxcywh()` | Kalman filter (implicit) | Internal state representation |

Each function is small, independently testable, and has documented mathematical properties. This follows the coding standard's principle of "one function, one responsibility."

## 12.8 Exercises

1. **Distance error analysis.** For a vehicle of width 1.8 m at a true distance of 30 m, compute the estimated distance if the pixel width is measured with ±2 pixel error. What is the percentage error? Repeat for 10 m, 50 m, 100 m. Where is the error worst?

2. **Focal length calibration.** Take a photo of a checkerboard pattern at a known distance. Measure the pixel width of a known checker square. Compute the actual focal length from: $f = Z \cdot w / W$. Compare to the configured default of 650.

3. **Filter response analysis.** Feed a step function (0, 0, 0, 10, 10, 10, ...) through the low-pass filter with $\alpha = 0.3$. Plot the output. How many samples does it take to reach 90% of the step value?

4. **Implement BEV projection.** Given camera height h = 1.2 m and pitch $\theta = 5^\circ$, write a function that converts $(u, v)$ image coordinates to $(X, Z)$ ground-plane coordinates. Test with a grid of points.

## 12.9 Key Takeaways

- The pinhole camera model converts pixel width to real-world distance, with known limitations (width assumptions, angle effects, calibration errors).
- Relative velocity is estimated from the rate of change of distance, requiring both temporal filtering and measurement smoothing.
- The low-pass filter ($\alpha = 0.3$) provides moderate smoothing with ~0.1 s response time — fast enough for alerts, slow enough to eliminate jitter.
- IoU is the fundamental geometric metric for both detection NMS and tracking association.
- Bounding box format conversion (corner ↔ center-size) bridges the detector and Kalman filter representations.
- BEV projection is planned but requires camera extrinsic calibration to be accurate.

In Chapter 13, we bring everything together in the first capstone module: intersection intelligence.
