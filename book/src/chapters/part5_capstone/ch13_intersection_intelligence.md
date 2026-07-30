# Chapter 13: Intersection Intelligence

> *"Intersections are where roads meet, traffic rules apply, and ADAS systems fail."*

The National Highway Traffic Safety Administration (NHTSA) reports that approximately 40% of all crashes in the United States occur at intersections. "Blocking the box" (entering an intersection when the exit is not clear) and stop sign violations are among the most common causes.

The CivicSense intersection module (`src/modules/intersection.rs`) addresses two specific failure modes:

1. **Stop sign violations**: The driver is approaching a stop sign too fast to stop safely.
2. **Blocked intersections**: The driver is about to enter a congested intersection where they would get stuck.

This chapter walks through the intersection analysis code, from the detection semantics to alert generation.

## 13.1 The IntersectionAlert Enum

The module defines two alert types as a Rust enum:

```rust
#[derive(Debug, Clone)]
pub enum IntersectionAlert {
    StopSignViolation {
        confidence: f32,
        distance_to_stop_line: f32,
        ego_speed: f32,
    },
    BlockedIntersection {
        confidence: f32,
        occupancy_pct: f32,
        distance_to_stop_line: f32,
        ego_speed: f32,
    },
}
```

Each variant carries the data needed to:
- Log the alert with enough context for debugging.
- Drive the voice output system ("Stop sign in 30 feet, you're going 25 mph").
- Escalate to the mesh network if the violation is severe.

Why an enum and not two separate structs? The enum allows the alert priority engine to handle all intersection alerts uniformly, while the variants provide specific data. The `match` expression on the alert type is exhaustive — the compiler ensures every variant is handled.

## 13.2 The IntersectionAnalyzer Structure

```rust
pub struct IntersectionAnalyzer {
    config: IntersectionConfig,
    stop_sign_width_m: f32,    // 0.75 m (US standard)
    frame_width: u32,
    frame_height: u32,
    focal_length: f32,
}
```

The analyzer is stateless between frames — each call to `analyze()` is independent. This is by design: intersection conditions change rapidly (within a second, an intersection can go from clear to blocked), so temporal filtering would add latency to safety-critical alerts.

### 13.2.1 Configuration Parameters

```rust
pub struct IntersectionConfig {
    pub stop_sign_warning_distance: f32,    // 50.0 m
    pub stop_sign_warning_speed: f32,      // 10.0 mph
    pub blocked_intersection_speed: f32,   // 15.0 mph
    pub blocked_distance_to_stop: f32,     // 30.0 m
    pub grid_resolution: f32,              // 0.5 m/cell
    pub grid_ahead_distance: f32,          // 20.0 m
}
```

These parameters are calibrated for US urban streets. For European or Asian markets, they would need adjustment:
- European streets are narrower → shorter warning distances.
- Roundabout-heavy traffic patterns → different intersection semantics.

The values are documented defaults in `configs/default.yaml` and can be overridden per deployment.

## 13.3 Stop Sign Detection

### 13.3.1 The Detection Pipeline

```rust
fn check_stop_signs(
    &self,
    detections: &[Detection],
    ego_speed: f32,
    alerts: &mut Vec<IntersectionAlert>,
) {
    const MIN_CONFIDENCE: f32 = 0.5;
    for sign in detections.iter().filter(|d| d.class_id == 0 && d.confidence >= MIN_CONFIDENCE) {
        let pixel_width = (sign.x2 - sign.x1).abs();
        if pixel_width < 1.0 { continue; }

        let distance = estimate_distance(pixel_width, self.stop_sign_width_m, self.focal_length)
            .clamp(1.0, 200.0);

        if distance <= self.config.stop_sign_warning_distance
            && ego_speed >= self.config.stop_sign_warning_speed
        {
            alerts.push(IntersectionAlert::StopSignViolation {
                confidence: sign.confidence,
                distance_to_stop_line: distance,
                ego_speed,
            });
        }
    }
}
```

**The three filters:**

1. **Class filter**: Only detections with `class_id == 0` (stop_sign) are considered. Traffic lights, crosswalks, and vehicles are ignored for this check.

2. **Confidence filter**: `MIN_CONFIDENCE = 0.5` ensures that low-confidence detections (false positives) do not trigger alerts. In production, this could be lowered for safety-critical scenarios (better to false-alarm on a non-existent stop sign than to miss a real one).

3. **Distance + speed filter**: Both conditions must be met:
   - $Z \leq 50 \text{ m}$: The stop sign is close enough to be relevant.
   - $v_{\text{ego}} \geq 10 \text{ mph}$: The vehicle is moving fast enough that stopping requires active braking.

### 13.3.2 The Distance-Speed Tradeoff

The combination of distance and speed creates a **stopping distance envelope**. At 10 mph, a typical passenger vehicle needs about 15-20 meters to stop under normal braking. At 50 mph, it needs about 75 meters.

The CivicSense defaults (50 m distance, 10 mph speed) are conservative:
- At 10 mph and 50 m distance: the driver has 30+ meters of margin (comfortable stop).
- At 25 mph and 50 m distance: the driver has ~20 meters of margin (requires braking).
- At 35 mph and 50 m distance: the driver is at the edge of the stopping distance (definite alert).

A production deployment might dynamically adjust the distance threshold based on the current stopping distance:

```rust
fn compute_warning_distance(ego_speed_mph: f32) -> f32 {
    // Stopping distance ≈ speed (mph) * 1.5 for normal braking
    // Safety margin: 2x the stopping distance
    ego_speed_mph * 3.0  // 35 mph → 105 ft ≈ 32 m
}
```

## 13.4 Blocked Intersection Detection

### 13.4.1 Occupancy Estimation

```rust
fn check_blocked_intersection(
    &self,
    detections: &[Detection],
    ego_speed: f32,
    alerts: &mut Vec<IntersectionAlert>,
) {
    let vehicles: Vec<&Detection> = detections
        .iter()
        .filter(|d| (3..=5).contains(&d.class_id) && d.confidence > 0.4)
        .collect();

    if vehicles.is_empty() { return; }

    let total_area: f32 = (self.frame_width * self.frame_height) as f32;
    let occupied_area: f32 = vehicles.iter()
        .map(|d| (d.x2 - d.x1) * (d.y2 - d.y1))
        .sum();
    let occupancy_pct = ((occupied_area / total_area) * 100.0).clamp(0.0, 100.0);

    if occupancy_pct > 30.0 && ego_speed >= self.config.blocked_intersection_speed {
        alerts.push(IntersectionAlert::BlockedIntersection {
            confidence: occupancy_pct / 100.0,
            occupancy_pct,
            distance_to_stop_line: self.config.blocked_distance_to_stop,
            ego_speed,
        });
    }
}
```

**The occupancy heuristic** is simple: if vehicles (classes 3-5: car, truck, bus) occupy more than 30% of the frame area, and the ego vehicle is moving at 15+ mph, the intersection is likely blocked.

**Why 30%?** This threshold was determined empirically: a typical intersection fills about 10-15% of the frame with vehicles when traffic is flowing normally. When traffic is backed up through the intersection, vehicles can occupy 40-60% of the frame. The 30% threshold provides a safety margin.

**Why only vehicles (classes 3-5)?** Pedestrians and cyclists (not in the current class set) would also indicate a blocked intersection, but the CivicSense model is not trained for them. For production, the class set could be extended.

### 13.4.2 Limitations of the Occupancy Heuristic

The area-based occupancy metric has known limitations:

1. **Distance bias**: A vehicle 5 meters away occupies much more frame area than a vehicle 50 meters away, even though both block the intersection equally. A BEV projection (mapping to ground-plane coordinates) would fix this.

2. **Perspective distortion**: Vehicles at the edges of a wide-angle lens appear smaller than vehicles at the center, even at the same distance. This undercounts occupancy at the periphery.

3. **No temporal fusion**: The current frame's occupancy is considered in isolation. A vehicle that has been stationary in the intersection for 3 seconds is more concerning than one that is moving through.

These limitations are documented in the code as future work items. The BEV occupancy grid (using `grid_resolution` and `grid_ahead_distance` from the config) would replace the simple area metric with a proper ground-plane projection.

## 13.5 Alert Prioritization

In the full pipeline, alerts from multiple modules are collected and prioritized:

```rust
fn generate_voice_output(alerts: &[Alert]) -> Option<String> {
    // Priority: Safety-critical > Courtesy > Informational
    for alert in alerts {
        match alert {
            Alert::Intersection(IntersectionAlert::StopSignViolation { .. }) => {
                return Some("Stop sign ahead! Brake now.".to_string());
            }
            Alert::Intersection(IntersectionAlert::BlockedIntersection { .. }) => {
                return Some("Intersection blocked. Do not enter.".to_string());
            }
            Alert::LaneSpeed(_) => {
                // Accumulate lane alerts; only speak if persistent
            }
        }
    }
    None
}
```

Stop sign violations always take priority because they have an immediate safety implication. Blocked intersection alerts are second. Lane courtesy reminders are non-critical and can be queued.

## 13.6 Logging and Visualization

Each alert is logged with structured context:

```rust
fn log_intersection_alerts(alerts: &[IntersectionAlert]) {
    for alert in alerts {
        match alert {
            IntersectionAlert::StopSignViolation { confidence, distance_to_stop_line, ego_speed } => {
                log::warn!(
                    "STOP SIGN VIOLATION! conf={:.2}, dist={:.1}ft, speed={:.1}mph",
                    confidence, distance_to_stop_line, ego_speed
                );
            }
            IntersectionAlert::BlockedIntersection { confidence, occupancy_pct, distance_to_stop_line, ego_speed } => {
                log::warn!(
                    "BLOCKED INTERSECTION! conf={:.2}, occupancy={:.1}%, dist={:.1}ft, speed={:.1}mph",
                    confidence, occupancy_pct, distance_to_stop_line, ego_speed
                );
            }
        }
    }
}
```

In `--visualize` mode, the frame overlay shows a colored banner:

```rust
if !intersection_alerts.is_empty() {
    visualization::draw_alert_text(
        &mut viz, self.frame_width, self.frame_height, "STOP SIGN VIOLATION",
    );
}
```

## 13.7 Testing the Intersection Module

The module has unit tests covering three scenarios:

1. **No detections → no alerts**: Verifies the module is silent when YOLO sees nothing.
2. **No speed → no violation**: A stop sign is detected but the vehicle is stopped (0 mph) → no alert.
3. **High occupancy → blocked intersection**: Two large vehicle detections covering >30% of the frame with ego speed > 15 mph → blocked intersection alert.

```rust
#[test]
fn test_intersection_occupancy_detection() {
    let cfg = Config::default();
    let mut analyzer = IntersectionAnalyzer::new(&cfg, 1280, 720);
    let dets = vec![
        make_det(3, 0.0, 0.0, 640.0, 360.0, 0.9),
        make_det(3, 640.0, 0.0, 1280.0, 360.0, 0.9),
    ];
    let alerts = analyzer.analyze(&dets, 20.0, 0.033);
    let blocked = alerts.iter().any(|a| {
        matches!(a, IntersectionAlert::BlockedIntersection { .. })
    });
    assert!(blocked, "Expected BlockedIntersection alert");
}
```

This property-based testing approach (test with explicit conditions, verify the expected output) is applied to every alert condition.

## 13.8 Exercises

1. **Add temporal filtering to stop sign detection.** Implement a counter that requires a stop sign to be detected in 3 out of 5 consecutive frames before alerting. Measure the reduction in false positives.

2. **Extend occupancy to BEV.** Replace the area-based occupancy metric with a BEV grid occupancy. Use the pinhole model to project vehicle positions onto the ground plane and count occupied grid cells.

3. **Calibrate for your camera.** Measure the actual focal length of your camera and update the config. Measure the real-world width of stop signs in your area (US: 30 inches, EU: 600-900 mm). What happens to the distance estimates?

4. **Add roundabout detection.** Modify the intersection analyzer to detect roundabouts (a different traffic pattern). When a roundabout is detected, suppress "blocked intersection" alerts (since waiting is expected in a roundabout).

## 13.9 Key Takeaways

- The intersection module detects two failure modes: stop sign violations and blocked intersections.
- Stop sign alerts require both a proximal sign (≤ 50 m) and sufficient speed (≥ 10 mph).
- Blocked intersection alerts use an occupancy heuristic (vehicle bounding box area > 30% of frame).
- Alerts are logged with structured context for debugging and voice output generation.
- The module is stateless between frames — no temporal filtering (by design, to minimize alert latency).
- The occupancy heuristic has known limitations (distance bias, perspective distortion) that a BEV projection would address.

In Chapter 14, we cover the lane courtesy module — the socially-aware component that reminds drivers to keep right except to pass.
