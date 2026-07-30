# Chapter 16: Verification, Testing & Performance

> *"A test is not a test if it only passes. A test is a proof that the code is wrong  -  and the code is wrong until proven otherwise."*

The CivicSense coding standards define six formal verification properties that every PR must satisfy:

1. **Complexity proof**: Worst-case time complexity must be documented and verified not to be \\( O(n^2) \\) due to hidden nested iteration.
2. **Null-safety proof**: Every nullable type must be traced; no `!!` or forced unwrap.
3. **Async liveness**: Every I/O call must have a timeout; parents must know if child tasks panic.
4. **Lazy termination**: Every iterator/generator must have a documented terminal operator.
5. **SOLID compliance**: Every module must point to its abstraction point.
6. **Mechanical sympathy**: Hot paths must document cache miss and allocation rates.

This chapter covers how these properties are enforced through testing, benchmarking, and profiling.

## 16.1 Unit Testing: The Foundation

Every non-trivial function in CivicSense has a `#[test]` annotation. The tests are structured to cover:

- **Normal operation**: What happens with valid inputs.
- **Edge cases**: Empty inputs, extreme values, boundary conditions.
- **Error handling**: What happens when inputs are invalid.

### 16.1.1 A Representative Test Suite

From `src/detection/yolo.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detector_constructs_without_model() {
        let cfg = YoloConfig {
            model_path: "nonexistent.onnx".into(),
            conf_threshold: 0.5,
            iou_threshold: 0.45,
            input_width: 640,
            input_height: 640,
            class_names: vec!["stop_sign".into()],
        };
        let detector = YoloDetector::new(cfg).expect("Constructor should not fail");
        assert!(!detector.is_model_available());
    }

    #[test]
    fn test_detect_returns_empty_when_no_model() {
        let cfg = YoloConfig {
            model_path: "nonexistent.onnx".into(),
            conf_threshold: 0.5,
            iou_threshold: 0.45,
            input_width: 640,
            input_height: 640,
            class_names: vec![],
        };
        let mut detector = YoloDetector::new(cfg).unwrap();
        let results = detector.detect(&[], 640, 480).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_anchor_grid_size() {
        let grid = AnchorGrid::new(640);
        assert_eq!(grid.num_predictions, 8400);
    }

    #[test]
    fn test_nms_keeps_best() {
        let candidates = vec![
            BBox { x1: 10.0, y1: 10.0, x2: 100.0, y2: 100.0, confidence: 0.9, class_id: 0 },
            BBox { x1: 15.0, y1: 15.0, x2: 95.0, y2: 95.0, confidence: 0.8, class_id: 0 },
            BBox { x1: 200.0, y1: 200.0, x2: 300.0, y2: 300.0, confidence: 0.7, class_id: 0 },
        ];
        let kept = non_max_suppression(candidates, 0.5);
        assert_eq!(kept.len(), 2);
        assert!((kept[0].confidence - 0.9).abs() < 1e-6);
    }
}
```

**What makes these tests good:**

- They test **public API** behavior (constructor, detect, grid size), not internal implementation details.
- They test the **documented behavior** (no model file → no panic, empty results).
- They use **assert! with epsilon** for floating-point comparisons (`1e-6`).
- They cover **normal** (grid size), **edge** (empty detections), and **error** (missing model) cases.

### 16.1.2 The TDD Approach in CivicSense

The codebase follows a modified test-driven development cycle:

1. **Write the test first** (defines expected behavior).
2. **Write the function** (makes the test pass).
3. **Write the documentation** (doc-comment with examples).
4. **Run the test** (verify the function is correct).
5. **Benchmark** (measure latency against the budget).

This cycle ensures every function is tested, documented, and performant before it is merged.

## 16.2 Property-Based Testing: Beyond Example-Based Tests

Example-based tests ("assert that 2 + 2 = 4") are necessary but not sufficient. Property-based tests ("assert that addition is commutative for all inputs") catch edge cases that example-based tests miss.

### 16.2.1 Invariant Testing

The `compute_iou` function should satisfy:

- **Symmetry**: \(IoU(A, B) = IoU(B, A)\)
- **Reflexivity**: \(IoU(A, A) = 1.0\)
- **Non-negativity**: \\( IoU(A, B) \geq 0.0 \\)
- **Boundedness**: \\( IoU(A, B) \leq 1.0 \\)

Example-based tests verify these for specific cases. Property-based tests verify them for random cases:

```rust
proptest! {
    #[test]
    fn iou_is_symmetric(
        ax1 in 0.0..1000.0f32, ay1 in 0.0..1000.0f32,
        ax2 in 0.0..1000.0f32, ay2 in 0.0..1000.0f32,
        bx1 in 0.0..1000.0f32, by1 in 0.0..1000.0f32,
        bx2 in 0.0..1000.0f32, by2 in 0.0..1000.0f32,
    ) {
        // Ensure valid boxes (x1 < x2, y1 < y2)
        let (ax1, ax2) = if ax1 < ax2 { (ax1, ax2) } else { (ax2, ax1) };
        let (ay1, ay2) = if ay1 < ay2 { (ay1, ay2) } else { (ay2, ay1) };
        let (bx1, bx2) = if bx1 < bx2 { (bx1, bx2) } else { (bx2, bx1) };
        let (by1, by2) = if by1 < by2 { (by1, by2) } else { (by2, by1) };
        
        let iou_ab = compute_iou((ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2));
        let iou_ba = compute_iou((bx1, by1, bx2, by2), (ax1, ay1, ax2, ay2));
        
        prop_assert!((iou_ab - iou_ba).abs() < 1e-6, "IoU must be symmetric");
    }
}
```

This test generates random bounding boxes and verifies symmetry. If there is any input pair for which `compute_iou` is not symmetric, `proptest` finds it and reports the minimal failing input (shrinking).

### 16.2.2 Invariant Documentation

Every struct with internal invariants documents them:

```rust
/// Kalman filter state for a tracked vehicle.
///
/// INVARIANT: `covariance` must always be symmetric positive-definite.
/// Violating this produces incorrect (and possibly NaN) state estimates.
/// All update steps must enforce this via `covariance = (covariance + covariance.t()) / 2.0`.
pub struct KalmanState {
    mean: [f32; 8],
    covariance: [[f32; 8]; 8],
}
```

These invariants are checked in debug builds:

```rust
debug_assert!(self.is_covariance_valid(), "Kalman covariance invariant violated");
```

## 16.3 Latency Budget Testing

The coding standards define strict latency budgets for each pipeline stage. These are enforced through benchmark tests:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_nms(c: &mut Criterion) {
    let candidates: Vec<BBox> = (0..200)
        .map(|i| BBox {
            x1: i as f32 * 10.0,
            y1: i as f32 * 10.0,
            x2: i as f32 * 10.0 + 50.0,
            y2: i as f32 * 10.0 + 50.0,
            confidence: 0.5 + (i as f32 / 200.0) * 0.5,
            class_id: 0,
        })
        .collect();

    c.bench_function("nms_200_candidates", |b| {
        b.iter(|| non_max_suppression(black_box(candidates.clone()), 0.45))
    });
}
```

Any PR that increases latency beyond the budget without profiling data is rejected. The CI pipeline compares benchmark results against the previous commit:

```bash
# Run benchmarks and compare to baseline
cargo bench -- --save-baseline current
cargo bench -- --baseline previous --baseline current
```

If the new code is >10% slower on any benchmark, the PR requires an explanation and a profiling trace.

## 16.4 Integration Testing: The Full Pipeline

Integration tests verify that all modules work together correctly:

```rust
#[test]
fn test_full_pipeline_cycle() {
    // 1. Create a test frame (1280x720 gray)
    let frame = vec![128u8; 1280 * 720 * 3];
    
    // 2. Initialize the pipeline (with no model file → empty detections)
    let config = Config::default();
    let mut detector = YoloDetector::new(YoloConfig::from(&config.model)).unwrap();
    let mut tracker = MultiObjectTracker::new(30, 3, 0.2);
    let mut intersection = IntersectionAnalyzer::new(&config, 1280, 720);
    let mut lane_speed = LaneSpeedAnalyzer::new(&config);
    
    // 3. Process 5 frames
    for _ in 0..5 {
        let detections = detector.detect(&frame, 1280, 720).unwrap();
        let tracks = tracker.update(&detections);
        
        let intersection_alerts = intersection.analyze(&detections, 35.0, 0.033);
        let lane_alerts = lane_speed.analyze(&tracks, 35.0, 0.033);
        
        // No detections → no alerts
        assert!(intersection_alerts.is_empty());
        assert!(lane_alerts.is_empty());
    }
    
    // 4. No tracks should survive past max_age
    assert_eq!(tracker.track_count(), 0);
}
```

This test exercises the entire pipeline: detection (returns empty because no model), tracking (nothing to track), intersection analysis (no objects to analyze), lane speed analysis (no tracks to evaluate). It verifies that the system degrades gracefully when no model is loaded.

## 16.5 The PR Verification Checklist

Before merging any PR, the developer must verify:

```markdown
- [ ] `cargo test` passes
- [ ] `cargo clippy -- -D warnings` passes
- [ ] `cargo fmt` has been run
- [ ] No `todo!()` or `unimplemented!()` in production code
- [ ] All new public items have doc comments
- [ ] `// SAFETY:` comments on every `unsafe` block
- [ ] Property-based tests for critical math/logic
- [ ] Benchmarks added for hot-path functions
- [ ] No allocations on the inference hot path
- [ ] Every function ≤ 50 lines (or justified in doc comment)
- [ ] No dead code or commented-out code
```

This checklist is enforced by CI tooling (a GitHub Actions workflow that runs clippy, tests, and benchmarks).

## 16.6 The Capstone Connection: Verification in Action

The CivicSense codebase currently has:

- **47 unit tests** across all modules.
- **8 property-based tests** for geometry and tracking invariants.
- **6 benchmarks** for hot-path functions.
- **100% doc comment coverage** on public items.
- **0 `unsafe` blocks** in the inference pipeline (the only `unsafe` is in `mlockall()` in main.rs, with a `// SAFETY:` justification).

The test suite runs in under 30 seconds on a development machine. The benchmarks take about 2 minutes. Both must pass before any PR is merged.

## 16.7 Exercises

1. **Add a property-based test for `estimate_distance`.** Verify that: (a) larger pixel width → smaller distance, (b) distance is always positive, (c) the formula matches the mathematical derivation.

2. **Create a latency budget test.** Write a criterion benchmark that runs the full `process_frame()` pipeline on a synthetic frame and asserts it completes within 33 ms.

3. **Set up CI.** Create a GitHub Actions workflow that runs `cargo test`, `cargo clippy`, and `cargo bench` on every PR. Add a check that benchmarks do not regress by more than 10%.

4. **Mutation testing.** Use a mutation testing tool (like `cargo-mutants`) to see if your tests catch intentionally introduced bugs. What is your mutation score?

## 16.8 Key Takeaways

- Every non-trivial function must have a unit test covering normal, edge, and error cases.
- Property-based tests verify invariants (symmetry, bounds, correctness) across random inputs.
- Invariant documentation (`/// INVARIANT:`) makes hidden assumptions explicit.
- Latency budget tests enforce performance requirements at the CI level.
- Integration tests verify that modules compose correctly.
- The PR verification checklist formalizes the quality gate for every contribution.

This is the last chapter of the book  -  not because verification is an afterthought, but because it is the final, essential step before the code meets the real road.
