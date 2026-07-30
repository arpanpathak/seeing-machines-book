# Chapter 11: Deep SORT — Tracking by Detection

> *"An object is not just a detection. It is a story that unfolds over time."*

Deep SORT (Simple Online and Realtime Tracking with a Deep Association Metric) is the tracking architecture used in CivicSense. It follows the "tracking-by-detection" paradigm: a detector (YOLO) finds objects in each frame, and a tracker (Deep SORT) links detections into coherent tracks across time.

This chapter covers the complete tracker implementation in `src/tracking/deep_sort.rs`, including the predict-match-update-birth-death cycle that maintains persistent object identities.

## 11.1 The Tracking-by-Detection Paradigm

Tracking-by-detection decouples the problem into two independent stages:

1. **Detection**: Find all objects in the current frame (YOLO).
2. **Association**: Match detections to existing tracks (Deep SORT).

The advantage of this decoupling is modularity: you can swap the detector (YOLOv8, YOLOv11, etc.) without changing the tracker, and vice versa. The disadvantage is that the tracker cannot correct detection errors — if YOLO misses a vehicle for 3 frames, the track dies.

Deep SORT addresses this through:
- **Motion prediction** (Kalman filter): predicts where each track will be in the next frame, bridging brief detection gaps.
- **Appearance matching** (cosine distance of feature embeddings): associates objects based on visual similarity, not just position overlap.
- **IoU gating**: rejects associations that are geometrically impossible (IoU < 0.3).

The CivicSense implementation uses IoU-only matching (appearance matching is reserved for future Re-ID model training). This is sufficient for traffic scenes where vehicles are well-separated and move predictably.

## 11.2 The Tracking Cycle

The `MultiObjectTracker::update()` method executes one complete tracking cycle per frame:

```rust
pub fn update(&mut self, detections: &[Detection]) -> Vec<Track> {
    self.predict_all();                              // 1. Predict
    let unmatched_det = self.match_and_update(detections);  // 2. Match
    self.birth_new_tracks(detections, &unmatched_det);      // 3. Birth
    self.confirm_tracks();                                    // 4. Confirm
    self.remove_stale_tracks();                               // 5. Death
    self.tracks.clone()
}
```

### 11.2.1 Predict All

Advance every active track's Kalman state by one frame:

```rust
fn predict_all(&mut self) {
    for track in &mut self.tracks {
        track.predict();
    }
}
```

After prediction, each track's `bbox` reflects where it expects to find the object in the current frame. The covariance has increased (process noise added), representing growing uncertainty about the object's position.

### 11.2.2 Match and Update

Match detections to predictions using greedy IoU association:

```rust
fn match_and_update(&mut self, detections: &[Detection]) -> Vec<usize> {
    if self.tracks.is_empty() || detections.is_empty() {
        return (0..detections.len()).collect();
    }

    let matches = self.build_match_candidates(detections);
    self.apply_matches(detections, &matches)
}
```

**Building match candidates:**

```rust
fn build_match_candidates(&self, detections: &[Detection]) -> Vec<(usize, usize, f32)> {
    let mut candidates = Vec::new();
    for (ti, track) in self.tracks.iter().enumerate() {
        for (di, det) in detections.iter().enumerate() {
            let iou_val = compute_iou(track.bbox, (det.x1, det.y1, det.x2, det.y2));
            if iou_val > 0.3 {  // IoU gating threshold
                candidates.push((ti, di, iou_val));
            }
        }
    }
    // Sort by descending IoU (best matches first)
    candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    candidates
}
```

**Greedy assignment:**

```rust
fn apply_matches(&mut self, detections: &[Detection], matches: &[(usize, usize, f32)]) -> Vec<usize> {
    let mut used_trk = vec![false; self.tracks.len()];
    let mut used_det = vec![false; detections.len()];

    for &(ti, di, _) in matches {
        if !used_trk[ti] && !used_det[di] {
            self.tracks[ti].update(&detections[di]);  // Kalman update
            used_trk[ti] = true;
            used_det[di] = true;
        }
    }

    // Return indices of unmatched detections
    detections.iter()
        .enumerate()
        .filter(|(i, _)| !used_det[*i])
        .map(|(i, _)| i)
        .collect()
}
```

**Why greedy and not Hungarian?** The Hungarian algorithm (also called the Munkres algorithm) finds the optimal assignment that minimizes the total cost. Greedy assignment picks the best match first, then the next best, etc. Greedy is \\( O(n^2) \\) and Hungarian is \\( O(n^3) \\). For < 50 detections per frame, the difference is negligible, and greedy produces near-optimal results for well-separated objects.

The IoU gating threshold (0.3) rejects matches where the predicted track and detection have less than 30% overlap. This prevents a track from jumping to a completely different object that happens to be nearby.

### 11.2.3 Birth: New Tracks from Unmatched Detections

```rust
fn birth_new_tracks(&mut self, detections: &[Detection], unmatched: &[usize]) {
    for &di in unmatched {
        let track = Track::new(self.next_id, &detections[di]);
        self.tracks.push(track);
        self.next_id += 1;
    }
}
```

Every unmatched detection becomes a new **tentative** track. The track starts with `hits = 1` and `is_confirmed = false`. It must accumulate `n_init` (3) successful matches before it is promoted to confirmed status.

**Why n_init = 3?** A single unmatched detection might be a false positive (YOLO occasionally detects a cloud as a truck). By requiring 3 consecutive matches, we filter out spurious detections while keeping the initialization latency acceptable (~100 ms at 30 FPS).

### 11.2.4 Confirm: Promoting Stable Tracks

```rust
fn confirm_tracks(&mut self) {
    for track in &mut self.tracks {
        if track.hits() >= self.n_init {
            track.is_confirmed = true;
        }
    }
}
```

Confirmed tracks are returned to the analysis modules. Tentative tracks are tracked internally but not exposed — they could still be spurious.

### 11.2.5 Death: Removing Lost Tracks

```rust
fn remove_stale_tracks(&mut self) {
    self.tracks.retain(|t| t.time_since_update() <= self.max_age);
}
```

Tracks that have been unmatched for more than `max_age` (30 frames, ~1 second) are removed. This handles:
- **Occlusion**: A vehicle behind a truck for 0.5 seconds. The track survives because `max_age = 30` (30 frames at 30 FPS = 1 second).
- **Departure**: A vehicle that exits the frame. The track dies naturally.
- **Detection failures**: YOLO misses a vehicle for a few frames due to lighting or pose. The track bridges the gap.

## 11.3 The Track Lifecycle

```
        ┌─────────────────────────┐
        │   Detection (YOLO)      │
        │   [frame n]             │
        └───────────┬─────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │   Birth: Track.new()     │
        │   hits = 1, tentative    │
        └───────────┬─────────────┘
                    │
         ┌─────────┴─────────┐
         │                   │
         ▼                   ▼
  ┌──────────────┐    ┌──────────────┐
  │ Matched to   │    │ No match for │
  │ detection    │    │ max_age      │
  │ hits += 1    │    │ frames       │
  └──────┬───────┘    └──────┬───────┘
         │                   │
         ▼                   ▼
  ┌──────────────┐    ┌──────────────┐
  │ hits >=      │    │ REMOVED      │
  │ n_init (3)   │    │ (stale)      │
  └──────┬───────┘    └──────────────┘
         │
         ▼
  ┌─────────────────────────────────┐
  │ CONFIRMED                       │
  │ Track is reported to modules    │
  │ Can receive alerts              │
  └──────────┬──────────────────────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
  ┌────────┐  ┌──────────┐
  │ Still  │  │ No match │
  │ active │  │ for >    │
  │        │  │ max_age  │
  └────────┘  └────┬─────┘
                   │
                   ▼
             ┌──────────┐
             │ REMOVED  │
             │ (stale)  │
             └──────────┘
```

## 11.4 Tracking Performance Considerations

The tracker operates within a 5 ms budget per frame. The key operations and their costs:

| Operation | Cost (100 tracks, 50 detections) |
|-----------|----------------------------------|
| Kalman predict (100 tracks) | ~2 μs |
| Build candidates (5000 pairs) | ~50 μs |
| Sort candidates (5000 items) | ~80 μs |
| Apply matches (50 matches) | ~1 μs |
| Birth new tracks (10) | ~1 μs |
| Remove stale tracks | ~1 μs |
| **Total** | **~135 μs** |

The tracker is well within its budget. The dominant cost is the IoU computation for candidate pairs — 5000 pairs × ~10 ns/IoU = 50 μs. This leaves room for adding appearance-based matching (CNN feature embedding + cosine distance) in the future.

## 11.5 The IoU Metric Under the Hood

The IoU computation, from `src/utils/geometry.rs`:

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

The `max(0.0)` on the intersection dimensions ensures non-overlapping boxes produce IoU = 0 (not negative). The `union <= 0.0` check prevents division by zero.

## 11.6 The Capstone Connection: Tracks in the Analysis Pipeline

The analysis modules (intersection, lane speed) receive tracks, not raw detections. This is deliberate:

- **Tracks have memory.** A `Track` knows how long it has been tracked, how many times it has been matched, and its velocity (from the Kalman filter). A raw detection has none of this context.
- **Tracks are smooth.** The Kalman filter has removed jitter. A track's bounding box changes gradually across frames, which is critical for distance estimation (which depends on stable pixel widths).
- **Tracks have identity.** The `track_id` allows the analysis modules to associate events with specific vehicles across time. "The truck that was following too closely 3 seconds ago" is a meaningful concept.

The `process_frame()` method passes tracks to the lane speed analyzer and raw detections to the intersection analyzer:

```rust
let intersection_alerts = self.intersection_analyzer
    .analyze(&detections, self.ego_speed, dt_secs);
let lane_alerts = self.lane_speed_analyzer
    .analyze(&tracks, self.ego_speed, dt_secs);
```

The intersection analyzer uses raw detections because it only needs the current frame's information (is there a stop sign? how occupied is the intersection?). The lane speed analyzer uses tracks because it needs temporal information (how fast is that vehicle in the right lane moving relative to me?).

## 11.7 Exercises

1. **Implement Hungarian matching.** Replace the greedy matching with the Hungarian algorithm. Compare track ID switches on a MOT benchmark dataset.

2. **Add appearance features.** Implement a simple color histogram-based Re-ID feature. Extract HSV histograms from each detection and compute cosine distance during matching.

3. **Analyze occlusion robustness.** Create a synthetic sequence where a vehicle is occluded for 10, 20, 30, and 60 frames. At what point does the tracker lose the ID?

4. **Track visualization.** Output a video with track IDs rendered on each bounding box. Verify that the same vehicle keeps the same ID across the video.

## 11.8 Key Takeaways

- Deep SORT follows a five-stage cycle: predict → match → birth → confirm → death.
- Greedy IoU matching with a 0.3 gating threshold provides fast, near-optimal association for traffic scenes.
- Tracks start tentative and become confirmed after `n_init` (3) successful matches.
- Tracks die after `max_age` (30) unmatched frames, bridging brief occlusions.
- The tracker operates well within its 5 ms budget (~135 μs typical).
- Analysis modules receive tracks (with memory and identity) rather than raw detections.

In Chapter 12, we cover the geometric computations that bridge pixels and real-world distances — the pinhole model, velocity estimation, and the Bird's Eye View projection.
