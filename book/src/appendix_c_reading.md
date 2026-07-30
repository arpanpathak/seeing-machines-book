# Appendix C  -  Recommended Reading & Tools

The books, papers, and tools that shaped the thinking behind this book and the CivicSense project.

## C.1 Books

### Computer Vision & Deep Learning

- **"Deep Learning" by Goodfellow, Bengio, and Courville** (MIT Press, 2016)  -  The definitive textbook. Chapters 6-9 (feedforward networks, regularization, CNNs) are essential reading. The chapter on practical methodology (Ch 11) is criminally underread.

- **"Computer Vision: Algorithms and Applications" by Richard Szeliski** (Springer, 2nd ed, 2022)  -  The comprehensive reference for geometric computer vision. Chapters 1-4 cover image formation, processing, and feature detection. Free online.

- **"Multiple View Geometry in Computer Vision" by Hartley and Zisserman** (Cambridge, 2nd ed, 2004)  -  The bible of multi-view geometry. Dense but essential for anyone doing 3D reconstruction or advanced geometric CV.

- **"Programming Computer Vision with Python" by Jan Erik Solem** (O'Reilly, 2012)  -  A practical, code-first introduction to CV fundamentals. Good companion to the theory-heavy books above.

### Rust

- **"The Rust Programming Language" by Steve Klabnik and Carol Nichols** (No Starch Press, 2nd ed, 2023)  -  The official Rust book. Free online at doc.rust-lang.org/book/. Chapters 4 (ownership), 10 (generics/traits), and 15 (smart pointers) are particularly relevant to CV systems programming.

- **"Rust for Rustaceans" by Jon Gjengset** (No Starch Press, 2022)  -  The intermediate Rust book. Chapters on async Rust, unsafe Rust, and FFI are directly applicable to edge deployment.

- **"Programming Rust" by Jim Blandy, Jason Orendorff, and Leonora Tindall** (O'Reilly, 2nd ed, 2021)  -  An alternative reference that covers the same material from a systems programming perspective.

### Mathematics

- **"Linear Algebra and Its Applications" by Gilbert Strang** (Cengage, 5th ed, 2016)  -  The gold standard for intuition-based linear algebra. Chapter 1 (matrix operations) and Chapter 5 (eigenvalues/covariance) are directly relevant.

- **"Probability and Statistics for Computer Science" by David Forsyth** (Springer, 2018)  -  Accessible introduction to probability from a CS perspective. The chapters on Bayes' theorem and Gaussian distributions are essential for Kalman filters.

- **"Calculus" by Michael Spivak** (Publish or Perish, 4th ed, 2008)  -  If you want the rigorous treatment of differentiation and integration. Overkill for most CV work, but the chain rule derivation is beautiful.

## C.2 Papers

### Object Detection

- **"You Only Look Once: Unified, Real-Time Object Detection" (Redmon et al., 2016)**  -  The original YOLO paper. Primarily of historical interest; the modern versions are architecturally different.
- **"YOLOv8: A State-of-the-Art Real-Time Object Detector" (Ultralytics, 2023)**  -  The technical report for YOLOv8. Describes the CSPDarknet backbone, C2f module, and DFL loss.
- **"Focal Loss for Dense Object Detection" (Lin et al., 2017)**  -  The foundation for class imbalance handling in modern detectors.

### Tracking

- **"Simple Online and Realtime Tracking (SORT)" (Bewley et al., 2016)**  -  The original SORT paper. The baseline for all modern online trackers.
- **"Deep SORT: Simple Online and Realtime Tracking with a Deep Association Metric" (Wojke et al., 2017)**  -  Adds appearance-based matching to SORT. The architecture used in CivicSense.
- **"A Kalman Filter for Robust Tracking" (Maybeck, 1979)**  -  The classic Kalman filter reference. Stochastics, Estimation, and Control, Vol 1.

### Kalman Filters

- **"An Introduction to the Kalman Filter" (Welch & Bishop, 2006)**  -  The most widely-cited Kalman filter tutorial. Clear derivation with worked examples.
- **"Understanding the Basis of the Kalman Filter Via a Simple and Intuitive Derivation" (Faragher, 2012)**  -  Less formal but more intuitive than Welch & Bishop. Published in IEEE Signal Processing Magazine.

### Edge Deployment

- **"Efficient Inference on ARM Devices" (Arm Research, 2023)**  -  Practical guide to optimizing ML inference on ARM CPUs.
- **"ONNX Runtime: A High-Performance Cross-Platform Inference Engine" (Microsoft, 2021)**  -  The architecture paper for ONNX Runtime.

## C.3 Tools

### Development

- **Netron** (github.com/lutzroeder/netron)  -  ONNX model visualization tool. Invaluable for debugging model exports.
- **rerun.io**  -  Real-time visualization SDK for computer vision. Stream bounding boxes, tracks, and alerts in a web dashboard.
- **label-studio** (labelstud.io)  -  Open-source data labeling platform. Supports YOLO format export.

### Profiling & Benchmarking

- **perf** (Linux)  -  CPU profiling. Essential for identifying hot paths.
- **samply** (github.com/mstange/samply)  -  Rust-native sampling profiler with Firefox Profiler integration.
- **cargo-criterion**  -  Cargo extension for criterion benchmarks with CI integration.
- **flamegraph** (github.com/flamegraph-rs/flamegraph)  -  Generate flame graphs from perf or samply output.

### Edge Deployment

- **Raspberry Pi Imager** (raspberrypi.com/software)  -  Write Raspberry Pi OS to SD card.
- **Hailo Model Zoo** (github.com/hailo-ai/hailo-model-zoo)  -  Pre-compiled models for Hailo NPUs.
- **pi-gen** (github.com/RPi-Distro/pi-gen)  -  Build custom Raspberry Pi OS images with pre-installed software.

## C.4 Online Resources

- **Ultralytics Documentation** (docs.ultralytics.com)  -  The official YOLO documentation. The training guide and export guide are essential references.
- **ONNX Runtime Docs** (onnxruntime.ai/docs)  -  Execution provider configuration, optimization options, and C API reference.
- **Rust Performance Book** (nnethercote.github.io/perf-book)  -  Practical Rust optimization techniques by Nicholas Nethercote.
- **The Rustonomicon** (doc.rust-lang.org/nomicon)  -  Required reading for any `unsafe` Rust code.
- **CivicSense Repository** (github.com/arpanpathak/driving-civicsense-vision-model)  -  The living reference. Open issues, PRs, and discussions.

## C.5 Conferences to Follow

- **CVPR** (Computer Vision and Pattern Recognition)  -  June annually. The premier CV conference.
- **ECCV** (European Conference on Computer Vision)  -  Even years, August.
- **ICCV** (International Conference on Computer Vision)  -  Odd years, October.
- **RustConf**  -  Annual Rust conference. Talks on embedded and systems programming.
- **Embedded Vision Summit**  -  May annually (Santa Clara). Commercial and industrial CV deployment.
