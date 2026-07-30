# 📘 Seeing Machines

**Deep Learning & Computer Vision from Python to Bare Metal**

A comprehensive 16-chapter book about building production-grade computer vision systems. Train models with **typed Python**, deploy with **high-performance Rust** on edge devices, and apply everything through the **CivicSense** capstone project.

## Read the Book

**🌐 [arpanpathak.github.io/seeing-machines-book](https://arpanpathak.github.io/seeing-machines-book/)**

## Contents

| Part | Chapters | Topics |
|------|----------|--------|
| **I — Foundations** | 1-3 | Mathematics of Seeing, Neural Networks, CNNs |
| **II — Training** | 4-6 | YOLO, Typed Python Pipelines, ONNX Export |
| **III — Rust Inference** | 7-9 | Why Rust, Inference Engine, Video Pipeline |
| **IV — Tracking** | 10-12 | Kalman Filters, Deep SORT, Geometric Vision |
| **V — Capstone** | 13-15 | Intersection Intelligence, Lane Speed, Edge Deployment |
| **VI — Engineering** | 16 | Verification, Testing & Performance |

## Build Locally

```bash
# Install mdBook
cargo install mdbook

# Build & serve
cd book && mdbook serve --open
```

## Related Project

The capstone project: [driving-civicsense-vision-model](https://github.com/arpanpathak/driving-civicsense-vision-model)

## License

**GNU AGPL v3** — Same as the CivicSense project it documents.
