# Chapter 1: The Mathematics of Seeing

> *"The eye is the first circle; the horizon which it forms is the second; and throughout nature this primary figure is repeated without end."*  -  Ralph Waldo Emerson

Before we can teach a machine to see, we must understand what "seeing" means in mathematical terms. Your visual cortex performs approximately 10 billion calculations per second. We are going to build something less ambitious but more explicit  -  a system that takes pixels and produces meaning, one linear algebra operation at a time.

This chapter covers the mathematical foundations you need to understand before writing a single line of code. Every concept here appears directly in the CivicSense codebase. If you skip this chapter, the code in later chapters will compile but it will not *click*. So do not skip it.

## 1.1 Pixels as Vectors: The Bridge Between Light and Mathematics

A digital image is, at its core, a rectangular grid of numbers. A grayscale image of width \(W\) and height \(H\) is a matrix \\( I \in \mathbb{R}^{H \times W} \\) where each entry \\( I\_{i,j} \\) represents the intensity of light at pixel \((i, j)\), typically in the range \([0, 255]\).

But in computer vision, we rarely work with raw pixel grids. We reshape them. We convolve them. We embed them into higher-dimensional spaces. And to do any of this, we must think of images as **vectors** in high-dimensional spaces.

### 1.1.1 The Vector Space of Images

Consider a 3-channel RGB image of dimensions \\( 640 \times 640 \\). This image can be represented as a vector:

\\[\mathbf{x} \in \mathbb{R}^{640 \times 640 \times 3} \cong \mathbb{R}^{1,228,800}\\]

That is 1.2 million dimensions. You cannot visualize this space, but you can reason about it algebraically. Every image is a point in this space. Two images of stop signs form a cluster. A random noise image is far away. The operation of "detecting a stop sign" is equivalent to finding a decision boundary in this 1.2-million-dimensional space.

The reason neural networks work is that they learn to project this enormous space into a lower-dimensional *latent space* where the geometry of the data becomes linearly separable. A YOLO model, for instance, takes our \\( 640 \times 640 \times 3 \\) input and compresses it through a series of transformations until it produces, say, an \\( 84 \times 8400 \\) output tensor  -  84 channels (4 bounding box coordinates + 80 COCO class probabilities) for each of 8400 anchor points.

### 1.1.2 Why This Matters for CivicSense

The CivicSense detector takes frames at \\( 1280 \times 720 \times 3 \\)  -  that is 2.76 million input dimensions. It letterboxes them to \\( 640 \times 640 \times 3 \\) (1.23 million) and runs them through a YOLOv11n model that has 2.6 million parameters. Each parameter is a learned coefficient in the transformation that maps input pixels to output detections.

If you do not understand that an image is a vector, you will not understand why data augmentation works (it is perturbation in the input space), why adversarial attacks work (they are small movements in input space that cross a decision boundary), or why your model fails when the camera is mounted at a slightly different angle (the training distribution and the deployment distribution are different regions of the same vector space).

## 1.2 Linear Algebra: The Language of Transformations

A neural network is a composition of linear transformations and nonlinear activation functions. If you master four operations  -  matrix multiplication, transposition, broadcasting, and the dot product  -  you can understand 90% of the arithmetic that happens in a forward pass.

### 1.2.1 Matrix Multiplication as a Batch Operation

When you write `output = W @ x + b` in Python (or Rust), you are performing:

\\[\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}\\]

where \\( \mathbf{W} \in \mathbb{R}^{m \times n} \\), \\( \mathbf{x} \in \mathbb{R}^{n} \\), and \\( \mathbf{b} \in \mathbb{R}^{m} \\).

Each row of \\( \mathbf{W} \\) computes a dot product with \\( \mathbf{x} \\):

\\[y\_j = \sum\_{k=1}^{n} W\_{j,k} \cdot x\_k + b\_j\\]

In a convolutional neural network, the weight matrix is structured  -  it is sparse (most entries are zero) and it has a special Toeplitz-like structure where the same weights are applied to different patches of the input. But the fundamental operation is the same: a weighted sum followed by a bias shift.

**Why this matters for the inference engine:** When you look at the `ort` crate's inference call in Chapter 8, you will see that it returns a tensor. That tensor is the result of a sequence of these matrix multiplications. The shape of that tensor  -  what is called its *semantic layout*  -  determines how you decode it into bounding boxes. If you do not know the output shape convention of YOLOv8 (84 channels × 8400 predictions), you will misinterpret the tensor.

### 1.2.2 The Dot Product as a Similarity Measure

The dot product of two vectors \\( \mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta \\) is the workhorse of attention mechanisms, cosine similarity in Deep SORT re-identification, and the confidence scoring in detection heads.

In YOLO, the class probability for a given anchor is computed by passing the class logits through a sigmoid function. But the logits themselves are dot products: each is the dot product between a row of the classification weight matrix and the feature vector at that grid cell.

In Deep SORT, the cosine distance between appearance embeddings is:

\\[d(\mathbf{a}, \mathbf{b}) = 1 - \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}\\]

This is the gating function that determines whether two detections across frames are likely the same object. In our current implementation, we use IoU-based matching (which is geometric), but the architecture is designed to accept appearance-based matching when we train a Re-ID model.

### 1.2.3 The Covariance Matrix: The Heart of Kalman Filters

The Kalman filter (Chapter 10) revolves around two matrices: the state covariance \\( \mathbf{P} \\) and the measurement covariance \\( \mathbf{R} \\). These are the mathematical expression of uncertainty.

A covariance matrix \\( \mathbf{P} \in \mathbb{R}^{n \times n} \\) for an \(n\)-dimensional state vector \\( \mathbf{x} \\) has entries:

\\[P\_{i,j} = \text{Cov}(x\_i, x\_j) = \mathbb{E}[(x\_i - \mu\_i)(x\_j - \mu\_j)]\\]

The diagonal entries \\( P\_{i,i} \\) are the variances  -  how uncertain we are about each state variable. The off-diagonal entries capture *correlations* between state variables. If \\( P\_{i,j} > 0 \\), then when \\( x\_i \\) is high, \\( x\_j \\) tends to be high too.

In the CivicSense Kalman filter (see `src/tracking/deep_sort.rs`), the state vector is 8-dimensional:

\\[\mathbf{x} = [c\_x, c\_y, w, h, v\_x, v\_y, v\_w, v\_h]\\]

The covariance matrix is \\( 8 \times 8 = 64 \\) entries. Our implementation stores it flattened row-major and, in a simplification, only maintains the diagonal. The update step uses a scalar-gain approximation:

\\[\text{gain}\_i = \frac{P\_{i,i}}{P\_{i,i} + R}\\]

instead of the full matrix inversion \\( \mathbf{K} = \mathbf{P}\mathbf{H}^T(\mathbf{H}\mathbf{P}\mathbf{H}^T + \mathbf{R})^{-1} \\).

This simplification is justified because the state variables are approximately independent in the measurement space  -  the bounding box center \\( (c\_x, c\_y) \\) is only weakly correlated with its dimensions \((w, h)\). But you need to understand the full matrix form to know when this simplification breaks down (it does, for example, when vehicles are heavily occluded and the width-height correlation becomes significant).

## 1.3 Calculus: How Learning Happens

Neural networks learn by gradient descent. Gradient descent works by computing partial derivatives of a loss function with respect to every parameter in the network, then moving each parameter in the direction that reduces the loss.

### 1.3.1 The Chain Rule Is the Only Rule

The gradient computation in a neural network is a repeated application of the chain rule:

\\[\frac{\partial L}{\partial \mathbf{W}^{(l)}} = \frac{\partial L}{\partial \mathbf{a}^{(L)}} \cdot \frac{\partial \mathbf{a}^{(L)}}{\partial \mathbf{a}^{(L-1)}} \cdots \frac{\partial \mathbf{a}^{(l+1)}}{\partial \mathbf{z}^{(l)}} \cdot \frac{\partial \mathbf{z}^{(l)}}{\partial \mathbf{W}^{(l)}}\\]

where \\( \mathbf{a}^{(l)} \\) is the activation at layer \(l\), \\( \mathbf{z}^{(l)} = \mathbf{W}^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)} \\), and \(L\) is the loss.

When you train a YOLO model, the loss function is a weighted sum of three terms:

\\[\mathcal{L} = \lambda\_{\text{box}} \mathcal{L}\_{\text{CIoU}} + \lambda\_{\text{cls}} \mathcal{L}\_{\text{BCE}} + \lambda\_{\text{dfl}} \mathcal{L}\_{\text{DFL}}\\]

- **CIoU Loss**  -  Complete Intersection-over-Union loss, which penalizes incorrect bounding box positions, aspect ratios, and overlap simultaneously.
- **Binary Cross-Entropy Loss**  -  \\( \mathcal{L}\_{\text{BCE}} = -\sum [y\_i \log(\hat{y}\_i) + (1 - y\_i) \log(1 - \hat{y}\_i)] \\), used for multi-label classification in YOLO's head.
- **Distribution Focal Loss (DFL)**  -  A distributional formulation that refines the bounding box coordinates by learning a probability distribution over the box boundaries rather than regressing them directly.

### 1.3.2 The Gradient of the Sigmoid

The sigmoid function appears everywhere in detection:

\\[\sigma(x) = \frac{1}{1 + e^{-x}}\\]

Its derivative is elegant:

\\[\frac{d\sigma}{dx} = \sigma(x) \cdot (1 - \sigma(x))\\]

This means that when \\( \sigma(x) \\) is near 0 or 1 (the network is very confident), the gradient is nearly zero. This is the *vanishing gradient problem*  -  confident predictions do not learn easily, which is why focal loss was invented to down-weight well-classified examples.

In the YOLO decoder in `src/detection/yolo.rs`, we apply sigmoid to decode the bounding box center coordinates:

```rust
let cx = (sigmoid(output[i]) * 2.0 - 0.5 + gx) * s;
```

This sigmoid ensures the predicted center offset is in \([0, 1]\), which is then scaled and shifted to the grid cell coordinate system. Understanding this single line requires understanding: (1) what sigmoid does, (2) why \\( 2\sigma - 0.5 \\) maps \([0,1]\) to \([-0.5, 1.5]\) (allowing the center to be slightly outside the grid cell for better edge cases), and (3) how the anchor stride \(s\) maps from grid-cell space to pixel space.

### 1.3.3 Numerical Stability: The Log-Sum-Exp Trick

When computing cross-entropy loss with sigmoid outputs, you should *never* compute sigmoid first and then the logarithm. The reason is numerical: if \\( \sigma(x) \\) is very close to 1, \\( \log(\sigma(x)) \\) underflows to \\( -\infty \\).

Instead, use the log-sum-exp trick:

\\[\log(\sigma(x)) = -\log(1 + e^{-x})\\]

which is stable for all values of \(x\). This is implemented as `F.binary_cross_entropy_with_logits` in PyTorch  -  it takes the raw logits, applies sigmoid internally in a numerically stable way, and computes the BCE loss in one fused operation.

## 1.4 Probability: Uncertainty Is Inevitable

Computer vision is a probabilistic enterprise. You never know for certain what is in an image. The best you can do is estimate a probability distribution over possible interpretations.

### 1.4.1 Conditional Probability and Bayes' Rule

The entire detection pipeline can be framed in Bayesian terms. Given an image \\( \mathbf{I} \\), we want:

\\[P(\text{object} \mid \mathbf{I}) = \frac{P(\mathbf{I} \mid \text{object}) \cdot P(\text{object})}{P(\mathbf{I})}\\]

The YOLO model learns \\( P(\text{object} \mid \mathbf{I}) \\) directly (discriminative modeling). But the Kalman filter is a Bayesian filter  -  it maintains a belief state and updates it using Bayes' rule at each time step.

The Kalman filter update (Chapter 10) is:

\\[P(\mathbf{x}\_t \mid \mathbf{z}\_{1:t}) \propto P(\mathbf{z}\_t \mid \mathbf{x}\_t) \cdot P(\mathbf{x}\_t \mid \mathbf{z}\_{1:t-1})\\]

The **predict** step computes the prior \\( P(\mathbf{x}\_t \mid \mathbf{z}\_{1:t-1}) \\) by applying the motion model. The **update** step multiplies by the likelihood \\( P(\mathbf{z}\_t \mid \mathbf{x}\_t) \\) to get the posterior.

### 1.4.2 IoU as a Probability of Overlap

The Intersection-over-Union metric, computed in `src/utils/geometry.rs`:

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

IoU is not a probability in the strict sense, but it behaves like one: it is in \([0, 1]\), it is symmetric, and it measures the degree of overlap between two events (the events being "this pixel belongs to object A" and "this pixel belongs to object B").

In Deep SORT, we use IoU as the association metric: a detection matches a track if their IoU exceeds 0.3 (an empirically determined threshold). This is a probabilistic gating function  -  we reject associations that are unlikely to represent the same physical object.

### 1.4.3 The Confidence Score: Calibration Matters

YOLO outputs a confidence score \\( c \in [0, 1] \\) for each detection. A well-calibrated model satisfies:

\\[P(\text{detection is correct} \mid c = 0.9) \approx 0.9\\]

Most detection models are **miscalibrated**  -  they are overconfident (predict high confidence when wrong) or underconfident (predict low confidence when right). This is why the CivicSense config has a `conf_threshold` parameter that you tune on your validation set, not the training set.

The default threshold is 0.5, but in practice, you want to calibrate this. The procedure:

1. Run your model on the validation set with confidence threshold 0.0.
2. For each threshold \\( t \in [0, 1] \\), compute precision and recall.
3. Choose \(t\) that maximizes \\( F\_1 \\) score or satisfies your precision/recall requirements.
4. For safety-critical alerts (stop signs), bias toward recall (lower threshold, accept more false positives).

## 1.5 The Pinhole Camera Model: From 3D to 2D

The CivicSense pipeline estimates real-world distances from pixel coordinates. This requires the **pinhole camera model**  -  the mathematical relationship between 3D points in the world and their 2D projections onto the image sensor.

### 1.5.1 The Projection Equation

A point \\( \mathbf{P} = (X, Y, Z) \\) in the real world (where \(Z\) is depth along the optical axis) projects to pixel coordinates \((u, v)\) through:

\\[u = f\_x \cdot \frac{X}{Z} + u\_0\\]
\\[v = f\_y \cdot \frac{Y}{Z} + v\_0\\]

where \\( (f\_x, f\_y) \\) is the focal length in pixels and \\( (u\_0, v\_0) \\) is the principal point (usually the image center).

For the distance estimation used in CivicSense, we invert this relationship. Given a known real-world width \(W\) (e.g., a vehicle is approximately 1.8 meters wide), and its pixel width \(w\) in the image:

\\[Z = \frac{f \cdot W}{w}\\]

This is implemented in `src/utils/geometry.rs`:

```rust
pub fn estimate_distance(pixel_width: f32, real_width: f32, focal_length: f32) -> f32 {
    if pixel_width <= 0.0 {
        return f32::MAX;
    }
    (focal_length * real_width) / pixel_width
}
```

**Why this is an approximation:** The formula assumes the object is perpendicular to the optical axis and that its width in the image is proportional to its angular width. For vehicles viewed from behind (common in dashcam footage), this is reasonable. For vehicles at an angle, the pixel width underestimates the true angular width, and our distance estimate is too large (we think they are farther than they are). This is a known limitation documented in the code.

### 1.5.2 The Focal Length: What Is 650 Pixels?

In the default CivicSense config, `focal_length = 650.0`. This is not a physical focal length in millimeters  -  it is the focal length expressed in pixel units:

\\[f\_{\text{pixels}} = f\_{\text{mm}} \cdot \frac{\text{image width in pixels}}{\text{sensor width in mm}}\\]

For a typical dashcam with a 3.7 mm lens and a 1/2.3" sensor (6.17 mm wide) capturing 1280-pixel-wide images:

\\[f\_{\text{pixels}} = 3.7 \cdot \frac{1280}{6.17} \approx 767\\]

The default 650 is a reasonable approximation for a slightly wider-angle lens. In production, you calibrate this using a checkerboard pattern or manufacturer specifications.

## 1.6 The Capstone Connection

Every mathematical concept in this chapter appears directly in the CivicSense codebase:

| Concept | Location in Code | What It Does |
|---------|-----------------|--------------|
| Vector space representation | `yolo.rs` `letterbox()` | Converts image to CHW tensor |
| Matrix multiplication | `ort` `session.run()` | Forward pass through YOLO |
| Dot product / similarity | `deep_sort.rs` IoU matching | Associates detections across frames |
| Covariance matrix | `deep_sort.rs` `KalmanFilter` | Tracks uncertainty in state estimate |
| Chain rule / gradient | Python training script | Backpropagation during training |
| Sigmoid function | `yolo.rs` `sigmoid()` | Decodes YOLO output to [0,1] |
| Bayes' rule (Kalman) | `deep_sort.rs` predict/update | Recursive state estimation |
| Pinhole projection | `geometry.rs` `estimate_distance()` | Converts pixels to meters |
| IoU | `geometry.rs` `compute_iou()` | Association metric for tracking |
| Log-Sum-Exp stability | Python BCE loss | Numerically safe loss computation |

## 1.7 Exercises

1. **Derive the letterbox transform.** Given a \\( 1280 \times 720 \\) image and a target size of \\( 640 \times 640 \\), compute the scale factor, the new dimensions, and the padding. Verify against the `letterbox()` function in `src/detection/yolo.rs`.

2. **Covariance visualization.** Simulate a 2D Kalman filter tracking a point moving in a straight line. Plot the covariance ellipse (the contour of the Gaussian distribution) at each time step to see how uncertainty grows during prediction and shrinks during update.

3. **Distance estimation error analysis.** For a vehicle of real width 1.8 m viewed at a 30-degree angle (so the apparent width is \\( 1.8 \cdot \cos(30^\circ) \\)), compute the distance estimate error at \(Z = 10\) m, \(Z = 25\) m, \(Z = 50\) m using the pinhole formula.

4. **Calibration experiment.** Run the YOLO model on your validation set with `conf_threshold = 0.0` and plot precision-recall curves. Find the threshold that maximizes \\( F\_1 \\) score.

## 1.8 Key Takeaways

- Every image is a vector in a high-dimensional space. The geometry of this space determines what is easy and hard to learn.
- Matrix multiplication, the sigmoid function, and the chain rule account for approximately 95% of all arithmetic in neural network inference and training.
- The covariance matrix is the mathematical expression of uncertainty. Kalman filters are just Bayesian belief updates applied to moving objects.
- The pinhole camera model is a simple geometric relationship that lets you estimate real-world distances from a single camera. It is approximate but sufficient for many ADAS applications.
- Numerical stability is not a detail  -  it is a correctness requirement. Always use log-space computation for probability calculations involving sigmoid or softmax.

In the next chapter, we take these mathematical tools and build a neural network from scratch  -  in typed Python, with forward and backward passes written explicitly, no autograd, no training loops from frameworks. Just matrix multiplication, the chain rule, and sheer determination.
