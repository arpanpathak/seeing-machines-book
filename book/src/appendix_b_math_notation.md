# Appendix B  -  Mathematical Notation Reference

A quick reference for the mathematical notation used in this book.

## B.1 Sets and Spaces

| Symbol | Meaning |
|--------|---------|
| \\( \mathbb{R} \\) | Real numbers |
| \\( \mathbb{R}^{n} \\) | \(n\)-dimensional real vector space |
| \\( \mathbb{R}^{m \times n} \\) | Space of \\( m \times n \\) real matrices |
| \([a, b]\) | Closed interval (inclusive of endpoints) |
| \((a, b)\) | Open interval (exclusive of endpoints) |

## B.2 Vectors and Matrices

| Symbol | Meaning |
|--------|---------|
| \\( \mathbf{x} \\) | Column vector (bold lowercase) |
| \\( \mathbf{X} \\) | Matrix (bold uppercase) |
| \\( \mathbf{x}\_i \\) | The \(i\)-th element of vector \\( \mathbf{x} \\) |
| \\( X\_{i,j} \\) | Element at row \(i\), column \(j\) of matrix \\( \mathbf{X} \\) |
| \\( \mathbf{x}^T \\) | Vector transpose (row vector) |
| \\( \mathbf{X}^T \\) | Matrix transpose |
| \\( \mathbf{I}\_n \\) | \\( n \times n \\) identity matrix |
| \\( \|\mathbf{x}\| \\) | Euclidean norm: \\( \sqrt{x\_1^2 + \cdots + x\_n^2} \\) |
| \\( \mathbf{a} \cdot \mathbf{b} \\) | Dot product: \\( \sum\_i a\_i b\_i \\) |

## B.3 Calculus

| Symbol | Meaning |
|--------|---------|
| \\( \frac{dy}{dx} \\) | Derivative of \(y\) with respect to \(x\) |
| \\( \frac{\partial L}{\partial w} \\) | Partial derivative of \(L\) with respect to \(w\) |
| \\( \nabla L \\) | Gradient vector of \(L\) |
| \\( \nabla\_{\mathbf{W}} L \\) | Gradient of \(L\) with respect to matrix \\( \mathbf{W} \\) |
| \\( \sum\_{i=1}^{n} a\_i \\) | Summation: \\( a\_1 + a\_2 + \cdots + a\_n \\) |

## B.4 Probability and Statistics

| Symbol | Meaning |
|--------|---------|
| \(P(A)\) | Probability of event \(A\) |
| \\( P(A \mid B) \\) | Probability of \(A\) given \(B\) |
| \\( \mathbb{E}[X] \\) | Expected value of random variable \(X\) |
| \\( \mathcal{N}(\mu, \sigma^2) \\) | Normal (Gaussian) distribution with mean \\( \mu \\), variance \\( \sigma^2 \\) |
| \\( \text{Cov}(X, Y) \\) | Covariance of \(X\) and \(Y\) |
| \\( \mathbf{P} \\) | Covariance matrix (in Kalman filter context) |

## B.5 Detection and Tracking

| Symbol | Meaning |
|--------|---------|
| IoU | Intersection-over-Union |
| \\( \mathbf{x} \\) | State vector \\( [c\_x, c\_y, w, h, v\_x, v\_y, v\_w, v\_h]^T \\) |
| \\( \mathbf{z} \\) | Measurement vector \\( [c\_x, c\_y, w, h]^T \\) |
| \\( \mathbf{F} \\) | State transition matrix |
| \\( \mathbf{H} \\) | Measurement (observation) matrix |
| \\( \mathbf{Q} \\) | Process noise covariance |
| \\( \mathbf{R} \\) | Measurement noise covariance |
| \\( \mathbf{K} \\) | Kalman gain matrix |
| \\( \sigma(x) \\) | Sigmoid function: \\( 1 / (1 + e^{-x}) \\) |
| \(c\) | Confidence score |

## B.6 Convolution Notation

| Symbol | Meaning |
|--------|---------|
| \\( H\_{\text{in}}, W\_{\text{in}} \\) | Input height and width |
| \\( H\_{\text{out}}, W\_{\text{out}} \\) | Output height and width |
| \\( C\_{\text{in}}, C\_{\text{out}} \\) | Input and output channels |
| \\( K\_h, K\_w \\) | Kernel height and width |
| \(S\) | Stride |
| \(P\) | Padding |

## B.7 YOLO-Specific Notation

| Symbol | Meaning |
|--------|---------|
| \\( g\_x, g\_y \\) | Grid cell coordinates (in grid-space) |
| \(s\) | Anchor stride (8, 16, or 32) |
| \\( t\_x, t\_y, t\_w, t\_h \\) | Raw network outputs for box coordinates |
| \\( \lambda\_{\text{box}} \\) | Box loss weight (default 7.5) |
| \\( \lambda\_{\text{cls}} \\) | Class loss weight (default 0.5) |
| \\( \lambda\_{\text{DFL}} \\) | Distribution Focal Loss weight (default 1.5) |
| mAP@0.5 | Mean Average Precision at IoU threshold 0.5 |

## B.8 Greek Letters

| Symbol | Usage |
|--------|-------|
| \\( \alpha \\) | Low-pass filter smoothing factor, or DFL tradeoff parameter |
| \\( \theta \\) | Camera pitch angle |
| \\( \rho \\) | Euclidean distance between box centers |
| \\( \sigma \\) | Sigmoid activation function |
| \\( \mu \\) | Mean of a distribution |
| \\( \Sigma \\) | Covariance matrix (alternative notation) |
