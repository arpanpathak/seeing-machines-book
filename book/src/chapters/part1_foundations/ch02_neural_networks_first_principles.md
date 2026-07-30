# Chapter 2: Neural Networks from First Principles

> *"What I cannot create, I do not understand."* — Richard Feynman

There is a scene in every engineer's life where they realize that a neural network is not magic — it is just a very long chain of matrix multiplications with occasional nonlinearities, and the entire field of deep learning is about making that chain longer without breaking the gradient flow.

This chapter is that realization.

We are going to build a neural network from scratch in typed Python. No PyTorch. No TensorFlow. No autograd. Just `numpy` arrays, manual forward passes, manual backward passes, and the iron discipline of type annotations.

When you finish this chapter, you will understand exactly what happens inside a `model.forward()` call — not because a tutorial told you the math, but because you wrote the math yourself and watched it converge.

## 2.1 The Neuron: A Biological Metaphor That Is Technically Incorrect

A biological neuron receives electrical signals through dendrites, integrates them in the cell body, and fires an action potential down the axon if the integrated signal exceeds a threshold.

An artificial neuron does something mathematically analogous but physiologically dubious:

\\[y = \sigma\left(\sum_{i=1}^{n} w_i x_i + b\right)\\]

where \\( x_i \\) are inputs, \\( w_i \\) are weights, $b$ is a bias, and \\( \sigma \\) is a nonlinear activation function.

The key insight — and the one that took AI research decades to internalize — is that **composition is the secret**. A single neuron can only learn a linear decision boundary (plus the sigmoid curve). But two layers of neurons can approximate any continuous function to arbitrary accuracy (the Universal Approximation Theorem). And deep networks (many layers) can do this with exponentially fewer parameters than shallow networks.

This is not theoretical. When you look at the YOLO backbone in Chapter 4, it has approximately 70 layers of neurons. The first layers learn edges and textures. The middle layers learn object parts (wheels, windows, license plates). The final layers learn whole-object detectors. The composition of simple functions creates complex understanding.

### 2.1.1 The Forward Pass in Type-Annotated Python

Let us define a single neuron as a function, with all types explicit:

```python
import numpy as np
from numpy.typing import NDArray
from typing import Callable

# Type aliases for clarity
Vector = NDArray[np.float64]   # shape (n,)
Matrix = NDArray[np.float64]   # shape (m, n)

def neuron_forward(
    x: Vector,          # input, shape (n_features,)
    w: Vector,          # weights, shape (n_features,)
    b: float,           # bias, scalar
    activation: Callable[[Vector], Vector]  # e.g., sigmoid
) -> float:
    """Forward pass of a single neuron.
    
    Computes z = dot(w, x) + b, then returns activation(z).
    """
    z: float = float(np.dot(w, x)) + b
    return float(activation(np.array([z]))[0])
```

This function is not useful in practice — real networks use vectorized operations across entire batches — but it illustrates the atomic unit of computation.

## 2.2 Building a Layer: Vectorization Is Performance

A single neuron is slow. A **layer** of $m$ neurons, operating on a batch of $b$ inputs simultaneously, is fast — because GPUs and modern CPUs have SIMD units that perform matrix multiplication in hardware.

A fully-connected (dense) layer performs:

\\[\mathbf{Z} = \mathbf{X}\mathbf{W}^T + \mathbf{b}\\]
\\[\mathbf{A} = \sigma(\mathbf{Z})\\]

where \\( \mathbf{X} \in \mathbb{R}^{b \times n} \\), \\( \mathbf{W} \in \mathbb{R}^{m \times n} \\), and \\( \mathbf{b} \in \mathbb{R}^{m} \\).

Note the dimensions carefully:
- The input matrix has rows = batch samples, columns = features.
- The weight matrix has rows = output neurons, columns = input features.
- The bias is broadcast across all batch samples.

### 2.2.1 Implementing a Dense Layer with Types

```python
from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray

class DenseLayer:
    """A fully-connected neural network layer.
    
    y = activation(x @ W.T + b)
    
    Attributes:
        W: Weight matrix, shape (output_dim, input_dim)
        b: Bias vector, shape (output_dim,)
        input_dim: Number of input features
        output_dim: Number of output neurons
        activation: Nonlinearity applied after the affine transform
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        rng: Optional[np.random.Generator] = None
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        
        # Xavier/Glorot initialization: variance = 2 / (fan_in + fan_out)
        # This prevents vanishing/exploding gradients in deep networks.
        scale: float = np.sqrt(2.0 / (input_dim + output_dim))
        
        if rng is None:
            rng = np.random.default_rng()
        
        self.W: NDArray[np.float64] = rng.normal(
            0, scale, size=(output_dim, input_dim)
        ).astype(np.float64)
        self.b: NDArray[np.float64] = np.zeros(output_dim, dtype=np.float64)
        
        # Cached values for backward pass
        self._x: Optional[NDArray[np.float64]] = None
        self._z: Optional[NDArray[np.float64]] = None
    
    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Forward pass: compute activations.
        
        Args:
            x: Input array of shape (batch_size, input_dim)
        
        Returns:
            Activations of shape (batch_size, output_dim)
        """
        self._x = x
        self._z = x @ self.W.T + self.b  # shape (batch_size, output_dim)
        return self.activation(self._z)
```

**What is Xavier initialization?** Named after Xavier Glorot, it sets the initial weight variance to \\( 2 / (n_{\text{in}} + n_{\text{out}}) \\). The intuition: if you initialize weights too large, the activations in deep layers explode toward \\( \pm \infty \\) (sigmoid saturates, gradients vanish). If too small, activations shrink to zero and nothing learns. Xavier initialization keeps the variance of activations roughly constant across layers, which keeps gradients flowing.

## 2.3 The Backward Pass: The Chain Rule, Materialized

Forward propagation computes the output. Backpropagation computes the gradient of the loss with respect to every parameter. The key insight is that the gradient at each layer can be computed from the gradient at the next layer — the chain rule propagated backward.

For a single dense layer \\( \mathbf{A} = \sigma(\mathbf{X}\mathbf{W}^T + \mathbf{b}) \\), the gradients are:

Let \\( \mathbf{Z} = \mathbf{X}\mathbf{W}^T + \mathbf{b} \\) and \\( \mathbf{A} = \sigma(\mathbf{Z}) \\).

If \\( \frac{\partial L}{\partial \mathbf{A}} \\) is known (the "upstream gradient"), then:

\\[\frac{\partial L}{\partial \mathbf{Z}} = \frac{\partial L}{\partial \mathbf{A}} \odot \sigma'(\mathbf{Z})\\]

\\[\frac{\partial L}{\partial \mathbf{W}} = \left(\frac{\partial L}{\partial \mathbf{Z}}\right)^T \mathbf{X}\\]

\\[\frac{\partial L}{\partial \mathbf{b}} = \sum_{\text{batch}} \frac{\partial L}{\partial \mathbf{Z}} \quad \text{(sum over batch dimension)}\\]

\\[\frac{\partial L}{\partial \mathbf{X}} = \frac{\partial L}{\partial \mathbf{Z}} \mathbf{W}\\]

The last gradient (\\( \partial L / \partial \mathbf{X} \\)) is what gets passed to the previous layer.

### 2.3.1 Implementing Backpropagation

```python
def backward(
    self,
    grad_output: NDArray[np.float64]  # upstream gradient, shape (batch, output_dim)
) -> NDArray[np.float64]:             # returns grad_input, shape (batch, input_dim)
    """Backward pass: compute gradients of loss w.r.t. parameters.
    
    Args:
        grad_output: Gradient of loss with respect to layer output.
                     Shape (batch_size, output_dim).
    
    Returns:
        grad_input: Gradient of loss with respect to layer input.
                    Shape (batch_size, input_dim).
                    This is what flows to the previous layer.
    """
    assert self._z is not None, "Must run forward before backward"
    assert self._x is not None, "Must run forward before backward"
    
    batch_size: int = grad_output.shape[0]
    
    # Gradient through activation: dL/dZ = dL/dA * sigma'(Z)
    # For a sigmoid activation: sigma'(z) = sigma(z) * (1 - sigma(z))
    d_activation: NDArray[np.float64]
    if self.activation == sigmoid:
        a: NDArray[np.float64] = sigmoid(self._z)
        d_activation = grad_output * (a * (1.0 - a))
    elif self.activation == relu:
        d_activation = grad_output * (self._z > 0).astype(np.float64)
    else:
        raise ValueError(f"Unknown activation: {self.activation}")
    
    # Gradient w.r.t. weights: dL/dW = dL/dZ^T @ X
    self._grad_W: NDArray[np.float64] = d_activation.T @ self._x  # shape (out, in)
    
    # Gradient w.r.t. bias: sum over batch dimension
    self._grad_b: NDArray[np.float64] = d_activation.sum(axis=0)  # shape (out,)
    
    # Gradient w.r.t. input (to pass to previous layer): dL/dX = dL/dZ @ W
    grad_input: NDArray[np.float64] = d_activation @ self.W  # shape (batch, in)
    
    return grad_input
```

**Why the transpose?** The arrangement of transposes is a common source of bugs. The rule: if \\( \mathbf{Z} = \mathbf{X}\mathbf{W}^T \\), then \\( \partial L / \partial \mathbf{W} = (\partial L / \partial \mathbf{Z})^T \mathbf{X} \\). The shapes work out: \\( \mathbb{R}^{m \times b} \cdot \mathbb{R}^{b \times n} = \mathbb{R}^{m \times n} \\), matching \\( \mathbf{W} \\)'s shape. Always verify your gradient shapes when implementing backprop.

## 2.4 The Loss Function: What Are We Optimizing?

The loss function quantifies "how wrong" the network's prediction is. For object detection, the loss has three components, but for classification (the foundation), we use **cross-entropy**.

### 2.4.1 Cross-Entropy Loss

For a multi-class classification problem with $C$ classes:

\\[\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})\\]

where \\( y_{i,c} \\) is 1 if sample $i$ belongs to class $c$ (0 otherwise), and \\( \hat{y}_{i,c} \\) is the predicted probability.

The gradient of cross-entropy with respect to the logits (input to softmax) has a beautiful closed form:

\\[\frac{\partial \mathcal{L}}{\partial \mathbf{z}_i} = \hat{\mathbf{y}}_i - \mathbf{y}_i\\]

That is: the gradient is simply the difference between the predicted probability distribution and the ground truth distribution. If the network predicts 0.9 for the correct class but the true label is 1.0, the gradient is $0.9 - 1.0 = -0.1$ (pushing the logit up). If it predicts 0.1 for the correct class, the gradient is $0.1 - 1.0 = -0.9$ (a stronger push).

```python
def cross_entropy_loss(
    logits: NDArray[np.float64],   # shape (batch_size, num_classes)
    targets: NDArray[np.int64]     # shape (batch_size,)  — class indices
) -> Tuple[float, NDArray[np.float64]]:
    """Compute cross-entropy loss and its gradient.
    
    Args:
        logits: Raw class scores (not softmaxed).
        targets: Ground-truth class indices in [0, num_classes).
    
    Returns:
        loss: Scalar loss value (averaged over batch).
        grad: Gradient of loss w.r.t. logits, shape (batch_size, num_classes).
    """
    batch_size: int = logits.shape[0]
    
    # Softmax with numerical stability: subtract max logit
    logits_stable: NDArray[np.float64] = logits - logits.max(axis=1, keepdims=True)
    exp_logits: NDArray[np.float64] = np.exp(logits_stable)
    probs: NDArray[np.float64] = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    
    # Loss: negative log probability of the correct class
    correct_log_probs: NDArray[np.float64] = -np.log(
        probs[np.arange(batch_size), targets] + 1e-15  # epsilon to avoid log(0)
    )
    loss: float = float(correct_log_probs.mean())
    
    # Gradient: prob - target_one_hot
    grad: NDArray[np.float64] = probs.copy()
    grad[np.arange(batch_size), targets] -= 1.0
    grad /= batch_size
    
    return loss, grad
```

The `1e-15` epsilon is critical. Without it, if the network becomes overconfident and assigns probability 1.0 to the correct class, \\( \log(1.0) = 0 \\) which is fine. But if it assigns 0.0 to the correct class (which happens when softmax saturates), \\( \log(0) = -\infty \\) which breaks everything. The epsilon ensures numerical stability.

## 2.5 The Training Loop: Putting It All Together

A training step consists of:
1. **Forward pass** through all layers.
2. **Loss computation** at the output.
3. **Backward pass** through all layers (in reverse order).
4. **Parameter update** using computed gradients.

```python
def train_step(
    model: List[DenseLayer],
    x_batch: NDArray[np.float64],    # input, shape (batch, input_dim)
    y_batch: NDArray[np.int64],      # targets, shape (batch,)
    learning_rate: float
) -> float:
    """One step of gradient descent on a batch of data.
    
    Args:
        model: List of layers (forward order).
        x_batch: Input data.
        y_batch: Ground-truth labels.
        learning_rate: Step size for gradient descent.
    
    Returns:
        loss: Scalar loss for this batch.
    """
    # 1. Forward pass
    activations: NDArray[np.float64] = x_batch
    for layer in model:
        activations = layer.forward(activations)
    
    # 2. Loss computation
    loss: float
    grad: NDArray[np.float64]
    loss, grad = cross_entropy_loss(activations, y_batch)
    
    # 3. Backward pass (reverse order)
    for layer in reversed(model):
        grad = layer.backward(grad)
    
    # 4. Parameter update (SGD)
    for layer in model:
        if hasattr(layer, '_grad_W'):
            layer.W -= learning_rate * layer._grad_W
            layer.b -= learning_rate * layer._grad_b
    
    return loss
```

### 2.5.1 Why This Works: The Loss Landscape

The loss function defines a surface in parameter space — an \\( (n_{\text{params}}) \\)-dimensional landscape where each point is a specific weight configuration and the height is the loss value. Gradient descent walks downhill on this landscape.

But here is the uncomfortable truth: for deep networks, this landscape is not a nice convex bowl. It is a rugged, high-dimensional terrain with:
- **Local minima** that are often good enough (contrary to myth, local minima in deep nets are rare; saddle points are the real problem).
- **Saddle points** where the gradient is zero but the curvature is negative in some directions and positive in others.
- **Ravines** where the loss changes rapidly in one direction and slowly in another (this is why momentum helps).

The fact that SGD navigates this landscape at all is a minor miracle, partially explained by the **Lottery Ticket Hypothesis**: within a randomly initialized network, there exists a subnetwork (a "winning ticket") that, if trained in isolation, can match the performance of the full network. SGD finds one of these winning tickets.

## 2.6 The Capstone Connection: From Toy Networks to YOLO

The neural network you just built in this chapter — with its dense layers, sigmoid activations, cross-entropy loss, and SGD optimization — is the same fundamental machinery that drives YOLOv11, just scaled up by several orders of magnitude and augmented with specialized operations:

| Your Implementation | YOLOv11 Equivalent | Purpose |
|---------------------|-------------------|---------|
| `DenseLayer` | `nn.Conv2d` (convolutional) | Feature extraction with spatial structure |
| `sigmoid` activation | `SiLU` / `Sigmoid` | Nonlinearity (SiLU = sigmoid * x) |
| Cross-entropy loss | `BCEWithLogitsLoss` + CIoU + DFL | Multi-component detection loss |
| SGD optimizer | `AdamW` + cosine LR schedule | Adaptive learning rate per parameter |
| Batch of samples | Batch of images + augmentations | Data parallelism and regularization |

But the pipeline is identical:
1. Forward pass to compute predictions.
2. Compute loss between predictions and ground truth.
3. Backward pass to compute gradients.
4. Update parameters.

When you train a YOLO model in Chapter 5, you will not write the backward pass manually — PyTorch's autograd handles that. But you will understand what `loss.backward()` does because you have written it yourself. And when your model does not converge, you will debug the right thing: not "is the magic framework working?" but "is the gradient flowing through my architecture?"

## 2.7 Exercises

1. **Backprop by hand.** Compute the forward and backward pass for a 2-layer network (input \\( \to \\) hidden \\( \to \\) output) with sigmoid activations for a single sample. Verify that your hand-computed gradients match your autograd implementation.

2. **Implement a 3-layer network.** Extend the code from this chapter to add a hidden layer. Train it on the Iris dataset and achieve >90% accuracy. Type-annotate everything.

3. **Gradient checking.** Write a function that numerically approximates gradients using finite differences: \\( \frac{\partial L}{\partial w_i} \approx \frac{L(w_i + \epsilon) - L(w_i - \epsilon)}{2\epsilon} \\). Verify that your backprop gradients match the numerical gradients to within \\( 1 \times 10^{-5} \\).

4. **Visualize the loss landscape.** Train a 2-layer network on a 2D synthetic dataset. After training, sample the loss on a 2D slice of parameter space and plot it as a contour map. Identify the minimum your SGD found.

## 2.8 Key Takeaways

- A neural network is a composition of affine transformations and nonlinearities. The composition is what gives it representational power.
- Backpropagation is the chain rule applied efficiently to compute gradients through a computation graph.
- The gradient of cross-entropy loss with softmax is the simple difference \\( (\hat{y} - y) \\), which makes training numerically well-behaved.
- Xavier initialization is essential for deep networks. Without it, gradients vanish or explode.
- Type annotations in Python are not just documentation — they prevent shape-mismatch bugs that would otherwise silently produce wrong results.

In the next chapter, we generalize from dense layers to convolutional layers — the operation that made computer vision possible by exploiting spatial locality in images.
