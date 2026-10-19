# Normalization in Neural Networks

## 1. What is Normalization

Normalization transforms values into a different scale (typically with a controlled mean and/or variance) while preserving the relative relationships between them: the order and proportions are maintained, and only the numerical representation changes.

```
before: [1, 2, 3, 4]
after:  [0, 0.33, 0.67, 1]    (min-max scaling)
```

## 2. Why it is necessary

### 2.1 Numerical stability

Computers store numbers with finite precision. In float16, the largest representable value is 65,504. If a layer produces a larger value, it overflows to `inf`, which quickly turns into `NaN` and breaks the training. At the other extreme, tiny values (such as 10⁻¹⁰) underflow to 0 and the information is lost. Controlling the scale of activations keeps the arithmetic inside the representable range. (bfloat16 has the same range as float32, but much lower precision, so scale control still matters there.)

### 2.2 Saturation of non-linearities

Functions like sigmoid and softmax saturate when their inputs are large in magnitude: the output flattens (softmax becomes almost one-hot) and the gradient goes to nearly zero, which causes vanishing gradients. Keeping the inputs in a moderate range keeps these functions in their useful region.

### 2.3 Backpropagation

In backpropagation, the gradient of the loss with respect to each weight is computed with the chain rule. For a linear layer $y = Wx + b$:

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot x^\top$$

The magnitude of the gradient is proportional to the magnitude of the input $x$ (which, in a deep network, is the output of the previous layer). Very large inputs therefore produce very large gradients.

In a deep network this effect accumulates. The gradient that reaches an early layer is a product of many terms, one for each subsequent layer:

$$\frac{\partial L}{\partial x_1} = \frac{\partial L}{\partial x_L} \cdot \frac{\partial x_L}{\partial x_{L-1}} \cdot \frac{\partial x_{L-1}}{\partial x_{L-2}} \cdots \frac{\partial x_2}{\partial x_1}$$

- If these terms are consistently **greater than 1**, the product grows exponentially with depth (**exploding gradients**). Training becomes unstable and fails to converge.
- If they are consistently **smaller than 1**, the product shrinks exponentially with depth (**vanishing gradients**), and the early layers practically stop learning.

### Summary

Normalization **helps** with vanishing and exploding gradients and stabilizes training, but it is **not a complete solution** on its own. Initialization, residual connections and learning-rate schedules also matter.

## 3. How it works

Normalization rescales the output of a layer using statistics of the data:

- **Min-max scaling**: uses the minimum and maximum values.
- **Z-score / LayerNorm / BatchNorm**: uses the mean $\mu$ and variance $\sigma^2$:

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

- **RMSNorm**: skips the mean subtraction and divides only by the root mean square. It is cheaper and works just as well in practice, so most modern LLMs use it:

$$\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}}$$

In neural networks, the transformation includes learnable parameters ($\gamma$, and $\beta$ in LayerNorm). They let the network adjust, or even partially undo, the normalization if that is useful for the task.

In transformers, the statistics are computed **per token, across the feature dimension**, so the result does not depend on batch size or on other tokens.

## 4. Problems of normalization

Normalization gives better control of scale, but it has costs:

- **Computational cost**: normalization layers are memory-bound operations (they read and write the whole activation tensor), so stacking many of them slows down training and inference. Kernel fusion mitigates this.
- **Loss of information**: normalizing each token discards its overall magnitude. If the magnitude carries useful information, it is lost. The learnable $\gamma$ restores a per-channel scale, but not the per-token magnitude.
- **Wrong placement**: placing the normalization in the wrong position can make training harder (see the next section).

## 5. Normalization in Transformers

The transformer is the most common place where normalization is used, and the **position** of the norm changes how the gradient flows. For a block with input $x$, sublayer $f$ (attention or MLP) and output $y$, what matters is the Jacobian $\frac{\partial y}{\partial x}$, which is a product of the Jacobians of the norm and of $f$ (chain rule).

### Post-LN (original transformer)

$$y = \text{norm}(x + f(x))$$

$$\frac{\partial y}{\partial x} = J_{norm} \cdot (I + J_f)$$

The norm sits on the residual path, so the gradient must cross a normalization at every layer. There is no clean identity path, and the gradient can become unstable in deep models. In practice, it requires careful learning-rate warmup.

### Pre-LN

$$y = x + f(\text{norm}(x))$$

$$\frac{\partial y}{\partial x} = I + J_f \cdot J_{norm}$$

The residual path stays untouched, so the identity term $I$ gives the gradient a direct route to earlier layers. This makes training much more stable and is the most common choice.

### OLMo 2

$$y = x + \text{norm}(f(x))$$

$$Q, K = \text{norm}(Q), \text{norm}(K)$$

$$\frac{\partial y}{\partial x} = I + J_{norm} \cdot J_f$$

The norm is applied to the **output** of the sublayer, and the residual path stays clean. It also uses **QK-norm**: Q and K are normalized (separately) before the attention scores, which prevents the attention logits from growing too large and saturating the softmax. OLMo uses RMSNorm.

### Gemma (2 and 3)

$$y = x + \text{norm}_2(f(\text{norm}_1(x)))$$

$$\frac{\partial y}{\partial x} = I + J_{norm_2} \cdot J_f \cdot J_{norm_1}$$

It normalizes both the input and the output of each sublayer, and the residual path stays clean. Gemma 3 also uses QK-norm. Gemma uses RMSNorm.