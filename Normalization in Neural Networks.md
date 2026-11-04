# Normalization in Neural Networks

*Focus: normalization of activations, with emphasis on Transformers.*

## 1. What is Normalization

Normalization transforms values into a different scale, typically with a controlled mean and/or variance. Two different things go by this name, and it helps to keep them apart:

- **Input (data) normalization**: a preprocessing step applied to the dataset before training (min-max scaling, z-score with dataset statistics). Statistics are fixed.
- **Activation normalization**: a *layer* inside the network (BatchNorm, LayerNorm, RMSNorm, ...). Statistics are computed on the fly from the activations, and the layer usually has learnable parameters.

What is preserved depends on the transformation:

| Transformation | Formula | Preserves |
| --- | --- | --- |
| Min-max scaling | (x - min) / (max - min) | Order, and the ratio between *differences* (it is affine) |
| Z-score / LayerNorm | (x - μ) / σ | Order, and the ratio between differences (affine) |
| RMSNorm | x / rms(x) | Order and the ratio between the *values themselves* (pure rescaling) |

Note: with a negative learnable γ the order of a channel can be inverted, so "order is preserved" holds for the pure transformation, not necessarily after the learnable affine step.

```
before:  [1, 2, 3, 4]
min-max: [0, 0.33, 0.67, 1]      # the 1:2 ratio is NOT preserved, the equal spacing is
```

## 2. Why it is necessary

### 2.1 Numerical stability

Computers store numbers with finite precision. In float16 the largest representable value is 65,504. If a layer produces a larger value, it overflows to inf, which quickly turns into NaN and breaks training. At the other extreme, tiny values (such as 1e-10) underflow to 0 and information is lost. Controlling the scale of activations keeps the arithmetic inside the representable range. (bfloat16 has the same range as float32 but only about 3 significant decimal digits, so scale control still matters there: large activations lose absolute precision.)

### 2.2 Saturation of non-linearities

Functions like sigmoid, tanh and softmax saturate when their inputs are large in magnitude: the output flattens (softmax becomes almost one-hot) and the gradient goes to nearly zero, causing vanishing gradients. Keeping inputs in a moderate range keeps these functions in their useful region.

### 2.3 Backpropagation

For a linear layer y = Wx + b, the gradient of the loss with respect to the weights is:


$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot x^\top$$

Its magnitude is proportional to the magnitude of the input x (in a deep network, the output of the previous layer). Very large inputs therefore produce very large gradients.

In a deep network the effect accumulates. The gradient that reaches an early layer is a product of one Jacobian per subsequent layer:


$$\frac{\partial L}{\partial x_1} = \frac{\partial L}{\partial x_L} \cdot \frac{\partial x_L}{\partial x_{L-1}} \cdot \frac{\partial x_{L-1}}{\partial x_{L-2}} \cdots \frac{\partial x_2}{\partial x_1}$$


These terms are matrices, so "greater or smaller than 1" really refers to their **spectral norm** (largest singular value):

- If the Jacobians consistently have norm **greater than 1**, the product grows exponentially with depth (**exploding gradients**). Training becomes unstable and fails to converge.
- If they consistently have norm **smaller than 1**, the product shrinks exponentially with depth (**vanishing gradients**), and early layers practically stop learning.

### 2.4 Conditioning and learning rate

Stability is not the only benefit. Normalization also makes the optimization problem easier:

- It keeps layer inputs at a consistent scale, which improves the conditioning of the loss surface, so gradient descent behaves more uniformly across directions.
- It makes the network less sensitive to the scale of the weights, which allows **larger learning rates** and faster convergence.

The original explanation for BatchNorm was "reducing internal covariate shift" (Ioffe & Szegedy, 2015). This explanation has been challenged: Santurkar et al. (2018) showed that the main effect is a **smoother loss landscape** (better Lipschitz behavior of the loss and its gradients), not a reduction of distribution shift.

### Summary of this section

Normalization **helps** with vanishing and exploding gradients, improves conditioning and stabilizes training, but it is **not a complete solution** on its own. Initialization, residual connections, gradient clipping and learning-rate schedules also matter.

## 3. How it works

Activation normalization rescales the output of a layer using statistics computed from the data. The variants differ in **which axes the statistics are computed over**. For an activation tensor of shape (B, T, D) (batch, tokens, features):

| Method | Statistics computed over | Notes |
| --- | --- | --- |
| **BatchNorm** | batch (and spatial/token) axes, per feature | Depends on batch size; needs running averages at inference, so train and inference behave differently |
| **LayerNorm** | feature axis D, per token | Independent of batch and of other tokens |
| **RMSNorm** | feature axis D, per token | Like LayerNorm without mean subtraction |
| **GroupNorm** | groups of channels, per sample | Common in CNNs with small batches |
| **InstanceNorm** | spatial axes, per sample and channel | Common in style transfer |

Transformers use LayerNorm or RMSNorm because batch statistics are a poor fit for them: batch sizes can be small (large models), sequences have variable length (padding corrupts the statistics), and per-token statistics make training and inference behave identically.

### Formulas

**LayerNorm** uses the mean μ and variance σ² over the d features of one token:


$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$


**RMSNorm** skips mean subtraction and the bias β, dividing only by the root mean square:


$$\text{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}}$$


- ε is a small constant (e.g. 1e-5 or 1e-6) that avoids division by zero and keeps the computation stable when the variance is tiny.
- γ (and β in LayerNorm) are **learnable** parameters, one per feature. They let the network adjust, or even partially undo, the normalization if that helps the task.
- RMSNorm is cheaper (no mean, no bias) and works just as well in practice, so most modern LLMs use it. Its premise is that the useful property of LayerNorm is the re-scaling invariance, not the re-centering.

### Small numeric example

Take x = \[1, 2, 3, 4\], with γ = 1, β = 0 and ε ≈ 0:

|  | Computation | Result |
| --- | --- | --- |
| LayerNorm | μ = 2.5, σ = sqrt(1.25) ≈ 1.118 | \[-1.342, -0.447, 0.447, 1.342\] |
| RMSNorm | rms = sqrt(7.5) ≈ 2.739 | \[0.365, 0.730, 1.095, 1.461\] |

Both are invariant to scaling the input: \[10, 20, 30, 40\] gives the same outputs as above. Only LayerNorm is also invariant to a **shift**: \[11, 12, 13, 14\] gives the same LayerNorm output, but a different RMSNorm output. RMSNorm preserves the ratios between values (1 : 2 : 3 : 4); LayerNorm does not.

### Reference implementation (PyTorch)

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        # statistics per token, over the feature dimension
        return self.g * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class PreLNBlock(nn.Module):
    def __init__(self, d, attn, mlp):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn, self.mlp = attn, mlp

    def forward(self, x):
        x = x + self.attn(self.n1(x))   # residual path untouched
        x = x + self.mlp(self.n2(x))
        return x
```

In practice, implementations often upcast to float32 inside the norm for numerical safety, then cast back.

### Why normalization stabilizes the gradient

Ignoring ε and γ, RMSNorm is y = x / r with r = rms(x). Its Jacobian is:

$$
J_{norm} = \frac{1}{r}\left(I - \hat{x}\hat{x}^\top\right), \qquad \hat{x} = \frac{x}{\|x\|}$$


Two consequences:

1. The Jacobian **projects out the radial direction** (the component along x): changing only the scale of x does not change the output.
2. Its magnitude scales as **1/r**: the larger the activations, the smaller the gradient passed back through the norm. This acts as a built-in feedback that counteracts growth of the activations, and it is the quantitative version of the argument in Section 2.3.

This J\_norm is the object that appears in the transformer analysis below.

## 4. Problems of normalization

Normalization gives better control of scale, but it has costs:

- **Computational cost**: normalization layers are memory-bound operations (they read and write the whole activation tensor), so stacking many of them slows down training and inference. Kernel fusion mitigates this.
- **Loss of information**: normalizing each token discards its overall magnitude. If the magnitude carries useful information, it is lost. The learnable γ restores a per-channel scale, but not the per-token magnitude.
- **Batch dependence** (BatchNorm only): results depend on batch composition and size, and train/inference behavior differs.
- **Wrong placement**: placing the normalization in the wrong position can make training harder (see the next section).

## 5. Normalization in Transformers

The transformer is the most common place where normalization is used, and the **position** of the norm changes how the gradient flows. For a block with input x, sublayer f (attention or MLP) and output y, what matters is the Jacobian dy/dx, which is built from the Jacobians of the norm and of f (chain rule). Whenever the residual path carries a clean identity term I, the gradient has a direct route to earlier layers.

### Block diagrams

```
Post-LN         Pre-LN          OLMo 2          Gemma 2 / 3

x --+---+       x --+---+       x --+---+       x --+---+
    |   |           |   |           |   |           |   |
    |   f           |  norm         |   f           |  norm1
    |   |           |   |           |   |           |   |
    +--(+)          |   f           |  norm         |   f
        |           |   |           |   |           |   |
      norm          +--(+)          +--(+)          |  norm2
        |               |               |           |   |
        y               y               y           +--(+)
                                                        |
                                                        y
```

### Post-LN (original transformer)


$$
y = \text{norm}(x + f(x)), \qquad \frac{\partial y}{\partial x} = J_{norm} \cdot (I + J_f)
$$

The norm sits on the residual path, so the gradient must cross a normalization at every layer. There is no clean identity path, and the gradient can become unstable in deep models. Xiong et al. (2020) showed that at initialization the gradients near the output are large, which is why Post-LN requires careful learning-rate warmup.

### Pre-LN

$$
y = x + f(\text{norm}(x)), \qquad \frac{\partial y}{\partial x} = I + J_f \cdot J_{norm}
$$



The residual path stays untouched, so the identity term I gives the gradient a direct route to earlier layers. This makes training much more stable (warmup becomes much less critical) and is the most common choice.

Two caveats:

- **Final norm**: since the residual stream itself is never normalized, Pre-LN models apply one extra norm after the last block, before the LM head.
- **Growth with depth ("curse of depth")**: the residual stream is a running sum, so its norm tends to grow with depth. Deeper blocks then see a relatively small contribution to the stream and their Jacobian is close to the identity, so they contribute less than expected. Pre-LN trades Post-LN's instability for this inefficiency in deep layers (see Sun et al., 2025).

### OLMo 2

$$
y = x + \text{norm}(f(x)), \qquad \frac{\partial y}{\partial x} = I + J_{norm} \cdot J_f
$$

And, inside attention:

$$
Q \leftarrow \text{norm}(Q), \qquad K \leftarrow \text{norm}(K)
$$

The norm is applied to the **output** of the sublayer (attention or MLP), and the residual path stays clean. Because the output of every sublayer is normalized, the amount each block adds to the residual stream is bounded by γ, which also limits growth with depth.

OLMo 2 also uses **QK-norm**: Q and K are normalized before the attention scores, which prevents the attention logits from growing too large and saturating the softmax. In OLMo 2 the norm is applied to the full Q and K projections (before splitting into heads), not per head. The norm used is RMSNorm.

### Gemma (2 and 3)

$$
y = x + \text{norm}_2\big(f(\text{norm}_1(x))\big), \qquad \frac{\partial y}{\partial x} = I + J_{norm_2} \cdot J_f \cdot J_{norm_1}
$$

It normalizes both the input and the output of each sublayer, and the residual path stays clean. Gemma uses RMSNorm.

The attention-logit problem is handled differently in the two versions: **Gemma 2 uses soft-capping** of the attention logits, while **Gemma 3 replaces soft-capping with QK-norm**.

### Comparison

| Variant | Formula | Residual path | Notes |
| --- | --- | --- | --- |
| **Post-LN** | norm(x + f(x)) | Normalized every layer | Needs warmup; unstable when deep |
| **Pre-LN** | x + f(norm(x)) | Clean | Stable; needs final norm; residual norm grows with depth |
| **OLMo 2** | x + norm(f(x)) | Clean | Bounded sublayer outputs; QK-norm on full Q, K |
| **Gemma 2** | x + norm2(f(norm1(x))) | Clean | Pre + post norm; attention soft-capping |
| **Gemma 3** | x + norm2(f(norm1(x))) | Clean | Pre + post norm; QK-norm instead of soft-capping |

## 6. Summary

- Normalization controls the **scale** of activations, which gives numerical stability, avoids saturation of softmax and sigmoid, improves conditioning, and allows larger learning rates.
- It **helps** with vanishing and exploding gradients but does not solve them alone: initialization, residual connections, clipping and schedules also matter.
- Variants differ mainly in **which axes the statistics use**. Transformers use per-token LayerNorm or RMSNorm, and RMSNorm dominates modern LLMs because it is cheaper and just as good.
- The **position** of the norm determines gradient flow through the Jacobian dy/dx. Keeping the residual path clean (Pre-LN, OLMo 2, Gemma) gives a direct identity route for the gradient, and is the key to stability in deep networks.
- Each choice has a trade-off: Pre-LN needs a final norm and suffers from residual growth; QK-norm and post-norms add cost but bound attention logits and sublayer outputs.

## References

- Ioffe & Szegedy (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.*
- Ba, Kiros & Hinton (2016). *Layer Normalization.*
- Santurkar et al. (2018). *How Does Batch Normalization Help Optimization?*
- Zhang & Sennrich (2019). *Root Mean Square Layer Normalization.*
- Xiong et al. (2020). *On Layer Normalization in the Transformer Architecture.*
- Sun et al. (2025). *The Curse of Depth in Large Language Models.*
- OLMo Team (2024). *2 OLMo 2 Furious.*
- Gemma Team (2024, 2025). *Gemma 2* and *Gemma 3* technical reports.
