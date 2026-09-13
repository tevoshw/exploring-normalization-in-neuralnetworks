# 1. What is Normalization

Normalization is the process of transforming values into a different scale (typically with controlled mean and/or variance), while preserving the relative relationships between them — that is, the order and proportions of the data are maintained, only the numerical representation changes.

```
before: [1, 2, 3, 4]
after:  [0, 0.33, 0.67, 1] 
using the math = (x - min) / (max - min)
```

# 2. Why it's necessary

We know that in the backpropagation algorithm, one of the central components is the calculation of the gradient of the Loss with respect to each weight in the network, via the chain rule. For a simple linear layer (y = W·x + b), the derivative of the Loss with respect to the weight W depends directly on the input to that layer:

```
∂L/∂W = ∂L/∂y · x
```

In other words, the magnitude of the gradient is proportional to the magnitude of the input x (which, in a deep network, is the output of the previous layer).

What happens if this input consists of very large values? The resulting gradient will also be very large — and as it propagates through multiple layers (again via the chain rule), this effect can multiply, causing gradient explosion. This makes training unstable and prevents the network from converging.

The opposite is also a problem: very small inputs lead to very small gradients, causing vanishing gradients, where the network practically stops learning in the earlier layers.

For this reason, controlling the scale of outputs between layers — through normalization — is essential to keep training stable.

# 3. How it works

Normalization works by rescaling the output values of a layer, using statistics such as mean and variance (in the case of z-score/LayerNorm/BatchNorm) or min/max (in the case of min-max scaling), applying a transformation that preserves the relative structure of the data while controlling its scale.

In the context of neural networks, it usually includes learnable parameters (γ and β in LayerNorm, for example) that allow the network to adjust or even partially undo the normalization, if that turns out to be useful for the task.

# 4. Where do we find it?

The position of normalization within a layer is not a fixed rule — different conventions exist:

- **Before the activation**: the original BatchNorm proposal (Linear → BatchNorm → ReLU), which controls the scale of the values entering the non-linearity.
- **After the activation**: also used in practice, with mixed empirical results depending on the architecture.
- **In Transformers**: normalization (LayerNorm) is applied relative to the attention and feed-forward sub-layers. Here, the exact position gives rise to two well-known variants:
  - **Post-LN** (original "Attention is All You Need" paper): `LayerNorm(x + Sublayer(x))` — normalizes after the residual sum.
  - **Pre-LN** (used in most modern models, such as GPT): `x + Sublayer(LayerNorm(x))` — normalizes the input before it passes through the sub-layer, which brings more stability in very deep networks.

In other words: normalization always exists at some point in the flow between layers, but "exactly where" is a design decision that affects training stability, not a universal rule.