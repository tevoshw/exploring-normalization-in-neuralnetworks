# 1. What is Normalization

Normalization is the process of transforming values into a different scale (typically with controlled mean and/or variance), while preserving the relative relationships between them — that is, the order and proportions of the data are maintained, only the numerical representation changes.

```
before: [1, 2, 3, 4]
after:  [0, 0.33, 0.67, 1] 
using the math = (x - min) / (max - min)
```

Basically, normalizing means rescaling its magnitude to 1. In neuralnetworks we want to rescaling using mean and variance. In other cases, like L2 normalization we use the norm to rescaling the values of magnitude.

# 2. Why it's necessary

We know that in the backpropagation algorithm, one of the central components is the calculation of the gradient of the Loss with respect to each weight in the network, via the chain rule. For a simple linear layer (y = W·x + b), the derivative of the Loss with respect to the weight W depends directly on the input to that layer:

```
∂L/∂W = ∂L/∂y · x
```

In other words, the magnitude of the gradient is proportional to the magnitude of the input x (which, in a deep network, is the output of the previous layer).

What happens if this input consists of very large values? The resulting gradient will also be very large. In a deep network, this effect accumulates: by the chain rule, the gradient that reaches an early layer is the product of many terms spanning all the subsequent layers:

```
∂L/∂x_1 = ∂L/∂x_L · (∂x_L/∂x_{L-1}) · (∂x_{L-1}/∂x_{L-2}) · ... · (∂x_2/∂x_1)
```

If these terms are consistently greater than 1, the product grows exponentially with depth — this is gradient explosion. This makes training unstable and prevents the network from converging.

The opposite is also a problem: if these terms are consistently smaller than 1, the product shrinks exponentially with depth, causing vanishing gradients, where the network practically stops learning in the earlier layers.

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

## A concrete argument for normalizing before the activation

ReLU creates intentional sparsity: zeroing a neuron is an explicit decision that it doesn't contribute for that input (`ReLU(x) = max(0, x)`). If normalization is applied **after** the activation, z-score normalization can reintroduce non-zero values exactly where ReLU had decided to "turn off" a neuron — partially undoing that sparsity.

Example: take the post-ReLU output `[0, 0, 0, 1]` and apply z-score normalization to it.

```
μ = (0+0+0+1)/4 = 0.25
σ ≈ 0.433

z-score:
0 → (0 - 0.25) / 0.433 ≈ -0.577
0 → -0.577
0 → -0.577
1 → (1 - 0.25) / 0.433 ≈  1.732

result: [-0.577, -0.577, -0.577, 1.732]
```

In this case the zeros stayed negative, so a subsequent ReLU would still kill them. But whether a zero turns positive or negative after z-score depends entirely on the mean of the group. If the surrounding values are different, the same kind of input can flip sign:

```
[0, 0, -5, -3] → μ = -2, σ ≈ 2.29

z-score of 0: (0 - (-2)) / 2.29 ≈ 0.87   ← now positive!
```

So a value that ReLU explicitly zeroed out can come back as positive after normalization, effectively "reviving" a neuron the activation had turned off. This is one of the practical arguments in favor of normalizing **before** the activation: it keeps ReLU's categorical decision (on/off) intact for that layer, instead of letting a later normalization step reinterpret zeros relative to the rest of the batch.

This doesn't break training — the network's weights are optimized end-to-end considering the full pipeline (linear → normalization → activation), so it learns to work within whichever order is used. But it is a real trade-off, not just a theoretical curiosity, and it's part of why "before the activation" remains the more common default.