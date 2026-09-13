# Types of Normalization

When talking about normalization in neural networks, there are two different axes to consider:

1. **Where / over what** the statistics (mean, variance, etc.) are calculated — BatchNorm, LayerNorm, InstanceNorm, GroupNorm...
2. **Which formula** is used to transform the values — z-score, min-max, RMSNorm...

These two axes are independent: most "where" types use the same z-score formula underneath, while RMSNorm changes the formula itself and can be applied in a "per-sample" style similar to LayerNorm.

---

# 1. Types by "where" the statistics are calculated

Imagine a batch represented as a matrix, where each **row** is a sample and each **column** is a feature/neuron:

```
           feature1  feature2  feature3
sample1  [   2,        10,        -1   ]
sample2  [   4,         8,         3   ]
sample3  [   6,        12,         1   ]
```

## BatchNorm

Normalizes **column by column**. For each feature, it computes μ and σ by looking at that feature's values across **all samples in the batch**.

```
feature1: μ=4, σ calculated using [2, 4, 6]
feature2: μ=10, σ calculated using [10, 8, 12]
feature3: μ=1, σ calculated using [-1, 3, 1]
```

In other words: "how does this neuron behave, on average, across the whole batch?"

**Limitations**: depends on having a batch that's large enough to give meaningful statistics. Performs poorly with small batches (or batch size = 1, e.g. during inference on a single example), and is awkward for variable-length sequences, since padding distorts the per-column statistics.

## LayerNorm

Normalizes **row by row**. For each sample, it computes μ and σ by looking at **all the features of that single example**.

```
sample1: μ and σ calculated using [2, 10, -1]
sample2: μ and σ calculated using [4, 8, 3]
sample3: μ and σ calculated using [6, 12, 1]
```

In other words: "how does this specific example distribute across its own features?"

**Advantages**: doesn't depend on other examples in the batch — each sample normalizes independently, so behavior is identical during training and inference, works with any batch size, and is robust to variable sequence lengths. This is why it's the standard choice in Transformers/NLP.

## InstanceNorm

Normalizes each sample **and each channel** individually (common in computer vision, e.g. style transfer). It's like LayerNorm but restricted to spatial dimensions within a single channel, rather than across all features.

## GroupNorm

Splits channels into groups and normalizes within each group — a middle ground between BatchNorm and InstanceNorm. Useful when the batch size is small (e.g. object detection, where large batches don't fit in memory).

## Quick summary

| Type        | Computed over                          | Batch-dependent? |
|-------------|------------------------------------------|-------------------|
| BatchNorm   | same feature, across all samples (column)| Yes               |
| LayerNorm   | all features, within one sample (row)    | No                |
| InstanceNorm| one sample, one channel                  | No                |
| GroupNorm   | one sample, group of channels            | No                |

---

# 2. Types by formula

## Z-score (standardization)

The most common formula, used by both BatchNorm and LayerNorm:

```
y = (x - μ) / σ
```

Centers the data around 0 (mean) with unit variance. Usually followed by learnable parameters γ and β:

```
y = γ * (x - μ) / √(σ² + ε) + β
```

`ε` is a small constant to avoid division by zero; `γ` and `β` let the network rescale or shift the normalized output if that's useful for the task — effectively allowing it to partially undo the normalization.

## Min-max scaling

Maps values into a fixed range, usually [0, 1]:

```
y = (x - min) / (max - min)
```

Simple and preserves proportions relative to the range, but is sensitive to outliers (a single extreme value compresses everything else) and is not commonly used inside neural network layers — more common in data preprocessing.

## RMSNorm (Root Mean Square Normalization)

A simplified variant of z-score that skips re-centering (doesn't subtract the mean) and only rescales by the root-mean-square magnitude:

```
y = x / RMS(x)        where RMS(x) = √(mean(x²) + ε)
```

Often paired with a learnable scale parameter γ:

```
y = γ * x / RMS(x)
```

Cheaper to compute than full z-score (no mean subtraction), and used in modern LLMs such as LLaMA in place of LayerNorm.



## Quick summary

| Formula       | Subtracts mean? | Output range        | Notes                          |
|---------------|------------------|----------------------|---------------------------------|
| Z-score       | Yes              | Unbounded (≈ -3 to 3)| Standard in BatchNorm/LayerNorm|
| Min-max       | Yes (via min)    | [0, 1] (fixed)       | Sensitive to outliers           |
| RMSNorm       | No               | Unbounded            | Cheaper, used in LLaMA-style models |
