# 02 · CNNs and ResNet

Companion to the notebook's "Convolutional neural networks" and "ResNet18 and residual connections" cells.

## 1. Why convolution? Counting parameters

Compare a first layer producing 16 output channels from a 3 × 224 × 224 image:

| Layer type | Parameters | Calculation |
|---|---|---|
| Fully connected to 16 × 224 × 224 outputs | ≈ 1.2 × 10¹¹ | $(3\cdot224^2)\cdot(16\cdot224^2)$ |
| Fully connected to just 16 outputs | 2,408,464 | $3\cdot224^2\cdot16 + 16$ |
| Conv 3×3, 16 filters | **448** | $16\cdot(3\cdot3\cdot3) + 16$ |

Convolution gets its efficiency from two assumptions:

- **Local connectivity**: each output depends only on a small neighbourhood.
- **Weight sharing**: the same filter is used at every position.

Together they give **translation equivariance**: shift the input, and the feature map shifts the same way. Pooling and global average pooling then add some translation *invariance* (the final answer barely changes).

## 2. Parameter count of a conv layer

$$\#\text{params} = C_{\text{out}}\cdot(C_{\text{in}}\cdot k\cdot k) + C_{\text{out}}$$

| SmallCNN layer | Calculation | Parameters |
|---|---|---|
| Conv(3→16) | 16·27 + 16 | 448 |
| BN(16) | 2·16 ($\gamma, \beta$) | 32 |
| Conv(16→16) | 16·144 + 16 | 2,320 |
| BN(16) | | 32 |
| **Block 1 total** | | **2,832** |
| Block 2 (16→32) | | 14,016 |
| Block 3 (32→64) | | 55,680 |
| Block 4 (64→128) | | 221,952 |
| Linear(128→2) | 256 + 2 | 258 |
| **SmallCNN total** | | **294,738** |

Notice that the last block holds 75% of the parameters: channel counts grow, and parameters grow with $C_{\text{in}}\cdot C_{\text{out}}$.

## 3. Receptive field, the general rule

Let $r_l$ be the receptive field after layer $l$ and $j_l$ the *jump* (the distance in input pixels between neighbouring outputs). For a layer with kernel $k$ and stride $s$:

$$r_l = r_{l-1} + (k - 1)\,j_{l-1}, \qquad j_l = j_{l-1}\cdot s, \qquad r_0 = j_0 = 1$$

SmallCNN, layer by layer:

| Layer | k | s | j (after) | r (after) |
|---|---|---|---|---|
| input | | | 1 | 1 |
| conv 3×3 | 3 | 1 | 1 | 3 |
| conv 3×3 | 3 | 1 | 1 | 5 |
| maxpool 2 | 2 | 2 | 2 | 6 |
| conv, conv | 3 | 1 | 2 | 10, 14 |
| maxpool 2 | 2 | 2 | 4 | 16 |
| conv, conv | 3 | 1 | 4 | 24, 32 |
| maxpool 2 | 2 | 2 | 8 | 36 |
| conv, conv | 3 | 1 | 8 | 52, 68 |
| maxpool 2 | 2 | 2 | 16 | **76** |

The *effective* receptive field is smaller than this theoretical one: influence is concentrated near the centre, falling off roughly like a Gaussian (Luo et al., 2016).

## 4. Batch normalisation in more depth

**Training mode.** For each channel, BN uses the mini-batch mean and variance (computed over batch, height and width) and updates running estimates:

$$\mu_{\text{run}} \leftarrow (1-m)\,\mu_{\text{run}} + m\,\mu_{\mathcal B}, \qquad m = 0.1 \text{ (PyTorch default)}$$

**Evaluation mode.** BN uses $\mu_{\text{run}}, \sigma^2_{\text{run}}$, so each image's output no longer depends on the other images in its batch.

**Why it helps.** The original paper's explanation, reducing "internal covariate shift", is disputed. Later work (Santurkar et al., 2018) argues that BN mainly smooths the loss landscape, allowing larger learning rates. Either way, the practical effect is faster, more stable training.

**Pitfall with small data.** If training and test images differ systematically (for example a different camera), the running statistics no longer match the test images, which hurts accuracy.

## 5. Dropout as an ensemble

With $n$ droppable units, dropout samples one of $2^n$ "thinned" networks at every step. Evaluation with all units (and the $1/(1-p)$ scaling done during training) approximates averaging that huge ensemble. Larger $p$ means stronger regularisation, but also slower learning.

## 6. Residual networks: the gradient argument

Stack $L$ residual blocks, $x_{l+1} = x_l + F_l(x_l)$. Unrolling gives

$$x_L = x_l + \sum_{i=l}^{L-1} F_i(x_i) \quad\Rightarrow\quad \frac{\partial \mathcal L}{\partial x_l} = \frac{\partial \mathcal L}{\partial x_L}\left(1 + \frac{\partial}{\partial x_l}\sum_{i=l}^{L-1}F_i(x_i)\right)$$

The "1" means the gradient from the loss reaches every earlier layer directly, without passing through a chain of multiplications (He et al., 2016b). In a plain network, the gradient is the product of all the Jacobians, which can shrink or explode exponentially with depth.

**Degradation problem.** He et al. (2016a) observed that a plain 56-layer network had higher *training* error than a 20-layer one. That isn't overfitting: the deeper network was harder to optimise. Residual connections removed the gap.

**Downsampling blocks.** When a block halves spatial size and doubles channels (the first block of `layer2`, `layer3` and `layer4`), $x$ and $F(x)$ have different shapes. ResNet18 then uses a 1×1 convolution with stride 2 (plus BN) on the shortcut so the two can be added.

## 7. ResNet18 parameter budget

| Part | Parameters (approx.) |
|---|---|
| conv1 + bn1 | 9.5 k |
| layer1 | 148 k |
| layer2 | 526 k |
| layer3 | 2.1 M |
| layer4 | 8.4 M |
| fc (512 → 2) | 1 k |
| **Total** | **≈ 11.18 M** |

Three quarters of the parameters sit in `layer4`, the most ImageNet-specific layer. Fine-tuning `layer4` on a few hundred images is therefore where most overfitting risk lies.

## Exercises

1. Output size of `Conv2d(3, 64, kernel_size=7, stride=2, padding=3)` on a 224 × 224 input?
2. How many parameters does that layer have (with bias)?
3. A friend removes all pooling from SmallCNN. What happens to the receptive field, to memory use, and to the parameter count?
4. Why can't Grad-CAM in `explain.py` be applied to SmallCNN without changes?

<details><summary>Answers</summary>

1. $\lfloor(224 + 6 - 7)/2\rfloor + 1 = 112$.
2. $64\cdot(3\cdot7\cdot7) + 64 = 9{,}472$. (ResNet's real conv1 has `bias=False`, so 9,408.)
3. The receptive field grows only 2 pixels per conv, giving $1 + 8\cdot2 = 17$ pixels. Feature maps stay 224 × 224, so memory and compute balloon, roughly 16× for the deepest layers. The parameter count is unchanged (GAP still produces 128 values).
4. It targets `model.layer4[-1]`, which doesn't exist in SmallCNN. You would target `model.features[-1]` (or a specific conv inside it) instead.

</details>

## Further reading

- He, Zhang, Ren & Sun (2016a), "Deep Residual Learning for Image Recognition", *CVPR*.
- He, Zhang, Ren & Sun (2016b), "Identity Mappings in Deep Residual Networks", *ECCV*.
- Ioffe & Szegedy (2015), "Batch Normalization", *ICML*; Santurkar et al. (2018), "How Does Batch Normalization Help Optimization?", *NeurIPS*.
- Srivastava et al. (2014), "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", *JMLR*.
- Lin, Chen & Yan (2014), "Network In Network" (introduces global average pooling), *ICLR*.
- Luo et al. (2016), "Understanding the Effective Receptive Field in Deep CNNs", *NeurIPS*.
- Dumoulin & Visin (2016), "A guide to convolution arithmetic for deep learning" (excellent diagrams).
