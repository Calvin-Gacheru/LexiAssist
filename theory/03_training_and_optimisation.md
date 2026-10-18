# 03 · Training and optimisation

Companion to the notebook's "Loss, backpropagation and optimisation", "Transfer learning, early stopping and the metrics helper" and "Reading learning curves" cells.

## 1. Where cross-entropy comes from

Treat the network as a model of $P(y \mid x)$. **Maximum likelihood** picks the parameters that make the observed labels most probable:

$$\theta^* = \arg\max_\theta \prod_i p_\theta(y_i \mid x_i) = \arg\min_\theta \sum_i -\log p_\theta(y_i \mid x_i)$$

The right-hand side is exactly the summed cross-entropy. So minimising cross-entropy *is* maximum-likelihood estimation. An information-theory view says the same thing: cross-entropy $H(q, p) = -\sum_k q_k \log p_k$ measures how many bits are wasted coding the true labels $q$ with the model's distribution $p$.

**Why not mean squared error on probabilities?** Combined with softmax, it gives tiny gradients when the model is confidently wrong (the softmax is saturated). Cross-entropy's gradient $p - \mathbb 1_y$ stays large exactly when the model is badly wrong.

## 2. Derivation of the softmax + cross-entropy gradient

With $p_k = e^{z_k}/\sum_j e^{z_j}$ and $\ell = -\log p_y$:

$$\frac{\partial p_y}{\partial z_k} = p_y(\mathbb 1[k=y] - p_k) \quad\Rightarrow\quad \frac{\partial \ell}{\partial z_k} = -\frac{1}{p_y}\cdot p_y(\mathbb 1[k=y]-p_k) = p_k - \mathbb 1[k=y]$$

**Numerical stability.** Computing $e^{z}$ directly overflows for large logits. `CrossEntropyLoss` uses the log-sum-exp trick: $\log\sum_j e^{z_j} = m + \log\sum_j e^{z_j - m}$ with $m = \max_j z_j$. This is another reason to pass it logits, not probabilities.

## 3. Backpropagation on a tiny network

Take one input $x = 2$, one hidden ReLU unit and one output, with a squared-error loss to keep the arithmetic short:

$$h = \mathrm{ReLU}(w_1 x), \qquad \hat y = w_2 h, \qquad \ell = \tfrac12(\hat y - y)^2$$

With $w_1 = 0.5$, $w_2 = 3$, $y = 1$:

| Forward | Backward (chain rule) |
|---|---|
| $w_1 x = 1$, $h = 1$ | $\partial\ell/\partial\hat y = \hat y - y = 2$ |
| $\hat y = 3$ | $\partial\ell/\partial w_2 = 2\cdot h = 2$ |
| $\ell = \tfrac12(3-1)^2 = 2$ | $\partial\ell/\partial h = 2\cdot w_2 = 6$ |
| | $\partial\ell/\partial w_1 = 6\cdot \mathbb 1[w_1x>0]\cdot x = 12$ |

Each gradient reuses the one computed just before it. That reuse is why backpropagation costs about the same as one forward pass, whatever the number of parameters.

## 4. Optimisers compared

| Optimiser | Update | Strengths | Weaknesses |
|---|---|---|---|
| SGD | $\theta \leftarrow \theta - \eta g$ | Simple; often generalises well | Sensitive to LR; slow in narrow valleys |
| SGD + momentum | $v \leftarrow \mu v + g$; $\theta \leftarrow \theta - \eta v$ | Smooths noise; faster along consistent directions | One more hyperparameter |
| Adam | Per-parameter step $\hat m/(\sqrt{\hat v}+\epsilon)$ | Works well with little tuning | L2 regularisation interacts badly with the adaptive scaling |
| **AdamW** | Adam + separate $-\eta\lambda\theta$ | Weight decay behaves as intended | |

**Bias correction.** $m_0 = v_0 = 0$, so the early averages are biased towards zero. Adam divides by $1-\beta^t$:

$$\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}$$

**Why decoupling matters** (Loshchilov & Hutter, 2019). Adding $\frac{\lambda}{2}\lVert\theta\rVert^2$ to the loss puts $\lambda\theta$ into $g$. Adam then divides it by $\sqrt{\hat v}$, so parameters with large gradients get *less* regularisation. AdamW applies the decay directly, uniformly for every parameter.

## 5. Learning-rate schedules

| Schedule | Shape | Notes |
|---|---|---|
| Constant | flat | Simple; may never settle |
| Step decay | drops ×0.1 at set epochs | Needs the drop epochs chosen by hand |
| **Cosine annealing** (used) | smooth half-cosine to 0 | No step epochs to choose |
| Warm-up + cosine | short linear ramp, then cosine | Helps when early gradients are wild |
| One-cycle | up then down | Fast convergence; popular for fine-tuning |

The two-stage setup here restarts the cosine at stage 2 with a new, lower peak. That is a simple form of the *warm restarts* in SGDR (Loshchilov & Hutter, 2017).

## 6. Regularisation toolkit used in this project

| Technique | Where | Effect |
|---|---|---|
| Data augmentation | `RandomAffineBG`, `ColorJitter` | More varied training inputs |
| Dropout | 0.4 (SmallCNN), 0.3 (ResNet18 head) | Discourages co-adaptation |
| Weight decay | `weight_decay=1e-4` | Prefers small weights |
| Early stopping | `patience=7` on validation loss | Stops before memorisation |
| Transfer learning + freezing | stage 1 | Restricts the hypothesis space |
| Low LR in fine-tuning | stage 2 | Stays near the pretrained solution |

## 7. Bias–variance, the derivation

For squared error, with $y = f(x) + \varepsilon$, noise variance $\sigma^2$, and $\hat f$ trained on a random training set $D$:

$$\mathbb E_{D,\varepsilon}\big[(y-\hat f(x))^2\big] = \underbrace{\big(f(x) - \mathbb E_D[\hat f(x)]\big)^2}_{\text{bias}^2} + \underbrace{\mathbb E_D\big[(\hat f(x) - \mathbb E_D[\hat f(x)])^2\big]}_{\text{variance}} + \sigma^2$$

Variance is "how much the model would change if you trained it on a different sample of children". With a few hundred images it is large, which shows up as the fold-to-fold standard deviation in Phase 5.

**A modern caveat.** Very large networks can show "double descent": test error can fall again past the point of interpolating the training data (Belkin et al., 2019). The classic U-shaped curve is still the right mental model for a small dataset with early stopping.

## Exercises

1. Logits $(0.0, 0.0)$, true class 1. What are the loss and the gradient with respect to the logits?
2. Counts are 200 negative and 50 positive. Compute the class weights as in `train_model`.
3. Stage 1 uses LR $10^{-3}$ for 8 epochs with cosine annealing. What is the LR after epoch 4 (i.e. when `sched.step()` has been called 4 times)?
4. Training loss keeps falling, validation loss rose for 7 epochs in a row. What does the code do, and which weights end up in the model?

<details><summary>Answers</summary>

1. $p = (0.5, 0.5)$; $\ell = -\log 0.5 \approx 0.693$; gradient $(0.5, -0.5)$.
2. $N = 250$: $w_0 = 250/400 = 0.625$, $w_1 = 250/100 = 2.5$.
3. $\tfrac12\cdot10^{-3}(1+\cos(\pi\cdot 4/8)) = 0.5\times10^{-3}$.
4. `wait` reaches `patience` and the stage loop breaks. The weights from the epoch with the lowest validation loss (stored in `best_state`) are loaded back.

</details>

## Further reading

- Goodfellow, Bengio & Courville, *Deep Learning* (2016), chapters 6 (feedforward networks and backprop), 7 (regularisation) and 8 (optimisation). Free online.
- Kingma & Ba (2015), "Adam: A Method for Stochastic Optimization", *ICLR*.
- Loshchilov & Hutter (2019), "Decoupled Weight Decay Regularization", *ICLR*.
- Loshchilov & Hutter (2017), "SGDR: Stochastic Gradient Descent with Warm Restarts", *ICLR*.
- Prechelt (1998), "Early Stopping: But When?", in *Neural Networks: Tricks of the Trade*.
- Belkin et al. (2019), "Reconciling modern machine-learning practice and the classical bias–variance trade-off", *PNAS*.
