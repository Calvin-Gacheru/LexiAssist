# 07 · Explainability

Companion to the notebook's "Grad-CAM" and "Integrated Gradients" cells.

## 1. A map of explanation methods

| Family | Examples | Question answered | Cost |
|---|---|---|---|
| Gradient-based | Vanilla gradient, Gradient × Input, SmoothGrad | How sensitive is the output to each pixel? | 1 backward pass (×N for SmoothGrad) |
| Path-based | **Integrated Gradients** | How much did each pixel contribute, relative to a baseline? | `n_steps` backward passes |
| CAM-based | CAM, **Grad-CAM**, Grad-CAM++ | Which regions of a conv layer's map support the class? | 1 backward pass |
| Perturbation | Occlusion, RISE, LIME | What happens to the output if I hide this region? | Many forward passes |
| Example-based | Nearest neighbours, prototypes | Which training images does this resemble? | Depends |

LexiAssist uses one CAM method and one path method because they rest on different assumptions. Agreement between them is weak evidence; disagreement is a useful warning.

## 2. From CAM to Grad-CAM

**CAM** (Zhou et al., 2016) works for networks ending in GAP → linear. The class score is

$$y^c = \sum_k w_k^c \cdot \frac{1}{Z}\sum_{i,j}A^k_{ij} = \frac1Z\sum_{i,j}\underbrace{\sum_k w_k^c A^k_{ij}}_{\text{CAM}(i,j)}$$

So the class score is literally the average of a spatial map, and that map is the explanation.

**Grad-CAM** (Selvaraju et al., 2017) generalises CAM to any architecture by replacing the weights $w_k^c$ with gradients averaged over space, $\alpha_k^c$. For a network that *is* GAP → linear, like ResNet18 here, $\alpha_k^c = w_k^c / Z$. Grad-CAM then equals CAM up to a constant factor and the ReLU.

The dropout in the `fc` head is inactive in `eval()` mode, so this equivalence holds for LexiAssist's ResNet18.

## 3. Why the ReLU in Grad-CAM?

Negative values mark regions whose activation pushes the score *down*, i.e. evidence for the other class. Grad-CAM keeps only positive evidence for the chosen class. To see evidence for "low indicators", compute Grad-CAM for class 0 instead; it is not simply the inverse of the class-1 map.

## 4. Integrated Gradients: the axioms

Sundararajan, Taly & Yan (2017) argue that attribution methods should satisfy:

| Axiom | Meaning | Plain gradient | IG |
|---|---|---|---|
| **Completeness** | Attributions sum to $F(x) - F(x')$ | ✗ | ✓ |
| **Sensitivity** | If input and baseline differ in one feature and the outputs differ, that feature gets non-zero attribution | ✗ (fails under saturation) | ✓ |
| **Implementation invariance** | Two networks computing the same function get the same attributions | ✓ | ✓ |
| Linearity | Attributions of a sum of models = sum of attributions | ✓ | ✓ |

**Saturation example.** $F(x) = \min(1, x)$ at $x = 2$ has gradient 0, so the plain gradient says $x$ doesn't matter. IG with baseline 0 integrates the gradient along the path from 0 to 2: it is 1 for $\alpha x < 1$ and 0 afterwards. The integral is $\int_0^1 \mathbb 1[2\alpha < 1]\,d\alpha = 0.5$, so $\mathrm{IG} = 2 \times 0.5 = 1 = F(2) - F(0)$. Completeness holds.

## 5. Baselines matter

| Baseline | Meaning | Problem |
|---|---|---|
| Zeros in normalised space (used) | ImageNet-mean gray | Blank paper differs from it, so paper gets attribution |
| White image | Blank paper | Natural for handwriting; ink pixels get the attribution |
| Black image | No light | Paper gets large attribution |
| Blurred input | Same image without fine detail | Highlights fine detail specifically |
| Average over random baselines | Expected gradients (Erion et al., 2021) | More compute |

For handwriting, a white-paper baseline is arguably the most meaningful ("what if there were no writing?"). In normalised space that is `((1 - MEAN) / STD)` per channel.

## 6. What explanations cannot do

1. **They show correlation inside the model, not causation in the world.** A heatmap on letter shapes doesn't mean the model has learned anything about dyslexia, only that its score is sensitive to those pixels.
2. **They can be fragile.** Tiny, imperceptible input changes can alter saliency maps a lot while leaving the prediction unchanged (Ghorbani et al., 2019).
3. **They can pass the eye test and still be wrong.** Adebayo et al. (2018) showed that some methods produce plausible-looking maps even for randomly initialised networks, behaving more like edge detectors than explanations. A cheap check: compute Grad-CAM on an untrained ResNet18. If the maps look the same as for the trained model, they are not explaining what it learned.
4. **People over-trust them.** Studies have found that showing explanations can make users accept wrong predictions more often.

A good practice for the write-up is to report heatmaps for correct and incorrect predictions in both classes, describe patterns across many images, and state these limits explicitly.

## Exercises

1. ResNet18's `layer4` output is 512 × 7 × 7. How many $\alpha_k^c$ values does Grad-CAM compute, and how big is the raw heatmap?
2. IG with `n_steps=40` on 4 images: roughly how many forward + backward passes does `show_ig` run?
3. Write the code for a white-paper IG baseline for a normalised input `x`.
4. A heatmap lights up the right-hand padding region on most positive images. Give one hypothesis and one experiment to test it.

<details><summary>Answers</summary>

1. 512 weights (one per channel); a 7 × 7 heatmap, upsampled to 224 × 224 for display.
2. 4 × 40 = 160 (Captum batches the 40 steps per image internally).
3. `baseline = ((1 - torch.tensor(MEAN)) / torch.tensor(STD))[None, :, None, None].expand_as(x).to(x.device)`
4. Hypothesis: positive samples tend to be shorter sentences, so the padding is larger and the model uses padding size as a shortcut. Experiment: crop images to the text before padding (or randomise the padding position) and retrain. If performance drops sharply, the shortcut was being used.

</details>

## Further reading

- Selvaraju et al. (2017), "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", *ICCV*.
- Zhou et al. (2016), "Learning Deep Features for Discriminative Localization" (CAM), *CVPR*.
- Sundararajan, Taly & Yan (2017), "Axiomatic Attribution for Deep Networks" (IG), *ICML*.
- Adebayo et al. (2018), "Sanity Checks for Saliency Maps", *NeurIPS*.
- Ghorbani, Abid & Zou (2019), "Interpretation of Neural Networks Is Fragile", *AAAI*.
- Sturmfels, Lundberg & Lee (2020), "Visualizing the Impact of Feature Attribution Baselines", *Distill*.
- Molnar, *Interpretable Machine Learning* (free online book).
