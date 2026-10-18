# 04 · Transfer learning

Companion to the notebook's "ResNet18 and residual connections", "Transfer learning, early stopping and the metrics helper" and "Training the deployment model" cells.

## 1. The core idea

Learning good visual features needs a lot of data; a few hundred handwriting images are nowhere near enough. ImageNet (about 1.2 million labelled photos) is. Transfer learning reuses what a network learned on the big task as a starting point for the small one.

$$\theta_{\text{start}} = \theta_{\text{ImageNet}} \quad\text{instead of}\quad \theta_{\text{start}} \sim \text{random}$$

Starting near a good solution means:

- fewer steps are needed;
- the search effectively stays in a smaller region of parameter space, a form of regularisation;
- generic low-level features come for free.

## 2. What transfers, layer by layer

Yosinski et al. (2014) measured how transferable each layer of a CNN is:

- **First layers** learn Gabor-like edge and colour-blob detectors. These appear in almost every image network, whatever the dataset, so they are highly general.
- **Middle layers** learn textures and simple shapes. Moderately general.
- **Last layers** learn object parts and whole objects. Specific to the original task.

Transferability drops as the source and target tasks get further apart. Handwriting on paper is far from ImageNet's photos of objects, so:

- `conv1`–`layer2` (strokes, edges, curves) probably transfer well;
- `layer4` (dog faces, wheels) probably needs the most adaptation;
- yet `layer4` holds ~75% of the parameters, so it is also where overfitting risk concentrates.

## 3. Strategies, and when to use each

| Data size | Similar to ImageNet | Different from ImageNet |
|---|---|---|
| Small | Freeze backbone, train head | Freeze early layers, fine-tune later ones carefully (**our case**) |
| Large | Fine-tune everything | Fine-tune everything, or train from scratch |

This project uses a two-stage approximation of the "small, different" cell: train the head only, then fine-tune everything at a 10× lower LR.

## 4. Why a lower learning rate in stage 2

Two reasons.

1. **Catastrophic forgetting.** A large LR moves pretrained weights far from their starting point in a few steps, destroying the features we wanted to keep.
2. **Loss-landscape geometry.** The pretrained solution sits in a reasonable basin. Small steps explore that basin; big steps can jump out of it into regions fitted only to our few hundred images.

**Discriminative learning rates** take this further, giving early layers smaller LRs than later ones (e.g. $\eta_{\text{layer}l} = \eta_0 / 2.6^{L-l}$, as in ULMFiT). This notebook doesn't use them; they would be a natural extension, implemented with AdamW parameter groups:

```python
opt = torch.optim.AdamW([
    {"params": model.layer1.parameters(), "lr": 1e-5},
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(),     "lr": 1e-3},
], weight_decay=1e-4)
```

(Illustrative only; a real version must include every parameter group.)

## 5. The BatchNorm subtlety

ImageNet BN running statistics describe natural photos. Handwriting images are mostly white, so their activation statistics differ. During fine-tuning in `train()` mode the running statistics adapt, which is usually good. But with batch size 16 they adapt noisily, and in stage 1 they change even though `requires_grad=False`. Some practitioners keep BN layers in `eval()` mode when freezing the backbone. Neither choice is universally better; try both if results are unstable.

## 6. How to tell whether transfer helped

The SmallCNN vs ResNet18 comparison in Phase 5 is exactly this test. Read it carefully:

- ResNet18 better by more than the fold standard deviation → pretraining likely helps.
- Similar results → the task may be too noisy or too easy, or the shared preprocessing (224 px, padding) may be the bottleneck for both.
- SmallCNN better → possible if ResNet18 overfits badly; check the learning curves.

Also remember that ResNet18 has ~38× more parameters, so this compares *capacity* as well as *pretraining*. A fully fair ablation would add a third arm: ResNet18 trained from random initialisation.

## Exercises

1. In stage 1 of ResNet18, which parameters are in the optimiser?
2. Why does `build_model("resnet18", pretrained=False)` in the app still give good predictions?
3. Suggest one experiment to test whether `layer4`'s ImageNet features are useful for handwriting.

<details><summary>Answers</summary>

1. Only those whose names start with `fc` (the dropout has no parameters, so just the linear layer's weight and bias, 1,026 values). The optimiser receives `[p for p in model.parameters() if p.requires_grad]`.
2. `load_checkpoint` immediately overwrites all weights with the fine-tuned ones from `lexiassist.pth`; ImageNet weights would just be replaced.
3. For example: re-initialise `layer4` randomly before fine-tuning and compare CV results; or freeze `layer1`–`layer3` and fine-tune only `layer4` + `fc`; or truncate the network after `layer3` and attach the head there.

</details>

## Further reading

- Yosinski, Clune, Bengio & Lipson (2014), "How transferable are features in deep neural networks?", *NeurIPS*.
- Kornblith, Shlens & Le (2019), "Do Better ImageNet Models Transfer Better?", *CVPR*.
- Raghu, Zhang, Kleinberg & Bengio (2019), "Transfusion: Understanding Transfer Learning for Medical Imaging", *NeurIPS* (pretraining helps less than expected on some medical images, and small models can match big ones).
- Howard & Ruder (2018), "Universal Language Model Fine-tuning" (ULMFiT; discriminative LRs, gradual unfreezing).
- CS231n (Stanford) notes, "Transfer Learning".
