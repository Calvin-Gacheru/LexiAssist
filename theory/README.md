# LexiAssist theory notes

These notes go deeper than the "Theory:" cells in `LexiAssist.ipynb`. The notebook cells explain each step where it happens; these files add longer derivations, worked exercises (with answers) and further reading. Read the notebook cell first, then the matching file here.

## Reading order

| File | Covers | Notebook phases |
|---|---|---|
| [01_images_and_preprocessing.md](01_images_and_preprocessing.md) | Images as tensors, padding, interpolation, normalisation, augmentation | 2, 4 |
| [02_cnns_and_resnet.md](02_cnns_and_resnet.md) | Convolution, receptive fields, BN, dropout, GAP, residual networks | 4 |
| [03_training_and_optimisation.md](03_training_and_optimisation.md) | Softmax, cross-entropy, backprop, AdamW, schedules, early stopping, bias–variance | 5 |
| [04_transfer_learning.md](04_transfer_learning.md) | Why pretrained features help, freezing, fine-tuning, learning rates | 5, 7 |
| [05_validation_and_leakage.md](05_validation_and_leakage.md) | Leakage, grouped/stratified k-fold, OOF predictions, variance of estimates | 2, 3, 5, 7 |
| [06_metrics_and_thresholds.md](06_metrics_and_thresholds.md) | Confusion matrix, ROC/AUC, thresholds, prevalence, calibration | 5, 6 |
| [07_explainability.md](07_explainability.md) | Grad-CAM, Integrated Gradients, limits of saliency | 9 |
| [08_responsible_ai.md](08_responsible_ai.md) | Screening vs diagnosis, harms, children's data, dataset shift | 8, 12 |

## Notation

| Symbol | Meaning |
|---|---|
| $x$, $y$ | Input image (tensor), true label (0 or 1) |
| $\theta$ | All learnable parameters (weights and biases) |
| $z = (z_0, z_1)$ | Logits: raw scores output by the network |
| $p_k$ | Softmax probability of class $k$ |
| $\ell$, $\mathcal L$ | Loss for one example, average loss over a batch |
| $\eta$, $\lambda$ | Learning rate, weight-decay coefficient |
| $B$, $N$ | Batch size, number of training images |
| $A^k$ | $k$-th feature map of a convolutional layer |

## Glossary of acronyms

| Acronym | Meaning |
|---|---|
| AdamW | Adam optimiser with decoupled weight decay |
| AUC | Area under the (ROC) curve |
| BN | Batch normalisation |
| CNN | Convolutional neural network |
| CPU / GPU | Central / graphics processing unit |
| CUDA | NVIDIA's platform for running computations on GPUs |
| CV | Cross-validation (in these notes; never "computer vision") |
| ECE | Expected calibration error |
| FN / FP / TN / TP | False negative / false positive / true negative / true positive |
| FPR / TPR | False positive rate / true positive rate |
| GAP | Global average pooling |
| Grad-CAM | Gradient-weighted class activation mapping |
| IG | Integrated Gradients |
| IID | Independent and identically distributed |
| LR | Learning rate |
| NPV / PPV | Negative / positive predictive value |
| OOF | Out-of-fold (predictions) |
| PIL | Python Imaging Library (Pillow) |
| ReLU | Rectified linear unit, $\max(0, z)$ |
| RGB | Red, green, blue |
| ROC | Receiver operating characteristic |
| SGD | Stochastic gradient descent |
| XAI | Explainable artificial intelligence |

## A note on honesty

Every number in the notebook comes from a few hundred images of one group of children, labelled by assessors for potential *dysgraphia*. Keep that in mind when reading any metric, and read [08_responsible_ai.md](08_responsible_ai.md) before presenting results.
