# 01 · Images and preprocessing

Companion to the notebook's "Images as tensors", "The preprocessing pipeline" and "Datasets, data loaders and augmentation" cells.

## 1. From photo to tensor, step by step

Take a 1200×400 phone photo of one sentence and follow it through `app/preprocess.py`.

| Step | Shape | Value range | Code |
|---|---|---|---|
| Decoded JPEG | 400 × 1200 × 3 (H × W × C) | 0–255 `uint8` | `Image.open(...).convert("RGB")` |
| Grayscale | 400 × 1200 | 0–255 | `img.convert("L")` in `PadToSquare` |
| Padded square | 1200 × 1200 | 0–255 | `Image.new("L", (s, s), fill)` + `paste` |
| Resized | 224 × 224 | 0–255 | `T.Resize((224, 224))` |
| 3-channel gray | 224 × 224 × 3 | 0–255 | `T.Grayscale(num_output_channels=3)` |
| Tensor | 3 × 224 × 224 | 0.0–1.0 `float32` | `T.ToTensor()` |
| Normalised | 3 × 224 × 224 | about −2.1 to +2.6 | `T.Normalize(MEAN, STD)` |
| Batched | 1 × 3 × 224 × 224 | | `.unsqueeze(0)` |

**Where does the range "−2.1 to +2.6" come from?** The smallest possible value is black in the red channel, $(0 - 0.485)/0.229 \approx -2.12$; the largest is white in the blue channel, $(1 - 0.406)/0.225 \approx 2.64$.

## 2. How much of the image is handwriting?

After padding a $W \times H$ image (with $W > H$) to $W \times W$ and resizing to $S = 224$:

$$\text{text band height} = S\cdot\frac{H}{W}, \qquad \text{stroke width after resizing} = w_{\text{stroke}}\cdot\frac{S}{W}$$

| Original size | Text band height | 5-pixel stroke becomes | Share of pixels that are padding |
|---|---|---|---|
| 1200 × 400 | 75 px | 0.93 px | 67% |
| 800 × 600 | 168 px | 1.4 px | 25% |
| 3000 × 1000 | 75 px | 0.37 px | 67% |

Lesson: a high-resolution photo doesn't help after resizing, and wide sentence strips waste most of the input on padding. Cropping to the text, or splitting long lines into tiles, would keep far more detail.

## 3. Interpolation methods compared

| Method | How it computes a new pixel | Quality when shrinking | Speed |
|---|---|---|---|
| Nearest neighbour | Copies the closest pixel | Jagged; thin strokes can vanish entirely | Fastest |
| Bilinear (used here) | Weighted average of 4 neighbours | Smooth; slightly blurry | Fast |
| Bicubic | Weighted average of 16 neighbours | Sharper | Slower |
| Area / antialiased | Averages every source pixel that falls in the target pixel | Best for large shrink factors | Moderate |

**Aliasing.** Shrinking 1200 → 224 keeps only about one pixel in every five. Without a low-pass filter first, detail finer than the new pixel grid folds into false patterns (the classic moiré effect). torchvision applies antialiasing for PIL images.

## 4. Why the border median?

The mean is pulled towards dark ink and shadows; the median ignores up to half the pixels being "wrong". On a clean scan the border is almost all paper, so the median is the paper colour. It fails when the photo border shows a desk, a hand or a dark shadow. Then the padding becomes a visible band that the model may learn from.

## 5. Normalisation, more precisely

Standardising inputs to roughly zero mean and unit variance helps optimisation: when input features have very different scales, the loss surface becomes a long, narrow valley, and gradient descent zig-zags. Pretrained networks add a second reason: their first-layer filters were tuned for ImageNet-normalised inputs, so feeding them a different scale shifts every activation downstream.

## 6. Augmentation as a prior

Augmentation tells the model which changes should *not* change the answer.

| Transform | Invariance it teaches | Safe for handwriting? |
|---|---|---|
| Small rotation (±5°) | Slightly tilted page | Yes |
| Small shift / scale | Different framing | Yes |
| Brightness / contrast | Lighting, scanner settings | Yes |
| Horizontal flip | Mirror symmetry | **No**: changes b↔d, reverses writing direction |
| Large rotation (±30°+) | Arbitrary orientation | Risky: slant may be informative |
| Elastic distortion | Natural variation in strokes | Risky: may mimic or erase the very irregularities being screened for |
| Blur / noise | Camera quality | Mild only; strong blur removes stroke detail |

## Exercises

1. A scan is 1000 × 1000 pixels with 4-pixel strokes. How wide are the strokes after preprocessing?
2. Why is it a problem if all positive images were scanned and all negative images photographed?
3. Compute the normalised value of a mid-gray pixel ($x = 0.5$) in each channel.
4. You add `T.RandomHorizontalFlip()` and validation accuracy goes *up*. Give two possible explanations, and say why you still shouldn't keep it.

<details><summary>Answers</summary>

1. No padding needed; scale $224/1000 = 0.224$, so $4 \times 0.224 \approx 0.9$ pixels.
2. The model can learn the acquisition method (sharpness, lighting, noise) instead of handwriting: a shortcut that will fail on new data.
3. R: $(0.5-0.485)/0.229 \approx 0.066$; G: $(0.5-0.456)/0.224 \approx 0.196$; B: $(0.5-0.406)/0.225 \approx 0.418$.
4. (a) Random variation: one validation fold of a small dataset is noisy. (b) The extra regularisation helps even though it is label-unsafe. You shouldn't keep it because it destroys a potentially meaningful signal (letter reversals), and the gain is likely noise.

</details>

## Further reading

- Gonzalez & Woods, *Digital Image Processing*: chapters on sampling, interpolation and aliasing.
- Shorten & Khoshgoftaar (2019), "A survey on image data augmentation for deep learning", *Journal of Big Data*.
- torchvision documentation, "Transforming and augmenting images".
