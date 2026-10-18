# 05 · Validation and data leakage

Companion to the notebook's "Labels, metadata and data leakage", "Stratified group k-fold cross-validation", "The cross-validation loop and out-of-fold predictions" and "Training the deployment model" cells.

## 1. A catalogue of leakage

| Type | Example in a project like this | Effect |
|---|---|---|
| **Group leakage** | The same child's samples in train and test | Model recognises the writer, not the indicators |
| Duplicate leakage | The same photo (or a crop of it) appears twice | Test images were effectively trained on |
| Preprocessing leakage | Normalisation statistics computed on all data, including test | Mild optimism (avoided here: fixed ImageNet statistics) |
| Selection leakage | Threshold or architecture chosen using test results | Optimistic test score (partly present: threshold tuned on OOF) |
| Temporal / site leakage | Positive and negative images collected at different sites or times | Model learns the site, not the condition |

Group leakage is the big one here. Its strength depends on how recognisable writers are, and handwriting is *very* recognisable: writer identification is a whole research field.

## 2. Clustered data shrinks your effective sample size

When images come in groups (several per child) and images within a group are correlated, the data contains less information than its raw count suggests. The **design effect** quantifies this:

$$\mathrm{DE} = 1 + (m - 1)\rho, \qquad n_{\text{eff}} = \frac{n}{\mathrm{DE}}$$

where $m$ is the average number of images per child and $\rho$ the intra-class correlation (how similar a child's images are, relative to the variation between children).

**Example.** 300 images from 60 children ($m = 5$) with $\rho = 0.5$ give $\mathrm{DE} = 3$ and $n_{\text{eff}} = 100$. Confidence intervals are about $\sqrt 3 \approx 1.7\times$ wider than a naive per-image calculation suggests.

## 3. How StratifiedGroupKFold works

Roughly (scikit-learn's implementation):

1. Count the class distribution inside each group.
2. Sort groups by the standard deviation of their class counts, largest first, so big single-class groups are placed early (with `shuffle=True`, groups are shuffled before this sort, which breaks ties randomly).
3. Assign each group, in turn, to the fold where adding it keeps the per-fold class proportions closest to the overall proportions.

It is greedy, so balance is approximate. That's why the notebook prints `pd.crosstab(df.fold, df.label)`.

## 4. Why validation and test must differ

If one fold were used both to pick the early-stopping epoch *and* to report the score, we would report the best of ~30 epochs on that fold: a maximum over noisy estimates, which is biased upward. The notebook avoids this by using fold $k+1$ for stopping and fold $k$ for testing.

**Nested cross-validation** extends the idea to *all* choices. An inner CV loop selects hyperparameters (LR, threshold, architecture); an outer loop measures the whole selection procedure. It is expensive but the gold standard when you tune a lot on small data. The screening threshold here is chosen on the same OOF predictions it is evaluated on, so its reported recall is optimistic; nested CV would fix that.

## 5. Uncertainty of CV estimates

The standard deviation across 5 folds is a rough guide, not a confidence interval. The folds share training data, so the fold scores are correlated, and no unbiased estimator of the variance of k-fold CV exists (Bengio & Grandvalet, 2004).

Better options for reporting:

- **Bootstrap the OOF predictions *by writer*.** Resample writers with replacement, recompute AUC or recall each time, and report the 2.5th–97.5th percentiles.
- **Repeated CV.** Repeat the 5-fold split with different seeds and report the spread of the means.
- **Wilson interval** for a proportion such as recall, with $\hat p$ observed over $n$ positives and $z = 1.96$:

$$\frac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

**Example.** Recall 0.90 measured on 40 positives gives a Wilson 95% interval of roughly 0.77–0.96.

## 6. Shortcut learning and dataset bias

Geirhos et al. (2020) define shortcuts as decision rules that work on standard benchmarks but fail under slightly different conditions. Classic examples: pneumonia classifiers that learned which hospital's scanner produced an X-ray, and skin-lesion models that keyed on rulers that dermatologists place next to worrying lesions.

Candidate shortcuts in handwriting data:

- paper type, ruled vs plain lines;
- scanner or camera, lighting, image size or aspect ratio;
- sentence length, which changes how much of the image is padding;
- pen colour (removed here by converting to grayscale);
- classroom-specific writing exercises.

**Ways to detect them:**

- Grad-CAM and IG heatmaps.
- Training a model on deliberately uninformative inputs, such as only the image border or only the image size, to see if it still predicts the label.
- External validation.
- Stratifying errors by site or condition.

## Exercises

1. Filenames look like `S12_03.jpg`, `S12_04.jpg`, `S47_01.jpg`. Write a suitable `WRITER_REGEX`.
2. Your writer-aware CV gives recall 0.71; an image-level split gave 0.88. What does the gap tell you?
3. Why does the `assert df.groupby("writer").fold.nunique().max() == 1` check prove nothing when `WRITER_REGEX = None`?
4. With 20 positive children split over 5 folds, about how many positive children are in each test fold, and what does that imply?

<details><summary>Answers</summary>

1. `r"^(S\d+)_"` (captures `S12`, `S47`).
2. Much of the image-level performance came from recognising writers seen in training, i.e. leakage. The writer-aware number is the honest estimate.
3. Each image is its own group, so every group trivially appears in exactly one fold.
4. About 4. Recall per fold can then only take values like 0, 0.25, 0.5, 0.75 or 1, so per-fold numbers swing wildly and the mean has large uncertainty.

</details>

## Further reading

- Kaufman, Rosset & Perlich (2012), "Leakage in Data Mining: Formulation, Detection, and Avoidance", *ACM TKDD*.
- Kapoor & Narayanan (2023), "Leakage and the reproducibility crisis in machine-learning-based science", *Patterns*.
- Varma & Simon (2006), "Bias in error estimation when using cross-validation for model selection", *BMC Bioinformatics*.
- Bengio & Grandvalet (2004), "No Unbiased Estimator of the Variance of K-Fold Cross-Validation", *JMLR*.
- Geirhos et al. (2020), "Shortcut learning in deep neural networks", *Nature Machine Intelligence*.
- Zech et al. (2018), "Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs", *PLOS Medicine*.
