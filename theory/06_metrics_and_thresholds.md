# 06 · Metrics and thresholds

Companion to the notebook's "Transfer learning, early stopping and the metrics helper", "ROC curves, AUC, the confusion matrix and calibration" and "Choosing a screening threshold" cells.

## 1. One confusion matrix, every metric

Using the worked example from the notebook (TP = 13, FN = 3, FP = 6, TN = 18):

| Metric | Also called | Formula | Value | Question it answers |
|---|---|---|---|---|
| Recall | Sensitivity, TPR | TP/(TP+FN) | 0.81 | Of children with indicators, how many are flagged? |
| Specificity | TNR (true negative rate) | TN/(TN+FP) | 0.75 | Of children without, how many are cleared? |
| FPR | Fall-out | FP/(FP+TN) | 0.25 | Of children without, how many are falsely flagged? |
| Precision | PPV | TP/(TP+FP) | 0.68 | Of flagged children, how many really have indicators? |
| NPV | | TN/(TN+FN) | 0.86 | Of cleared children, how many really are fine? |
| F1 | | 2TP/(2TP+FP+FN) | 0.74 | Harmonic mean of precision and recall |
| Accuracy | | (TP+TN)/N | 0.78 | Overall share correct |
| Balanced accuracy | | (Recall+Specificity)/2 | 0.78 | Accuracy that ignores class imbalance |

**Why F1 uses the harmonic mean.** It is dominated by the smaller value. Precision 1.0 with recall 0.1 gives an arithmetic mean of 0.55 but F1 of 0.18, which correctly calls that model poor.

## 2. Which metrics depend on prevalence?

| Prevalence-*independent* (properties of the test) | Prevalence-*dependent* (depend on who is tested) |
|---|---|
| Recall, specificity, FPR, ROC, AUC | Precision (PPV), NPV, accuracy, F1 |

That is why recall and specificity transfer from a study to the real world better than precision does. In the notebook, the class balance is far higher than real-world prevalence, so the measured precision overstates real-world precision.

## 3. Bayes' rule for screening

$$\mathrm{PPV} = \frac{\text{sens}\cdot\pi}{\text{sens}\cdot\pi + (1-\text{spec})(1-\pi)}$$

where $\pi$ is prevalence. With sensitivity 0.90 and specificity 0.70:

| Prevalence $\pi$ | PPV | NPV |
|---|---|---|
| 5% | 14% | 99.3% |
| 10% | 25% | 98.4% |
| 20% | 43% | 96.6% |
| 40% (similar to a balanced study set) | 67% | 91.3% |

A screening tool with these numbers is good at *ruling out* (high NPV) and weak at *ruling in* (low PPV). That is exactly the profile of a screener: a positive result means "look closer", never "has the condition".

## 4. ROC and AUC in more depth

**Constructing the curve.** Sort images by score, descending. Walk down the list: each positive moves the curve up by $1/n_+$, each negative moves it right by $1/n_-$.

**AUC equals the Mann–Whitney U statistic** divided by $n_+ n_-$. Ties count as half a win:

$$\mathrm{AUC} = \frac{1}{n_+ n_-}\sum_{i\in +}\sum_{j \in -}\Big(\mathbb 1[\hat p_i > \hat p_j] + \tfrac12\mathbb 1[\hat p_i = \hat p_j]\Big)$$

**Limits of AUC:**

- It weights all thresholds equally, including ones you would never use. For screening only the high-recall region matters; partial AUC focuses there.
- It says nothing about calibration.
- On imbalanced data, the precision–recall curve is often more informative about the positive class.

## 5. Threshold-selection rules compared

| Rule | Chooses | Good when |
|---|---|---|
| 0.5 | Default | Probabilities are calibrated and errors cost the same |
| Youden's $J$ | max(sens + spec − 1) | Equal costs, no prevalence information |
| Fixed recall (used) | best spec with recall ≥ target | A clear minimum sensitivity requirement |
| Cost-based | min $c_{FN}FN + c_{FP}FP$ | You can put rough costs on each error |

For calibrated probabilities, the cost-optimal threshold has a closed form:

$$t^* = \frac{c_{FP}}{c_{FP} + c_{FN}}$$

If a missed case is judged 4 times as costly as a false alarm, then $t^* = 1/5 = 0.2$.

## 6. Calibration in more depth

**Reliability diagram.** Bin predictions, then plot the mean predicted probability against the observed positive rate in each bin. Points below the diagonal mean over-confidence.

**Brier score.** The mean squared error of probabilities, $\frac1n\sum_i(\hat p_i - y_i)^2$. It combines calibration and discrimination.

**Recalibration methods** (fit on held-out data):

- *Platt scaling*: logistic regression on the logit.
- *Temperature scaling*: divide logits by a learned $T > 0$. A single parameter, popular for neural networks (Guo et al., 2017).
- *Isotonic regression*: monotone, non-parametric; needs more data.

Class-weighted training systematically inflates minority-class probabilities. If you want the app's percentage to mean something, recalibrate on OOF predictions, or show only the category ("possible / few indicators") instead of a percentage.

## Exercises

1. Compute NPV and balanced accuracy for TP = 30, FN = 10, FP = 20, TN = 140.
2. Positive scores {0.8, 0.5}, negative scores {0.5, 0.2}. Compute AUC, counting ties as half.
3. Your screener has sensitivity 0.85 and specificity 0.80; real prevalence is 8%. What share of flagged children truly have indicators?
4. Why might you pick a *lower* recall target for a tool used by parents at home than for one used by school specialists?

<details><summary>Answers</summary>

1. NPV = 140/150 ≈ 0.93. Recall = 0.75, specificity = 140/160 = 0.875, so balanced accuracy ≈ 0.81.
2. The pairs: (0.8, 0.5) win, (0.8, 0.2) win, (0.5, 0.5) tie = 0.5, (0.5, 0.2) win. Total 3.5/4 = 0.875.
3. PPV = (0.85·0.08)/(0.85·0.08 + 0.20·0.92) = 0.068/0.252 ≈ 27%.
4. At home there is no professional to absorb false alarms, so each false positive causes direct worry. In a specialist setting the next assessment step catches false positives cheaply. The costs differ, so the threshold should too. (The opposite argument, that at home missed cases are never caught later, is also valid; the point is that the choice depends on context and must be justified.)

</details>

## Further reading

- Saito & Rehmsmeier (2015), "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets", *PLOS ONE*.
- Fawcett (2006), "An introduction to ROC analysis", *Pattern Recognition Letters*.
- Guo, Pleiss, Sun & Weinberger (2017), "On Calibration of Modern Neural Networks", *ICML*.
- Altman & Bland (1994), "Diagnostic tests 2: predictive values", *BMJ*.
- Elkan (2001), "The Foundations of Cost-Sensitive Learning", *IJCAI*.
