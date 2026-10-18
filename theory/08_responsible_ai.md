# 08 · Responsible AI

Companion to the notebook's "External validation and dataset shift" and "Responsible AI" cells.

## 1. What the label actually measures

```
what we care about          what the label records          what the model sees
-------------------         -----------------------         -------------------
dyslexia (a specific   -->  assessor's judgement of    -->  a 224x224 grayscale
learning difficulty         "potential dysgraphia"          picture of one writing
mainly affecting            in one writing sample           sample
reading and spelling)
         ^ gap 1: different condition          ^ gap 2: noisy, single-sample rating
```

Each arrow loses information. Results should be described in terms of the right-hand side ("resembles samples rated as potential dysgraphia in dataset X"), never the left.

## 2. Harms of each error type

| | False negative (missed) | False positive (false alarm) |
|---|---|---|
| Child | Support delayed; struggles attributed to effort | Unnecessary assessment; may be labelled |
| Parent | False reassurance | Anxiety, cost of private assessment |
| Teacher / school | Misses a child who needed help | Time spent on unnecessary referrals |
| System | Erodes trust when the miss is discovered | Overloads specialists; erodes trust in flags |

A threshold choice is a value judgement about these harms. Make it explicit, and ideally make it with the people affected.

## 3. Bias and generalisation

Sources of possible systematic error in LexiAssist:

- **Language and script.** Trained on Malay sentences in Latin script. Letter frequencies, common reversals and writing direction differ in other languages.
- **Age.** Handwriting changes rapidly through childhood; a 6-year-old's normal writing may resemble an older child's "indicators".
- **Motor and other conditions.** Dyspraxia, ADHD, visual impairment or injury can affect handwriting without any reading difficulty.
- **Handedness.** Left-handed writers often have a different slant and smudging.
- **Instrument and paper.** Pencil vs pen, ruled vs plain paper.
- **Image capture.** Phone camera quality, lighting, angle.

If the data records any of these, report metrics for each subgroup. If it doesn't, list them as untested.

## 4. Children's data, consent and privacy

- **Legal basis.** In many places, children's data has extra protection (for example the EU and UK GDPR, the General Data Protection Regulation, which treats children as needing specific protection). Check the dataset's licence and terms before any use beyond research.
- **Consent and assent.** Parents or guardians consent; the child should also agree in an age-appropriate way.
- **Data minimisation.** The app processes uploads in memory and doesn't store them. Keep it that way, and don't add analytics that capture images.
- **Model weights are derived data.** Networks can memorise training examples. Treat `lexiassist.pth` as sensitive, and don't publish it alongside identifying metadata.
- **Re-identification.** Handwriting is a biometric-like identifier. Never combine dataset images with other information to identify a child.

## 5. Documenting the model

A **model card** (Mitchell et al., 2019) is a one-page summary that travels with the model. A minimal one for LexiAssist would cover:

| Section | Content |
|---|---|
| Intended use | Educational demonstration of a screening-style classifier |
| Out-of-scope use | Diagnosis; decisions about support, placement or resources; non-Malay writers; adults |
| Training data | Dataset name and DOI, number of images and writers, labelling process, writer-ID caveat |
| Evaluation | Writer-aware CV mean ± sd; external test results; threshold and how it was chosen |
| Limitations | Concept gap (dysgraphia ≠ dyslexia), resolution loss, small data, possible shortcuts |
| Ethical considerations | Children's data, harms of each error type, consent |

**Datasheets for datasets** (Gebru et al., 2021) do the same for the data.

## 6. Communicating results honestly

| Instead of | Say |
|---|---|
| "Detects dyslexia with 85% accuracy" | "On held-out writers from one small dataset, the model ranked samples rated as potential dysgraphia above others with AUC 0.85 (95% CI …)" |
| "The model focuses on letter formation" | "In most of the heatmaps we inspected, attribution concentrated on the writing rather than the background" |
| "Generalises to other languages" | "Performance on a bilingual external dataset dropped to …, suggesting limited transfer" |

## Exercises

1. Write a one-sentence "intended use" statement for LexiAssist that a parent would understand.
2. Name two subgroups for which you would want separate error rates, and explain why.
3. A school wants to use LexiAssist to decide which children get reading support. What would you tell them?

<details><summary>Answers (one possible version)</summary>

1. "LexiAssist is a learning project that shows how a computer can look for some handwriting patterns; it cannot tell you whether your child has dyslexia, and its result should never replace a teacher's or specialist's judgement."
2. For example: age bands (handwriting maturity differs), and left- vs right-handed writers (different slant and stroke patterns). Both could produce errors that look like "indicators".
3. That it was never validated for that purpose: it was trained on a different condition (dysgraphia ratings), in one language, on a small dataset, and has known false-negative and false-positive rates. Allocation decisions should rest on established assessments by qualified staff.

</details>

## Further reading

- Mitchell et al. (2019), "Model Cards for Model Reporting", *FAccT*.
- Gebru et al. (2021), "Datasheets for Datasets", *Communications of the ACM*.
- Wilson & Jungner (1968), *Principles and Practice of Screening for Disease*, World Health Organization (classic screening criteria).
- Obermeyer et al. (2019), "Dissecting racial bias in an algorithm used to manage the health of populations", *Science* (a proxy label causing harm).
- UNICEF (2021), *Policy Guidance on AI for Children*.
- Information Commissioner's Office (UK), *Age Appropriate Design Code*.
