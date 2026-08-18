# GSE39833 — CRC classification from serum exosomal miRNA

Nested cross-validation with fold-internal feature selection on the serum
exosomal miRNA microarray **GSE39833** (11 healthy controls, 88 colorectal
cancer patients, 15,739 probes).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jay99Sohn/GEOexosome/blob/main/GEOexosome.ipynb)

> Analysis performed November–December 2025 for a CHA University research
> poster (first prize, November 2025). Pipeline refactored and re-run in
> August 2026; commit dates reflect publication, not when the work was done.
> Commit `45d8fe5` (2025-11-15) is the version submitted with the poster.

## Headline

The refactored pipeline reaches a cross-validated AUC of **0.997**. That number
should not be used. **Sample accession order is perfectly confounded with the
class label**, so the classifier cannot be shown to be separating disease from
run batch. Details in [Confounding](#confounding).

## Method

- **Imbalance:** `class_weight='balanced'` instead of SMOTE. Oversampling before
  cross-validation inflates AUC, and with 11 minority samples the SMOTE
  neighbourhood is not well defined (Demircioğlu 2024, *Sci Rep* 14:15744).
- **Feature selection, inside each training fold only:** differential-expression
  filter (|log2FC|, Benjamini–Hochberg FDR) → LASSO, SVM-RFE and random-forest
  importance → probes selected by ≥ 2 of the 3.
- **Evaluation:** 5-fold × 10 independent repeats. Specificity is reported with
  its spread, since one control is 9.1 percentage points.
- **Preprocessing:** `VALUE` is background-subtracted intensity and contains
  negative entries. These are floored at zero before `log2(x+1)` rather than
  allowed to become `NaN`, which would silently drop the affected probes.

## Results

| Model | AUC (10 repeats) | Sensitivity | Specificity |
|---|---|---|---|
| SVM | 0.9974 ± 0.0028 | 0.994 | 0.873 (0.73–1.00) |
| Logistic regression | 0.9966 ± 0.0036 | 0.985 | 0.955 (0.91–1.00) |
| Random forest | 0.9965 ± 0.0040 | 0.998 | 0.736 (0.55–0.91) |

13 probes were selected in ≥ 70 % of the 50 folds, corresponding to 9 distinct
annotations.

## Confounding

Removing SMOTE — a technique that *inflates* AUC — raised AUC from 0.96 to
0.997. That is the wrong direction, and it prompted the following checks.

**1. Accession order.** The 11 controls are `GSM980024`–`GSM980034` and the 88
cases are `GSM980035`–`GSM980122`: two disjoint, contiguous blocks with no
interleaving. A Wald–Wolfowitz runs test gives 2 blocks against ~21 expected
under random ordering (p = 0.0001).

```
label sequence in accession order (C = control)
CCCCCCCCCCC........................................................................................
```

**2. Implausible features.** Among the most frequently selected probes:

- `hsv1-miR-H6-3p` — a **herpes simplex virus 1** miRNA, selected in 86 % of
  folds. There is no biological reason for a viral probe to discriminate
  colorectal cancer from healthy serum.
- `hsa-miR-1825` — selected in 76 % of folds; reclassified as a tRNA fragment
  rather than a genuine miRNA.

**Conclusion.** Run order and class label are perfectly aligned in this series,
so no analysis of GSE39833 alone can attribute the signal to disease rather than
to batch. This is a property of the dataset, not of the pipeline. Accession
clustering does not by itself prove that samples were processed in separate
batches — submitters sometimes deposit all controls first — but it removes any
basis for excluding it, and the two implausible probes point the same way.
Resolving it requires hybridisation or scan dates, which this series does not
report.

The selected probes are therefore **not reported as candidate biomarkers**. The
value of this repository is the pipeline and the confounding analysis.

## Other limitations

- 11 healthy controls only; specificity moves in steps of 9.1 points.
- The `±` on AUC is between-repeat variation, not sampling uncertainty for a
  new cohort.
- 4 of the 13 stable probes measure the same miRNA (`hsa-miR-654-5p`) and 2 have
  no annotation, so the panel is 9 distinct features, not 13.
- Single cohort, single platform, no external validation.

## Repository layout

`GEOexosome.ipynb` — full pipeline in six cells, outputs included. Each cell
checkpoints to Drive, so an interrupted run continues where it stopped. Delete
`checkpoints/` to recompute from scratch.

## Requirements

Python 3 with `GEOparse`, `scikit-learn`, `scipy`, `pandas`, `shap`,
`matplotlib`. Cell 1 installs the two non-default packages. Raw data is
downloaded from GEO at run time; no local files are needed.
