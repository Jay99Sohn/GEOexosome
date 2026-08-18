# GSE39833 — CRC classification from serum exosomal miRNA

Nested cross-validation with fold-internal feature selection on the serum
exosomal miRNA microarray **GSE39833** (11 healthy controls, 88 colorectal
cancer patients, 15,739 probes).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jay99Sohn/GEOexosome/blob/main/GEOexosome.ipynb)

> Analysis performed November–December 2025 for a CHA University research
> poster (first prize, November 2025). Pipeline refactored and re-run in
> August 2026; commit dates reflect publication, not when the work was done.

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

## Interpretation and open problems

**These AUCs are almost certainly optimistic and should not be read as
diagnostic performance.** Two observations argue against a purely biological
explanation:

1. **A herpes simplex virus 1 probe (`hsv1-miR-H6-3p`) is among the most
   frequently selected features** (86 % of folds). A viral probe should not
   discriminate colorectal cancer from healthy serum on biological grounds.
2. **`hsa-miR-1825` is also selected** (76 % of folds). This probe has been
   reclassified as a tRNA fragment rather than a genuine miRNA.

Together these suggest the classifier is partly separating a **technical or
batch difference** between the control and case samples rather than disease
biology. GEO series frequently process controls and cases in separate batches.
Confirming or excluding this requires the array scan dates and batch metadata,
which are not resolved here.

Other limitations:

- 11 healthy controls only; specificity moves in steps of 9.1 points.
- The `±` on AUC is between-repeat variation, not sampling uncertainty for a
  new cohort.
- 4 of the 13 stable probes measure the same miRNA (`hsa-miR-654-5p`), and 2
  have no annotation, so the panel is 9 distinct features, not 13.
- Single cohort, single platform, no external validation.

The selected probes are **candidate biomarkers pending batch-effect
adjudication**, not a validated diagnostic panel.

## Repository layout

| Path | Contents |
|---|---|
| `GEOexosome.ipynb` | Full pipeline, six cells, outputs included |
| `docs/references.md` | Methodological references and how each was applied |

The notebook checkpoints each cell to Drive, so an interrupted run continues
where it stopped. Delete `checkpoints/` to recompute from scratch.

Earlier revisions are in the commit history; `45d8fe5` (2025-11-15) is the
version submitted with the poster, which used SMOTE and single-pass ANOVA
feature selection.

## Requirements

Python 3 with `GEOparse`, `scikit-learn`, `scipy`, `pandas`, `shap`,
`matplotlib`. Cell 1 installs the two non-default packages. Raw data is
downloaded from GEO at run time; no local files are needed.
