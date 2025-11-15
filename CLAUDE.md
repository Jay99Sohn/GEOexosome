# CLAUDE.md - AI Assistant Guide for GEOexosome Project

**Last Updated:** 2025-11-15
**Repository:** Jay99Sohn/GEOexosome
**Project Status:** Poster Submission Ready (1차 poster 제출 코드)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Development Workflow](#development-workflow)
4. [Technical Architecture](#technical-architecture)
5. [Key Conventions](#key-conventions)
6. [Working with This Repository](#working-with-this-repository)
7. [Common Tasks](#common-tasks)
8. [Important Notes for AI Assistants](#important-notes-for-ai-assistants)

---

## Project Overview

### Purpose

This is a **bioinformatics machine learning research project** focused on:
- **Objective:** Classify colorectal cancer (CRC) patients from healthy controls using exosomal microRNA expression profiles
- **Method:** Machine learning classification with rigorous nested cross-validation
- **Goal:** Discover biomarker miRNAs for CRC diagnosis
- **Output:** Academic poster presentation with publication-ready figures

### Research Context

- **Data Source:** NCBI GEO database (Accession: GSE39833)
- **Platform:** GPL14767 (Agilent microarray)
- **Sample Classes:**
  - HC (Healthy Controls) = Label 0
  - CRC (Colorectal Cancer) = Label 1
- **Analysis Type:** Supervised binary classification with explainability analysis

### Key Features

✓ Automated GEO dataset download and processing
✓ Multi-model comparison (Random Forest, SVM, Logistic Regression)
✓ Nested cross-validation for unbiased evaluation
✓ SHAP-based explainability for biomarker discovery
✓ Publication-ready visualization generation
✓ Comprehensive overfitting diagnostics

---

## Repository Structure

```
GEOexosome/
├── .git/                      # Git version control
├── GEOexosome.ipynb          # Main analysis notebook (~1000+ lines)
├── README.md                  # Minimal repository description
└── CLAUDE.md                  # This file - AI assistant guide

Generated Outputs (when run):
├── data/                      # GEO dataset download location
├── confusion_matrix_final.png
├── roc_curve_final.png
├── model_comparison.png
├── learning_curves.png
├── cv_score_distribution.png
├── feature_importance_all_mirnas.csv
├── shap_importance_all_mirnas.csv
├── final_research_results.json
└── research_summary_for_poster.txt
```

### File Descriptions

| File | Purpose | Size | Critical |
|------|---------|------|----------|
| **GEOexosome.ipynb** | Main analysis pipeline | ~880KB | ✓ YES |
| **README.md** | Repository description | ~1B | No |
| **CLAUDE.md** | AI assistant guide | This file | No |

---

## Development Workflow

### Git Branching Strategy

**Branch Naming Convention:**
- Feature branches: `claude/claude-md-<session-id>`
- Current branch: `claude/claude-md-mi0dk2l4liosecto-01SKgDmQqmRJXBRA4riFygNu`
- Main branch: Not explicitly set (likely `main` or `master`)

**Important Git Rules:**
1. **ALWAYS** develop on `claude/` prefixed branches
2. **NEVER** push directly to main/master without permission
3. **MUST** use `git push -u origin <branch-name>` for pushing
4. Branch names **MUST** start with 'claude/' and match session ID
5. Implement exponential backoff (2s, 4s, 8s, 16s) for network failures

### Commit History

```
45d8fe5 - 1차 poster 제출 코드 (Initial poster submission code)
10b7309 - Create README.md
```

**Commit Message Style:**
- Korean language used for main commits
- Descriptive and concise
- Focus on milestone achievements

### Typical Development Cycle

1. **Checkout** appropriate `claude/` branch
2. **Modify** GEOexosome.ipynb with analysis changes
3. **Test** notebook execution (cells must run in order)
4. **Commit** with descriptive Korean/English message
5. **Push** to feature branch with retry logic
6. **Create PR** when ready for main branch merge

---

## Technical Architecture

### Execution Environment

**Primary Platform:** Google Colab
**Secondary Platform:** Local Jupyter Notebook
**Python Version:** 3.6+
**Dependencies:** See [Dependencies](#dependencies) section

**Environment Detection:**
```python
# Notebook automatically detects Colab vs. local
try:
    from google.colab import drive
    # Colab-specific code
except:
    # Local execution fallback
```

### Analysis Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DATA ACQUISITION (GEOparse)                              │
│    └─ Download GSE39833 → Extract expression & metadata     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. DATA PREPROCESSING                                        │
│    ├─ miRNA ID mapping                                      │
│    ├─ Label extraction (HC=0, CRC=1)                        │
│    └─ Quality checks                                        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. PIPELINE CONSTRUCTION (scikit-learn Pipeline)            │
│    ├─ VarianceThreshold (threshold=0.01)                   │
│    ├─ SelectKBest (k=30-40, f_classif)                     │
│    ├─ SMOTE (k_neighbors=2-3)                              │
│    ├─ StandardScaler                                        │
│    └─ Classifier (RF/SVM/LR)                               │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. NESTED CROSS-VALIDATION                                  │
│    ├─ Outer CV: 5-Fold Stratified (performance estimation) │
│    └─ Inner CV: 3-Fold Stratified (hyperparameter tuning)  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. MODEL EVALUATION & EXPLAINABILITY                        │
│    ├─ Performance metrics (ROC-AUC, Accuracy, etc.)        │
│    ├─ Overfitting diagnostics                              │
│    ├─ SHAP analysis                                         │
│    └─ Feature importance ranking                           │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. VISUALIZATION & EXPORT                                   │
│    ├─ Generate 5 main figures (300 DPI)                    │
│    ├─ Export biomarker CSV files                           │
│    ├─ Create JSON results dictionary                       │
│    └─ Generate poster summary text                         │
└─────────────────────────────────────────────────────────────┘
```

### Machine Learning Models

| Model | Hyperparameters Tuned | Kernel/Solver Options |
|-------|----------------------|----------------------|
| **Random Forest** | n_estimators, max_depth, min_samples_split, min_samples_leaf | N/A |
| **SVM** | C, gamma, kernel | rbf, linear, poly |
| **Logistic Regression** | C, penalty, solver | lbfgs, liblinear, saga |

**Evaluation Metric:** ROC-AUC (primary), Accuracy, Precision, Recall, F1-Score

### Dependencies

```python
# Core Scientific Stack
numpy                    # Latest
pandas                   # Latest
scikit-learn            # Latest (Pipeline, CV, Models)
imbalanced-learn >= 0.8 # SMOTE within Pipeline
matplotlib              # Latest
seaborn                 # Latest

# Bioinformatics
GEOparse                # GEO database access

# Explainability
shap                    # Latest (upgraded in cells 3 & 13)

# Optional (Colab-specific)
google.colab            # Conditional import
```

**Installation:**
```bash
pip install GEOparse pandas numpy scikit-learn imbalanced-learn matplotlib seaborn shap
pip install shap --upgrade  # Important: ensure latest version
```

---

## Key Conventions

### Code Style

1. **Random Seed:** Always `SEED = 42` for reproducibility
2. **Language:** Mixed Korean (comments/outputs) and English (code/variables)
3. **Pipeline-First:** All preprocessing inside scikit-learn Pipeline (prevents leakage)
4. **Print Statements:** Progress indicators with ✓ checkmarks and section headers
5. **Error Handling:** Try-except blocks for file I/O and external dependencies

### Notebook Structure

**Cell Organization:**
```
[Markdown] → Section Header
[Code] → Implementation
[Code] → Visualization/Output (if applicable)
[Markdown] → Next Section Header
...
```

**Execution Order:** Cells **MUST** be run sequentially (top to bottom)

### Variable Naming

| Pattern | Example | Purpose |
|---------|---------|---------|
| `df_*` | `df_expression` | Pandas DataFrames |
| `best_*` | `best_model` | Optimal model/parameters |
| `final_*` | `final_results` | Final outputs |
| `*_path` | `output_path` | File system paths |
| Snake_case | `feature_importance` | Standard variables |
| UPPER_CASE | `SEED` | Constants |

### Data Conventions

- **Labels:** 0 = Healthy Control (HC), 1 = Colorectal Cancer (CRC)
- **Missing Data:** Handled via variance filtering and feature selection
- **Feature Names:** Preserved original miRNA probe IDs from platform
- **Splits:** Always stratified to maintain class distribution

### Output Conventions

**File Naming:**
- `*_final.png` → Final version ready for publication
- `*_all_mirnas.csv` → Complete feature set results
- `research_summary_for_poster.txt` → Presentation-ready summary

**Figure DPI:** 300 (publication quality)

**Results Format:**
- Metrics: Mean ± Std with 95% CI
- JSON: Structured dictionary for programmatic access
- CSV: Tab-separated or comma-separated based on use case

---

## Working with This Repository

### Initial Setup

```bash
# 1. Clone repository
git clone <repository-url>
cd GEOexosome

# 2. Install dependencies
pip install GEOparse pandas numpy scikit-learn imbalanced-learn matplotlib seaborn shap
pip install shap --upgrade

# 3. Create feature branch
git checkout -b claude/claude-md-<your-session-id>

# 4. Open notebook
jupyter notebook GEOexosome.ipynb
# OR upload to Google Colab
```

### Running the Analysis

**Option 1: Google Colab (Recommended)**
1. Upload `GEOexosome.ipynb` to Colab
2. Mount Google Drive (Cell 2)
3. Run all cells sequentially
4. Outputs saved to `/content/drive/MyDrive/cha poster/`

**Option 2: Local Jupyter**
1. Ensure all dependencies installed
2. Run cells in order
3. Outputs saved to current working directory

**Expected Runtime:** ~10-30 minutes (depends on dataset size and compute)

### Modifying the Analysis

**Safe to Modify:**
- Hyperparameter grids (expand search space)
- Feature selection parameters (k, variance threshold)
- SMOTE k_neighbors
- Visualization styles (colors, fonts)
- Output paths

**Modify with Caution:**
- Random seed (affects reproducibility)
- CV fold numbers (affects statistical validity)
- Pipeline order (can cause data leakage)
- GEO accession number (changes entire dataset)

**DO NOT Modify Unless Expert:**
- Nested CV structure (prevents leakage)
- SMOTE placement in pipeline (must be after split)
- Stratification settings (maintains class balance)

### Troubleshooting

| Issue | Solution |
|-------|----------|
| GEOparse download fails | Check internet connection, verify GSE39833 availability |
| SHAP errors | Run upgrade cells (3 & 13) again |
| Google Drive mount fails | Re-authenticate in Colab |
| Memory errors | Reduce k in SelectKBest, use fewer CV folds |
| Kernel crashes | Restart runtime, reduce hyperparameter grid size |

---

## Common Tasks

### Task 1: Change Dataset

```python
# In Cell 5, modify:
gse = GEOparse.get_GEO(geo="GSE39833", destdir="./data/")
# To:
gse = GEOparse.get_GEO(geo="GSE_NEW_ACCESSION", destdir="./data/")

# Important: Verify new dataset has compatible structure
# (expression matrix + sample phenotype data)
```

### Task 2: Add New Model

```python
# In Cell 6, add to classifiers dictionary:
classifiers = {
    "Random Forest": (RandomForestClassifier(random_state=SEED), param_grid_rf),
    "SVM": (SVC(random_state=SEED, probability=True), param_grid_svm),
    "Logistic Regression": (LogisticRegression(random_state=SEED, max_iter=1000), param_grid_lr),
    # NEW MODEL:
    "Gradient Boosting": (GradientBoostingClassifier(random_state=SEED), param_grid_gb)
}

# Define param_grid_gb before classifiers dict
```

### Task 3: Adjust Feature Selection

```python
# In Cell 6, modify pipeline steps:
Pipeline([
    ('variance_filter', VarianceThreshold(threshold=0.01)),  # Change threshold
    ('feature_selection', SelectKBest(f_classif, k=30)),     # Change k
    ('smote', SMOTE(random_state=SEED, k_neighbors=2)),
    ('scaler', StandardScaler()),
    ('classifier', classifier)
])
```

### Task 4: Export Additional Metrics

```python
# In Cell 11-12, add to final_results dictionary:
final_results[name]['sensitivity'] = recall_score(y_true, y_pred)
final_results[name]['specificity'] = specificity_calculation(y_true, y_pred)
final_results[name]['auprc'] = average_precision_score(y_true, y_pred_proba)
```

### Task 5: Modify Cross-Validation

```python
# In Cell 6, change CV folds:
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)  # Change n_splits
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)  # Change n_splits

# Warning: Reducing folds decreases statistical robustness
# Increasing folds increases computation time significantly
```

---

## Important Notes for AI Assistants

### Critical Best Practices

1. **ALWAYS Preserve Reproducibility**
   - Never change `SEED = 42` without explicit user request
   - Maintain stratified CV (prevents class imbalance bias)
   - Keep pipeline structure intact (prevents data leakage)

2. **Data Leakage Prevention**
   - All preprocessing MUST be inside Pipeline
   - Never fit scalers/transformers on full dataset before CV
   - SMOTE MUST be inside CV loop, not before

3. **Notebook Execution**
   - Cells MUST run in sequential order
   - Cell dependencies exist (later cells use variables from earlier cells)
   - Re-running middle cells without earlier cells will cause errors

4. **Version Control**
   - ALWAYS use `claude/` prefixed branches
   - Commit messages can be Korean or English
   - Push with retry logic (network failures are common)

5. **Environment Awareness**
   - Code adapts to Colab vs. local automatically
   - Don't hardcode paths (use conditional logic)
   - Test in target environment (Colab preferred)

### What to Check Before Making Changes

✓ Does change affect reproducibility? (seed, CV strategy)
✓ Could change introduce data leakage? (preprocessing placement)
✓ Will change break cell execution order?
✓ Are hyperparameter ranges reasonable? (avoid excessive grid search)
✓ Is SMOTE still inside pipeline and CV loop?
✓ Are visualization outputs still publication-quality? (DPI=300)
✓ Will outputs still be saved correctly? (path handling)

### Understanding Research Context

This is **academic research** with specific requirements:
- **Statistical rigor** is critical (nested CV, proper metrics)
- **Reproducibility** is mandatory (fixed seeds, documented methods)
- **Explainability** matters (SHAP analysis for biomarker discovery)
- **Publication quality** outputs (high DPI, proper formatting)
- **Overfitting prevention** is essential (nested CV, diagnostics)

### When to Ask User

Ask before:
- Changing random seed or CV strategy
- Modifying GEO accession number (changes entire dataset)
- Altering pipeline structure or order
- Removing SHAP or explainability analysis
- Changing primary evaluation metric (ROC-AUC)
- Major refactoring of notebook structure

Can proceed without asking:
- Fixing bugs or errors
- Adding new visualizations
- Expanding hyperparameter grids
- Improving code comments/documentation
- Optimizing runtime without changing logic
- Adding additional models for comparison

### Common Pitfalls to Avoid

❌ Fitting preprocessors before train-test split
❌ Using SMOTE on test data
❌ Changing random seed without tracking
❌ Breaking sequential cell execution
❌ Hardcoding file paths without environment detection
❌ Removing stratification from CV
❌ Ignoring class imbalance
❌ Forgetting to upgrade SHAP library
❌ Using GridSearchCV without nested CV
❌ Testing on training data (overfitting)

### Performance Optimization Tips

If notebook runs slowly:
1. Reduce hyperparameter grid size (fewer combinations)
2. Decrease CV folds (e.g., 3-fold outer instead of 5-fold)
3. Lower SelectKBest k value (fewer features)
4. Use `n_jobs=-1` for parallel processing (already implemented)
5. Sample dataset if very large (use stratified sampling)

### Code Modification Safety Levels

| Level | Description | Examples | Safe? |
|-------|-------------|----------|-------|
| **SAFE** | Style/output changes | Print statements, plot colors, figure sizes | ✓ |
| **LOW RISK** | Parameter tuning | Hyperparameter ranges, k in SelectKBest | ✓ |
| **MEDIUM RISK** | Algorithm changes | Add new models, change metrics | ⚠️ |
| **HIGH RISK** | Pipeline modification | Reorder steps, remove SMOTE | ⚠️⚠️ |
| **CRITICAL** | Core methodology | Change CV strategy, remove nesting | ❌ |

---

## Technical Details Reference

### Complete Notebook Cell Map

| Cell | Type | Purpose | Execution Time | Dependencies |
|------|------|---------|---------------|--------------|
| 1 | MD | Colab badge + header | N/A | None |
| 2 | Code | Drive mount + paths | ~5s | google.colab |
| 3 | Code | SHAP upgrade | ~10s | pip |
| 4 | Code | Imports + seed | ~5s | All libraries |
| 5 | Code | GEO data download | ~30-120s | Internet, GEOparse |
| 6 | Code | Nested CV + training | ~300-900s | Cell 4, 5 |
| 7 | MD | Overfitting section | N/A | None |
| 8 | Code | Overfitting analysis | ~10s | Cell 6 |
| 9 | MD | Visualization section | N/A | None |
| 10 | Code | Final evaluation | ~30s | Cell 6 |
| 11 | Code | Results aggregation | ~5s | Cell 6, 10 |
| 12 | Code | Generate figures | ~20s | Cell 11 |
| 13 | Code | SHAP upgrade check | ~10s | pip |
| 14 | Code | Feature importance | ~30s | Cell 6, 11 |
| 15 | Code | Poster summary | ~5s | Cell 11, 14 |
| 16 | MD | miRNA mapping header | N/A | None |
| 17 | Code | Top 5 miRNA analysis | ~10s | Cell 14 |

**Total Estimated Runtime:** 10-30 minutes (varies by compute and network)

### Key Algorithms Explained

**SMOTE (Synthetic Minority Over-sampling Technique):**
- Addresses class imbalance (HC vs CRC may be unbalanced)
- Creates synthetic samples for minority class
- Uses k-nearest neighbors (k=2-3 in this project)
- Applied INSIDE pipeline to prevent data leakage

**SelectKBest with f_classif:**
- ANOVA F-value feature selection
- Selects k best features (k=30-40)
- Reduces dimensionality and noise
- Improves model generalization

**Nested Cross-Validation:**
```
Outer Loop (5-fold):
  For each fold:
    Inner Loop (3-fold on training data only):
      - Try hyperparameter combinations
      - Select best hyperparameters
    - Train model with best hyperparameters on outer training set
    - Evaluate on outer test set
Return: Unbiased performance estimate
```

**SHAP (SHapley Additive exPlanations):**
- Model-agnostic explainability method
- Calculates feature contribution to predictions
- Based on game theory (Shapley values)
- Identifies most important biomarker miRNAs

### Performance Metrics Explained

| Metric | Formula | Interpretation | Priority |
|--------|---------|----------------|----------|
| **ROC-AUC** | Area under ROC curve | Overall classification ability (0.5-1.0) | PRIMARY |
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correctness | Secondary |
| **Precision** | TP/(TP+FP) | Positive prediction accuracy | Secondary |
| **Recall/Sensitivity** | TP/(TP+FN) | True positive rate | Secondary |
| **F1-Score** | 2×(Precision×Recall)/(Precision+Recall) | Balanced metric | Secondary |

**Why ROC-AUC is Primary:**
- Handles class imbalance well
- Threshold-independent
- Standard in medical diagnostics
- Compares performance across models fairly

---

## Project History

### Milestones

| Date | Event | Commit |
|------|-------|--------|
| Unknown | Initial repository creation | 10b7309 |
| Unknown | First poster submission code | 45d8fe5 |
| 2025-11-15 | CLAUDE.md creation | (current) |

### Known Issues / Future Work

(To be documented as project evolves)

Potential improvements:
- Add confidence interval visualization
- Include permutation importance analysis
- Test additional classifiers (XGBoost, Neural Networks)
- Implement feature selection comparison study
- Add biological pathway enrichment analysis
- Create interactive visualizations (plotly)
- Develop automated report generation
- Add statistical significance testing between models

---

## Contact & Collaboration

**Repository Owner:** Jay99Sohn
**Repository:** Jay99Sohn/GEOexosome
**Primary Branch:** (To be determined)
**Feature Branches:** `claude/claude-md-*`

### Collaboration Guidelines

1. **Code Review:** All changes should be reviewed before merging to main
2. **Testing:** Run complete notebook before committing
3. **Documentation:** Update this CLAUDE.md when making structural changes
4. **Issues:** Document any bugs or enhancement requests
5. **Reproducibility:** Always test with `SEED=42` for consistency

---

## Appendix: Quick Reference

### Essential Commands

```bash
# Git workflow
git checkout -b claude/claude-md-<session-id>
git add GEOexosome.ipynb
git commit -m "Descriptive message"
git push -u origin claude/claude-md-<session-id>

# Dependency installation
pip install GEOparse pandas numpy scikit-learn imbalanced-learn matplotlib seaborn shap
pip install shap --upgrade

# Jupyter notebook
jupyter notebook GEOexosome.ipynb
```

### Key Variables Quick Lookup

```python
SEED = 42                      # Random seed
X                              # Feature matrix (expression data)
y                              # Labels (0=HC, 1=CRC)
df_expression                  # Complete expression DataFrame
final_results                  # Dictionary with all results
best_model                     # Best trained classifier
final_model                    # Final fitted pipeline
```

### Hyperparameter Grids Quick Reference

```python
# Random Forest
n_estimators: [100]
max_depth: [4, 5]
min_samples_split: [5, 10]
min_samples_leaf: [2, 4]

# SVM
C: [0.1, 1.0, 10.0]
gamma: ['scale', 0.01]
kernel: ['rbf', 'linear', 'poly']

# Logistic Regression
C: [0.01, 0.1, 1.0]
penalty: ['l2']
solver: ['lbfgs', 'liblinear', 'saga']
```

### File Outputs Quick Reference

| File | Type | Content |
|------|------|---------|
| `confusion_matrix_final.png` | Image | 2×2 confusion matrix heatmap |
| `roc_curve_final.png` | Image | ROC curves with AUC values |
| `model_comparison.png` | Image | Bar chart of model performance |
| `learning_curves.png` | Image | Train/test score vs sample size |
| `cv_score_distribution.png` | Image | Box plots of CV scores |
| `feature_importance_all_mirnas.csv` | Data | RF feature importances |
| `shap_importance_all_mirnas.csv` | Data | SHAP-based rankings |
| `final_research_results.json` | Data | Complete results dictionary |
| `research_summary_for_poster.txt` | Text | Poster summary |

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-11-15 | Initial CLAUDE.md creation | AI Assistant |

---

**End of CLAUDE.md**

*This document should be updated whenever significant changes are made to the repository structure, analysis pipeline, or development workflows.*
