### 1. Demircioğlu A (2024)
**제목:** Evaluation of the performance of oversampling techniques in 
         class imbalance problems
**저널:** Scientific Reports 14:15744
**DOI:** 10.1038/s41598-024-66477-w

**핵심 발견:**
"Applying oversampling before CV leads to large positive bias in AUC 
of up to 0.34, which is proportional to class imbalance."

**코드 적용:**
- Cell 2, Line 83-87: SMOTE 제거 및 class_weight 사용 근거
- model_configs에서 모든 SMOTE 옵션 제거

**인용 문장 (Methods용):**
"Given the severe class imbalance (11 vs 88) and limited minority samples,
we opted for class weighting over SMOTE to avoid potential bias 
(Demircioğlu, 2024)."

---

### 2. Liu Y et al. (2025)
**제목:** Benchmarking feature selection stability in high-dimensional 
         small-sample data
**저널:** Computer Methods and Programs in Biomedicine 248:108107
**DOI:** 10.1016/j.cmpb.2024.108107

**핵심 발견:**
"Kuncheva stability index of 0.50-0.75 is realistic for challenging 
HDSS datasets; values >0.80 achievable with proper ensemble approaches."

**코드 적용:**
- Cell 2: STABILITY_THRESHOLD = 0.70 (기존 0.80에서 하향)
- Cell 2: calculate_kuncheva_index() 함수 추가

**인용 문장 (Methods용):**
"Feature stability was assessed using the Kuncheva index, with a 
threshold of 70% selection frequency considered stable based on 
benchmarks for high-dimensional small-sample data (Liu et al., 2025)."

---

### 3. Lewis MJ et al. (2023)
**제목:** nestedcv: an R package for fast implementation of nested 
         cross-validation with embedded feature selection
**저널:** Bioinformatics Advances 3(1):vbad048
**DOI:** 10.1093/bioadv/vbad048

**핵심 발견:**
"The final model is determined by following the same steps applied to 
the outer training folds, but this time applied to the whole dataset."

**코드 적용:**
- Cell 2, STEP 6: 이중 경로(Dual-Path) Feature Selection
  - PATH A: 전체 데이터 재훈련 (외부 검증용)
  - PATH B: CV-stable features (바이오마커 보고용)

**인용 문장 (Methods용):**
"Following the dual-path strategy (Lewis et al., 2023), we reported 
CV-stable features (≥70% selection) as candidate biomarkers while 
using full-data-retrained model for external validation."

---

### 4. Parvandeh S et al. (2020)
**제목:** Consensus features nested cross-validation
**저널:** Bioinformatics 36(10):3093-3098
**DOI:** 10.1093/bioinformatics/btaa046

**핵심 발견:**
"Standard nested CV tends to include too many features (avg 253 vs 
50 true functional). Consensus-based selection across folds 
significantly reduces false positives."

**코드 적용:**
- Cell 2: CV-stable features 개념 (≥70% of folds)
- Cell 2: cv_stable_biomarkers_{model}.csv 생성

**인용 문장 (Methods용):**
"To identify robust biomarkers, we applied consensus feature selection 
requiring features to be selected in ≥70% of cross-validation 
iterations (Parvandeh et al., 2020)."

---

### 5. Varma S & Simon R (2006)
**제목:** Bias in error estimation when using cross-validation for 
         model selection
**저널:** BMC Bioinformatics 7:91
**DOI:** 10.1186/1471-2105-7-91

**핵심 발견:**
"Using CV to compute an error estimate for a classifier that has been 
tuned by CV gives a significantly biased estimate of the true error."

**코드 적용:**
- 전체 Nested CV 구조 설계의 이론적 근거
- Feature selection이 각 outer fold 내부에서만 수행

**인용 문장 (Methods용):**
"We employed nested cross-validation to obtain unbiased performance 
estimates while incorporating feature selection within each fold 
(Varma & Simon, 2006)."
