# Home-Credit-Default-Risk
 Data:https://www.kaggle.com/competitions/home-credit-default-risk/data
# Machine Learning Theory - Credit Risk Modeling

## Table of Contents
1. [Gradient Boosting Fundamentals](#gradient-boosting-fundamentals)
2. [Ensemble Learning](#ensemble-learning)
3. [Feature Engineering](#feature-engineering)
4. [Model Evaluation](#model-evaluation)
5. [Overfitting & Regularization](#overfitting--regularization)
6. [Cross-Validation](#cross-validation)

---

## 1. Gradient Boosting Fundamentals

### What is Gradient Boosting?

Gradient Boosting is an ensemble machine learning technique that builds a strong predictive model by combining multiple weak learners (typically decision trees) sequentially. Each new tree corrects the errors made by the previous trees.

### Mathematical Foundation

**Objective Function:**
```
L(y, F(x)) + Σ Ω(fₖ)
```

Where:
- `L(y, F(x))` = Loss function (measures prediction error)
- `Ω(fₖ)` = Regularization term (prevents overfitting)
- `fₖ` = k-th tree in the ensemble

**Additive Model:**
```
F_m(x) = F_{m-1}(x) + η·f_m(x)
```

Where:
- `F_m(x)` = Prediction after m trees
- `η` = Learning rate (controls contribution of each tree)
- `f_m(x)` = m-th weak learner

**Gradient Descent in Function Space:**

At each iteration, we fit a new tree to the negative gradient of the loss function:

```
f_m(x) ≈ -∇L(y, F_{m-1}(x))
```

### Why Gradient Boosting Works

1. **Bias-Variance Tradeoff**
   - Individual trees (weak learners) have high bias, low variance
   - Boosting reduces bias while controlling variance
   - Sequential learning focuses on hard-to-predict examples

2. **Additive Learning**
   - Each tree learns from residuals of previous trees
   - Progressive refinement of predictions
   - Flexibility to fit complex patterns

3. **Regularization**
   - Learning rate prevents overfitting
   - Tree depth limits complexity
   - Feature subsampling adds randomness

---

## 2. Ensemble Learning

### Types of Ensemble Methods

#### A. Bagging (Bootstrap Aggregating)
- **Example:** Random Forest
- **Method:** Train multiple models on random subsets of data
- **Combination:** Average predictions
- **Goal:** Reduce variance

#### B. Boosting
- **Examples:** LightGBM, XGBoost, CatBoost
- **Method:** Train models sequentially, focusing on errors
- **Combination:** Weighted sum
- **Goal:** Reduce bias

#### C. Stacking
- **Method:** Use predictions from multiple models as input to meta-learner
- **Combination:** Meta-model learns optimal combination
- **Goal:** Leverage strengths of different models

### Our Ensemble Strategy

We use **weighted averaging** of three gradient boosting models:

```python
P_final = 0.45·P_LightGBM + 0.35·P_XGBoost + 0.20·P_CatBoost
```

**Why This Works:**

1. **Diversity:** Each model uses different algorithms:
   - LightGBM: Leaf-wise tree growth
   - XGBoost: Level-wise tree growth
   - CatBoost: Ordered boosting

2. **Complementary Strengths:**
   - LightGBM: Fast, handles large datasets
   - XGBoost: Robust regularization
   - CatBoost: Native categorical handling

3. **Error Correlation:**
   - Models make different mistakes
   - Averaging reduces individual errors
   - Consensus is more reliable

### Ensemble Performance Gain

**Theoretical Upper Bound:**

If models are perfectly independent with error rate `ε`:
```
Ensemble Error ≈ ε²
```

In practice, models are correlated, but we still see:
- Single model: AUC ≈ 0.79
- Ensemble: AUC ≈ 0.80
- **Gain: +1-2% AUC**

---

## 3. Feature Engineering

### Why Feature Engineering?

> "Applied machine learning is basically feature engineering." - Andrew Ng

**Impact:**
- Raw features: AUC ≈ 0.74
- + Basic engineering: AUC ≈ 0.77
- + Advanced engineering: AUC ≈ 0.80
- **Total gain: +6% AUC**

### Feature Types

#### 1. **Aggregate Features**

Summarize information from related tables:

```python
# Example: Bureau credit summary
BUREAU_TOTAL_CREDIT = SUM(bureau.AMT_CREDIT_SUM)
BUREAU_AVG_CREDIT = MEAN(bureau.AMT_CREDIT_SUM)
BUREAU_ACTIVE_LOANS = COUNT(bureau WHERE CREDIT_ACTIVE='Active')
```

**Why:** Captures credit history in single features

#### 2. **Ratio Features**

Create normalized comparisons:

```python
CREDIT_INCOME_RATIO = AMT_CREDIT / AMT_INCOME_TOTAL
DEBT_CREDIT_RATIO = AMT_CREDIT_SUM_DEBT / AMT_CREDIT_SUM
```

**Why:** 
- Scale-invariant
- Captures relative burden
- More interpretable than raw amounts

#### 3. **Interaction Features**

Combine multiple features:

```python
EXT_SOURCE_PROD = EXT_SOURCE_1 × EXT_SOURCE_2 × EXT_SOURCE_3
EXT_WEIGHTED = 2·EXT_SOURCE_1 + 1·EXT_SOURCE_2 + 3·EXT_SOURCE_3
```

**Why:**
- Captures non-linear relationships
- Models feature synergies
- Boosts predictive power

#### 4. **Polynomial Features**

Non-linear transformations:

```python
EXT_SOURCE_1_SQUARED = (EXT_SOURCE_1)²
EXT_SOURCE_1_CUBED = (EXT_SOURCE_1)³
EXT_SOURCE_1_SQRT = √(EXT_SOURCE_1)
```

**Why:**
- Captures non-linear patterns
- Helps with curved decision boundaries

#### 5. **Domain-Specific Features**

Based on credit risk knowledge:

```python
LATE_PAYMENT_RATIO = LATE_PAYMENTS / TOTAL_PAYMENTS
CREDIT_UTILIZATION = BALANCE / CREDIT_LIMIT
PAYMENT_BURDEN = ANNUITY / INCOME
```

**Why:**
- Incorporates domain expertise
- Directly measures risk indicators
- Highly predictive

### Feature Engineering Best Practices

1. **Start with Domain Knowledge**
   - What matters for credit risk?
   - What do underwriters look at?

2. **Handle Missing Values Carefully**
   - Missing can be informative
   - Use median/mode imputation
   - Create "is_missing" flags

3. **Avoid Data Leakage**
   - Don't use future information
   - Validate temporal ordering
   - Be careful with aggregations

4. **Test Incrementally**
   - Add features in groups
   - Measure impact on CV score
   - Remove low-impact features

---

## 4. Model Evaluation

### AUC-ROC Metric

**ROC Curve:** Plots True Positive Rate vs False Positive Rate

```
TPR = TP / (TP + FN)  # Sensitivity/Recall
FPR = FP / (FP + TN)  # 1 - Specificity
```

**AUC (Area Under Curve):**
- Measures discrimination ability
- Probability that model ranks random positive > random negative
- Range: [0, 1], where 0.5 = random, 1.0 = perfect

### Why AUC for Credit Risk?

1. **Threshold-Independent**
   - Don't need to choose cutoff
   - Evaluates ranking quality
   - Useful when cost of FP/FN unknown

2. **Imbalanced Data**
   - Works well with 8% default rate
   - Focuses on ranking, not absolute probabilities
   - More robust than accuracy

3. **Business Interpretation**
   - AUC = 0.80 means 80% chance of correct ranking
   - Directly relates to risk stratification
   - Easy to explain to stakeholders

### Confusion Matrix Analysis

For threshold t = 0.5:

```
                Predicted
              0         1
Actual 0   TN=45000   FP=2000
       1   FN=1500    TP=1500
```

**Metrics:**
- Accuracy = (TN+TP) / Total = 93%
- Precision = TP / (TP+FP) = 43%
- Recall = TP / (TP+FN) = 50%
- F1-Score = 2·(P·R)/(P+R) = 46%

**Business Impact:**
- FP = Deny good customer (lost revenue)
- FN = Approve bad customer (credit loss)
- Cost ratio guides threshold selection

---

## 5. Overfitting & Regularization

### What is Overfitting?

**Definition:** Model learns training data too well, including noise, and fails to generalize.

**Signs:**
- High training accuracy, low test accuracy
- Large gap between train/validation performance
- Model too complex for available data

### Regularization Techniques

#### 1. **L1 Regularization (Lasso)**

```
Loss = MSE + α·Σ|wᵢ|
```

**Effect:**
- Pushes some weights to exactly zero
- Performs automatic feature selection
- Creates sparse models

#### 2. **L2 Regularization (Ridge)**

```
Loss = MSE + α·Σ(wᵢ)²
```

**Effect:**
- Shrinks weights toward zero
- Prevents large coefficients
- More stable than L1

#### 3. **Tree-Specific Regularization**

**Max Depth:**
- Limits tree complexity
- Prevents overfitting to noise
- Typical values: 6-12

**Min Child Samples:**
- Minimum data points in leaf
- Prevents tiny, overfit leaves
- Typical values: 20-50

**Learning Rate:**
- Shrinks contribution of each tree
- Lower = better generalization, slower training
- Typical values: 0.01-0.05

**Feature/Sample Subsampling:**
- Use random subset of features/samples per tree
- Adds randomness, reduces overfitting
- Typical values: 0.8-0.9

### Our Regularization Strategy

```python
params = {
    'learning_rate': 0.015,      # Slow learning
    'max_depth': 8-9,            # Moderate complexity
    'min_child_samples': 25-30,  # Prevent small leaves
    'subsample': 0.88,           # Row sampling
    'colsample_bytree': 0.88,    # Column sampling
    'reg_alpha': 0.6,            # L1 penalty
    'reg_lambda': 0.6,           # L2 penalty
}
```

**Result:** Gap between train/val < 1%, indicating good generalization

---

## 6. Cross-Validation

### Why Cross-Validation?

1. **Robust Evaluation**
   - Single train/val split can be lucky/unlucky
   - CV averages over multiple splits
   - More reliable performance estimate

2. **Use All Data**
   - Every sample used for both training and validation
   - No data wasted
   - Important with limited data

3. **Detect Overfitting**
   - High variance across folds = overfitting
   - Consistent scores = good generalization

### Stratified K-Fold

**Standard K-Fold:**
```
Split data into K equal parts
For each fold:
  Train on K-1 folds
  Validate on 1 fold
Average results
```

**Stratified K-Fold:**
- Maintains target distribution in each fold
- Critical for imbalanced data (8% defaults)
- Ensures representative validation sets

### Our CV Strategy

```python
n_folds = 10
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
```

**Benefits:**
- 10 folds = robust estimate
- Stratification = consistent target ratio
- Shuffle = remove ordering bias
- Random state = reproducibility

**Results:**
- Fold scores: [0.799, 0.801, 0.803, 0.802, 0.800, ...]
- Mean: 0.801
- Std: 0.002
- **Low variance = stable model**

### Out-of-Fold (OOF) Predictions

```python
oof_preds = np.zeros(len(train))

for fold in range(n_folds):
    # Train on other folds
    model.fit(X_train_fold, y_train_fold)
    
    # Predict on this fold
    oof_preds[val_idx] = model.predict(X_val_fold)

# OOF predictions = predictions on entire train set
# Used for: stacking, ensemble, calibration
```

**Advantage:** Unbiased predictions on training data

---

## Key Takeaways

1. **Gradient Boosting** builds strong models by combining weak learners sequentially
2. **Ensembles** leverage diversity to outperform individual models
3. **Feature Engineering** is the most impactful step (6% AUC gain)
4. **AUC-ROC** is ideal for imbalanced classification and ranking
5. **Regularization** prevents overfitting through multiple techniques
6. **Stratified CV** provides robust, unbiased performance estimates

---

## Further Reading

1. **Books:**
   - "The Elements of Statistical Learning" - Hastie, Tibshirani, Friedman
   - "Hands-On Machine Learning" - Aurélien Géron

2. **Papers:**
   - LightGBM: https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree
   - XGBoost: https://arxiv.org/abs/1603.02754
   - CatBoost: https://arxiv.org/abs/1706.09516

3. **Online Courses:**
   - Andrew Ng's Machine Learning (Coursera)
   - Fast.ai Practical Deep Learning
   - Kaggle Learn

4. **Communities:**
   - Kaggle Forums
   - Cross Validated (Stack Exchange)
   - Reddit r/MachineLearning


