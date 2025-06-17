
# 🧠 Bank Defaulter Risk Prediction Model (Convolve 3.0)

> ⚡ Built for Convolve 3.0 by **Team Neural Thinkers**

## 👋 Author
**Uttkarsh Solanki**  
📧 uttkarsh2003.solanki@gmail.com

---

## 🚀 Overview

This project builds a **high-performance defaulter prediction system** for a large anonymized credit dataset provided by a leading bank. The key goal is to develop a **Behavior Score**—a comprehensive customer-level metric to assess creditworthiness and enable dynamic risk management decisions.

The solution leverages:
- ⚙️ Deep Data Preprocessing
- 📉 PCA & Dimensionality Reduction Trials
- 🌲 Ensemble Learning (LightGBM, XGBoost, CatBoost)
- 📊 Advanced Metrics (PR AUC, F1-Score for minority class)
- 🧮 Final Credit Behavior Score (0 to 100 scale)

---

## 🗂️ Dataset Description

The raw dataset included **over 1200 features** with four major types:

1. **Onus Attributes**: Credit limits, obligations  
2. **Transaction Attributes**: Merchant-level activity  
3. **Bureau Tradelines**: Repayment history, products held  
4. **Bureau Inquiries**: Personal loan queries  

**🔑 Target**: `BAD_FLAG` (1 = Defaulter, 0 = Good)

### 🔍 Challenges

- 96,806 training rows with **1,216 columns**  
- 41,792 validation rows with **1,215 columns**
- High **null value frequency**
- Significant **class imbalance** (0 ≫ 1)

---

## 🧪 Methodology

### 🧼 Data Cleaning
- Removed columns with **>30% nulls**
- Dropped **low-variance** columns (only 1 unique value)
- Nulls imputed using:
  - **Median** for continuous features
  - **Mode** for categorical/binary

### ⚖️ Class Imbalance Strategy
- Used **manual class weights** to emphasize minority class
- Split data into **5 balanced datasets**
  - Each set: All class-1 rows + a split of class-0 rows
  - Trained separate models and ensembled results

### 📉 Dimensionality Reduction
Tried:
- PCA  
- ANOVA  
- Chi-Square  

⛔ But all led to degraded PR AUC and F1 scores  
✅ Final model used **no dimensionality reduction** to retain rare patterns

---

## 🧠 Model Selection

We evaluated:
- ✅ **LightGBM** (best performer)
- ⚠️ XGBoost
- ⚠️ CatBoost

### 💡 Why LightGBM?
- Highest **PR AUC (0.074)** and **F1 Score (0.12)** on minority class
- Leaf-wise growth + histogram optimization = **faster + better**

### 🔁 Ensemble (Grouped Resampling + StratifiedKFold)
- Built 5 LightGBM models with different subsets
- Final predictions = **average of 5 outputs**
- Improved:
  - PR AUC → **0.0851**
  - ROC AUC → **0.8285**
  - Minority class recall

---

## 📈 Evaluation Metrics

| Metric            | Value     |
|-------------------|-----------|
| Baseline PR AUC   | 0.014     |
| LightGBM PR AUC   | 0.0740    |
| Ensemble PR AUC   | 0.0851    |
| F1 Score (Class 1)| 0.12      |
| ROC AUC           | 0.8285    |

**🧠 Note:** PR AUC preferred over accuracy due to severe class imbalance

---

## 📊 Behavior Score

Introduced a **Behavior Score**:
```text
Behavior Score = (1 - Predicted Default Probability) × 100
```

- **0** = Worst behavior (definite default)
- **100** = Best behavior (very low risk)
- Used to **segment customers** and recommend:
  - Lower credit limits
  - Custom payment plans
  - Better offers to reliable customers

---

## 🎯 Final Results

- LightGBM ensemble achieved the **best trade-off** between class separation and recall on minority class
- Validated model on unseen dataset with cleaned and imputed features
- Generated behavior scores for individual customers

---

## 🔮 Future Scope

- Integrate real-time data streams (e.g., customer transactions)
- Expand to **loans, mortgages, savings accounts**
- Add explainability modules (e.g., SHAP values for scoring)
- Deploy scoring API with automated updates

---

## 📂 Files in this Repository

| File              | Description                            |
|------------------|----------------------------------------|
| `Convolve_3.ipynb` | Full modeling notebook with preprocessing, modeling, and prediction logic |
| `README.md`      | Project summary (this file)            |
| `*.csv` / `*.pkl` | (Optional) Input data and model files  |

---

## 🙌 Acknowledgement

Special thanks to the **Convolve 3.0** organizing team and the data providers for this opportunity.  
Project by **Team Neural Thinkers**.

---
