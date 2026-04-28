# Credit Risk Prediction System — Architecture Document

## 1. Project Overview

This project implements an end-to-end **Credit Risk Prediction System** that classifies loan applicants as either **low risk** (likely to repay) or **high risk** (likely to default). The system includes a trained machine learning model and an interactive Streamlit web interface for real-time predictions.

---

## 2. Dataset

### Source
- **File:** `credit_risk_dataset.csv`
- **Records:** ~32,000 rows
- **Features:** 12 columns (11 input features + 1 target)

### Feature Description

| Feature | Type | Description |
|---------|------|-------------|
| `person_age` | Numeric | Age of the applicant |
| `person_income` | Numeric | Annual income ($) |
| `person_home_ownership` | Categorical | RENT, OWN, MORTGAGE, OTHER |
| `person_emp_length` | Numeric | Employment length (years) |
| `loan_intent` | Categorical | Purpose: PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION |
| `loan_grade` | Categorical | Risk grade assigned: A (best) to G (worst) |
| `loan_amnt` | Numeric | Loan amount requested ($) |
| `loan_int_rate` | Numeric | Interest rate (%) |
| `loan_percent_income` | Numeric | Loan amount as a fraction of income (0–1) |
| `cb_person_default_on_file` | Categorical | Historical default: Y / N |
| `cb_person_cred_hist_length` | Numeric | Credit history length (years) |

### Target Variable
- **`loan_status`**: Binary — `0` = Non-default (repaid), `1` = Default

---

## 3. Data Preprocessing Pipeline

```
Raw CSV → Handle Missing Values → Encode Categoricals → Feature/Target Split → Train/Test Split
```

### 3.1 Missing Value Treatment
| Data Type | Strategy |
|-----------|----------|
| Numeric | Filled with **median** (robust to outliers) |
| Categorical | Filled with **mode** (most frequent value) |

### 3.2 Categorical Encoding
- **Method:** `LabelEncoder` (sklearn)
- Converts categorical strings to integer labels (alphabetically sorted)
- Detection uses `is_string_dtype()` for compatibility with pandas 3.x `StringDtype`

### 3.3 Train/Test Split
- **Ratio:** 80% training / 20% testing
- **Random State:** 42 (reproducibility)

---

## 4. Model Selection

### Algorithm: Logistic Regression

| Aspect | Rationale |
|--------|-----------|
| **Interpretability** | Coefficients directly indicate feature importance and direction of influence |
| **Binary Classification** | Native support for two-class problems (default vs. non-default) |
| **Probability Output** | Produces calibrated probabilities, useful for risk scoring |
| **Efficiency** | Fast to train and predict, suitable for real-time inference |
| **Baseline Strength** | Strong baseline for tabular data; easy to benchmark against |
| **Regulatory Compliance** | Transparent, explainable model preferred in financial domains |

### Hyperparameters
| Parameter | Value | Reason |
|-----------|-------|--------|
| `max_iter` | 2000 | Allow convergence on high-dimensional encoded data |
| `solver` | lbfgs (default) | Efficient for small-to-medium datasets |

---

## 5. Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | ~82.8% |

### Interpretation
- The model correctly classifies approximately 83% of applicants
- This is a solid baseline; further improvements possible with feature scaling, ensemble methods, or hyperparameter tuning

### Known Limitation
- Convergence warning observed — the model benefits from feature scaling (StandardScaler) which was not applied in this iteration

---

## 6. Dimensionality Reduction (Exploratory)

- **Method:** PCA (Principal Component Analysis)
- **Components:** Reduced from 11 features → 2 components
- **Purpose:** Visualization only — scatter plot colored by loan status to observe class separability
- **Finding:** Partial separability visible, confirming that the feature set carries discriminative signal

---

## 7. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Training Pipeline                      │
│                                                          │
│  credit_risk_dataset.csv                                 │
│         │                                                │
│         ▼                                                │
│  Data Preprocessing (fillna, LabelEncoder)               │
│         │                                                │
│         ▼                                                │
│  Train/Test Split (80/20)                                │
│         │                                                │
│         ▼                                                │
│  Logistic Regression (max_iter=2000)                     │
│         │                                                │
│         ▼                                                │
│  Evaluation (accuracy_score) ──► 82.8%                   │
│         │                                                │
│         ▼                                                │
│  joblib.dump() ──► credit_risk_model.pkl                 │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   Inference Pipeline                      │
│                                                          │
│  Streamlit UI (app.py)                                   │
│         │                                                │
│         ▼                                                │
│  User inputs 11 feature values                           │
│         │                                                │
│         ▼                                                │
│  Encode categoricals (same mapping as training)          │
│         │                                                │
│         ▼                                                │
│  model.predict() + model.predict_proba()                 │
│         │                                                │
│         ▼                                                │
│  Display: Low Risk ✅ / High Risk ⚠️ + Confidence %     │
└─────────────────────────────────────────────────────────┘
```

---

## 8. Project Structure

```
roshan-project/
├── credit_risk_dataset.csv      # Raw dataset
├── main.ipynb                   # Training notebook (EDA + model)
├── app.py                       # Streamlit prediction UI
├── credit_risk_model.pkl        # Serialized trained model
├── feature_names.pkl            # Feature column names
├── test_model.py                # Test cases (pytest)
└── README.md                    # This document
```

---

## 9. Future Improvements

| Area | Recommendation |
|------|----------------|
| **Feature Scaling** | Apply `StandardScaler` to resolve convergence warning and potentially boost accuracy |
| **Advanced Models** | Try Random Forest, XGBoost, or LightGBM for higher accuracy |
| **Cross-Validation** | Use k-fold CV instead of single train/test split |
| **Class Imbalance** | Apply SMOTE or class weights if target is imbalanced |
| **Feature Engineering** | Create interaction features (e.g., loan_amnt × int_rate) |
| **Model Explainability** | Add SHAP values for per-prediction explanations |
| **Deployment** | Containerize with Docker, deploy on cloud (Azure/AWS) |

---

## 10. Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.14 |
| ML Library | scikit-learn 1.8 |
| Data | pandas 3.0 |
| Visualization | matplotlib |
| Serialization | joblib |
| Web UI | Streamlit |
| Testing | pytest |

---

*Document prepared as part of the Credit Risk ML project.*
