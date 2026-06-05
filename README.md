# 🏦 Bank Customer Churn Analysis

A complete end-to-end data science project that identifies bank customers at risk of churning using machine learning. Built with a real-world dataset of 10,000 customers, this project walks through every stage of the data science lifecycle — from raw data to production-ready models.

---

## 📌 Problem Statement

Customer churn is one of the most costly challenges in banking. Acquiring a new customer costs 5–7× more than retaining an existing one. This project builds a churn prediction system to help banks proactively identify high-risk customers and take targeted retention actions before it's too late.

---

## 📊 Dataset

- **Source:** Kaggle — [Bank Customer Churn Modelling](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)
- **Size:** 10,000 customer records × 14 features
- **Target:** `Exited` (1 = churned, 0 = retained)
- **Churn Rate:** ~20.4% — a moderately imbalanced classification problem

| Feature | Type | Description |
|---|---|---|
| `CreditScore` | Numerical | Customer's credit score |
| `Geography` | Categorical | Country: France, Germany, Spain |
| `Gender` | Categorical | Male / Female |
| `Age` | Numerical | Age of the customer |
| `Tenure` | Numerical | Years as a bank customer |
| `Balance` | Numerical | Account balance (€) |
| `NumOfProducts` | Numerical | Number of bank products held |
| `HasCrCard` | Binary | Owns a credit card (1/0) |
| `IsActiveMember` | Binary | Is an active member (1/0) |
| `EstimatedSalary` | Numerical | Estimated annual salary (€) |
| `Exited` | Binary | **Target** — churned (1) or not (0) |

---

## 🔧 Project Pipeline

### 1. Data Cleaning
- Dropped non-predictive identifier columns: `RowNumber`, `CustomerId`, `Surname`
- Confirmed zero null values and no duplicate records — dataset is clean out of the box
- Applied a domain-logic outlier filter removing any `CreditScore` values below 300, which are practically invalid for a functioning bank customer

### 2. Exploratory Data Analysis (EDA)
- Visualised the overall churn distribution — 79.6% retained vs. 20.4% churned
- Overlaid histograms across all numerical features split by churn status, revealing that older customers (45–60) and those with very high or zero balances churn significantly more
- Computed churn rates per categorical group — Germany shows nearly double the churn rate (~32%) compared to France and Spain (~16%), and inactive members churn at ~27% vs. ~14% for active ones
- Scatter-plotted Age vs. Balance to expose the clustering of churned customers in the high-age, high-balance quadrant

### 3. Feature Engineering
Five new features were crafted to capture relationships that raw columns cannot express individually:

- **`BalancePerProduct`** — normalises account balance by the number of products held, helping distinguish between large-balance customers with many products vs. those concentrating wealth in few accounts
- **`SalaryToBalance`** — compares estimated salary to actual balance, surfacing customers who earn well but hold little — a potential dissatisfaction signal
- **`IsZeroBalance`** — binary flag marking customers with a €0 balance, who may be dormant or transitioning away from the bank
- **`AgeGroup`** — buckets age into four life-stage bands (Under 30 / 30–45 / 45–60 / 60+), allowing models to pick up non-linear age effects
- **`TenurePerAge`** — ratio of tenure to age, capturing whether a customer has been with the bank for most of their adult life or joined recently relative to their age

### 4. Modelling
Three classifiers were trained and evaluated, covering the spectrum from interpretable to complex:

- **Logistic Regression** — a linear baseline that provides interpretable coefficients and serves as a sanity check for the more powerful models
- **Random Forest** — an ensemble of 200 decision trees that captures non-linear feature interactions and ranks feature importance by mean decrease in impurity
- **XGBoost** — a gradient-boosted tree model fine-tuned with a lower learning rate and capped depth, offering the strongest predictive performance in this experiment

All models were evaluated on a stratified 80/20 train-test split, and ROC-AUC scores were validated with 5-fold cross-validation to confirm there was no overfitting.

---

## 📈 Results

| Model | Accuracy | F1 Score | Precision | Recall | ROC-AUC | CV AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 80.5% | 0.278 | 0.564 | 0.184 | 0.773 | 0.763 |
| Random Forest | 86.7% | 0.584 | 0.799 | 0.460 | 0.850 | 0.852 |
| **XGBoost** | **86.9%** | **0.599** | **0.790** | **0.482** | **0.864** | **0.861** |

**XGBoost is the best-performing model**, achieving an ROC-AUC of 0.864 — meaning the model correctly ranks a churner above a non-churner ~86% of the time. It correctly identifies ~48% of all churners (recall) while keeping false alarms low (79% precision), making it practically deployable for a targeted retention campaign.

### Key Findings from EDA
- **Age is the strongest churn predictor** — customers aged 45–60 churn at nearly 3× the rate of customers under 30
- **Germany has a significantly higher churn rate (~32%)** compared to France and Spain, suggesting geography-specific service or competitive pressures
- **Inactive members are twice as likely to churn** as active members, making engagement the most actionable intervention point
- **Customers holding only 1 product churn the most** — cross-selling is a natural retention lever
- **Zero-balance customers are disproportionately represented among churners**, often indicating dormant or transitioning customers

---

## 🗂️ Project Structure

```
bank-churn-analysis/
│
├── data/
│   └── Churn_Modelling.csv          # Raw dataset
│
├── notebooks/
│   └── bank_churn_analysis.ipynb    # Full interactive notebook
│
├── src/
│   └── analysis.py                  # Standalone Python pipeline script
│
├── models/
│   ├── Logistic_Regression.pkl      # Saved trained model
│   ├── Random_Forest.pkl            # Saved trained model
│   ├── XGBoost.pkl                  # Saved trained model
│   ├── scaler.pkl                   # Fitted StandardScaler
│   └── model_results.json           # Evaluation metrics (JSON)
│
├── plots/
│   ├── 01_churn_distribution.png
│   ├── 02_feature_distributions.png
│   ├── 03_categorical_churn_rates.png
│   ├── 04_age_vs_balance.png
│   ├── 05_model_comparison.png
│   ├── 06_roc_curves.png
│   ├── 07_confusion_matrices.png
│   ├── 08_feature_importances.png
│   └── 09_correlation_heatmap.png
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/Chakravarthi-roy/ChurnPrediction.git
cd bank-churn-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline
```bash
python src/analysis.py
```
This will regenerate all plots in `/plots` and re-save all models in `/models`.

### 4. Explore interactively
```bash
jupyter notebook notebooks/bank_churn_analysis.ipynb
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
joblib
jupyter
```

---

## 🚀 Future Improvements

- **Handle class imbalance** with SMOTE or class-weight tuning to further improve recall on churners
- **Hyperparameter tuning** via `GridSearchCV` or `Optuna` for XGBoost and Random Forest
- **SHAP values** for individual-level explainability — understanding *why* a specific customer is predicted to churn
- **Deploy as a REST API** using FastAPI + Docker so the model can be integrated into a CRM system
- **A/B test retention strategies** by using model scores to segment customers and measure real-world retention lift
