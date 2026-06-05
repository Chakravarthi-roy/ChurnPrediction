"""
Bank Customer Churn Analysis
Full pipeline: Cleaning → EDA → Feature Engineering → Modeling → Evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os
import json
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, f1_score, accuracy_score, precision_score, recall_score
)
from xgboost import XGBClassifier
import joblib

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE, "data", "Churn_Modelling.csv")
PLOTS_DIR   = os.path.join(BASE, "plots")
MODELS_DIR  = os.path.join(BASE, "models")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

PALETTE = {"churned": "#E63946", "retained": "#457B9D", "neutral": "#1D3557"}
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "font.family": "DejaVu Sans"})

# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD & CLEAN
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 1: DATA LOADING & CLEANING")
print("="*60)

df = pd.read_csv(DATA_PATH)
print(f"Raw shape: {df.shape}")

# Drop identifiers — not predictive
df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)
print(f"After dropping identifiers: {df.shape}")

# Null check
nulls = df.isnull().sum()
print(f"\nNull values:\n{nulls[nulls > 0] if nulls.any() else 'None — clean dataset ✓'}")

# Duplicate check
dupes = df.duplicated().sum()
print(f"Duplicate rows: {dupes}")
if dupes:
    df.drop_duplicates(inplace=True)

# Outlier flag: CreditScore < 300 is practically invalid
low_cs = (df["CreditScore"] < 300).sum()
print(f"CreditScore < 300 (outliers): {low_cs}")
df = df[df["CreditScore"] >= 300]

print(f"\nClean shape: {df.shape}")
print("\nData types after cleaning:")
print(df.dtypes)

# ═══════════════════════════════════════════════════════════════════════════
# 2. EDA
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*60)

churn_counts = df["Exited"].value_counts()
churn_rate   = df["Exited"].mean() * 100
print(f"Overall churn rate: {churn_rate:.2f}%")
print(f"Retained: {churn_counts[0]:,}  |  Churned: {churn_counts[1]:,}")

# ── Plot 1: Churn Distribution ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Churn Distribution Overview", fontsize=15, fontweight="bold", y=1.01)

colors = [PALETTE["retained"], PALETTE["churned"]]
axes[0].bar(["Retained", "Churned"], churn_counts.values, color=colors, edgecolor="white", linewidth=1.5)
axes[0].set_title("Customer Count by Churn Status")
axes[0].set_ylabel("Number of Customers")
for i, v in enumerate(churn_counts.values):
    axes[0].text(i, v + 50, f"{v:,}", ha="center", fontweight="bold")

axes[1].pie(churn_counts.values, labels=["Retained", "Churned"],
            autopct="%1.1f%%", colors=colors, startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2))
axes[1].set_title("Churn Rate Proportion")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "01_churn_distribution.png"), bbox_inches="tight")
plt.close()
print("Saved: 01_churn_distribution.png")

# ── Plot 2: Numerical Feature Distributions ───────────────────────────────
num_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Numerical Feature Distributions by Churn", fontsize=14, fontweight="bold")

for ax, col in zip(axes.flatten(), num_cols):
    for val, label, color in [(0, "Retained", PALETTE["retained"]),
                               (1, "Churned",  PALETTE["churned"])]:
        ax.hist(df[df["Exited"] == val][col], bins=30, alpha=0.6,
                label=label, color=color, edgecolor="white")
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "02_feature_distributions.png"), bbox_inches="tight")
plt.close()
print("Saved: 02_feature_distributions.png")

# ── Plot 3: Categorical Churn Rates ──────────────────────────────────────
cat_cols = ["Geography", "Gender", "HasCrCard", "IsActiveMember", "NumOfProducts"]
fig, axes = plt.subplots(1, len(cat_cols), figsize=(18, 5))
fig.suptitle("Churn Rate by Categorical Features", fontsize=14, fontweight="bold")

for ax, col in zip(axes, cat_cols):
    churn_by = df.groupby(col)["Exited"].mean() * 100
    bars = ax.bar(churn_by.index.astype(str), churn_by.values,
                  color=[PALETTE["churned"] if v > churn_rate else PALETTE["retained"]
                         for v in churn_by.values],
                  edgecolor="white", linewidth=1.5)
    ax.axhline(churn_rate, color="black", linestyle="--", linewidth=1, label=f"Avg {churn_rate:.1f}%")
    ax.set_title(col)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel(col)
    ax.legend(fontsize=7)
    for bar, val in zip(bars, churn_by.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", fontsize=8, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "03_categorical_churn_rates.png"), bbox_inches="tight")
plt.close()
print("Saved: 03_categorical_churn_rates.png")

# ── Plot 4: Age vs Balance scatter ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
for val, label, color in [(0, "Retained", PALETTE["retained"]),
                           (1, "Churned",  PALETTE["churned"])]:
    subset = df[df["Exited"] == val]
    ax.scatter(subset["Age"], subset["Balance"], alpha=0.3, s=12,
               label=label, color=color)
ax.set_title("Age vs. Account Balance by Churn Status", fontsize=13, fontweight="bold")
ax.set_xlabel("Age")
ax.set_ylabel("Account Balance (€)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "04_age_vs_balance.png"), bbox_inches="tight")
plt.close()
print("Saved: 04_age_vs_balance.png")

# ═══════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 3: FEATURE ENGINEERING")
print("="*60)

df_fe = df.copy()

# Encode categoricals
le_gender = LabelEncoder()
df_fe["Gender"] = le_gender.fit_transform(df_fe["Gender"])          # Female=0, Male=1

geo_dummies = pd.get_dummies(df_fe["Geography"], prefix="Geo", drop_first=False)
df_fe = pd.concat([df_fe.drop("Geography", axis=1), geo_dummies], axis=1)

# New engineered features
df_fe["BalancePerProduct"]   = df_fe["Balance"] / (df_fe["NumOfProducts"] + 1)
df_fe["SalaryToBalance"]     = df_fe["EstimatedSalary"] / (df_fe["Balance"] + 1)
df_fe["IsZeroBalance"]       = (df_fe["Balance"] == 0).astype(int)
df_fe["AgeGroup"]            = pd.cut(df_fe["Age"],
                                       bins=[0, 30, 45, 60, 100],
                                       labels=[0, 1, 2, 3]).astype(int)
df_fe["TenurePerAge"]        = df_fe["Tenure"] / (df_fe["Age"] + 1)

print("Engineered features added:")
print("  • BalancePerProduct  — balance normalised by number of products held")
print("  • SalaryToBalance    — estimated salary relative to account balance")
print("  • IsZeroBalance      — binary flag for customers with zero balance")
print("  • AgeGroup           — binned age: 0=<30, 1=30-45, 2=45-60, 3=60+")
print("  • TenurePerAge       — tenure relative to customer age")
print(f"\nFinal feature set shape: {df_fe.shape}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. TRAIN / TEST SPLIT + SCALING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 4: TRAIN / TEST SPLIT & SCALING")
print("="*60)

X = df_fe.drop("Exited", axis=1)
y = df_fe["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train set: {X_train.shape}  |  Test set: {X_test.shape}")
print(f"Train churn rate: {y_train.mean()*100:.2f}%  |  Test churn rate: {y_test.mean()*100:.2f}%")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
print("Scaler saved.")

# ═══════════════════════════════════════════════════════════════════════════
# 5. MODEL TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 5: MODEL TRAINING & EVALUATION")
print("="*60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost":              XGBClassifier(n_estimators=200, learning_rate=0.05,
                                          max_depth=5, random_state=42,
                                          eval_metric="logloss", verbosity=0),
}

results = {}

for name, model in models.items():
    print(f"\n── {name} ──")
    X_tr = X_train_sc if name == "Logistic Regression" else X_train
    X_te = X_test_sc  if name == "Logistic Regression" else X_test

    model.fit(X_tr, y_train)
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    cv_auc = cross_val_score(model, X_tr, y_train, cv=cv,
                              scoring="roc_auc").mean()

    results[name] = dict(accuracy=acc, f1=f1, precision=prec,
                         recall=rec, roc_auc=auc, cv_auc=cv_auc,
                         y_pred=y_pred.tolist(), y_proba=y_proba.tolist())

    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  CV AUC   : {cv_auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

    joblib.dump(model, os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}.pkl"))

# Save results (without lists for JSON)
results_json = {k: {m: v for m, v in vals.items() if not isinstance(v, list)}
                for k, vals in results.items()}
with open(os.path.join(MODELS_DIR, "model_results.json"), "w") as f:
    json.dump(results_json, f, indent=2)

# ═══════════════════════════════════════════════════════════════════════════
# 6. VISUALISATIONS — EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 6: EVALUATION PLOTS")
print("="*60)

# ── Plot 5: Model Comparison Bar Chart ───────────────────────────────────
metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
model_names = list(results.keys())
x = np.arange(len(metrics))
width = 0.25
colors_models = ["#457B9D", "#2A9D8F", "#E9C46A"]

fig, ax = plt.subplots(figsize=(13, 6))
for i, (name, color) in enumerate(zip(model_names, colors_models)):
    vals = [results[name][m] for m in metrics]
    bars = ax.bar(x + i*width, vals, width, label=name, color=color, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax.set_xticks(x + width)
ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
ax.set_ylim(0, 1.12)
ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
ax.set_ylabel("Score")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "05_model_comparison.png"), bbox_inches="tight")
plt.close()
print("Saved: 05_model_comparison.png")

# ── Plot 6: ROC Curves ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
line_styles = ["-", "--", "-."]
for (name, color, ls) in zip(model_names, colors_models, line_styles):
    fpr, tpr, _ = roc_curve(y_test, results[name]["y_proba"])
    auc = results[name]["roc_auc"]
    ax.plot(fpr, tpr, color=color, lw=2, ls=ls, label=f"{name} (AUC={auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "06_roc_curves.png"), bbox_inches="tight")
plt.close()
print("Saved: 06_roc_curves.png")

# ── Plot 7: Confusion Matrices ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")

for ax, name in zip(axes, model_names):
    cm = confusion_matrix(y_test, results[name]["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Retained", "Churned"],
                yticklabels=["Retained", "Churned"])
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "07_confusion_matrices.png"), bbox_inches="tight")
plt.close()
print("Saved: 07_confusion_matrices.png")

# ── Plot 8: Feature Importances (RF & XGB) ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Feature Importances", fontsize=14, fontweight="bold")

for ax, name, color in [
    (axes[0], "Random Forest",  "#2A9D8F"),
    (axes[1], "XGBoost",        "#E9C46A"),
]:
    model = joblib.load(os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}.pkl"))
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=True).tail(15)
    feat_imp.plot(kind="barh", ax=ax, color=color, edgecolor="white")
    ax.set_title(name)
    ax.set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "08_feature_importances.png"), bbox_inches="tight")
plt.close()
print("Saved: 08_feature_importances.png")

# ── Plot 9: Correlation Heatmap ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 10))
corr = df_fe.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, linewidths=0.5, annot_kws={"size": 7})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "09_correlation_heatmap.png"), bbox_inches="tight")
plt.close()
print("Saved: 09_correlation_heatmap.png")

print("\n" + "="*60)
print("  ALL DONE — models saved to /models, plots to /plots")
print("="*60)
