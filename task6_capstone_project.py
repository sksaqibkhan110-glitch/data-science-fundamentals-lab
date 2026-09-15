# Task 6 - Final Data Science Capstone Project
# Intern: Saqib Khan
# Domain: Data Analyst / Data Science

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# 1. Capstone Data Generation & Feature Engineering
np.random.seed(42)
n_samples = 1500

customer_age = np.random.randint(18, 70, size=n_samples)
tenure_months = np.random.randint(1, 72, size=n_samples)
monthly_charges = np.random.uniform(20.0, 120.0, size=n_samples)
total_charges = tenure_months * monthly_charges + np.random.normal(0, 25, size=n_samples)
total_charges = np.maximum(total_charges, monthly_charges)
support_tickets = np.random.poisson(1.5, size=n_samples)
contract_type = np.random.choice(['Month-to-Month', 'One-Year', 'Two-Year'], size=n_samples, p=[0.55, 0.25, 0.20])
tech_support = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])

# Engineered Feature: Ticket frequency relative to tenure
avg_monthly_support = support_tickets / (tenure_months + 1)

contract_factor = np.where(contract_type == 'Month-to-Month', 0.8, np.where(contract_type == 'One-Year', -0.3, -1.0))
logit = (
    -1.2
    - 0.03 * tenure_months
    + 0.02 * (monthly_charges - 60)
    + 0.45 * support_tickets
    - 0.50 * tech_support
    + contract_factor
    + np.random.normal(0, 0.4, size=n_samples)
)
prob = 1 / (1 + np.exp(-logit))
churn = (prob > 0.50).astype(int)

df = pd.DataFrame({
    'Customer_ID': [f'CUST-{1000+i}' for i in range(n_samples)],
    'Age': customer_age,
    'Tenure_Months': tenure_months,
    'Monthly_Charges': np.round(monthly_charges, 2),
    'Total_Charges': np.round(total_charges, 2),
    'Contract_Type': contract_type,
    'Tech_Support': tech_support,
    'Support_Tickets': support_tickets,
    'Avg_Monthly_Support': np.round(avg_monthly_support, 3),
    'Churn': churn
})

# 2. Data Quality Assurance & Schema Validation
print("--- Data Quality Checks ---")
print(f"Total Records: {len(df)}, Features: {df.shape[1]}")
print(f"Missing Values: {df.isnull().sum().sum()}")
assert df.isnull().sum().sum() == 0, "Missing values detected!"
assert df.shape == (1500, 10), f"Unexpected shape: {df.shape}"
print("Schema Validation: PASSED")

# 3. Model Preparation & Stratified Splitting
df_encoded = pd.get_dummies(df.drop(columns=['Customer_ID']), columns=['Contract_Type'], drop_first=True)

X = df_encoded.drop(columns=['Churn'])
y = df_encoded['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Model Training (Random Forest Classifier)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf_model.fit(X_train, y_train)

# Default 0.50 Decision Threshold
y_prob = rf_model.predict_proba(X_test)[:, 1]
y_pred_50 = (y_prob >= 0.50).astype(int)

acc = accuracy_score(y_test, y_pred_50)
prec = precision_score(y_test, y_pred_50)
rec = recall_score(y_test, y_pred_50)
f1 = f1_score(y_test, y_pred_50)
auc = roc_auc_score(y_test, y_prob)
cm_50 = confusion_matrix(y_test, y_pred_50)

print("\n--- Capstone Model Performance (Threshold: 0.50) ---")
print(f"Accuracy:  {acc * 100:.2f}%")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {auc:.4f}")
print("Confusion Matrix:\n", cm_50)

# Alternative 0.35 Decision Threshold
y_pred_35 = (y_prob >= 0.35).astype(int)
print("\n--- Model Performance (Threshold: 0.35) ---")
print(f"Precision: {precision_score(y_test, y_pred_35):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_35):.4f}")

# Feature Importance Ranking
feat_imp = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n--- Feature Importances ---")
print(feat_imp)

# 5. Capstone Dashboard Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm_50, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Retained', 'Churned'], yticklabels=['Retained', 'Churned'])
axes[0].set_title('Test Set Confusion Matrix (Threshold = 0.50)')
axes[0].set_xlabel('Predicted Status')
axes[0].set_ylabel('Actual Status')

feat_imp.head(6).plot(kind='barh', ax=axes[1]).invert_yaxis()
axes[1].set_title('Top 6 Feature Importances (Random Forest)')
axes[1].set_xlabel('Relative Importance Score')

plt.tight_layout()
plt.savefig('task6_capstone_dashboard.png')
print("\nSaved capstone dashboard to 'task6_capstone_dashboard.png'.")
