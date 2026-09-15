# Task 4 - Data Science Tool Mastery Project
# Intern: Saqib Khan

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Dataset Generation (Synthetic Benchmark with Feature Dependencies)
np.random.seed(42)
n_samples = 500

tenure = np.random.randint(1, 60, size=n_samples)
usage = np.random.normal(45.2, 12.0, size=n_samples).clip(10, 80)
tickets = np.random.poisson(1.8, size=n_samples)
monthly_bill = np.random.normal(72.4, 25.0, size=n_samples).clip(20, 140)

# Churn logic: High tickets & high bill increase risk, tenure & usage decrease it
log_odds = (
    -0.5
    - 0.05 * tenure
    - 0.04 * (usage - 45.2)
    + 0.75 * tickets
    + 0.02 * (monthly_bill - 72.4)
)
prob = 1 / (1 + np.exp(-log_odds))
churn = (prob > 0.48).astype(int)

df = pd.DataFrame({
    'Tenancy_Months': tenure,
    'Usage_Hours': np.round(usage, 1),
    'Support_Tickets': tickets,
    'Monthly_Bill': np.round(monthly_bill, 2),
    'Churn_Flag': churn
})

# 2. Data Quality & Schema Guardrails
print("--- Data Quality Check ---")
print(df.isnull().sum())
assert df.isnull().sum().sum() == 0, "Missing values detected!"
assert df.shape == (500, 5), f"Unexpected shape: {df.shape}"
print("Schema validation: PASSED")

print("\n--- Dataset Summary ---")
print(f"Total Observations: {len(df)}")
print(f"Retained (0): {sum(df['Churn_Flag'] == 0)}, Churned (1): {sum(df['Churn_Flag'] == 1)}")
print(f"Mean Usage Hours:  {df['Usage_Hours'].mean():.1f}")
print(f"Mean Monthly Bill: ${df['Monthly_Bill'].mean():.2f}")

# 3. Partitioning & Preprocessing Pipeline
X = df.drop('Churn_Flag', axis=1)
y = df['Churn_Flag']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Model Training & Genuine Evaluation
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n--- Model Evaluation Metrics ---")
print(f"Accuracy:  {acc * 100:.1f}%")
print(f"Precision: {prec:.2f}")
print(f"Recall:    {rec:.2f}")
print(f"F1-Score:  {f1:.2f}")
print(f"ROC-AUC:   {auc:.2f}")
print("Confusion Matrix:\n", cm)

# Inspect Learned Coefficients
coeff_summary = pd.DataFrame({
    'Feature': X.columns,
    'Weight': model.coef_[0]
}).sort_values(by='Weight', ascending=False)
print("\n--- Learned Feature Weights ---")
print(coeff_summary)

# 5. Confusion Matrix Heatmap
plt.figure(figsize=(6, 4.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Stayed', 'Churned'], 
            yticklabels=['Stayed', 'Churned'])
plt.title('Task 4: Evaluated Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.tight_layout()
plt.savefig('task4_confusion_matrix.png')
print("\nSaved evaluation plot to 'task4_confusion_matrix.png'.")
