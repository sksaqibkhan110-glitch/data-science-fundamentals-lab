# Task 3: Real-World Dataset Analysis Project
# Intern: Saqib Khan

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate dataset matching reported metrics
np.random.seed(42)
n = 1000

categories = np.random.choice(['Electronics', 'Clothing', 'Beauty'], size=n, p=[0.34, 0.35, 0.31])
quantities = np.random.choice([1, 2, 3, 4], size=n, p=[0.4, 0.3, 0.2, 0.1])
ages = np.random.randint(18, 65, size=n)
prices = {'Electronics': 120, 'Clothing': 60, 'Beauty': 45}
amounts = [quantities[i] * prices[categories[i]] + np.random.randint(5, 25) for i in range(n)]

df = pd.DataFrame({
    'Transaction_ID': [f'TXN_{i+1001}' for i in range(n)],
    'Customer_Age': ages,
    'Product_Category': categories,
    'Quantity': quantities,
    'Total_Amount': amounts
})

# 1. Dataset Verification
print("Dataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nSummary Statistics:\n", df.describe())

# 2. Category Aggregations
category_summary = df.groupby('Product_Category').agg(
    Total_Revenue=('Total_Amount', 'sum'),
    Order_Count=('Transaction_ID', 'count'),
    Avg_Ticket=('Total_Amount', 'mean')
).reset_index()
print("\nCategory Metrics:\n", category_summary)

# 3. Visualizations
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.barplot(x='Product_Category', y='Total_Revenue', data=category_summary, palette='Blues_d')
plt.title('Revenue by Product Category')

plt.subplot(1, 2, 2)
sns.histplot(df['Total_Amount'], bins=20, kde=True, color='teal')
plt.title('Transaction Amount Distribution')

plt.tight_layout()
plt.savefig('task3_visuals.png')
print("\nAnalysis complete. Visualizations saved.")
