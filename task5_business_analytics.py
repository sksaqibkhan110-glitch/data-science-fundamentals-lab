# Task 5 - Professional Data Analysis & Executive Reporting
# Analyst: Saqib Khan

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Dataset Generation (Synthetic Commercial Sales & Profitability Benchmark)
np.random.seed(101)
n_records = 1000

regions = np.random.choice(['North', 'South', 'East', 'West'], size=n_records, p=[0.30, 0.25, 0.20, 0.25])
categories = np.random.choice(['Technology', 'Furniture', 'Office Supplies'], size=n_records, p=[0.35, 0.30, 0.35])
customer_segments = np.random.choice(['Consumer', 'Corporate', 'Home Office'], size=n_records, p=[0.50, 0.30, 0.20])

units = np.random.randint(1, 10, size=n_records)
unit_price = np.where(categories == 'Technology', np.random.uniform(150, 800, size=n_records),
              np.where(categories == 'Furniture', np.random.uniform(80, 450, size=n_records),
                       np.random.uniform(10, 90, size=n_records)))
discount = np.random.choice([0.0, 0.05, 0.1, 0.15, 0.2, 0.3], size=n_records, p=[0.3, 0.2, 0.2, 0.15, 0.1, 0.05])

revenue = units * unit_price * (1 - discount)
margin_base = np.where(categories == 'Technology', 0.25,
              np.where(categories == 'Furniture', 0.08, 0.18))
margin = margin_base - (discount * 0.9) + np.random.normal(0, 0.02, size=n_records)
profit = revenue * margin

df = pd.DataFrame({
    'Order_ID': [f'ORD-{10000+i}' for i in range(n_records)],
    'Region': regions,
    'Category': categories,
    'Customer_Segment': customer_segments,
    'Units_Sold': units,
    'Unit_Price': np.round(unit_price, 2),
    'Discount_Rate': discount,
    'Gross_Revenue': np.round(revenue, 2),
    'Net_Profit': np.round(profit, 2)
})

# 2. Schema Verification & Sanity Checks
print("--- Data Quality Checks ---")
print(f"Total Rows: {len(df)}, Columns: {df.shape[1]}")
print(f"Null Values Found: {df.isnull().sum().sum()}")
assert df.isnull().sum().sum() == 0, "Missing values detected!"
assert df.shape == (1000, 9), f"Unexpected shape: {df.shape}"
print("Schema validation: PASSED")

# 3. Macro Financial Metrics
total_rev = df['Gross_Revenue'].sum()
total_profit = df['Net_Profit'].sum()
overall_margin = (total_profit / total_rev) * 100
aov = df['Gross_Revenue'].mean()

print("\n--- Executive Summary KPIs ---")
print(f"Total Orders: {len(df)}")
print(f"Total Gross Revenue: ${total_rev:,.2f}")
print(f"Total Net Profit:    ${total_profit:,.2f}")
print(f"Blended Profit Margin: {overall_margin:.2f}%")
print(f"Average Order Value (AOV): ${aov:.2f}")

# 4. Multidimensional Segmentation Analysis
cat_perf = df.groupby('Category').agg(
    Revenue=('Gross_Revenue', 'sum'),
    Profit=('Net_Profit', 'sum'),
    Volume=('Order_ID', 'count')
)
cat_perf['Margin_Pct'] = (cat_perf['Profit'] / cat_perf['Revenue']) * 100
print("\n--- Category Breakdown ---")
print(cat_perf)

discount_perf = df.groupby('Discount_Rate').agg(
    Volume=('Order_ID', 'count'),
    Revenue=('Gross_Revenue', 'sum'),
    Profit=('Net_Profit', 'sum')
)
discount_perf['Margin_Pct'] = (discount_perf['Profit'] / discount_perf['Revenue']) * 100
print("\n--- Discount Impact Breakdown ---")
print(discount_perf)

# 5. Visualization Dashboard Export
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(data=cat_perf.reset_index(), x='Category', y='Revenue', ax=axes[0], palette='Blues_d')
axes[0].set_title('Revenue by Product Category')
axes[0].set_ylabel('Total Revenue ($)')
axes[0].set_xlabel('Product Category')

sns.lineplot(data=discount_perf.reset_index(), x='Discount_Rate', y='Margin_Pct', marker='o', ax=axes[1], color='crimson')
axes[1].axhline(0, color='gray', linestyle='--')
axes[1].set_title('Profit Margin Erosion vs Discount Rate')
axes[1].set_ylabel('Profit Margin (%)')
axes[1].set_xlabel('Discount Rate Applied')

plt.tight_layout()
plt.savefig('task5_business_dashboard.png')
print("\nDashboard chart saved as 'task5_business_dashboard.png'.")
