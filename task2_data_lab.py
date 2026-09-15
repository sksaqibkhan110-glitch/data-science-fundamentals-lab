# Task 2 - Data Analysis Lab Practical
# Name: Saqib Khan

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Basic library check
print("Libraries imported successfully!")
print("Pandas:", pd.__version__)
print("Numpy:", np.__version__)

# Creating sample student/employee dataset for analysis
data = {
    'EmpID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'Dept': ['IT', 'HR', 'IT', 'Sales', 'Marketing', 'Sales', 'IT', 'HR', 'Marketing', 'Sales'],
    'Experience': [2, 5, 3, 7, 1, 4, 8, 3, 2, 6],
    'Salary': [45000, 60000, 52000, 75000, 38000, 58000, 95000, 50000, 42000, 68000],
    'Score': [85, 78, 90, 88, 70, 75, 95, 82, 79, 84]
}

df = pd.DataFrame(data)

# Looking at the data
print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Summary Info ---")
df.info()

print("\n--- Basic Stats ---")
print(df.describe())

# Checking department wise average salary
avg_salary = df.groupby('Dept')['Salary'].mean()
print("\n--- Average Salary by Dept ---")
print(avg_salary)

# Simple column based logic
df['Level'] = df['Experience'].apply(lambda x: 'Senior' if x >= 5 else 'Junior')
print("\n--- Updated Dataframe ---")
print(df)

# Visualizing the findings
plt.figure(figsize=(8, 4))

# Bar plot for Dept salary
plt.subplot(1, 2, 1)
avg_salary.plot(kind='bar', color='skyblue')
plt.title('Avg Salary per Dept')
plt.xlabel('Department')
plt.ylabel('Salary')

# Scatter plot: Experience vs Salary
plt.subplot(1, 2, 2)
plt.scatter(df['Experience'], df['Salary'], color='coral')
plt.title('Exp vs Salary')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')

plt.tight_layout()
plt.show()
