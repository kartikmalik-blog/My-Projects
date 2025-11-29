import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split

# --- 1. Load Data (Recreating the stable mock dataset) ---
np.random.seed(42) # Ensure the data is the same as before

# FIX: Define the necessary lists OUTSIDE the dictionary first
date_list = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06', '2024-01-07'] * 100)[:700]
promo_list = np.random.randint(0, 2, 700)

data = {
    'Date': date_list,
    'Promotion': promo_list,
    'StoreType': np.random.choice(['A', 'B', 'C'], 700),
    # CORRECTED: Use 'promo_list' instead of 'data[Promotion]'
    'Sales': np.random.randint(1000, 10000, 700) * (np.random.rand(700) + 1 + (promo_list * 0.5)) 
}

sales_df = pd.DataFrame(data)
sales_df['DayOfWeek'] = sales_df['Date'].dt.dayofweek # 0=Monday, 6=Sunday
sales_df['DayName'] = sales_df['Date'].dt.day_name()

print("--- Data Analyst Preparation Complete ---")

#Task 2: Descriptive Statistics (Answering 'What Happened?')
# 1. Sales Performance by Day of Week
day_performance = sales_df.groupby('DayName')['Sales'].agg(['mean', 'median', 'std']).reset_index()

# Sort days for logical plotting (Monday=0, Sunday=6)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_performance['DayName'] = pd.Categorical(day_performance['DayName'], categories=day_order, ordered=True)
day_performance = day_performance.sort_values('DayName')

print("\n--- Descriptive Statistics: Sales by Day ---")
print(day_performance.to_string(index=False))

# 2. Promotion Impact Summary
promo_summary = sales_df.groupby('Promotion')['Sales'].agg(['mean', 'std']).reset_index()
promo_summary['Promotion'] = promo_summary['Promotion'].replace({0: 'No Promotion', 1: 'With Promotion'})

print("\n--- Descriptive Statistics: Promotion Effectiveness ---")
print(promo_summary.to_string(index=False))

#Task 3: Advanced Visualization (Data Storytelling)
# Set up the visualization style
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 6)) # 1 row, 2 columns for side-by-side plots
plt.tight_layout(pad=4.0)

# --- PLOT A: Average Sales by Day (Answering 'When are we busiest?') ---
sns.barplot(
    x='DayName',
    y='mean',
    data=day_performance,
    palette='viridis',
    ax=axes[0]
)
axes[0].set_title('Average Daily Sales Performance', fontsize=14)
axes[0].set_xlabel('Day of Week')
axes[0].set_ylabel('Mean Sales ($)')
axes[0].tick_params(axis='x', rotation=45)


# --- PLOT B: Sales Distribution (Answering 'Does the Promotion work?') ---
sns.boxplot(
    x='Promotion',
    y='Sales',
    data=sales_df.replace({'Promotion': {0: 'No Promo', 1: 'With Promo'}}),
    palette=['lightcoral', 'skyblue'],
    ax=axes[1]
)
axes[1].set_title('Sales Distribution: Promotion vs. No Promotion', fontsize=14)
axes[1].set_xlabel('Promotion Status')
axes[1].set_ylabel('Sales ($)')

#Save the combined charts for the portfolio
plt.savefig('Sales_Analyst_Insights.png')
plt.show()

print("\n--- Advanced Visualization Complete ---")
print("Saved 'Sales_Analyst_Insights.png' showing key business insights.")