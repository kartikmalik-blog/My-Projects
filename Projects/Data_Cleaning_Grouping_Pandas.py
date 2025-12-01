import pandas as pd
import numpy as np

# 1. Create sample data with missing values (NaN)
data = {
    'Category': ['A', 'B', 'A', 'C', 'B', 'A'],
    'Value': [100, 200, 150, np.nan, 250, 120],  # NaN represents missing data
    'Region': ['North', 'South', 'North', 'East', 'South', 'East']
}
df = pd.DataFrame(data)
print("Original Data:\n", df)

# 2. Data Cleaning: Fill missing 'Value' with the median 
median_value = df['Value'].median()
df['Value'].fillna(median_value, inplace=True)
print("\nData after Cleaning (NaN filled with Median):\n", df)

#3. Analytical Grouping: Find the sum of values for each 'Region' 
region_summary = df.groupby('Region')['Value'].sum().reset_index()
print("\nAnalytical Summary (Sum of Value per Region):\n", region_summary)