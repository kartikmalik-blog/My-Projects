'''📊 Phase 2: Visualization and Machine Learning Project
4. Matplotlib Challenge: Visualizing Distributions
Now we use the cleaned Pandas data to create a meaningful visualization.

Goal: Create a visualization that shows the distribution of values for each category.

Skill Focus: Using matplotlib.pyplot to create side-by-side plots (subplots).
'''
import matplotlib.pyplot as plt
import seaborn as sns #Used for easier plotting,common in Data Science

#Reusing the Cleaned 'df' from the Pandas 

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

#Create a figure with a single plot
plt.figure(figsize=(8, 5))

#Create a box plot to show the distribution of 'Value' across different 'Categories'
sns.boxplot(x='Category', y='Value', data=df)

plt.title('Distribution of Value across Categories')
plt.xlabel('Category Type')
plt.ylabel('Value (USD)')
plt.grid(axis='y')
plt.show()