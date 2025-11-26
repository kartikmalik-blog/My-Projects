import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. DATA LOADING (Guaranteed Stable Source) ---
# Load the Iris dataset directly from the seaborn library (bypasses broken URLs).
iris_df = sns.load_dataset('iris')

print("--- 1. DATA LOADING COMPLETE ---")
print(iris_df.head()) 

# =================================================================
# TASK 2: DATA CLEANING, FILTERING, AND WRANGLING
# =================================================================

# 1. Cleaning: Drop any rows with missing data (NaN values)
clean_df = iris_df.dropna() 

# 2. Filtering: Create a subset of the data for advanced analysis
# We filter out the smallest flowers (Petal Length <= 1.5 cm) to focus on the two larger species.
filtered_df = clean_df[clean_df['petal_length'] > 1.5].copy()

print("\n--- 2. FILTERING AND CLEANING COMPLETE ---")
print(f"Original Data Rows: {len(clean_df)}")
print(f"Filtered Data Rows: {len(filtered_df)} (Focused on larger flowers)")

# =================================================================
# TASK 3: SUMMARY STATISTICS (Pandas Reporting)
# =================================================================

# Calculate descriptive statistics for the key feature (Petal Length)
summary_statistics = filtered_df['petal_length'].describe()

print("\n--- 3. SUMMARY STATISTICS (Petal Length) ---")
print(summary_statistics)

# =================================================================
# TASK 4: PROFESSIONAL VISUALIZATION (Segmentation)
# =================================================================

plt.figure(figsize=(10, 6)) # Set a professional size for the chart

# Use Seaborn's histplot to create a segmented visualization:
sns.histplot(
    data=filtered_df,     # Use the filtered data
    x='petal_length',     # X-axis shows the measurement
    hue='species',        # Segment the bars by flower species (color coding)
    multiple='stack',     # Stack the segments to show total count
    bins=15               # Use 15 bins for good detail
)

# Add clear, professional labels and title
plt.title('Distribution of Petal Length by Flower Species (Length > 1.5 cm)', fontsize=14)
plt.xlabel('Petal Length (cm)', fontsize=12)
plt.ylabel('Count of Flowers (Frequency)', fontsize=12)

plt.legend(title='Species') # The legend is crucial for interpretation
plt.show()