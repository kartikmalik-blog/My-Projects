import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score, classification_report, confusion_matrix
import sqlite3

# =================================================================
# PART 1: DATA SIMULATION (SQL Access & Integration)
# =================================================================

# We simulate complex data tables from Astrophysics and Chemistry
np.random.seed(42)
N_PLANETS = 1000

data = {
    # 1. Astrophysics/Physics Features
    'Planet_ID': np.arange(N_PLANETS),
    'Planet_Mass': np.random.uniform(0.1, 10.0, N_PLANETS),
    'Orbital_Period': np.random.uniform(50, 400, N_PLANETS),
    
    # 2. Chemistry/Atmosphere Features (Biomarkers)
    'H2O_Signature': np.random.uniform(0.01, 0.99, N_PLANETS),
    'O2_Signature': np.random.uniform(0.01, 0.99, N_PLANETS),
    'CH4_Signature': np.random.uniform(0.01, 0.99, N_PLANETS),
}

df_planet = pd.DataFrame(data)

# --- Define the Complex Target Variables (y) ---

# Target 1 (Regression): Numerical Habitability Index (0.0 to 1.0)
# Formula: Weighted sum of biomarkers, plus a factor for mass
df_planet['Habitability_Index'] = (
    0.5 * df_planet['H2O_Signature'] + 
    0.3 * df_planet['O2_Signature'] + 
    0.1 * df_planet['Planet_Mass'] + 
    np.random.normal(0, 0.05, N_PLANETS)  # Add small noise
).clip(0.0, 1.0) # Ensure score stays between 0 and 1

# Target 2 (Classification): Life Type
def classify_life(score):
    if score > 0.8:
        return 'Complex'
    elif score > 0.5:
        return 'Microbial'
    else:
        return 'None'
df_planet['Life_Type'] = df_planet['Habitability_Index'].apply(classify_life)


# --- SQL Integration Simulation ---
conn = sqlite3.connect(':memory:')
df_planet.to_sql('Exoplanet_Survey', conn, if_exists='replace', index=False)

# Write the final SQL query to extract the training data
sql_query = """
SELECT Planet_Mass, Orbital_Period, H2O_Signature, O2_Signature, CH4_Signature, Habitability_Index, Life_Type
FROM Exoplanet_Survey
WHERE Planet_Mass < 5.0 -- Example filter (e.g., only terrestrial planets)
"""
final_data = pd.read_sql_query(sql_query, conn)
conn.close()

print("--- Data Integration (SQL) Complete ---")
print(f"Data set size after SQL filtering: {len(final_data)} planets")

# =================================================================
# PART 2: MODELING (Regression & Classification)
# =================================================================

# Define X and the two Y targets
features = ['Planet_Mass', 'Orbital_Period', 'H2O_Signature', 'O2_Signature', 'CH4_Signature']
X = final_data[features]
y_reg = final_data['Habitability_Index'] # Regression Target
y_cls = final_data['Life_Type']          # Classification Target

# Split data for both models (using a larger test size to show clearer differences)
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_reg, test_size=0.3, random_state=42
)
_, _, y_cls_train, y_cls_test = train_test_split(
    X, y_cls, test_size=0.3, random_state=42, stratify=y_cls # Stratify for balanced classes
)


# --- MODEL 1: RANDOM FOREST REGRESSOR (Habitability Score) ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_reg_train)
y_reg_pred = rf_model.predict(X_test)

r2 = r2_score(y_reg_test, y_reg_pred)
mse = np.mean((y_reg_test - y_reg_pred)**2)

print("\n--- Model 1: Random Forest Regression (Habitability) ---")
print(f"R-squared (R2) Score: {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")


# --- MODEL 2: K-NN CLASSIFIER (Life Type) ---
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_cls_train)
y_cls_pred = knn_model.predict(X_test)

print("\n--- Model 2: K-NN Classification (Life Type) ---")
print("Classification Report (Microbial, Complex, None):")
print(classification_report(y_cls_test, y_cls_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_cls_test, y_cls_pred))


# =================================================================
# PART 3: MISSION INSIGHTS & EXCEL REPORTING
# =================================================================

# Extract Feature Importance from the Random Forest Model
feature_importances = pd.Series(
    rf_model.feature_importances_, 
    index=X_train.columns
).sort_values(ascending=False)


# --- EXCEL REPORT GENERATION ---
# Create a DataFrame for the final business/mission report
report_summary = pd.DataFrame({
    'Metric': [
        'MODEL PERFORMANCE R2 (Habitability)', 
        'Dominant Feature 1',
        'Dominant Feature 2',
        'Model Accuracy (Classification)'
    ],
    'Value': [
        r2, 
        feature_importances.index[0], 
        feature_importances.index[1], 
        f'{np.mean(y_cls_test == y_cls_pred):.4f}'
    ]
})

report_summary['Value'] = report_summary.apply(
    lambda x: f'{x["Value"]:.4f}' if isinstance(x["Value"], (int, float)) else x["Value"], axis=1
)

# Export the summary to a CSV file (ready for Excel/Presentation)
report_summary.to_csv('Exoplanet_Mission_Report.csv', index=False)

print("\n--- Mission Insights & Excel Report ---")
print("Feature Importances (for Robotics Prioritization):")
print(feature_importances)
print("\n'Exoplanet_Mission_Report.csv' created for funding agencies.")