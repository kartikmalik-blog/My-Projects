'''🛍️ Project 3: Advanced Sales Prediction (Random Forest Regression)
We will use a stable version of a mock retail dataset to predict sales based on factors like time, promotions, and store type.

The Challenge: Non-Linearity and Categorical Data
House prices often have a simple linear relationship, but sales are complex: a holiday promotion on a Monday behaves differently than a regular Tuesday. We need a model that can capture these non-linear relationships—the Random Forest Regressor.
'''

#Task1: Data Preparation and Feature Engineering
'''Before we train  model, we must prepare the complex date and categorical data.'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # THE ADVANCED MODEL
from sklearn.metrics import mean_squared_error, r2_score

#--- 1. LOAT DATA (Mock Sales Data) ---
#We'll create a stable mock dataset for this complex project 
data = {
    'Date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06', '2024-01-07'] * 100),
    'Promotion': np.random.randint(0,2,700), # 0 or 1
    'StoreType': np.random.choice(['A','B','C'], 700),
    'Sales': np.random.randint(1000, 10000, 700) * (np.random.rand(700) + 1)
}
sales_df = pd.DataFrame(data)

# --- 2. FEATURE ENGINEERING (Time-Series Data) ---
# Extracting featires from the 'Date' coulmn is crucial for time-series analysis
sales_df['DayOfWeek'] = sales_df['Date'].dt.dayofweek
sales_df['Month'] = sales_df['Date'].dt.month

#---3. HANDLING CATEGORICAL DATA (One-Hot Encoding) ---
# Models can only understand numbers, so we convert 'StoreType' into numerical columns
sales_df = pd.get_dummies(sales_df, columns=['StoreType'], drop_first=True)

#---4. Define X and y ---
#Exclude the original 'Date' column and 'Sales' (our target)
features = ['Promotion', 'DayOfWeek', 'Month', 'StoreType_B', 'StoreType_C']
X = sales_df[features]
y = sales_df['Sales']

#Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("--- TASK 1: Advanced Data Preparation Complete ----")
print(f"Final Features (X columns): {X_train.columns.tolist()}")


#Task 2: Train the Random Foerst Regressor
''''We're moving from the simple Linear Regression (a straight line) to a Random Forest (which is a collection of decision trees). This model is excellent at finding complex, non-linear patterns.'''

#---5. Instantiate the Random Forest Model ---
#We use n_estimators=100 (100 decision trees) for robust prediction 
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

#--- 6. Train the Model ---
rf_model.fit(X_train, y_train)

#---7. Predict and Evaluate ---
y_pred = rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\n--- TASK 2: Random Forest Model Tained and Evaluated ---")
print(f"Model Used: Random Forest Regressor")
print(f"R-squared (R2) Score: {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")

#--- 8. Feature Importance (The Business Insigth) ---
# Random Forest allows us to see which features were most influential 
feature_importances = pd.Series(
    rf_model.feature_importances_,
    index=X_train.columns   
).sort_values(ascending=False)

print("\n--- TASK 3: Feature Importance (Driving Factors)---")
print(feature_importances)