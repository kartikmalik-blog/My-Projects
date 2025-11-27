'''🏠 Project 2: 
House Price Regression (The Business Project)
This project focuses on Regression (predicting a number) and introduces the key concept of Feature Selection (choosing the right inputs).
'''

'''Goal:-
We will improve your previous low-$R^2$ model by adding more relevant features to predict house prices, which will significantly increase the model's accuracy and make the project more impressive.
'''

'''The most critical part will be the new $R^2$ score. It should be drastically higher, proving that your feature selection improved the model!'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# =================================================================
# PART 1: DATA LOADING, PREPARATION, AND SPLITTING
# =================================================================

#Load the stable California Housing Data 
housing = fetch_california_housing(as_frame=True)
housing_df = housing.frame

#1. Define Features (X) and Target (y) - NOW WITH MORE FEATURES !
#We use Median Income, Average Rooms, and Latitude 
X = housing_df[['MedInc', 'AveRooms', 'Latitude']]
y = housing_df['MedHouseVal']

#2. Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("---Data Prepared for Multiple Regression ---")
print(f"Number of Features (X columns): {X_train.shape[1]}")

# =================================================================
# PART 2: MODEL TRAINING AND PREDICTION
# =================================================================

#3. Instantiate the Multiple Linear Regression Model
#(It uses the same LinearRegression() class, but with multiple inputs)
reg_model = LinearRegression()

#4. Train the model
reg_model.fit(X_train, y_train)

#5. Predict the house price on the unseen test data
y_pred = reg_model.predict(X_test)


# =================================================================
# PART 3: ADVANCED EVALUATION AND REPORTING
# =================================================================

#6. Calculate the evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Regressin Performance Report ---")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R2) Score: {r2:.4f}")

#7. Print the Coefficients (Model Interpretation)
print("\n---Model Interpretation (Coefficients) ---")
coefficients = pd.Series(reg_model.coef_, index=X_train.columns)
print(coefficients)