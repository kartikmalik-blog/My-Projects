'''Goal: Perform an end-to-end Linear Regression on a dataset to predict a continuous variable.

Skill Focus: Data loading, train-test split, model training (LinearRegression), and evaluation (R2 Score).
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

#1. Load Data
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

#2. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

#3. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

#4. Predict and Evaluate
y_pred = model.predict(X_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f"Final Model R-squared Score: {r2:.4f}")