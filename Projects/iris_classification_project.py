'''🌸 Portfolio Project 1: Iris Classification (End-to-End)'''

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

#--- SKLEARN MODULES FOR MACHINE LEARNING ---
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

# =================================================================
# PART 1: DATA LOADING, PREPARATION, AND SPLITTING (Pandas)
# =================================================================

#1. Load Data from Stable Source 
iris_df = sns.load_dataset("iris")
 
#2. Define Features (X) and Target (y)
X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris_df['species']

#3. Split Data into Training and Testing Sets (80/20 split)
# The random_state=42 ensures the result in reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("---Data Prepared---")
print(f"Total Samples: {len(iris_df)}")
print(f"Training Samples: {len(X_train)} | Testing Samples: {len(X_test)}")

# =================================================================
# PART 2: MODEL TRAINING (K-NN Classifier)
# =================================================================

#4. Instantiate the Model (K-Nearest Neighbors)
# We choose K=5 as a well-balanced, odd number to avoid ties and reduce variance.
k_best = 5
model = KNeighborsClassifier(n_neighbors=k_best)

#5. Train the Model (The model learns the function f(Measurement) -> Species)
model.fit(X_train,y_train)

print("\n---Model Trained ---")
print(f"classifier used: K-Nearest Neighbors (k={k_best})")

# =================================================================
# PART 3: ADVANCED EVALUATION AND REPORTING
# =================================================================

#6. Make Predictions on the Test set (Unseen Data)
y_pred = model.predict(X_test)

#7. Calculate and Print Core Meterics
final_accuracy = accuracy_score(y_test, y_pred)

print("\n---Model Performance Report---")
print(f"Overall Accuracy: {final_accuracy*100:.2f}%\n")

#A. Detailed Classification Report (Precision, Recall, F1-Score)
print("CLASSIFICATION REPORT:\n")
print(classification_report(y_test, y_pred))

#B. Confusion Matrix (Visualizing where errors occur)
print("\nCONFUSION MATRIX (Actual vs. Predicted):\n")
print(confusion_matrix(y_test, y_pred))

# =================================================================
# PART 4: PRACTICAL PREDICTION EXAMPLE
# =================================================================

#Predict a new, hypothtical flower (Petal Length 5.0, Petal Width 1.7)
new_flower_data = np.array([[6.0,3.0,5.0,1.7]])
predicted_species = model.predict(new_flower_data)

print("\n---Practical Application ---")
print(f"Prediction for new flower (5.0/1.7cm): {predicted_species[0]}")

