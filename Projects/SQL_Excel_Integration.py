'''💻 Project 2: SQL and Excel Integration
For this project, we will use the sqlite3 library (built into Python) to simulate a corporate database environment. This is the professional way to show you can handle data access.'''

#Task 1: SQL Data Access (Simulation)
'''Instead of loading the data directly with Scikit-learn, we'll first load it into a virtual table and then use a SQL Query to extract the data—just like you would in a real company.'''

import pandas as pd
import sqlite3 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing

#--- 1. SET UP THE VIRTUAL DATABASE (SQL) ---
housing = fetch_california_housing(as_frame=True)
housing_df = housing.frame

#Connect to an in-memory database (only exists for this script)
conn = sqlite3.connect(':memory:')

#Load the DataFrame into an SQL table named 'housing_data'
housing_df.to_sql('housing_data', conn, if_exists= 'replace', index=False)

#---2. WRITE THE SQL QUERY (DATA Access) ---
#We SELECT the relevant features and the target,  fitlered by low Average Rooms
sql_query = """
SELECT MedInc, AveRooms, Latitude, MedHouseVal
FROM housing_data
WHERE AveRooms < 8
"""

#Load the SQL result directly into a new Pandas DataFrame 
#THIS IS THE KEY STEP SHOWING SQL SKILL
project_data = pd.read_sql_query(sql_query, conn)

print("--- TASK 1: SQL Data Access Complete ---")
print(f"Data accessed via SQL. Filtered down to {len(project_data)} rows.")

#Close the database connection
conn.close()

#Define X and y from the SQL result
X = project_data[['MedInc', 'AveRooms', 'Latitude']]
y = project_data['MedHouseVal']

#Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


#Task 2: ML Analysis and Business Reporting (Python + Excel)

'''This part runs your improved regression model and generates the final output in an Excel-ready format.'''

#---3. MODEL TRAINING ----
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X_test)
r2 = r2_score(y_test, y_pred)


print("\n--- TASK 2: ML Analysis Complete---")

#---4. EXCEL REPORT GENERATION ---
#Create a DataFrame to hold the key findings for easy export/sharing
report_summary = pd.DataFrame({
    'Metric': ['Model R2 Score (Goodness of Fit)', 'Prediction Coefficient (MedInc)','Prediction Coefficient (AveRooms)', 'Prediction Coefficient (Latitude)'],

    'Value': [r2,
              reg_model.coef_[0],
              reg_model.coef_[1],
              reg_model.coef_[2]]
})

#Format the values for clean presentation (Excel-ready)
report_summary['Value'] = report_summary['Value'].map('{:.4f}'.format)

#Export the summary to a CSV file (which opens easily in Excel/Sheets)
report_summary.to_csv('Regresssion_Business_Summary.csv', index=False)

print("\n--- TASK 3: BUSINESS REPORTING COMPLETE ---")
print("Report Summary (Ready for Excel/Presentation):")

print("\nSuccessfully exported 'Regression_Business_Summary.csv'to you project folder")