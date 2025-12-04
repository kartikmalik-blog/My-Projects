import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(data_path = "Data/raw/telco_churn.csv"):
    df = pd.read_csv(data_path)

    # TotalCharges has spaces -> fix
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    #New features
    df['TenureGroup'] = pd.cut(df['tenure'], 
    bins=[0,12,24,48,60,100],
    labels=['0-12','12-24','48-60', '60+'])

    df['IsContractM2M'] = (df['Contract'] == 'Month-to-month').astype(int)
    df['AvgMonthly'] = df['TotalCharges'] / (df['tenure'] + 1) #+1 to avoid divide by 0


    #Binary encoding
    binary_cols = ['Partner', 'Dependents','PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No':0})

    df['gender'] = df['gender'].map({'Female':1, 'Male':0})

    # One-hot encoding
    categorical_cols = ['InternetService', 'Contract', 'PaymentMethod',
    'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'TenureGroup']

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    #Target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    #Drop CustomerID
    df.drop(['customerID'], axis=1, inplace=True)

    return df

# Save processed data (optional)
if __name__ == "__main__":
    df = load_and_preprocess()
    df.to_csv("data/processed/processed_Data.csv", index=False)
    print("Preprocessing done! Shape:", df.shape)