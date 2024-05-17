# modules/data_prep.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    return pd.read_csv(file_path)

def handle_missing_values(data):
    # Implement your missing value handling logic here
    return data

def encode_categorical_variables(data, categorical_cols):
    # Implement your categorical variable encoding logic here
    return data

def scale_features(data, numeric_cols):
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data

file_path = "data/telecom_churn.csv"