
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    df['credit_utilization'] = df['current_balance'] / (df['credit_limit'] + 1)
    df['credit_history_length'] = 2025 - df['credit_history_start']

    X = df[['age', 'income', 'loan_amount', 'credit_utilization', 'credit_history_length', 'num_credit_lines']]
    y = df['defaulted']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.3, random_state=42)
