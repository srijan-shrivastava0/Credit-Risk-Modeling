
import pandas as pd
import numpy as np
from src.data_preprocessing import preprocess_data
from src.model_training import train_logistic, train_lightgbm
from src.evaluation import evaluate_model

# Load Data
df = pd.read_csv("data/synthetic_credit_data.csv")

# Preprocessing
X_train, X_test, y_train, y_test = preprocess_data(df)

# Train models
logistic_model = train_logistic(X_train, y_train)
lgbm_model = train_lightgbm(X_train, y_train)

# Evaluate
print("\nLogistic Regression Performance:")
evaluate_model(logistic_model, X_test, y_test)

print("\nLightGBM Performance:")
evaluate_model(lgbm_model, X_test, y_test)
