# ===== dataset.py =====
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.iloc[:, :4].values
    y = df.iloc[:, 4].values.reshape(-1, 1)

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X = X_scaler.fit_transform(X)
    y = y_scaler.fit_transform(y)

    return X, y, X_scaler, y_scaler