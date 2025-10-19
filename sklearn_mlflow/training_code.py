from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np

params= {
    "n_estimators": 100,
    "max_depth": 6,
    "min_samples_split": 10,
    "min_samples_leaf": 4,
    "bootstrap": True,
    "oob_score": False,
    "random_state": 888,
    }


def prepare_data(data: pd.DataFrame)->tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = data.drop(columns=["date","demand"])
    y = data["demand"]
    return train_test_split(X,y, test_size=0.2,random_state=24)

def train_RFmodel(X_train: pd.DataFrame, y_train: pd.DataFrame, params: dict = params) -> RandomForestRegressor:
    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)
    return rf

def data_val(model: RandomForestRegressor, X_val: pd.DataFrame, y_val: pd.DataFrame) -> dict:
    y_pred = model.predict(X_val)
    # Calculate error metrics
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)
    
    metrics = {
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "r2": r2
                }
    
    return metrics