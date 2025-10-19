# Important Libs for clean coding
from typing import Dict, Tuple, Any, Optional
import logging
import warnings

# Basic libs
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# Machine learning Libs
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Deep Learning Libs
from sqlalchemy import true
import tensorflow as tf
from tensorflow import keras

# Additional Deep Learning Libs 
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# MLOPS Libs
import mlflow
from mlflow.models import infer_signature

logger = logging.getLogger("__main__")

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)

class DataProcessor:
    """ Handles data loading, preprocessing, and splitting"""
    
    def __init__(self, separator: str = ";", test_size: float = 0.25, val_size: float = 0.2, random_state: int = SEED):
        self.test_size = test_size
        self.val_size = val_size
        self.separator = separator
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.is_fitted = False
        # .....
    
    def load_data(self, url: str) -> pd.DataFrame:
        """Load dataset from URL"""

        logger.info(f"Loading data from {url}")
        try:
            data = pd.read_csv(url, sep= self.separator)
            logger.info(f"Loaded dataset with shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def prepare_features_target(self, data: pd.DataFrame, target_col: str = "quality")-> Tuple[pd.DataFrame,pd.Series]:
        """Separate features and target"""
        
        X = data.drop(columns = target_col)
        y = data[target_col]
        return X, y

    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, np.ndarray]:
        """Create train/validation/test splits with proper scaling"""
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.test_size, random_state= self.random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = self.val_size, random_state= self .random_state)

        # Scale features
        if not self.is_fitted:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.fit_transform(X_val)
            X_test_scaled = self.scaler.fit_transform(X_test)
            self.is_fitted= True

        logger.info(f"Data splits - Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")

        return {
            'X_train': X_train_scaled, 'y_train': y_train.values,
            'X_val': X_val_scaled, 'y_val': y_val.values,
            'X_test': X_test_scaled, 'y_test': y_test.values,
        }

