# Important Libs for clean coding
from typing import Dict, Tuple, Any, Optional
import logging
import warnings

# Basic libs
from mlflow.entities import metric
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# Machine learning Libs
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Deep Learning Libs
import tensorflow as tf
from tensorflow import keras

# Additional Deep Learning Libs 
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# MLOPS Libs
import mlflow
from mlflow.models import infer_signature

# Configure logging
logger = logging.getLogger("__main__")

class WineQualityModel:
    """Neural network model for wine quality prediction"""

    def __init__(self,input_dim: int):
        self.input_dim = input_dim
        self.model = None
        self.history = None
    
    def build_model(self, learning_rate: float, momentum: float, hidden_layers: Tuple[int,...]= (64,32), dropout_rate: 0.2)->keras.Model:
        """Build and compile the neural network model"""

        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(self.input_dim,)))

        # Add hidden layers
        for units in hidden_layers:
            model.add(keras.layers.Dense(units, activation= "relu"))
            if dropout_rate > 0:
                model.add(keras.layers.Dropout(dropout_rate))

        # Output layers
        model.add(keras.layers.Dense(1))

        # Compile model
        optimizer = keras.optimizers.adam(learning_rate = learning_rate, momentum = momentum)
        model.compile(
            optimizer = optimizer,
            loss = "mse",
            metrics = [keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError()]
        )
        self.model = model
        return model

        def train(self):
            pass