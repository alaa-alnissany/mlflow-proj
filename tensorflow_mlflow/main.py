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
import tensorflow as tf
from tensorflow import keras

# Additional Deep Learning Libs 
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# MLOPS Libs
import mlflow
from mlflow.models import infer_signature

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore') 


def main():
    pass


if __name__ == "__main__":
    main()