# Important Libs for clean coding
from inspect import signature
from tkinter import NO
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
from prepare_data import SEED

# Configure logging
logger = logging.getLogger("__main__")
tf.random.set_seed(SEED)

class WineQualityModel:
    """Neural network model for wine quality prediction"""

    def __init__(self,input_dim: int):
        self.input_dim = input_dim
        self.model = None
        self.history = None
    
    def build_model(self, learning_rate: float, momentum: float, hidden_layers: Tuple[int,...]= (64,32), dropout_rate: float = 0.2)->keras.Model:
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

    def train(self,X_tarin: np.ndarray, y_train:np.ndarray,
                X_val: np.ndarray, y_val:np.ndarray,
                epochs: int = 50, batch_size: int= 32,
                patience: int = 10) -> Dict[str, Any]:
        early_stopping = keras.callbacks.EarlyStopping(
                    monitor = 'val_loss',
                    patience = patience,
                    restore_best_weights = True,
                    verbose= 0
                )
                
        reduce_lr = keras. callbacks.ReduceLROnPlateau(
                    monitor = 'val_loss',
                    factor = 0.5,
                    patience = 5,
                    min_lr = 1e-7,
                    verbose= 0
                )
        self.history= self.model.fit(
                    X_tarin, y_train,
                    validation_data = (X_val, y_val),
                    epochs= epochs,
                    batch_size = batch_size,
                    callbacks = [early_stopping, reduce_lr],
                    verbose = 0
                )
        val_loss, val_rmse, val_mae = self.model.evaluate(X_val,y_val,verbose = 0)

        return {
                "val_loss": val_loss,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "epochs_trained": len(self.history.history["loss"]),
                "history": self.history
                }

class HyperparameterOptimizer:
    """Handles hyperparameter optimization using Hyperopt"""

    def __init__(self, data: dict[str, np.ndarray], experiment_name: str = "wine-quality-optimization"):
        self.data = data
        self.experiment_name = experiment_name
        self.best_params = None
        self.trials = None

        mlflow.set_experiment(experiment_name)
    
    def create_model_and_train(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create and train model with given hyperparameters"""
        model = WineQualityModel(input_dim=self.data['X_train'].shape(1))
        model.build_model(
            learning_rate = params['learning_rate'],
            momentum = params['momentum'],
            hidden_layers = params.get('hidden_layers',(64,32)),
            dropout_rate = params.get('dropout_rate',0.2) 
        )

        results = model.train(
            self.data['X_train'],self.data['y_train'],
            self.data['X_val'], self.data['y_val'],
            epochs = params.get('epochs',50),
            batch_size = params.get('batch_size', 32),
            patience = params.get('patience',10)
        )
        return {**results, 'model':model}
    
    def objective(self, params: Dict[str, Any]) -> Dict[str, Any]:
        with mlflow.start_run(nested= True):
            mlflow.log_params(params)
            result = self.create_model_and_train(params)
            # Log metrics
            mlflow.log_metrics({
                "val_rmse": result["val_rmse"],
                "val_loss": result["val_loss"],
                "val_mae": result["val_mae"],
                "epochs_trained": result["epochs_trained"]
            })
            # Log model
            signature = infer_signature(self.data['X_train'], result['model'].model.predict(self.data['X_train']))
            mlflow.tensorflow.log_model(
                result['model'].model,
                'model',
                signature= signature
            )
            self._log_training_curves(result['history'])
            return {'loss': result['val_rmse'], 'status': STATUS_OK}
        
    def _log_training_curves(self, history: keras.callbacks.History):
            """Create and log training visualization"""
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
            # Plot loss
            ax1.plot(history.history["loss"], label="Training Loss")
            ax1.plot(history.history["val_loss"], label="Validation Loss")
            ax1.set_title("Model Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()
        
            # Plot RMSE
            ax2.plot(history.history["root_mean_squared_error"], label="Training RMSE")
            ax2.plot(history.history["val_root_mean_squared_error"], label="Validation RMSE")
            ax2.set_title("Model RMSE")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("RMSE")
            ax2.legend()
        
            plt.tight_layout()
            mlflow.log_figure(fig, "training_curves.png")
            plt.close()
        
    def optimize(self, max_evals: int = 15) -> Dict[str,Any]:
            """Run hyperparameter optimization"""
            search_space = {
                'learning_rate': hp.loguniform('learning_rate',np.log(1e-5),np.log(1e-1)),
                'momentum': hp.uniform('momentum', 0.0, 0.9),
                'dropout_rate': hp.uniform('dropout_rate',0.1,0.5),
                'batch_size': hp.choice('batch_size',[32,64,128]),
                'hidden_layers': hp.choice('hidden_layers', [(64,32),(128,64),(64,32,16)])
            }
            logger.info("Starting hyperparameter optimization")
            logger.info(f"Search space: {search_space.keys()}")

            with mlflow.start_run(run_name= "hyperparameter-sweep"):
                mlflow.log_params(
                    {
                        "optimization_method":"TPE",
                        "max_evaluations": max_evals,
                        "objective_metric": "validation_rmse",
                        "dataset": "win-quality",
                        "model_type": "neural_network",
                    }
                )
                # Run optimization
                self.trials = Trials()
                self.best_params = fmin(
                    fn = self.objective,
                    space = search_space,
                    algo = tpe.suggest,
                    max_evals = max_evals,
                    trials= self.trials,
                    verbose = True,
                    rstate= np.random.default_rng(SEED)
                )
                # Log best results
                best_trial = min(self.trials.results, key = lambda x: x['loss'])

                mlflow.log_params(
                    {
                        'best_learning_rate': self.best_params['learning_rate'],
                        'best_momentum': self.best_params['momentum'],
                        'best_dropout_rate': self.best_params['dropout_rate']
                    }
                )
                mlflow.log_metrics(
                    {
                        "best_val_rmse": best_trial['loss'],
                        "total_trials": len(self.trials.trials)

                    }
                )
                logger.info(f"Optimization completed. Best validation RMSE: {best_trial['loss']:.4f}")
                return {
                    "best_params": self.best_params,
                    "best_rmse": best_trial["loss"],
                    "trials": self.trials
                }


