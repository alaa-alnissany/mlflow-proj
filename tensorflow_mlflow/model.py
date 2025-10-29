# Important Libs for clean coding
from typing import Dict, Tuple, Any
import logging

# Basic libs
import numpy as np
import matplotlib.pyplot as plt

# Deep Learning Libs
import tensorflow as tf
from tensorflow import keras

# Additional Deep Learning Libs for hyperparameter fine-tuning
import optuna
from optuna import Trial

# MLOPS Libs
import mlflow
from mlflow.models import infer_signature
from prepare_data import SEED

# Configure logging
logger = logging.getLogger("__main__")
tf.random.set_seed(SEED)


class Model:
    def __init__(self,input_dim: int):
        self.input_dim = input_dim
        self.model = None
        self.history = None
    def build(self):
        pass
    def train(self):
        pass
    def test(self):
        pass


class WineQualityModel(Model):
    """Neural network model for wine quality prediction"""

    def __init__(self,input_dim: int):
        self.input_dim = input_dim
        self.model = None
        self.history = None
    
    def build(self, learning_rate: float, momentum: float, hidden_layers: Tuple[int,...]= (64,32), dropout_rate: float = 0.2):
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
        optimizer = keras.optimizers.Adam(learning_rate = learning_rate, ema_momentum = momentum)
        model.compile(
            optimizer = optimizer,
            loss = "mse",
            metrics = [keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError()]
        )
        self.model = model

    def train(self, X_tarin: np.ndarray, y_train:np.ndarray,
                X_val: np.ndarray, y_val:np.ndarray,
                epochs: int = 50, batch_size: int= 32,
                patience: int = 10) -> Dict[str, Any]:
        early_stopping = keras.callbacks.EarlyStopping(
                    monitor = 'val_loss',
                    patience = patience,
                    restore_best_weights = True,
                    verbose= 0
                )
                
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
                    monitor = 'val_loss',
                    factor = 0.5,
                    patience = 5,
                    min_lr = 1e-7,
                    verbose= 0
                )
        self.history = self.model.fit(
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
        self.study = None
        self.best_trial = None

        mlflow.set_experiment(experiment_name)
    
    def create_model_and_train(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create and train model with given hyperparameters"""
        model = WineQualityModel(input_dim= self.data['X_train'].shape[1])
        model.build(
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
    
    def objective(self, trial: Trial) -> float:
        """Optuna objective function"""
        # Define hyperparameter search space
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
            'momentum': trial.suggest_float('momentum', 0.0, 0.9),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'hidden_layers': trial.suggest_categorical('hidden_layers', [(64, 32), (128, 64), (64, 32, 16)]),
            'epochs': 50,
            'patience': 10 
        }
        
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
            # Store trial number in user attributes for reference
            trial.set_user_attr("mlflow_run_id", mlflow.active_run().info.run_id)
            return result['val_rmse']
        
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
        
    def optimize(self, n_trials: int = 15) -> Dict[str,Any]:
            """Run hyperparameter optimization"""
            logger.info("Starting hyperparameter optimization")

            with mlflow.start_run(run_name= "hyperparameter-sweep"):
                mlflow.log_params(
                    {
                        "optimization_method":"TPE",
                        "number_trials": n_trials,
                        "objective_metric": "validation_rmse",
                        "dataset": "wine-quality",
                        "model_type": "neural_network",
                    }
                )
                # Create Optuna study
                self.study = optuna.create_study(direction='minimize',
                                                sampler= optuna.samplers.TPESampler(seed= SEED))
                # Run optimization
                self.study.optimize(self.objective, n_trials= n_trials)
                
                # Get best results
                self.best_trial = self.study.best_trial
                self.best_params = self.study.best_params
                
                # Log best results
                mlflow.log_params(
                    {
                        'best_learning_rate': self.best_params['learning_rate'],
                        'best_momentum': self.best_params['momentum'],
                        'best_dropout_rate': self.best_params['dropout_rate'],
                        'best_batch_size': self.best_params['batch_size'],
                        'best_hidden_layers': str(self.best_params['hidden_layers'])
                    }
                )
                mlflow.log_metrics(
                    {
                        "best_val_rmse": self.best_trial.value,
                        "total_trials": len(self.study.trials)

                    }
                )
                logger.info(f"Optimization completed. Best validation RMSE: {self.best_trial.value:.4f}")
                logger.info(f"Best parameters: {self.best_params}")
                
                return {
                    "best_params": self.best_params,
                    "best_rmse": self.best_trial.value,
                    "study": self.study,
                    "best_trial": self.best_trial
                }
    def get_trials_dataframe(self):
        """Get all trials as a pandas DataFrame (useful for analysis)"""
        if self.study:
            return self.study.trials_dataframe()
        return None
        
    def get_optimization_history(self):
        """Get optimization history for plotting"""
        if self.study:
            return optuna.visualization.plot_optimization_history(self.study)
        return None
    
    def get_parallel_coordinate_plot(self):
        """Get parallel coordinate plot of all trials"""
        if self.study:
            return optuna.visualization.plot_parallel_coordinate(self.study)
        return None
        


