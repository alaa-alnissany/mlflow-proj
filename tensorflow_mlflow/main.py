from prepare_data import DataProcessor
from model import HyperparameterOptimizer

# Important Libs for clean coding
import logging
import warnings
import dotenv
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore') 

dotenv.load_dotenv()

def main():
    try:
        DATA_URL = os.getenv("DATA_URL")
        EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")
        MAX_EVALS = os.getenv("MAX_EVALS")
        logger.info("Step 1: Preparing data")
        processor = DataProcessor()
        data = processor.load_data(DATA_URL)
        X,y = processor.prepare_features_target(data)
        processed_data = processor.split_data(X,y)

        # Step 2: Run hyperparameter optimization
        logger.info("Step 2: Starting hyperparameter optimization")
        optimizer = HyperparameterOptimizer(processed_data, EXPERIMENT_NAME)
        results = optimizer.optimize(max_evals=MAX_EVALS)
        
        # Step 3: Final evaluation (optional - train best model on full training data)
        logger.info("Step 3: Optimization completed")
        logger.info(f"Best parameters: {results['best_params']}")
        logger.info(f"Best validation RMSE: {results['best_rmse']:.4f}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()