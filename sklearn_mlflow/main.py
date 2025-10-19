from mlflow import MlflowClient
import mlflow
import dotenv
import os

from mlflow.models import infer_signature
from training_code import prepare_data, train_RFmodel, data_val, params
import pandas as pd
from generate_data import generate_apple_sales_data_with_promo_adjustment

dotenv.load_dotenv()

### USING MLFLOW CLIENT
# client = MlflowClient(tracking_uri= os.getenv('MLFLOW_TRACKING_URI'))
# all_experiments = client.search_experiments()
# default_experiment = [
#                     {"name":experiments.name,"lifecycle_stage":experiments.lifecycle_stage} 
#                     for  experiments in all_experiments
#                    if experiments.name == "Default"
#                    ][0]
# print(default_experiment)


### USING MLFLOW FLUENT API
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))


def safe_experiment_setup(experiment_name=None):
    """
    Safely set up MLflow experiment, handling various edge cases
    """
    if experiment_name is None:
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "Default_Experiment")
    
    client = MlflowClient()
    
    try:
        # Try to get the experiment
        experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment:
            if experiment.lifecycle_stage == "deleted":
                # Restore if deleted
                client.restore_experiment(experiment.experiment_id)
                print(f"✅ Restored previously deleted experiment: {experiment_name}")
            else:
                print(f"✅ Using existing experiment: {experiment_name}")
        else:
            # Create new experiment if it doesn't exist
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"✅ Created new experiment: {experiment_name}")
        
        # Set as active experiment
        return mlflow.set_experiment(experiment_name)
        
    except Exception as e:
        print(f"❌ Error setting up experiment: {e}")
        # Fallback: create with modified name
        fallback_name = f"{experiment_name}_fallback"
        print(f"🔄 Trying fallback experiment: {fallback_name}")
        return mlflow.create_experiment(fallback_name)



def get_artifact_internal_id(run_id, experiment_id=None):
    """Extract the internal artifact ID from the filesystem"""
    
    if experiment_id is None:
        experiment_id = mlflow.active_run().info.experiment_id
    
    artifacts_dir = f"mlruns/{experiment_id}/{run_id}/outputs"
    
    if os.path.exists(artifacts_dir):
        for item in os.listdir(artifacts_dir):
            if item.startswith('m-'): 
                return item

# Usage
apple_experiment = safe_experiment_setup()
RUN_NAME = os.getenv("MLFLOW_RUN_NAME")
ARTIFACT_PATH = os.getenv("MLFLOW_ARTIFACT_PATH")
data_path = "./data/apple_sales_data.csv"


def main():
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"{e}\nGenerating Data ...")
        df = generate_apple_sales_data_with_promo_adjustment( base_demand= 1000, n_rows = 5000)
        df.to_csv(data_path)
        print("The Data was generated successfully")
    X_train, X_val, y_train, y_val = prepare_data(df)
    rf = train_RFmodel(X_train= X_train, y_train= y_train, params = params)
    metrics = data_val(model= rf, X_val= X_val, y_val= y_val)
    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        signature = infer_signature(X_val,y_val)
        print(mlflow.get_artifact_uri())
        mlflow.sklearn.log_model(sk_model= rf,
                                signature= signature, 
                                registered_model_name= "appels_model",
                                name= ARTIFACT_PATH)
        
        # Hard code part, cause the log_model is not working properly
        dest_id = get_artifact_internal_id(run.info._run_id)
        mlflow.log_artifacts(f"./mlartifacts/{run.info.experiment_id}/models/{dest_id}/artifacts", artifact_path= ARTIFACT_PATH)

if __name__ =="__main__":
    main()