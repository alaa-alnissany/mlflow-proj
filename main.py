from mlflow import MlflowClient
import mlflow
import dotenv
import os
import training_code
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
apple_experiment=mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME")) 
RUN_NAME = os.getenv("MLFLOW_RUN_NAME")
ARTIFACT_PATH = os.getenv("MLFLOW_ARTIFACT_PATH")
data_path = "./data/apple_sales_data.csv"


try:
    df = pd.read_csv(data_path)
except Exception as e:
    print(f"{e}\nGenerating Data ...")
    df = generate_apple_sales_data_with_promo_adjustment( base_demand= 1000, n_rows = 5000)
    df.to_csv(data_path)
    print("The Data was generated successfully")

