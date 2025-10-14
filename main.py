from mlflow import MlflowClient
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
import dotenv
import os

dotenv.load_dotenv()
client = MlflowClient(tracking_uri= os.getenv('MLFLOW_TRACKING_URI'))
all_experiments = client.search_experiments()
default_experiment = [
                    {"name":experiments.name,"lifecycle_stage":experiments.lifecycle_stage} 
                    for  experiments in all_experiments
                    if experiments.name == "Default"
                    ][0]
print(default_experiment)