import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.models import infer_signature
from datetime import datetime

# Set the MLflow tracking URI
TRACKING_URI = "http://127.0.0.1:5001"  # Replace with external URL if hosted
mlflow.set_tracking_uri(TRACKING_URI)

# Create an experiment
experiment_name = "Test_Experiment"
experiment = mlflow.get_experiment_by_name(name=experiment_name)
if not experiment:
    experiment_id = mlflow.create_experiment(name=experiment_name)
else:
    experiment_id = experiment.experiment_id

# Load data
url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
data = pd.read_csv(url, sep=";")
train, test = train_test_split(data)
train_x, train_y = train.drop("quality", axis=1), train["quality"]
test_x, test_y = test.drop("quality", axis=1), test["quality"]

# Train model
alpha = 0.5
l1_ratio = 0.5
model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
model.fit(train_x, train_y)
predictions = model.predict(test_x)

# Evaluate model
rmse = np.sqrt(mean_squared_error(test_y, predictions))
mae = mean_absolute_error(test_y, predictions)
r2 = r2_score(test_y, predictions)

# Log to MLflow
try:
    with mlflow.start_run(run_name=datetime.now().strftime("%Y-%m-%d_%H:%M"), experiment_id=experiment_id):
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # Log model with signature
        signature = infer_signature(train_x, predictions)
        mlflow.sklearn.log_model(model, "model", signature=signature)

        # Log environment
        mlflow.log_artifact("requirements.txt")  # Ensure this file exists
except Exception as e:
    print(f"Error logging to MLflow: {e}")
