import mlflow
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://35.227.96.52:5001")  # Replace with your actual IP

# Start an MLflow experiment
mlflow.set_experiment("Test_Experiment")

# Load sample data
data = pd.read_csv("https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv", sep=";")
train, test = train_test_split(data)
train_x, train_y = train.drop("quality", axis=1), train["quality"]
test_x, test_y = test.drop("quality", axis=1), test["quality"]

# Train a model
model = ElasticNet(alpha=0.5, l1_ratio=0.5)
model.fit(train_x, train_y)
predictions = model.predict(test_x)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(test_y, predictions))

# Log parameters, metrics, and model to MLflow
with mlflow.start_run():
    mlflow.log_param("alpha", 0.5)
    mlflow.log_param("l1_ratio", 0.5)
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(model, "model")
