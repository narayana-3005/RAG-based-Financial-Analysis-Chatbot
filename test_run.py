"Simple script to log a experiment on MLFlow Server"

# Import libraries
import pandas as pd
import numpy as np
from datetime import datetime

import mlflow
from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define the tracking URI for the MLflow experiment
TRACKING_URI = " https://rag-based-financial-analysis-chatbot-jzgmntpbbq-ue.a.run.app"


# Read the wine-quality dataset from a CSV file
csv_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
data = pd.read_csv(csv_url, sep=";")

# Split the data into training and testing sets
train, test = train_test_split(data)

# Extract the features and target variable from the data
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]

# Define hyperparameters for the Elastic Net model
alpha = 0.5
l1_ratio = 0.5
random_state = 42
max_iter = 1000

# Create an Elastic Net model with the defined hyperparameters
lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state, max_iter=max_iter)

# Fit the model to the training data
lr.fit(train_x, train_y)

# Make predictions on the testing data
predictions = lr.predict(test_x)

# Evaluate the model's performance
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

(rmse, mae, r2) = eval_metrics(test_y, predictions)

# Print the evaluation metrics
print(f"Elasticnet model (alpha={alpha:f}, l1_ratio={l1_ratio:f}, random_state={random_state}, max_iter={max_iter}):")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R2: {r2}")

# Set the tracking URI for the MLflow experiment
mlflow.set_tracking_uri(TRACKING_URI)

# Create an experiment if it doesn't exist
experiment_name = "Test_Experiment"
if not mlflow.get_experiment_by_name(name=experiment_name):
    mlflow.create_experiment(
        name=experiment_name
    )
experiment = mlflow.get_experiment_by_name(experiment_name)

# Define the run name and tags for the experiment
run_name = datetime.now().strftime("%Y-%m-%d_%H:%M")
tags = {
    "env": "test",
    "data_date": "2023-11-24",
    "model_type": "ElasticNet",
    "experiment_description": "Tutorial MLFlow experiment"
    # ... other tags ...
}

# Start the MLflow run
with mlflow.start_run(
    experiment_id=experiment.experiment_id, 
    run_name=run_name, 
    tags=tags
):
    
    # Log the hyperparameters used in the model
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("max_iter", max_iter)
    
    # Log the metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    
    # Log model:
    signature = infer_signature(train_x, predictions)
    mlflow.sklearn.log_model(lr, "model", signature=signature)
