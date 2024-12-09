from google.cloud import monitoring_v3
from datetime import datetime
import os

def push_metrics_to_monitoring(metric_name, value):
    """
    Push custom metrics to Google Cloud Monitoring.

    Args:
        metric_name (str): Name of the metric (e.g., "model_accuracy").
        value (float): Value of the metric to push.
    """
    # Initialize the client
    client = monitoring_v3.MetricServiceClient()

    # Set the project ID explicitly
    project_id = "theta-function-429605-j0"
    project_name = f"projects/{project_id}"

    # Define the metric and resource type
    series = monitoring_v3.TimeSeries()
    series.metric.type = f"custom.googleapis.com/{metric_name}"
    series.resource.type = "global"
    series.resource.labels["project_id"] = project_id

    # Add additional labels to the metric (optional)
    series.metric.labels["environment"] = "production"
    series.metric.labels["component"] = "model-monitoring"

    # Set the metric value and timestamp
    point = series.points.add()
    point.value.double_value = value  # Adjust to int64_value or bool_value if necessary
    point.interval.end_time.FromDatetime(datetime.utcnow())

    # Push the metric to Google Cloud Monitoring
    try:
        client.create_time_series(name=project_name, time_series=[series])
        print(f"Metric {metric_name} pushed successfully.")
    except Exception as e:
        print(f"Error pushing metric {metric_name}: {e}")

# Example usage
if __name__ == "__main__":
    # Push some example metrics
    push_metrics_to_monitoring("model_accuracy", 0.95)
    push_metrics_to_monitoring("data_drift_score", 0.15)
