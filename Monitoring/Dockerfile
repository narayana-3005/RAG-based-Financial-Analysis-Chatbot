# Use Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . ./

# Expose the port for Cloud Run
EXPOSE 8080

# Command to run the application
CMD ["python", "monitoring/log_predictions.py"]
