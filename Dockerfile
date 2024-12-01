


# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory to the Model_Pipeline folder
WORKDIR /Model_Pipeline

# Copy the requirements file into the container
COPY requirements.txt ./


# Install any required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire Model_Pipeline directory into the container
COPY . ./

# Expose the port your app runs on (if applicable)
EXPOSE 8000

# Set the entry point for the container
CMD ["python", "scripts/ml_ops_model.py"]

