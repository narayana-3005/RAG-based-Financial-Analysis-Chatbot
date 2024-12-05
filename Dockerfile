


# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory to the Model_Pipeline folder
WORKDIR /Model_Pipeline

#The error occurs because the pyfarmhash package requires gcc (GNU Compiler Collection) to build its components,
#and the Docker image python:3.9-slim used in your Dockerfile does not include gcc by default.

RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
&& apt-get clean && rm -rf /var/lib/apt/lists/*
 
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

