FROM python:3.9-buster

# Set the working directory
WORKDIR /app

# Install system dependencies including gcloud CLI
RUN apt-get update && apt-get install -y curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && apt-get install -y google-cloud-sdk

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install mlflow globally to ensure it's available in PATH
RUN pip install mlflow

# Copy application scripts
COPY server.sh server.sh

# Make server.sh executable
RUN chmod +x server.sh

# Expose the port for MLflow
EXPOSE 5001

# Set entrypoint to server.sh
ENTRYPOINT ["./server.sh"]
